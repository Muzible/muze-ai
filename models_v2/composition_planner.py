"""
🎵 Composition Planner - Mały transformer generujący strukturę utworu

Generuje plan kompozycji przed uruchomieniem głównego modelu LDM:
- Struktura sekcji (Intro -> Verse -> Chorus -> ...)
- Mapa tempa per sekcja
- Krzywa energii
- Progresja tonacji

Użycie:
    planner = CompositionPlanner()
    plan = planner.generate(
        prompt="sad love ballad, 3 minutes",
        target_duration=180.0,
        genre="ballad"
    )
    # plan.sections = [SectionPlan(type='intro', duration=15, tempo=75, ...), ...]

Trening:
    python -m models_v2.composition_planner --train --data ./data_v2/segments.json
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Dict, Any, Tuple
from dataclasses import dataclass, asdict, field
import json
import math
import numpy as np
from enum import IntEnum


class SectionTypeToken(IntEnum):
    """Tokeny sekcji dla modelu"""
    PAD = 0
    BOS = 1      # Beginning of sequence
    EOS = 2      # End of sequence
    INTRO = 3
    VERSE = 4
    PRE_CHORUS = 5
    CHORUS = 6
    POST_CHORUS = 7
    BRIDGE = 8
    INSTRUMENTAL = 9
    SOLO = 10
    BREAKDOWN = 11
    BUILDUP = 12
    DROP = 13
    OUTRO = 14


# Mapowanie string -> token
SECTION_TO_TOKEN = {
    'intro': SectionTypeToken.INTRO,
    'verse': SectionTypeToken.VERSE,
    'pre_chorus': SectionTypeToken.PRE_CHORUS,
    'chorus': SectionTypeToken.CHORUS,
    'post_chorus': SectionTypeToken.POST_CHORUS,
    'bridge': SectionTypeToken.BRIDGE,
    'instrumental': SectionTypeToken.INSTRUMENTAL,
    'solo': SectionTypeToken.SOLO,
    'breakdown': SectionTypeToken.BREAKDOWN,
    'buildup': SectionTypeToken.BUILDUP,
    'drop': SectionTypeToken.DROP,
    'outro': SectionTypeToken.OUTRO,
    # Fallback dla nieznanego (mapuj na verse jako bezpieczny default)
    'unknown': SectionTypeToken.VERSE,
}

TOKEN_TO_SECTION = {v: k for k, v in SECTION_TO_TOKEN.items() if k != 'unknown'}


@dataclass
class SectionPlan:
    """Plan pojedynczej sekcji"""
    section_type: str
    duration: float          # w sekundach
    tempo: float             # BPM
    energy: float            # 0-1
    key: str                 # np. "Am", "C"
    has_vocals: bool
    position_start: float    # 0-1 (pozycja w utworze)
    position_end: float      # 0-1
    
    # Optional
    transition_type: str = "crossfade"  # "cut", "crossfade", "buildup"
    prompt_hint: str = ""


@dataclass
class CompositionPlan:
    """Full composition plan for a track"""
    total_duration: float
    global_tempo: float
    global_key: str
    genre: str
    mood: str
    
    sections: List[SectionPlan]
    
    # Metadane
    generation_params: Dict = field(default_factory=dict)
    
    def to_conditioning_sequence(self) -> List[Dict]:
        """Converts plan to conditioning sequence for LDM"""
        return [
            {
                'section_type': s.section_type,
                'tempo': s.tempo,
                'energy': s.energy,
                'key': s.key,
                'has_vocals': s.has_vocals,
                'duration': s.duration,
                'position': (s.position_start + s.position_end) / 2,
            }
            for s in self.sections
        ]


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding"""
    
    def __init__(self, d_model: int, max_len: int = 100):
        super().__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe.unsqueeze(0))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, :x.size(1)]


class CompositionTransformer(nn.Module):
    """
    Transformer for generating composition structure.
    
    Input:
    - Text embedding (from T5/CLAP) - track description
    - Target duration (continuous)
    - Genre embedding
    - Mood embedding
    
    Output (autoregressive):
    - Sequence of section tokens
    - For each section: duration, tempo, energy, key, has_vocals
    """
    
    def __init__(
        self,
        vocab_size: int = 15,           # Number of section types + special tokens
        d_model: int = 256,             # Model dimension
        nhead: int = 4,                 # Number of attention heads
        num_encoder_layers: int = 3,    # Encoder layers (for prompt)
        num_decoder_layers: int = 4,    # Decoder layers (autoregressive)
        dim_feedforward: int = 512,
        dropout: float = 0.1,
        max_sections: int = 20,         # Max sections in track
        text_embed_dim: int = 768,      # Text embedding dimension (T5)
        num_genres: int = 50,
        num_moods: int = 30,
        num_keys: int = 24,             # 12 major + 12 minor
    ):
        super().__init__()
        
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.max_sections = max_sections
        
        # ===== ENCODER (for context) =====
        
        # Text embedding projection
        self.text_proj = nn.Linear(text_embed_dim, d_model)
        
        # Duration embedding (continuous)
        self.duration_embed = nn.Sequential(
            nn.Linear(1, 64),
            nn.SiLU(),
            nn.Linear(64, d_model),
        )
        
        # Genre embedding
        self.genre_embed = nn.Embedding(num_genres, d_model)
        
        # Mood embedding
        self.mood_embed = nn.Embedding(num_moods, d_model)
        
        # Encoder transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)
        
        # ===== DECODER (autoregresywny) =====
        
        # Section type embedding
        self.section_embed = nn.Embedding(vocab_size, d_model)
        
        # Continuous attributes embedding
        # (tempo_normalized, energy, duration_normalized, has_vocals)
        self.attr_embed = nn.Linear(4, d_model)
        
        # Key embedding
        self.key_embed = nn.Embedding(num_keys, d_model // 4)
        
        # Positional encoding - max_sections + 2 for BOS/EOS tokens
        self.pos_encoder = PositionalEncoding(d_model, max_sections + 2)
        
        # Decoder transformer
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_decoder_layers)
        
        # ===== OUTPUT HEADS =====
        
        # Section type prediction
        self.section_head = nn.Linear(d_model, vocab_size)
        
        # Continuous attributes prediction (tempo, energy, duration)
        self.attr_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.SiLU(),
            nn.Linear(d_model // 2, 3),  # tempo, energy, duration (all normalized 0-1)
        )
        
        # Key prediction
        self.key_head = nn.Linear(d_model, num_keys)
        
        # Has vocals (binary)
        self.vocals_head = nn.Linear(d_model, 1)
        
        # Causal mask - max_sections + 2 for BOS/EOS tokens
        self.register_buffer(
            'causal_mask',
            torch.triu(torch.ones(max_sections + 2, max_sections + 2), diagonal=1).bool()
        )
    
    def encode_context(
        self,
        text_embed: torch.Tensor,       # [B, seq_len, text_embed_dim]
        target_duration: torch.Tensor,   # [B, 1]
        genre_idx: torch.Tensor,         # [B]
        mood_idx: torch.Tensor,          # [B]
    ) -> torch.Tensor:
        """Enkoduje kontekst (prompt + atrybuty)"""
        
        B = text_embed.size(0)
        
        # Project text
        text_proj = self.text_proj(text_embed)  # [B, seq_len, d_model]
        
        # Duration embedding
        dur_emb = self.duration_embed(target_duration)  # [B, d_model]
        dur_emb = dur_emb.unsqueeze(1)  # [B, 1, d_model]
        
        # Genre embedding
        genre_emb = self.genre_embed(genre_idx)  # [B, d_model]
        genre_emb = genre_emb.unsqueeze(1)  # [B, 1, d_model]
        
        # Mood embedding
        mood_emb = self.mood_embed(mood_idx)  # [B, d_model]
        mood_emb = mood_emb.unsqueeze(1)  # [B, 1, d_model]
        
        # Concatenate context
        context = torch.cat([text_proj, dur_emb, genre_emb, mood_emb], dim=1)
        
        # Encode
        memory = self.encoder(context)
        
        return memory
    
    def forward(
        self,
        text_embed: torch.Tensor,       # [B, seq_len, text_embed_dim]
        target_duration: torch.Tensor,   # [B, 1]
        genre_idx: torch.Tensor,         # [B]
        mood_idx: torch.Tensor,          # [B]
        tgt_sections: torch.Tensor,      # [B, T] - target section tokens
        tgt_attrs: torch.Tensor,         # [B, T, 4] - tempo, energy, duration, has_vocals
        tgt_keys: torch.Tensor,          # [B, T] - key indices
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass dla treningu (teacher forcing).
        
        Returns:
            Dict with logits for each output head
        """
        B, T = tgt_sections.shape
        
        # Encode context
        memory = self.encode_context(text_embed, target_duration, genre_idx, mood_idx)
        
        # Prepare decoder input
        section_emb = self.section_embed(tgt_sections)  # [B, T, d_model]
        attr_emb = self.attr_embed(tgt_attrs)  # [B, T, d_model]
        key_emb = self.key_embed(tgt_keys)  # [B, T, d_model//4]
        key_emb = F.pad(key_emb, (0, self.d_model - key_emb.size(-1)))  # Pad to d_model
        
        # Combine embeddings
        decoder_input = section_emb + attr_emb + key_emb
        decoder_input = self.pos_encoder(decoder_input)
        
        # Causal mask
        causal_mask = self.causal_mask[:T, :T]
        
        # Decode
        decoder_output = self.decoder(
            decoder_input,
            memory,
            tgt_mask=causal_mask,
        )
        
        # Output heads
        section_logits = self.section_head(decoder_output)  # [B, T, vocab_size]
        attr_pred = torch.sigmoid(self.attr_head(decoder_output))  # [B, T, 3] - normalized
        key_logits = self.key_head(decoder_output)  # [B, T, num_keys]
        vocals_logits = self.vocals_head(decoder_output).squeeze(-1)  # [B, T]
        
        return {
            'section_logits': section_logits,
            'attr_pred': attr_pred,
            'key_logits': key_logits,
            'vocals_logits': vocals_logits,
        }
    
    @torch.no_grad()
    def generate(
        self,
        text_embed: torch.Tensor,       # [1, seq_len, text_embed_dim]
        target_duration: torch.Tensor,   # [1, 1]
        genre_idx: torch.Tensor,         # [1]
        mood_idx: torch.Tensor,          # [1]
        temperature: float = 0.8,
        max_sections: int = 15,
    ) -> List[Dict]:
        """
        Generuje strukturę kompozycji autoregresywnie.
        
        Returns:
            Lista słowników z informacjami o każdej sekcji
        """
        self.eval()
        device = text_embed.device
        
        # Encode context
        memory = self.encode_context(text_embed, target_duration, genre_idx, mood_idx)
        
        # Start with BOS token
        current_sections = torch.tensor([[SectionTypeToken.BOS]], device=device)
        current_attrs = torch.zeros(1, 1, 4, device=device)
        current_keys = torch.zeros(1, 1, dtype=torch.long, device=device)
        
        generated = []
        total_duration = 0.0
        # Denormalize target_duration (input was normalized by /300)
        target_dur = target_duration.item() * 300.0
        
        for step in range(max_sections):
            # Decoder input
            section_emb = self.section_embed(current_sections)
            attr_emb = self.attr_embed(current_attrs)
            key_emb = self.key_embed(current_keys)
            key_emb = F.pad(key_emb, (0, self.d_model - key_emb.size(-1)))
            
            decoder_input = section_emb + attr_emb + key_emb
            decoder_input = self.pos_encoder(decoder_input)
            
            # Causal mask
            T = current_sections.size(1)
            causal_mask = self.causal_mask[:T, :T]
            
            # Decode
            decoder_output = self.decoder(
                decoder_input,
                memory,
                tgt_mask=causal_mask,
            )
            
            # Get last position predictions
            last_output = decoder_output[:, -1]  # [1, d_model]
            
            # Sample section type
            section_logits = self.section_head(last_output) / temperature
            # Mask special tokens (except EOS)
            section_logits[:, SectionTypeToken.PAD] = -float('inf')
            section_logits[:, SectionTypeToken.BOS] = -float('inf')
            
            section_probs = F.softmax(section_logits, dim=-1)
            next_section = torch.multinomial(section_probs, 1)
            
            # Check for EOS
            if next_section.item() == SectionTypeToken.EOS:
                break
            
            # Predict attributes
            attr_pred = torch.sigmoid(self.attr_head(last_output))  # [1, 3]
            tempo_norm, energy, dur_norm = attr_pred[0].tolist()
            
            # Denormalize
            tempo = 60 + tempo_norm * 140  # 60-200 BPM
            duration = 8 + dur_norm * 52   # 8-60 seconds per section
            
            # Adjust duration if we're near the end
            remaining = target_dur - total_duration
            if remaining < duration * 1.5:
                duration = remaining
            
            # Sample key
            key_logits = self.key_head(last_output) / temperature
            key_probs = F.softmax(key_logits, dim=-1)
            next_key = torch.multinomial(key_probs, 1)
            
            # Predict vocals
            vocals_prob = torch.sigmoid(self.vocals_head(last_output))
            has_vocals = vocals_prob.item() > 0.5
            
            # Key index to string
            key_idx = next_key.item()
            keys = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
            key_str = keys[key_idx % 12] + (' minor' if key_idx >= 12 else '')
            
            # Section type to string
            section_type = TOKEN_TO_SECTION.get(next_section.item(), 'verse')
            
            generated.append({
                'section_type': section_type,
                'tempo': tempo,
                'energy': energy,
                'duration': duration,
                'key': key_str,
                'has_vocals': has_vocals,
                'position_start': total_duration / target_dur,
                'position_end': (total_duration + duration) / target_dur,
            })
            
            total_duration += duration
            
            # Update sequences for next step
            # next_section has shape [1, 1] from multinomial, need [1, 1] for concat
            current_sections = torch.cat([
                current_sections,
                next_section  # already [1, 1]
            ], dim=1)
            
            next_attrs = torch.tensor([[[tempo_norm, energy, dur_norm, float(has_vocals)]]], device=device)
            current_attrs = torch.cat([current_attrs, next_attrs], dim=1)
            
            # next_key has shape [1, 1] from multinomial
            current_keys = torch.cat([current_keys, next_key], dim=1)
            
            # Stop if we've reached target duration
            if total_duration >= target_dur * 0.95:
                break
        
        return generated


class CompositionPlanner:
    """
    Wysokopoziomowy interfejs do generowania planów kompozycji.
    
    Użycie:
        planner = CompositionPlanner.from_pretrained("./checkpoints/composition_planner.pt")
        plan = planner.generate(
            prompt="energetic EDM track with big drops",
            target_duration=180.0,
            genre="electronic",
            mood="energetic"
        )
    """
    
    # Predefiniowane mapowania
    GENRE_TO_IDX = {
        'pop': 0, 'rock': 1, 'electronic': 2, 'hip_hop': 3, 'r&b': 4,
        'jazz': 5, 'classical': 6, 'country': 7, 'metal': 8, 'folk': 9,
        'indie': 10, 'punk': 11, 'soul': 12, 'funk': 13, 'disco': 14,
        'house': 15, 'techno': 16, 'trance': 17, 'dubstep': 18, 'ambient': 19,
        'ballad': 20, 'blues': 21, 'reggae': 22, 'latin': 23, 'world': 24,
    }
    
    MOOD_TO_IDX = {
        'happy': 0, 'sad': 1, 'energetic': 2, 'calm': 3, 'angry': 4,
        'romantic': 5, 'melancholic': 6, 'euphoric': 7, 'dark': 8, 'uplifting': 9,
        'chill': 10, 'aggressive': 11, 'dreamy': 12, 'nostalgic': 13, 'epic': 14,
    }
    
    def __init__(
        self,
        model: CompositionTransformer,
        text_encoder: Optional[nn.Module] = None,
        device: str = 'cpu',
    ):
        self.model = model.to(device)
        self.text_encoder = text_encoder
        self.device = device
    
    @classmethod
    def _infer_config_from_state_dict(cls, state_dict: dict) -> dict:
        """
        Automatycznie wykrywa konfigurację modelu z wag checkpointu.
        
        Sprawdza kształty tensorów aby określić:
        - d_model (z genre_embed.weight shape[1])
        - num_genres (z genre_embed.weight shape[0])
        - num_moods (z mood_embed.weight shape[0])
        - vocab_size (z section_embed.weight shape[0])
        - num_keys (z key_embed.weight shape[0])
        """
        config = {}
        
        for key, tensor in state_dict.items():
            if key == 'genre_embed.weight':
                config['num_genres'] = tensor.shape[0]
                config['d_model'] = tensor.shape[1]
            elif key == 'mood_embed.weight':
                config['num_moods'] = tensor.shape[0]
            elif key == 'section_embed.weight':
                config['vocab_size'] = tensor.shape[0]
            elif key == 'key_embed.weight':
                config['num_keys'] = tensor.shape[0]
            elif key == 'text_proj.weight':
                config['text_embed_dim'] = tensor.shape[1]
        
        return config
    
    @classmethod
    def from_pretrained(cls, checkpoint_path: str, device: str = 'cpu'):
        """
        Ładuje wytrenowany model z automatyczną detekcją konfiguracji.
        
        Priorytet konfiguracji:
        1. model_config z checkpointu (jeśli kompletny)
        2. Automatyczna detekcja z wag state_dict
        3. Wartości domyślne
        """
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        
        state_dict = checkpoint.get('model_state_dict', checkpoint)
        saved_config = checkpoint.get('model_config', {})
        
        # Automatically detect config from weights
        inferred_config = cls._infer_config_from_state_dict(state_dict)
        
        # Merge: inferred_config overwrites gaps in saved_config
        final_config = {**inferred_config, **saved_config}
        
        # Make sure inferred values are used (saved may have incomplete data)
        for key in ['num_genres', 'num_moods', 'vocab_size', 'num_keys', 'd_model']:
            if key in inferred_config:
                final_config[key] = inferred_config[key]
        
        print(f"   CompositionPlanner config: {final_config}")
        
        model = CompositionTransformer(**final_config)
        model.load_state_dict(state_dict)
        
        return cls(model, device=device)
    
    def generate(
        self,
        prompt: str,
        target_duration: float = 180.0,
        genre: str = 'pop',
        mood: str = 'energetic',
        temperature: float = 0.8,
        text_embed: Optional[torch.Tensor] = None,
    ) -> CompositionPlan:
        """
        Generuje plan kompozycji.
        
        Args:
            prompt: Opis utworu
            target_duration: Docelowa długość w sekundach
            genre: Gatunek muzyczny
            mood: Nastrój
            temperature: Temperatura próbkowania (wyższa = bardziej kreatywne)
            text_embed: Opcjonalny gotowy embedding tekstu
            
        Returns:
            CompositionPlan z listą sekcji
        """
        self.model.eval()
        
        # Get text embedding
        if text_embed is None:
            if self.text_encoder is not None:
                text_embed = self.text_encoder([prompt])
            else:
                # Dummy embedding if no encoder
                text_embed = torch.randn(1, 10, 768)
        
        text_embed = text_embed.to(self.device)
        
        # Prepare inputs
        duration_tensor = torch.tensor([[target_duration / 300.0]], device=self.device)  # Normalize
        genre_idx = torch.tensor([self.GENRE_TO_IDX.get(genre.lower(), 0)], device=self.device)
        mood_idx = torch.tensor([self.MOOD_TO_IDX.get(mood.lower(), 0)], device=self.device)
        
        # Generate
        sections_data = self.model.generate(
            text_embed,
            duration_tensor,
            genre_idx,
            mood_idx,
            temperature=temperature,
        )
        
        # Convert to SectionPlan objects
        sections = []
        for data in sections_data:
            sections.append(SectionPlan(
                section_type=data['section_type'],
                duration=data['duration'],
                tempo=data['tempo'],
                energy=data['energy'],
                key=data['key'],
                has_vocals=data['has_vocals'],
                position_start=data['position_start'],
                position_end=data['position_end'],
            ))
        
        # Calculate global values
        global_tempo = np.mean([s.tempo for s in sections]) if sections else 120.0
        global_key = sections[0].key if sections else 'C'
        
        return CompositionPlan(
            total_duration=target_duration,
            global_tempo=global_tempo,
            global_key=global_key,
            genre=genre,
            mood=mood,
            sections=sections,
            generation_params={
                'prompt': prompt,
                'temperature': temperature,
            }
        )
    
    def generate_from_template(
        self,
        template: str,
        target_duration: float = 180.0,
        **kwargs
    ) -> CompositionPlan:
        """
        Generuje plan z predefiniowanego template'u.
        
        Templates:
        - "verse_chorus": Klasyczna struktura Verse-Chorus
        - "edm": EDM z buildupami i dropami
        - "ballad": Wolna ballada
        - "progressive": Progressive rock/metal
        """
        
        templates = {
            'verse_chorus': [
                ('intro', 0.08), ('verse', 0.17), ('chorus', 0.17),
                ('verse', 0.17), ('chorus', 0.17), ('bridge', 0.08),
                ('chorus', 0.12), ('outro', 0.04),
            ],
            'edm': [
                ('intro', 0.07), ('buildup', 0.10), ('drop', 0.15),
                ('breakdown', 0.10), ('buildup', 0.10), ('drop', 0.20),
                ('breakdown', 0.08), ('buildup', 0.08), ('drop', 0.08),
                ('outro', 0.04),
            ],
            'ballad': [
                ('intro', 0.10), ('verse', 0.20), ('chorus', 0.15),
                ('verse', 0.15), ('chorus', 0.15), ('bridge', 0.10),
                ('chorus', 0.10), ('outro', 0.05),
            ],
            'progressive': [
                ('intro', 0.08), ('verse', 0.12), ('instrumental', 0.10),
                ('verse', 0.12), ('chorus', 0.12), ('solo', 0.15),
                ('bridge', 0.08), ('chorus', 0.15), ('outro', 0.08),
            ],
        }
        
        if template not in templates:
            raise ValueError(f"Unknown template: {template}. Available: {list(templates.keys())}")
        
        structure = templates[template]
        sections = []
        current_position = 0.0
        
        for section_type, fraction in structure:
            duration = target_duration * fraction
            sections.append(SectionPlan(
                section_type=section_type,
                duration=duration,
                tempo=kwargs.get('tempo', 120.0),
                energy=0.5,  # Default, can be refined
                key=kwargs.get('key', 'C'),
                has_vocals=section_type in ['verse', 'chorus', 'bridge'],
                position_start=current_position / target_duration,
                position_end=(current_position + duration) / target_duration,
            ))
            current_position += duration
        
        return CompositionPlan(
            total_duration=target_duration,
            global_tempo=kwargs.get('tempo', 120.0),
            global_key=kwargs.get('key', 'C'),
            genre=kwargs.get('genre', 'pop'),
            mood=kwargs.get('mood', 'neutral'),
            sections=sections,
            generation_params={'template': template},
        )


def compute_loss(
    outputs: Dict[str, torch.Tensor],
    targets: Dict[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    """
    Oblicza loss dla treningu.
    
    Args:
        outputs: Dict z logits z modelu
        targets: Dict z ground truth
    
    Returns:
        Dict z poszczególnymi lossami i total
    """
    # Section type loss (cross-entropy)
    section_loss = F.cross_entropy(
        outputs['section_logits'].view(-1, outputs['section_logits'].size(-1)),
        targets['sections'].view(-1),
        ignore_index=SectionTypeToken.PAD,
    )
    
    # Attribute loss (MSE dla tempo, energy, duration)
    # Maskujemy PAD pozycje
    mask = (targets['sections'] != SectionTypeToken.PAD).float().unsqueeze(-1)
    attr_loss = F.mse_loss(
        outputs['attr_pred'] * mask,
        targets['attrs'] * mask,
    )
    
    # Key loss (cross-entropy)
    key_loss = F.cross_entropy(
        outputs['key_logits'].view(-1, outputs['key_logits'].size(-1)),
        targets['keys'].view(-1),
        ignore_index=-1,  # Ignore PAD
    )
    
    # Vocals loss (binary cross-entropy)
    vocals_loss = F.binary_cross_entropy_with_logits(
        outputs['vocals_logits'],
        targets['vocals'].float(),
        reduction='none',
    )
    vocals_loss = (vocals_loss * mask.squeeze(-1)).mean()
    
    # Total weighted loss
    total_loss = (
        1.0 * section_loss +
        0.5 * attr_loss +
        0.3 * key_loss +
        0.2 * vocals_loss
    )
    
    return {
        'total': total_loss,
        'section': section_loss,
        'attr': attr_loss,
        'key': key_loss,
        'vocals': vocals_loss,
    }


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Composition Planner')
    parser.add_argument('--test', action='store_true', help='Test generation')
    parser.add_argument('--prompt', type=str, default='energetic electronic track with big drops')
    parser.add_argument('--duration', type=float, default=180.0)
    parser.add_argument('--genre', type=str, default='electronic')
    parser.add_argument('--mood', type=str, default='energetic')
    parser.add_argument('--template', type=str, default=None, help='Use template instead of AI generation')
    
    args = parser.parse_args()
    
    print("🎵 Composition Planner Test")
    print("="*60)
    
    # Create model (untrained)
    model = CompositionTransformer()
    planner = CompositionPlanner(model)
    
    if args.template:
        # Use template
        plan = planner.generate_from_template(
            args.template,
            target_duration=args.duration,
            genre=args.genre,
            mood=args.mood,
        )
    else:
        # Use AI generation (untrained model - random output)
        plan = planner.generate(
            prompt=args.prompt,
            target_duration=args.duration,
            genre=args.genre,
            mood=args.mood,
        )
    
    print(f"\n📋 Composition Plan")
    print(f"   Duration: {plan.total_duration:.0f}s ({plan.total_duration/60:.1f} min)")
    print(f"   Genre: {plan.genre}")
    print(f"   Mood: {plan.mood}")
    print(f"   Global tempo: {plan.global_tempo:.0f} BPM")
    print(f"   Global key: {plan.global_key}")
    
    print(f"\n🎼 Sections ({len(plan.sections)}):")
    for i, section in enumerate(plan.sections):
        print(f"\n   [{i+1}] {section.section_type.upper()}")
        print(f"       Duration: {section.duration:.1f}s")
        print(f"       Position: {section.position_start:.1%} - {section.position_end:.1%}")
        print(f"       Tempo: {section.tempo:.0f} BPM | Energy: {section.energy:.2f}")
        print(f"       Key: {section.key} | Vocals: {'Yes' if section.has_vocals else 'No'}")
    
    # Model size
    params = sum(p.numel() for p in model.parameters())
    print(f"\n📊 Model size: {params:,} parameters ({params/1e6:.1f}M)")
