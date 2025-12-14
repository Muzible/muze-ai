"""
🎵 Enhanced Text Encoder v2 - CLAP per-section + Audio Encoder

Ulepszenia względem v1:
1. CLAP Audio Encoder - embedding z segmentów audio (nie tylko tekst)
2. Per-section encoding - osobny embedding dla każdej sekcji
3. Section-aware conditioning - embedding zawiera info o typie sekcji
4. Cross-modal alignment - lepsze powiązanie tekst-muzyka

Architektura:
                                ┌────────────────────┐
     Text Prompt ──────────────▶│    CLAP Text       │──▶ text_embed
                                │    Encoder         │
                                └────────────────────┘
                                          │
                                          ▼
                                ┌────────────────────┐
     Section Type ─────────────▶│    Section         │──▶ combined_embed
     + Position                 │    Fusion          │
                                └────────────────────┘
                                          │
                                          ▼
                                ┌────────────────────┐
     Reference Audio ──────────▶│    CLAP Audio      │──▶ style_embed
     (optional)                 │    Encoder         │
                                └────────────────────┘

Użycie:
    encoder = EnhancedMusicEncoder()
    
    # Text + section encoding
    embed = encoder.encode_section(
        text="powerful chorus with soaring vocals",
        section_type="chorus",
        position=0.5,
        tempo=128,
        energy=0.9,
    )
    
    # Audio reference encoding
    style_embed = encoder.encode_audio_reference(audio_segment)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Dict, Tuple, Union
import numpy as np


class CLAPEncoder(nn.Module):
    """
    Wrapper dla CLAP (Contrastive Language-Audio Pretraining).
    
    v2 Updates:
    - LoRA fine-tuning instead of full freezing
    - Better adaptation to music domain
    
    CLAP is trained on audio-text pairs, so it understands music descriptions
    and can create embeddings from both text and audio.
    """
    
    def __init__(
        self,
        model_name: str = "laion/clap-htsat-unfused",
        freeze: bool = False,        # v2: default False (we use LoRA)
        use_lora: bool = True,       # v2: LoRA fine-tuning
        lora_r: int = 16,            # LoRA rank
        lora_alpha: int = 32,        # LoRA alpha (scaling)
        lora_dropout: float = 0.1,   # LoRA dropout
        lora_target_modules: list = None,  # Which modules to adapt
        device: str = 'cpu',
    ):
        super().__init__()
        
        self.model_name = model_name
        self.device = device
        self._model = None
        self._processor = None
        self.output_dim = 512  # CLAP projection dim
        self.freeze = freeze
        self.use_lora = use_lora
        self.lora_r = lora_r
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        self.lora_target_modules = lora_target_modules or ["q_proj", "v_proj", "k_proj", "out_proj"]
        self._lora_applied = False
    
    def _apply_lora(self):
        """Aplikuje LoRA adaptery do modelu CLAP"""
        if self._lora_applied or not self.use_lora:
            return
        
        try:
            from peft import LoraConfig, get_peft_model, TaskType
            
            print(f"  Applying LoRA (r={self.lora_r}, alpha={self.lora_alpha})...")
            
            lora_config = LoraConfig(
                r=self.lora_r,
                lora_alpha=self.lora_alpha,
                lora_dropout=self.lora_dropout,
                target_modules=self.lora_target_modules,
                bias="none",
                task_type=TaskType.FEATURE_EXTRACTION,
            )
            
            self._model = get_peft_model(self._model, lora_config)
            self._lora_applied = True
            
            # Policz trainable parameters
            trainable_params = sum(p.numel() for p in self._model.parameters() if p.requires_grad)
            total_params = sum(p.numel() for p in self._model.parameters())
            print(f"  LoRA applied! Trainable: {trainable_params:,} / {total_params:,} ({100*trainable_params/total_params:.2f}%)")
            
        except ImportError:
            print("  ⚠️ PEFT not installed - falling back to frozen CLAP")
            print("  Install with: pip install peft")
            self.use_lora = False
            if self.freeze:
                for param in self._model.parameters():
                    param.requires_grad = False
        except Exception as e:
            print(f"  ⚠️ Error applying LoRA: {e}")
            self.use_lora = False
    
    def _load_model(self):
        """Lazy loading modelu CLAP"""
        if self._model is not None:
            return
        
        try:
            from transformers import ClapModel, ClapProcessor
            
            print(f"Loading CLAP model: {self.model_name}...")
            self._model = ClapModel.from_pretrained(self.model_name)
            self._processor = ClapProcessor.from_pretrained(self.model_name)
            self._model = self._model.to(self.device)
            
            # v2: First try LoRA, then optionally freeze
            if self.use_lora:
                self._apply_lora()
            elif self.freeze:
                for param in self._model.parameters():
                    param.requires_grad = False
                print("  CLAP frozen (no LoRA)")
            
            self.output_dim = self._model.config.projection_dim
            print(f"  CLAP loaded. Output dim: {self.output_dim}")
            
        except ImportError:
            print("CLAP not available (transformers not installed with CLAP support)")
            print("Falling back to dummy encoder")
            self._model = "dummy"
        except Exception as e:
            print(f"Error loading CLAP: {e}")
            self._model = "dummy"
    
    def get_trainable_parameters(self):
        """Returns parameters for training (only LoRA if active)"""
        self._load_model()
        if self._model == "dummy":
            return []
        return [p for p in self._model.parameters() if p.requires_grad]
    
    def save_lora_weights(self, path: str):
        """Zapisuje tylko wagi LoRA"""
        if self._lora_applied:
            self._model.save_pretrained(path)
            print(f"  LoRA weights saved to {path}")
    
    def load_lora_weights(self, path: str):
        """Wczytuje wagi LoRA"""
        if self.use_lora:
            from peft import PeftModel
            self._load_model()  # Ensure base model is loaded
            if self._model != "dummy":
                self._model = PeftModel.from_pretrained(self._model, path)
                print(f"  LoRA weights loaded from {path}")
    
    @torch.no_grad()
    def encode_text(self, texts: List[str]) -> torch.Tensor:
        """
        Enkoduje teksty do embeddingów CLAP.
        
        Returns:
            [B, output_dim] embeddings
        """
        self._load_model()
        
        if self._model == "dummy":
            return torch.randn(len(texts), self.output_dim, device=self.device)
        
        inputs = self._processor(
            text=texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        outputs = self._model.get_text_features(**inputs)
        return outputs
    
    @torch.no_grad()
    def encode_audio(
        self, 
        audio: torch.Tensor,
        sample_rate: int = 48000,
    ) -> torch.Tensor:
        """
        Enkoduje audio do embeddingów CLAP.
        
        Args:
            audio: [B, samples] waveform
            sample_rate: sample rate audio
            
        Returns:
            [B, output_dim] embeddings
        """
        self._load_model()
        
        if self._model == "dummy":
            return torch.randn(audio.shape[0], self.output_dim, device=self.device)
        
        # CLAP oczekuje 48kHz
        if sample_rate != 48000:
            import torchaudio
            resampler = torchaudio.transforms.Resample(sample_rate, 48000).to(audio.device)
            audio = resampler(audio)
        
        # Process each audio in batch
        audio_np = audio.cpu().numpy()
        inputs = self._processor(
            audios=list(audio_np),
            return_tensors="pt",
            sampling_rate=48000,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        outputs = self._model.get_audio_features(**inputs)
        return outputs
    
    def get_similarity(
        self,
        text_embed: torch.Tensor,
        audio_embed: torch.Tensor,
    ) -> torch.Tensor:
        """
        Oblicza podobieństwo między embeddingami tekstu i audio.
        
        Returns:
            [B_text, B_audio] cosine similarity matrix
        """
        text_embed = F.normalize(text_embed, dim=-1)
        audio_embed = F.normalize(audio_embed, dim=-1)
        return torch.matmul(text_embed, audio_embed.T)


class SectionConditioner(nn.Module):
    """
    Kondycjonuje embedding informacją o sekcji.
    
    Dodaje do embeddingu:
    - Typ sekcji (verse, chorus, etc.)
    - Pozycję w utworze (0-1)
    - Cechy muzyczne (tempo, energy, key)
    """
    
    SECTION_TYPES = [
        'intro', 'verse', 'pre_chorus', 'chorus', 'post_chorus',
        'bridge', 'instrumental', 'solo', 'breakdown', 'buildup',
        'drop', 'outro', 'unknown'
    ]
    
    def __init__(
        self,
        input_dim: int = 512,
        output_dim: int = 768,
        num_keys: int = 24,
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        # Section type embedding
        self.section_embed = nn.Embedding(len(self.SECTION_TYPES), 128)
        self.section_to_idx = {s: i for i, s in enumerate(self.SECTION_TYPES)}
        
        # Position embedding (continuous)
        self.position_embed = nn.Sequential(
            nn.Linear(1, 64),
            nn.SiLU(),
            nn.Linear(64, 128),
        )
        
        # Musical attributes embedding
        # tempo (normalized), energy, key_index
        self.attr_embed = nn.Sequential(
            nn.Linear(3, 64),
            nn.SiLU(),
            nn.Linear(64, 128),
        )
        
        # Key embedding
        self.key_embed = nn.Embedding(num_keys, 64)
        
        # Fusion layer
        # input_dim + section(128) + position(128) + attr(128) + key(64)
        fusion_dim = input_dim + 128 + 128 + 128 + 64
        
        self.fusion = nn.Sequential(
            nn.Linear(fusion_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.SiLU(),
            nn.Linear(output_dim, output_dim),
        )
    
    def forward(
        self,
        text_embed: torch.Tensor,           # [B, input_dim]
        section_type: List[str],            # ['verse', 'chorus', ...]
        position: torch.Tensor,             # [B] (0-1)
        tempo: Optional[torch.Tensor] = None,  # [B] BPM
        energy: Optional[torch.Tensor] = None, # [B] (0-1)
        key_idx: Optional[torch.Tensor] = None, # [B] key index
    ) -> torch.Tensor:
        """
        Łączy embedding tekstu z informacją o sekcji.
        
        Returns:
            [B, output_dim] conditioned embedding
        """
        B = text_embed.shape[0]
        device = text_embed.device
        
        # Section embedding
        section_indices = torch.tensor(
            [self.section_to_idx.get(s.lower(), self.section_to_idx['unknown']) 
             for s in section_type],
            device=device
        )
        section_emb = self.section_embed(section_indices)  # [B, 128]
        
        # Position embedding
        pos_emb = self.position_embed(position.unsqueeze(-1))  # [B, 128]
        
        # Musical attributes
        if tempo is None:
            tempo = torch.ones(B, device=device) * 120
        if energy is None:
            energy = torch.ones(B, device=device) * 0.5
        
        tempo_norm = (tempo - 60) / 140  # Normalize to ~0-1
        attr_input = torch.stack([tempo_norm, energy, torch.zeros_like(tempo)], dim=-1)
        attr_emb = self.attr_embed(attr_input)  # [B, 128]
        
        # Key embedding
        if key_idx is None:
            key_idx = torch.zeros(B, dtype=torch.long, device=device)
        key_emb = self.key_embed(key_idx)  # [B, 64]
        
        # Concatenate and fuse
        combined = torch.cat([
            text_embed,
            section_emb,
            pos_emb,
            attr_emb,
            key_emb,
        ], dim=-1)
        
        output = self.fusion(combined)
        
        return output


class EnhancedMusicEncoder(nn.Module):
    """
    Główny encoder v2 - łączy CLAP, section conditioning, i audio reference.
    
    Użycie:
        encoder = EnhancedMusicEncoder()
        
        # Dla pojedynczego segmentu
        embed = encoder(
            texts=["powerful chorus with soaring vocals"],
            section_types=["chorus"],
            positions=torch.tensor([0.5]),
            tempos=torch.tensor([128.0]),
            energies=torch.tensor([0.9]),
        )
        
        # Z audio reference (dla style transfer)
        embed, style = encoder(
            texts=["..."],
            ...,
            reference_audio=audio_tensor,
        )
    """
    
    def __init__(
        self,
        clap_model: str = "laion/clap-htsat-unfused",
        output_dim: int = 768,
        use_clap: bool = True,
        use_t5_fallback: bool = True,
        device: str = 'cpu',
    ):
        super().__init__()
        
        self.output_dim = output_dim
        self.device = device
        self.use_clap = use_clap
        
        # CLAP encoder
        if use_clap:
            self.clap = CLAPEncoder(clap_model, device=device)
            clap_dim = 512
        else:
            self.clap = None
            clap_dim = 768
            
            # T5 fallback
            if use_t5_fallback:
                from transformers import T5EncoderModel, T5Tokenizer
                import warnings
                print("Using T5 fallback instead of CLAP")
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", category=FutureWarning)
                    self.t5_tokenizer = T5Tokenizer.from_pretrained(
                        "t5-base",
                        model_max_length=512,
                        legacy=False,
                    )
                self.t5_model = T5EncoderModel.from_pretrained("t5-base")
                self.t5_model = self.t5_model.to(device)
                for param in self.t5_model.parameters():
                    param.requires_grad = False
        
        # Section conditioner
        self.section_conditioner = SectionConditioner(
            input_dim=clap_dim,
            output_dim=output_dim,
        )
        
        # Audio reference projection (for style embedding)
        self.audio_proj = nn.Linear(clap_dim, output_dim)
        
        # Cross-attention for merging text and audio reference
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=output_dim,
            num_heads=8,
            batch_first=True,
        )
        
        # Final projection
        self.output_proj = nn.Linear(output_dim, output_dim)
    
    def encode_text_only(self, texts: List[str]) -> torch.Tensor:
        """Enkoduje tylko tekst (bez section conditioning)"""
        if self.clap is not None:
            return self.clap.encode_text(texts)
        else:
            # T5 fallback
            inputs = self.t5_tokenizer(
                texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512,
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.t5_model(**inputs)
            
            # Mean pooling
            return outputs.last_hidden_state.mean(dim=1)
    
    def encode_audio_reference(
        self, 
        audio: torch.Tensor,
        sample_rate: int = 22050,
    ) -> torch.Tensor:
        """
        Enkoduje audio reference do style embedding.
        
        Args:
            audio: [B, samples] lub [samples]
            sample_rate: sample rate
            
        Returns:
            [B, output_dim] style embedding
        """
        if audio.dim() == 1:
            audio = audio.unsqueeze(0)
        
        if self.clap is not None:
            audio_embed = self.clap.encode_audio(audio, sample_rate)
        else:
            # Dummy
            audio_embed = torch.randn(audio.shape[0], 768, device=audio.device)
        
        return self.audio_proj(audio_embed)
    
    def forward(
        self,
        texts: List[str],
        section_types: List[str],
        positions: torch.Tensor,
        tempos: Optional[torch.Tensor] = None,
        energies: Optional[torch.Tensor] = None,
        key_indices: Optional[torch.Tensor] = None,
        reference_audio: Optional[torch.Tensor] = None,
        reference_sample_rate: int = 22050,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass - enkoduje tekst z section conditioning.
        
        Args:
            texts: Lista promptów tekstowych
            section_types: Lista typów sekcji
            positions: [B] pozycje w utworze (0-1)
            tempos: [B] tempo w BPM
            energies: [B] energia (0-1)
            key_indices: [B] indeksy tonacji
            reference_audio: [B, samples] opcjonalne audio reference
            reference_sample_rate: sample rate reference audio
            
        Returns:
            [B, output_dim] conditioned embedding
            lub (embedding, style_embedding) jeśli podano reference_audio
        """
        # Text encoding
        text_embed = self.encode_text_only(texts)  # [B, clap_dim]
        
        # Section conditioning
        conditioned = self.section_conditioner(
            text_embed,
            section_types,
            positions,
            tempos,
            energies,
            key_indices,
        )  # [B, output_dim]
        
        # Reference audio (style transfer)
        if reference_audio is not None:
            style_embed = self.encode_audio_reference(
                reference_audio,
                reference_sample_rate,
            )  # [B, output_dim]
            
            # Cross-attention: conditioned jako query, style jako key/value
            conditioned = conditioned.unsqueeze(1)  # [B, 1, output_dim]
            style_embed_kv = style_embed.unsqueeze(1)  # [B, 1, output_dim]
            
            attended, _ = self.cross_attn(
                conditioned,
                style_embed_kv,
                style_embed_kv,
            )
            
            conditioned = (conditioned + attended).squeeze(1)  # [B, output_dim]
            
            output = self.output_proj(conditioned)
            return output, style_embed
        
        output = self.output_proj(conditioned)
        return output


class SequenceEncoder(nn.Module):
    """
    Enkoder dla sekwencji sekcji.
    
    Zamiast enkodować pojedyncze sekcje, enkoduje całą strukturę utworu:
    [Intro -> Verse -> Chorus -> ...]
    
    Używa lightweight transformer do modelowania zależności między sekcjami.
    """
    
    def __init__(
        self,
        section_encoder: EnhancedMusicEncoder,
        d_model: int = 768,
        nhead: int = 8,
        num_layers: int = 2,
        max_sections: int = 20,
    ):
        super().__init__()
        
        self.section_encoder = section_encoder
        self.d_model = d_model
        
        # Positional encoding for section sequence
        self.pos_embedding = nn.Parameter(torch.randn(1, max_sections, d_model) * 0.02)
        
        # Transformer for modeling dependencies
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 2,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Global summary token
        self.global_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
    
    def forward(
        self,
        texts: List[List[str]],           # [B, num_sections]
        section_types: List[List[str]],   # [B, num_sections]
        positions: torch.Tensor,          # [B, num_sections]
        tempos: Optional[torch.Tensor] = None,
        energies: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encodes section sequence.
        
        Returns:
            global_embed: [B, d_model] global summary
            section_embeds: [B, num_sections, d_model] embeddings per section
        """
        B = len(texts)
        num_sections = len(texts[0])
        device = positions.device
        
        # Encode each section separately
        section_embeds = []
        for i in range(num_sections):
            section_texts = [t[i] for t in texts]
            section_types_i = [s[i] for s in section_types]
            positions_i = positions[:, i]
            
            tempos_i = tempos[:, i] if tempos is not None else None
            energies_i = energies[:, i] if energies is not None else None
            
            embed = self.section_encoder(
                section_texts,
                section_types_i,
                positions_i,
                tempos_i,
                energies_i,
            )
            section_embeds.append(embed)
        
        section_embeds = torch.stack(section_embeds, dim=1)  # [B, num_sections, d_model]
        
        # Add positional embedding
        section_embeds = section_embeds + self.pos_embedding[:, :num_sections]
        
        # Add global token
        global_token = self.global_token.expand(B, -1, -1)  # [B, 1, d_model]
        sequence = torch.cat([global_token, section_embeds], dim=1)  # [B, 1+num_sections, d_model]
        
        # Transformer
        output = self.transformer(sequence)
        
        # Extract outputs
        global_embed = output[:, 0]  # [B, d_model]
        section_embeds = output[:, 1:]  # [B, num_sections, d_model]
        
        return global_embed, section_embeds


if __name__ == "__main__":
    print("🎵 Testing Enhanced Music Encoder v2")
    print("="*60)
    
    device = 'cpu'
    
    # Test bez CLAP (szybciej dla testu)
    print("\n📝 Testing EnhancedMusicEncoder (without CLAP)...")
    encoder = EnhancedMusicEncoder(use_clap=False, use_t5_fallback=False, device=device)
    
    # Test inputs
    texts = [
        "powerful chorus with soaring vocals and big drums",
        "quiet verse with acoustic guitar",
    ]
    section_types = ["chorus", "verse"]
    positions = torch.tensor([0.5, 0.2])
    tempos = torch.tensor([128.0, 90.0])
    energies = torch.tensor([0.9, 0.3])
    
    # Forward
    output = encoder(texts, section_types, positions, tempos, energies)
    print(f"  Output shape: {output.shape}")
    
    # Test z audio reference
    print("\n🎸 Testing with audio reference...")
    dummy_audio = torch.randn(2, 22050 * 5)  # 5 seconds
    output, style = encoder(
        texts, section_types, positions, tempos, energies,
        reference_audio=dummy_audio,
    )
    print(f"  Output shape: {output.shape}")
    print(f"  Style shape: {style.shape}")
    
    # Test CLAP (if available)
    print("\n🔊 Testing CLAP encoder (if available)...")
    clap = CLAPEncoder(device=device)
    
    text_embed = clap.encode_text(texts)
    print(f"  Text embedding shape: {text_embed.shape}")
    
    audio_embed = clap.encode_audio(dummy_audio)
    print(f"  Audio embedding shape: {audio_embed.shape}")
    
    # Similarity
    sim = clap.get_similarity(text_embed, audio_embed)
    print(f"  Similarity matrix shape: {sim.shape}")
    
    # Model size
    params = sum(p.numel() for p in encoder.parameters())
    print(f"\n📊 EnhancedMusicEncoder parameters: {params:,}")
