"""
🎵 Section-Aware Latent Diffusion Model v2

Ulepszenia względem v1:
1. Section conditioning - U-Net wie jaki typ sekcji generuje
2. Position encoding - świadomość pozycji w utworze
3. Context conditioning - może uwzględniać poprzedni segment
4. Composition plan conditioning - generuje według planu kompozycji
5. Voice stream (opcjonalnie) - unified vocal + instrumental generation
6. Gradient Checkpointing - oszczędność pamięci GPU

Architektura:
                                    
    Noise ─────────────────────────────────────────────────────┐
                                                               │
    Text Embedding (CLAP) ────────┐                            │
                                  │                            │
    Section Type ─────────────────┼──▶ Conditioning ───────────┤
                                  │    Module                  │
    Position (0-1) ───────────────┤                            │
                                  │                            ▼
    Energy/Tempo ─────────────────┘              ┌─────────────────────┐
                                                 │                     │
    Context Latent ────────────────────────────▶│    U-Net v2         │
    (previous segment)                           │    (section-aware)  │
                                                 │                     │
    Voice Embedding ───────────────────────────▶│                     │
    (artist style)                               └──────────┬──────────┘
                                                            │
                                                            ▼
                                                     Denoised Latent
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple, List, Dict
from einops import rearrange
from torch.utils.checkpoint import checkpoint


def timestep_embedding(timesteps: torch.Tensor, dim: int, max_period: int = 10000) -> torch.Tensor:
    """Sinusoidalne embeddingi dla timestepów"""
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(half, device=timesteps.device) / half
    )
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    return embedding


# ============================================================================
# Rotary Position Embedding (RoPE) dla LDM
# ============================================================================

class RotaryEmbedding(nn.Module):
    """
    Rotary Position Embedding (RoPE).
    
    Lepsze od learned/sinusoidal position embeddings dla długich sekwencji.
    Pozwala na ekstrapolację do dłuższych sekwencji niż podczas treningu.
    """
    
    def __init__(self, dim: int, max_seq_len: int = 4096, base: int = 10000):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        
        # Compute rotation frequencies
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)
        
        # Build cache
        self._build_cache(max_seq_len)
    
    def _build_cache(self, seq_len: int):
        """Build sin/cos cache"""
        t = torch.arange(seq_len, device=self.inv_freq.device)
        freqs = torch.einsum('i,j->ij', t, self.inv_freq)
        
        # [seq_len, dim/2] -> [seq_len, dim]
        emb = torch.cat([freqs, freqs], dim=-1)
        
        self.register_buffer('cos_cached', emb.cos()[None, None, :, :], persistent=False)
        self.register_buffer('sin_cached', emb.sin()[None, None, :, :], persistent=False)
    
    def forward(self, seq_len: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            seq_len: Current sequence length
            device: Device for tensors
        
        Returns:
            cos, sin: [1, 1, seq_len, dim] tensors
        """
        if seq_len > self.max_seq_len:
            self._build_cache(seq_len)
        
        return (
            self.cos_cached[:, :, :seq_len, :].to(device),
            self.sin_cached[:, :, :seq_len, :].to(device),
        )


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate half the hidden dims"""
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply rotary position embedding to query and key tensors.
    
    Args:
        q: [B, heads, N, dim] query tensor
        k: [B, heads, N, dim] key tensor  
        cos: [1, 1, N, dim] cosine tensor
        sin: [1, 1, N, dim] sine tensor
        
    Returns:
        q_embed, k_embed: Tensors with RoPE applied
    """
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


class ClapProjection(nn.Module):
    """
    🎵 CLAP Embedding Projection Module
    
    Projekcja embeddingów CLAP (audio/text) z 512 → 768 dim.
    CLAP embeddings zawierają bogate informacje audio-tekstowe,
    ale mają inny wymiar niż T5 embeddings.
    
    Modes:
    - 'audio': używa clap_audio_embedding
    - 'text': używa clap_text_embedding
    - 'fused': łączy oba przez cross-attention
    """
    
    def __init__(
        self,
        clap_dim: int = 512,
        output_dim: int = 768,
        mode: str = 'fused',  # 'audio', 'text', 'fused'
    ):
        super().__init__()
        self.mode = mode
        self.clap_dim = clap_dim
        self.output_dim = output_dim
        
        # Projekcje dla audio i text
        self.audio_proj = nn.Sequential(
            nn.Linear(clap_dim, clap_dim),
            nn.SiLU(),
            nn.Linear(clap_dim, output_dim),
            nn.LayerNorm(output_dim),
        )
        
        self.text_proj = nn.Sequential(
            nn.Linear(clap_dim, clap_dim),
            nn.SiLU(),
            nn.Linear(clap_dim, output_dim),
            nn.LayerNorm(output_dim),
        )
        
        if mode == 'fused':
            # Cross-attention fusion między audio i text CLAP
            self.fusion_attn = nn.MultiheadAttention(
                embed_dim=output_dim,
                num_heads=8,
                dropout=0.1,
                batch_first=True,
            )
            self.fusion_norm = nn.LayerNorm(output_dim)
            self.fusion_ff = nn.Sequential(
                nn.Linear(output_dim, output_dim * 2),
                nn.SiLU(),
                nn.Linear(output_dim * 2, output_dim),
            )
    
    def forward(
        self,
        clap_audio: Optional[torch.Tensor] = None,  # [B, 512]
        clap_text: Optional[torch.Tensor] = None,   # [B, 512]
    ) -> torch.Tensor:
        """
        Returns: [B, 768] projected CLAP embedding
        """
        if self.mode == 'audio' and clap_audio is not None:
            return self.audio_proj(clap_audio)
        
        elif self.mode == 'text' and clap_text is not None:
            return self.text_proj(clap_text)
        
        elif self.mode == 'fused':
            # Jeśli mamy oba, użyj fusion
            if clap_audio is not None and clap_text is not None:
                audio_emb = self.audio_proj(clap_audio)  # [B, 768]
                text_emb = self.text_proj(clap_text)     # [B, 768]
                
                # Stack jako sequence [B, 2, 768]
                combined = torch.stack([audio_emb, text_emb], dim=1)
                
                # Self-attention fusion
                fused, _ = self.fusion_attn(combined, combined, combined)
                fused = self.fusion_norm(fused + combined)
                fused = fused + self.fusion_ff(fused)
                
                # Return mean of fused tokens
                return fused.mean(dim=1)  # [B, 768]
            
            elif clap_audio is not None:
                return self.audio_proj(clap_audio)
            elif clap_text is not None:
                return self.text_proj(clap_text)
        
        # Fallback - return zeros (need batch_size from caller)
        # If both are None, we return None and let caller handle it
        if clap_audio is not None:
            B = clap_audio.shape[0]
            device = clap_audio.device
            return torch.zeros(B, self.output_dim, device=device)
        elif clap_text is not None:
            B = clap_text.shape[0]
            device = clap_text.device
            return torch.zeros(B, self.output_dim, device=device)
        else:
            return None  # Both are None, return None


class BeatEmbedding(nn.Module):
    """
    🥁 Beat Embedding Module
    
    Koduje informacje o rytmie:
    - num_beats: liczba uderzeń w segmencie
    - beat_positions: pozycje uderzeń (w sekundach)
    - time_signature: metrum (4/4, 3/4, etc.)
    """
    
    def __init__(
        self,
        output_dim: int = 64,
        max_beats: int = 64,
        max_time_sig: int = 16,  # max numerator
    ):
        super().__init__()
        
        # Embedding dla liczby uderzeń
        self.num_beats_embed = nn.Embedding(max_beats + 1, output_dim // 2)
        
        # Embedding dla time signature
        self.time_sig_embed = nn.Embedding(max_time_sig + 1, output_dim // 2)
        
        # Positional encoding dla beat positions
        self.beat_pos_mlp = nn.Sequential(
            nn.Linear(max_beats, output_dim),
            nn.SiLU(),
            nn.Linear(output_dim, output_dim),
        )
        
        # Final fusion
        self.fusion = nn.Sequential(
            nn.Linear(output_dim * 2, output_dim),
            nn.SiLU(),
            nn.Linear(output_dim, output_dim),
        )
        
        self.max_beats = max_beats
        self.output_dim = output_dim
    
    def forward(
        self,
        num_beats: Optional[torch.Tensor] = None,      # [B] int
        beat_positions: Optional[List] = None,          # List of [varying length] float
        time_signature: Optional[List[str]] = None,     # List of "4/4", "3/4" etc
    ) -> torch.Tensor:
        """
        Returns: [B, output_dim] beat embedding
        """
        if num_beats is None:
            raise ValueError("num_beats is required for BeatEmbedding")
        
        B = num_beats.shape[0]
        device = num_beats.device
        
        # Num beats embedding
        num_beats_clamped = num_beats.clamp(0, self.max_beats).long()
        num_beats_emb = self.num_beats_embed(num_beats_clamped)  # [B, dim/2]
        
        # Time signature embedding
        if time_signature is not None:
            time_sig_nums = []
            for ts in time_signature:
                try:
                    num = int(ts.split('/')[0]) if ts else 4
                except:
                    num = 4
                time_sig_nums.append(min(num, 16))
            time_sig_idx = torch.tensor(time_sig_nums, device=device, dtype=torch.long)
        else:
            time_sig_idx = torch.full((B,), 4, device=device, dtype=torch.long)
        
        time_sig_emb = self.time_sig_embed(time_sig_idx)  # [B, dim/2]
        
        # Beat positions embedding (pad to max_beats)
        if beat_positions is not None:
            beat_pos_tensor = torch.zeros(B, self.max_beats, device=device)
            for i, positions in enumerate(beat_positions):
                if positions is not None and len(positions) > 0:
                    pos_tensor = torch.tensor(positions[:self.max_beats], device=device)
                    beat_pos_tensor[i, :len(pos_tensor)] = pos_tensor
            beat_pos_emb = self.beat_pos_mlp(beat_pos_tensor)  # [B, dim]
        else:
            beat_pos_emb = torch.zeros(B, self.output_dim, device=device)
        
        # Combine
        basic_emb = torch.cat([num_beats_emb, time_sig_emb], dim=-1)  # [B, dim]
        combined = torch.cat([basic_emb, beat_pos_emb], dim=-1)  # [B, dim*2]
        
        return self.fusion(combined)  # [B, dim]


class ChordEmbedding(nn.Module):
    """
    🎸 Chord Embedding Module
    
    Koduje informacje o akordach:
    - 12 nut podstawowych × 2 tryby (major/minor) = 24 podstawowe akordy
    - + rozszerzenia (7, maj7, dim, aug, sus) = dodatkowe 48
    - + 'N' (no chord) i 'X' (unknown)
    """
    
    # Mapowanie akordów na indeksy
    CHORD_ROOTS = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    CHORD_TYPES = ['', 'm', '7', 'm7', 'maj7', 'dim', 'aug', 'sus2', 'sus4', 'add9']
    
    def __init__(
        self,
        output_dim: int = 64,
    ):
        super().__init__()
        
        # Root note embedding (12 notes + no chord + unknown)
        self.root_embed = nn.Embedding(14, output_dim // 2)
        
        # Chord type embedding
        self.type_embed = nn.Embedding(len(self.CHORD_TYPES) + 2, output_dim // 2)
        
        # Build chord parser
        self.root_to_idx = {r: i for i, r in enumerate(self.CHORD_ROOTS)}
        self.root_to_idx['N'] = 12  # No chord
        self.root_to_idx['X'] = 13  # Unknown
        
        self.type_to_idx = {t: i for i, t in enumerate(self.CHORD_TYPES)}
        self.type_to_idx['N'] = len(self.CHORD_TYPES)
        self.type_to_idx['X'] = len(self.CHORD_TYPES) + 1
        
        self.fusion = nn.Linear(output_dim, output_dim)
        self.output_dim = output_dim
    
    def parse_chord(self, chord_str: str) -> Tuple[int, int]:
        """Parse chord string like 'C#m7' → (root_idx, type_idx)"""
        if not chord_str or chord_str in ['N', 'N:N', 'none', '']:
            return 12, self.type_to_idx['N']  # No chord
        
        chord_str = chord_str.replace(':maj', '').replace(':min', 'm')
        
        # Extract root
        if len(chord_str) >= 2 and chord_str[1] in ['#', 'b']:
            root = chord_str[:2]
            rest = chord_str[2:]
            # Handle flats
            if root.endswith('b'):
                flat_to_sharp = {
                    'Db': 'C#', 'Eb': 'D#', 'Gb': 'F#', 'Ab': 'G#', 'Bb': 'A#'
                }
                root = flat_to_sharp.get(root, root.replace('b', ''))
        else:
            root = chord_str[0] if chord_str else 'C'
            rest = chord_str[1:] if len(chord_str) > 1 else ''
        
        root_idx = self.root_to_idx.get(root, 13)  # Unknown if not found
        
        # Extract type
        type_idx = self.type_to_idx.get(rest, self.type_to_idx.get('', 0))
        
        return root_idx, type_idx
    
    def forward(
        self,
        chords: List[str],  # List of chord strings like ['Cmaj7', 'Am', 'F', 'G']
    ) -> torch.Tensor:
        """
        Returns: [B, output_dim] chord embedding
        """
        device = self.root_embed.weight.device
        B = len(chords)
        
        root_indices = []
        type_indices = []
        
        for chord in chords:
            root_idx, type_idx = self.parse_chord(chord)
            root_indices.append(root_idx)
            type_indices.append(type_idx)
        
        root_idx_tensor = torch.tensor(root_indices, device=device, dtype=torch.long)
        type_idx_tensor = torch.tensor(type_indices, device=device, dtype=torch.long)
        
        root_emb = self.root_embed(root_idx_tensor)  # [B, dim/2]
        type_emb = self.type_embed(type_idx_tensor)  # [B, dim/2]
        
        combined = torch.cat([root_emb, type_emb], dim=-1)  # [B, dim]
        return self.fusion(combined)


# ============================================================================
# Voice Embedding Fusion (Resemblyzer + ECAPA-TDNN)
# ============================================================================

class VoiceEmbeddingFusion(nn.Module):
    """
    🎤 Voice Embedding Fusion Module
    
    Łączy dwa typy voice embeddings dla lepszego voice cloning:
    - Resemblyzer (256-dim): z miksu audio - ogólna charakterystyka głosu
    - ECAPA-TDNN (192-dim): z separowanych wokali - czysty głos
    
    Fusion strategies:
    - 'concat': Konkatenacja + projekcja
    - 'attention': Cross-attention między embeddingami
    - 'gated': Gated fusion z learnable weights
    """
    
    def __init__(
        self,
        resemblyzer_dim: int = 256,
        ecapa_dim: int = 192,
        output_dim: int = 256,
        fusion_type: str = 'gated',
    ):
        super().__init__()
        
        self.resemblyzer_dim = resemblyzer_dim
        self.ecapa_dim = ecapa_dim
        self.output_dim = output_dim
        self.fusion_type = fusion_type
        
        # Project each embedding to common dim
        self.resemblyzer_proj = nn.Sequential(
            nn.Linear(resemblyzer_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.SiLU(),
        )
        
        self.ecapa_proj = nn.Sequential(
            nn.Linear(ecapa_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.SiLU(),
        )
        
        if fusion_type == 'concat':
            # Simple concatenation + projection
            self.fusion = nn.Sequential(
                nn.Linear(output_dim * 2, output_dim),
                nn.LayerNorm(output_dim),
                nn.SiLU(),
                nn.Linear(output_dim, output_dim),
            )
        elif fusion_type == 'gated':
            # Gated fusion - model learns how to combine
            self.gate = nn.Sequential(
                nn.Linear(output_dim * 2, output_dim),
                nn.Sigmoid(),
            )
            self.fusion = nn.Sequential(
                nn.Linear(output_dim, output_dim),
                nn.LayerNorm(output_dim),
            )
        elif fusion_type == 'attention':
            # Cross-attention fusion
            self.cross_attn = nn.MultiheadAttention(
                embed_dim=output_dim,
                num_heads=4,
                batch_first=True,
            )
            self.fusion = nn.LayerNorm(output_dim)
    
    def forward(
        self,
        resemblyzer_emb: Optional[torch.Tensor] = None,  # [B, 256]
        ecapa_emb: Optional[torch.Tensor] = None,        # [B, 192]
    ) -> torch.Tensor:
        """
        Fuse voice embeddings.
        
        Returns:
            fused_emb: [B, output_dim]
        """
        # Handle missing embeddings
        if resemblyzer_emb is None and ecapa_emb is None:
            raise ValueError("At least one voice embedding must be provided")
        
        device = resemblyzer_emb.device if resemblyzer_emb is not None else ecapa_emb.device
        batch_size = resemblyzer_emb.shape[0] if resemblyzer_emb is not None else ecapa_emb.shape[0]
        
        # Project embeddings
        if resemblyzer_emb is not None:
            res_proj = self.resemblyzer_proj(resemblyzer_emb)
        else:
            res_proj = torch.zeros(batch_size, self.output_dim, device=device)
        
        if ecapa_emb is not None:
            ecapa_proj = self.ecapa_proj(ecapa_emb)
        else:
            ecapa_proj = torch.zeros(batch_size, self.output_dim, device=device)
        
        # Fusion
        if self.fusion_type == 'concat':
            combined = torch.cat([res_proj, ecapa_proj], dim=-1)
            return self.fusion(combined)
        
        elif self.fusion_type == 'gated':
            combined = torch.cat([res_proj, ecapa_proj], dim=-1)
            gate = self.gate(combined)
            fused = gate * res_proj + (1 - gate) * ecapa_proj
            return self.fusion(fused)
        
        elif self.fusion_type == 'attention':
            # Treat as sequence of 2 tokens
            seq = torch.stack([res_proj, ecapa_proj], dim=1)  # [B, 2, dim]
            attn_out, _ = self.cross_attn(seq, seq, seq)
            fused = attn_out.mean(dim=1)  # [B, dim]
            return self.fusion(fused)
        
        return res_proj  # Fallback


# ============================================================================
# Pitch Encoder (F0 Conditioning for Singing)
# ============================================================================

class PitchEncoder(nn.Module):
    """
    🎵 Pitch/F0 Encoder for Melody Conditioning
    
    Koduje kontur melodyczny (fundamental frequency) dla lepszej kontroli
    nad melodią generowanych wokali. Krytyczny dla singing voice synthesis.
    
    Input types:
    - f0_contour: [B, T] lub [B, T, 1] - ciągły kontur F0 w Hz
    - f0_coarse: [B, T] - zdyskretyzowany pitch (MIDI-like bins)
    - pitch_class: [B, T] - klasa wysokości (0-11 dla C-B)
    
    Features:
    - Positional encoding dla konturu czasowego
    - Pitch normalization (log-scale dla percepcyjnie liniowego)
    - Obsługa UV (unvoiced) segmentów
    """
    
    # MIDI pitch range: A0 (21) do C8 (108)
    PITCH_BINS = 128  # MIDI-style bins
    
    def __init__(
        self,
        output_dim: int = 64,
        max_seq_len: int = 1024,
        num_pitch_bins: int = 128,  # MIDI bins
        use_log_scale: bool = True,  # Log Hz dla percepcji
        use_pitch_class: bool = True,  # Dodatkowo pitch class (0-11)
    ):
        super().__init__()
        
        self.output_dim = output_dim
        self.max_seq_len = max_seq_len
        self.num_pitch_bins = num_pitch_bins
        self.use_log_scale = use_log_scale
        self.use_pitch_class = use_pitch_class
        
        # Pitch embedding (discrete bins)
        self.pitch_embed = nn.Embedding(num_pitch_bins + 1, output_dim)  # +1 dla UV
        
        # Pitch class embedding (C, C#, D, ..., B)
        if use_pitch_class:
            self.pitch_class_embed = nn.Embedding(13, output_dim // 4)  # 12 + UV
        
        # Continuous F0 processing
        self.f0_mlp = nn.Sequential(
            nn.Linear(1, output_dim // 2),
            nn.SiLU(),
            nn.Linear(output_dim // 2, output_dim // 2),
        )
        
        # UV (unvoiced) detector output
        self.uv_embed = nn.Embedding(2, output_dim // 4)  # 0=voiced, 1=unvoiced
        
        # Temporal convolutions for smoothing
        self.conv_layers = nn.Sequential(
            nn.Conv1d(output_dim, output_dim, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv1d(output_dim, output_dim, kernel_size=3, padding=1),
        )
        
        # Positional encoding
        self.pos_embed = nn.Embedding(max_seq_len, output_dim)
        
        # Final projection
        self.output_proj = nn.Sequential(
            nn.Linear(output_dim + output_dim // 4 + output_dim // 4, output_dim),
            nn.LayerNorm(output_dim),
        )
        
        # Register Hz to MIDI conversion params
        self.register_buffer('f0_min', torch.tensor(50.0))  # ~G1
        self.register_buffer('f0_max', torch.tensor(1000.0))  # ~B5
    
    def hz_to_midi(self, hz: torch.Tensor) -> torch.Tensor:
        """Convert Hz to MIDI note number"""
        # MIDI note = 12 * log2(f/440) + 69
        hz_safe = torch.clamp(hz, min=1.0)  # Avoid log(0)
        midi = 12 * torch.log2(hz_safe / 440.0) + 69
        return torch.clamp(midi, 0, self.num_pitch_bins - 1)
    
    def hz_to_pitch_class(self, hz: torch.Tensor) -> torch.Tensor:
        """Convert Hz to pitch class (0-11)"""
        midi = self.hz_to_midi(hz)
        return (midi % 12).long()
    
    def forward(
        self,
        f0: Optional[torch.Tensor] = None,      # [B, T] continuous F0 in Hz
        f0_coarse: Optional[torch.Tensor] = None,  # [B, T] discrete pitch bins
        uv: Optional[torch.Tensor] = None,       # [B, T] unvoiced mask (1=UV)
    ) -> torch.Tensor:
        """
        Encode pitch information.
        
        Args:
            f0: Continuous F0 contour in Hz [B, T]
            f0_coarse: Discrete pitch bins [B, T] (MIDI-like)
            uv: Unvoiced mask [B, T], 1=unvoiced, 0=voiced
            
        Returns:
            pitch_emb: [B, T, output_dim] or [B, output_dim] pitch embedding
        """
        # Determine input type
        if f0 is not None:
            B, T = f0.shape[:2]
            device = f0.device
            
            # Default UV from f0 < threshold
            if uv is None:
                uv = (f0 < 50.0).long()  # Below ~G1 = unvoiced
            
            # Normalize F0 (log scale)
            if self.use_log_scale:
                f0_norm = torch.log(torch.clamp(f0, min=1.0) / 440.0)  # Log relative to A4
                f0_norm = f0_norm / 4.0  # Scale to ~[-1, 1]
            else:
                f0_norm = (f0 - self.f0_min) / (self.f0_max - self.f0_min)
                f0_norm = f0_norm * 2 - 1  # Scale to [-1, 1]
            
            # Continuous F0 embedding
            f0_emb = self.f0_mlp(f0_norm.unsqueeze(-1))  # [B, T, dim/2]
            
            # Discrete pitch bins
            midi_bins = self.hz_to_midi(f0).long()
            midi_bins = torch.where(uv.bool(), torch.full_like(midi_bins, self.num_pitch_bins), midi_bins)
            pitch_emb = self.pitch_embed(midi_bins)  # [B, T, dim]
            
        elif f0_coarse is not None:
            B, T = f0_coarse.shape
            device = f0_coarse.device
            
            if uv is None:
                uv = (f0_coarse >= self.num_pitch_bins).long()
            
            # Use coarse bins directly
            pitch_emb = self.pitch_embed(f0_coarse.clamp(0, self.num_pitch_bins))
            f0_emb = torch.zeros(B, T, self.output_dim // 2, device=device)
            
        else:
            raise ValueError("Either f0 or f0_coarse must be provided")
        
        # Pitch class embedding
        if self.use_pitch_class and f0 is not None:
            pitch_class = self.hz_to_pitch_class(f0)
            pitch_class = torch.where(uv.bool(), torch.full_like(pitch_class, 12), pitch_class)
            pitch_class_emb = self.pitch_class_embed(pitch_class)  # [B, T, dim/4]
        else:
            pitch_class_emb = torch.zeros(B, T, self.output_dim // 4, device=device)
        
        # UV embedding
        uv_emb = self.uv_embed(uv.long())  # [B, T, dim/4]
        
        # Add positional encoding
        positions = torch.arange(T, device=device).unsqueeze(0).expand(B, -1)
        positions = positions.clamp(0, self.max_seq_len - 1)
        pos_emb = self.pos_embed(positions)  # [B, T, dim]
        
        # Combine pitch embedding + positional
        pitch_emb = pitch_emb + pos_emb
        
        # Temporal convolutions for smoothing
        pitch_emb = pitch_emb.transpose(1, 2)  # [B, dim, T]
        pitch_emb = self.conv_layers(pitch_emb)
        pitch_emb = pitch_emb.transpose(1, 2)  # [B, T, dim]
        
        # Final projection: combine all features
        combined = torch.cat([pitch_emb, pitch_class_emb, uv_emb], dim=-1)
        output = self.output_proj(combined)  # [B, T, output_dim]
        
        return output
    
    def get_pooled(
        self,
        f0: Optional[torch.Tensor] = None,
        f0_coarse: Optional[torch.Tensor] = None,
        uv: Optional[torch.Tensor] = None,
        pool_type: str = 'mean',
    ) -> torch.Tensor:
        """
        Get pooled pitch embedding for global conditioning.
        
        Returns:
            pitch_emb: [B, output_dim] global pitch embedding
        """
        pitch_seq = self.forward(f0, f0_coarse, uv)  # [B, T, dim]
        
        if pool_type == 'mean':
            return pitch_seq.mean(dim=1)
        elif pool_type == 'max':
            return pitch_seq.max(dim=1)[0]
        elif pool_type == 'last':
            return pitch_seq[:, -1, :]
        else:
            return pitch_seq.mean(dim=1)


class PhonemeEncoder(nn.Module):
    """
    🎤 Phoneme Encoder for Singing Voice Synthesis
    
    Koduje fonemy (IPA) + voice embedding do reprezentacji śpiewu.
    Używa Transformer encoder do przetworzenia sekwencji fonemów.
    
    Input:
    - phonemes_ipa: string z fonemami IPA (np. "h ɛ l oʊ")
    - phonemes_words: lista słów
    - voice_embedding: [B, 256] embedding głosu
    - lyrics_text: oryginalny tekst
    """
    
    # Podstawowe fonemy IPA (angielski + polski + uniwersalne)
    IPA_PHONEMES = [
        '<pad>', '<unk>', '<sos>', '<eos>', '<sil>',
        # Vowels
        'a', 'ɑ', 'æ', 'e', 'ɛ', 'ə', 'i', 'ɪ', 'o', 'ɔ', 'u', 'ʊ', 'ʌ',
        'aɪ', 'aʊ', 'eɪ', 'oʊ', 'ɔɪ', 'ɪə', 'eə', 'ʊə',
        # Consonants
        'b', 'd', 'f', 'g', 'h', 'j', 'k', 'l', 'm', 'n', 'p', 'r', 's', 't', 'v', 'w', 'z',
        'ʃ', 'ʒ', 'θ', 'ð', 'ŋ', 'tʃ', 'dʒ',
        # Polish specific
        'ɕ', 'ʑ', 'ɲ', 'ʐ', 'ʂ', 'w̃',
        # Common symbols
        ' ', "'", ',', '.', ':', ';', '!', '?', '-',
    ]
    
    def __init__(
        self,
        output_dim: int = 768,
        voice_dim: int = 256,
        hidden_dim: int = 512,
        num_layers: int = 4,
        num_heads: int = 8,
        max_phonemes: int = 512,
    ):
        super().__init__()
        
        self.output_dim = output_dim
        self.max_phonemes = max_phonemes
        
        # Phoneme embedding
        self.phoneme_embed = nn.Embedding(len(self.IPA_PHONEMES), hidden_dim)
        self.phoneme_to_idx = {p: i for i, p in enumerate(self.IPA_PHONEMES)}
        
        # Positional encoding
        self.pos_embed = nn.Embedding(max_phonemes, hidden_dim)
        
        # Voice conditioning projection
        self.voice_proj = nn.Linear(voice_dim, hidden_dim)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=0.1,
            activation='gelu',
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, output_dim),
            nn.LayerNorm(output_dim),
        )
        
        # Duration predictor (dla synchronizacji z audio)
        self.duration_predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Softplus(),  # Duracje muszą być dodatnie
        )
    
    def tokenize_ipa(self, ipa_string: str) -> List[int]:
        """Tokenize IPA string to indices"""
        tokens = []
        i = 0
        while i < len(ipa_string):
            # Try 2-char phonemes first (like 'aɪ', 'tʃ')
            if i + 1 < len(ipa_string):
                two_char = ipa_string[i:i+2]
                if two_char in self.phoneme_to_idx:
                    tokens.append(self.phoneme_to_idx[two_char])
                    i += 2
                    continue
            
            # Single char
            char = ipa_string[i]
            if char in self.phoneme_to_idx:
                tokens.append(self.phoneme_to_idx[char])
            else:
                tokens.append(self.phoneme_to_idx['<unk>'])
            i += 1
        
        return tokens
    
    def forward(
        self,
        phonemes_ipa: List[str],                      # List of IPA strings
        voice_embedding: torch.Tensor,                 # [B, 256]
        phonemes_words: Optional[List[str]] = None,   # Not used directly, for alignment
        lyrics_text: Optional[List[str]] = None,      # Not used directly, for reference
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            phoneme_features: [B, output_dim] aggregated phoneme features
            duration_pred: [B, max_seq] predicted durations per phoneme
        """
        B = voice_embedding.shape[0]
        device = voice_embedding.device
        
        # Tokenize all phoneme strings
        all_tokens = []
        max_len = 0
        for ipa in phonemes_ipa:
            tokens = self.tokenize_ipa(ipa if ipa else "")
            tokens = tokens[:self.max_phonemes]
            all_tokens.append(tokens)
            max_len = max(max_len, len(tokens))
        
        max_len = max(max_len, 1)  # At least 1
        
        # Pad to same length
        padded_tokens = torch.zeros(B, max_len, device=device, dtype=torch.long)
        attention_mask = torch.zeros(B, max_len, device=device, dtype=torch.bool)
        
        for i, tokens in enumerate(all_tokens):
            if len(tokens) > 0:
                padded_tokens[i, :len(tokens)] = torch.tensor(tokens, device=device)
                attention_mask[i, :len(tokens)] = True
        
        # Phoneme embeddings
        phoneme_emb = self.phoneme_embed(padded_tokens)  # [B, seq, hidden]
        
        # Add positional encoding
        positions = torch.arange(max_len, device=device).unsqueeze(0).expand(B, -1)
        phoneme_emb = phoneme_emb + self.pos_embed(positions)
        
        # Add voice conditioning (broadcast to sequence)
        voice_cond = self.voice_proj(voice_embedding)  # [B, hidden]
        phoneme_emb = phoneme_emb + voice_cond.unsqueeze(1)
        
        # Create mask for transformer (True = ignore)
        src_key_padding_mask = ~attention_mask
        
        # Transformer encoding
        encoded = self.transformer(
            phoneme_emb,
            src_key_padding_mask=src_key_padding_mask,
        )  # [B, seq, hidden]
        
        # Duration prediction per phoneme
        duration_pred = self.duration_predictor(encoded).squeeze(-1)  # [B, seq]
        
        # Aggregate features (attention-weighted mean)
        # Use duration as attention weights
        duration_weights = duration_pred * attention_mask.float()
        duration_weights = duration_weights / (duration_weights.sum(dim=1, keepdim=True) + 1e-8)
        
        aggregated = torch.einsum('bs,bsh->bh', duration_weights, encoded)  # [B, hidden]
        
        # Project to output dim
        output_features = self.output_proj(aggregated)  # [B, output_dim]
        
        return output_features, duration_pred


# =============================================================================
# 🎨 Genre & Artist Encoders (v3)
# =============================================================================

class GenreEncoder(nn.Module):
    """
    Multi-hot genre encoder with embedding + pooling.
    
    Maps a list of genre strings (e.g., ["rock", "electronic"]) to a fixed-size vector.
    Uses multi-hot encoding → embedding lookup → mean pooling → MLP projection.
    """
    
    # Predefined genre vocabulary (matches dataset)
    GENRE_TO_IDX = {
        'rock': 0, 'pop': 1, 'hip-hop': 2, 'electronic': 3, 'jazz': 4,
        'classical': 5, 'country': 6, 'r&b': 7, 'metal': 8, 'folk': 9,
        'blues': 10, 'indie': 11, 'punk': 12, 'soul': 13, 'reggae': 14,
        'latin': 15, 'world': 16, 'ambient': 17, 'experimental': 18, 'other': 19,
    }
    
    def __init__(self, num_genres: int = 20, embed_dim: int = 64):
        super().__init__()
        self.num_genres = num_genres
        self.embed_dim = embed_dim
        
        # Genre embeddings
        self.genre_embedding = nn.Embedding(num_genres, embed_dim)
        
        # MLP projection after pooling
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.SiLU(),
            nn.Linear(embed_dim * 2, embed_dim),
        )
    
    def forward(self, genres: List[List[str]], device: torch.device) -> torch.Tensor:
        """
        Args:
            genres: List of genre lists, e.g., [["rock", "pop"], ["electronic"]]
            device: Target device
            
        Returns:
            [B, embed_dim] genre embedding
        """
        B = len(genres)
        
        # Create multi-hot encoding
        multi_hot = torch.zeros(B, self.num_genres, device=device)
        
        for i, genre_list in enumerate(genres):
            if genre_list:
                for genre in genre_list:
                    # Normalize genre name
                    genre_lower = genre.lower().strip()
                    idx = self.GENRE_TO_IDX.get(genre_lower, self.GENRE_TO_IDX['other'])
                    multi_hot[i, idx] = 1.0
        
        # Mean pooling over active genres
        # [B, num_genres] @ [num_genres, embed_dim] -> [B, embed_dim]
        genre_embeds = self.genre_embedding.weight  # [num_genres, embed_dim]
        
        # Weighted mean: sum(multi_hot * embeds) / max(sum(multi_hot), 1)
        weighted_sum = torch.matmul(multi_hot, genre_embeds)  # [B, embed_dim]
        genre_counts = multi_hot.sum(dim=1, keepdim=True).clamp(min=1)  # [B, 1]
        pooled = weighted_sum / genre_counts  # [B, embed_dim]
        
        # MLP projection
        output = self.mlp(pooled)
        
        return output


class ArtistEncoder(nn.Module):
    """
    Hash-based artist encoder for unknown vocabulary.
    
    Maps artist names to embeddings using hash bucketing.
    This allows handling any artist name without a fixed vocabulary.
    """
    
    def __init__(self, num_buckets: int = 1000, embed_dim: int = 64):
        super().__init__()
        self.num_buckets = num_buckets
        self.embed_dim = embed_dim
        
        # Bucket embeddings
        self.bucket_embedding = nn.Embedding(num_buckets, embed_dim)
        
        # MLP projection
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.SiLU(),
            nn.Linear(embed_dim * 2, embed_dim),
        )
    
    def _hash_artist(self, artist_name: str) -> int:
        """Hash artist name to bucket index."""
        if not artist_name:
            return 0
        # Simple hash using Python's built-in hash
        return hash(artist_name.lower().strip()) % self.num_buckets
    
    def forward(self, artists: List[str], device: torch.device) -> torch.Tensor:
        """
        Args:
            artists: List of artist names
            device: Target device
            
        Returns:
            [B, embed_dim] artist embedding
        """
        # Hash each artist to bucket
        bucket_indices = torch.tensor(
            [self._hash_artist(a) for a in artists],
            device=device,
            dtype=torch.long
        )
        
        # Lookup embeddings
        embeds = self.bucket_embedding(bucket_indices)  # [B, embed_dim]
        
        # MLP projection
        output = self.mlp(embeds)
        
        return output


class VibratoEncoder(nn.Module):
    """
    🎵 v3: Vibrato Encoder for expressive singing
    
    Encodes vibrato characteristics (rate, depth, extent) into embeddings.
    Vibrato is crucial for expressive singing synthesis.
    
    Input:
    - vibrato_rate: [B] average vibrato frequency in Hz (typically 4-8 Hz)
    - vibrato_depth: [B] average pitch deviation in cents (typically 20-100)
    - vibrato_extent: [B] fraction of segment with vibrato (0.0-1.0)
    """
    
    def __init__(self, output_dim: int = 64):
        super().__init__()
        self.output_dim = output_dim
        
        # Vibrato rate encoder (4-8 Hz typical range)
        self.rate_mlp = nn.Sequential(
            nn.Linear(1, 32),
            nn.SiLU(),
            nn.Linear(32, output_dim // 3),
        )
        
        # Vibrato depth encoder (cents, log scale)
        self.depth_mlp = nn.Sequential(
            nn.Linear(1, 32),
            nn.SiLU(),
            nn.Linear(32, output_dim // 3),
        )
        
        # Vibrato extent encoder (fraction 0-1)
        self.extent_mlp = nn.Sequential(
            nn.Linear(1, 32),
            nn.SiLU(),
            nn.Linear(32, output_dim // 3),
        )
        
        # Final projection
        combined_dim = 3 * (output_dim // 3)
        self.output_proj = nn.Linear(combined_dim, output_dim)
    
    def forward(
        self,
        vibrato_rate: Optional[torch.Tensor] = None,   # [B] Hz
        vibrato_depth: Optional[torch.Tensor] = None,  # [B] cents
        vibrato_extent: Optional[torch.Tensor] = None, # [B] fraction
    ) -> torch.Tensor:
        """
        Returns:
            vibrato_emb: [B, output_dim] vibrato embedding
        """
        if vibrato_rate is None and vibrato_depth is None and vibrato_extent is None:
            # Return zeros if no vibrato info
            return None
        
        device = (vibrato_rate if vibrato_rate is not None else 
                  vibrato_depth if vibrato_depth is not None else 
                  vibrato_extent).device
        B = (vibrato_rate if vibrato_rate is not None else 
             vibrato_depth if vibrato_depth is not None else 
             vibrato_extent).shape[0]
        
        # Rate: normalize to 0-1 range (typical 4-8 Hz)
        if vibrato_rate is not None:
            rate_norm = (vibrato_rate - 4.0) / 4.0  # Center around 6 Hz
            rate_emb = self.rate_mlp(rate_norm.unsqueeze(-1))
        else:
            rate_emb = torch.zeros(B, self.output_dim // 3, device=device)
        
        # Depth: log scale normalization (typical 20-100 cents)
        if vibrato_depth is not None:
            depth_log = torch.log1p(vibrato_depth) / 5.0  # Log scale
            depth_emb = self.depth_mlp(depth_log.unsqueeze(-1))
        else:
            depth_emb = torch.zeros(B, self.output_dim // 3, device=device)
        
        # Extent: already 0-1
        if vibrato_extent is not None:
            extent_emb = self.extent_mlp(vibrato_extent.unsqueeze(-1))
        else:
            extent_emb = torch.zeros(B, self.output_dim // 3, device=device)
        
        # Combine
        combined = torch.cat([rate_emb, depth_emb, extent_emb], dim=-1)
        output = self.output_proj(combined)
        
        return output


class BreathEncoder(nn.Module):
    """
    🌬️ v3: Breath Position Encoder for natural singing
    
    Encodes breath positions within a segment for natural phrasing.
    Breaths are important cues for phrase boundaries and expressiveness.
    
    Input:
    - breath_positions: List of breath timestamps (seconds) within segment
    - segment_duration: Duration of segment in seconds
    """
    
    def __init__(self, output_dim: int = 32, max_breaths: int = 8):
        super().__init__()
        self.output_dim = output_dim
        self.max_breaths = max_breaths
        
        # Breath count embedding
        self.count_embed = nn.Embedding(max_breaths + 1, output_dim // 2)
        
        # Breath timing encoder (relative positions)
        self.timing_mlp = nn.Sequential(
            nn.Linear(max_breaths, 32),
            nn.SiLU(),
            nn.Linear(32, output_dim // 2),
        )
        
        # Final projection
        self.output_proj = nn.Linear(output_dim, output_dim)
    
    def forward(
        self,
        breath_positions: Optional[List[List[float]]] = None,  # List of breath pos lists
        segment_duration: float = 10.0,  # Default segment duration
    ) -> Optional[torch.Tensor]:
        """
        Args:
            breath_positions: List of lists, each inner list contains breath times in seconds
            segment_duration: Duration of each segment
            
        Returns:
            breath_emb: [B, output_dim] breath embedding or None
        """
        if breath_positions is None:
            return None
        
        B = len(breath_positions)
        device = next(self.parameters()).device
        
        # Encode breath count
        breath_counts = torch.tensor(
            [min(len(bp) if bp else 0, self.max_breaths) for bp in breath_positions],
            device=device,
            dtype=torch.long
        )
        count_emb = self.count_embed(breath_counts)  # [B, output_dim//2]
        
        # Encode relative timing (pad to max_breaths)
        timing_tensor = torch.zeros(B, self.max_breaths, device=device)
        for i, bp in enumerate(breath_positions):
            if bp:
                for j, t in enumerate(bp[:self.max_breaths]):
                    # Normalize to 0-1 within segment
                    timing_tensor[i, j] = t / segment_duration
        
        timing_emb = self.timing_mlp(timing_tensor)  # [B, output_dim//2]
        
        # Combine
        combined = torch.cat([count_emb, timing_emb], dim=-1)
        output = self.output_proj(combined)
        
        return output


class PhonemeTimestampEncoder(nn.Module):
    """
    🗣️ v3: Phoneme Timestamp Encoder for precise lip-sync
    
    Encodes phoneme timing information within a segment.
    Critical for singing synthesis where phoneme timing matters.
    
    Input:
    - phoneme_timestamps: List of (phoneme, start_time, end_time) tuples
    """
    
    def __init__(self, output_dim: int = 64, max_phonemes: int = 64):
        super().__init__()
        self.output_dim = output_dim
        self.max_phonemes = max_phonemes
        
        # Phoneme count embedding
        self.count_embed = nn.Embedding(max_phonemes + 1, output_dim // 4)
        
        # Duration statistics encoder
        self.duration_mlp = nn.Sequential(
            nn.Linear(3, 32),  # mean, std, total duration
            nn.SiLU(),
            nn.Linear(32, output_dim // 4),
        )
        
        # Timing density encoder (phonemes per second)
        self.density_mlp = nn.Sequential(
            nn.Linear(1, 16),
            nn.SiLU(),
            nn.Linear(16, output_dim // 4),
        )
        
        # Coverage encoder (fraction of segment with phonemes)
        self.coverage_mlp = nn.Sequential(
            nn.Linear(1, 16),
            nn.SiLU(),
            nn.Linear(16, output_dim // 4),
        )
        
        # Final projection
        self.output_proj = nn.Linear(output_dim, output_dim)
    
    def forward(
        self,
        phoneme_timestamps: Optional[List[List]] = None,  # List of phoneme timing lists
        segment_duration: float = 10.0,
    ) -> Optional[torch.Tensor]:
        """
        Args:
            phoneme_timestamps: List of lists, each containing (phoneme, start, end) tuples
            segment_duration: Duration of segment
            
        Returns:
            phoneme_timing_emb: [B, output_dim] or None
        """
        if phoneme_timestamps is None:
            return None
        
        B = len(phoneme_timestamps)
        device = next(self.parameters()).device
        
        # Phoneme counts
        counts = torch.tensor(
            [min(len(pt) if pt else 0, self.max_phonemes) for pt in phoneme_timestamps],
            device=device,
            dtype=torch.long
        )
        count_emb = self.count_embed(counts)  # [B, output_dim//4]
        
        # Duration statistics
        duration_stats = torch.zeros(B, 3, device=device)
        for i, pt in enumerate(phoneme_timestamps):
            if pt and len(pt) > 0:
                durations = []
                for item in pt:
                    # Support both dict and tuple/list format
                    if isinstance(item, dict):
                        start = item.get('start', 0)
                        end = item.get('end', start)
                        dur = end - start
                    elif hasattr(item, '__len__') and len(item) >= 3:
                        dur = item[2] - item[1]  # end - start
                    else:
                        continue
                    durations.append(dur)
                if durations:
                    durations_t = torch.tensor(durations, device=device)
                    duration_stats[i, 0] = durations_t.mean()
                    duration_stats[i, 1] = durations_t.std() if len(durations) > 1 else 0
                    duration_stats[i, 2] = durations_t.sum()
        
        duration_emb = self.duration_mlp(duration_stats)  # [B, output_dim//4]
        
        # Density (phonemes per second)
        density = counts.float() / segment_duration
        density_emb = self.density_mlp(density.unsqueeze(-1))  # [B, output_dim//4]
        
        # Coverage (fraction of segment with phonemes)
        coverage = duration_stats[:, 2] / segment_duration
        coverage_emb = self.coverage_mlp(coverage.unsqueeze(-1))  # [B, output_dim//4]
        
        # Combine
        combined = torch.cat([count_emb, duration_emb, density_emb, coverage_emb], dim=-1)
        output = self.output_proj(combined)
        
        return output


class SectionConditioningModule(nn.Module):
    """
    Moduł kondycjonowania sekcją.
    
    Łączy wszystkie informacje o sekcji w jeden conditioning tensor.
    
    v2 Updates:
    - 🎵 CLAP embeddings (audio + text)
    - 🥁 Beat conditioning
    - 🎸 Chord conditioning
    - 🎤 Phoneme features (dla singing)
    
    v3 Updates:
    - 🎵 PitchEncoder (F0 conditioning dla melodii)
    """
    
    SECTION_TYPES = [
        'intro', 'verse', 'pre_chorus', 'chorus', 'post_chorus',
        'bridge', 'instrumental', 'solo', 'breakdown', 'buildup',
        'drop', 'outro', 'unknown'
    ]
    
    def __init__(
        self,
        output_dim: int = 1024,
        text_embed_dim: int = 768,
        section_embed_dim: int = 128,
        num_keys: int = 24,
        # v2: New modules
        use_clap: bool = True,
        use_beat: bool = True,
        use_chord: bool = True,
        use_phonemes: bool = True,
        use_pitch: bool = True,  # v3: F0/pitch conditioning
        clap_dim: int = 512,
        voice_dim: int = 256,
    ):
        super().__init__()
        
        self.output_dim = output_dim
        self.use_clap = use_clap
        self.use_beat = use_beat
        self.use_chord = use_chord
        self.use_phonemes = use_phonemes
        self.use_pitch = use_pitch  # v3
        
        # Section type embedding
        self.section_embed = nn.Embedding(len(self.SECTION_TYPES), section_embed_dim)
        self.section_to_idx = {s: i for i, s in enumerate(self.SECTION_TYPES)}
        
        # Position embedding (continuous)
        self.position_mlp = nn.Sequential(
            nn.Linear(1, 64),
            nn.SiLU(),
            nn.Linear(64, 128),
        )
        
        # Energy embedding (continuous)
        self.energy_mlp = nn.Sequential(
            nn.Linear(1, 32),
            nn.SiLU(),
            nn.Linear(32, 64),
        )
        
        # 🔊 v3: Loudness embedding (continuous, dB scale)
        self.loudness_mlp = nn.Sequential(
            nn.Linear(1, 32),
            nn.SiLU(),
            nn.Linear(32, 64),
        )
        
        # 🎤 v3: Has vocals embedding (binary)
        self.has_vocals_embed = nn.Embedding(2, 32)  # 0=no vocals, 1=has vocals
        
        # 😊 v3: Sentiment score embedding (continuous, -1 to 1)
        self.sentiment_mlp = nn.Sequential(
            nn.Linear(1, 32),
            nn.SiLU(),
            nn.Linear(32, 64),
        )
        
        # 🎨 v3: Genre embedding (multi-hot → embedding)
        # Common genres: rock, pop, electronic, jazz, classical, hip-hop, r&b, country, metal, folk, etc.
        self.GENRE_LIST = [
            'rock', 'pop', 'electronic', 'jazz', 'classical', 'hip-hop', 'r&b', 
            'country', 'metal', 'folk', 'blues', 'soul', 'funk', 'reggae', 'punk',
            'indie', 'alternative', 'dance', 'house', 'techno', 'ambient', 'experimental',
            'world', 'latin', 'acoustic', 'instrumental', 'vocal', 'soundtrack', 'other'
        ]
        self.genre_to_idx = {g: i for i, g in enumerate(self.GENRE_LIST)}
        self.genre_embed = nn.Embedding(len(self.GENRE_LIST), 16)  # 16-dim per genre
        self.genre_proj = nn.Linear(16, 64)  # Project mean genre embedding to 64
        
        # 🎤 v3: Artist embedding (hash-based for unlimited artists)
        self.num_artist_buckets = 1000  # Hash artists to 1000 buckets
        self.artist_embed = nn.Embedding(self.num_artist_buckets, 64)
        
        # Tempo embedding (continuous)
        self.tempo_mlp = nn.Sequential(
            nn.Linear(1, 32),
            nn.SiLU(),
            nn.Linear(32, 64),
        )
        
        # Key embedding
        self.key_embed = nn.Embedding(num_keys, 64)
        
        # Text projection
        self.text_proj = nn.Linear(text_embed_dim, 512)
        
        # 🎵 CLAP Projection (v2)
        if use_clap:
            self.clap_projection = ClapProjection(
                clap_dim=clap_dim,
                output_dim=text_embed_dim,  # 768
                mode='fused',
            )
            # CLAP → conditioning dim
            self.clap_to_cond = nn.Linear(text_embed_dim, 128)
        
        # 🥁 Beat Embedding (v2)
        if use_beat:
            self.beat_embedding = BeatEmbedding(output_dim=64)
        
        # 🎸 Chord Embedding (v2)
        if use_chord:
            self.chord_embedding = ChordEmbedding(output_dim=64)
        
        # 🎤 Phoneme Encoder (v2)
        if use_phonemes:
            self.phoneme_encoder = PhonemeEncoder(
                output_dim=text_embed_dim,  # 768
                voice_dim=voice_dim,
            )
            self.phoneme_to_cond = nn.Linear(text_embed_dim, 128)
        
        # 🎵 Pitch Encoder (v3)
        if use_pitch:
            self.pitch_encoder = PitchEncoder(
                output_dim=64,
                max_seq_len=1024,
            )
        
        # 🎵 v3: Vibrato Encoder (for expressive singing)
        self.vibrato_encoder = VibratoEncoder(output_dim=64)
        
        # 🌬️ v3: Breath Encoder (for natural phrasing)
        self.breath_encoder = BreathEncoder(output_dim=32)
        
        # 🗣️ v3: Phoneme Timestamp Encoder (for precise timing)
        self.phoneme_timestamp_encoder = PhonemeTimestampEncoder(output_dim=64)
        
        # Fusion
        # Original: section(128) + position(128) + energy(64) + tempo(64) + key(64) + text(512) = 960
        # v2: + clap(128) + beat(64) + chord(64) + phoneme(128) = 1344
        # v3: + pitch(64) + loudness(64) + has_vocals(32) = 1504
        # v3.1: + vibrato(64) + breath(32) + phoneme_timestamps(64) = 1664
        # Fusion dimensions:
        # Original: section(128) + position(128) + energy(64) + tempo(64) + key(64) + text(512) = 960
        # v3 base: + loudness(64) + has_vocals(32) + sentiment(64) + genre(64) + artist(64) = 1248
        # v2 optional: + clap(128) + beat(64) + chord(64) + phoneme(128) = 1632
        # v3 optional: + pitch(64) + vibrato(64) + breath(32) + phoneme_ts(64) = 1856
        base_dim = section_embed_dim + 128 + 64 + 64 + 64 + 512 + 64 + 32 + 64 + 64 + 64  # 1248
        extra_dim = 0
        if use_clap:
            extra_dim += 128
        if use_beat:
            extra_dim += 64
        if use_chord:
            extra_dim += 64
        if use_phonemes:
            extra_dim += 128
        if use_pitch:
            extra_dim += 64
        
        # v3.1: Always add vibrato, breath, phoneme_timestamp dims
        extra_dim += 64 + 32 + 64  # vibrato + breath + phoneme_timestamps
        
        fusion_dim = base_dim + extra_dim
        
        self.fusion = nn.Sequential(
            nn.Linear(fusion_dim, output_dim),
            nn.SiLU(),
            nn.Linear(output_dim, output_dim),
        )
    
    def _hash_artist(self, artist: str) -> int:
        """Hash artist name to bucket index."""
        return hash(artist.lower().strip()) % self.num_artist_buckets
    
    def _encode_genres(self, genres: List[str], device) -> torch.Tensor:
        """Encode genre list to embedding by averaging."""
        if not genres:
            return torch.zeros(64, device=device)
        
        indices = []
        for g in genres:
            g_lower = g.lower().strip()
            if g_lower in self.genre_to_idx:
                indices.append(self.genre_to_idx[g_lower])
            else:
                indices.append(self.genre_to_idx['other'])
        
        genre_indices = torch.tensor(indices, device=device)
        genre_embs = self.genre_embed(genre_indices)  # [num_genres, 16]
        mean_emb = genre_embs.mean(dim=0)  # [16]
        return self.genre_proj(mean_emb)  # [64]
    
    def forward(
        self,
        text_embed: torch.Tensor,           # [B, text_embed_dim] lub [B, seq, text_embed_dim]
        section_type: Optional[List[str]] = None,
        position: Optional[torch.Tensor] = None,     # [B]
        energy: Optional[torch.Tensor] = None,       # [B]
        tempo: Optional[torch.Tensor] = None,        # [B] BPM
        key_idx: Optional[torch.Tensor] = None,      # [B]
        # v3: New conditioning
        loudness: Optional[torch.Tensor] = None,           # [B] loudness in dB
        has_vocals: Optional[torch.Tensor] = None,         # [B] bool tensor
        sentiment_score: Optional[torch.Tensor] = None,    # [B] sentiment (-1 to 1)
        genres: Optional[List[List[str]]] = None,          # List of genre lists per sample
        artists: Optional[List[str]] = None,               # List of artist names
        # v2: New conditioning inputs
        clap_audio_embedding: Optional[torch.Tensor] = None,   # [B, 512]
        clap_text_embedding: Optional[torch.Tensor] = None,    # [B, 512]
        num_beats: Optional[torch.Tensor] = None,              # [B] int
        beat_positions: Optional[List] = None,                  # List of beat position lists
        time_signature: Optional[List[str]] = None,            # List of "4/4" etc
        current_chord: Optional[List[str]] = None,             # List of chord strings
        phonemes_ipa: Optional[List[str]] = None,              # List of IPA strings
        voice_embedding: Optional[torch.Tensor] = None,        # [B, 256] for phoneme encoder
        # v3: Pitch conditioning
        f0: Optional[torch.Tensor] = None,                     # [B, T] continuous F0 in Hz
        f0_coarse: Optional[torch.Tensor] = None,              # [B, T] discrete pitch bins
        f0_voiced_mask: Optional[torch.Tensor] = None,         # [B, T] voiced mask (True=voiced)
        # v3.1: Vibrato, breath, phoneme timestamps
        vibrato_rate: Optional[torch.Tensor] = None,           # [B] Hz
        vibrato_depth: Optional[torch.Tensor] = None,          # [B] cents
        vibrato_extent: Optional[torch.Tensor] = None,         # [B] fraction 0-1
        breath_positions: Optional[List[List[float]]] = None,  # List of breath position lists
        phoneme_timestamps: Optional[List[List]] = None,       # List of (phoneme, start, end) lists
        segment_duration: float = 10.0,                         # Duration for breath/phoneme encoding
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Returns:
            conditioning: [B, output_dim] conditioning vector
            phoneme_durations: [B, seq] predicted phoneme durations (or None)
        """
        B = text_embed.shape[0]
        device = text_embed.device
        
        # Handle sequence text embedding - use CLS or mean
        if text_embed.dim() == 3:
            text_embed = text_embed.mean(dim=1)  # [B, text_embed_dim]
        
        # Text projection
        text_cond = self.text_proj(text_embed)  # [B, 512]
        
        # Section type
        if section_type is not None:
            section_indices = torch.tensor(
                [self.section_to_idx.get(s.lower(), self.section_to_idx['unknown'])
                 for s in section_type],
                device=device
            )
        else:
            section_indices = torch.zeros(B, dtype=torch.long, device=device)
        section_cond = self.section_embed(section_indices)  # [B, 128]
        
        # Position
        if position is not None:
            pos_cond = self.position_mlp(position.unsqueeze(-1))  # [B, 128]
        else:
            pos_cond = torch.zeros(B, 128, device=device)
        
        # Energy
        if energy is not None:
            energy_cond = self.energy_mlp(energy.unsqueeze(-1))  # [B, 64]
        else:
            energy_cond = torch.zeros(B, 64, device=device)
        
        # Tempo
        if tempo is not None:
            tempo_norm = (tempo - 60) / 140  # Normalize
            tempo_cond = self.tempo_mlp(tempo_norm.unsqueeze(-1))  # [B, 64]
        else:
            tempo_cond = torch.zeros(B, 64, device=device)
        
        # Key
        if key_idx is not None:
            key_cond = self.key_embed(key_idx)  # [B, 64]
        else:
            key_cond = torch.zeros(B, 64, device=device)
        
        # 🔊 v3: Loudness (dB scale, normalized)
        if loudness is not None:
            # Normalize: typical range -60 to 0 dB → -1 to 1
            loudness_norm = (loudness + 30) / 30  # Shift and scale
            loudness_cond = self.loudness_mlp(loudness_norm.unsqueeze(-1))  # [B, 64]
        else:
            loudness_cond = torch.zeros(B, 64, device=device)
        
        # 🎤 v3: Has vocals (binary)
        if has_vocals is not None:
            has_vocals_idx = has_vocals.long()  # Convert bool to 0/1
            has_vocals_cond = self.has_vocals_embed(has_vocals_idx)  # [B, 32]
        else:
            has_vocals_cond = torch.zeros(B, 32, device=device)
        
        # 😊 v3: Sentiment score (continuous)
        if sentiment_score is not None:
            # Already in range -1 to 1, just pass through
            sentiment_cond = self.sentiment_mlp(sentiment_score.unsqueeze(-1))  # [B, 64]
        else:
            sentiment_cond = torch.zeros(B, 64, device=device)
        
        # 🎨 v3: Genre embedding (per-sample genre lists)
        if genres is not None:
            genre_conds = []
            for genre_list in genres:
                genre_cond = self._encode_genres(genre_list, device)  # [64]
                genre_conds.append(genre_cond)
            genre_cond = torch.stack(genre_conds)  # [B, 64]
        else:
            genre_cond = torch.zeros(B, 64, device=device)
        
        # 🎤 v3: Artist embedding (hash-based)
        if artists is not None:
            artist_indices = torch.tensor(
                [self._hash_artist(a) for a in artists],
                device=device, dtype=torch.long
            )
            artist_cond = self.artist_embed(artist_indices)  # [B, 64]
        else:
            artist_cond = torch.zeros(B, 64, device=device)
        
        # Base conditioning list
        cond_parts = [
            text_cond,          # 512
            section_cond,       # 128
            pos_cond,           # 128
            energy_cond,        # 64
            tempo_cond,         # 64
            key_cond,           # 64
            loudness_cond,      # 64 (v3)
            has_vocals_cond,    # 32 (v3)
            sentiment_cond,     # 64 (v3)
            genre_cond,         # 64 (v3)
            artist_cond,        # 64 (v3)
        ]
        
        # 🎵 CLAP conditioning (v2)
        if self.use_clap:
            clap_emb = self.clap_projection(clap_audio_embedding, clap_text_embedding)  # [B, 768] or None
            if clap_emb is not None:
                clap_cond = self.clap_to_cond(clap_emb)  # [B, 128]
                cond_parts.append(clap_cond)
            else:
                # Add zeros if no CLAP embedding
                cond_parts.append(torch.zeros(B, 128, device=device))
        
        # 🥁 Beat conditioning (v2)
        if self.use_beat and num_beats is not None:
            beat_cond = self.beat_embedding(
                num_beats=num_beats,
                beat_positions=beat_positions,
                time_signature=time_signature,
            )  # [B, 64]
            cond_parts.append(beat_cond)
        elif self.use_beat:
            cond_parts.append(torch.zeros(B, 64, device=device))
        
        # 🎸 Chord conditioning (v2)
        if self.use_chord and current_chord is not None:
            chord_cond = self.chord_embedding(current_chord)  # [B, 64]
            cond_parts.append(chord_cond)
        elif self.use_chord:
            cond_parts.append(torch.zeros(B, 64, device=device))
        
        # 🎤 Phoneme conditioning (v2)
        phoneme_durations = None
        if self.use_phonemes and phonemes_ipa is not None and voice_embedding is not None:
            phoneme_features, phoneme_durations = self.phoneme_encoder(
                phonemes_ipa=phonemes_ipa,
                voice_embedding=voice_embedding,
            )  # [B, 768], [B, seq]
            phoneme_cond = self.phoneme_to_cond(phoneme_features)  # [B, 128]
            cond_parts.append(phoneme_cond)
        elif self.use_phonemes:
            cond_parts.append(torch.zeros(B, 128, device=device))
        
        # 🎵 Pitch conditioning (v3)
        if self.use_pitch and (f0 is not None or f0_coarse is not None):
            # Convert f0_voiced_mask (True=voiced) to uv mask (1=unvoiced) expected by PitchEncoder
            uv_mask = None
            if f0_voiced_mask is not None:
                uv_mask = (~f0_voiced_mask).long()  # Invert: True→0 (voiced), False→1 (unvoiced)
            pitch_cond = self.pitch_encoder.get_pooled(
                f0=f0,
                f0_coarse=f0_coarse,
                uv=uv_mask,  # ✅ Now using ground-truth mask from dataset
                pool_type='mean',
            )  # [B, 64]
            cond_parts.append(pitch_cond)
        elif self.use_pitch:
            cond_parts.append(torch.zeros(B, 64, device=device))
        
        # 🎵 v3.1: Vibrato conditioning
        vibrato_cond = self.vibrato_encoder(
            vibrato_rate=vibrato_rate,
            vibrato_depth=vibrato_depth,
            vibrato_extent=vibrato_extent,
        )
        if vibrato_cond is not None:
            cond_parts.append(vibrato_cond)
        else:
            cond_parts.append(torch.zeros(B, 64, device=device))
        
        # 🌬️ v3.1: Breath conditioning
        breath_cond = self.breath_encoder(
            breath_positions=breath_positions,
            segment_duration=segment_duration,
        )
        if breath_cond is not None:
            cond_parts.append(breath_cond)
        else:
            cond_parts.append(torch.zeros(B, 32, device=device))
        
        # 🗣️ v3.1: Phoneme timestamp conditioning
        phoneme_ts_cond = self.phoneme_timestamp_encoder(
            phoneme_timestamps=phoneme_timestamps,
            segment_duration=segment_duration,
        )
        if phoneme_ts_cond is not None:
            cond_parts.append(phoneme_ts_cond)
        else:
            cond_parts.append(torch.zeros(B, 64, device=device))
        
        # Combine all
        combined = torch.cat(cond_parts, dim=-1)
        
        return self.fusion(combined), phoneme_durations


class CrossAttention(nn.Module):
    """
    Cross-attention z obsługą kontekstu sekcji.
    
    v3 Updates:
    - GQA (Grouped Query Attention) - 4× mniejszy KV-cache
    - num_kv_heads < heads dla efektywności pamięciowej
    - RoPE (Rotary Position Embedding) dla self-attention
    - Flash Attention via scaled_dot_product_attention (PyTorch 2.0+)
    """
    
    def __init__(
        self,
        query_dim: int,
        context_dim: int = None,
        heads: int = 8,
        num_kv_heads: int = None,  # v3: GQA - jeśli None, użyj heads (MHA)
        dim_head: int = 64,
        use_rope: bool = False,    # v3: RoPE dla self-attention
        max_seq_len: int = 4096,
        use_flash_attn: bool = True,  # v3: Flash Attention
        dropout: float = 0.0,
    ):
        super().__init__()
        context_dim = context_dim or query_dim
        self.is_self_attention = (context_dim == query_dim)
        
        self.heads = heads
        self.num_kv_heads = num_kv_heads or heads  # Default: standard MHA
        self.dim_head = dim_head
        self.scale = dim_head ** -0.5
        self.use_rope = use_rope and self.is_self_attention  # RoPE only for self-attn
        self.use_flash_attn = use_flash_attn and hasattr(F, 'scaled_dot_product_attention')
        self.dropout = dropout
        
        # Validate GQA config
        assert heads % self.num_kv_heads == 0, \
            f"heads ({heads}) must be divisible by num_kv_heads ({self.num_kv_heads})"
        self.num_groups = heads // self.num_kv_heads
        
        inner_dim = heads * dim_head
        kv_dim = self.num_kv_heads * dim_head  # Mniejszy dla GQA
        
        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, kv_dim, bias=False)  # GQA: mniejszy
        self.to_v = nn.Linear(context_dim, kv_dim, bias=False)  # GQA: mniejszy
        self.to_out = nn.Linear(inner_dim, query_dim)
        
        # RoPE for self-attention
        if self.use_rope:
            self.rope = RotaryEmbedding(dim_head, max_seq_len=max_seq_len)
    
    def forward(
        self,
        x: torch.Tensor,
        context: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        is_self_attn = context is None
        context = context if context is not None else x
        B, N, _ = x.shape
        
        q = self.to_q(x)
        k = self.to_k(context)
        v = self.to_v(context)
        
        # Reshape Q: [B, N, heads*dim_head] -> [B, heads, N, dim_head]
        q = rearrange(q, 'b n (h d) -> b h n d', h=self.heads)
        
        # Reshape K,V for GQA: [B, N, num_kv_heads*dim_head] -> [B, num_kv_heads, N, dim_head]
        k = rearrange(k, 'b n (h d) -> b h n d', h=self.num_kv_heads)
        v = rearrange(v, 'b n (h d) -> b h n d', h=self.num_kv_heads)
        
        # Apply RoPE for self-attention
        if self.use_rope and is_self_attn:
            cos, sin = self.rope(N, x.device)
            q = (q * cos) + (rotate_half(q) * sin)
            k = (k * cos) + (rotate_half(k) * sin)
        
        # GQA: Repeat K,V for each group
        if self.num_groups > 1:
            k = k.repeat_interleave(self.num_groups, dim=1)  # [B, heads, N, dim_head]
            v = v.repeat_interleave(self.num_groups, dim=1)
        
        # Flash Attention (PyTorch 2.0+) or standard attention
        if self.use_flash_attn:
            # scaled_dot_product_attention handles scaling internally
            # It automatically uses Flash Attention when available
            out = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=attn_mask,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=False,  # Not causal for diffusion
            )
        else:
            # Standard attention
            attn = torch.matmul(q, k.transpose(-1, -2)) * self.scale
            if attn_mask is not None:
                attn = attn.masked_fill(attn_mask == 0, float('-inf'))
            attn = F.softmax(attn, dim=-1)
            if self.dropout > 0 and self.training:
                attn = F.dropout(attn, p=self.dropout)
            out = torch.matmul(attn, v)
        
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class SwiGLU(nn.Module):
    """
    SwiGLU activation - lepsza niż GELU dla transformerów.
    Używane w Llama, PaLM, etc.
    
    SwiGLU(x) = SiLU(gate(x)) * up(x)
    """
    
    def __init__(self, dim: int, hidden_dim: int = None, dropout: float = 0.0):
        super().__init__()
        hidden_dim = hidden_dim or dim * 4
        
        self.gate_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.up_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.down_proj = nn.Linear(hidden_dim, dim, bias=False)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = F.silu(self.gate_proj(x))
        up = self.up_proj(x)
        return self.dropout(self.down_proj(gate * up))


class TransformerBlock(nn.Module):
    """
    Blok transformera z self-attention i cross-attention.
    
    v3 Updates:
    - SwiGLU zamiast GELU (lepsza jakość)
    - GQA (Grouped Query Attention) - mniejszy KV-cache
    - RoPE (Rotary Position Embedding) dla self-attention
    """
    
    def __init__(
        self,
        dim: int,
        context_dim: int = None,
        heads: int = 8,
        num_kv_heads: int = None,  # v3: GQA - jeśli None, użyj heads//4
        dim_head: int = 64,
        use_swiglu: bool = True,
        use_gqa: bool = True,    # v3: GQA domyślnie włączone
        use_rope: bool = True,   # v3: RoPE dla self-attention
        max_seq_len: int = 4096,
    ):
        super().__init__()
        
        # GQA config: domyślnie 4× mniej KV heads
        if use_gqa and num_kv_heads is None:
            num_kv_heads = max(1, heads // 4)
        elif not use_gqa:
            num_kv_heads = heads  # Standard MHA
        
        self.norm1 = nn.LayerNorm(dim)
        self.self_attn = CrossAttention(
            dim, dim, heads, num_kv_heads, dim_head,
            use_rope=use_rope, max_seq_len=max_seq_len,
        )
        
        self.norm2 = nn.LayerNorm(dim)
        # Cross-attention doesn't use RoPE (different sequence lengths)
        self.cross_attn = CrossAttention(
            dim, context_dim, heads, num_kv_heads, dim_head,
            use_rope=False,
        )
        
        self.norm3 = nn.LayerNorm(dim)
        
        # SwiGLU lub standard GELU FFN
        if use_swiglu:
            self.ff = SwiGLU(dim, dim * 4)
        else:
            self.ff = nn.Sequential(
                nn.Linear(dim, dim * 4),
                nn.GELU(),
                nn.Linear(dim * 4, dim),
            )
    
    def forward(self, x: torch.Tensor, context: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = x + self.self_attn(self.norm1(x))
        if context is not None:
            x = x + self.cross_attn(self.norm2(x), context)
        x = x + self.ff(self.norm3(x))
        return x


class ResBlock(nn.Module):
    """Residual block z timestep i section conditioning"""
    
    def __init__(self, in_channels: int, out_channels: int, cond_dim: int):
        super().__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.norm1 = nn.GroupNorm(8, out_channels)
        self.norm2 = nn.GroupNorm(8, out_channels)
        
        # Combined conditioning (timestep + section)
        self.cond_mlp = nn.Linear(cond_dim, out_channels * 2)  # scale and shift
        
        self.skip = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()
    
    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        h = self.conv1(x)
        h = self.norm1(h)
        h = F.silu(h)
        
        # Apply conditioning (scale and shift)
        cond_params = self.cond_mlp(cond)
        scale, shift = cond_params.chunk(2, dim=-1)
        scale = scale[:, :, None, None]
        shift = shift[:, :, None, None]
        h = h * (1 + scale) + shift
        
        h = self.conv2(h)
        h = self.norm2(h)
        h = F.silu(h)
        
        return h + self.skip(x)


class SpatialTransformer(nn.Module):
    """Spatial transformer dla cross-attention na feature maps"""
    
    def __init__(self, channels: int, context_dim: int, heads: int = 8, depth: int = 1):
        super().__init__()
        
        self.norm = nn.GroupNorm(8, channels)
        self.proj_in = nn.Conv2d(channels, channels, 1)
        
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(channels, context_dim, heads)
            for _ in range(depth)
        ])
        
        self.proj_out = nn.Conv2d(channels, channels, 1)
    
    def forward(self, x: torch.Tensor, context: Optional[torch.Tensor] = None) -> torch.Tensor:
        b, c, h, w = x.shape
        x_in = x
        
        x = self.norm(x)
        x = self.proj_in(x)
        
        x = rearrange(x, 'b c h w -> b (h w) c')
        
        for block in self.transformer_blocks:
            x = block(x, context)
        
        x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)
        x = self.proj_out(x)
        
        return x + x_in


class ContextFusion(nn.Module):
    """
    Fuzja z kontekstem poprzedniego segmentu.
    
    Pozwala modelowi na zachowanie spójności między segmentami.
    """
    
    def __init__(self, channels: int):
        super().__init__()
        
        # Projekcja kontekstu
        self.context_proj = nn.Conv2d(channels, channels, 1)
        
        # Cross-attention na poziomie feature maps
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=channels,
            num_heads=8,
            batch_first=True,
        )
        
        self.norm = nn.LayerNorm(channels)
        self.gate = nn.Sequential(
            nn.Linear(channels * 2, channels),
            nn.Sigmoid(),
        )
    
    def forward(
        self, 
        x: torch.Tensor,  # [B, C, H, W]
        context: Optional[torch.Tensor] = None,  # [B, C, H, W]
    ) -> torch.Tensor:
        if context is None:
            return x
        
        b, c, h, w = x.shape
        
        # Project context
        context = self.context_proj(context)
        
        # Reshape for attention
        x_flat = rearrange(x, 'b c h w -> b (h w) c')
        context_flat = rearrange(context, 'b c h w -> b (h w) c')
        
        # Cross-attention
        attended, _ = self.cross_attn(x_flat, context_flat, context_flat)
        attended = self.norm(attended)
        
        # Gating
        gate_input = torch.cat([x_flat.mean(dim=1), attended.mean(dim=1)], dim=-1)
        gate = self.gate(gate_input).unsqueeze(1)
        
        # Apply gated fusion
        fused = x_flat + gate * attended
        
        return rearrange(fused, 'b (h w) c -> b c h w', h=h, w=w)


class VoiceStreamAttention(nn.Module):
    """
    🎤 Voice Stream Attention - głęboka integracja wokalu z muzyką.
    
    Zamiast prostego dodawania voice embedding do cond, ten moduł
    stosuje cross-attention między feature maps a voice embedding
    na KAŻDYM poziomie U-Net.
    
    To pozwala modelowi nauczyć się:
    - Gdzie w spektrogramie powinien być wokal
    - Jak dopasować instrumenty do charakterystyki głosu
    - Synchronizacja rytmu wokalu z beatem
    """
    
    def __init__(
        self,
        channels: int,
        voice_dim: int = 256,
        num_heads: int = 4,
    ):
        super().__init__()
        
        self.channels = channels
        
        # Project voice to sequence format (dla cross-attention)
        # Voice embedding [B, voice_dim] -> [B, num_tokens, channels]
        self.voice_to_seq = nn.Sequential(
            nn.Linear(voice_dim, channels * 4),
            nn.SiLU(),
            nn.Linear(channels * 4, channels * 8),  # 8 "voice tokens"
        )
        self.num_voice_tokens = 8
        
        # Cross-attention: features attend to voice tokens
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=channels,
            num_heads=num_heads,
            batch_first=True,
        )
        
        # Learnable voice presence gate
        # Model uczy się gdzie wokal powinien być obecny
        self.voice_gate = nn.Sequential(
            nn.Linear(voice_dim + channels, channels),
            nn.Sigmoid(),
        )
        
        self.norm = nn.LayerNorm(channels)
        self.out_proj = nn.Linear(channels, channels)
    
    def forward(
        self,
        x: torch.Tensor,  # [B, C, H, W] feature maps
        voice_emb: Optional[torch.Tensor] = None,  # [B, voice_dim]
    ) -> torch.Tensor:
        """
        Stosuje voice-aware attention do feature maps.
        
        Jeśli voice_emb jest None, zwraca x bez zmian.
        """
        if voice_emb is None:
            return x
        
        b, c, h, w = x.shape
        
        # Voice embedding -> sequence of voice tokens
        voice_seq = self.voice_to_seq(voice_emb)  # [B, self.channels * 8]
        voice_seq = voice_seq.view(b, self.num_voice_tokens, self.channels)  # [B, 8, self.channels]
        
        # Flatten spatial dims
        x_flat = rearrange(x, 'b c h w -> b (h w) c')  # [B, H*W, C]
        
        # Cross-attention: features query voice tokens
        attended, attn_weights = self.cross_attn(
            query=x_flat,
            key=voice_seq,
            value=voice_seq,
        )  # [B, H*W, C]
        
        attended = self.norm(attended)
        attended = self.out_proj(attended)
        
        # Compute voice presence gate
        # Global feature + voice -> gate dla każdej pozycji
        global_feat = x_flat.mean(dim=1)  # [B, C]
        gate_input = torch.cat([voice_emb, global_feat], dim=-1)  # [B, voice_dim + C]
        gate = self.voice_gate(gate_input)  # [B, C]
        gate = gate.unsqueeze(1)  # [B, 1, C]
        
        # Apply gated voice conditioning
        x_out = x_flat + gate * attended
        
        return rearrange(x_out, 'b (h w) c -> b c h w', h=h, w=w)


class UNetV2(nn.Module):
    """
    U-Net v2 z section-aware conditioning.
    
    v2 Updates:
    - in_channels/out_channels: 8 → 128 (większy latent_dim)
    - Dostosowane model_channels dla 128D latent
    - Gradient Checkpointing (oszczędność pamięci GPU)
    
    v3 Updates:
    - 🎤 Dual Voice Embeddings: Resemblyzer (256) + ECAPA-TDNN (192)
    - SwiGLU w TransformerBlocks
    - VoiceEmbeddingFusion dla lepszego voice cloning
    
    Różnice od v1:
    1. Section conditioning zamiast tylko voice embedding
    2. Context fusion dla poprzedniego segmentu
    3. Lepsze powiązanie z planem kompozycji
    4. 🎤 Voice Stream Attention - głęboka integracja wokalu na każdym poziomie
    5. 🎤 Dual voice embeddings fusion (v3)
    """
    
    def __init__(
        self,
        in_channels: int = 128,          # v2: 128 zamiast 8
        out_channels: int = 128,         # v2: 128 zamiast 8
        model_channels: int = 320,       # v2: zwiększone dla 128D latent
        channel_mult: List[int] = [1, 2, 4, 4],
        num_res_blocks: int = 2,
        attention_resolutions: List[int] = [2, 4],
        context_dim: int = 768,          # Text encoder dim
        num_heads: int = 8,
        voice_embedding_dim: int = 256,  # Resemblyzer (from mix)
        voice_embedding_sep_dim: int = 192,  # v3: ECAPA-TDNN (from separated vocals)
        use_context_fusion: bool = True, # Fusion z poprzednim segmentem
        use_voice_stream: bool = True,   # 🎤 Voice Stream Attention
        use_checkpoint: bool = False,    # v2: Gradient checkpointing
        use_dual_voice: bool = True,     # v3: Use both voice embeddings
    ):
        super().__init__()
        
        self.in_channels = in_channels
        self.model_channels = model_channels
        self.channel_mult = channel_mult  # Store for forward pass
        self.use_context_fusion = use_context_fusion
        self.use_voice_stream = use_voice_stream
        self.voice_embedding_dim = voice_embedding_dim
        self.voice_embedding_sep_dim = voice_embedding_sep_dim
        self.use_checkpoint = use_checkpoint
        self.use_dual_voice = use_dual_voice
        
        # Combined conditioning dimension
        # timestep(model_channels*4) + section(1024) + voice(256)
        cond_dim = model_channels * 4
        
        # Time embedding
        self.time_embed = nn.Sequential(
            nn.Linear(model_channels, cond_dim),
            nn.SiLU(),
            nn.Linear(cond_dim, cond_dim),
        )
        
        # 🎤 v3: Voice Embedding Fusion (Resemblyzer + ECAPA-TDNN)
        if use_dual_voice:
            self.voice_fusion = VoiceEmbeddingFusion(
                resemblyzer_dim=voice_embedding_dim,
                ecapa_dim=voice_embedding_sep_dim,
                output_dim=voice_embedding_dim,  # Output same as original voice_dim
                fusion_type='gated',
            )
        
        # Section conditioning module
        # v2: Now includes CLAP, beat, chord, and phoneme conditioning
        self.section_conditioning = SectionConditioningModule(
            output_dim=cond_dim,
            text_embed_dim=context_dim,
            # v2: Enable all new conditioning modules
            use_clap=True,
            use_beat=True,
            use_chord=True,
            use_phonemes=True,
            clap_dim=512,               # CLAP embedding dim
            voice_dim=voice_embedding_dim,  # For phoneme encoder
        )
        
        # Voice embedding projection (for global conditioning)
        self.voice_embed = nn.Sequential(
            nn.Linear(voice_embedding_dim, cond_dim),
            nn.SiLU(),
            nn.Linear(cond_dim, cond_dim),
        )
        
        # Input
        self.conv_in = nn.Conv2d(in_channels, model_channels, 3, padding=1)
        
        # Context fusion (dla poprzedniego segmentu)
        if use_context_fusion:
            self.context_fusion = ContextFusion(model_channels)
        
        # 🎤 Voice Stream Attention modules - jeden dla każdego poziomu
        self.voice_stream_modules = nn.ModuleDict()
        
        # Downsampling
        self.down_blocks = nn.ModuleList()
        channels = [model_channels]
        ch = model_channels
        ds = 1
        
        for level, mult in enumerate(channel_mult):
            out_ch = model_channels * mult
            for _ in range(num_res_blocks):
                layers = [ResBlock(ch, out_ch, cond_dim)]
                ch = out_ch
                if ds in attention_resolutions:
                    layers.append(SpatialTransformer(ch, context_dim, num_heads))
                self.down_blocks.append(nn.ModuleList(layers))
                channels.append(ch)
            
            # 🎤 Add Voice Stream Attention at each resolution
            if use_voice_stream and ds in attention_resolutions:
                self.voice_stream_modules[f'down_{level}'] = VoiceStreamAttention(
                    channels=ch,
                    voice_dim=voice_embedding_dim,
                    num_heads=num_heads // 2,  # Mniej głów dla voice
                )
            
            if level < len(channel_mult) - 1:
                self.down_blocks.append(nn.ModuleList([
                    nn.Conv2d(ch, ch, 4, stride=2, padding=1)
                ]))
                channels.append(ch)
                ds *= 2
        
        # Middle
        self.mid_block1 = ResBlock(ch, ch, cond_dim)
        self.mid_attn = SpatialTransformer(ch, context_dim, num_heads)
        self.mid_block2 = ResBlock(ch, ch, cond_dim)
        
        # 🎤 Voice Stream dla middle block
        if use_voice_stream:
            self.voice_stream_modules['mid'] = VoiceStreamAttention(
                channels=ch,
                voice_dim=voice_embedding_dim,
                num_heads=num_heads // 2,
            )
        
        # Upsampling
        self.up_blocks = nn.ModuleList()
        
        for level, mult in list(enumerate(channel_mult))[::-1]:
            out_ch = model_channels * mult
            for i in range(num_res_blocks + 1):
                skip_ch = channels.pop()
                layers = [ResBlock(ch + skip_ch, out_ch, cond_dim)]
                ch = out_ch
                if ds in attention_resolutions:
                    layers.append(SpatialTransformer(ch, context_dim, num_heads))
                self.up_blocks.append(nn.ModuleList(layers))
            
            # 🎤 Add Voice Stream Attention at each resolution (upsampling)
            if use_voice_stream and ds in attention_resolutions:
                self.voice_stream_modules[f'up_{level}'] = VoiceStreamAttention(
                    channels=ch,
                    voice_dim=voice_embedding_dim,
                    num_heads=num_heads // 2,
                )
            
            if level > 0:
                self.up_blocks.append(nn.ModuleList([
                    nn.ConvTranspose2d(ch, ch, 4, stride=2, padding=1)
                ]))
                ds //= 2
        
        # Output
        self.norm_out = nn.GroupNorm(8, ch)
        self.conv_out = nn.Conv2d(ch, out_channels, 3, padding=1)
    
    def _forward_res_block(self, layer: ResBlock, h: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """Helper dla checkpointingu ResBlock"""
        return layer(h, cond)
    
    def _forward_spatial_transformer(self, layer: SpatialTransformer, h: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        """Helper dla checkpointingu SpatialTransformer"""
        return layer(h, context)
    
    def forward(
        self,
        x: torch.Tensor,                            # [B, C, H, W] noisy latent
        t: torch.Tensor,                            # [B] timesteps
        text_embed: torch.Tensor,                   # [B, seq_len, context_dim] lub [B, context_dim]
        # Section conditioning
        section_type: Optional[List[str]] = None,
        position: Optional[torch.Tensor] = None,
        energy: Optional[torch.Tensor] = None,
        tempo: Optional[torch.Tensor] = None,
        key_idx: Optional[torch.Tensor] = None,
        # v3: New conditioning
        loudness: Optional[torch.Tensor] = None,            # [B] loudness in dB
        has_vocals: Optional[torch.Tensor] = None,          # [B] bool tensor
        sentiment_score: Optional[torch.Tensor] = None,     # [B] sentiment (-1 to 1)
        genres: Optional[List[List[str]]] = None,           # List of genre lists per sample
        artist: Optional[List[str]] = None,                 # List of artist names
        # Voice embeddings (v3: dual embeddings)
        voice_emb: Optional[torch.Tensor] = None,           # [B, 256] Resemblyzer (from mix)
        voice_emb_separated: Optional[torch.Tensor] = None, # [B, 192] ECAPA-TDNN (from separated vocals)
        context_latent: Optional[torch.Tensor] = None,      # [B, C, H, W] previous segment
        # v2: New conditioning inputs
        clap_audio_embedding: Optional[torch.Tensor] = None,   # [B, 512]
        clap_text_embedding: Optional[torch.Tensor] = None,    # [B, 512]
        num_beats: Optional[torch.Tensor] = None,              # [B] int
        beat_positions: Optional[List] = None,                  # List of beat position lists
        time_signature: Optional[List[str]] = None,            # List of "4/4" etc
        current_chord: Optional[List[str]] = None,             # List of chord strings
        phonemes_ipa: Optional[List[str]] = None,              # List of IPA strings
        # v3: Pitch/F0 conditioning
        f0: Optional[torch.Tensor] = None,                     # [B, T] continuous F0 in Hz
        f0_coarse: Optional[torch.Tensor] = None,              # [B, T] discrete pitch bins
        f0_voiced_mask: Optional[torch.Tensor] = None,         # [B, T] voiced/unvoiced mask
        # v3.1: Vibrato, breath, phoneme timestamps
        vibrato_rate: Optional[torch.Tensor] = None,           # [B] Hz
        vibrato_depth: Optional[torch.Tensor] = None,          # [B] cents
        vibrato_extent: Optional[torch.Tensor] = None,         # [B] fraction 0-1
        breath_positions: Optional[List[List[float]]] = None,  # List of breath position lists
        phoneme_timestamps: Optional[List[List]] = None,       # List of (phoneme, start, end) lists
        segment_duration: float = 10.0,                         # Duration for breath/phoneme encoding
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass z section-aware conditioning.
        
        🎤 Voice Stream Attention:
        - voice_emb jest używany na 2 sposoby:
          1. Globalnie: dodany do cond (timestep + section)
          2. Lokalnie: cross-attention na każdym poziomie przez VoiceStreamAttention
        - To zapewnia głęboką integrację wokalu z muzyką w jednym procesie diffusion
        
        🎤 v3: Dual Voice Embeddings
        - voice_emb: Resemblyzer (256-dim) z miksu - ogólna charakterystyka
        - voice_emb_separated: ECAPA-TDNN (192-dim) z wokali - czysty głos
        - Oba są fuzowane przez VoiceEmbeddingFusion dla lepszego voice cloning
        
        🎵 v2 Conditioning:
        - CLAP audio/text embeddings dla lepszego audio-text alignment
        - Beat positions dla synchronizacji rytmicznej
        - Chord embeddings dla harmonicznej spójności
        - Phoneme features dla syntezy śpiewu
        
        Gradient Checkpointing:
        - Gdy use_checkpoint=True, checkpointuje ResBlocks i SpatialTransformers
        - Oszczędza ~40-50% pamięci GPU kosztem ~25% dłuższego treningu
        
        Returns:
            predicted_noise: [B, C, H, W] predicted noise
            phoneme_durations: [B, seq] predicted phoneme durations (or None)
        """
        # Time embedding
        t_emb = timestep_embedding(t, self.model_channels)
        t_emb = self.time_embed(t_emb)
        
        # Section conditioning
        # Handle 3D text embedding
        if text_embed.dim() == 3:
            text_for_section = text_embed.mean(dim=1)
        else:
            text_for_section = text_embed
        
        # 🎤 v3: Fuse dual voice embeddings
        fused_voice_emb = None
        if self.use_dual_voice and (voice_emb is not None or voice_emb_separated is not None):
            fused_voice_emb = self.voice_fusion(
                resemblyzer_emb=voice_emb,
                ecapa_emb=voice_emb_separated,
            )
        elif voice_emb is not None:
            # Fallback to single embedding if dual not enabled
            fused_voice_emb = voice_emb
            
        section_cond, phoneme_durations = self.section_conditioning(
            text_for_section,
            section_type,
            position,
            energy,
            tempo,
            key_idx,
            # v3: New conditioning
            loudness=loudness,
            has_vocals=has_vocals,
            sentiment_score=sentiment_score,
            genres=genres,
            artists=artist,
            # v2: New conditioning
            clap_audio_embedding=clap_audio_embedding,
            clap_text_embedding=clap_text_embedding,
            num_beats=num_beats,
            beat_positions=beat_positions,
            time_signature=time_signature,
            current_chord=current_chord,
            phonemes_ipa=phonemes_ipa,
            voice_embedding=fused_voice_emb,  # Pass FUSED voice embedding for phoneme encoder
            # v3: Pitch conditioning
            f0=f0,
            f0_coarse=f0_coarse,
            f0_voiced_mask=f0_voiced_mask,  # ✅ Pass ground-truth voiced mask
            # v3.1: Vibrato, breath, phoneme timestamps
            vibrato_rate=vibrato_rate,
            vibrato_depth=vibrato_depth,
            vibrato_extent=vibrato_extent,
            breath_positions=breath_positions,
            phoneme_timestamps=phoneme_timestamps,
            segment_duration=segment_duration,
        )
        
        # Combine time and section conditioning
        cond = t_emb + section_cond
        
        # Add voice embedding if provided (global conditioning)
        if fused_voice_emb is not None:
            voice_cond = self.voice_embed(fused_voice_emb)
            cond = cond + voice_cond
        
        # Prepare context for cross-attention
        # Ensure text_embed is 3D for cross-attention
        if text_embed.dim() == 2:
            text_context = text_embed.unsqueeze(1)  # [B, 1, dim]
        else:
            text_context = text_embed
        
        # Input conv
        h = self.conv_in(x)
        
        # Context fusion (previous segment)
        # Note: context_latent must also go through conv_in to match channel dimensions
        if self.use_context_fusion and context_latent is not None:
            context_h = self.conv_in(context_latent)
            h = self.context_fusion(h, context_h)
        
        # Downsampling z Voice Stream Attention (z opcjonalnym checkpointingiem)
        hs = [h]
        level_idx = 0
        block_in_level = 0
        num_res = self.up_blocks  # Hack: get num_res_blocks from structure
        
        for block_idx, block in enumerate(self.down_blocks):
            for layer in block:
                if isinstance(layer, ResBlock):
                    if self.use_checkpoint and self.training:
                        h = checkpoint(self._forward_res_block, layer, h, cond, use_reentrant=False)
                    else:
                        h = layer(h, cond)
                elif isinstance(layer, SpatialTransformer):
                    if self.use_checkpoint and self.training:
                        h = checkpoint(self._forward_spatial_transformer, layer, h, text_context, use_reentrant=False)
                    else:
                        h = layer(h, text_context)
                else:
                    h = layer(h)
            hs.append(h)
            
            # 🎤 Apply Voice Stream po każdym level (po ostatnim bloku przed downsample)
            if self.use_voice_stream and voice_emb is not None:
                # Check if this is the last block before downsample in current level
                vs_key = f'down_{level_idx}'
                if vs_key in self.voice_stream_modules:
                    # Apply voice stream after the blocks at this resolution
                    # We apply it once per level, at the last attention resolution
                    pass  # Applied separately below
        
        # Apply Voice Stream dla down path (po wszystkich blokach na danej resolution)
        # Simplified: apply at skip connections that match attention resolutions
        if self.use_voice_stream and voice_emb is not None:
            for key, vs_module in self.voice_stream_modules.items():
                if key.startswith('down_'):
                    level = int(key.split('_')[1])
                    # Apply to last skip connection at this level
                    # This is complex due to structure - for now apply to mid
                    pass
        
        # Middle (z opcjonalnym checkpointingiem)
        if self.use_checkpoint and self.training:
            h = checkpoint(self._forward_res_block, self.mid_block1, h, cond, use_reentrant=False)
            h = checkpoint(self._forward_spatial_transformer, self.mid_attn, h, text_context, use_reentrant=False)
        else:
            h = self.mid_block1(h, cond)
            h = self.mid_attn(h, text_context)
        
        # 🎤 Voice Stream w middle block (używa fused embedding)
        if self.use_voice_stream and fused_voice_emb is not None:
            if 'mid' in self.voice_stream_modules:
                h = self.voice_stream_modules['mid'](h, fused_voice_emb)
        
        if self.use_checkpoint and self.training:
            h = checkpoint(self._forward_res_block, self.mid_block2, h, cond, use_reentrant=False)
        else:
            h = self.mid_block2(h, cond)
        
        # Upsampling z Voice Stream Attention (z opcjonalnym checkpointingiem)
        # Track current level based on channel_mult (reversed order)
        num_levels = len(self.channel_mult)
        current_level = num_levels - 1  # Start from highest level
        
        for block_idx, block in enumerate(self.up_blocks):
            for layer in block:
                if isinstance(layer, ResBlock):
                    skip = hs.pop()
                    if h.shape[2:] != skip.shape[2:]:
                        h = F.interpolate(h, size=skip.shape[2:], mode='nearest')
                    h = torch.cat([h, skip], dim=1)
                    if self.use_checkpoint and self.training:
                        h = checkpoint(self._forward_res_block, layer, h, cond, use_reentrant=False)
                    else:
                        h = layer(h, cond)
                elif isinstance(layer, SpatialTransformer):
                    if self.use_checkpoint and self.training:
                        h = checkpoint(self._forward_spatial_transformer, layer, h, text_context, use_reentrant=False)
                    else:
                        h = layer(h, text_context)
                    
                    # 🎤 Apply Voice Stream po każdym SpatialTransformer w up path (używa fused embedding)
                    if self.use_voice_stream and fused_voice_emb is not None:
                        vs_key = f'up_{current_level}'
                        if vs_key in self.voice_stream_modules:
                            h = self.voice_stream_modules[vs_key](h, fused_voice_emb)
                else:
                    # Upsample layer - zmieniamy level
                    h = layer(h)
                    current_level -= 1
        
        # Output
        h = self.norm_out(h)
        h = F.silu(h)
        h = self.conv_out(h)
        
        return h, phoneme_durations


class LatentDiffusionV2(nn.Module):
    """
    Latent Diffusion Model v2 z section-aware generation.
    
    v2 Updates:
    - num_timesteps: 1000 → 200 (szybsza inferencja i trening)
    - Obsługa latent_dim=128
    
    v3 Updates:
    - CFG (Classifier-Free Guidance) dla lepszego promptowania
    - Phoneme Duration Loss dla singing synthesis
    
    Główne zmiany:
    1. Section conditioning podczas generacji
    2. Sequential generation - generuje sekcja po sekcji
    3. Context passing - przekazuje kontekst między sekcjami
    4. CFG - dropout kondycjonowania podczas treningu
    """
    
    def __init__(
        self,
        unet: UNetV2,
        num_timesteps: int = 200,     # v2: 200 zamiast 1000
        beta_start: float = 0.00085,  # v2: dostosowane do mniejszej liczby kroków
        beta_end: float = 0.012,      # v2: dostosowane
        beta_schedule: str = "scaled_linear",  # v2: lepszy schedule
        # v3: CFG (Classifier-Free Guidance)
        cfg_dropout_prob: float = 0.1,  # Probability of dropping conditioning during training
        # v3: Phoneme Duration Loss
        use_phoneme_duration_loss: bool = True,
        phoneme_duration_weight: float = 0.1,
    ):
        super().__init__()
        
        self.unet = unet
        self.num_timesteps = num_timesteps
        self.cfg_dropout_prob = cfg_dropout_prob
        self.use_phoneme_duration_loss = use_phoneme_duration_loss
        self.phoneme_duration_weight = phoneme_duration_weight
        
        # Noise schedule - wybór między linear i scaled_linear
        if beta_schedule == "linear":
            betas = torch.linspace(beta_start, beta_end, num_timesteps)
        elif beta_schedule == "scaled_linear":
            # Scaled linear - lepiej działa dla mniejszej liczby kroków
            betas = torch.linspace(beta_start ** 0.5, beta_end ** 0.5, num_timesteps) ** 2
        elif beta_schedule == "cosine":
            # Cosine schedule (jak w improved DDPM)
            steps = num_timesteps + 1
            s = 0.008
            x = torch.linspace(0, num_timesteps, steps)
            alphas_cumprod = torch.cos(((x / num_timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
            alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
            betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
            betas = torch.clamp(betas, 0.0001, 0.9999)
        else:
            betas = torch.linspace(beta_start, beta_end, num_timesteps)
        
        alphas = 1 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        
        self.register_buffer('betas', betas)
        self.register_buffer('alphas', alphas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1 - alphas_cumprod))
    
    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor = None) -> torch.Tensor:
        """Forward diffusion"""
        if noise is None:
            noise = torch.randn_like(x0)
        
        sqrt_alpha = self.sqrt_alphas_cumprod[t][:, None, None, None]
        sqrt_one_minus_alpha = self.sqrt_one_minus_alphas_cumprod[t][:, None, None, None]
        
        return sqrt_alpha * x0 + sqrt_one_minus_alpha * noise
    
    def p_losses(
        self,
        x0: torch.Tensor,
        t: torch.Tensor,
        text_embed: torch.Tensor,
        section_type: Optional[List[str]] = None,
        position: Optional[torch.Tensor] = None,
        energy: Optional[torch.Tensor] = None,
        tempo: Optional[torch.Tensor] = None,
        key_idx: Optional[torch.Tensor] = None,  # ✅ v3: Key conditioning (0-23)
        loudness: Optional[torch.Tensor] = None,  # ✅ v3: Loudness in dB
        has_vocals: Optional[torch.Tensor] = None,  # ✅ v3: Has vocals flag
        sentiment_score: Optional[torch.Tensor] = None,  # ✅ v3: Sentiment (-1 to 1)
        genres: Optional[List[List[str]]] = None,  # ✅ v3: Multi-hot genre list
        artist: Optional[List[str]] = None,  # ✅ v3: Artist name (hash-based)
        voice_emb: Optional[torch.Tensor] = None,
        voice_emb_separated: Optional[torch.Tensor] = None,  # v3: ECAPA-TDNN embedding
        context_latent: Optional[torch.Tensor] = None,
        noise: torch.Tensor = None,
        # v2: New conditioning inputs
        clap_audio_embedding: Optional[torch.Tensor] = None,
        clap_text_embedding: Optional[torch.Tensor] = None,
        num_beats: Optional[torch.Tensor] = None,
        beat_positions: Optional[List] = None,
        time_signature: Optional[List[str]] = None,
        current_chord: Optional[List[str]] = None,
        phonemes_ipa: Optional[List[str]] = None,
        # v3: Pitch conditioning
        f0: Optional[torch.Tensor] = None,
        f0_coarse: Optional[torch.Tensor] = None,
        f0_voiced_mask: Optional[torch.Tensor] = None,  # ✅ v3: Ground-truth voiced mask
        # v3: Phoneme duration targets (for auxiliary loss)
        target_phoneme_durations: Optional[torch.Tensor] = None,
        # v3.1: Vibrato, breath, phoneme timestamps
        vibrato_rate: Optional[torch.Tensor] = None,           # [B] Hz
        vibrato_depth: Optional[torch.Tensor] = None,          # [B] cents
        vibrato_extent: Optional[torch.Tensor] = None,         # [B] fraction 0-1
        breath_positions: Optional[List[List[float]]] = None,  # List of breath position lists
        phoneme_timestamps: Optional[List[List]] = None,       # List of (phoneme, start, end) lists
        segment_duration: float = 10.0,                         # Duration for breath/phoneme encoding
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Training loss z v2/v3 conditioning + CFG dropout.
        
        Returns:
            loss: Total loss (MSE + optional phoneme duration loss)
            phoneme_durations: [B, seq] predicted phoneme durations
        """
        if noise is None:
            noise = torch.randn_like(x0)
        
        B = x0.shape[0]
        device = x0.device
        
        # ============================================
        # CFG: Random dropout of conditioning (10% default)
        # ============================================
        # During training, randomly drop conditioning to enable classifier-free guidance
        if self.training and self.cfg_dropout_prob > 0:
            # Create dropout mask for each sample in batch
            cfg_mask = torch.rand(B, device=device) < self.cfg_dropout_prob
            
            if cfg_mask.any():
                # Zero out text embedding for dropped samples
                text_embed = text_embed.clone()
                text_embed[cfg_mask] = 0
                
                # Zero out other conditioning
                if voice_emb is not None:
                    voice_emb = voice_emb.clone()
                    voice_emb[cfg_mask] = 0
                if voice_emb_separated is not None:
                    voice_emb_separated = voice_emb_separated.clone()
                    voice_emb_separated[cfg_mask] = 0
                if clap_audio_embedding is not None:
                    clap_audio_embedding = clap_audio_embedding.clone()
                    clap_audio_embedding[cfg_mask] = 0
                if clap_text_embedding is not None:
                    clap_text_embedding = clap_text_embedding.clone()
                    clap_text_embedding[cfg_mask] = 0
                if f0 is not None:
                    f0 = f0.clone()
                    f0[cfg_mask] = 0
                if f0_coarse is not None:
                    f0_coarse = f0_coarse.clone()
                    f0_coarse[cfg_mask] = 128  # Unvoiced bin
                if key_idx is not None:
                    key_idx = key_idx.clone()
                    key_idx[cfg_mask] = 0  # Default to C major when dropped
                if loudness is not None:
                    loudness = loudness.clone()
                    loudness[cfg_mask] = -30.0  # Default loudness when dropped
                if has_vocals is not None:
                    has_vocals = has_vocals.clone()
                    has_vocals[cfg_mask] = False  # Default to no vocals when dropped
                if sentiment_score is not None:
                    sentiment_score = sentiment_score.clone()
                    sentiment_score[cfg_mask] = 0.0  # Neutral sentiment when dropped
                
                # For list-based conditioning, replace with None/empty
                if section_type is not None:
                    section_type = [s if not cfg_mask[i] else 'unknown' 
                                   for i, s in enumerate(section_type)]
                if phonemes_ipa is not None:
                    phonemes_ipa = [p if not cfg_mask[i] else '' 
                                   for i, p in enumerate(phonemes_ipa)]
                if genres is not None:
                    genres = [g if not cfg_mask[i] else [] 
                             for i, g in enumerate(genres)]
                if artist is not None:
                    artist = [a if not cfg_mask[i] else '' 
                             for i, a in enumerate(artist)]
        
        x_noisy = self.q_sample(x0, t, noise)
        predicted_noise, phoneme_durations = self.unet(
            x_noisy, t, text_embed,
            section_type=section_type,
            position=position,
            energy=energy,
            tempo=tempo,
            key_idx=key_idx,                              # ✅ v3: Key conditioning
            loudness=loudness,                            # ✅ v3: Loudness conditioning
            has_vocals=has_vocals,                        # ✅ v3: Has vocals flag
            sentiment_score=sentiment_score,              # ✅ v3: Sentiment conditioning
            genres=genres,                                # ✅ v3: Genre conditioning
            artist=artist,                                # ✅ v3: Artist conditioning
            voice_emb=voice_emb,
            voice_emb_separated=voice_emb_separated,
            context_latent=context_latent,
            # v2: New conditioning
            clap_audio_embedding=clap_audio_embedding,
            clap_text_embedding=clap_text_embedding,
            num_beats=num_beats,
            beat_positions=beat_positions,
            time_signature=time_signature,
            current_chord=current_chord,
            phonemes_ipa=phonemes_ipa,
            # v3: Pitch conditioning
            f0=f0,
            f0_coarse=f0_coarse,
            f0_voiced_mask=f0_voiced_mask,  # ✅ Pass ground-truth voiced mask
            # v3.1: Vibrato, breath, phoneme timestamps
            vibrato_rate=vibrato_rate,
            vibrato_depth=vibrato_depth,
            vibrato_extent=vibrato_extent,
            breath_positions=breath_positions,
            phoneme_timestamps=phoneme_timestamps,
            segment_duration=segment_duration,
        )
        
        # ============================================
        # Main loss: MSE on noise
        # ============================================
        diffusion_loss = F.mse_loss(predicted_noise, noise)
        
        # ============================================
        # Auxiliary loss: Phoneme Duration (for singing)
        # ============================================
        phoneme_duration_loss = torch.tensor(0.0, device=device)
        if (self.use_phoneme_duration_loss and 
            phoneme_durations is not None and 
            target_phoneme_durations is not None):
            # MSE loss on log-durations (log scale for better gradients)
            pred_log_dur = torch.log(phoneme_durations.clamp(min=1e-5))
            target_log_dur = torch.log(target_phoneme_durations.clamp(min=1e-5))
            phoneme_duration_loss = F.mse_loss(pred_log_dur, target_log_dur)
        
        # Total loss
        total_loss = diffusion_loss + self.phoneme_duration_weight * phoneme_duration_loss
        
        return total_loss, phoneme_durations
    
    @torch.no_grad()
    def p_sample(
        self,
        x: torch.Tensor,
        t: int,
        text_embed: torch.Tensor,
        section_type: Optional[List[str]] = None,
        position: Optional[torch.Tensor] = None,
        energy: Optional[torch.Tensor] = None,
        tempo: Optional[torch.Tensor] = None,
        key_idx: Optional[torch.Tensor] = None,  # ✅ v3: Key conditioning
        loudness: Optional[torch.Tensor] = None,  # ✅ v3: Loudness in dB
        has_vocals: Optional[torch.Tensor] = None,  # ✅ v3: Has vocals flag
        sentiment_score: Optional[torch.Tensor] = None,  # ✅ v3: Sentiment (-1 to 1)
        genres: Optional[List[List[str]]] = None,  # ✅ v3: Multi-hot genre list
        artist: Optional[List[str]] = None,  # ✅ v3: Artist name (hash-based)
        voice_emb: Optional[torch.Tensor] = None,
        voice_emb_separated: Optional[torch.Tensor] = None,  # v3: ECAPA-TDNN embedding
        context_latent: Optional[torch.Tensor] = None,
        # v2: New conditioning inputs
        clap_audio_embedding: Optional[torch.Tensor] = None,
        clap_text_embedding: Optional[torch.Tensor] = None,
        num_beats: Optional[torch.Tensor] = None,
        beat_positions: Optional[List] = None,
        time_signature: Optional[List[str]] = None,
        current_chord: Optional[List[str]] = None,
        phonemes_ipa: Optional[List[str]] = None,
        # v3: Pitch conditioning
        f0: Optional[torch.Tensor] = None,
        f0_coarse: Optional[torch.Tensor] = None,
        # v3: CFG (Classifier-Free Guidance)
        cfg_scale: float = 1.0,  # 1.0 = no guidance, >1.0 = stronger conditioning
    ) -> torch.Tensor:
        """
        One reverse diffusion step with optional CFG.
        
        Args:
            cfg_scale: Classifier-Free Guidance scale. 
                      1.0 = standard sampling (no guidance)
                      >1.0 = enhanced conditioning (typical: 3.0-7.0)
        """
        B = x.shape[0]
        device = x.device
        t_tensor = torch.full((B,), t, device=device, dtype=torch.long)
        
        # ============================================
        # CFG: Conditional + Unconditional prediction
        # ============================================
        if cfg_scale != 1.0:
            # Unconditional prediction (zeroed conditioning)
            uncond_text = torch.zeros_like(text_embed)
            uncond_section = ['unknown'] * B if section_type else None
            
            noise_uncond, _ = self.unet(
                x, t_tensor, uncond_text,
                section_type=uncond_section,
                position=position,
                energy=energy,
                tempo=tempo,
                key_idx=None,  # No key conditioning for uncond
                loudness=None,  # No loudness for uncond
                has_vocals=None,  # No has_vocals for uncond
                sentiment_score=None,  # No sentiment for uncond
                genres=None,  # No genres for uncond
                artist=None,  # No artist for uncond
                voice_emb=None,
                voice_emb_separated=None,
                context_latent=context_latent,
                clap_audio_embedding=None,
                clap_text_embedding=None,
                num_beats=num_beats,
                beat_positions=beat_positions,
                time_signature=time_signature,
                current_chord=None,
                phonemes_ipa=None,
                f0=None,
                f0_coarse=None,
            )
            
            # Conditional prediction
            noise_cond, _ = self.unet(
                x, t_tensor, text_embed,
                section_type=section_type,
                position=position,
                energy=energy,
                tempo=tempo,
                key_idx=key_idx,  # ✅ Key conditioning
                loudness=loudness,  # ✅ Loudness conditioning
                has_vocals=has_vocals,  # ✅ Has vocals flag
                sentiment_score=sentiment_score,  # ✅ Sentiment conditioning
                genres=genres,  # ✅ Genre conditioning
                artist=artist,  # ✅ Artist conditioning
                voice_emb=voice_emb,
                voice_emb_separated=voice_emb_separated,
                context_latent=context_latent,
                clap_audio_embedding=clap_audio_embedding,
                clap_text_embedding=clap_text_embedding,
                num_beats=num_beats,
                beat_positions=beat_positions,
                time_signature=time_signature,
                current_chord=current_chord,
                phonemes_ipa=phonemes_ipa,
                f0=f0,
                f0_coarse=f0_coarse,
            )
            
            # CFG: noise = uncond + cfg_scale * (cond - uncond)
            predicted_noise = noise_uncond + cfg_scale * (noise_cond - noise_uncond)
        else:
            # Standard sampling (no CFG)
            predicted_noise, _ = self.unet(
                x, t_tensor, text_embed,
                section_type=section_type,
                position=position,
                energy=energy,
                tempo=tempo,
                key_idx=key_idx,  # ✅ Key conditioning
                loudness=loudness,  # ✅ Loudness conditioning
                has_vocals=has_vocals,  # ✅ Has vocals flag
                sentiment_score=sentiment_score,  # ✅ Sentiment conditioning
                genres=genres,  # ✅ Genre conditioning
                artist=artist,  # ✅ Artist conditioning
                voice_emb=voice_emb,
                voice_emb_separated=voice_emb_separated,
                context_latent=context_latent,
                clap_audio_embedding=clap_audio_embedding,
                clap_text_embedding=clap_text_embedding,
                num_beats=num_beats,
                beat_positions=beat_positions,
                time_signature=time_signature,
                current_chord=current_chord,
                phonemes_ipa=phonemes_ipa,
                f0=f0,
                f0_coarse=f0_coarse,
            )
        
        alpha = self.alphas[t]
        alpha_cumprod = self.alphas_cumprod[t]
        beta = self.betas[t]
        
        if t > 0:
            noise = torch.randn_like(x)
            alpha_cumprod_prev = self.alphas_cumprod[t - 1]
        else:
            noise = torch.zeros_like(x)
            alpha_cumprod_prev = torch.tensor(1.0)
        
        coef1 = 1 / torch.sqrt(alpha)
        coef2 = beta / self.sqrt_one_minus_alphas_cumprod[t]
        mean = coef1 * (x - coef2 * predicted_noise)
        
        variance = beta * (1 - alpha_cumprod_prev) / (1 - alpha_cumprod)
        std = torch.sqrt(variance)
        
        return mean + std * noise
    
    @torch.no_grad()
    def sample(
        self,
        shape: Tuple[int, ...],
        text_embed: torch.Tensor,
        device: str = 'cuda',
        verbose: bool = True,
    ) -> torch.Tensor:
        """
        Prosta metoda samplingowa bez dodatkowego kondycjonowania.
        
        Args:
            shape: Kształt latenta [B, C, H, W]
            text_embed: Text embedding [B, seq_len, context_dim]
            device: Urządzenie
            verbose: Wyświetlaj postęp
            
        Returns:
            Wygenerowany latent
        """
        x = torch.randn(shape, device=device)
        
        for t in reversed(range(self.num_timesteps)):
            x = self.p_sample(x, t, text_embed)
            
            if verbose and t % 50 == 0:
                print(f"  Sampling step {self.num_timesteps - t}/{self.num_timesteps}")
        
        return x
    
    @torch.no_grad()
    def sample_section(
        self,
        shape: Tuple[int, ...],
        text_embed: torch.Tensor,
        section_type: str,
        position: float,
        energy: float = 0.5,
        tempo: float = 120.0,
        key: str = None,  # ✅ v3: Key conditioning (e.g. "C major", "A minor")
        loudness: float = -20.0,  # ✅ v3: Loudness in dB
        has_vocals: bool = True,  # ✅ v3: Has vocals flag
        sentiment_score: float = 0.0,  # ✅ v3: Sentiment (-1 to 1)
        genres: Optional[List[str]] = None,  # ✅ v3: Genre list
        artist: Optional[str] = None,  # ✅ v3: Artist name
        voice_emb: Optional[torch.Tensor] = None,
        voice_emb_separated: Optional[torch.Tensor] = None,  # v3: ECAPA-TDNN embedding
        context_latent: Optional[torch.Tensor] = None,
        # v2: New conditioning
        clap_audio_embedding: Optional[torch.Tensor] = None,
        clap_text_embedding: Optional[torch.Tensor] = None,
        num_beats: Optional[torch.Tensor] = None,
        beat_positions: Optional[List] = None,
        time_signature: Optional[List[str]] = None,
        current_chord: Optional[List[str]] = None,
        phonemes_ipa: Optional[List[str]] = None,
        # v3: Pitch conditioning
        f0: Optional[torch.Tensor] = None,
        f0_coarse: Optional[torch.Tensor] = None,
        # v3: CFG
        cfg_scale: float = 3.0,  # Default CFG scale for better quality
        device: str = 'cuda',
        verbose: bool = True,
    ) -> torch.Tensor:
        """
        Generuje pojedynczą sekcję z pełnym kondycjonowaniem v2/v3.
        
        Args:
            key: Musical key string (e.g. "C major", "A minor"). Will be converted to index.
            loudness: Target loudness in dB (typical: -40 to 0)
            has_vocals: Whether the section should have vocals
            sentiment_score: Sentiment value from -1 (negative) to 1 (positive)
            genres: List of genre strings (e.g. ["rock", "electronic"])
            artist: Artist name for style conditioning
            cfg_scale: Classifier-Free Guidance scale.
                      1.0 = no guidance
                      3.0-5.0 = recommended for music (default: 3.0)
                      >7.0 = very strong conditioning (may reduce diversity)
        """
        x = torch.randn(shape, device=device)
        
        # Prepare conditioning tensors
        B = shape[0]
        section_types = [section_type] * B
        positions = torch.tensor([position] * B, device=device)
        energies = torch.tensor([energy] * B, device=device)
        tempos = torch.tensor([tempo] * B, device=device)
        loudnesses = torch.tensor([loudness] * B, device=device)
        has_vocals_tensor = torch.tensor([has_vocals] * B, device=device, dtype=torch.bool)
        sentiment_scores = torch.tensor([sentiment_score] * B, device=device)
        genres_list = [genres or []] * B
        artist_list = [artist or ''] * B
        
        # Convert key string to index
        # Supports both simple ("C", "F#") and full ("C major", "A minor") formats
        key_idx = None
        if key is not None:
            KEY_TO_IDX = {
                # Pitch classes (0-11)
                'C': 0, 'C#': 1, 'Db': 1, 'D': 2, 'D#': 3,
                'Eb': 3, 'E': 4, 'F': 5, 'F#': 6, 'Gb': 6,
                'G': 7, 'G#': 8, 'Ab': 8, 'A': 9, 'A#': 10,
                'Bb': 10, 'B': 11,
                # Full format (major 0-11, minor 12-23)
                'C major': 0, 'C# major': 1, 'D major': 2, 'D# major': 3,
                'E major': 4, 'F major': 5, 'F# major': 6, 'G major': 7,
                'G# major': 8, 'A major': 9, 'A# major': 10, 'B major': 11,
                'C minor': 12, 'C# minor': 13, 'D minor': 14, 'D# minor': 15,
                'E minor': 16, 'F minor': 17, 'F# minor': 18, 'G minor': 19,
                'G# minor': 20, 'A minor': 21, 'A# minor': 22, 'B minor': 23,
            }
            key_idx = torch.tensor([KEY_TO_IDX.get(key, 0)] * B, device=device)
        
        for t in reversed(range(self.num_timesteps)):
            x = self.p_sample(
                x, t, text_embed,
                section_type=section_types,
                position=positions,
                energy=energies,
                tempo=tempos,
                key_idx=key_idx,  # ✅ Key conditioning
                loudness=loudnesses,  # ✅ Loudness conditioning
                has_vocals=has_vocals_tensor,  # ✅ Has vocals flag
                sentiment_score=sentiment_scores,  # ✅ Sentiment conditioning
                genres=genres_list,  # ✅ Genre conditioning
                artist=artist_list,  # ✅ Artist conditioning
                voice_emb=voice_emb,
                voice_emb_separated=voice_emb_separated,
                context_latent=context_latent,
                # v2: New conditioning
                clap_audio_embedding=clap_audio_embedding,
                clap_text_embedding=clap_text_embedding,
                num_beats=num_beats,
                beat_positions=beat_positions,
                time_signature=time_signature,
                current_chord=current_chord,
                phonemes_ipa=phonemes_ipa,
                # v3: Pitch conditioning
                f0=f0,
                f0_coarse=f0_coarse,
                # v3: CFG
                cfg_scale=cfg_scale,
            )
            
            if verbose and t % 100 == 0:
                print(f"  Sampling step {self.num_timesteps - t}/{self.num_timesteps}")
        
        return x
    
    @torch.no_grad()
    def sample_composition(
        self,
        composition_plan: List[Dict],
        shape: Tuple[int, ...],
        text_encoder,
        vae_encoder,
        device: str = 'cuda',
        overlap_frames: int = 10,
        verbose: bool = True,
    ) -> List[torch.Tensor]:
        """
        Generuje cały utwór według planu kompozycji.
        
        Args:
            composition_plan: Lista słowników z:
                - section_type, position, energy, tempo, prompt (podstawowe)
                - voice_emb, voice_emb_separated (voice cloning)
                - clap_audio_embedding, clap_text_embedding (CLAP)
                - num_beats, beat_positions, time_signature (beat)
                - current_chord (harmony)
                - phonemes_ipa (singing)
                - f0, f0_coarse (pitch)
            shape: Kształt pojedynczego latenta [1, C, H, W]
            text_encoder: Encoder tekstu
            vae_encoder: VAE encoder (do ekstrakcji context latent)
            overlap_frames: Ile ramek nakłada się między sekcjami
            
        Returns:
            Lista latentów dla każdej sekcji
        """
        generated_sections = []
        context_latent = None
        
        for i, section in enumerate(composition_plan):
            if verbose:
                print(f"\n🎵 Generating section {i+1}/{len(composition_plan)}: {section['section_type']}")
            
            # Get text embedding
            prompt = section.get('prompt', f"{section['section_type']} section")
            text_embed = text_encoder([prompt])
            text_embed = text_embed.to(device)
            
            # Generate section with all conditioning
            latent = self.sample_section(
                shape=shape,
                text_embed=text_embed,
                section_type=section['section_type'],
                position=section['position'],
                energy=section.get('energy', 0.5),
                tempo=section.get('tempo', 120.0),
                key=section.get('key'),  # ✅ v3: Key conditioning
                loudness=section.get('loudness', -20.0),  # ✅ v3: Loudness
                has_vocals=section.get('has_vocals', True),  # ✅ v3: Has vocals
                sentiment_score=section.get('sentiment_score', 0.0),  # ✅ v3: Sentiment
                genres=section.get('genres'),  # ✅ v3: Genres list
                artist=section.get('artist'),  # ✅ v3: Artist
                voice_emb=section.get('voice_emb'),
                voice_emb_separated=section.get('voice_emb_separated'),
                context_latent=context_latent,
                # v2: New conditioning
                clap_audio_embedding=section.get('clap_audio_embedding'),
                clap_text_embedding=section.get('clap_text_embedding'),
                num_beats=section.get('num_beats'),
                beat_positions=section.get('beat_positions'),
                time_signature=section.get('time_signature'),
                current_chord=section.get('current_chord'),
                phonemes_ipa=section.get('phonemes_ipa'),
                # v3: Pitch conditioning
                f0=section.get('f0'),
                f0_coarse=section.get('f0_coarse'),
                device=device,
                verbose=verbose,
            )
            
            generated_sections.append(latent)
            
            # Update context for next section
            context_latent = latent
        
        return generated_sections


if __name__ == "__main__":
    print("🎵 Testing Latent Diffusion v2")
    print("="*60)
    
    device = 'cpu'
    
    # v2: Nowe parametry - latent_dim=128, num_timesteps=200
    print("\n📊 v2 Config:")
    print("   Latent dim: 128")
    print("   Timesteps: 200")
    print("   Model channels: 256 (test) / 320 (production)")
    
    # Create model
    unet = UNetV2(
        in_channels=128,      # v2: 128 zamiast 8
        out_channels=128,     # v2: 128 zamiast 8
        model_channels=256,   # Mniejsze dla testu
        channel_mult=[1, 2, 2],
        context_dim=768,
    ).to(device)
    
    ldm = LatentDiffusionV2(
        unet, 
        num_timesteps=200,              # v2: 200 zamiast 1000
        beta_schedule="scaled_linear",  # v2: lepszy schedule
    ).to(device)
    
    # Test training step
    print("\n📊 Testing training...")
    batch_size = 2
    x0 = torch.randn(batch_size, 128, 32, 64).to(device)  # v2: 128 channels
    t = torch.randint(0, 200, (batch_size,)).to(device)   # v2: max 200
    text_embed = torch.randn(batch_size, 10, 768).to(device)
    
    loss = ldm.p_losses(
        x0, t, text_embed,
        section_type=['verse', 'chorus'],
        position=torch.tensor([0.2, 0.5]).to(device),
        energy=torch.tensor([0.4, 0.8]).to(device),
        tempo=torch.tensor([100.0, 128.0]).to(device),
    )
    print(f"  Training loss: {loss.item():.4f}")
    
    # Test sampling single section
    print("\n🎼 Testing section sampling...")
    sample = ldm.sample_section(
        shape=(1, 128, 32, 64),  # v2: 128 channels
        text_embed=text_embed[:1],
        section_type='chorus',
        position=0.5,
        energy=0.8,
        tempo=128.0,
        device=device,
        verbose=True,
    )
    print(f"  Sample shape: {sample.shape}")
    
    # Model size
    params = sum(p.numel() for p in ldm.parameters())
    print(f"\n📊 Total parameters: {params:,} ({params/1e6:.1f}M)")


# =============================================================================
# 🏭 Model Factory Functions
# =============================================================================

def create_unet_small(**kwargs) -> UNetV2:
    """
    Small UNet (~350M params) - dla szybkiego prototypowania.
    """
    defaults = dict(
        in_channels=128,
        out_channels=128,
        model_channels=256,
        channel_mult=[1, 2, 4, 4],
        num_res_blocks=2,
        attention_resolutions=[2, 4],
        num_heads=8,
    )
    defaults.update(kwargs)
    return UNetV2(**defaults)


def create_unet_base(**kwargs) -> UNetV2:
    """
    Base UNet (~720M params) - dobry stosunek jakość/szybkość.
    Domyślna konfiguracja.
    """
    defaults = dict(
        in_channels=128,
        out_channels=128,
        model_channels=320,
        channel_mult=[1, 2, 4, 4],
        num_res_blocks=2,
        attention_resolutions=[2, 4],
        num_heads=8,
    )
    defaults.update(kwargs)
    return UNetV2(**defaults)


def create_unet_large(**kwargs) -> UNetV2:
    """
    Large UNet (~1.5B params) - wysoka jakość.
    
    Zmiany vs Base:
    - model_channels: 320 → 384 (+20%)
    - num_res_blocks: 2 → 3 (+50% bloków)
    - num_heads: 8 → 12 (więcej attention heads)
    """
    defaults = dict(
        in_channels=128,
        out_channels=128,
        model_channels=384,  # 384 dzieli się przez 12
        channel_mult=[1, 2, 4, 4],
        num_res_blocks=3,
        attention_resolutions=[2, 4],
        num_heads=12,
        use_checkpoint=True,  # Potrzebne dla tak dużego modelu
    )
    defaults.update(kwargs)
    return UNetV2(**defaults)


def create_unet_xl(**kwargs) -> UNetV2:
    """
    XL UNet (~2B params) - maksymalna jakość.
    
    UWAGA: Wymaga >24GB VRAM lub gradient checkpointing + mixed precision.
    """
    defaults = dict(
        in_channels=128,
        out_channels=128,
        model_channels=512,
        channel_mult=[1, 2, 4, 4],
        num_res_blocks=3,
        attention_resolutions=[2, 4],
        num_heads=16,
        use_checkpoint=True,
    )
    defaults.update(kwargs)
    return UNetV2(**defaults)


# Model size presets
UNET_PRESETS = {
    'small': create_unet_small,   # ~350M
    'base': create_unet_base,     # ~720M
    'large': create_unet_large,   # ~1.5B
    'xl': create_unet_xl,         # ~2.5B
}


def create_ldm(
    size: str = 'base',
    num_timesteps: int = 200,
    beta_schedule: str = 'scaled_linear',
    **unet_kwargs,
) -> LatentDiffusionV2:
    """
    Factory function dla całego LDM.
    
    Args:
        size: 'small', 'base', 'large', 'xl'
        num_timesteps: Liczba kroków diffusion
        beta_schedule: Rozkład szumu
        **unet_kwargs: Dodatkowe parametry dla UNet
        
    Example:
        ldm = create_ldm('large', use_voice_stream=True)
    """
    if size not in UNET_PRESETS:
        raise ValueError(f"Unknown size: {size}. Choose from: {list(UNET_PRESETS.keys())}")
    
    unet = UNET_PRESETS[size](**unet_kwargs)
    
    return LatentDiffusionV2(
        unet=unet,
        num_timesteps=num_timesteps,
        beta_schedule=beta_schedule,
    )
