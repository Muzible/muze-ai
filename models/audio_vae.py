"""
Audio VAE - Variational Autoencoder dla spektrogramów mel
Kompresuje audio do przestrzeni latentnej i rekonstruuje

v2 Updates:
- latent_dim: 8 → 128 (większa pojemność)
- sample_rate: 22050 → 32000 (lepsza jakość)
- Multi-Resolution STFT Loss (lepsza rekonstrukcja)
- Gradient Checkpointing (oszczędność pamięci GPU)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import torchaudio.transforms as T
from typing import Tuple, Optional, List
from torch.utils.checkpoint import checkpoint
import math


# ============================================================================
# Multi-Resolution STFT Loss
# ============================================================================

class MultiResolutionSTFTLoss(nn.Module):
    """
    Multi-Resolution STFT Loss dla lepszej rekonstrukcji audio.
    
    Oblicza loss na spektrogramach o różnych rozdzielczościach czasowo-częstotliwościowych,
    co pozwala modelowi lepiej odwzorowywać zarówno szybkie transjenty jak i wolne zmiany.
    """
    
    def __init__(
        self,
        fft_sizes: List[int] = [512, 1024, 2048],
        hop_sizes: List[int] = [128, 256, 512],
        win_lengths: List[int] = [512, 1024, 2048],
        window: str = "hann",
        sc_weight: float = 1.0,  # Spectral Convergence weight
        mag_weight: float = 1.0,  # Magnitude loss weight
    ):
        super().__init__()
        
        assert len(fft_sizes) == len(hop_sizes) == len(win_lengths)
        
        self.fft_sizes = fft_sizes
        self.hop_sizes = hop_sizes
        self.win_lengths = win_lengths
        self.sc_weight = sc_weight
        self.mag_weight = mag_weight
        
        # Pre-compute windows
        self.windows = nn.ParameterList()
        for win_length in win_lengths:
            if window == "hann":
                win = torch.hann_window(win_length)
            elif window == "hamming":
                win = torch.hamming_window(win_length)
            else:
                win = torch.ones(win_length)
            # Register as buffer, not parameter (not trainable)
            self.register_buffer(f'window_{win_length}', win)
    
    def stft(self, x: torch.Tensor, fft_size: int, hop_size: int, win_length: int) -> torch.Tensor:
        """Compute STFT magnitude."""
        window = getattr(self, f'window_{win_length}')
        
        # Ensure x is 2D [B, T]
        if x.dim() == 3:
            x = x.squeeze(1)
        
        # STFT
        stft_out = torch.stft(
            x,
            n_fft=fft_size,
            hop_length=hop_size,
            win_length=win_length,
            window=window.to(x.device),
            return_complex=True,
            pad_mode='reflect',
        )
        
        # Magnitude
        mag = torch.abs(stft_out)  # [B, freq, time]
        return mag
    
    def spectral_convergence_loss(self, y_mag: torch.Tensor, y_hat_mag: torch.Tensor) -> torch.Tensor:
        """Spectral Convergence Loss - normalized Frobenius norm."""
        return torch.norm(y_mag - y_hat_mag, p='fro') / (torch.norm(y_mag, p='fro') + 1e-8)
    
    def log_magnitude_loss(self, y_mag: torch.Tensor, y_hat_mag: torch.Tensor) -> torch.Tensor:
        """Log Magnitude Loss - L1 on log spectrograms."""
        return F.l1_loss(torch.log(y_mag + 1e-8), torch.log(y_hat_mag + 1e-8))
    
    def forward(self, y: torch.Tensor, y_hat: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute multi-resolution STFT loss.
        
        Args:
            y: Target waveform [B, T] or [B, 1, T]
            y_hat: Predicted waveform [B, T] or [B, 1, T]
            
        Returns:
            Tuple of (spectral_convergence_loss, magnitude_loss)
        """
        sc_loss = 0.0
        mag_loss = 0.0
        
        for fft_size, hop_size, win_length in zip(self.fft_sizes, self.hop_sizes, self.win_lengths):
            y_mag = self.stft(y, fft_size, hop_size, win_length)
            y_hat_mag = self.stft(y_hat, fft_size, hop_size, win_length)
            
            # Ensure same size
            min_time = min(y_mag.shape[-1], y_hat_mag.shape[-1])
            y_mag = y_mag[..., :min_time]
            y_hat_mag = y_hat_mag[..., :min_time]
            
            sc_loss += self.spectral_convergence_loss(y_mag, y_hat_mag)
            mag_loss += self.log_magnitude_loss(y_mag, y_hat_mag)
        
        sc_loss /= len(self.fft_sizes)
        mag_loss /= len(self.fft_sizes)
        
        return self.sc_weight * sc_loss, self.mag_weight * mag_loss


class ResidualBlock(nn.Module):
    """Blok rezydualny z GroupNorm"""
    
    def __init__(self, in_channels: int, out_channels: int, groups: int = 8):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.norm1 = nn.GroupNorm(groups, out_channels)
        self.norm2 = nn.GroupNorm(groups, out_channels)
        
        self.skip = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv1(x)
        h = self.norm1(h)
        h = F.silu(h)
        h = self.conv2(h)
        h = self.norm2(h)
        return F.silu(h + self.skip(x))


class AttentionBlock(nn.Module):
    """Self-attention dla feature maps"""
    
    def __init__(self, channels: int, num_heads: int = 4):
        super().__init__()
        self.norm = nn.GroupNorm(8, channels)
        self.attention = nn.MultiheadAttention(channels, num_heads, batch_first=True)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        x_norm = self.norm(x)
        x_flat = x_norm.view(b, c, h * w).permute(0, 2, 1)  # [B, H*W, C]
        attn_out, _ = self.attention(x_flat, x_flat, x_flat)
        attn_out = attn_out.permute(0, 2, 1).view(b, c, h, w)
        return x + attn_out


class Encoder(nn.Module):
    """Encoder: Mel-spectrogram -> Latent
    
    Supports gradient checkpointing for memory efficiency.
    """
    
    def __init__(
        self,
        in_channels: int = 1,
        latent_dim: int = 8,
        channels: list = [64, 128, 256, 512],
        num_res_blocks: int = 2,
        use_checkpoint: bool = False,  # Gradient checkpointing
    ):
        super().__init__()
        
        self.use_checkpoint = use_checkpoint
        self.conv_in = nn.Conv2d(in_channels, channels[0], 3, padding=1)
        
        # Downsampling blocks
        self.down_blocks = nn.ModuleList()
        in_ch = channels[0]
        for i, out_ch in enumerate(channels):
            block = nn.ModuleList()
            for _ in range(num_res_blocks):
                block.append(ResidualBlock(in_ch, out_ch))
                in_ch = out_ch
            if i < len(channels) - 1:
                block.append(nn.Conv2d(out_ch, out_ch, 4, stride=2, padding=1))  # Downsample
            if i >= 2:  # Attention on deeper layers
                block.append(AttentionBlock(out_ch))
            self.down_blocks.append(block)
        
        # Middle
        self.mid_block1 = ResidualBlock(channels[-1], channels[-1])
        self.mid_attn = AttentionBlock(channels[-1])
        self.mid_block2 = ResidualBlock(channels[-1], channels[-1])
        
        # Output -> mean i log_var
        self.norm_out = nn.GroupNorm(8, channels[-1])
        self.conv_out = nn.Conv2d(channels[-1], latent_dim * 2, 3, padding=1)
    
    def _forward_block(self, h: torch.Tensor, block: nn.ModuleList) -> torch.Tensor:
        """Forward through a single block (for checkpointing)"""
        for layer in block:
            h = layer(h)
        return h
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.conv_in(x)
        
        for block in self.down_blocks:
            if self.use_checkpoint and self.training:
                h = checkpoint(self._forward_block, h, block, use_reentrant=False)
            else:
                for layer in block:
                    h = layer(h)
        
        # Middle (always checkpointed if enabled)
        if self.use_checkpoint and self.training:
            h = checkpoint(self.mid_block1, h, use_reentrant=False)
            h = checkpoint(self.mid_attn, h, use_reentrant=False)
            h = checkpoint(self.mid_block2, h, use_reentrant=False)
        else:
            h = self.mid_block1(h)
            h = self.mid_attn(h)
            h = self.mid_block2(h)
        
        h = self.norm_out(h)
        h = F.silu(h)
        h = self.conv_out(h)
        
        mean, log_var = torch.chunk(h, 2, dim=1)
        return mean, log_var


class Decoder(nn.Module):
    """Decoder: Latent -> Mel-spectrogram
    
    Supports gradient checkpointing for memory efficiency.
    """
    
    def __init__(
        self,
        out_channels: int = 1,
        latent_dim: int = 8,
        channels: list = [512, 256, 128, 64],
        num_res_blocks: int = 2,
        use_checkpoint: bool = False,  # Gradient checkpointing
    ):
        super().__init__()
        
        self.use_checkpoint = use_checkpoint
        self.conv_in = nn.Conv2d(latent_dim, channels[0], 3, padding=1)
        
        # Middle
        self.mid_block1 = ResidualBlock(channels[0], channels[0])
        self.mid_attn = AttentionBlock(channels[0])
        self.mid_block2 = ResidualBlock(channels[0], channels[0])
        
        # Upsampling blocks
        self.up_blocks = nn.ModuleList()
        in_ch = channels[0]
        for i, out_ch in enumerate(channels):
            block = nn.ModuleList()
            for _ in range(num_res_blocks):
                block.append(ResidualBlock(in_ch, out_ch))
                in_ch = out_ch
            if i < len(channels) - 1:
                block.append(nn.ConvTranspose2d(out_ch, out_ch, 4, stride=2, padding=1))  # Upsample
            if i < 2:  # Attention on deeper layers
                block.append(AttentionBlock(out_ch))
            self.up_blocks.append(block)
        
        # Output
        self.norm_out = nn.GroupNorm(8, channels[-1])
        self.conv_out = nn.Conv2d(channels[-1], out_channels, 3, padding=1)
    
    def _forward_block(self, h: torch.Tensor, block: nn.ModuleList) -> torch.Tensor:
        """Forward through a single block (for checkpointing)"""
        for layer in block:
            h = layer(h)
        return h
    
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        h = self.conv_in(z)
        
        # Middle (checkpointed if enabled)
        if self.use_checkpoint and self.training:
            h = checkpoint(self.mid_block1, h, use_reentrant=False)
            h = checkpoint(self.mid_attn, h, use_reentrant=False)
            h = checkpoint(self.mid_block2, h, use_reentrant=False)
        else:
            h = self.mid_block1(h)
            h = self.mid_attn(h)
            h = self.mid_block2(h)
        
        for block in self.up_blocks:
            if self.use_checkpoint and self.training:
                h = checkpoint(self._forward_block, h, block, use_reentrant=False)
            else:
                for layer in block:
                    h = layer(h)
        
        h = self.norm_out(h)
        h = F.silu(h)
        h = self.conv_out(h)
        return h


class AudioVAE(nn.Module):
    """
    Variational Autoencoder dla audio (mel-spektrogramy)
    
    v2 Updates:
    - latent_dim: 8 → 128 (większa pojemność)
    - sample_rate: 22050 → 32000 (lepsza jakość)
    - Multi-Resolution STFT Loss (lepsza rekonstrukcja)
    - Gradient Checkpointing (oszczędność pamięci GPU)
    
    Pipeline:
    1. Audio -> Mel-spectrogram
    2. Mel -> Encoder -> Latent (z)
    3. Latent -> Decoder -> Reconstructed Mel
    4. Mel -> Audio (vocoder, np. HiFi-GAN)
    """
    
    # Default channels for different latent sizes
    LATENT_CONFIGS = {
        8: [64, 128, 256, 512],        # Small (~55M params)
        32: [64, 128, 256, 512],        # Medium (~56M params)
        64: [96, 192, 384, 768],        # Large (~125M params)
        128: [128, 256, 512, 1024],     # Very large (~224M params)
        256: [192, 384, 768, 1536],     # XXL (~450M params)
        512: [256, 512, 1024, 2048],    # Maximum (~890M params)
    }
    
    def __init__(
        self,
        sample_rate: int = 32000,  # v2: 32kHz zamiast 22050
        n_mels: int = 128,
        n_fft: int = 1024,
        hop_length: int = 320,     # v2: dostosowane do 32kHz (10ms hop)
        latent_dim: int = 128,     # v2: 128 zamiast 8
        channels: list = None,     # Auto-select based on latent_dim
        use_stft_loss: bool = True,  # v2: Multi-Resolution STFT Loss
        use_checkpoint: bool = False,  # v2: Gradient checkpointing
    ):
        super().__init__()
        
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.hop_length = hop_length
        self.latent_dim = latent_dim
        self.use_stft_loss = use_stft_loss
        self.use_checkpoint = use_checkpoint
        
        # Auto-select channels based on latent_dim
        if channels is None:
            channels = self.LATENT_CONFIGS.get(latent_dim, [128, 256, 512, 1024])
        
        # Mel spectrogram transform
        self.mel_transform = T.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            normalized=True,
        )
        
        # Griffin-Lim dla rekonstrukcji audio z mel (do STFT loss)
        if use_stft_loss:
            self.inverse_mel = T.InverseMelScale(
                n_stft=n_fft // 2 + 1,
                n_mels=n_mels,
                sample_rate=sample_rate,
            )
            self.griffin_lim = T.GriffinLim(
                n_fft=n_fft,
                hop_length=hop_length,
                power=1.0,
                n_iter=32,
            )
            # Multi-Resolution STFT Loss
            self.stft_loss = MultiResolutionSTFTLoss(
                fft_sizes=[512, 1024, 2048],
                hop_sizes=[128, 256, 512],
                win_lengths=[512, 1024, 2048],
            )
        
        # VAE components (with optional gradient checkpointing)
        self.encoder = Encoder(
            in_channels=1,
            latent_dim=latent_dim,
            channels=channels,
            use_checkpoint=use_checkpoint,
        )
        self.decoder = Decoder(
            out_channels=1,
            latent_dim=latent_dim,
            channels=channels[::-1],
            use_checkpoint=use_checkpoint,
        )
    
    def audio_to_mel(self, audio: torch.Tensor) -> torch.Tensor:
        """Konwertuje audio waveform na mel-spectrogram"""
        mel = self.mel_transform(audio)
        mel = torch.log(mel + 1e-5)  # Log scale
        return mel.unsqueeze(1)  # Add channel dim [B, 1, n_mels, time]
    
    def reparameterize(self, mean: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        """Reparametrization trick"""
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mean + eps * std
    
    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Enkoduje mel-spectrogram do przestrzeni latentnej"""
        mean, log_var = self.encoder(x)
        z = self.reparameterize(mean, log_var)
        return z, mean, log_var
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Dekoduje z przestrzeni latentnej do mel-spectrogramu"""
        return self.decoder(z)
    
    def forward(self, audio: torch.Tensor) -> dict:
        """
        Forward pass
        
        Args:
            audio: [B, samples] audio waveform
            
        Returns:
            dict z: latent, mean, log_var, mel_input, mel_recon
        """
        # Audio -> Mel
        mel = self.audio_to_mel(audio)
        
        # Encode
        z, mean, log_var = self.encode(mel)
        
        # Decode
        mel_recon = self.decode(z)
        
        return {
            'z': z,
            'mean': mean,
            'log_var': log_var,
            'mel_input': mel,
            'mel_recon': mel_recon,
        }
    
    def loss(
        self, 
        output: dict, 
        beta: float = 0.001,
        audio_input: torch.Tensor = None,  # v2: dla STFT loss
        stft_weight: float = 0.5,          # v2: waga STFT loss
    ) -> dict:
        """
        VAE loss: reconstruction + KL divergence + Multi-Resolution STFT
        
        Args:
            output: dict from forward()
            beta: weight dla KL term (beta-VAE)
            audio_input: oryginalne audio [B, samples] dla STFT loss
            stft_weight: waga dla multi-resolution STFT loss
        """
        mel_recon = output['mel_recon']
        mel_input = output['mel_input']
        
        # Match sizes (encoder/decoder may have different sizes due to stride)
        min_time = min(mel_recon.shape[-1], mel_input.shape[-1])
        mel_recon = mel_recon[..., :min_time]
        mel_input = mel_input[..., :min_time]
        
        # Reconstruction loss (MSE na mel-spectrogramach)
        recon_loss = F.mse_loss(mel_recon, mel_input)
        
        # KL divergence
        kl_loss = -0.5 * torch.mean(
            1 + output['log_var'] - output['mean'].pow(2) - output['log_var'].exp()
        )
        
        total_loss = recon_loss + beta * kl_loss
        
        loss_dict = {
            'total': total_loss,
            'recon': recon_loss,
            'kl': kl_loss,
        }
        
        # v2: Multi-Resolution STFT Loss (if audio available)
        if self.use_stft_loss and audio_input is not None:
            try:
                # Rekonstruuj audio z mel
                audio_recon = self.mel_to_audio(mel_recon)
                
                # Match length
                min_len = min(audio_input.shape[-1], audio_recon.shape[-1])
                audio_input_trimmed = audio_input[..., :min_len]
                audio_recon_trimmed = audio_recon[..., :min_len]
                
                # STFT loss
                sc_loss, mag_loss = self.stft_loss(audio_input_trimmed, audio_recon_trimmed)
                stft_total = sc_loss + mag_loss
                
                total_loss = total_loss + stft_weight * stft_total
                
                loss_dict.update({
                    'total': total_loss,
                    'stft_sc': sc_loss,
                    'stft_mag': mag_loss,
                    'stft_total': stft_total,
                })
            except Exception as e:
                # Fallback if STFT loss doesn't work
                pass
        
        return loss_dict
    
    def mel_to_audio(self, mel: torch.Tensor) -> torch.Tensor:
        """
        Konwertuje mel-spectrogram z powrotem na audio (przybliżenie).
        Używane do STFT loss podczas treningu.
        
        Args:
            mel: [B, 1, n_mels, time] log-mel spectrogram
            
        Returns:
            [B, samples] audio waveform
        """
        # Remove channel dim and exp (undo log)
        mel = mel.squeeze(1)  # [B, n_mels, time]
        mel = torch.exp(mel) - 1e-5  # Undo log scale
        mel = torch.clamp(mel, min=0)
        
        # Inverse mel scale
        spec = self.inverse_mel(mel)
        
        # Griffin-Lim
        audio = self.griffin_lim(spec)
        
        return audio


if __name__ == "__main__":
    # Test
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print("="*60)
    print("🎵 Testing Audio VAE v2")
    print("="*60)
    
    # v2: Nowe parametry
    model = AudioVAE(
        sample_rate=32000,    # v2: 32kHz
        n_mels=128,
        latent_dim=128,       # v2: 128 zamiast 8
        use_stft_loss=True,   # v2: Multi-Resolution STFT Loss
    ).to(device)
    
    # Fake audio (5 seconds at 32kHz)
    audio = torch.randn(2, 32000 * 5).to(device)
    
    print(f"\n📊 Model Config:")
    print(f"   Sample rate: {model.sample_rate} Hz")
    print(f"   Latent dim: {model.latent_dim}")
    print(f"   N mels: {model.n_mels}")
    print(f"   Use STFT loss: {model.use_stft_loss}")
    
    output = model(audio)
    
    # Test z STFT loss
    loss = model.loss(output, audio_input=audio, stft_weight=0.5)
    
    print(f"\n📐 Shapes:")
    print(f"   Input audio: {audio.shape}")
    print(f"   Mel input: {output['mel_input'].shape}")
    print(f"   Latent z: {output['z'].shape}")
    print(f"   Mel recon: {output['mel_recon'].shape}")
    
    print(f"\n📉 Losses:")
    print(f"   Total: {loss['total']:.4f}")
    print(f"   Recon (mel): {loss['recon']:.4f}")
    print(f"   KL: {loss['kl']:.4f}")
    if 'stft_total' in loss:
        print(f"   STFT SC: {loss['stft_sc']:.4f}")
        print(f"   STFT Mag: {loss['stft_mag']:.4f}")
        print(f"   STFT Total: {loss['stft_total']:.4f}")
    
    # Number of parameters
    params = sum(p.numel() for p in model.parameters())
    print(f"\n📊 Total parameters: {params:,} ({params/1e6:.1f}M)")
    
    # Test mniejszej wersji
    print("\n" + "="*60)
    print("🔬 Testing smaller latent_dim=32 variant")
    model_small = AudioVAE(
        sample_rate=32000,
        latent_dim=32,
        use_stft_loss=False,  # Szybciej bez STFT
    ).to(device)
    
    output_small = model_small(audio)
    print(f"   Latent z shape: {output_small['z'].shape}")
    params_small = sum(p.numel() for p in model_small.parameters())
    print(f"   Parameters: {params_small:,} ({params_small/1e6:.1f}M)")
