"""
🎵 Vocoder Fine-tuning Script

Dotrenowanie HiFi-GAN na mel-spektrogramach generowanych przez nasz VAE.

Dlaczego to ważne:
- Pretrained HiFi-GAN jest trenowany na "prawdziwych" mel spektrogramach
- Nasz VAE generuje mel spektrogramy, które mogą się różnić od prawdziwych
- Fine-tuning dostosowuje vocoder do charakterystyki outputu VAE
- Rezultat: lepsza jakość finalnego audio

Pipeline:
1. Załaduj wytrenowany VAE
2. Wygeneruj mel spektrogramy z VAE (reconstruction)
3. Trenuj HiFi-GAN na parach (VAE mel, prawdziwe audio)

GAN Training:
- Generator (HiFi-GAN): mel → audio
- Discriminator (MPD + MSD): rozróżnia prawdziwe od wygenerowanych
- Feature Matching Loss: dopasowanie feature maps dyskryminatora
- Mel Loss: mel(generated) ≈ input mel
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import torchaudio.transforms as T
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import ExponentialLR
from tqdm import tqdm
import argparse
from typing import Dict, Optional, List
import json


def feature_loss(fmap_r: List[torch.Tensor], fmap_g: List[torch.Tensor]) -> torch.Tensor:
    """Feature matching loss - dopasowanie feature maps dyskryminatora"""
    loss = 0
    for dr, dg in zip(fmap_r, fmap_g):
        for rl, gl in zip(dr, dg):
            loss += F.l1_loss(rl, gl)
    return loss * 2  # Skalowanie jak w oryginalnym HiFi-GAN


def discriminator_loss(disc_real_outputs: List[torch.Tensor], 
                       disc_generated_outputs: List[torch.Tensor]) -> tuple:
    """
    Discriminator loss (hinge loss).
    
    Real powinien być > 1, fake powinien być < -1.
    """
    loss = 0
    r_losses = []
    g_losses = []
    
    for dr, dg in zip(disc_real_outputs, disc_generated_outputs):
        r_loss = torch.mean(torch.clamp(1 - dr, min=0))
        g_loss = torch.mean(torch.clamp(1 + dg, min=0))
        loss += r_loss + g_loss
        r_losses.append(r_loss.item())
        g_losses.append(g_loss.item())
    
    return loss, r_losses, g_losses


def generator_loss(disc_outputs: List[torch.Tensor]) -> torch.Tensor:
    """Generator adversarial loss - chce żeby fake wyglądał jak real (> 0)"""
    loss = 0
    for dg in disc_outputs:
        loss += torch.mean(torch.clamp(1 - dg, min=0))
    return loss


class VocoderTrainer:
    """
    Trener dla HiFi-GAN zintegrowany z VAE.
    
    Workflow:
    1. Załaduj batch audio
    2. Audio → VAE → reconstructed mel
    3. Reconstructed mel → HiFi-GAN → generated audio
    4. Trenuj dyskryminatory: rozróżnij real vs generated
    5. Trenuj generator: oszukaj dyskryminatory + mel matching
    """
    
    def __init__(
        self,
        generator: nn.Module,
        mpd: nn.Module,  # Multi-Period Discriminator
        msd: nn.Module,  # Multi-Scale Discriminator
        vae: nn.Module,  # AudioVAE (zamrożony)
        sample_rate: int = 32000,
        n_mels: int = 128,
        hop_length: int = 320,
        device: str = 'cuda',
    ):
        self.generator = generator.to(device)
        self.mpd = mpd.to(device)
        self.msd = msd.to(device)
        self.vae = vae.to(device)
        self.device = device
        
        # Zamroź VAE
        self.vae.eval()
        for param in self.vae.parameters():
            param.requires_grad = False
        
        # Mel transform dla loss calculation
        self.mel_transform = T.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=1024,
            hop_length=hop_length,
            n_mels=n_mels,
            normalized=True,
        ).to(device)
        
        # Optimizers
        self.optim_g = AdamW(generator.parameters(), lr=2e-4, betas=(0.8, 0.99))
        self.optim_d = AdamW(
            list(mpd.parameters()) + list(msd.parameters()),
            lr=2e-4, betas=(0.8, 0.99)
        )
        
        # Schedulers
        self.scheduler_g = ExponentialLR(self.optim_g, gamma=0.999)
        self.scheduler_d = ExponentialLR(self.optim_d, gamma=0.999)
        
        # Loss weights
        self.lambda_fm = 2.0       # Feature matching
        self.lambda_mel = 45.0     # Mel reconstruction
        self.lambda_adv = 1.0      # Adversarial
    
    def train_step(self, audio: torch.Tensor) -> Dict[str, float]:
        """
        Pojedynczy krok treningowy.
        
        Args:
            audio: Waveform [B, T] lub [B, 1, T]
            
        Returns:
            Dict ze stratami
        """
        if audio.dim() == 2:
            audio = audio.unsqueeze(1)  # [B, 1, T]
        
        audio = audio.to(self.device)
        
        # 1. Przepuść audio przez VAE (dostajemy mel)
        with torch.no_grad():
            vae_out = self.vae(audio)
            mel_vae = vae_out['mel']  # [B, 1, n_mels, T]
            mel_recon = vae_out['mel_recon']  # [B, 1, n_mels, T]
        
        # Przygotuj mel dla generatora [B, n_mels, T]
        mel_input = mel_recon.squeeze(1)  # [B, n_mels, T]
        
        # 2. Generator: mel → audio
        audio_gen = self.generator(mel_input)  # [B, 1, T]
        
        # Dopasuj długości (audio_gen może być innej długości)
        min_len = min(audio.shape[-1], audio_gen.shape[-1])
        audio = audio[..., :min_len]
        audio_gen = audio_gen[..., :min_len]
        
        # ==================== DISCRIMINATOR ====================
        self.optim_d.zero_grad()
        
        # MPD
        y_df_r, y_df_g, fmap_f_r, fmap_f_g = self.mpd(audio, audio_gen.detach())
        loss_disc_f, _, _ = discriminator_loss(y_df_r, y_df_g)
        
        # MSD
        y_ds_r, y_ds_g, fmap_s_r, fmap_s_g = self.msd(audio, audio_gen.detach())
        loss_disc_s, _, _ = discriminator_loss(y_ds_r, y_ds_g)
        
        loss_disc = loss_disc_f + loss_disc_s
        loss_disc.backward()
        self.optim_d.step()
        
        # ==================== GENERATOR ====================
        self.optim_g.zero_grad()
        
        # Recompute discriminator outputs dla generatora
        y_df_r, y_df_g, fmap_f_r, fmap_f_g = self.mpd(audio, audio_gen)
        y_ds_r, y_ds_g, fmap_s_r, fmap_s_g = self.msd(audio, audio_gen)
        
        # Adversarial loss
        loss_gen_f = generator_loss(y_df_g)
        loss_gen_s = generator_loss(y_ds_g)
        loss_gen = loss_gen_f + loss_gen_s
        
        # Feature matching loss
        loss_fm_f = feature_loss(fmap_f_r, fmap_f_g)
        loss_fm_s = feature_loss(fmap_s_r, fmap_s_g)
        loss_fm = loss_fm_f + loss_fm_s
        
        # Mel loss (generated audio → mel powinno być podobne do input mel)
        mel_gen = self.mel_transform(audio_gen.squeeze(1))
        if mel_gen.shape[-1] != mel_input.shape[-1]:
            min_t = min(mel_gen.shape[-1], mel_input.shape[-1])
            mel_gen = mel_gen[..., :min_t]
            mel_input_crop = mel_input[..., :min_t]
        else:
            mel_input_crop = mel_input
        loss_mel = F.l1_loss(mel_gen, mel_input_crop)
        
        # Total generator loss
        loss_g = (
            self.lambda_adv * loss_gen +
            self.lambda_fm * loss_fm +
            self.lambda_mel * loss_mel
        )
        
        loss_g.backward()
        self.optim_g.step()
        
        return {
            'loss_disc': loss_disc.item(),
            'loss_gen': loss_gen.item(),
            'loss_fm': loss_fm.item(),
            'loss_mel': loss_mel.item(),
            'loss_total': loss_g.item(),
        }
    
    def step_schedulers(self):
        """Aktualizuje learning rate"""
        self.scheduler_g.step()
        self.scheduler_d.step()


def train_vocoder(
    vae_checkpoint: str,
    annotations_json: str,
    audio_dir: str,
    checkpoint_dir: str = './checkpoints_v2',
    epochs: int = 100,
    batch_size: int = 16,
    sample_rate: int = 32000,
    segment_duration: float = 1.0,  # Krótsze segmenty dla vocoder
    device: str = 'cuda',
    save_every: int = 10,
    num_workers: int = 4,
):
    """
    Główna funkcja treningu vocodera.
    
    Args:
        vae_checkpoint: Ścieżka do wytrenowanego VAE
        annotations_json: Plik z datasetek
        audio_dir: Katalog z audio
        checkpoint_dir: Gdzie zapisywać
        epochs: Liczba epok
        batch_size: Rozmiar batcha
        sample_rate: Sample rate (powinien być taki sam jak VAE)
        segment_duration: Długość segmentu w sekundach
        device: Device
        save_every: Co ile epok zapisywać
    """
    from models.audio_vae import AudioVAE
    from models.vocoder import HiFiGANGenerator, MultiPeriodDiscriminator, MultiScaleDiscriminator
    from models_v2.segmented_dataset import SegmentedMusicDataset, collate_segmented
    
    print("="*60)
    print("🎵 Training HiFi-GAN Vocoder (VAE Fine-tuning)")
    print("="*60)
    
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(exist_ok=True)
    
    # Load VAE
    print(f"\n📂 Loading VAE from {vae_checkpoint}...")
    vae_ckpt = torch.load(vae_checkpoint, map_location=device)
    vae_config = vae_ckpt.get('config', {})
    
    vae = AudioVAE(
        sample_rate=vae_config.get('sample_rate', sample_rate),
        latent_dim=vae_config.get('latent_dim', 128),
        use_stft_loss=False,
    ).to(device)
    vae.load_state_dict(vae_ckpt['model_state_dict'])
    vae.eval()
    print(f"   VAE loaded (latent_dim={vae.latent_dim})")
    
    # Create vocoder components
    print("\n🏗️  Creating HiFi-GAN...")
    generator = HiFiGANGenerator.from_sample_rate(
        sample_rate=sample_rate,
        n_mels=vae.n_mels,
    )
    mpd = MultiPeriodDiscriminator()
    msd = MultiScaleDiscriminator()
    
    g_params = sum(p.numel() for p in generator.parameters())
    d_params = sum(p.numel() for p in mpd.parameters()) + sum(p.numel() for p in msd.parameters())
    print(f"   Generator: {g_params:,} params")
    print(f"   Discriminators: {d_params:,} params")
    
    # Dataset
    print(f"\n📂 Loading dataset...")
    dataset = SegmentedMusicDataset(
        annotations_json=annotations_json,
        audio_dir=audio_dir,
        sample_rate=sample_rate,
        segment_duration=segment_duration,  # Krótsze dla vocoder
        include_context=False,
    )
    
    def vocoder_collate(batch):
        """Collate tylko dla audio"""
        return {
            'audio': torch.stack([b['audio'] for b in batch]),
        }
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=vocoder_collate,
        pin_memory=True,
    )
    
    print(f"   Samples: {len(dataset)}")
    print(f"   Batches: {len(dataloader)}")
    
    # Trainer
    trainer = VocoderTrainer(
        generator=generator,
        mpd=mpd,
        msd=msd,
        vae=vae,
        sample_rate=sample_rate,
        n_mels=vae.n_mels,
        hop_length=vae.hop_length,
        device=device,
    )
    
    # Training loop
    print(f"\n🚀 Starting training for {epochs} epochs...")
    
    best_mel_loss = float('inf')
    
    for epoch in range(epochs):
        generator.train()
        mpd.train()
        msd.train()
        
        total_disc = 0
        total_gen = 0
        total_fm = 0
        total_mel = 0
        
        pbar = tqdm(dataloader, desc=f"Vocoder Epoch {epoch+1}/{epochs}")
        
        for batch in pbar:
            audio = batch['audio']
            
            losses = trainer.train_step(audio)
            
            total_disc += losses['loss_disc']
            total_gen += losses['loss_gen']
            total_fm += losses['loss_fm']
            total_mel += losses['loss_mel']
            
            pbar.set_postfix({
                'mel': f"{losses['loss_mel']:.4f}",
                'gen': f"{losses['loss_gen']:.4f}",
            })
        
        trainer.step_schedulers()
        
        # Epoch stats
        n = len(dataloader)
        avg_disc = total_disc / n
        avg_gen = total_gen / n
        avg_fm = total_fm / n
        avg_mel = total_mel / n
        
        print(f"\nEpoch {epoch+1} - D: {avg_disc:.4f}, G: {avg_gen:.4f}, FM: {avg_fm:.4f}, Mel: {avg_mel:.4f}")
        
        # Save best
        if avg_mel < best_mel_loss:
            best_mel_loss = avg_mel
            torch.save({
                'epoch': epoch,
                'generator_state_dict': generator.state_dict(),
                'mpd_state_dict': mpd.state_dict(),
                'msd_state_dict': msd.state_dict(),
                'optim_g_state_dict': trainer.optim_g.state_dict(),
                'optim_d_state_dict': trainer.optim_d.state_dict(),
                'mel_loss': avg_mel,
                'sample_rate': sample_rate,
            }, checkpoint_dir / 'vocoder_best.pt')
            print(f"   ✅ Saved best vocoder (mel_loss: {best_mel_loss:.4f})")
        
        # Regular checkpoint
        if (epoch + 1) % save_every == 0:
            torch.save({
                'epoch': epoch,
                'generator_state_dict': generator.state_dict(),
                'mel_loss': avg_mel,
            }, checkpoint_dir / f'vocoder_epoch_{epoch+1}.pt')
    
    print(f"\n✅ Vocoder training complete! Best mel loss: {best_mel_loss:.4f}")
    
    return generator


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train HiFi-GAN Vocoder')
    parser.add_argument('--vae_checkpoint', type=str, required=True,
                        help='Path to VAE checkpoint')
    parser.add_argument('--annotations', type=str, default='./data_v2/dataset.json',
                        help='Dataset annotations JSON')
    parser.add_argument('--audio_dir', type=str, default='./music/own',
                        help='Audio directory')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints_v2',
                        help='Checkpoint directory')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--sample_rate', type=int, default=32000)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--save_every', type=int, default=10)
    
    args = parser.parse_args()
    
    train_vocoder(
        vae_checkpoint=args.vae_checkpoint,
        annotations_json=args.annotations,
        audio_dir=args.audio_dir,
        checkpoint_dir=args.checkpoint_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        sample_rate=args.sample_rate,
        device=args.device,
        save_every=args.save_every,
    )
