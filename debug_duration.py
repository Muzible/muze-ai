#!/usr/bin/env python3
"""Debug script for duration issue"""
import sys
sys.path.insert(0, '.')
import torch
from models.audio_vae import AudioVAE
from models.vocoder import HiFiGAN

device = torch.device('cpu')

# Load VAE checkpoint
print("Loading VAE checkpoint...")
vae_ckpt = torch.load('./checkpoints_v2/vae_best.pt', map_location=device)
vae_config = vae_ckpt.get('config', {})
vae_sr = vae_config.get('sample_rate', 32000)
vae_hop = vae_config.get('hop_length', 320)
latent_dim = vae_config.get('latent_dim', 128)

print(f'VAE config: sr={vae_sr}, hop={vae_hop}, latent_dim={latent_dim}')

# Calculate latent_time for 10s
duration = 10.0
latent_time = int(duration * vae_sr / vae_hop / 8)
print(f'latent_time for {duration}s: {latent_time}')

# Create latent
latent = torch.randn(1, latent_dim, 16, latent_time)
print(f'Latent shape: {latent.shape}')

# VAE decode
print("Creating VAE...")
vae = AudioVAE(latent_dim=latent_dim, sample_rate=vae_sr, hop_length=vae_hop)
vae.load_state_dict(vae_ckpt.get('model_state_dict', vae_ckpt))
vae.eval()

print("Decoding latent to mel...")
with torch.no_grad():
    mel = vae.decode(latent)
print(f'Decoded mel shape: {mel.shape}')

if mel.dim() == 4:
    mel = mel.squeeze(1)

# Vocoder
print("Creating vocoder...")
vocoder = HiFiGAN()
print("Vocoding mel to audio...")
with torch.no_grad():
    audio = vocoder.inference(mel.float())
print(f'Audio shape: {audio.shape}')
print(f'Duration at {vae_sr}Hz: {audio.shape[-1] / vae_sr:.2f}s')
print(f'Expected: 10.00s')
