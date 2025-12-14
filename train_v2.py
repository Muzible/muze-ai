"""
🎵 Train v2 - Kompletny system treningu Muze AI

Pipeline treningu:
1. Faza 0: Budowanie datasetu (build_dataset_v2.py) - segmenty, features, vocals, prompts
2. Faza 1: Trening Audio VAE (kompresja audio do latent space)
3. Faza 2: Trening Composition Planner
4. Faza 3: Trening Latent Diffusion v2 (section-aware)

Użycie:
    # Budowanie datasetu (z wokalami, lyrics, LLM prompts)
    python build_dataset_v2.py --audio_dir ./music/own --output ./data_v2/dataset.json
    
    # Trening VAE
    python train_v2.py --phase 1 --annotations ./data_v2/dataset.json --epochs 100
    
    # Trening Composition Planner
    python train_v2.py --phase 2 --annotations ./data_v2/dataset.json --epochs 100
    
    # Trening LDM v2
    python train_v2.py --phase 3 --annotations ./data_v2/dataset.json --vae_checkpoint ./checkpoints_v2/vae_best.pt
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from tqdm import tqdm


def train_vae(args):
    """Trening Audio VAE (Faza 1)
    
    v2 Updates:
    - latent_dim: 128 (zwiększone z 8)
    - sample_rate: 32000 (zwiększone z 22050)
    - Multi-Resolution STFT Loss
    """
    from models.audio_vae import AudioVAE
    from models_v2.segmented_dataset import SegmentedMusicDataset
    
    print("="*60)
    print("🎵 Training Audio VAE v2 (Phase 1)")
    print("="*60)
    
    device = args.device
    
    # v2: Nowe parametry
    sample_rate = getattr(args, 'sample_rate', 32000)
    latent_dim = getattr(args, 'latent_dim', 128)
    use_stft_loss = getattr(args, 'use_stft_loss', True)
    
    print(f"\n📊 v2 Config:")
    print(f"   Sample rate: {sample_rate} Hz")
    print(f"   Latent dim: {latent_dim}")
    print(f"   STFT loss: {use_stft_loss}")
    
    # Dataset - używamy SegmentedMusicDataset ale tylko audio
    print(f"\n📂 Loading dataset from {args.annotations}...")
    dataset = SegmentedMusicDataset(
        annotations_json=args.annotations,
        audio_dir=args.audio_dir,
        sample_rate=sample_rate,  # v2: 32kHz
        segment_duration=10.0,
        include_context=False,  # VAE nie potrzebuje kontekstu
        max_tracks=args.max_tracks,
    )
    
    # Split train/val
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )
    
    def collate_fn(batch):
        """Collate dla VAE - tylko audio"""
        return {
            'audio': torch.stack([b['audio'] for b in batch]),
        }
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )
    
    print(f"   Dataset size: {len(dataset)} segments")
    print(f"   Train: {len(train_dataset)}, Val: {len(val_dataset)}")
    print(f"   Batches per epoch: {len(train_loader)}")
    
    # Model - v2: Nowe parametry
    print("\n🏗️  Creating VAE model...")
    model = AudioVAE(
        sample_rate=sample_rate,  # v2: 32kHz
        latent_dim=latent_dim,    # v2: 128
        use_stft_loss=use_stft_loss,  # v2: Multi-Resolution STFT Loss
    ).to(device)
    
    params = sum(p.numel() for p in model.parameters())
    print(f"   Parameters: {params:,} ({params/1e6:.1f}M)")
    
    # Optimizer
    optimizer = AdamW(model.parameters(), lr=args.lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    # Checkpoint directory
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(exist_ok=True)
    
    # Training
    print(f"\n🚀 Starting training for {args.epochs} epochs...")
    print(f"   Device: {device}")
    print(f"   Batch size: {args.batch_size}")
    print(f"   Learning rate: {args.lr}")
    
    best_loss = float('inf')
    patience_counter = 0
    beta = 0.001  # KL weight
    stft_weight = 0.5  # v2: STFT loss weight
    
    for epoch in range(1, args.epochs + 1):
        # Train
        model.train()
        train_loss = 0
        train_recon = 0
        train_kl = 0
        train_stft = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}")
        for batch in pbar:
            audio = batch['audio'].to(device)
            
            optimizer.zero_grad()
            output = model(audio)
            
            # v2: Loss z STFT
            losses = model.loss(
                output, 
                beta=beta, 
                audio_input=audio if use_stft_loss else None,
                stft_weight=stft_weight,
            )
            
            losses['total'].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            train_loss += losses['total'].item()
            train_recon += losses['recon'].item()
            train_kl += losses['kl'].item()
            if 'stft_total' in losses:
                train_stft += losses['stft_total'].item()
            
            pbar.set_postfix({
                'loss': f"{losses['total'].item():.4f}",
                'recon': f"{losses['recon'].item():.4f}",
            })
        
        train_loss /= len(train_loader)
        train_recon /= len(train_loader)
        train_kl /= len(train_loader)
        train_stft /= len(train_loader)
        
        # Validate
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                audio = batch['audio'].to(device)
                output = model(audio)
                losses = model.loss(output, beta=beta)
                val_loss += losses['total'].item()
        val_loss /= len(val_loader)
        
        scheduler.step()
        
        # v2: Rozszerzone logowanie
        log_msg = f"\nEpoch {epoch} - Train: {train_loss:.4f} (recon: {train_recon:.4f}, kl: {train_kl:.4f}"
        if use_stft_loss:
            log_msg += f", stft: {train_stft:.4f}"
        log_msg += f") | Val: {val_loss:.4f}"
        print(log_msg)
        
        # Save best
        if val_loss < best_loss:
            best_loss = val_loss
            patience_counter = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': val_loss,
                # v2: Zapisz konfigurację
                'config': {
                    'sample_rate': sample_rate,
                    'latent_dim': latent_dim,
                    'use_stft_loss': use_stft_loss,
                },
            }, checkpoint_dir / 'vae_best.pt')
            print(f"   ✅ Saved best model (val_loss: {best_loss:.4f})")
        else:
            patience_counter += 1
            print(f"   ⚠️  No improvement ({patience_counter}/{args.patience})")
        
        # Early stopping
        if args.patience and patience_counter >= args.patience:
            print(f"\n🛑 Early stopping at epoch {epoch}")
            break
        
        # Regular checkpoint
        if epoch % args.save_every == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': val_loss,
            }, checkpoint_dir / f'vae_epoch_{epoch}.pt')
    
    print(f"\n✅ VAE Training complete! Best val_loss: {best_loss:.4f}")
    print(f"   Checkpoint: {checkpoint_dir / 'vae_best.pt'}")


def train_composition_planner(args):
    """Trening Composition Planner (Faza 2)
    
    v2 Updates:
    - Używa prawdziwych embeddingów tekstu z CLAP/T5 zamiast dummy
    - Genre i mood indices z datasetu
    """
    from models_v2.composition_planner import (
        CompositionTransformer, 
        compute_loss,
    )
    from models_v2.segmented_dataset import CompositionDataset
    from models_v2.text_encoder import EnhancedMusicEncoder
    
    print("="*60)
    print("🎵 Training Composition Planner (Phase 2)")
    print("="*60)
    
    device = args.device
    
    # Dataset
    print(f"\n📂 Loading dataset from {args.annotations}...")
    dataset = CompositionDataset(
        annotations_json=args.annotations,
        max_sections=20,
        max_tracks=args.max_tracks,
    )
    
    def collate_composition(batch):
        """Custom collate dla composition dataset"""
        return {
            'sections': torch.stack([b['sections'] for b in batch]),
            'attrs': torch.stack([b['attrs'] for b in batch]),
            'keys': torch.stack([b['keys'] for b in batch]),
            'vocals': torch.stack([b['vocals'] for b in batch]),
            'duration': torch.stack([b['duration'] for b in batch]),
            'prompts': [b['prompt'] for b in batch],  # v2: Lista promptów
            'genre_idx': torch.tensor([b['genre_idx'] for b in batch], dtype=torch.long),
            'mood_idx': torch.tensor([b['mood_idx'] for b in batch], dtype=torch.long),
        }
    
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_composition,  # v2: Custom collate
    )
    
    print(f"   Dataset size: {len(dataset)}")
    print(f"   Batches per epoch: {len(dataloader)}")
    
    # v2: Text Encoder dla prawdziwych embeddingów
    print("\n🏗️  Creating text encoder...")
    text_encoder = EnhancedMusicEncoder(
        use_clap=args.use_clap,
        use_t5_fallback=True,
        device=device,
    )
    
    # Model
    print("🏗️  Creating Composition Planner...")
    model = CompositionTransformer(
        vocab_size=15,
        d_model=args.d_model,
        nhead=4,
        num_encoder_layers=3,
        num_decoder_layers=4,
        text_embed_dim=768,
        num_genres=len(CompositionDataset.GENRES),  # v2: Właściwa liczba genres
        num_moods=len(CompositionDataset.MOODS),    # v2: Właściwa liczba moods
    ).to(device)
    
    params = sum(p.numel() for p in model.parameters())
    print(f"   Parameters: {params:,} ({params/1e6:.1f}M)")
    
    # Optimizer
    optimizer = AdamW(model.parameters(), lr=args.lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    # Checkpoint directory
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(exist_ok=True)
    
    # Training loop
    print(f"\n🚀 Starting training for {args.epochs} epochs...")
    
    best_loss = float('inf')
    
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        total_section = 0
        total_attr = 0
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{args.epochs}")
        
        for batch in pbar:
            # Move to device
            sections = batch['sections'].to(device)
            attrs = batch['attrs'].to(device)
            keys = batch['keys'].to(device)
            vocals = batch['vocals'].to(device)
            duration = batch['duration'].to(device)
            
            # v2: Prawdziwe embeddingi tekstu z CLAP/T5
            prompts = batch['prompts']
            B = sections.shape[0]
            
            # Encode prompts
            with torch.no_grad():  # Text encoder nie jest trenowany tutaj
                text_embed = text_encoder.encode_text_only(prompts)
            text_embed = text_embed.to(device)
            
            # Rozszerz do sekwencji (composition planner oczekuje [B, seq, dim])
            if text_embed.dim() == 2:
                text_embed = text_embed.unsqueeze(1).expand(-1, 10, -1)
            
            # v2: Genre i mood indices z datasetu (nie dummy!)
            genre_idx = batch['genre_idx'].to(device)
            mood_idx = batch['mood_idx'].to(device)
            
            # Forward
            optimizer.zero_grad()
            
            outputs = model(
                text_embed=text_embed,
                target_duration=duration,
                genre_idx=genre_idx,
                mood_idx=mood_idx,
                tgt_sections=sections[:, :-1],  # Teacher forcing - bez EOS
                tgt_attrs=attrs[:, :-1],
                tgt_keys=keys[:, :-1],
            )
            
            # Loss
            targets = {
                'sections': sections[:, 1:],  # Target = shifted by 1
                'attrs': attrs[:, 1:, :3],    # Tylko tempo, energy, duration
                'keys': keys[:, 1:],
                'vocals': vocals[:, 1:],
            }
            
            losses = compute_loss(outputs, targets)
            
            # Backward
            losses['total'].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            total_loss += losses['total'].item()
            total_section += losses['section'].item()
            total_attr += losses['attr'].item()
            
            pbar.set_postfix({
                'loss': losses['total'].item(),
                'section': losses['section'].item(),
            })
        
        scheduler.step()
        
        # Epoch stats
        n_batches = len(dataloader)
        avg_loss = total_loss / n_batches
        avg_section = total_section / n_batches
        avg_attr = total_attr / n_batches
        
        print(f"\nEpoch {epoch+1} - Loss: {avg_loss:.4f} (section: {avg_section:.4f}, attr: {avg_attr:.4f})")
        
        # Save checkpoint
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
                'model_config': {
                    'd_model': args.d_model,
                },
            }, checkpoint_dir / 'composition_planner_best.pt')
            print(f"   ✅ Saved best model (loss: {best_loss:.4f})")
        
        # Regular checkpoint
        if (epoch + 1) % args.save_every == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
            }, checkpoint_dir / f'composition_planner_epoch_{epoch+1}.pt')
    
    print(f"\n✅ Training complete! Best loss: {best_loss:.4f}")


def train_ldm_v2(args):
    """Trening Latent Diffusion v2 (Faza 3)
    
    v2 Updates:
    - latent_dim: 128 (zwiększone z 8)
    - num_timesteps: 200 (zmniejszone z 1000)
    - sample_rate: 32000 (zwiększone z 22050)
    - LoRA dla text encoder (opcjonalnie)
    - End-to-end training z vocoderem (opcjonalnie)
    """
    from models_v2.latent_diffusion import UNetV2, LatentDiffusionV2
    from models_v2.text_encoder import EnhancedMusicEncoder
    from models_v2.segmented_dataset import SegmentedMusicDataset, collate_segmented
    
    # Import VAE from original
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from models.audio_vae import AudioVAE
    
    print("="*60)
    print("🎵 Training Latent Diffusion v2 (Phase 3)")
    print("="*60)
    
    device = args.device
    
    # v2: Nowe parametry
    sample_rate = getattr(args, 'sample_rate', 32000)
    latent_dim = getattr(args, 'latent_dim', 128)
    num_timesteps = getattr(args, 'num_timesteps', 200)
    train_text_encoder = getattr(args, 'train_text_encoder', False)
    
    # Auto-adjust patience based on model size
    model_channels = getattr(args, 'model_channels', 256)
    if args.patience == 10:  # Default value, auto-adjust
        if model_channels >= 768:
            recommended_patience = 25
            print(f"   📊 Auto-adjusted patience: {args.patience} → {recommended_patience} (XXL model)")
            args.patience = recommended_patience
        elif model_channels >= 512:
            recommended_patience = 20
            print(f"   📊 Auto-adjusted patience: {args.patience} → {recommended_patience} (Large model)")
            args.patience = recommended_patience
        elif model_channels >= 320:
            recommended_patience = 15
            print(f"   📊 Auto-adjusted patience: {args.patience} → {recommended_patience} (Default model)")
            args.patience = recommended_patience
    
    print(f"\n📊 v2 Config:")
    print(f"   Sample rate: {sample_rate} Hz")
    print(f"   Latent dim: {latent_dim}")
    print(f"   Diffusion steps: {num_timesteps}")
    print(f"   Train text encoder: {train_text_encoder}")
    
    # Load VAE
    print(f"\n📂 Loading VAE from {args.vae_checkpoint}...")
    vae = AudioVAE(
        sample_rate=sample_rate,
        latent_dim=latent_dim,
        use_stft_loss=False,  # Nie potrzebne przy inferencji
    ).to(device)
    
    vae_ckpt = torch.load(args.vae_checkpoint, map_location=device)
    vae.load_state_dict(vae_ckpt['model_state_dict'])
    vae.eval()
    for param in vae.parameters():
        param.requires_grad = False
    
    # Sprawdź czy VAE ma poprawny latent_dim
    vae_latent_dim = vae_ckpt.get('config', {}).get('latent_dim', 8)
    if vae_latent_dim != latent_dim:
        print(f"   ⚠️ Warning: VAE latent_dim={vae_latent_dim}, expected {latent_dim}")
        print(f"   Retrain VAE with --latent_dim {latent_dim}")
    print(f"   VAE loaded and frozen (latent_dim={vae_latent_dim})")
    
    # Dataset
    print(f"\n📂 Loading dataset...")
    dataset = SegmentedMusicDataset(
        annotations_json=args.annotations,
        audio_dir=args.audio_dir,
        sample_rate=sample_rate,  # v2: 32kHz
        segment_duration=10.0,
        include_context=True,
        max_tracks=args.max_tracks,
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_segmented,
    )
    
    print(f"   Dataset size: {len(dataset)}")
    
    # Text encoder - v2: z LoRA
    print("\n🏗️  Creating text encoder...")
    text_encoder = EnhancedMusicEncoder(
        use_clap=args.use_clap,
        use_t5_fallback=True,
        device=device,
    )
    
    # U-Net v2 - v2: 128 channels
    print("🏗️  Creating U-Net v2...")
    unet = UNetV2(
        in_channels=latent_dim,       # v2: 128
        out_channels=latent_dim,      # v2: 128
        model_channels=args.model_channels,
        channel_mult=[1, 2, 4, 4],
        context_dim=768,
        use_context_fusion=True,
    ).to(device)
    
    # LDM v2 - v2: 200 timesteps
    ldm = LatentDiffusionV2(
        unet, 
        num_timesteps=num_timesteps,  # v2: 200
        beta_schedule="scaled_linear",
    ).to(device)
    
    params = sum(p.numel() for p in ldm.parameters())
    print(f"   LDM Parameters: {params:,} ({params/1e6:.1f}M)")
    
    # Optimizer - v2: opcjonalnie trenuj text encoder
    if train_text_encoder and hasattr(text_encoder, 'clap') and text_encoder.clap is not None:
        # Pobierz parametry LoRA z CLAP
        text_encoder_params = text_encoder.clap.get_trainable_parameters()
        all_params = list(ldm.parameters()) + text_encoder_params
        print(f"   Text encoder trainable params: {len(text_encoder_params)}")
    else:
        all_params = ldm.parameters()
    
    optimizer = AdamW(all_params, lr=args.lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    # Checkpoint directory
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(exist_ok=True)
    
    # Training loop
    print(f"\n🚀 Starting training for {args.epochs} epochs...")
    
    best_loss = float('inf')
    
    for epoch in range(args.epochs):
        ldm.train()
        total_loss = 0
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{args.epochs}")
        
        for batch in pbar:
            # Move to device
            audio = batch['audio'].to(device)
            prompts = batch['prompt']
            section_types = batch['section_type']
            positions = batch['position'].to(device)
            tempos = batch['tempo'].to(device)
            energies = batch['energy'].to(device)
            
            # 🔊 v3: Loudness conditioning (dB)
            loudness = batch.get('loudness')
            if loudness is not None:
                loudness = loudness.to(device)
            
            # 🎤 v3: Has vocals flag (for vocal/instrumental distinction)
            has_vocals = batch.get('has_vocals')
            if has_vocals is not None:
                has_vocals = has_vocals.to(device)
            
            # 😊 v3: Sentiment score (-1 to 1)
            sentiment_score = batch.get('sentiment_score')
            if sentiment_score is not None:
                sentiment_score = sentiment_score.to(device)
            
            # 🎨 v3: Artist and genre conditioning
            artists = batch.get('artist', None)  # List of artist strings
            genres = batch.get('genres', None)   # List of genre lists
            
            # 🎵 Key conditioning (string → index)
            # Dataset has key as single note (e.g. "C", "F#", "Bb") without major/minor info
            # Model has 24 key embeddings but we only use first 12 (pitch class)
            keys = batch.get('key', None)  # List of key strings like "C", "F", "A#"
            key_idx = None
            if keys is not None:
                KEY_TO_IDX = {
                    # Pitch classes (0-11) - dataset only provides note, no major/minor
                    'C': 0, 'C#': 1, 'Db': 1, 'D': 2, 'D#': 3,
                    'Eb': 3, 'E': 4, 'F': 5, 'F#': 6, 'Gb': 6,
                    'G': 7, 'G#': 8, 'Ab': 8, 'A': 9, 'A#': 10,
                    'Bb': 10, 'B': 11,
                    # Also support full format if future dataset has it
                    'C major': 0, 'C# major': 1, 'D major': 2, 'D# major': 3,
                    'E major': 4, 'F major': 5, 'F# major': 6, 'G major': 7,
                    'G# major': 8, 'A major': 9, 'A# major': 10, 'B major': 11,
                    'C minor': 12, 'C# minor': 13, 'D minor': 14, 'D# minor': 15,
                    'E minor': 16, 'F minor': 17, 'F# minor': 18, 'G minor': 19,
                    'G# minor': 20, 'A minor': 21, 'A# minor': 22, 'B minor': 23,
                }
                key_indices = [KEY_TO_IDX.get(k, 0) for k in keys]  # Default to C
                key_idx = torch.tensor(key_indices, device=device)
            
            # 🎤 Voice embeddings (dla voice cloning/conditioning)
            # v3: Obsługa obu typów embeddings
            voice_emb = batch.get('voice_embedding')
            if voice_emb is not None:
                voice_emb = voice_emb.to(device)
            
            # 🎤 v3: ECAPA-TDNN embedding z separowanych wokali
            voice_emb_separated = batch.get('voice_embedding_separated')
            if voice_emb_separated is not None:
                voice_emb_separated = voice_emb_separated.to(device)
            
            # 🔊 CLAP embeddings (pre-computed) - możemy użyć zamiast text_encoder
            clap_audio_emb = batch.get('clap_audio_embedding')
            clap_text_emb = batch.get('clap_text_embedding')
            if clap_audio_emb is not None:
                clap_audio_emb = clap_audio_emb.to(device)
            if clap_text_emb is not None:
                clap_text_emb = clap_text_emb.to(device)
            use_precomputed_clap = (clap_text_emb is not None and 
                                    clap_text_emb.shape[-1] == 512 and 
                                    clap_text_emb.abs().sum() > 0)
            
            # 📝 Lyrics per segment (może być użyte do kondycjonowania)
            lyrics_texts = batch.get('lyrics_text', [])
            
            # 🎵 Beat info (może być użyte do sync)
            num_beats = batch.get('num_beats')
            if num_beats is not None:
                num_beats = num_beats.to(device)
            beat_positions = batch.get('beat_positions', None)  # List of lists
            
            # 🎸 Chord info
            current_chord = batch.get('current_chord', None)  # List of chord strings
            
            # ⏱️ Time signature
            time_signature = batch.get('time_signature', None)  # List of "4/4" etc
            
            # 🎤 Phonemes for singing synthesis
            phonemes_ipa = batch.get('phonemes_ipa', None)  # List of IPA strings
            
            # 🔤 v3: Phoneme timestamps per segment
            phoneme_timestamps = batch.get('phoneme_timestamps', None)  # List of dicts
            
            # 🎤 v3: Vibrato analysis per segment
            vibrato_rate = batch.get('vibrato_rate', None)    # List of floats/None [Hz]
            vibrato_depth = batch.get('vibrato_depth', None)  # List of floats/None [semitones]
            vibrato_extent = batch.get('vibrato_extent', None)  # List of floats/None [semitones]
            
            # 💨 v3: Breath detection per segment
            breath_positions = batch.get('breath_positions', None)  # List of lists [seconds]
            
            # 🎵 v3: F0/Pitch conditioning
            f0 = batch.get('f0')  # [B, T] continuous F0 in Hz (from dataset)
            if f0 is not None:
                f0 = f0.to(device)
            f0_coarse = batch.get('f0_coarse')  # [B, T] discrete pitch bins
            if f0_coarse is not None:
                f0_coarse = f0_coarse.to(device)
            f0_voiced_mask = batch.get('f0_voiced_mask')  # [B, T] boolean mask
            if f0_voiced_mask is not None:
                f0_voiced_mask = f0_voiced_mask.to(device)
            
            # Context audio (previous segment)
            context_audio = batch.get('context_audio')
            if context_audio is not None:
                context_audio = context_audio.to(device)
            
            # Encode audio to latent
            with torch.no_grad():
                vae_output = vae(audio)
                z = vae_output['z']
                
                # Context latent
                if context_audio is not None and context_audio.shape[-1] > 0:
                    context_vae = vae(context_audio)
                    context_z = context_vae['z']
                else:
                    context_z = None
            
            # Text embedding - użyj pre-computed CLAP lub encode runtime
            if use_precomputed_clap and not train_text_encoder:
                # Użyj pre-computed CLAP text embedding (512 dim)
                # Musimy dopasować do context_dim=768
                clap_text_emb = clap_text_emb.to(device)
                # Projection 512 → 768 (lub użyj bezpośrednio jeśli model to obsługuje)
                # Na razie używamy text_encoder jako fallback
                with torch.no_grad():
                    text_embed = text_encoder(
                        prompts,
                        section_types,
                        positions,
                        tempos,
                        energies,
                    )
            elif train_text_encoder:
                text_embed = text_encoder(
                    prompts,
                    section_types,
                    positions,
                    tempos,
                    energies,
                )
            else:
                with torch.no_grad():
                    text_embed = text_encoder(
                        prompts,
                        section_types,
                        positions,
                        tempos,
                        energies,
                    )
            
            # Random timesteps
            B = z.shape[0]
            t = torch.randint(0, ldm.num_timesteps, (B,), device=device)
            
            # Forward
            optimizer.zero_grad()
            
            # 🎤 Przekaż voice_embedding do modelu!
            # 🎵 v2: Przekaż wszystkie nowe kondycjonowania (CLAP, beat, chord, phonemes)
            # 🎤 v3: Oba voice embeddings (resemblyzer + ECAPA-TDNN)
            # 🎵 v3: F0/pitch conditioning
            # 🎵 v3: Key, loudness, has_vocals, sentiment, genre, artist conditioning
            loss, phoneme_durations = ldm.p_losses(
                z, t, text_embed,
                section_type=section_types,
                position=positions,
                energy=energies,
                tempo=tempos,
                key_idx=key_idx,                            # ✅ Key conditioning (0-23)
                loudness=loudness,                          # ✅ v3: Loudness in dB
                has_vocals=has_vocals,                      # ✅ v3: Has vocals flag
                sentiment_score=sentiment_score,            # ✅ v3: Sentiment (-1 to 1)
                genres=genres,                              # ✅ v3: Genre list (multi-hot)
                artist=artists,                             # ✅ v3: Artist name (hash-based)
                voice_emb=voice_emb,                        # ✅ Voice conditioning (256-dim)
                voice_emb_separated=voice_emb_separated,    # ✅ v3: ECAPA-TDNN embedding (192-dim)
                context_latent=context_z,
                # v2: New conditioning
                clap_audio_embedding=clap_audio_emb,        # ✅ CLAP audio embedding [B, 512]
                clap_text_embedding=clap_text_emb,          # ✅ CLAP text embedding [B, 512]
                num_beats=num_beats,                        # ✅ Beat count [B]
                beat_positions=beat_positions,              # ✅ Beat positions (list of lists)
                time_signature=time_signature,              # ✅ Time signature (list of "4/4" etc)
                current_chord=current_chord,                # ✅ Chord (list of chord strings)
                phonemes_ipa=phonemes_ipa,                  # ✅ IPA phonemes (list of strings)
                # v3: Pitch conditioning
                f0=f0,                                      # ✅ Continuous F0 in Hz [B, T]
                f0_coarse=f0_coarse,                        # ✅ Discrete pitch bins [B, T]
                f0_voiced_mask=f0_voiced_mask,              # ✅ Ground-truth voiced mask [B, T]
                # v3.1: Vibrato, breath, phoneme timestamps
                vibrato_rate=vibrato_rate,                  # ✅ v3.1: Vibrato frequency Hz
                vibrato_depth=vibrato_depth,                # ✅ v3.1: Vibrato depth cents
                vibrato_extent=vibrato_extent,              # ✅ v3.1: Vibrato extent (fraction)
                breath_positions=breath_positions,          # ✅ v3.1: Breath positions (list)
                phoneme_timestamps=phoneme_timestamps,      # ✅ v3.1: Phoneme timing (list)
                # v3: Phoneme duration targets (opcjonalnie - jeśli dostępne w datasecie)
                # target_phoneme_durations=batch.get('phoneme_durations'),
            )
            # ℹ️ Phoneme duration loss jest teraz wbudowany w p_losses()
            # (use_phoneme_duration_loss=True w LatentDiffusionV2)
            
            # Backward
            loss.backward()
            torch.nn.utils.clip_grad_norm_(ldm.parameters(), 1.0)
            optimizer.step()
            
            total_loss += loss.item()
            pbar.set_postfix({'loss': loss.item()})
        
        scheduler.step()
        
        # Epoch stats
        avg_loss = total_loss / len(dataloader)
        print(f"\nEpoch {epoch+1} - Loss: {avg_loss:.4f}")
        
        # Konfiguracja do zapisania
        ldm_config = {
            'latent_dim': latent_dim,
            'model_channels': args.model_channels,
            'channel_mult': [1, 2, 4, 4],
            'num_timesteps': num_timesteps,
            'use_voice_stream': True,
            'use_dual_voice': True,
            'sample_rate': sample_rate,
        }
        
        # Save checkpoint
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': ldm.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
                'config': ldm_config,  # ✅ Zapisz konfigurację
            }, checkpoint_dir / 'ldm_v2_best.pt')
            print(f"   ✅ Saved best model (loss: {best_loss:.4f})")
        
        if (epoch + 1) % args.save_every == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': ldm.state_dict(),
                'loss': avg_loss,
                'config': ldm_config,  # ✅ Zapisz konfigurację
            }, checkpoint_dir / f'ldm_v2_epoch_{epoch+1}.pt')
    
    print(f"\n✅ Training complete! Best loss: {best_loss:.4f}")


def annotate_segments(args):
    """Anotacja segmentów (Faza 0)"""
    
    if args.full_pipeline:
        # Pełny pipeline: ekstrakcja cech + segmenty + prompty + wokale
        from build_dataset_v2 import DatasetBuilderV2
        
        print("="*60)
        print("🎵 Building Full Dataset v2 (Phase 0 - Full Pipeline)")
        print("="*60)
        
        builder = DatasetBuilderV2(
            audio_dir=args.audio_dir,
            sample_rate=args.sample_rate,
            tracks_csv=args.tracks_csv,
            genres_csv=args.genres_csv,
            artists_csv=args.artists_csv,
            min_segment_duration=args.min_segment,
            # Vocal options
            extract_vocals=args.extract_vocals,
            extract_lyrics=args.extract_lyrics,
            use_demucs=args.use_demucs,
            whisper_model=args.whisper_model,
            device=args.device,
        )
        
        builder.build_dataset(
            output_path=args.output,
            max_tracks=args.max_tracks,
            extract_features=args.extract_features,
            with_segments=True,
        )
    else:
        # Tylko segmenty (szybciej)
        from tools_v2.segment_annotator import SegmentAnnotator, BatchAnnotator
        
        print("="*60)
        print("🎵 Annotating Segments Only (Phase 0)")
        print("="*60)
        print("💡 Tip: Use --full_pipeline for complete dataset with features and prompts")
        
        annotator = SegmentAnnotator(
            sample_rate=args.sample_rate,
            min_segment_duration=args.min_segment,
        )
        
        batch = BatchAnnotator(annotator)
        batch.annotate_directory(
            input_dir=args.audio_dir,
            output_file=args.output,
            max_files=args.max_tracks,
        )


def main():
    parser = argparse.ArgumentParser(
        description='🎵 Train Muzible Muze AI v2 - Section-Aware Generation'
    )
    
    # Phase
    parser.add_argument('--phase', type=int, required=True, 
                        help='Training phase: 1=VAE, 2=composition, 3=ldm (0=deprecated, use build_dataset_v2.py)')
    
    # Paths
    parser.add_argument('--annotations', type=str, default='./data_v2/dataset.json',
                        help='Path to dataset JSON (from build_dataset_v2.py)')
    parser.add_argument('--audio_dir', type=str, default='./music/own',
                        help='Path to audio files (must match file_path in dataset JSON)')
    parser.add_argument('--output', type=str, default='./data_v2/segments.json',
                        help='Output path for annotations')
    parser.add_argument('--vae_checkpoint', type=str, default='./checkpoints/vae_best.pt',
                        help='Path to VAE checkpoint')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints_v2',
                        help='Directory for saving checkpoints')
    
    # Training
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--save_every', type=int, default=10)
    parser.add_argument('--max_tracks', type=int, default=None)
    parser.add_argument('--patience', type=int, default=10, 
                        help='Early stopping patience (epochs without improvement)')
    
    # Model
    parser.add_argument('--d_model', type=int, default=256, help='Composition Planner dim')
    parser.add_argument('--model_channels', type=int, default=256, help='U-Net base channels')
    parser.add_argument('--use_clap', action='store_true', help='Use CLAP encoder')
    parser.add_argument('--latent_dim', type=int, default=128, help='VAE latent dimension')
    parser.add_argument('--sample_rate', type=int, default=32000, help='Audio sample rate (Hz)')
    parser.add_argument('--num_timesteps', type=int, default=200, help='Diffusion timesteps')
    
    # Annotation
    parser.add_argument('--min_segment', type=float, default=4.0)
    parser.add_argument('--full_pipeline', action='store_true',
                        help='Phase 0: Full dataset build (features + segments + prompts)')
    parser.add_argument('--extract_features', action='store_true', default=True,
                        help='Phase 0: Extract detailed audio features')
    parser.add_argument('--tracks_csv', type=str, default='./data/muz_raw_tracks_mod.csv',
                        help='Phase 0: Tracks metadata CSV')
    parser.add_argument('--genres_csv', type=str, default='./data/muz_raw_genres_mod.csv',
                        help='Phase 0: Genres metadata CSV')
    parser.add_argument('--artists_csv', type=str, default='./data/muz_raw_artists_mod.csv',
                        help='Phase 0: Artists metadata CSV')
    
    # 🎤 Vocal processing (Phase 0)
    parser.add_argument('--extract_vocals', action='store_true',
                        help='Phase 0: Extract vocal detection and voice embeddings')
    parser.add_argument('--extract_lyrics', action='store_true',
                        help='Phase 0: Extract lyrics using Whisper')
    parser.add_argument('--use_demucs', action='store_true',
                        help='Phase 0: Use Demucs for vocal separation')
    parser.add_argument('--whisper_model', type=str, default='base',
                        choices=['tiny', 'base', 'small', 'medium', 'large'],
                        help='Phase 0: Whisper model size')
    
    args = parser.parse_args()
    
    # Create checkpoint directory
    Path(args.checkpoint_dir).mkdir(exist_ok=True)
    
    if args.phase == 0:
        annotate_segments(args)
    elif args.phase == 1:
        train_vae(args)
    elif args.phase == 2:
        train_composition_planner(args)
    elif args.phase == 3:
        train_ldm_v2(args)
    else:
        print(f"Unknown phase: {args.phase}")
        print("Available phases:")
        print("  0 - Build dataset (use build_dataset_v2.py instead)")
        print("  1 - Train Audio VAE")
        print("  2 - Train Composition Planner")
        print("  3 - Train Latent Diffusion v2")


if __name__ == "__main__":
    main()
