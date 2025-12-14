# Muzible Muze AI - Technical Documentation v2

> **Text-to-Music Generation with Latent Diffusion & Voice Conditioning**

---

## Table of Contents

1. [System Architecture](#system-architecture)
2. [V2 Architecture - Voice Stream Attention](#v2-architecture---voice-stream-attention)
3. [Training Pipeline](#training-pipeline)
4. [Dataset Format](#dataset-format)
5. [Conditioning System](#conditioning-system)
6. [File Structure](#file-structure)
7. [Usage Scenarios](#usage-scenarios)
8. [Inference - Music Generation](#inference---music-generation)
9. [Detailed File Descriptions](#detailed-file-descriptions)
10. [FAQ & Troubleshooting](#faq--troubleshooting)
11. [Model Size Configuration](#model-size-configuration)
12. [Requirements](#requirements)

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         MUZIBLE MUZE AI v2                                  │
│                   Text-to-Music Generation Pipeline                         │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐                  │
│  │   INPUTS     │    │   ENCODERS   │    │   OUTPUTS    │                  │
│  ├──────────────┤    ├──────────────┤    ├──────────────┤                  │
│  │ Text Prompt  │───▶│ T5/CLAP      │───▶│              │                  │
│  │ Voice Sample │───▶│ Resemblyzer  │───▶│  UNet V2     │                  │
│  │ Style Ref    │───▶│ ECAPA-TDNN   │───▶│  (Diffusion) │──▶ Audio WAV    │
│  │ Lyrics       │───▶│ Gruut/eSpeak │───▶│              │                  │
│  └──────────────┘    └──────────────┘    └──────────────┘                  │
│                                                                             │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │                    CORE COMPONENTS                                    │  │
│  ├──────────────────────────────────────────────────────────────────────┤  │
│  │  AudioVAE (224M)  │  UNet V2 (722M-6.1B)  │  Vocos Vocoder          │  │
│  │  - Mel → Latent   │  - Noise → Latent     │  - Mel → Waveform       │  │
│  │  - Latent → Mel   │  - Voice Attention    │  - 44.1kHz output       │  │
│  │  - KL + STFT Loss │  - Section Cond.      │  - High quality         │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Main Models

| Component | Parameters | Function |
|-----------|------------|----------|
| **AudioVAE** | 55-889M | Audio compression to latent space |
| **UNet V2** | 722M-6.1B | Latent diffusion denoising |
| **T5 Encoder** | 250M | Text prompt encoding |
| **CLAP** | 600M | Audio-text joint embeddings |
| **Vocos** | 13M | High-quality vocoder |

---

## V2 Architecture - Voice Stream Attention

### What is VoiceStreamAttention?

**VoiceStreamAttention** is a **dedicated cross-attention mechanism** that allows the diffusion model to attend to voice embedding **separately** from text embedding.

```
Standard Cross-Attention (v1):
    Q = latent, K,V = text_embedding
    
V2 Voice Stream Attention:
    Branch 1: Q = latent, K,V = text_embedding      → text_attn
    Branch 2: Q = latent, K,V = voice_embedding     → voice_attn
    Output: gate * voice_attn + (1-gate) * text_attn
```

### Why is it important?

1. **Voice quality** - Model can "focus" on voice characteristics independently
2. **Timbre control** - Voice gate allows dynamic balance between text and voice
3. **Better disentanglement** - Voice separated from semantics

### V2 Architecture Diagram

```
                    ┌─────────────────────────────────────────────┐
                    │              UNet V2 Block                   │
                    ├─────────────────────────────────────────────┤
Input Latent ──────▶│  ResBlock  │  Self-Attn  │  Cross-Attn     │
    [B,128,H,W]     │            │             │                  │
                    │            │             │   ┌────────────┐ │
                    │            │             │   │ Text K,V   │ │
                    │            │             │   │ [B,768]    │ │
                    │            │             │   └─────┬──────┘ │
                    │            │             │         │        │
                    │            │             │   ┌─────▼──────┐ │
                    │            │             │   │ text_attn  │ │
                    │            │             │   └─────┬──────┘ │
                    │            │             │         │        │
                    │            │             │   ┌─────▼──────┐ │
                    │            │             │   │ GATED MIX  │◀── gate (learnable)
                    │            │             │   └─────┬──────┘ │
                    │            │             │         │        │
                    │            │             │   ┌─────▼──────┐ │
                    │            │             │   │ voice_attn │ │
                    │            │             │   └─────┬──────┘ │
                    │            │             │         │        │
                    │            │             │   ┌─────▼──────┐ │
                    │            │             │   │ Voice K,V  │ │
                    │            │             │   │ [B,256]    │ │
                    │            │             │   └────────────┘ │
                    └─────────────────────────────────────────────┘
                                        │
                                        ▼
                              Output Latent [B,128,H,W]
```

### VoiceEmbeddingFusion (v2)

In v2, we use **two voice embeddings**:

| Embedding | Dimension | Model | Characteristics |
|-----------|-----------|-------|-----------------|
| **Resemblyzer** | 256 | GE2E | General speaker verification |
| **ECAPA-TDNN** | 192 | SpeechBrain | Better for singing voice |

```python
# Fusion
voice_fused = VoiceEmbeddingFusion(
    resemblyzer_embed,  # [B, 256]
    ecapa_embed         # [B, 192]
)
# Output: [B, 256] - weighted projection
```

---

## Training Pipeline

### Phase Overview

```
┌────────────────────────────────────────────────────────────────────────────┐
│                        TRAINING PIPELINE v2                                 │
├────────────────────────────────────────────────────────────────────────────┤
│                                                                            │
│  Phase 1: VAE (Audio Compression)                                          │
│  ────────────────────────────────                                          │
│  Audio WAV → Mel Spectrogram → Encoder → μ, σ → z (latent) → Decoder → Mel │
│                                                                            │
│  Loss: MSE(mel, mel_recon) + β*KL(z) + STFT_loss                          │
│  Target: Reconstruct audio with minimal latent dim (128)                   │
│                                                                            │
├────────────────────────────────────────────────────────────────────────────┤
│                                                                            │
│  Phase 2: Composition Planner (Optional)                                   │
│  ───────────────────────────────────────                                   │
│  Track features → MLP → Section plan (verse, chorus, bridge, etc.)         │
│                                                                            │
│  Loss: CrossEntropy(predicted_sections, ground_truth_sections)             │
│  Target: Learn song structure from metadata                                │
│                                                                            │
├────────────────────────────────────────────────────────────────────────────┤
│                                                                            │
│  Phase 3: Latent Diffusion Model (LDM)                                     │
│  ─────────────────────────────────────                                     │
│  Noise z_T → UNet V2 (conditioned) → ... → z_0 → VAE Decode → Audio        │
│                                                                            │
│  Conditioning:                                                             │
│  - Text: T5/CLAP embedding [768]                                           │
│  - Voice: Resemblyzer [256] + ECAPA [192]                                  │
│  - Section: type, position, energy, tempo, key                             │
│  - Audio: CLAP audio embedding [512]                                       │
│  - Beat/Chord/Phoneme encoders                                             │
│                                                                            │
│  Loss: MSE(predicted_noise, actual_noise) + cfg_loss                       │
│                                                                            │
└────────────────────────────────────────────────────────────────────────────┘
```

### LDM Training with All Conditioning

```
┌────────────────────────────────────────────────────────────────────────────┐
│                    LDM v2 TRAINING - FULL CONDITIONING                     │
├────────────────────────────────────────────────────────────────────────────┤
│                                                                            │
│  INPUTS (per batch):                                                       │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │ audio_path: "music/fma_small/000/000123.mp3"                         │  │
│  │ prompt: "Energetic rock with electric guitar and drums"              │  │
│  │ section_type: "chorus"                                               │  │
│  │ position: 0.35                                                       │  │
│  │ energy: 0.82                                                         │  │
│  │ tempo: 128.0                                                         │  │
│  │ key: "C major"                                                       │  │
│  │ voice_embedding: [256-dim tensor]                                    │  │
│  │ ecapa_embedding: [192-dim tensor]                                    │  │
│  │ clap_audio_embedding: [512-dim tensor]                               │  │
│  │ clap_text_embedding: [512-dim tensor]                                │  │
│  │ num_beats: 64                                                        │  │
│  │ beat_positions: [[0.0, 0.47], [0.47, 0.94], ...]                     │  │
│  │ current_chord: "C:maj"                                               │  │
│  │ phonemes_ipa: "ðɪs ɪz ə tɛst"                                        │  │
│  │ f0_contour: [440.0, 442.1, ...]                                      │  │
│  │ vibrato_rate, vibrato_depth, breath_positions, ...                   │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
│                                                                            │
│  PROCESSING:                                                               │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │                                                                      │  │
│  │  1. Load audio → Mel spectrogram                                     │  │
│  │  2. VAE.encode(mel) → z_0 (latent)                                   │  │
│  │  3. Sample timestep t ~ Uniform(0, T)                                │  │
│  │  4. Add noise: z_t = √ᾱₜ·z_0 + √(1-ᾱₜ)·ε                            │  │
│  │  5. Encode conditioning:                                             │  │
│  │     - text_embed = T5(prompt)           [768]                        │  │
│  │     - voice_fused = Fusion(voice, ecapa) [256]                       │  │
│  │     - section_cond = SectionModule(...)  [1024]                      │  │
│  │  6. UNet forward: ε_θ = UNet(z_t, t, text_embed, voice_fused, ...)  │  │
│  │  7. Loss = MSE(ε_θ, ε)                                               │  │
│  │                                                                      │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
│                                                                            │
└────────────────────────────────────────────────────────────────────────────┘
```

---

## Dataset Format

### Dataset JSON Structure (v3.1)

```json
{
  "audio_path": "music/fma_small/000/000123.mp3",
  "track_id": 123,
  "artist": "Artist Name",
  "title": "Song Title",
  "album": "Album Name",
  "genre": "rock",
  "year": 2023,
  
  "prompt": "Energetic rock song with electric guitar riffs and powerful drums...",
  "text_sentiment": "positive",
  
  "duration": 180.5,
  "sample_rate": 32000,
  "tempo": 128.0,
  "key": "C major",
  "time_signature": "4/4",
  "loudness_db": -8.5,
  "energy": 0.82,
  
  "has_vocals": true,
  "lyrics": "Transcribed lyrics from the song...",
  "phonemes_ipa": "ðɪs ɪz ðə faɪnəl transkrɪpʃən",
  
  "voice_embedding": [0.12, -0.34, ...],
  "voice_embedding_separated": [0.15, -0.31, ...],
  
  "clap_audio_embedding": [0.05, 0.12, ...],
  "clap_text_embedding": [0.08, 0.15, ...],
  
  "segments": [
    {
      "type": "intro",
      "start": 0.0,
      "end": 15.2,
      "energy": 0.3,
      "has_vocals": false
    },
    {
      "type": "verse",
      "start": 15.2,
      "end": 45.8,
      "energy": 0.6,
      "has_vocals": true,
      "lyrics": "First verse lyrics..."
    },
    {
      "type": "chorus",
      "start": 45.8,
      "end": 76.4,
      "energy": 0.9,
      "has_vocals": true,
      "lyrics": "Chorus lyrics..."
    }
  ],
  
  "beat_positions": [[0.0, 0.47], [0.47, 0.94], ...],
  "downbeat_positions": [0.0, 1.88, 3.76, ...],
  "chord_progression": ["C:maj", "G:maj", "Am:min", "F:maj"],
  
  "f0_contour": [440.0, 442.1, 438.5, ...],
  "f0_voiced_mask": [true, true, false, ...],
  "vibrato_rate": 5.2,
  "vibrato_depth": 0.15,
  "vibrato_extent": 0.8,
  "breath_positions": [[12.5, 12.8], [25.1, 25.4], ...],
  "phoneme_timestamps": [
    {"phoneme": "ð", "start": 0.0, "end": 0.05},
    {"phoneme": "ɪ", "start": 0.05, "end": 0.12}
  ]
}
```

---

## Conditioning System

### Conditioning Summary

| Parameter | Type | Dimension | Encoder |
|-----------|------|-----------|---------|
| `prompt` | str | → 768 | T5TextEncoder |
| `section_type` | str | → 128 | SectionEmbedding |
| `position` | float 0-1 | → 128 | SinusoidalPosEmb |
| `energy` | float 0-1 | → 64 | Linear |
| `tempo` | float BPM | → 64 | Linear (normalized) |
| `key` | int 0-23 | → 64 | KeyEmbedding |
| `loudness` | float dB | → 64 | Linear |
| `has_vocals` | bool | → 32 | Linear |
| `sentiment` | str | → 64 | SentimentEmbedding |
| `genre` | str | → 64 | GenreEmbedding |
| `artist` | str | → 64 | ArtistEmbedding |
| `clap_audio` | 512-dim | → 128 | Linear projection |
| `clap_text` | 512-dim | → 128 | Linear projection |
| `voice_embedding` | 256-dim | → 256 | VoiceStreamAttention |
| `ecapa_embedding` | 192-dim | → 256 | VoiceEmbeddingFusion |
| `num_beats` | int | → 64 | BeatEmbedding |
| `beat_positions` | List[List[float]] | → 64 | BeatEmbedding |
| `time_signature` | str | → 32 | TimeSignatureEmb |
| `current_chord` | str | → 64 | ChordEmbedding |
| `phonemes_ipa` | str | → 128 | PhonemeEncoder (GRU) |
| `f0_contour` | List[float] | → 64 | F0Encoder (Conv1d) |
| `f0_voiced_mask` | List[bool] | → 32 | VoicedMaskEncoder |
| `vibrato_rate` | float Hz | → 64 | VibratoEncoder |
| `vibrato_depth` | float cents | → 64 | VibratoEncoder |
| `vibrato_extent` | float 0-1 | → 64 | VibratoEncoder |
| `breath_positions` | List[List[float]] | → 32 | BreathEncoder |

### Fusion Dimensions

```
Base:     section(128) + position(128) + energy(64) + tempo(64) + key(64) + text(512)
          + loudness(64) + has_vocals(32) + sentiment(64) + genre(64) + artist(64) = 1248

Optional: + clap(128) + beat(64) + chord(64) + phoneme(128)
          + pitch(64) + vibrato(64) + breath(32) + phoneme_ts(64) = 1856

Final:    Fusion MLP → output_dim (1024)
```

---

## Inference Pipeline

```
┌────────────────────────────────────────────────────────────────────────────┐
│                      INFERENCE PIPELINE v2                                  │
├────────────────────────────────────────────────────────────────────────────┤
│                                                                            │
│  INPUT: "Energetic rock song with female vocals"                           │
│         + voice_sample.wav (optional)                                      │
│         + lyrics (optional)                                                │
│                                                                            │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ Step 1: Text Encoding                                               │   │
│  │   prompt → T5Encoder → text_embed [768]                             │   │
│  │   prompt → CLAPTextEncoder → clap_text_embed [512]                  │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                           │                                                │
│                           ▼                                                │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ Step 2: Voice Encoding (if voice_sample provided)                   │   │
│  │   voice.wav → Resemblyzer → voice_embed [256]                       │   │
│  │   voice.wav → ECAPA-TDNN → ecapa_embed [192]                        │   │
│  │   Fusion(voice, ecapa) → voice_fused [256]                          │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                           │                                                │
│                           ▼                                                │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ Step 3: Composition Planning                                        │   │
│  │   Template "verse_chorus" → [intro, verse, chorus, verse, chorus]   │   │
│  │   Each section: duration, energy, position                          │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                           │                                                │
│                           ▼                                                │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ Step 4: Per-Section Generation (DDPM/DDIM)                          │   │
│  │                                                                     │   │
│  │   For each section:                                                 │   │
│  │     z_T ~ N(0, I)                     # Start with noise            │   │
│  │     for t = T, T-1, ..., 1:                                         │   │
│  │       ε_θ = UNet(z_t, t, text_embed, voice_fused, section_cond)    │   │
│  │       z_{t-1} = DDPM_step(z_t, ε_θ, t)                             │   │
│  │     z_0 = final denoised latent                                     │   │
│  │                                                                     │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                           │                                                │
│                           ▼                                                │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ Step 5: Audio Decoding                                              │   │
│  │   z_0 → VAE.decode() → mel_spectrogram [128, T]                     │   │
│  │   mel → Vocos → audio_waveform [samples]                            │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                           │                                                │
│                           ▼                                                │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ Step 6: Concatenation                                               │   │
│  │   [intro_audio, verse_audio, chorus_audio, ...] → final_audio.wav   │   │
│  │   Apply crossfade between sections (50ms)                           │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                            │
│  OUTPUT: final_audio.wav (44.1kHz stereo)                                  │
│                                                                            │
└────────────────────────────────────────────────────────────────────────────┘
```

### DDPM vs DDIM

| Method | Steps | Speed | Quality |
|--------|-------|-------|---------|
| DDPM | 1000 | Slow (~2min/30s) | Best |
| DDPM-50 | 50 | Medium (~15s/30s) | Good |
| DDIM-50 | 50 | Medium (~15s/30s) | Good |
| DDIM-20 | 20 | Fast (~6s/30s) | Acceptable |

**Recommendation:** Use DDIM-50 for production, DDPM-1000 for final renders.

---

## UNet V2 - Key Modules

```python
UNetV2(
    in_channels=128,        # latent_dim from VAE (v2: increased from 8)
    out_channels=128,
    model_channels=320,     # main "size knob"
    num_res_blocks=2,
    attention_resolutions=[8, 4, 2],
    context_dim=768,        # text embedding dim
    num_heads=8,
    
    # v2: Voice conditioning
    voice_dim=256,              # Resemblyzer embedding
    ecapa_dim=192,              # ECAPA-TDNN embedding (voice_emb_separated)
    clap_dim=512,               # CLAP audio+text embedding
    use_voice_stream=True,      # VoiceStreamAttention
    use_dual_voice=True,        # Resemblyzer + ECAPA fusion
    
    # v2: Beat/Chord/Phoneme
    use_clap=True,
    use_beat=True,              # BeatEmbedding
    use_chord=True,             # ChordEmbedding
    use_phonemes=True,          # PhonemeEncoder
    
    # v3: Pitch conditioning
    use_pitch=True,
    f0_encoder_dim=64,
    
    # v3.1: Singing expression
    vibrato_encoder_dim=64,
    breath_encoder_dim=32,
    phoneme_timestamp_encoder_dim=64,
    
    # Performance
    use_gradient_checkpointing=True,
)
```

### SectionConditioningModule - Sections and Metadata

```python
SectionConditioningModule(
    output_dim=1024,
    text_embed_dim=768,
    section_embed_dim=128,
    num_keys=24,            # C-B major/minor
    
    # v2 modules:
    use_clap=True,
    use_beat=True,
    use_chord=True,
    use_phonemes=True,
    
    # v3 modules:
    use_pitch=True,
    clap_dim=512,
    voice_dim=256,
)

# Forward accepts 30+ conditioning parameters
section_cond.forward(
    text_embed,                     # [B, 768] or [B, seq, 768]
    section_type,                   # List[str]
    position, energy, tempo,        # [B] floats
    key_idx,                        # [B] int 0-23
    loudness, has_vocals,           # [B] v3 metadata
    sentiment_score, genres, artists,
    clap_audio_embedding, clap_text_embedding,
    num_beats, beat_positions, time_signature, current_chord,
    phonemes_ipa, voice_embedding,
    f0, f0_coarse, f0_voiced_mask,
    vibrato_rate, vibrato_depth, vibrato_extent,
    breath_positions, phoneme_timestamps,
    segment_duration,
)
→ (conditioning [B, 1024], phoneme_durations or None)
```

---

## File Structure

```
muzible-muze-ai/
├── 📄 train_v2.py                 # Training script v2 (3-phase)
├── 📄 inference_v2.py             # Music generation from model
├── 📄 build_dataset_v2.py         # Dataset builder v2 (full extraction)
│
├── 📁 docs_v2/                    # Documentation
│   └── 📄 DATASET_BUILDER.md      # Full dataset builder documentation
│
├── 📁 tools/
│   ├── 📄 f0_extractor.py         # F0/pitch extraction
│   └── 📄 analyze_metadata.py     # Metadata analysis
│
├── 📁 tools_v2/                   # Tools v2
│   ├── 📄 segment_annotator.py    # Segment detection (verse/chorus)
│   ├── 📄 generate_artist_embeddings.py  # Voice embeddings generation
│   └── 📄 scan_mp3_folder.py      # MP3 folder scanning
│
├── 📁 models/
│   ├── 📄 audio_vae.py            # Audio VAE (audio → latent compression)
│   ├── 📄 vocoder.py              # Vocoder (mel → waveform)
│   └── 📄 voice_synthesis.py      # Voice cloning (XTTS, Demucs)
│
├── 📁 models_v2/                  # 🆕 Architecture V2
│   └── 📄 latent_diffusion.py     # U-Net V2 + all encoders
│
├── 📁 data/                       # Data v1 (legacy)
│   ├── 📄 music_dataset.py        # PyTorch Dataset
│   └── 📄 training_dataset.json   # Dataset v1
│
├── 📁 data_v2/                    # 🆕 Data v2
│   ├── 📄 segmented_dataset.py    # SegmentedMusicDataset
│   └── 📄 *.json                  # Datasets v2
│
├── 📁 music/
│   └── 📁 fma_small/              # FMA audio files
│
├── 📁 checkpoints/                # Checkpoints v1
├── 📁 checkpoints_v2/             # 🆕 Checkpoints v2
│
└── 📁 output/                     # Generated audio
```

---

## Usage Scenarios

### Scenario 1: Training from Scratch on Your Own MP3s

**When to use:** You have your own MP3 collection and want to train a model from scratch.

#### Step 1: Prepare Folder Structure

```bash
mkdir -p my_music/artist_name
cp ~/Music/*.mp3 my_music/artist_name/
```

#### Step 2: Generate Dataset

```bash
# Full pipeline with audio analysis, vocals and voice embeddings
python build_dataset_v2.py \
    --audio_dir ./my_music \
    --output ./data_v2/my_dataset.json \
    --device cuda \
    --batch_size 4
```

**Generated files:**
- `my_dataset.json` - metadata + prompts + all audio features (CLAP, voice, F0, etc.)
- `my_dataset.artist_embeddings.json` - average voice embeddings per artist

#### Step 3: Train VAE (Phase 1)

```bash
python train_v2.py \
    --phase 1 \
    --annotations ./data_v2/my_dataset.json \
    --audio_dir ./my_music \
    --epochs 50 \
    --batch_size 4 \
    --device cuda
```

**Time:** ~2-4h for 1000 tracks (GPU RTX 3090)

#### Step 4: Train Diffusion (Phase 3)

```bash
python train_v2.py \
    --phase 3 \
    --annotations ./data_v2/my_dataset.json \
    --audio_dir ./my_music \
    --vae_checkpoint ./checkpoints/vae_epoch_50.pt \
    --epochs 100 \
    --batch_size 2 \
    --device cuda
```

**Time:** ~8-12h for 1000 tracks (GPU RTX 3090)

---

### Scenario 2: Training on FMA Dataset

**When to use:** You have the FMA dataset and want to train a model.

#### Step 1: Download FMA (if you don't have it)

```bash
# FMA Small (~8GB, 8000 tracks)
wget https://os.unil.cloud.switch.ch/fma/fma_small.zip
unzip fma_small.zip -d ./music/
```

#### Step 2: Build Dataset v2

```bash
python build_dataset_v2.py \
    --audio_dir ./music/fma_small \
    --output ./data_v2/fma_dataset.json \
    --device cuda \
    --batch_size 4
```

**What build_dataset_v2 generates:**
| Field | Description | Source |
|-------|-------------|--------|
| `has_vocals` | Whether track has vocals | Whisper |
| `lyrics` | Text transcription | Whisper |
| `voice_embedding` | 256-dim vector | Resemblyzer |
| `ecapa_embedding` | 192-dim vector | ECAPA-TDNN |
| `clap_audio_embedding` | 512-dim | CLAP |
| `clap_text_embedding` | 512-dim | CLAP |
| `f0_contour` | Pitch contour | CREPE/pYIN |
| `vibrato_*` | Vibrato features | Custom |
| `breath_positions` | Breath timings | Custom |
| `phoneme_timestamps` | IPA + timing | Gruut/eSpeak |

#### Step 3: Training

```bash
# Phase 1: VAE
python train_v2.py --phase 1 \
    --annotations ./data_v2/fma_dataset.json \
    --audio_dir ./music/fma_small \
    --epochs 50

# Phase 3: LDM with voice conditioning
python train_v2.py --phase 3 \
    --annotations ./data_v2/fma_dataset.json \
    --audio_dir ./music/fma_small \
    --vae_checkpoint ./checkpoints_v2/vae_best.pt \
    --epochs 100
```

---

### Scenario 3: Adding New Tracks to Dataset

**When to use:** You already have a dataset and want to add new tracks.

#### Method A: Rebuild with New Folder

```bash
# Add new MP3s to folder
cp ~/new_music/*.mp3 ./music/fma_small/new/

# Rebuild dataset (will detect new files)
python build_dataset_v2.py \
    --audio_dir ./music/fma_small \
    --output ./data_v2/dataset_updated.json \
    --device cuda
```

#### Method B: Merge JSON

```python
import json

# Load existing
with open('data_v2/dataset.json') as f:
    dataset = json.load(f)

# Load new
with open('data_v2/new_tracks.json') as f:
    new_tracks = json.load(f)

# Merge (check duplicates by audio_path)
existing_paths = {t['audio_path'] for t in dataset}
for track in new_tracks:
    if track['audio_path'] not in existing_paths:
        dataset.append(track)

# Save
with open('data_v2/dataset_merged.json', 'w') as f:
    json.dump(dataset, f, indent=2)
```

#### Method C: Continue Training (Fine-tuning)

```bash
# Fine-tune on new data
python train_v2.py --phase 3 \
    --annotations ./data_v2/dataset_merged.json \
    --audio_dir ./music \
    --vae_checkpoint ./checkpoints_v2/vae_best.pt \
    --ldm_checkpoint ./checkpoints_v2/ldm_epoch_100.pt \
    --epochs 20  # Fewer epochs for fine-tuning
```

---

## Inference - Music Generation

### Basic Generation

```bash
python inference_v2.py \
    --prompt "Energetic electronic dance track with heavy bass" \
    --output ./output/edm_track.wav \
    --duration 30 \
    --device cuda
```

### With Artist Style (Voice Embedding)

```bash
python inference_v2.py \
    --prompt "Melodic hip-hop beat" \
    --style_of "Artist Name" \
    --output ./output/artist_style.wav
```

### With Voice Cloning

```bash
python inference_v2.py \
    --prompt "Upbeat pop song" \
    --voice_clone "Artist Name" \
    --lyrics "Here are the lyrics to sing..." \
    --output ./output/cloned_voice.wav
```

### With Structure Template

```bash
python inference_v2.py \
    --prompt "Energetic pop with female vocals" \
    --template verse_chorus \
    --duration 120 \
    --output ./output/structured_song.wav
```

### All Options

```bash
python inference_v2.py --help

# Main options:
#   --prompt TEXT          Prompt describing the music
#   --output PATH          Output path (default: ./output/generated.wav)
#   --duration FLOAT       Duration in seconds (default: 30)
#   --cfg_scale FLOAT      Classifier-free guidance (default: 7.5)
#   --num_steps INT        Denoising steps (default: 50)
#   --template NAME        Structure template (verse_chorus, etc.)
#
# Voice conditioning:
#   --style_of NAME/PATH   Artist voice embedding or .wav file
#
# Voice cloning:
#   --voice_clone NAME     Artist to clone voice from
#   --voice_clone_samples PATH  Folder/file with voice samples
#   --lyrics TEXT          Text to sing
#   --language CODE        Language code (pl, en, de, etc.)
```

---

## Detailed File Descriptions

### 📄 `train_v2.py`

**Purpose:** Main v2 training script for VAE, Composition Planner and LDM.

**Training phases:**
1. **Phase 1 (VAE):** Audio → Mel → Latent → Mel (reconstruction)
2. **Phase 2 (Composition Planner):** Track features → Composition plan
3. **Phase 3 (LDM):** Noise → UNet V2 (conditioned) → Latent → VAE → Audio

**Key parameters:**
```python
# VAE
latent_dim = 128      # v2: increased from 8
sample_rate = 32000   # v2: 32kHz

# LDM
cfg_dropout = 0.1     # Classifier-free guidance dropout
voice_dropout = 0.1   # Voice conditioning dropout
```

---

### 📄 `inference_v2.py`

**Purpose:** Generate music from trained v2 model.

**Main functions:**
- `generate_composition_plan()` - plan track structure
- `generate_section_audio()` - generate single section
- `generate_full_song()` - generate full track section by section

**Pipeline:**
1. Prompt → T5/CLAP Encoder → text embedding
2. (optional) Voice sample → Resemblyzer/ECAPA → voice embedding
3. (optional) Lyrics → Gruut/eSpeak → phonemes IPA
4. Template → CompositionPlanner → section structure
5. Per section: Noise + embeddings → UNet V2 denoising → Latent
6. Latent → VAE Decoder → Mel spectrogram
7. Mel → Vocos → Audio WAV
8. Concat all sections → Final audio

---

### 📄 `build_dataset_v2.py`

**Purpose:** Full feature extraction from audio files.

**Extracts:**
- Metadata (ID3 tags)
- Audio features (librosa: tempo, key, energy, etc.)
- Voice embeddings (Resemblyzer 256-dim + ECAPA-TDNN 192-dim)
- CLAP embeddings (audio 512-dim + text 512-dim)
- Pitch/F0 (CREPE/pYIN)
- Vibrato, breath, phoneme features
- Segment detection (verse/chorus/bridge)
- Lyrics transcription (Whisper)

**Output:** JSON with v3.1 fields (see DATASET diagram above)

---

### 📄 `models_v2/latent_diffusion.py`

**Purpose:** UNet V2 + all conditioning modules.

**Main classes:**
- `UNetV2` - main diffusion model
- `SectionConditioningModule` - fusion of all conditioning
- `VoiceStreamAttention` - gated cross-attention for voice
- `VoiceEmbeddingFusion` - Resemblyzer + ECAPA fusion
- `PitchEncoder`, `VibratoEncoder`, `BreathEncoder` - feature encoders
- `BeatEmbedding`, `ChordEmbedding`, `PhonemeEncoder` - v2 encoders

---

### 📄 `models/audio_vae.py`

**Purpose:** Audio compression to latent space.

**Architecture v2:**
```
Mel [1, 128, T] → Encoder → μ, σ → z [128, H, W] → Decoder → Mel [1, 128, T]
```

**Parameters:**
- `latent_dim = 8` - latent channel dimension
- `channels = [64, 128, 256, 512]` - encoder channels
- `n_mels = 128` - number of mel filterbanks (v2: increased from 80)

**Loss:**
```python
loss = reconstruction_loss + beta * kl_divergence + stft_loss
```

---

### 📄 `models/text_encoder.py`

**Purpose:** Text prompt encoding.

**Backends:**
- `T5TextEncoder` - Flan-T5 (768-dim, good for long descriptions)
- `CLAPTextEncoder` - CLAP (specifically trained on audio-text)

---

### 📄 `models/voice_synthesis.py`

**Purpose:** Voice cloning and synthesis.

**Usage:**
```python
# 1. Extract vocals
extractor = VoiceExtractorFromSong()
vocals_path = extractor.extract_vocals("song.mp3")

# 2. Register voice
synth = VoiceSynthesizer(backend="coqui")
synth.register_voice("artist", vocals_path)

# 3. Synthesize new text
audio = synth.synthesize("New lyrics...", voice="artist")
```

---

### 📄 `data/music_dataset.py`

**Purpose:** PyTorch Dataset for training.

**Returns batch:**
```python
{
    'audio': torch.Tensor,           # [num_samples]
    'prompt': str,                   # "Energetic rock song..."
    'voice_embedding': torch.Tensor, # [256] or None
    'lyrics': str,                   # "Transcribed lyrics..."
    'has_vocals': bool,
    'text_sentiment': str,           # "positive"
    'track_id': int,
    'artist': str,
}
```

**Custom collate_fn:**
- Stacks tensors
- Groups strings into lists
- Handles None in voice_embedding

---

## FAQ & Troubleshooting

### ❓ Why `/var/folders/...` in vocals path?

**Question:** `Vocals saved to: /var/folders/fg/frwh54994k9gy6h5y_tc1_940000gn/T/2_Food_vocals.wav`

**Answer:** This is the **default macOS temporary folder** (`tempfile.gettempdir()`).

`VoiceExtractorFromSong` saves extracted vocals to the system temporary folder by default, which on macOS is:
```
/var/folders/XX/XXXX/T/
```

**Solution:** Set your own `output_dir`:

```python
extractor = VoiceExtractorFromSong(
    output_dir="./data/separated_vocals"  # Permanent folder
)
```

Or when building dataset with `build_dataset_v2.py` use `--separate_vocals` flag.

---

### ❓ Training is Very Slow on CPU

**Problem:** Training on CPU takes hours even for a few tracks.

**Solutions:**
1. Use GPU: `--device cuda`
2. Reduce batch size: `--batch_size 1`
3. Reduce number of tracks: `--max_tracks 10`
4. Use mixed precision (auto on GPU)

---

### ❓ `CUDA out of memory`

**Problem:** GPU doesn't have enough memory.

**Solutions:**
1. Reduce batch size: `--batch_size 1`
2. Use gradient checkpointing (enabled by default)
3. Use smaller VAE model
4. Shorten duration: change in code `duration=5.0`

---

### ❓ Voice Cloning Sounds Robotic

**Problem:** XTTS generates artificial voice.

**Solutions:**
1. Use longer voice sample (>30s)
2. Make sure sample has clean vocals (no instruments)
3. Use ElevenLabs instead of Coqui (better quality, paid)

---

### ❓ Whisper Doesn't Detect Vocals

**Problem:** `has_vocals: false` for tracks with vocals.

**Causes:**
1. Instrumental too loud
2. Vocals in unsupported language
3. Analyzed fragment too short

**Solutions:**
1. Use `--whisper_full` (analyze entire track)
2. Use larger model: `--whisper_model medium`
3. First separate vocals: `--separate_vocals`

---

### ❓ Missing Module `speechbrain`

**Warning:** `No module named 'speechbrain'`

**Solution:** System automatically uses `resemblyzer` as fallback. If you want SpeechBrain:
```bash
pip install speechbrain
```

---

## 📊 Model Size Configuration

### Model Size Parameters

| Parameter | Impact | Description |
|-----------|--------|-------------|
| `latent_dim` | Minimal (~3M) | VAE latent space dimension |
| `model_channels` | **KEY** | Base UNet channel width - main "size knob" |

### Model Size Table

| Config | latent_dim | model_channels | VAE | UNet | **Total** |
|--------|-----------|----------------|-----|------|-----------|
| Test/Dev | 128 | 256 | 224M | 722M | **~1B** |
| Production Default | 128 | 320 | 224M | 1.1B | **~1.3B** |
| Large Production | 128 | 512 | 224M | 2.8B | **~3B** |
| XL Production | 256 | 512 | 228M | 2.8B | **~3B** |
| XXL (multi-billion) | 256 | 768 | 228M | 6.1B | **~6.4B** |

### Conclusions

- **`latent_dim=128` is sufficient** - difference between 128 and 256 is only ~3M parameters in VAE (~1.5% difference)
- **`model_channels` is the real "size knob"** - increasing from 320→512 gives jump from 1.1B→2.8B
- For **several billion parameters**: `model_channels=512-768` is key

### Recommendations

| Use Case | Configuration | Size |
|----------|---------------|------|
| Local testing/dev | `latent_dim=128, model_channels=256` | ~1B |
| Standard production | `latent_dim=128, model_channels=320` | ~1.3B |
| Large production model | `latent_dim=128, model_channels=512` | ~3B |
| Very large model | `latent_dim=256, model_channels=768` | ~6.4B |

### Code Configuration Example

```python
# Test/Dev (~1B)
unet = UNetV2(
    in_channels=128,
    out_channels=128,
    model_channels=256,  # smaller for quick testing
    context_dim=768,
)

# Production (~3B)
unet = UNetV2(
    in_channels=128,
    out_channels=128,
    model_channels=512,  # larger for quality
    context_dim=768,
)
```

### AudioVAE - Full Configuration

**`AudioVAE.__init__` parameters:**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `sample_rate` | 32000 | v2: 32kHz (v1: 22050) |
| `n_mels` | 128 | Number of mel bins |
| `n_fft` | 1024 | FFT window size |
| `hop_length` | 320 | 10ms hop @ 32kHz |
| `latent_dim` | 128 | v2: increased from 8 |
| `channels` | None | Auto-select from `LATENT_CONFIGS` |
| `use_stft_loss` | True | Multi-Resolution STFT Loss |
| `use_checkpoint` | False | Gradient checkpointing (saves VRAM) |

**Auto-select channels (`LATENT_CONFIGS`):**

| latent_dim | channels (auto) | VAE Size |
|------------|-----------------|----------|
| 8 | [64, 128, 256, 512] | **55M** |
| 32 | [64, 128, 256, 512] | **56M** |
| 64 | [96, 192, 384, 768] | **125M** |
| 128 | [128, 256, 512, 1024] | **224M** |

**Custom channels - full scale:**

| Config | channels | Size |
|--------|----------|------|
| v2 Light | [64, 128, 256, 512] | **57M** |
| v2 Default | [128, 256, 512, 1024] | **224M** |
| v2 Heavy | [256, 512, 1024, 2048] | **889M** |

**VAE configuration examples:**

```python
# Default v2 (224M) - recommended
vae = AudioVAE(latent_dim=128)

# Light (57M) - quick tests
vae = AudioVAE(latent_dim=128, channels=[64, 128, 256, 512])

# Heavy (889M) - maximum reconstruction quality
vae = AudioVAE(latent_dim=128, channels=[256, 512, 1024, 2048])

# With gradient checkpointing (less VRAM)
vae = AudioVAE(latent_dim=128, use_checkpoint=True)
```

---

## Requirements

```txt
# Core
torch>=2.0
torchaudio>=2.0
transformers>=4.30
einops
vocos

# Audio processing
librosa
soundfile
mutagen

# Whisper (optional)
faster-whisper  # or openai-whisper

# Voice embeddings (one of):
resemblyzer        # lightweight (256-dim)
speechbrain        # better (192-dim ECAPA-TDNN)

# Voice cloning (optional)
TTS                # Coqui XTTS v2
demucs             # Vocal separation

# LLM (optional)
openai             # GPT-4
requests           # Ollama
```

---

## License

GPL-2.0 License - use for your own projects!

⚠️ **Legal notice:** Voice cloning may violate artists' voice likeness rights. Use only with your own voice or with the owner's consent.

---

## Related Documents

- 📘 [Dataset Builder - Full Documentation](docs_v2/DATASET_BUILDER.md)

---

*Documentation generated: December 14, 2025*

