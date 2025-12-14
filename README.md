# 🎵 Muzible Muze AI

> **Text-to-Music Generation with Latent Diffusion & Voice Conditioning**

Generate full songs from text prompts with optional voice style transfer or voice cloning.

---

## 🚀 Features

- **Text-to-Music** - Generate complete songs from natural language prompts
- **Voice Conditioning** - Apply artist voice style to generated music
- **Voice Cloning** - Clone any voice for singing synthesis (requires consent)
- **Section-Aware Generation** - Intelligent composition planning (intro, verse, chorus, bridge, outro)
- **Multi-language Lyrics** - Support for 140+ languages via G2P (Grapheme-to-Phoneme)
- **LLM-Enhanced Prompts** - GPT-4o-mini for creative, diverse music descriptions

---

## 📊 Model Sizes

| Config | Parameters | VRAM Required | Use Case |
|--------|-----------|---------------|----------|
| **Dev/Test** | ~1B | 8GB | Local testing |
| **Standard** | ~1.3B | 12GB | Production |
| **Large** | ~3B | 24GB | High quality |
| **XL** | ~6.4B | 48GB+ | Maximum quality |

### Size Configuration

```python
# ~1B (Dev)
unet = UNetV2(model_channels=256)

# ~3B (Production)  
unet = UNetV2(model_channels=512)

# ~6.4B (XL)
unet = UNetV2(model_channels=768)
```

---

## 🏗️ Architecture

```
Text Prompt ──▶ T5/CLAP Encoder ──▶ Text Embedding [768-dim]
                                          │
Voice Sample ──▶ Resemblyzer ────────────┤ (optional)
                 [256-dim]                │
                                          ▼
Noise ──────────▶ UNet Diffusion ──▶ Latent ──▶ VAE Decoder ──▶ Mel ──▶ HiFi-GAN ──▶ Audio
                  (200 steps)       [128-dim]
```

**Key Components:**
- **Audio VAE** - Compresses mel-spectrograms to 128-dim latent space (~224M params)
- **Latent Diffusion** - UNet with Voice Stream Attention (~1-6B params)
- **Composition Planner** - Transformer for song structure planning (~50M params)
- **Text Encoder** - CLAP/T5 with optional LoRA fine-tuning (768-dim)
- **HiFi-GAN Vocoder** - Mel to waveform conversion (32kHz)

---

## ⚡ Quick Start

### Installation

```bash
# System dependencies (required for G2P)
# macOS
brew install espeak-ng ffmpeg

# Linux
apt install espeak-ng ffmpeg libsndfile1

# Python dependencies
pip install -r requirements.txt
```

### Environment Variables

```bash
# Required for LLM prompt enhancement
export OPENAI_API_KEY="sk-..."
```

### Training Pipeline

```bash
# Phase 1: Train VAE
python train_v2.py --phase 1 --audio_dir ./music --epochs 100

# Phase 2: Train Composition Planner  
python train_v2.py --phase 2 --annotations ./data_v2/dataset.json

# Phase 3: Train Latent Diffusion
python train_v2.py --phase 3 --vae_checkpoint ./checkpoints_v2/vae_best.pt
```

### Dataset Building

```bash
# Build dataset with all features (GPU recommended)
python build_dataset_v2.py \
    --audio_dir ./music \
    --device cuda \
    --batch_size 4 \
    --output ./data_v2/dataset.json
```

### Inference

```bash
# Generate music from prompt
python inference_v2.py \
    --prompt "Energetic electronic track with big drops and synth leads" \
    --duration 180 \
    --output ./output/generated.wav

# With artist voice style
python inference_v2.py \
    --prompt "Melancholic hip-hop with deep bass" \
    --artist_style "AWOL" \
    --output ./output/with_style.wav

# With voice cloning (requires reference audio)
python inference_v2.py \
    --prompt "Pop ballad with emotional vocals" \
    --clone_voice_from ./reference_vocal.wav \
    --output ./output/cloned.wav
```

---

## 📁 Project Structure

```
muzible-muze-ai/
├── train_v2.py              # Training pipeline (3 phases)
├── inference_v2.py          # Music generation
├── build_dataset_v2.py      # Dataset builder with GPU batch processing
├── models/
│   ├── audio_vae.py         # Audio VAE (mel ↔ latent)
│   ├── vocoder.py           # HiFi-GAN vocoder
│   └── voice_synthesis.py   # TTS/voice cloning
├── models_v2/
│   ├── latent_diffusion.py  # UNet + Voice Stream Attention
│   ├── composition_planner.py  # Song structure planning
│   ├── text_encoder.py      # CLAP/T5 with LoRA
│   └── lcm_distillation.py  # LCM for fast inference (4 steps)
├── tools_v2/
│   ├── segment_annotator.py # Automatic section detection
│   └── generate_artist_embeddings.py
└── docs_v2/
    └── DOCUMENTATION.md     # Full documentation
```

---

## 🎤 Voice Modes

| Mode | Flag | Description | Legal |
|------|------|-------------|-------|
| **Style Transfer** | `--artist_style NAME` | Voice embedding influences music "vibe" | ✅ Legal |
| **Voice Cloning** | `--clone_voice_from FILE` | Synthesize vocals with cloned voice | ⚠️ Requires consent |

---

## 🔧 Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `latent_dim` | 128 | Latent space dimension |
| `model_channels` | 320 | UNet base width (**main size knob**) |
| `num_timesteps` | 200 | Diffusion steps (v2: reduced from 1000) |
| `sample_rate` | 32000 | Audio sample rate (v2: increased from 22050) |

---

## 📚 Documentation

- [Full Documentation](DOCUMENTATION.md) - Complete technical reference
- [Dataset Builder Guide](docs_v2/DATASET_BUILDER.md) - Dataset creation details

---

## ⚠️ Legal Notice

Voice cloning may infringe on artists' voice rights. Use only with:
- Your own voice
- Explicit consent from the voice owner

---

## 📄 License

GPL-3.0 License

---

*Made with 🎵 by Muzible*
