# 🎵 Dataset Builder v2 - Dokumentacja

> Automatyczny pipeline do budowania datasetów muzycznych

---

## 📑 Spis treści

1. [Przegląd](#przegląd)
2. [Quick Start](#quick-start)
3. [Architektura Pipeline](#architektura-pipeline)
4. [Argumenty CLI](#argumenty-cli)
5. [Ekstrakcja cech](#ekstrakcja-cech)
6. [Format wyjściowy](#format-wyjściowy)
7. [GPU vs CPU](#gpu-vs-cpu)
8. [Checkpoint/Resume](#checkpointresume)
9. [Sharding](#sharding)
10. [Troubleshooting](#troubleshooting)

---

## Przegląd

`build_dataset_v2.py` to kompletny pipeline do przygotowania datasetu treningowego dla Muze AI.

### Automatycznie ekstrahuje:

| Kategoria | Cechy | Model/Narzędzie |
|-----------|-------|-----------------|
| 🎵 Audio | tempo, key, energy, chroma, mel | librosa |
| 🎤 Vocals | separacja, voice embedding | Demucs, Resemblyzer/ECAPA-TDNN |
| 📝 Lyrics | transkrypcja, język, timestamps | Whisper large-v3 |
| 🔤 Phonemes | IPA, per-word | Gruut/espeak-ng |
| 🎼 Segments | verse/chorus/bridge detection | segment_annotator |
| 🧠 Embeddings | CLAP (audio+text) | LAION CLAP |
| 🎸 F0/Pitch | fundamental frequency | f0_extractor |
| 🌊 Vibrato/Breath | singing characteristics | custom |
| 🤖 Prompts | LLM-enhanced descriptions | GPT-4o-mini |

---

## Quick Start

### Minimalny build (test)

```bash
python build_dataset_v2.py \
    --audio_dir ./music/fma_small/000 \
    --output ./data_v2/test_dataset.json \
    --max_tracks 1 \
    --device cpu
```

### Pełny build z GPU

```bash
python build_dataset_v2.py \
    --audio_dir ./music/fma_small \
    --output ./data_v2/full_dataset.json \
    --device cuda \
    --batch_size 4 \
    --checkpoint_dir ./checkpoints \
    --whisper_model large-v3
```

### Szacowanie czasu

```bash
python build_dataset_v2.py \
    --audio_dir ./music \
    --estimate_time \
    --device cuda \
    --batch_size 4
```

---

## Architektura Pipeline

```
┌──────────────────────────────────────────────────────────────────────────┐
│                        build_dataset_v2.py                                │
├──────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐                  │
│  │  Audio File │───▶│  Load Audio │───▶│ Segment     │                  │
│  │  (.mp3)     │    │  (librosa)  │    │ Annotator   │                  │
│  └─────────────┘    └─────────────┘    └──────┬──────┘                  │
│                                               │                          │
│         ┌─────────────────────────────────────┼─────────────────────┐   │
│         │                                     │                     │   │
│         ▼                                     ▼                     ▼   │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌──────────┐ │
│  │   Demucs    │    │  Librosa    │    │   CLAP      │    │  F0/     │ │
│  │  (vocals)   │    │ (features)  │    │ (embed)     │    │ Vibrato  │ │
│  └──────┬──────┘    └──────┬──────┘    └──────┬──────┘    └────┬─────┘ │
│         │                  │                  │                │       │
│         ▼                  │                  │                │       │
│  ┌─────────────┐           │                  │                │       │
│  │  Whisper    │           │                  │                │       │
│  │ (lyrics)    │           │                  │                │       │
│  └──────┬──────┘           │                  │                │       │
│         │                  │                  │                │       │
│         ▼                  │                  │                │       │
│  ┌─────────────┐           │                  │                │       │
│  │   G2P       │           │                  │                │       │
│  │ (phonemes)  │           │                  │                │       │
│  └──────┬──────┘           │                  │                │       │
│         │                  │                  │                │       │
│         └──────────────────┼──────────────────┼────────────────┘       │
│                            │                  │                         │
│                            ▼                  ▼                         │
│                     ┌─────────────────────────────┐                     │
│                     │   Merge Features per       │                     │
│                     │   Segment + LLM Prompt     │                     │
│                     └──────────────┬─────────────┘                     │
│                                    │                                    │
│                                    ▼                                    │
│                           ┌─────────────┐                              │
│                           │  JSON Output│                              │
│                           └─────────────┘                              │
│                                                                          │
└──────────────────────────────────────────────────────────────────────────┘
```

---

## Argumenty CLI

### 📁 Input/Output

| Argument | Default | Opis |
|----------|---------|------|
| `--audio_dir` | **REQUIRED** | Folder z plikami audio |
| `--output` | `./data_v2/training_dataset_v2.json` | Ścieżka wyjściowa JSON |
| `--vocals_output_dir` | `./data_v2/vocals` | Folder na wokale |

### 📋 Metadane

| Argument | Default | Opis |
|----------|---------|------|
| `--tracks_csv` | None | CSV z metadanymi tracków |
| `--genres_csv` | None | CSV z mapowaniem genre_id → nazwa |
| `--metadata_mapping` | None | JSON z ręcznymi metadanymi |

### 🎤 Whisper

| Argument | Default | Opis |
|----------|---------|------|
| `--whisper_model` | `large-v3` | tiny/base/small/medium/large/large-v2/large-v3 |
| `--device` | `cpu` | cpu/cuda |

### 🎹 F0/Pitch

| Argument | Default | Opis |
|----------|---------|------|
| `--pitch_method` | `crepe` | crepe (accurate, default) / pyin (fast fallback) |

> **Note:** CREPE wymaga TensorFlow. Na macOS: `pip install tensorflow-macos`

### 🤖 LLM

| Argument | Default | Opis |
|----------|---------|------|
| `--llm_model` | `gpt-4o-mini` | Model OpenAI |
| `--llm_cache` | `./data_v2/.prompt_cache.json` | Cache promptów |

### ⚙️ Processing

| Argument | Default | Opis |
|----------|---------|------|
| `--max_tracks` | None | Limit tracków |
| `--min_segment` | 4.0 | Min. długość segmentu (s) |
| `--sample_rate` | 22050 | Sample rate |
| `--no_segments` | False | Wyłącz segmentację |
| `--no_features` | False | Minimalna ekstrakcja |

### 🚀 Parallel Processing

| Argument | Default | Opis |
|----------|---------|------|
| `--workers` | cpu_count-2 | CPU workers |
| `--batch_size` | 1 | GPU batch size |
| `--estimate_time` | False | Tylko szacuj czas |

### 💾 Checkpoint/Resume

| Argument | Default | Opis |
|----------|---------|------|
| `--checkpoint_dir` | None | Folder na checkpointy |
| `--run_name` | None | Nazwa runu |
| `--resume_run_id` | None | ID runu do wznowienia |
| `--merge` | False | Tylko merguj checkpointy |
| `--list_runs` | False | Pokaż listę runów |

### 📦 Sharding

| Argument | Default | Opis |
|----------|---------|------|
| `--shard_index` | None | Index shardu (0-based) |
| `--total_shards` | None | Łączna liczba shardów |
| `--shard_by` | `hash` | hash/alphabetical/directory |
| `--merge_shards` | None | Pliki do zmergowania |

---

## Ekstrakcja cech

### Per-Track Features

```json
{
  "track_id": "000010",
  "file_path": "./music/fma_small/000/000010.mp3",
  "artist": "Artist Name",
  "genre": "Rock",
  "duration": 222.4,
  
  "tempo": 95.7,
  "dominant_key": "A",
  "mode": "major",
  
  "voice_embedding": [0.12, -0.34, ...],
  "ecapa_embedding": [-0.21, 0.45, ...],
  "clap_embedding": [0.08, 0.15, ...],
  
  "lyrics": "I woke up this morning...",
  "lyrics_language": "en",
  
  "phonemes": {
    "phonemes_ipa": "aɪ woʊk ʌp ðɪs mɔrnɪŋ",
    "words": [
      {"word": "I", "phonemes": ["aɪ"]},
      {"word": "woke", "phonemes": ["w", "oʊ", "k"]}
    ],
    "backend": "gruut"
  }
}
```

### Per-Segment Features

```json
{
  "segment_idx": 0,
  "start_time": 0.0,
  "end_time": 15.2,
  "section_type": "intro",
  
  "energy": 0.35,
  "spectral_centroid": 2145.3,
  "rms": 0.124,
  
  "chroma": [0.45, 0.12, ...],
  "mfcc": [-12.3, 4.5, ...],
  
  "has_vocals": true,
  "vocal_ratio": 0.78,
  
  "f0_mean": 185.2,
  "f0_contour": [180.1, 182.3, ...],
  
  "vibrato_rate": 5.2,
  "vibrato_extent": 0.8,
  
  "breath_positions": [2.3, 5.1, 8.4],
  
  "segment_lyrics": "I woke up this morning",
  
  "prompt": "Gentle indie rock intro...",
  "prompt_enhanced": "A dreamy indie rock opening..."
}
```

---

## GPU vs CPU

### Szacunkowe czasy

| Hardware | Czas/track | 1000 tracków |
|----------|------------|--------------|
| CPU only | ~70s | ~19h |
| GTX 1080 | ~15s | ~4h |
| RTX 3080 | ~8s | ~2.2h |
| RTX 4090 | ~5s | ~1.4h |

### Rekomendacje VRAM

| VRAM | batch_size | Uwagi |
|------|------------|-------|
| 6GB | 1 | Minimum |
| 8GB | 2 | Bezpieczne |
| 12GB | 3 | Optymalne |
| 16GB+ | 4 | Maksymalne |

### Bottleneck Analysis

```
┌─────────────────────┬────────┬────────┬─────────────┐
│ Component           │ GPU    │ CPU    │ Time/track  │
├─────────────────────┼────────┼────────┼─────────────┤
│ Demucs (vocals)     │ ⭐⭐⭐  │ ⭐     │ 30-60s CPU  │
│ Whisper (lyrics)    │ ⭐⭐⭐  │ ⭐     │ 10-30s CPU  │
│ CLAP embeddings     │ ⭐⭐   │ ⭐     │ 2-5s        │
│ Resemblyzer         │ ⭐     │ ⭐⭐   │ 1-2s        │
│ Librosa features    │ ❌     │ ⭐⭐⭐  │ 2-5s        │
│ LLM (OpenAI API)    │ ❌     │ ❌     │ ~0.5s       │
└─────────────────────┴────────┴────────┴─────────────┘
```

---

## Checkpoint/Resume

### Włączenie

```bash
python build_dataset_v2.py \
    --audio_dir ./music \
    --checkpoint_dir ./checkpoints \
    --run_name "my_build" \
    --output ./dataset.json
```

### Wznowienie

```bash
# Lista runów
python build_dataset_v2.py --checkpoint_dir ./checkpoints --list_runs

# Wznów
python build_dataset_v2.py \
    --audio_dir ./music \
    --checkpoint_dir ./checkpoints \
    --resume_run_id abc123 \
    --output ./dataset.json
```

### Merge

```bash
python build_dataset_v2.py \
    --checkpoint_dir ./checkpoints \
    --merge \
    --output ./dataset.json
```

---

## Sharding

### Multi-machine processing

```bash
# Machine 1
python build_dataset_v2.py \
    --audio_dir ./music \
    --shard_index 0 \
    --total_shards 4 \
    --output ./shard_0.json

# Machine 2
python build_dataset_v2.py \
    --audio_dir ./music \
    --shard_index 1 \
    --total_shards 4 \
    --output ./shard_1.json

# Merge
python build_dataset_v2.py \
    --merge_shards shard_*.json \
    --output ./full_dataset.json
```

---

## Troubleshooting

### ❌ CUDA out of memory

```bash
--batch_size 1
--whisper_model medium
```

### ❌ espeak not found

```bash
# macOS
brew install espeak-ng

# Ubuntu
sudo apt install espeak-ng
```

### ❌ Wolne przetwarzanie

```bash
--estimate_time    # sprawdź szacowany czas
--device cuda      # użyj GPU
--workers 12       # zwiększ paralelizację
```

---

*Dokumentacja: 12 grudnia 2025*
