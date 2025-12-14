"""
🎵 Build Dataset v2 - Full pipeline for creating segmented datasets

Combines:
- Audio feature extraction (from build_dataset.py)
- Automatic segment detection (from segment_annotator.py)
- Per-segment prompt generation
- Metadata from CSV (if available)
- 🎤 Voice embeddings (Resemblyzer/ECAPA-TDNN)
- 📝 Lyrics extraction (Whisper large-v3)
- 💭 Text sentiment analysis
- 🎵 Vocal detection per segment
- 🥁 Beat grid and chord progression
- 🎵 CLAP embeddings (audio-text multimodal)
- 🔤 G2P Phoneme extraction (IPA)
- 🤖 LLM-enhanced prompts (GPT-4o-mini)
- 💾 Checkpoint/resume system

🚀 GPU BATCH PROCESSING:
    Use --batch_size > 1 and --device cuda for much faster processing.
    Batch processing groups GPU operations (Demucs, Whisper, CLAP).
    
    VRAM Recommendations:
    - 8GB VRAM:  --batch_size 2
    - 12GB VRAM: --batch_size 3
    - 16GB+ VRAM: --batch_size 4

Usage:
    # 🚀 FULL POWER (default - everything enabled!)
    python build_dataset_v2.py \\
        --audio_dir ./music/fma_small \\
        --output ./data_v2/dataset.json
    
    # 🖥️ With GPU batch processing (2-3x faster)
    python build_dataset_v2.py \\
        --audio_dir ./music \\
        --device cuda \\
        --batch_size 4 \\
        --checkpoint_dir ./checkpoints \\
        --output ./data_v2/dataset.json
    
    # ⏱️ Check estimated time
    python build_dataset_v2.py \\
        --audio_dir ./music \\
        --estimate_time \\
        --batch_size 4 \\
        --device cuda
    
    # 📋 With manual metadata (when no ID3 tags)
    python build_dataset_v2.py \\
        --audio_dir ./music \\
        --metadata_mapping ./metadata.json \\
        --output ./data_v2/dataset.json
    
    # 💾 With checkpoints (resume on crash)
    python build_dataset_v2.py \\
        --audio_dir ./music \\
        --checkpoint_dir ./checkpoints \\
        --run_name "gpu1_hiphop" \\
        --output ./data_v2/dataset.json
    
    # 🔄 Resume interrupted build
    python build_dataset_v2.py \\
        --audio_dir ./music \\
        --checkpoint_dir ./checkpoints \\
        --resume_run_id abc123 \\
        --output ./data_v2/dataset.json
"""

import os
import sys
import json
import ast
import hashlib
import glob
import time
import threading
from pathlib import Path
from datetime import datetime

# ============================================================================
# OPENAI API KEY - Load from environment variable
# Set via: export OPENAI_API_KEY="sk-..."
# ============================================================================
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    print("⚠️ WARNING: OPENAI_API_KEY not set. LLM prompt enhancement will be disabled.")
    print("   Set it via: export OPENAI_API_KEY=\"sk-...\"")
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum

import numpy as np
import pandas as pd
import librosa
from tqdm import tqdm

# Mutagen for ID3 tags (fallback when no CSV)
try:
    from mutagen.mp3 import MP3
    from mutagen.id3 import ID3
    from mutagen import MutagenError
    HAS_MUTAGEN = True
except ImportError:
    HAS_MUTAGEN = False
    print("⚠️ Mutagen not installed - ID3 tag extraction disabled. Install: pip install mutagen")

# Import segment annotator
sys.path.insert(0, str(Path(__file__).parent))
from tools_v2.segment_annotator import SegmentAnnotator, SectionType

# 🎵 F0 Extractor (v3)
try:
    from tools.f0_extractor import F0Extractor
    HAS_F0_EXTRACTOR = True
except ImportError:
    HAS_F0_EXTRACTOR = False
    print("⚠️ F0Extractor not found - pitch extraction disabled")

# 🔊 Loudness Meter (v3)
try:
    import pyloudnorm as pyln
    HAS_PYLOUDNORM = True
except ImportError:
    HAS_PYLOUDNORM = False
    print("⚠️ pyloudnorm not installed - loudness measurement disabled. Install: pip install pyloudnorm")


# ============================================================================
# LLM Prompt Enhancement (OpenAI)
# ============================================================================

class LLMPromptEnhancer:
    """
    Enhances automatically generated prompts using OpenAI API.
    
    Transforms rigid rule-based prompts into natural, diverse descriptions.
    
    Example:
        Input:  "Energetic hip-hop music with vocals featuring beats at fast tempo in C"
        Output: "Hard-hitting trap banger with aggressive 808s, rapid-fire vocals,
                 and a relentless energy that demands movement"
    """
    
    # Default OpenAI API key
    DEFAULT_API_KEY = "sk-proj-...."
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gpt-4o-mini",  # Cheaper, sufficient for prompts
        batch_size: int = 10,
        cache_file: Optional[str] = None,
    ):
        self.api_key = api_key or self.DEFAULT_API_KEY
        self.model = model
        self.batch_size = batch_size
        self.cache_file = cache_file
        self.cache: Dict[str, str] = {}
        
        # Load cache if exists
        if cache_file and Path(cache_file).exists():
            with open(cache_file, 'r') as f:
                self.cache = json.load(f)
            print(f"   📦 Loaded {len(self.cache)} cached prompts")
        
        self._client = None
    
    @property
    def client(self):
        """Lazy init OpenAI client"""
        if self._client is None:
            try:
                from openai import OpenAI
                self._client = OpenAI(api_key=self.api_key)
                print("   ✅ OpenAI client initialized")
            except ImportError:
                print("   ⚠️ OpenAI not installed: pip install openai")
                return None
            except Exception as e:
                print(f"   ⚠️ OpenAI init failed: {e}")
                return None
        return self._client
    
    def enhance_prompt(
        self,
        base_prompt: str,
        features: Optional[Dict] = None,
        artist: Optional[str] = None,
        genre: Optional[str] = None,
        language: str = "en",
    ) -> str:
        """
        Enhances a single prompt.
        
        Args:
            base_prompt: Rule-based prompt
            features: Audio features dict (tempo, energy, etc)
            artist: Artist name for style reference
            genre: Genre for context
            language: 'en' or 'pl' for output language
            
        Returns:
            Enhanced prompt string
        """
        # Check cache
        cache_key = f"{base_prompt}|{artist}|{genre}|{language}"
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        if self.client is None:
            return base_prompt
        
        # Build context
        context_parts = []
        if features:
            context_parts.append(f"Audio: tempo={features.get('tempo', 120):.0f}bpm, "
                                f"energy={features.get('energy', 0.5):.2f}, "
                                f"key={features.get('dominant_key', 'C')}")
        if artist:
            context_parts.append(f"Artist: {artist}")
        if genre:
            context_parts.append(f"Genre: {genre}")
        
        context = "; ".join(context_parts) if context_parts else ""
        
        # System prompt
        # ALWAYS generate prompts in English!
        # Prompts are for music description - text encoder trained on EN.
        # Lyrics remain in original language (PL/EN/other) for vocal generation.
        system_prompt = """You are a music description expert.
Transform the given description into a natural, rich prompt for music generation.
Keep key info (genre, tempo, mood), but:
- Use creative, descriptive language
- Add musical metaphors and comparisons  
- Avoid repeating the same pattern
- Max 2 sentences, in English
- Do NOT use quotes
- IMPORTANT: Always respond in ENGLISH regardless of input language"""
        
        user_prompt = f"Original: {base_prompt}"
        if context:
            user_prompt += f"\nContext: {context}"
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                max_tokens=150,
                temperature=0.8,
            )
            
            enhanced = response.choices[0].message.content.strip()
            enhanced = enhanced.strip('"\'')  # Remove quotes
            
            # Cache result
            self.cache[cache_key] = enhanced
            
            return enhanced
            
        except Exception as e:
            print(f"   ⚠️ LLM enhancement failed: {e}")
            return base_prompt
    
    def enhance_batch(
        self,
        prompts: List[Dict],  # [{'base_prompt': str, 'features': dict, ...}, ...]
    ) -> List[str]:
        """
        Enhances a batch of prompts (more efficient API usage).
        """
        results = []
        
        for item in tqdm(prompts, desc="Enhancing prompts"):
            enhanced = self.enhance_prompt(
                base_prompt=item.get('base_prompt', ''),
                features=item.get('features'),
                artist=item.get('artist'),
                genre=item.get('genre'),
                language=item.get('language', 'en'),
            )
            results.append(enhanced)
        
        return results
    
    def save_cache(self):
        """Saves cache to file"""
        if self.cache_file and self.cache:
            Path(self.cache_file).parent.mkdir(parents=True, exist_ok=True)
            with open(self.cache_file, 'w') as f:
                json.dump(self.cache, f, indent=2, ensure_ascii=False)
            print(f"   💾 Saved {len(self.cache)} prompts to cache")


# ============================================================================
# Phoneme Processing (G2P - Grapheme to Phoneme)
# ============================================================================

class PhonemeProcessor:
    """
    Text to phoneme (IPA) conversion.
    
    Uses:
    - Gruut for languages: en, de, es, fr, it, ru, cs, ar, fa, nl, sv, sw
    - Phonemizer/espeak-ng for Polish and other unsupported languages
    
    Output format:
    {
        'phonemes_ipa': 'jɛm kanapkɛ s panadɔlɛm',  # Full IPA string
        'words': [
            {'word': 'Jem', 'phonemes': ['j', 'ɛ', 'm']},
            {'word': 'kanapkę', 'phonemes': ['k', 'a', 'n', 'a', 'p', 'k', 'ɛ']},
            ...
        ]
    }
    """
    
    # Languages supported by Gruut
    GRUUT_LANGUAGES = {
        'en': 'en-us', 'en-us': 'en-us', 'en-gb': 'en-gb',
        'de': 'de-de', 'de-de': 'de-de',
        'es': 'es-es', 'es-es': 'es-es',
        'fr': 'fr-fr', 'fr-fr': 'fr-fr',
        'it': 'it-it', 'it-it': 'it-it',
        'ru': 'ru-ru', 'ru-ru': 'ru-ru',
        'cs': 'cs-cz', 'cs-cz': 'cs-cz',
        'nl': 'nl', 'nl-nl': 'nl',
        'sv': 'sv-se', 'sv-se': 'sv-se',
        'ar': 'ar', 'fa': 'fa',
    }
    
    # Languages for Phonemizer/espeak-ng (including Polish!)
    ESPEAK_LANGUAGES = {
        'pl': 'pl', 'polish': 'pl',
        'uk': 'uk', 'ukrainian': 'uk',
        'pt': 'pt', 'pt-br': 'pt-br', 'pt-pt': 'pt-pt',
        'ja': 'ja', 'japanese': 'ja',
        'ko': 'ko', 'korean': 'ko',
        'zh': 'zh', 'chinese': 'zh',
        'tr': 'tr', 'turkish': 'tr',
        'vi': 'vi', 'vietnamese': 'vi',
        'hi': 'hi', 'hindi': 'hi',
        # And many others - espeak supports 100+ languages
    }
    
    def __init__(self):
        self._gruut_available = None
        self._phonemizer_available = None
    
    def _check_gruut(self) -> bool:
        """Lazy check for Gruut"""
        if self._gruut_available is None:
            try:
                from gruut import sentences
                self._gruut_available = True
            except ImportError:
                self._gruut_available = False
        return self._gruut_available
    
    def _check_phonemizer(self) -> bool:
        """Lazy check for Phonemizer"""
        if self._phonemizer_available is None:
            try:
                from phonemizer import phonemize
                from phonemizer.backend import EspeakBackend
                # Check if espeak-ng is installed
                EspeakBackend.version()
                self._phonemizer_available = True
            except Exception:
                self._phonemizer_available = False
        return self._phonemizer_available
    
    def text_to_phonemes(
        self,
        text: str,
        language: str = 'en',
    ) -> Dict[str, Any]:
        """
        Converts text to IPA phonemes.
        
        Args:
            text: Text to convert
            language: Language code (e.g., 'pl', 'en', 'de')
            
        Returns:
            {
                'phonemes_ipa': str,  # Full phoneme string
                'words': [{'word': str, 'phonemes': List[str]}],  # Per-word
                'language': str,
                'backend': str,  # 'gruut' or 'espeak'
            }
        """
        if not text or not text.strip():
            return {
                'phonemes_ipa': '',
                'words': [],
                'language': language,
                'backend': None,
            }
        
        lang_lower = language.lower() if language else 'en'
        
        # Select backend
        if lang_lower in self.GRUUT_LANGUAGES and self._check_gruut():
            return self._phonemize_gruut(text, lang_lower)
        elif self._check_phonemizer():
            return self._phonemize_espeak(text, lang_lower)
        else:
            # Fallback - no G2P available
            return {
                'phonemes_ipa': '',
                'words': [],
                'language': language,
                'backend': None,
                'error': 'No G2P backend available',
            }
    
    def _phonemize_gruut(self, text: str, language: str) -> Dict[str, Any]:
        """Use Gruut for supported languages"""
        from gruut import sentences
        
        gruut_lang = self.GRUUT_LANGUAGES.get(language, 'en-us')
        
        words_data = []
        all_phonemes = []
        
        try:
            for sent in sentences(text, lang=gruut_lang):
                for word in sent:
                    if word.phonemes:
                        words_data.append({
                            'word': word.text,
                            'phonemes': list(word.phonemes),
                        })
                        all_phonemes.extend(word.phonemes)
                        all_phonemes.append(' ')  # Separator between words
            
            phonemes_ipa = ''.join(all_phonemes).strip()
            
            return {
                'phonemes_ipa': phonemes_ipa,
                'words': words_data,
                'language': language,
                'backend': 'gruut',
            }
            
        except Exception as e:
            # Fallback to espeak
            if self._check_phonemizer():
                return self._phonemize_espeak(text, language)
            return {
                'phonemes_ipa': '',
                'words': [],
                'language': language,
                'backend': 'gruut',
                'error': str(e),
            }
    
    def _phonemize_espeak(self, text: str, language: str) -> Dict[str, Any]:
        """Use Phonemizer/espeak-ng (especially for Polish!)"""
        from phonemizer import phonemize
        from phonemizer.separator import Separator
        
        # Map language to espeak code
        espeak_lang = self.ESPEAK_LANGUAGES.get(language, language)
        
        # Check if language is supported
        try:
            from phonemizer.backend import EspeakBackend
            supported = EspeakBackend.supported_languages()
            if espeak_lang not in supported:
                # Try without region
                espeak_lang = espeak_lang.split('-')[0]
                if espeak_lang not in supported:
                    espeak_lang = 'en-us'  # Fallback
        except:
            pass
        
        try:
            # Phonemize entire text
            phonemes_ipa = phonemize(
                text,
                language=espeak_lang,
                backend='espeak',
                strip=True,
                preserve_punctuation=False,
                language_switch='remove-flags',
            )
            
            # Phonemize per-word
            words = text.split()
            words_data = []
            
            for word in words:
                if word.strip():
                    word_phonemes = phonemize(
                        word,
                        language=espeak_lang,
                        backend='espeak',
                        strip=True,
                        preserve_punctuation=False,
                    )
                    # Split into individual phonemes (approximate - espeak returns string)
                    phoneme_list = list(word_phonemes.replace(' ', ''))
                    words_data.append({
                        'word': word,
                        'phonemes': phoneme_list,
                        'phonemes_str': word_phonemes,
                    })
            
            return {
                'phonemes_ipa': phonemes_ipa,
                'words': words_data,
                'language': language,
                'backend': 'espeak',
            }
            
        except Exception as e:
            return {
                'phonemes_ipa': '',
                'words': [],
                'language': language,
                'backend': 'espeak',
                'error': str(e),
            }


# ============================================================================
# 🚀 Parallel Processing Pipeline
# ============================================================================

class ParallelProcessor:
    """
    Orchestrator for parallel audio processing.
    
    Optimization strategy:
    
    📊 BOTTLENECK ANALYSIS:
    ┌─────────────────────┬────────┬────────┬─────────────┐
    │ Component           │ GPU    │ CPU    │ Time/track  │
    ├─────────────────────┼────────┼────────┼─────────────┤
    │ Demucs (vocals)     │ ⭐⭐⭐  │ ⭐     │ 30-60s CPU  │
    │ Whisper (lyrics)    │ ⭐⭐⭐  │ ⭐     │ 10-30s CPU  │
    │ CLAP embeddings     │ ⭐⭐   │ ⭐     │ 2-5s        │
    │ Resemblyzer/SB      │ ⭐     │ ⭐⭐   │ 1-2s        │
    │ Librosa features    │ ❌     │ ⭐⭐⭐  │ 2-5s        │
    │ LLM (OpenAI API)    │ ❌     │ ❌     │ ~0.5s       │
    └─────────────────────┴────────┴────────┴─────────────┘
    
    🎯 STRATEGY:
    1. GPU Queue: Demucs → Whisper → CLAP (sequential on GPU)
    2. CPU Pool: Librosa features (parallel on CPU cores)
    3. Async: OpenAI API calls (non-blocking)
    
    💡 USAGE:
    - With GPU (--device cuda): ~20-40s/track
    - Without GPU (--device cpu): ~3-5min/track
    
    🔧 TUNING:
    - --workers N: number of CPU workers for librosa (default: cpu_count-2)
    - --batch_size N: how many tracks in GPU batch (default: 1, more = more VRAM)
    """
    
    def __init__(
        self,
        device: str = "cpu",
        cpu_workers: Optional[int] = None,
        gpu_batch_size: int = 1,
        prefetch_audio: int = 4,  # How many files to preload
    ):
        import multiprocessing
        
        self.device = device
        self.gpu_batch_size = gpu_batch_size
        self.prefetch_audio = prefetch_audio
        
        # CPU workers - leave 2 cores for system
        max_workers = multiprocessing.cpu_count() - 2
        self.cpu_workers = min(cpu_workers or max_workers, max_workers)
        self.cpu_workers = max(1, self.cpu_workers)
        
        # Statistics
        self.stats = {
            'demucs_time': 0,
            'whisper_time': 0,
            'clap_time': 0,
            'librosa_time': 0,
            'total_tracks': 0,
        }
        
        print(f"\n🚀 ParallelProcessor initialized:")
        print(f"   Device: {device}")
        print(f"   CPU workers: {self.cpu_workers}")
        print(f"   GPU batch size: {gpu_batch_size}")
        
        # Check GPU availability
        if device == "cuda":
            try:
                import torch
                if torch.cuda.is_available():
                    gpu_name = torch.cuda.get_device_name(0)
                    vram = torch.cuda.get_device_properties(0).total_memory / 1e9
                    print(f"   GPU: {gpu_name} ({vram:.1f} GB VRAM)")
                    
                    # Batch size recommendations based on VRAM
                    if vram < 8:
                        print(f"   ⚠️ Low VRAM! Recommend --gpu_batch_size 1")
                    elif vram >= 16:
                        print(f"   💡 High VRAM - you can try --gpu_batch_size 2")
                else:
                    print(f"   ⚠️ CUDA requested but not available! Falling back to CPU")
                    self.device = "cpu"
            except ImportError:
                print(f"   ⚠️ PyTorch not installed, using CPU")
                self.device = "cpu"
    
    def estimate_time(self, num_tracks: int, avg_duration: float = 180) -> Dict[str, Any]:
        """
        Estimates processing time based on hardware.
        
        Returns:
            {
                'estimated_minutes': float,
                'estimated_hours': float,
                'per_track_seconds': float,
                'bottleneck': str,
            }
        """
        if self.device == "cuda":
            # GPU estimates (RTX 3080 level)
            demucs_per_track = 5  # seconds
            whisper_per_track = 3
            other_per_track = 2
            
            # Batch processing speedup
            # Batch > 1 gives ~1.5-2x speedup (transfer overhead amortization)
            batch_factor = 1.0
            if self.gpu_batch_size > 1:
                batch_factor = 1.0 / (1 + 0.3 * (self.gpu_batch_size - 1))
            
            per_track = (demucs_per_track + whisper_per_track + other_per_track) * batch_factor
        else:
            # CPU estimates
            demucs_per_track = 45  # seconds
            whisper_per_track = 20
            other_per_track = 5
            per_track = demucs_per_track + whisper_per_track + other_per_track
        
        # Account for CPU parallelization for librosa
        librosa_factor = 1.0 / self.cpu_workers
        per_track += 3 * librosa_factor  # ~3s librosa per track
        
        total_seconds = per_track * num_tracks
        
        return {
            'estimated_minutes': total_seconds / 60,
            'estimated_hours': total_seconds / 3600,
            'per_track_seconds': per_track,
            'bottleneck': 'Demucs (vocal separation)' if self.device == "cpu" else 'GPU pipeline',
            'device': self.device,
            'recommendation': self._get_recommendation(num_tracks, total_seconds),
        }
    
    def _get_recommendation(self, num_tracks: int, total_seconds: float) -> str:
        """Generates recommendation for user"""
        hours = total_seconds / 3600
        
        if self.device == "cpu" and hours > 24:
            return (
                f"⚠️ Estimated {hours:.1f}h on CPU! "
                f"Consider:\n"
                f"   1. Use GPU (--device cuda) - 10-20x faster\n"
                f"   2. Rent GPU cloud (vast.ai ~$0.30/h)\n"
                f"   3. Split into parts (--max_tracks {num_tracks//4})"
            )
        elif self.device == "cuda" and hours > 12:
            return (
                f"ℹ️ Estimated {hours:.1f}h on GPU. "
                f"Consider running in background:\n"
                f"   nohup python build_dataset_v2.py ... > build.log 2>&1 &"
            )
        elif hours < 1:
            return f"✅ Quick build (~{total_seconds/60:.0f} min)"
        else:
            return f"✅ Manageable ({hours:.1f}h)"
    
    def print_estimate(self, num_tracks: int):
        """Prints estimated time before starting"""
        est = self.estimate_time(num_tracks)
        
        print(f"\n⏱️  Time estimate for {num_tracks} tracks:")
        print(f"   Per track: ~{est['per_track_seconds']:.1f}s")
        print(f"   Total: ~{est['estimated_hours']:.1f}h ({est['estimated_minutes']:.0f} min)")
        print(f"   Bottleneck: {est['bottleneck']}")
        print(f"   {est['recommendation']}")
        print()


class BatchGPUProcessor:
    """
    Batch processing for GPU-heavy operations.
    
    Instead of processing one track at a time, we group:
    1. Loading audio (parallel CPU)
    2. Batch Demucs (GPU) - vocal separation for multiple tracks
    3. Batch Whisper (GPU) - transcription for multiple tracks
    4. Batch CLAP (GPU) - embeddings for multiple tracks
    5. CPU features (parallel) - librosa in background
    
    ┌─────────────────────────────────────────────────────────┐
    │  PIPELINE (batch_size=4)                                │
    ├─────────────────────────────────────────────────────────┤
    │  Load: [A1, A2, A3, A4]  ←── CPU parallel               │
    │           ↓                                              │
    │  Demucs: [A1, A2, A3, A4] → [V1, V2, V3, V4]  ←── GPU   │
    │           ↓                                              │
    │  Whisper: [V1, V2, V3, V4] → [L1, L2, L3, L4] ←── GPU   │
    │           ↓                                              │
    │  CLAP: [A1, A2, A3, A4] → [E1, E2, E3, E4]    ←── GPU   │
    │           ↓                                              │
    │  Librosa: [A1, A2, A3, A4] → features         ←── CPU   │
    └─────────────────────────────────────────────────────────┘
    
    Speedup:
    - Sequential: 4 tracks × 60s = 240s
    - Batched: ~90s (1.5-3x faster depending on VRAM)
    """
    
    def __init__(
        self,
        device: str = "cuda",
        batch_size: int = 4,
        sample_rate: int = 22050,
        whisper_model: str = "large-v3",
    ):
        self.device = device
        self.batch_size = batch_size
        self.sample_rate = sample_rate
        self.whisper_model_name = whisper_model
        
        # Lazy-loaded models
        self._demucs_model = None
        self._whisper_model = None
        self._clap_model = None
        self._clap_processor = None
        
        # Stats
        self.stats = {
            'batches_processed': 0,
            'tracks_processed': 0,
            'demucs_time': 0,
            'whisper_time': 0,
            'clap_time': 0,
        }
        
        print(f"\n🚀 BatchGPUProcessor initialized:")
        print(f"   Device: {device}")
        print(f"   Batch size: {batch_size}")
        print(f"   Whisper model: {whisper_model}")
        
        # VRAM check
        if device == "cuda":
            try:
                import torch
                if torch.cuda.is_available():
                    vram = torch.cuda.get_device_properties(0).total_memory / 1e9
                    # Estimated VRAM usage per batch
                    # Demucs: ~2GB base + ~0.5GB per track
                    # Whisper large-v3: ~3GB
                    # CLAP: ~1GB
                    estimated_vram = 2 + (0.5 * batch_size) + 3 + 1
                    
                    if estimated_vram > vram * 0.9:
                        recommended_batch = max(1, int((vram * 0.9 - 6) / 0.5))
                        print(f"   ⚠️ VRAM warning: {vram:.1f}GB available")
                        print(f"      Estimated usage: ~{estimated_vram:.1f}GB for batch_size={batch_size}")
                        print(f"      Recommended: --batch_size {recommended_batch}")
            except Exception:
                pass
    
    @property
    def demucs_model(self):
        """Lazy load Demucs model"""
        if self._demucs_model is None:
            try:
                import torch
                from demucs import pretrained
                print("   📦 Loading Demucs model...")
                self._demucs_model = pretrained.get_model('htdemucs')
                self._demucs_model.to(self.device)
                self._demucs_model.eval()
                print("   ✅ Demucs loaded")
            except ImportError:
                print("   ⚠️ Demucs not installed")
                return None
        return self._demucs_model
    
    @property
    def whisper_model(self):
        """Lazy load Whisper model"""
        if self._whisper_model is None:
            try:
                import whisper
                print(f"   📦 Loading Whisper {self.whisper_model_name}...")
                self._whisper_model = whisper.load_model(
                    self.whisper_model_name, 
                    device=self.device
                )
                print(f"   ✅ Whisper {self.whisper_model_name} loaded")
            except ImportError:
                print("   ⚠️ Whisper not installed")
                return None
        return self._whisper_model
    
    @property
    def clap_model(self):
        """Lazy load CLAP model"""
        if self._clap_model is None:
            try:
                from transformers import ClapModel, ClapProcessor
                print("   📦 Loading CLAP model...")
                model_id = "laion/clap-htsat-unfused"
                self._clap_processor = ClapProcessor.from_pretrained(model_id)
                self._clap_model = ClapModel.from_pretrained(model_id)
                self._clap_model.to(self.device)
                self._clap_model.eval()
                print("   ✅ CLAP loaded")
            except ImportError:
                print("   ⚠️ CLAP not installed")
                return None
        return self._clap_model
    
    def batch_separate_vocals(
        self,
        audio_batch: List[Tuple[np.ndarray, int]],  # [(audio, sr), ...]
    ) -> List[Optional[np.ndarray]]:
        """
        Batch vocal separation z Demucs.
        
        Args:
            audio_batch: List of tuples (audio_array, sample_rate)
            
        Returns:
            List of separated vocals (or None for errors)
        """
        import time
        start = time.time()
        
        if self.demucs_model is None:
            return [None] * len(audio_batch)
        
        try:
            import torch
            import torchaudio
            from demucs.apply import apply_model
            
            results = []
            model_sr = self.demucs_model.samplerate
            
            # Przygotuj tensory
            tensors = []
            for audio, sr in audio_batch:
                # Convert to torch
                t = torch.tensor(audio).unsqueeze(0).unsqueeze(0)  # [1, 1, samples]
                if t.shape[1] == 1:
                    t = t.repeat(1, 2, 1)  # Stereo
                
                # Resample if needed
                if sr != model_sr:
                    t = torchaudio.functional.resample(t, sr, model_sr)
                
                tensors.append(t)
            
            # Process batch
            # Demucs doesn't natively support batch, so we iterate
            # but keep model on GPU (amortize transfer overhead)
            for i, tensor in enumerate(tensors):
                try:
                    with torch.no_grad():
                        sources = apply_model(
                            self.demucs_model, 
                            tensor.to(self.device),
                            progress=False,
                        )
                    
                    # Vocals is index 3 in htdemucs
                    vocals = sources[0, 3].cpu().numpy()
                    vocals_mono = np.mean(vocals, axis=0)
                    
                    # Resample back
                    original_sr = audio_batch[i][1]
                    if original_sr != model_sr:
                        vocals_mono = librosa.resample(
                            vocals_mono, 
                            orig_sr=model_sr, 
                            target_sr=original_sr
                        )
                    
                    results.append(vocals_mono)
                except Exception as e:
                    print(f"      ⚠️ Demucs error for track {i}: {e}")
                    results.append(None)
            
            self.stats['demucs_time'] += time.time() - start
            return results
            
        except Exception as e:
            print(f"   ❌ Batch Demucs error: {e}")
            return [None] * len(audio_batch)
    
    def batch_transcribe(
        self,
        vocals_batch: List[Optional[np.ndarray]],
        sample_rate: int = 22050,
        contexts: Optional[List[Dict]] = None,  # [{'artist': ..., 'genres': [...]}, ...]
    ) -> List[Dict]:
        """
        Batch lyrics transcription with Whisper.
        
        Args:
            vocals_batch: List of separated vocals
            sample_rate: Sample rate
            contexts: Optional contexts (artist, genres) for each track
            
        Returns:
            List of transcription results
        """
        import time
        start = time.time()
        
        if self.whisper_model is None:
            return [{'text': '', 'language': 'en', 'segments': []}] * len(vocals_batch)
        
        results = []
        contexts = contexts or [{}] * len(vocals_batch)
        
        for i, (vocals, ctx) in enumerate(zip(vocals_batch, contexts)):
            if vocals is None or np.max(np.abs(vocals)) < 0.01:
                results.append({'text': '', 'language': 'en', 'segments': []})
                continue
            
            try:
                # Build initial prompt for better accuracy
                initial_prompt = self._build_whisper_prompt(ctx)
                
                # Transcribe
                result = self.whisper_model.transcribe(
                    vocals.astype(np.float32),
                    language=None,  # Auto-detect
                    task="transcribe",
                    initial_prompt=initial_prompt,
                    verbose=False,
                )
                
                results.append({
                    'text': result.get('text', ''),
                    'language': result.get('language', 'en'),
                    'segments': result.get('segments', []),
                })
            except Exception as e:
                print(f"      ⚠️ Whisper error for track {i}: {e}")
                results.append({'text': '', 'language': 'en', 'segments': []})
        
        self.stats['whisper_time'] += time.time() - start
        return results
    
    def _build_whisper_prompt(self, ctx: Dict) -> str:
        """Buduje initial_prompt dla Whisper na podstawie kontekstu"""
        parts = []
        
        artist = ctx.get('artist')
        genres = ctx.get('genres', [])
        
        if artist:
            parts.append(f"Artist: {artist}.")
        
        if genres:
            genre_str = ', '.join(genres[:2])
            parts.append(f"Genre: {genre_str}.")
        
        # Add Polish words if probably Polish
        if artist and any(pl in artist.lower() for pl in ['zeus', 'taco', 'sokół', 'pezet', 'quebonafide']):
            parts.append("Tekst po polsku.")
        
        return ' '.join(parts) if parts else ""
    
    def batch_clap_embeddings(
        self,
        audio_batch: List[Tuple[np.ndarray, int]],
        texts: Optional[List[str]] = None,
    ) -> Tuple[List[Optional[np.ndarray]], List[Optional[np.ndarray]]]:
        """
        Batch CLAP embeddings.
        
        Args:
            audio_batch: Lista krotek (audio, sr)
            texts: Opcjonalne teksty (prompty) do embeddingu
            
        Returns:
            (audio_embeddings, text_embeddings)
        """
        import time
        start = time.time()
        
        if self.clap_model is None:
            n = len(audio_batch)
            return [None] * n, [None] * n
        
        try:
            import torch
            
            audio_embeddings = []
            text_embeddings = []
            
            # Audio embeddings
            for audio, sr in audio_batch:
                try:
                    # CLAP expects 48kHz
                    if sr != 48000:
                        audio_48k = librosa.resample(audio, orig_sr=sr, target_sr=48000)
                    else:
                        audio_48k = audio
                    
                    inputs = self._clap_processor(
                        audios=[audio_48k],
                        return_tensors="pt",
                        sampling_rate=48000,
                    )
                    inputs = {k: v.to(self.device) for k, v in inputs.items()}
                    
                    with torch.no_grad():
                        emb = self.clap_model.get_audio_features(**inputs)
                    
                    audio_embeddings.append(emb[0].cpu().numpy())
                except Exception as e:
                    audio_embeddings.append(None)
            
            # Text embeddings (if provided)
            if texts:
                for text in texts:
                    if not text:
                        text_embeddings.append(None)
                        continue
                    try:
                        inputs = self._clap_processor(
                            text=[text],
                            return_tensors="pt",
                            padding=True,
                        )
                        inputs = {k: v.to(self.device) for k, v in inputs.items()}
                        
                        with torch.no_grad():
                            emb = self.clap_model.get_text_features(**inputs)
                        
                        text_embeddings.append(emb[0].cpu().numpy())
                    except Exception:
                        text_embeddings.append(None)
            else:
                text_embeddings = [None] * len(audio_batch)
            
            self.stats['clap_time'] += time.time() - start
            return audio_embeddings, text_embeddings
            
        except Exception as e:
            print(f"   ❌ Batch CLAP error: {e}")
            n = len(audio_batch)
            return [None] * n, [None] * n
    
    def process_batch(
        self,
        file_paths: List[Path],
        metadata_list: Optional[List[Dict]] = None,
    ) -> List[Dict]:
        """
        Processes batch of audio files through GPU pipeline.
        
        Args:
            file_paths: List of paths to audio files
            metadata_list: Optional metadata for each file
            
        Returns:
            List of dictionaries with GPU processing results:
            [
                {
                    'vocals': np.ndarray lub None,
                    'lyrics': {'text': str, 'language': str, 'segments': [...]},
                    'clap_audio': np.ndarray (512-dim) lub None,
                    'clap_text': np.ndarray (512-dim) lub None,
                },
                ...
            ]
        """
        import time
        batch_start = time.time()
        
        print(f"\n   🔄 Processing batch of {len(file_paths)} tracks...")
        
        # 1. Load audio (parallel CPU)
        print(f"      📂 Loading audio...")
        audio_batch = []
        for fp in file_paths:
            try:
                y, sr = librosa.load(str(fp), sr=self.sample_rate)
                audio_batch.append((y, sr))
            except Exception as e:
                print(f"         ⚠️ Failed to load {fp.name}: {e}")
                audio_batch.append((np.zeros(self.sample_rate * 10), self.sample_rate))
        
        # 2. Batch Demucs (vocal separation)
        print(f"      🎤 Separating vocals (Demucs)...")
        vocals_list = self.batch_separate_vocals(audio_batch)
        
        # 3. Batch Whisper (lyrics)
        print(f"      📝 Transcribing lyrics (Whisper)...")
        contexts = metadata_list or [{}] * len(file_paths)
        lyrics_list = self.batch_transcribe(vocals_list, self.sample_rate, contexts)
        
        # 4. Batch CLAP embeddings
        print(f"      🎵 Computing CLAP embeddings...")
        # Use lyrics as text for embedding (or placeholder)
        texts = [l.get('text', '')[:200] for l in lyrics_list]  # Max 200 chars
        clap_audio, clap_text = self.batch_clap_embeddings(audio_batch, texts)
        
        # Collect results
        results = []
        for i in range(len(file_paths)):
            results.append({
                'audio': audio_batch[i][0],
                'vocals': vocals_list[i],
                'lyrics': lyrics_list[i],
                'clap_audio': clap_audio[i],
                'clap_text': clap_text[i],
            })
        
        batch_time = time.time() - batch_start
        self.stats['batches_processed'] += 1
        self.stats['tracks_processed'] += len(file_paths)
        
        print(f"      ✅ Batch done in {batch_time:.1f}s ({batch_time/len(file_paths):.1f}s/track)")
        
        return results


# ============================================================================
# Voice & Lyrics Processing
# ============================================================================

class VocalProcessor:
    """
    Processes vocals: detection, embeddings, lyrics, sentiment
    
    Extracts TWO types of embeddings:
    
    1. voice_embedding (from mix) → for "in style of X" (style transfer)
       - Resemblyzer 256-dim
       - Used for averaging per-artist
    
    2. voice_embedding_separated (from Demucs) → for "like X" (voice cloning)
       - SpeechBrain ECAPA-TDNN 192-dim (more accurate)
       - Used when we want to exactly reproduce voice
    
    Mode selection happens in INFERENCE, not here!
    """
    
    def __init__(
        self,
        sample_rate: int = 22050,
        use_demucs: bool = False,
        whisper_model: str = "large-v3",  # Default best model!
        device: str = "cpu",
    ):
        self.sample_rate = sample_rate
        self.device = device
        self.use_demucs = use_demucs
        
        # Lazy loading - inicjalizuj tylko gdy potrzebne
        self._resemblyzer_encoder = None
        self._speechbrain_encoder = None
        self._whisper_model = None
        self._whisper_model_name = whisper_model
        self._demucs_model = None
        self._sentiment_analyzer = None
        self._lyrics_analyzer = None
        
        # G2P Phoneme processor
        self._phoneme_processor = PhonemeProcessor()
    
    @property
    def resemblyzer_encoder(self):
        """Lazy load Resemblyzer (256-dim)"""
        if self._resemblyzer_encoder is None:
            try:
                from resemblyzer import VoiceEncoder
                self._resemblyzer_encoder = VoiceEncoder(device=self.device)
                print("   ✅ Resemblyzer loaded (256-dim)")
            except ImportError:
                print("   ⚠️ Resemblyzer not installed: pip install resemblyzer")
                return None
        return self._resemblyzer_encoder
    
    @property
    def speechbrain_encoder(self):
        """Lazy load SpeechBrain ECAPA-TDNN (192-dim) - lepszy dla voice cloning"""
        if self._speechbrain_encoder is None:
            try:
                from speechbrain.inference.speaker import EncoderClassifier
                self._speechbrain_encoder = EncoderClassifier.from_hparams(
                    source="speechbrain/spkrec-ecapa-voxceleb",
                    savedir="pretrained_models/spkrec-ecapa-voxceleb",
                    run_opts={"device": self.device}
                )
                print("   ✅ SpeechBrain ECAPA-TDNN loaded (192-dim)")
            except ImportError:
                print("   ⚠️ SpeechBrain not installed: pip install speechbrain")
                return None
        return self._speechbrain_encoder
    
    @property
    def whisper_model(self):
        """Lazy load Whisper"""
        if self._whisper_model is None:
            try:
                import whisper
                self._whisper_model = whisper.load_model(self._whisper_model_name)
                print(f"   ✅ Whisper ({self._whisper_model_name}) loaded")
            except ImportError:
                print("   ⚠️ Whisper not installed: pip install openai-whisper")
                return None
        return self._whisper_model
    
    @property
    def sentiment_analyzer(self):
        """Lazy load sentiment analyzer"""
        if self._sentiment_analyzer is None:
            try:
                from transformers import pipeline
                self._sentiment_analyzer = pipeline(
                    "sentiment-analysis",
                    model="cardiffnlp/twitter-roberta-base-sentiment-latest",
                    device=0 if self.device == "cuda" else -1,
                )
                print("   ✅ Sentiment analyzer loaded")
            except ImportError:
                print("   ⚠️ Transformers not installed for sentiment")
                return None
        return self._sentiment_analyzer
    
    def detect_vocals(self, y: np.ndarray, sr: int) -> Tuple[bool, float]:
        """
        Detects if audio contains vocals.
        
        Uses combination of:
        - Spectral flatness (vocals have lower)
        - Harmonic ratio
        - Spectral contrast in vocal range (300-3000 Hz)
        
        Returns:
            (has_vocals, vocal_confidence)
        """
        try:
            # Harmonic-percussive separation
            y_harmonic, y_percussive = librosa.effects.hpss(y)
            
            # Spectral flatness - vocals are less "flat"
            flatness = librosa.feature.spectral_flatness(y=y_harmonic)
            avg_flatness = np.mean(flatness)
            
            # Spectral contrast w zakresie wokalnym
            contrast = librosa.feature.spectral_contrast(y=y_harmonic, sr=sr)
            vocal_band_contrast = np.mean(contrast[2:5])  # ~300-3000 Hz
            
            # Harmonic ratio
            harmonic_energy = np.sum(y_harmonic ** 2)
            total_energy = np.sum(y ** 2) + 1e-10
            harmonic_ratio = harmonic_energy / total_energy
            
            # Heuristics
            # Lower flatness + higher vocal contrast + high harmonic ratio = vocals
            vocal_score = (
                (1 - avg_flatness) * 0.3 +
                (vocal_band_contrast / 50) * 0.4 +  # Normalize
                harmonic_ratio * 0.3
            )
            
            has_vocals = vocal_score > 0.5
            confidence = min(1.0, max(0.0, vocal_score))
            
            return has_vocals, confidence
            
        except Exception as e:
            return False, 0.0
    
    def separate_vocals(self, y: np.ndarray, sr: int) -> Optional[np.ndarray]:
        """
        Separates vocals from mix using Demucs.
        
        Returns:
            Vocal track as numpy array or None
        """
        if not self.use_demucs:
            return None
        
        try:
            import torch
            import torchaudio
            from demucs import pretrained
            from demucs.apply import apply_model
            
            if self._demucs_model is None:
                self._demucs_model = pretrained.get_model('htdemucs')
                self._demucs_model.to(self.device)
                print("   ✅ Demucs loaded")
            
            # Convert to torch
            audio = torch.tensor(y).unsqueeze(0).unsqueeze(0)  # [1, 1, samples]
            if audio.shape[1] == 1:
                audio = audio.repeat(1, 2, 1)  # Stereo
            
            # Resample if needed
            if sr != self._demucs_model.samplerate:
                audio = torchaudio.functional.resample(
                    audio, sr, self._demucs_model.samplerate
                )
            
            # Separate
            with torch.no_grad():
                sources = apply_model(self._demucs_model, audio.to(self.device))
            
            # Vocals is usually index 3 in htdemucs
            vocals = sources[0, 3].cpu().numpy()
            vocals_mono = np.mean(vocals, axis=0)
            
            # Resample back
            if sr != self._demucs_model.samplerate:
                vocals_mono = librosa.resample(
                    vocals_mono, 
                    orig_sr=self._demucs_model.samplerate, 
                    target_sr=sr
                )
            
            return vocals_mono
            
        except Exception as e:
            print(f"   ⚠️ Vocal separation failed: {e}")
            return None
    
    def extract_all_embeddings(
        self,
        y: np.ndarray,
        sr: int,
    ) -> Dict[str, Any]:
        """
        Extracts BOTH types of embeddings for later use in inference.
        
        Args:
            y: Audio signal (full mix)
            sr: Sample rate
            
        Returns:
            {
                'embedding_mix': np.ndarray or None,  # From mix (for style_of)
                'embedding_separated': np.ndarray or None,  # From vocals (for voice_clone)
                'backend': str,  # 'resemblyzer', 'speechbrain'
                'embedding_dim': int,  # 256 or 192
                'separation_method': str,  # 'demucs', 'none'
                'separated_vocals': np.ndarray or None,  # Clean vocals to save
            }
        """
        result = {
            'embedding_mix': None,
            'embedding_separated': None,
            'backend': 'resemblyzer',
            'embedding_dim': 256,
            'separation_method': 'none',
            'separated_vocals': None,  # NEW: raw vocals audio
        }
        
        # 1. EMBEDDING FROM MIX (for style_of)
        # We always use Resemblyzer - sufficient for style transfer
        emb_mix = self._extract_resemblyzer_raw(y, sr)
        if emb_mix is not None:
            result['embedding_mix'] = emb_mix
            result['embedding_dim'] = 256
        
        # 2. EMBEDDING FROM SEPARATED VOCALS (for voice_clone)
        # Only if we have Demucs enabled
        if self.use_demucs:
            separated = self.separate_vocals(y, sr)
            if separated is not None and np.max(np.abs(separated)) > 0.01:
                result['separation_method'] = 'demucs'
                result['separated_vocals'] = separated  # NEW: save raw vocals
                
                # Try SpeechBrain (192-dim, more accurate)
                emb_sep = self._extract_speechbrain_raw(separated, sr)
                if emb_sep is not None:
                    result['embedding_separated'] = emb_sep
                    result['backend'] = 'speechbrain'
                else:
                    # Fallback do Resemblyzer
                    emb_sep = self._extract_resemblyzer_raw(separated, sr)
                    if emb_sep is not None:
                        result['embedding_separated'] = emb_sep
        
        return result
    
    def _extract_resemblyzer_raw(self, audio: np.ndarray, sr: int) -> Optional[np.ndarray]:
        """Extraction with Resemblyzer (256-dim) - only embedding"""
        if self.resemblyzer_encoder is None:
            return None
        
        try:
            # Resemblyzer requires 16kHz
            if sr != 16000:
                audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
                sr = 16000
            
            # Normalize to float32 [-1, 1]
            audio = audio.astype(np.float32)
            audio = audio / (np.max(np.abs(audio)) + 1e-10)
            
            # Resemblyzer's embed_utterance can take raw numpy array directly
            # It expects float32 array at 16kHz
            embedding = self.resemblyzer_encoder.embed_utterance(audio)
            return embedding  # [256]
            
        except Exception as e:
            print(f"   ⚠️ Resemblyzer embedding failed: {e}")
            return None
    
    def _extract_speechbrain_raw(self, audio: np.ndarray, sr: int) -> Optional[np.ndarray]:
        """Extraction with SpeechBrain ECAPA-TDNN (192-dim) - only embedding"""
        if self.speechbrain_encoder is None:
            return None
        
        try:
            import torch
            
            # SpeechBrain requires 16kHz
            if sr != 16000:
                audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
            
            # Convert to torch tensor
            signal = torch.tensor(audio).unsqueeze(0).float()
            
            # Extract embedding
            with torch.no_grad():
                embedding = self.speechbrain_encoder.encode_batch(signal)
            
            return embedding.squeeze().cpu().numpy()  # [192]
            
        except Exception as e:
            print(f"   ⚠️ SpeechBrain embedding failed: {e}")
            return None
    
    # Legacy methods for compatibility
    def extract_voice_embedding(
        self,
        y: np.ndarray,
        sr: int,
        use_separated: bool = True,
    ) -> Tuple[Optional[np.ndarray], str, str]:
        """
        [LEGACY] Extracts single voice embedding.
        Prefer extract_all_embeddings() for new code.
        """
        result = self.extract_all_embeddings(y, sr)
        
        if use_separated and result['embedding_separated'] is not None:
            return result['embedding_separated'], result['backend'], "speaker"
        elif result['embedding_mix'] is not None:
            return result['embedding_mix'], "resemblyzer", "speaker"
        else:
            return None, "", "audio"
    
    def _extract_resemblyzer(self, audio: np.ndarray, sr: int) -> Tuple[Optional[np.ndarray], str, str]:
        """[LEGACY] Extraction with Resemblyzer (256-dim)"""
        emb = self._extract_resemblyzer_raw(audio, sr)
        if emb is not None:
            return emb, "resemblyzer", "speaker"
        return None, "", "audio"
    
    def _extract_speechbrain(self, audio: np.ndarray, sr: int) -> Tuple[Optional[np.ndarray], str, str]:
        """[LEGACY] Extraction with SpeechBrain ECAPA-TDNN (192-dim)"""
        emb = self._extract_speechbrain_raw(audio, sr)
        if emb is not None:
            return emb, "speechbrain", "speaker"
        # Fallback to Resemblyzer
        return self._extract_resemblyzer(audio, sr)
    
    def transcribe_lyrics(
        self,
        y: np.ndarray,
        sr: int,
        language: Optional[str] = None,
        artist: Optional[str] = None,
        genres: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Transcribes lyrics using Whisper large-v3 with intelligent prompt.
        
        Args:
            y: Audio numpy array
            sr: Sample rate
            language: Optional language code (e.g. 'pl', 'en')
            artist: Artist name (helps with context)
            genres: List of genres (helps determine style)
        
        Returns:
            {
                'text': str,  # Full text
                'segments': [{'start': float, 'end': float, 'text': str}],
                'language': str,
                'confidence': float,
            }
        """
        if self.whisper_model is None:
            return {'text': '', 'segments': [], 'language': None, 'confidence': 0.0}
        
        try:
            # Whisper requires 16kHz
            if sr != 16000:
                audio = librosa.resample(y, orig_sr=sr, target_sr=16000)
            else:
                audio = y
            
            # Normalize
            audio = audio.astype(np.float32)
            audio = audio / (np.max(np.abs(audio)) + 1e-10)
            
            # Build intelligent initial_prompt for better transcription
            initial_prompt = self._build_transcription_prompt(artist, genres)
            
            # Transcribe with initial_prompt
            result = self.whisper_model.transcribe(
                audio,
                language=language,
                task="transcribe",
                initial_prompt=initial_prompt,
                # Additional options for better quality
                condition_on_previous_text=True,  # Context between segments
                no_speech_threshold=0.5,  # Less false positives
                logprob_threshold=-1.0,  # Accept more words
            )
            
            return {
                'text': result['text'].strip(),
                'segments': [
                    {
                        'start': seg['start'],
                        'end': seg['end'],
                        'text': seg['text'].strip(),
                    }
                    for seg in result.get('segments', [])
                ],
                'language': result.get('language', 'unknown'),
                'confidence': 1.0 - result.get('no_speech_prob', 0.0),
            }
            
        except Exception as e:
            print(f"   ⚠️ Transcription failed: {e}")
            return {'text': '', 'segments': [], 'language': None, 'confidence': 0.0}
    
    def _build_transcription_prompt(
        self,
        artist: Optional[str] = None,
        genres: Optional[List[str]] = None,
    ) -> str:
        """
        Builds intelligent prompt for Whisper based on context.
        
        Whisper uses initial_prompt as "previous context" - helps
        the model understand style, language and vocabulary.
        """
        parts = []
        
        # Detect if this is Polish rap/hip-hop
        genre_str = " ".join(genres or []).lower()
        is_rap = any(g in genre_str for g in ['rap', 'hip-hop', 'hip hop', 'trap'])
        is_polish = any(g in genre_str for g in ['polish', 'polski', 'pl'])
        
        if is_polish or is_rap:
            # Polish rap - special vocabulary
            parts.append("Polski rap hip-hop. Tekst po polsku.")
            parts.append("Slang uliczny: hajs, ziom, elo, spoko, git, nara, mordo, beka, ogar.")
            parts.append("Flow z rymami, szybkie wersy.")
        
        if artist:
            parts.append(f"Artysta: {artist}.")
        
        if genres:
            parts.append(f"Gatunek: {', '.join(genres[:3])}.")
        
        # General hints
        if not parts:
            parts.append("Tekst piosenki muzycznej. Lyrics z rymami i powtórzeniami.")
        
        return " ".join(parts)
    
    def analyze_sentiment(self, text: str) -> Dict[str, Any]:
        """
        Analyzes sentiment of lyrics text.
        
        Returns:
            {
                'label': str,  # positive/negative/neutral
                'score': float,  # 0-1
                'emotions': {emotion: score},  # if available
            }
        """
        if not text or len(text.strip()) < 10:
            return {'label': 'neutral', 'score': 0.5, 'emotions': {}}
        
        if self.sentiment_analyzer is None:
            # Fallback: simple keyword-based
            return self._simple_sentiment(text)
        
        try:
            result = self.sentiment_analyzer(text[:512])[0]  # Max 512 tokens
            
            return {
                'label': result['label'].lower(),
                'score': result['score'],
                'emotions': {},
            }
            
        except Exception as e:
            return self._simple_sentiment(text)
    
    def _simple_sentiment(self, text: str) -> Dict[str, Any]:
        """Simple fallback for sentiment analysis"""
        text_lower = text.lower()
        
        positive_words = ['love', 'happy', 'joy', 'beautiful', 'amazing', 'wonderful', 
                         'great', 'good', 'best', 'sunshine', 'smile', 'dance', 'party']
        negative_words = ['sad', 'pain', 'hurt', 'cry', 'alone', 'dark', 'lost',
                         'broken', 'hate', 'fear', 'die', 'death', 'gone', 'tears']
        
        pos_count = sum(1 for w in positive_words if w in text_lower)
        neg_count = sum(1 for w in negative_words if w in text_lower)
        
        if pos_count > neg_count:
            return {'label': 'positive', 'score': min(0.9, 0.5 + pos_count * 0.1), 'emotions': {}}
        elif neg_count > pos_count:
            return {'label': 'negative', 'score': min(0.9, 0.5 + neg_count * 0.1), 'emotions': {}}
        else:
            return {'label': 'neutral', 'score': 0.5, 'emotions': {}}
    
    def analyze_lyrics_content(self, text: str) -> Dict[str, Any]:
        """
        Deeper analysis of lyrics content.
        
        Returns:
            {
                'themes': [str],  # Detected themes
                'mood_keywords': [str],  # Mood keywords
                'is_repetitive': bool,  # Whether text is repetitive
                'word_count': int,
            }
        """
        if not text:
            return {
                'themes': [],
                'mood_keywords': [],
                'is_repetitive': False,
                'word_count': 0,
            }
        
        words = text.lower().split()
        word_count = len(words)
        
        # Tematy
        theme_keywords = {
            'love': ['love', 'heart', 'kiss', 'baby', 'romance', 'together'],
            'party': ['party', 'dance', 'night', 'club', 'fun', 'weekend'],
            'heartbreak': ['broken', 'leave', 'goodbye', 'over', 'tears', 'miss'],
            'empowerment': ['strong', 'power', 'rise', 'fight', 'winner', 'queen', 'king'],
            'nature': ['sun', 'moon', 'rain', 'ocean', 'sky', 'stars', 'wind'],
            'urban': ['city', 'street', 'money', 'hustle', 'ride', 'block'],
            'spiritual': ['soul', 'god', 'heaven', 'faith', 'believe', 'angel'],
        }
        
        themes = []
        for theme, keywords in theme_keywords.items():
            if any(kw in words for kw in keywords):
                themes.append(theme)
        
        # Mood keywords
        mood_keywords = {
            'happy': ['happy', 'smile', 'laugh', 'joy', 'good'],
            'sad': ['sad', 'cry', 'tears', 'pain', 'hurt'],
            'angry': ['angry', 'hate', 'mad', 'rage', 'fight'],
            'peaceful': ['peace', 'calm', 'quiet', 'rest', 'still'],
            'energetic': ['go', 'run', 'jump', 'move', 'fast'],
        }
        
        detected_moods = []
        for mood, keywords in mood_keywords.items():
            if any(kw in words for kw in keywords):
                detected_moods.append(mood)
        
        # Repetitiveness (choruses often repeat)
        unique_words = set(words)
        is_repetitive = len(unique_words) / max(1, word_count) < 0.5
        
        return {
            'themes': themes[:3],
            'mood_keywords': detected_moods,
            'is_repetitive': is_repetitive,
            'word_count': word_count,
        }


class CLAPProcessor:
    """
    🎵 CLAP (Contrastive Language-Audio Pretraining) Processor
    
    Generates audio-text embeddings for better conditioning.
    CLAP connects audio and text in shared embedding space.
    """
    
    def __init__(self, device: str = "cpu"):
        self.device = device
        self.model = None
        self.processor = None
        self._loaded = False
    
    def _load_model(self):
        """Lazy loading of CLAP model"""
        if self._loaded:
            return
        
        try:
            from transformers import ClapModel, ClapProcessor
            
            print("   Loading CLAP model...")
            model_name = "laion/clap-htsat-unfused"
            self.processor = ClapProcessor.from_pretrained(model_name)
            self.model = ClapModel.from_pretrained(model_name).to(self.device)
            self.model.eval()
            self._loaded = True
            print("   ✅ CLAP loaded (512-dim embeddings)")
        except ImportError:
            print("   ⚠️ transformers not installed, CLAP disabled")
            self._loaded = False
        except Exception as e:
            print(f"   ⚠️ CLAP loading error: {e}")
            self._loaded = False
    
    def get_audio_embedding(self, audio: np.ndarray, sr: int) -> Optional[List[float]]:
        """
        Generates CLAP embedding for audio.
        
        Args:
            audio: Audio signal (numpy array)
            sr: Sample rate
            
        Returns:
            512-dim embedding or None if error
        """
        self._load_model()
        if not self._loaded:
            return None
        
        try:
            import torch
            
            # CLAP requires 48kHz
            if sr != 48000:
                import librosa
                audio = librosa.resample(audio, orig_sr=sr, target_sr=48000)
            
            # Limit length (CLAP has limit)
            max_samples = 48000 * 10  # 10 seconds
            if len(audio) > max_samples:
                # Take middle part
                start = (len(audio) - max_samples) // 2
                audio = audio[start:start + max_samples]
            
            inputs = self.processor(
                audios=audio, 
                sampling_rate=48000, 
                return_tensors="pt"
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                audio_features = self.model.get_audio_features(**inputs)
            
            embedding = audio_features[0].cpu().numpy().tolist()
            return embedding
            
        except Exception as e:
            print(f"   ⚠️ CLAP audio embedding error: {e}")
            return None
    
    def get_text_embedding(self, text: str) -> Optional[List[float]]:
        """
        Generates CLAP embedding for text (prompt).
        
        Args:
            text: Text/prompt
            
        Returns:
            512-dim embedding or None if error
        """
        self._load_model()
        if not self._loaded:
            return None
        
        try:
            import torch
            
            inputs = self.processor(
                text=[text], 
                return_tensors="pt", 
                padding=True
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                text_features = self.model.get_text_features(**inputs)
            
            embedding = text_features[0].cpu().numpy().tolist()
            return embedding
            
        except Exception as e:
            print(f"   ⚠️ CLAP text embedding error: {e}")
            return None


class MoodCategory(Enum):
    """Music mood categories"""
    ENERGETIC = "energetic"
    CALM = "calm"
    HAPPY = "happy"
    SAD = "sad"
    AGGRESSIVE = "aggressive"
    DREAMY = "dreamy"
    DARK = "dark"
    UPLIFTING = "uplifting"


@dataclass
class AudioFeatures:
    """Audio features extracted from file - FULL VERSION as in v1"""
    # Energy
    energy: float = 0.0
    energy_std: float = 0.0
    
    # Spectral features
    spectral_centroid: float = 0.0
    spectral_bandwidth: float = 0.0
    spectral_rolloff: float = 0.0
    spectral_contrast_mean: float = 0.0
    
    # Other
    zcr: float = 0.0  # Zero crossing rate
    tempo: float = 120.0
    
    # 🥁 Beat grid - beat positions (seconds)
    beat_positions: List[float] = field(default_factory=list)
    downbeat_positions: List[float] = field(default_factory=list)  # Strong beats (1 in bar)
    time_signature: str = "4/4"  # Metrum
    
    # 🎸 Chord progression
    chords: List[Dict] = field(default_factory=list)  # [{time: 0.0, chord: "Am"}, ...]
    chord_sequence: List[str] = field(default_factory=list)  # ["Am", "F", "C", "G"]
    
    # Key/Tonality
    dominant_key: str = "C"
    key_strength: float = 0.0
    
    # MFCC (timber/brzmienie)
    mfcc_1_mean: float = 0.0
    mfcc_2_mean: float = 0.0
    mfcc_3_mean: float = 0.0
    mfcc_means: List[float] = field(default_factory=list)  # Full 13 MFCCs
    
    # Chroma
    chroma_means: List[float] = field(default_factory=list)
    
    # Duration
    duration: float = 0.0
    
    # Categorized (for prompts)
    energy_category: str = "moderate"  # very quiet/quiet/moderate/loud/very loud
    brightness_category: str = "balanced"  # dark/warm/balanced/bright/very bright


@dataclass
class SegmentData:
    """Dane pojedynczego segmentu"""
    segment_id: str
    section_type: str
    start_time: float
    end_time: float
    duration: float
    position: float  # 0-1 pozycja w utworze
    
    # Audio features for this segment
    tempo: float = 120.0
    energy: float = 0.5
    spectral_centroid: float = 2500.0
    dominant_key: str = "C"
    
    # Generated prompt for this segment
    prompt: str = ""
    
    # For training - is this segment similar to another?
    similar_to: Optional[str] = None
    
    # 🎤 Vocal data for this segment
    has_vocals: bool = False
    vocal_confidence: float = 0.0
    lyrics_text: str = ""
    lyrics_sentiment: str = "neutral"  # positive/negative/neutral
    sentiment_score: float = 0.5
    
    # 🎵 F0/Pitch contour for this segment (v3)
    f0: Optional[List[float]] = None           # [T] continuous F0 in Hz
    f0_coarse: Optional[List[int]] = None      # [T] discrete pitch bins (0-127 MIDI)
    f0_voiced_mask: Optional[List[bool]] = None  # [T] voiced/unvoiced mask
    f0_statistics: Optional[Dict[str, float]] = None  # mean, std, min, max F0
    
    # 🔊 Loudness (v3)
    loudness: Optional[float] = None           # Integrated loudness in LUFS
    
    # 🥁 Beat positions PER SEGMENT (v3) - filtered from track-level
    beat_positions: Optional[List[float]] = None      # Beat times relative to segment start
    downbeat_positions: Optional[List[float]] = None  # Downbeat times relative to segment start
    num_beats: Optional[int] = None                   # Number of beats in segment
    
    # 🔤 Phoneme timestamps PER SEGMENT (v3)
    phoneme_timestamps: Optional[List[Dict[str, Any]]] = None  # [{'phoneme': str, 'start': float, 'end': float}]
    
    # 🎤 Vibrato analysis (v3)
    vibrato_rate: Optional[float] = None       # Vibrato frequency in Hz (typically 4-8 Hz)
    vibrato_depth: Optional[float] = None      # Vibrato depth in cents (typically 20-100)
    vibrato_extent: Optional[float] = None     # % of voiced frames with vibrato
    
    # 😤 Breath detection (v3)
    breath_positions: Optional[List[float]] = None  # Times of detected breaths relative to segment start


@dataclass
class VocalData:
    """
    Dane wokalne dla całego utworu.
    
    WAŻNE: Przechowujemy embeddingi dla OBU trybów użycia:
    
    1. voice_embedding (z całego miksu lub lekko przetworzony)
       → Używany do "w stylu X" (style transfer)
       → Uśredniany per-artysta w artist_embeddings.json
       → 256-dim z Resemblyzer (wystarczający dla stylu)
    
    2. voice_embedding_separated (z wyizolowanych wokali przez Demucs)
       → Używany do "jak X" (voice cloning)
       → Dokładniejszy, bo bez instrumentów
       → Może być 192-dim z SpeechBrain ECAPA-TDNN
    
    Wybór trybu następuje w INFERENCE, nie tutaj!
    """
    has_vocals: bool = False
    vocal_confidence: float = 0.0
    
    # ============================================
    # EMBEDDING 1: For style transfer ("in style of X")
    # ============================================
    # From full mix - sufficient to capture overall style
    voice_embedding: Optional[List[float]] = None  # 256-dim Resemblyzer
    
    # ============================================
    # EMBEDDING 2: For voice cloning ("like X")
    # ============================================
    # From separated vocals (Demucs) - more accurate for voice cloning
    voice_embedding_separated: Optional[List[float]] = None  # 192-dim SpeechBrain or 256-dim Resemblyzer
    separation_method: str = ""  # 'demucs', 'spleeter', 'none'
    
    # Path to saved separated vocals
    vocals_path: Optional[str] = None  # e.g., "data_v2/vocals/artist_name/track_id.wav"
    
    # Embedding metadata
    embedding_backend: str = ""  # 'resemblyzer', 'speechbrain'
    embedding_dim: int = 0  # 256 lub 192
    
    # 🎵 CLAP embedding (audio-text multimodal)
    clap_audio_embedding: Optional[List[float]] = None  # 512-dim CLAP audio
    clap_text_embedding: Optional[List[float]] = None   # 512-dim CLAP text (z promptu)
    
    # Lyrics
    lyrics_full: str = ""
    lyrics_language: Optional[str] = None
    lyrics_language_name: str = "Instrumental"
    lyrics_segments: List[Dict] = field(default_factory=list)  # [{start, end, text}]
    
    # 🔤 G2P: Phoneme representation (IPA)
    phonemes_ipa: str = ""  # Full IPA phoneme string
    phonemes_words: List[Dict] = field(default_factory=list)  # [{'word': str, 'phonemes': List[str]}]
    phoneme_backend: Optional[str] = None  # 'gruut' lub 'espeak'
    
    # Sentiment & content analysis
    sentiment_label: str = "neutral"  # positive/negative/neutral
    sentiment_score: float = 0.5
    themes: List[str] = field(default_factory=list)  # love, party, life, struggle...
    mood_keywords: List[str] = field(default_factory=list)  # happy, sad, angry, romantic...
    text_energy: str = "moderate"  # calm/moderate/energetic (z tekstu)
    is_repetitive: bool = False
    word_count: int = 0
    explicit: bool = False  # Czy zawiera wulgaryzmy


@dataclass
class TrackData:
    """Full track data"""
    track_id: str
    file_path: str
    duration: float
    
    # Global features
    features: AudioFeatures = field(default_factory=AudioFeatures)
    
    # Segments
    segments: List[SegmentData] = field(default_factory=list)
    
    # Metadata (from CSV/ID3/filename) - tylko potrzebne do treningu
    artist: Optional[str] = None
    genres: List[str] = field(default_factory=list)
    
    # Content info (at track level for easy access)
    language: Optional[str] = None  # Text language (from CSV or Whisper)
    explicit: bool = False  # Czy zawiera wulgaryzmy
    
    # Metadata tracking
    metadata_source: str = "none"  # csv, id3, filename, folder
    missing_fields: List[str] = field(default_factory=list)  # Fields to fill in
    
    # Generated prompts
    global_prompt: str = ""
    
    # Processing info
    sample_rate: int = 22050
    
    # 🎤 Vocal data
    vocals: VocalData = field(default_factory=VocalData)


# ============================================================================
# 💾 CHECKPOINT MANAGER - Wznowienie po crashu, incremental builds
# ============================================================================

class CheckpointManager:
    """
    Zarządza checkpointami i incremental builds.
    
    Strategia:
    - Każdy przetworzony track zapisywany do osobnego pliku JSON (atomic writes)
    - Osobny plik progress.json śledzi co już przetworzono
    - Przy wznowieniu - skipujemy już przetworzone pliki
    - Na koniec - merge wszystkich do finalnego datasetu
    
    Struktura:
        checkpoints/
        ├── progress.json          # Stan: przetworzone pliki, statystyki
        ├── tracks/                # Pojedyncze tracki
        │   ├── abc123.json
        │   ├── def456.json
        │   └── ...
        └── failed/                # Files that failed processing
            ├── xyz789.json
            └── ...
    """
    
    def __init__(
        self,
        checkpoint_dir: str,
        run_id: Optional[str] = None,
        run_name: Optional[str] = None,
    ):
        """
        Args:
            checkpoint_dir: Katalog na checkpointy (np. "./data_v2/checkpoints")
            run_id: Unikalny ID runu (jeśli None - generowany z timestamp)
            run_name: Czytelna nazwa runu (np. "server1_hiphop", "gpu2_rock")
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        
        # Generate run_id with optional name
        if run_id:
            self.run_id = run_id
        elif run_name:
            # Sanitize name + timestamp for uniqueness
            safe_name = "".join(c if c.isalnum() or c in '-_' else '_' for c in run_name)
            self.run_id = f"{safe_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        else:
            self.run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        self.run_name = run_name
        
        # Paths
        self.run_dir = self.checkpoint_dir / self.run_id
        self.tracks_dir = self.run_dir / "tracks"
        self.failed_dir = self.run_dir / "failed"
        self.progress_file = self.run_dir / "progress.json"
        
        # Stan
        self.processed_files: set = set()  # Paths of already processed files
        self.failed_files: Dict[str, str] = {}  # {path: error_message}
        self.stats = {
            'started_at': None,
            'last_update': None,
            'total_files': 0,
            'processed': 0,
            'failed': 0,
            'skipped': 0,
        }
        
        # Thread lock for safe writes
        self._lock = threading.Lock()
        
        # Inicjalizuj lub wczytaj stan
        self._init_or_load()
    
    def _init_or_load(self):
        """Initializes a new run or loads existing state."""
        if self.progress_file.exists():
            self._load_progress()
            print(f"📂 Loaded checkpoint: {self.run_id}")
            print(f"   Processed: {len(self.processed_files)}, Failed: {len(self.failed_files)}")
        else:
            # Nowy run
            self.run_dir.mkdir(parents=True, exist_ok=True)
            self.tracks_dir.mkdir(exist_ok=True)
            self.failed_dir.mkdir(exist_ok=True)
            self.stats['started_at'] = datetime.now().isoformat()
            self._save_progress()
            print(f"🆕 New checkpoint run: {self.run_id}")
    
    def _load_progress(self):
        """Wczytuje stan z pliku progress."""
        try:
            with open(self.progress_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            self.processed_files = set(data.get('processed_files', []))
            self.failed_files = data.get('failed_files', {})
            self.stats = data.get('stats', self.stats)
            
            # Ensure directories exist
            self.tracks_dir.mkdir(parents=True, exist_ok=True)
            self.failed_dir.mkdir(parents=True, exist_ok=True)
            
        except Exception as e:
            print(f"⚠️ Error loading progress: {e}")
            # Reset do pustego stanu
            self.processed_files = set()
            self.failed_files = {}
    
    def _save_progress(self):
        """Atomowy zapis stanu progress."""
        with self._lock:
            self.stats['last_update'] = datetime.now().isoformat()
            
            data = {
                'run_id': self.run_id,
                'run_name': self.run_name,
                'processed_files': list(self.processed_files),
                'failed_files': self.failed_files,
                'stats': self.stats,
            }
            
            # Atomic write: zapisz do temp, potem rename
            temp_file = self.progress_file.with_suffix('.tmp')
            with open(temp_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            
            # Atomic rename
            temp_file.rename(self.progress_file)
    
    def is_processed(self, file_path: str) -> bool:
        """Checks if file was already processed."""
        return str(file_path) in self.processed_files
    
    def save_track(self, track_dict: Dict[str, Any], file_path: str) -> bool:
        """
        Zapisuje pojedynczy przetworzony track.
        
        Args:
            track_dict: Zserializowany track (dict)
            file_path: Oryginalna ścieżka pliku audio
            
        Returns:
            True jeśli zapis się powiódł
        """
        try:
            track_id = track_dict.get('track_id', 'unknown')
            output_file = self.tracks_dir / f"{track_id}.json"
            
            # Atomic write
            temp_file = output_file.with_suffix('.tmp')
            with open(temp_file, 'w', encoding='utf-8') as f:
                json.dump(track_dict, f, indent=2, ensure_ascii=False)
            temp_file.rename(output_file)
            
            # Aktualizuj stan
            with self._lock:
                self.processed_files.add(str(file_path))
                self.stats['processed'] += 1
            
            # Save progress every 10 tracks
            if self.stats['processed'] % 10 == 0:
                self._save_progress()
            
            return True
            
        except Exception as e:
            print(f"⚠️ Error saving track {file_path}: {e}")
            self.mark_failed(file_path, str(e))
            return False
    
    def mark_failed(self, file_path: str, error: str):
        """Marks file as failed."""
        with self._lock:
            self.failed_files[str(file_path)] = error
            self.stats['failed'] += 1
        
        # Save error info
        try:
            error_file = self.failed_dir / f"{Path(file_path).stem}.json"
            with open(error_file, 'w', encoding='utf-8') as f:
                json.dump({
                    'file_path': str(file_path),
                    'error': error,
                    'timestamp': datetime.now().isoformat(),
                }, f, indent=2)
        except:
            pass
    
    def get_files_to_process(self, all_files: List[Path]) -> List[Path]:
        """
        Zwraca listę plików do przetworzenia (pomijając już przetworzone).
        
        Args:
            all_files: Wszystkie pliki audio
            
        Returns:
            Lista plików do przetworzenia
        """
        to_process = []
        skipped = 0
        
        for f in all_files:
            if str(f) in self.processed_files:
                skipped += 1
            elif str(f) in self.failed_files:
                # Opcjonalnie: retry failed files
                # Na razie skipujemy
                skipped += 1
            else:
                to_process.append(f)
        
        self.stats['total_files'] = len(all_files)
        self.stats['skipped'] = skipped
        
        if skipped > 0:
            print(f"   ⏭️  Pomijam {skipped} już przetworzonych plików")
        
        return to_process
    
    def merge_to_final(
        self,
        output_path: str,
        additional_runs: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Merguje wszystkie checkpointy do finalnego datasetu.
        
        Args:
            output_path: Ścieżka do finalnego JSON
            additional_runs: Lista dodatkowych run_id do zmergowania
            
        Returns:
            Statystyki mergowania
        """
        print(f"\n🔄 Merging checkpoints to {output_path}...")
        
        # Zbierz wszystkie katalogi z trackami
        track_dirs = [self.tracks_dir]
        
        if additional_runs:
            for run_id in additional_runs:
                run_dir = self.checkpoint_dir / run_id / "tracks"
                if run_dir.exists():
                    track_dirs.append(run_dir)
                    print(f"   + Dodano run: {run_id}")
        
        # Wczytaj wszystkie tracki
        all_tracks = []
        seen_ids = set()
        duplicates = 0
        
        for tracks_dir in track_dirs:
            for track_file in tracks_dir.glob("*.json"):
                try:
                    with open(track_file, 'r', encoding='utf-8') as f:
                        track = json.load(f)
                    
                    track_id = track.get('track_id')
                    if track_id and track_id not in seen_ids:
                        all_tracks.append(track)
                        seen_ids.add(track_id)
                    else:
                        duplicates += 1
                        
                except Exception as e:
                    print(f"   ⚠️ Błąd wczytywania {track_file}: {e}")
        
        print(f"   Wczytano {len(all_tracks)} unikalnych tracków")
        if duplicates > 0:
            print(f"   Pominięto {duplicates} duplikatów")
        
        # Oblicz statystyki
        stats = self._calculate_stats(all_tracks)
        
        # Zbuduj finalny output
        output_data = {
            'version': '2.0',
            'generated_at': datetime.now().isoformat(),
            'checkpoint_runs': [self.run_id] + (additional_runs or []),
            'stats': stats,
            'tracks': all_tracks,
        }
        
        # Zapisz
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        print(f"✅ Zapisano finalny dataset: {output_path}")
        print(f"   Tracków: {len(all_tracks)}")
        
        # Zapisz progress na koniec
        self._save_progress()
        
        return stats
    
    def _calculate_stats(self, tracks: List[Dict]) -> Dict[str, Any]:
        """Calculates statistics for track list."""
        if not tracks:
            return {'total_tracks': 0}
        
        total_segments = sum(len(t.get('segments', [])) for t in tracks)
        total_duration = sum(t.get('duration', 0) for t in tracks)
        
        # Section distribution
        section_counts = {}
        for track in tracks:
            for seg in track.get('segments', []):
                st = seg.get('section_type', 'unknown')
                section_counts[st] = section_counts.get(st, 0) + 1
        
        # Genre distribution
        genre_counts = {}
        for track in tracks:
            for genre in track.get('genres', []):
                genre_counts[genre] = genre_counts.get(genre, 0) + 1
        
        # Vocal stats
        tracks_with_vocals = sum(1 for t in tracks if t.get('vocals', {}).get('has_vocals'))
        tracks_with_lyrics = sum(1 for t in tracks if t.get('vocals', {}).get('lyrics_full'))
        tracks_with_embedding = sum(1 for t in tracks if t.get('vocals', {}).get('voice_embedding'))
        
        # Theme distribution
        theme_counts = {}
        for track in tracks:
            for theme in track.get('vocals', {}).get('themes', []):
                theme_counts[theme] = theme_counts.get(theme, 0) + 1
        
        return {
            'total_tracks': len(tracks),
            'total_segments': total_segments,
            'total_duration_hours': total_duration / 3600,
            'avg_segments_per_track': total_segments / len(tracks) if tracks else 0,
            'section_distribution': section_counts,
            'genre_distribution': dict(sorted(genre_counts.items(), key=lambda x: -x[1])[:20]),
            'tracks_with_vocals': tracks_with_vocals,
            'tracks_with_lyrics': tracks_with_lyrics,
            'tracks_with_voice_embedding': tracks_with_embedding,
            'theme_distribution': dict(sorted(theme_counts.items(), key=lambda x: -x[1])[:10]),
        }
    
    @classmethod
    def list_runs(cls, checkpoint_dir: str) -> List[Dict[str, Any]]:
        """Lists all runs in checkpoint directory."""
        checkpoint_dir = Path(checkpoint_dir)
        runs = []
        
        if not checkpoint_dir.exists():
            return runs
        
        for run_dir in checkpoint_dir.iterdir():
            if run_dir.is_dir():
                progress_file = run_dir / "progress.json"
                if progress_file.exists():
                    try:
                        with open(progress_file, 'r') as f:
                            data = json.load(f)
                        runs.append({
                            'run_id': run_dir.name,
                            'run_name': data.get('run_name'),  # Readable name
                            'processed': len(data.get('processed_files', [])),
                            'failed': len(data.get('failed_files', {})),
                            'started_at': data.get('stats', {}).get('started_at'),
                            'last_update': data.get('stats', {}).get('last_update'),
                        })
                    except:
                        runs.append({
                            'run_id': run_dir.name,
                            'run_name': None,
                            'processed': '?',
                            'failed': '?',
                        })
        
        return sorted(runs, key=lambda x: x.get('last_update', ''), reverse=True)
    
    @classmethod
    def merge_runs(
        cls,
        checkpoint_dir: str,
        run_ids: List[str],
        output_path: str,
    ) -> Dict[str, Any]:
        """
        Statyczna metoda do mergowania wielu runów.
        
        Args:
            checkpoint_dir: Katalog z checkpointami
            run_ids: Lista run_id do zmergowania
            output_path: Ścieżka do finalnego JSON
            
        Returns:
            Statystyki
        """
        if not run_ids:
            raise ValueError("Provide at least one run_id!")
        
        # Use first run as base
        manager = cls(checkpoint_dir, run_id=run_ids[0])
        
        # Merge remaining
        additional = run_ids[1:] if len(run_ids) > 1 else None
        
        return manager.merge_to_final(output_path, additional_runs=additional)


class DatasetBuilderV2:
    """
    Kompletny builder datasetu v2 z segmentami i wokalami.
    
    Obsługuje:
    - 💾 Checkpointy - wznowienie po crashu
    - 📈 Incremental builds - dopisywanie do istniejącego datasetu
    - 🔄 Merge runów - łączenie wielu runów w jeden dataset
    """
    
    def __init__(
        self,
        audio_dir: str,
        sample_rate: int = 22050,
        tracks_csv: Optional[str] = None,
        genres_csv: Optional[str] = None,
        metadata_mapping_file: Optional[str] = None,  # JSON/CSV with manual metadata mapping
        require_metadata_check: bool = False,  # Requires metadata validation before building
        min_segment_duration: float = 4.0,
        # Vocal processing - ALWAYS ENABLED
        extract_vocals: bool = True,  # Always extract voice embeddings
        extract_lyrics: bool = True,  # Always extract lyrics (Whisper)
        use_demucs: bool = True,  # ENABLED: vocal separation - IMPORTANT for voice cloning!
        save_separated_vocals: bool = True,  # Zapisuj czyste wokale per artysta
        vocals_output_dir: Optional[str] = None,  # Katalog na wokale (default: data_v2/vocals/)
        whisper_model: str = "large-v3",  # Najlepszy model dla polskiego!
        device: str = "cpu",
        # F0/Pitch extraction
        pitch_method: str = "crepe",  # crepe (accurate, default) or pyin (fast fallback)
        # LLM prompt enhancement - ALWAYS ENABLED
        use_llm_prompts: bool = True,  # Always enhance prompts via LLM
        llm_model: str = "gpt-4o-mini",
        llm_cache_file: Optional[str] = None,
        # 💾 CHECKPOINT OPTIONS
        checkpoint_dir: Optional[str] = None,  # Directory for checkpoints (enables checkpointing)
        resume_run_id: Optional[str] = None,  # Run ID to resume (or new if None)
        run_name: Optional[str] = None,  # Czytelna nazwa runu (np. "server1_hiphop")
        # 🚀 BATCH GPU PROCESSING
        batch_size: int = 1,  # Batch size for GPU (more = faster, but more VRAM)
    ):
        self.audio_dir = Path(audio_dir)
        self.sample_rate = sample_rate
        self.min_segment_duration = min_segment_duration
        self.batch_size = batch_size
        self.device = device
        self.pitch_method = pitch_method
        
        # 💾 Checkpoint manager
        self.checkpoint_manager: Optional[CheckpointManager] = None
        if checkpoint_dir:
            self.checkpoint_manager = CheckpointManager(
                checkpoint_dir=checkpoint_dir,
                run_id=resume_run_id,
                run_name=run_name,
            )
        
        # Vocal processing - ALWAYS ENABLED
        self.extract_vocals = extract_vocals
        self.extract_lyrics = extract_lyrics
        self.save_separated_vocals = save_separated_vocals
        self.vocals_output_dir = Path(vocals_output_dir) if vocals_output_dir else Path("./data_v2/vocals")
        
        # Create directory for vocals
        if self.save_separated_vocals:
            self.vocals_output_dir.mkdir(parents=True, exist_ok=True)
        
        # LLM prompt enhancement - requires OPENAI_API_KEY environment variable
        self.use_llm_prompts = use_llm_prompts
        self.llm_enhancer = None
        if use_llm_prompts:
            if not OPENAI_API_KEY:
                print("WARNING: OPENAI_API_KEY not set. LLM prompt enhancement disabled.")
                self.use_llm_prompts = False
            else:
                self.llm_enhancer = LLMPromptEnhancer(
                    api_key=OPENAI_API_KEY,
                model=llm_model,
                cache_file=llm_cache_file or "./data_v2/.prompt_cache.json",
            )
        
        # Segment annotator
        self.annotator = SegmentAnnotator(
            sample_rate=sample_rate,
            min_segment_duration=min_segment_duration,
        )
        
        # Vocal processor - ALWAYS WITH DEMUCS for vocal separation
        self.vocal_processor = VocalProcessor(
            sample_rate=sample_rate,
            use_demucs=use_demucs,
            whisper_model=whisper_model,
            device=device,
        )
        
        # 🎵 CLAP processor - audio-text embeddings
        self.clap_processor = CLAPProcessor(device=device)
        
        # 🎵 F0 Extractor - pitch contour for singing synthesis (v3)
        self.f0_extractor = None
        if HAS_F0_EXTRACTOR:
            self.f0_extractor = F0Extractor(
                method=pitch_method,  # CREPE (accurate) or PYIN (fast)
                sr=sample_rate,
                hop_length=256,
                fmin=65.0,   # C2
                fmax=2000.0,  # ~B6
            )
        
        # Metadata validation flag
        self.require_metadata_check = require_metadata_check
        
        # Load CSV metadata if provided (tracks + genres only, artist info is in tracks)
        self.tracks_df = None
        self.genres_df = None
        self.genre_map = {}
        
        # 📋 Manual metadata mapping (for files without ID3 tags)
        # Format JSON: {"filename.mp3": {"artist": "...", "genre": "..."}}
        # Format CSV: filename,artist,genre (language optional - detected by Whisper!)
        self.metadata_mapping: Dict[str, Dict[str, Any]] = {}
        if metadata_mapping_file:
            self._load_metadata_mapping(metadata_mapping_file)
        
        if tracks_csv and Path(tracks_csv).exists():
            print(f"📂 Loading tracks CSV: {tracks_csv}")
            self.tracks_df = pd.read_csv(tracks_csv)
            if 'track_id' in self.tracks_df.columns:
                self.tracks_df.set_index('track_id', inplace=True)
        
        if genres_csv and Path(genres_csv).exists():
            print(f"📂 Loading genres CSV: {genres_csv}")
            self.genres_df = pd.read_csv(genres_csv)
            if 'genre_id' in self.genres_df.columns:
                self.genres_df.set_index('genre_id', inplace=True)
                self.genre_map = dict(zip(self.genres_df.index, self.genres_df['title']))
        
        # Genre to mood/instruments mapping
        self.genre_moods = {
            'electronic': ['energetic', 'futuristic', 'synthetic'],
            'rock': ['energetic', 'powerful', 'raw'],
            'hip-hop': ['rhythmic', 'urban', 'groovy'],
            'jazz': ['smooth', 'sophisticated', 'improvised'],
            'classical': ['elegant', 'orchestral', 'timeless'],
            'folk': ['acoustic', 'traditional', 'warm'],
            'pop': ['catchy', 'upbeat', 'melodic'],
            'experimental': ['avant-garde', 'unconventional'],
            'ambient': ['atmospheric', 'calm', 'spacious'],
            'blues': ['soulful', 'emotional', 'raw'],
            'metal': ['heavy', 'powerful', 'intense'],
        }
        
        self.genre_instruments = {
            'electronic': ['synthesizers', 'drum machines', 'bass'],
            'rock': ['electric guitars', 'drums', 'bass'],
            'jazz': ['saxophone', 'piano', 'double bass'],
            'classical': ['orchestra', 'strings', 'piano'],
            'hip-hop': ['beats', '808 bass', 'samples'],
            'folk': ['acoustic guitar', 'violin', 'banjo'],
            'ambient': ['pads', 'textures', 'drones'],
            'metal': ['distorted guitars', 'double bass drums'],
            'blues': ['blues guitar', 'harmonica'],
            'pop': ['synths', 'guitars', 'drums'],
        }
        
        # 🚀 BATCH GPU PROCESSOR (lazy loaded)
        self._batch_gpu_processor = None
    
    @property
    def batch_gpu_processor(self) -> Optional['BatchGPUProcessor']:
        """Lazy load BatchGPUProcessor gdy batch_size > 1 i device=cuda"""
        if self._batch_gpu_processor is None and self.batch_size > 1 and self.device == "cuda":
            self._batch_gpu_processor = BatchGPUProcessor(
                device=self.device,
                batch_size=self.batch_size,
                sample_rate=self.sample_rate,
                whisper_model=self.vocal_processor._whisper_model_name if self.vocal_processor else "large-v3",
            )
        return self._batch_gpu_processor
    
    def _load_metadata_mapping(self, mapping_file: str):
        """
        Loads manual metadata mapping from JSON or CSV file.
        
        Format JSON:
        {
            "tracks": [
                {"filename": "song.mp3", "artist": "Zeus", "genre": "Hip-Hop"},
                ...
            ]
        }
        
        Format CSV:
        filename,artist,genre
        song.mp3,Zeus,Hip-Hop
        
        NOTE: The 'language' column is optional - language is automatically
        detected by Whisper during lyrics transcription!
        """
        mapping_path = Path(mapping_file)
        if not mapping_path.exists():
            print(f"⚠️  Mapping file does not exist: {mapping_file}")
            return
        
        try:
            if mapping_path.suffix.lower() == '.json':
                with open(mapping_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # Handle different JSON formats
                tracks = data.get('tracks', [])
                if not tracks and isinstance(data, list):
                    tracks = data
                elif not tracks and isinstance(data, dict) and 'filename' not in data:
                    # Format: {"filename.mp3": {...}} or {"path/to/file.mp3": {...}}
                    for filepath, meta in data.items():
                        if isinstance(meta, dict):
                            # Save both full path and filename
                            self.metadata_mapping[filepath] = meta
                    print(f"📋 Loaded mapping for {len(self.metadata_mapping)} files from JSON")
                    return
                
                for entry in tracks:
                    filepath = entry.get('filename') or entry.get('file_path', '')
                    if filepath:
                        meta = {
                            'artist': entry.get('artist'),
                            'genre': entry.get('genre'),
                            'genres': entry.get('genres', [entry.get('genre')] if entry.get('genre') else []),
                            'language': entry.get('language'),
                        }
                        # Save with original path (can be full or just filename)
                        self.metadata_mapping[filepath] = meta
                        
            elif mapping_path.suffix.lower() == '.csv':
                import csv
                with open(mapping_path, 'r', encoding='utf-8') as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        filepath = row.get('filename') or row.get('file_path', '')
                        if filepath:
                            genre = row.get('genre', '')
                            meta = {
                                'artist': row.get('artist'),
                                'genre': genre,
                                'genres': [genre] if genre else [],
                                'language': row.get('language'),
                            }
                            # Save with original path (can be full or just filename)
                            self.metadata_mapping[filepath] = meta
            
            print(f"📋 Loaded metadata mapping for {len(self.metadata_mapping)} files")
            
        except Exception as e:
            print(f"❌ Error loading metadata mapping: {e}")
    
    def _generate_track_id(self, file_path: str) -> str:
        """Generates unique ID for track"""
        return hashlib.md5(file_path.encode()).hexdigest()[:12]
    
    def _get_track_id_from_path(self, file_path: Path) -> Optional[int]:
        """
        Extracts track_id from FMA filename.
        
        Requires full FMA structure:
        - fma_small/000/000002.mp3 → 2
        - fma_full/012/012345.mp3 → 12345
        
        NOT recognized as FMA:
        - my_mp3s/000002.mp3 (no fma_* folder)
        - fma_small/000002.mp3 (no xxx folder)
        - random/123.mp3 (not FMA)
        """
        try:
            # Check structure: .../fma_small|fma_full/XXX/XXXXXX.mp3
            parts = file_path.parts
            
            # Look for "fma_small" or "fma_full" in path
            fma_idx = None
            for i, part in enumerate(parts):
                if part in ('fma_small', 'fma_full'):
                    fma_idx = i
                    break
            
            if fma_idx is None:
                return None  # Not in FMA folder
            
            # Check if XXX folder (3 digits) exists after fma_*
            if len(parts) <= fma_idx + 2:
                return None  # Missing required structure
            
            folder_xxx = parts[fma_idx + 1]
            if not (len(folder_xxx) == 3 and folder_xxx.isdigit()):
                return None  # Folder is not in XXX format
            
            # Check filename - must be 6 digits
            stem = file_path.stem
            if not (len(stem) == 6 and stem.isdigit()):
                return None  # Filename is not in XXXXXX format
            
            return int(stem)
            
        except Exception:
            return None
    
    def _save_separated_vocals(
        self,
        vocals: np.ndarray,
        sr: int,
        artist: Optional[str],
        track_id: str,
    ) -> Optional[str]:
        """
        Zapisuje separowane wokale do folderu artysty.
        
        Struktura:
            data_v2/vocals/
            ├── unknown/
            │   └── abc123.wav
            ├── artist_name/
            │   └── def456.wav
            └── another_artist/
                └── ghi789.wav
        
        Args:
            vocals: Separowane wokale (numpy array)
            sr: Sample rate
            artist: Nazwa artysty (None -> 'unknown')
            track_id: Unikalny ID tracka
            
        Returns:
            Ścieżka do zapisanego pliku lub None
        """
        try:
            import soundfile as sf
            
            # Sanitize artist name for folder
            artist_name = artist or 'unknown'
            artist_name = "".join(c if c.isalnum() or c in (' ', '-', '_') else '_' for c in artist_name)
            artist_name = artist_name.strip().replace(' ', '_').lower()
            if not artist_name:
                artist_name = 'unknown'
            
            # Create artist folder
            artist_dir = self.vocals_output_dir / artist_name
            artist_dir.mkdir(parents=True, exist_ok=True)
            
            # Generate filename (just track_id)
            filename = f"{track_id}.wav"
            output_path = artist_dir / filename
            
            # Normalize audio
            vocals = vocals.astype(np.float32)
            max_val = np.max(np.abs(vocals))
            if max_val > 0:
                vocals = vocals / max_val * 0.95  # Normalize to -0.95 to 0.95
            
            # Save as WAV (lossless)
            sf.write(str(output_path), vocals, sr)
            
            return str(output_path)
            
        except ImportError:
            print("   ⚠️ soundfile not installed, cannot save vocals: pip install soundfile")
            return None
        except Exception as e:
            print(f"   ⚠️ Failed to save vocals: {e}")
            return None
    
    def _update_artist_embeddings(
        self,
        artist: Optional[str],
        track_id: str,
        voice_embedding: Optional[List[float]],
        voice_embedding_separated: Optional[List[float]],
        vocal_confidence: float = 0.0,
        genres: Optional[List[str]] = None,
    ) -> None:
        """
        Aktualizuje embeddings.json dla artysty przyrostowo.
        
        Struktura:
            data_v2/vocals/
            ├── artist_name/
            │   ├── track1.wav
            │   ├── track2.wav
            │   └── embeddings.json  # <-- TEN PLIK
            └── another_artist/
                ├── track3.wav
                └── embeddings.json
        
        Format embeddings.json:
        {
            "artist": "Artist Name",
            "style_embedding": [...],              # Averaged 256-dim
            "voice_embedding": [...],              # Alias = style_embedding
            "voice_embedding_separated": [...],    # Averaged 192-dim
            "track_count": 5,
            "tracks_with_separated": 4,
            "avg_vocal_confidence": 0.85,
            "genres": ["rock", "metal"],
            "tracks": {
                "track_id_1": {"conf": 0.9, "has_sep": true},
                "track_id_2": {"conf": 0.7, "has_sep": true},
            },
            "_raw_embeddings": {
                "style": [[...], [...], ...],
                "separated": [[...], [...], ...],
                "confidences": [0.9, 0.7, ...]
            }
        }
        """
        if voice_embedding is None and voice_embedding_separated is None:
            return  # Nothing to save
        
        try:
            # Sanitize artist name
            artist_name = artist or 'unknown'
            artist_name_clean = "".join(c if c.isalnum() or c in (' ', '-', '_') else '_' for c in artist_name)
            artist_name_clean = artist_name_clean.strip().replace(' ', '_').lower()
            if not artist_name_clean:
                artist_name_clean = 'unknown'
            
            # Path to embeddings.json
            artist_dir = self.vocals_output_dir / artist_name_clean
            artist_dir.mkdir(parents=True, exist_ok=True)
            embeddings_path = artist_dir / "embeddings.json"
            
            # Load existing data or create new
            if embeddings_path.exists():
                with open(embeddings_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
            else:
                data = {
                    "artist": artist_name,
                    "style_embedding": None,
                    "voice_embedding": None,
                    "voice_embedding_separated": None,
                    "track_count": 0,
                    "tracks_with_separated": 0,
                    "avg_vocal_confidence": 0.0,
                    "genres": [],
                    "tracks": {},
                    "_raw_embeddings": {
                        "style": [],
                        "separated": [],
                        "confidences": []
                    }
                }
            
            # Check if track already exists
            if track_id in data.get("tracks", {}):
                return  # Already processed
            
            # Dodaj nowy track
            raw = data.get("_raw_embeddings", {"style": [], "separated": [], "confidences": []})
            
            if voice_embedding:
                raw["style"].append(voice_embedding)
            
            if voice_embedding_separated:
                raw["separated"].append(voice_embedding_separated)
            
            raw["confidences"].append(vocal_confidence)
            
            # Zapisz info o tracku
            data["tracks"][track_id] = {
                "conf": vocal_confidence,
                "has_sep": voice_embedding_separated is not None
            }
            
            # Dodaj genres
            existing_genres = set(data.get("genres", []))
            for g in (genres or []):
                existing_genres.add(g)
            data["genres"] = sorted(list(existing_genres))
            
            # Calculate averaged embeddings
            if raw["style"]:
                # Weighted average by confidence
                weights = np.array(raw["confidences"][:len(raw["style"])])
                if weights.sum() > 0:
                    weights = weights / weights.sum()
                    data["style_embedding"] = np.average(raw["style"], axis=0, weights=weights).tolist()
                else:
                    data["style_embedding"] = np.mean(raw["style"], axis=0).tolist()
                data["voice_embedding"] = data["style_embedding"]  # Alias
            
            if raw["separated"]:
                weights = np.array(raw["confidences"][:len(raw["separated"])])
                if weights.sum() > 0:
                    weights = weights / weights.sum()
                    data["voice_embedding_separated"] = np.average(raw["separated"], axis=0, weights=weights).tolist()
                else:
                    data["voice_embedding_separated"] = np.mean(raw["separated"], axis=0).tolist()
            
            # Aktualizuj statystyki
            data["track_count"] = len(raw["style"])
            data["tracks_with_separated"] = len(raw["separated"])
            data["avg_vocal_confidence"] = float(np.mean(raw["confidences"])) if raw["confidences"] else 0.0
            data["_raw_embeddings"] = raw
            
            # Zapisz
            with open(embeddings_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
                
        except Exception as e:
            print(f"   ⚠️ Failed to update artist embeddings: {e}")
    
    def _extract_audio_features(self, y: np.ndarray, sr: int) -> AudioFeatures:
        """Extracts FULL audio features from signal (as in v1)"""
        features = AudioFeatures()
        
        try:
            # Duration
            features.duration = len(y) / sr
            
            # RMS Energy
            rms = librosa.feature.rms(y=y)[0]
            features.energy = float(np.mean(rms))
            features.energy_std = float(np.std(rms))
            
            # Spectral features
            cent = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
            features.spectral_centroid = float(np.mean(cent))
            
            bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
            features.spectral_bandwidth = float(np.mean(bw))
            
            rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
            features.spectral_rolloff = float(np.mean(rolloff))
            
            # Zero crossing rate
            zcr = librosa.feature.zero_crossing_rate(y)[0]
            features.zcr = float(np.mean(zcr))
            
            # Tempo - z fallback dla numpy/librosa compatibility issue
            try:
                tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
                features.tempo = float(tempo) if isinstance(tempo, (int, float)) else float(tempo[0])
            except Exception:
                # Fallback: estimate tempo using onset strength + tempogram
                try:
                    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
                    tempogram = librosa.feature.tempogram(onset_envelope=onset_env, sr=sr)
                    tempo_freqs = librosa.tempo_frequencies(tempogram.shape[0], sr=sr)
                    mean_tempogram = np.mean(tempogram, axis=1)
                    tempo_idx = np.argmax(mean_tempogram)
                    features.tempo = float(tempo_freqs[tempo_idx])
                except Exception:
                    features.tempo = 120.0  # Default fallback
            
            # Chroma (key detection) - with fallback for compatibility
            try:
                chroma = librosa.feature.chroma_cens(y=y, sr=sr)
            except Exception:
                # Fallback dla starszych wersji numpy/librosa
                chroma = librosa.feature.chroma_stft(y=y, sr=sr)
            chroma_means = np.mean(chroma, axis=1)
            features.chroma_means = chroma_means.tolist()
            
            notes = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
            dominant_idx = np.argmax(chroma_means)
            features.dominant_key = notes[dominant_idx]
            features.key_strength = float(chroma_means[dominant_idx])
            
            # MFCC (full 13 + individual 1-3)
            mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
            mfcc_means = np.mean(mfcc, axis=1)
            features.mfcc_means = mfcc_means.tolist()
            features.mfcc_1_mean = float(mfcc_means[0]) if len(mfcc_means) > 0 else 0.0
            features.mfcc_2_mean = float(mfcc_means[1]) if len(mfcc_means) > 1 else 0.0
            features.mfcc_3_mean = float(mfcc_means[2]) if len(mfcc_means) > 2 else 0.0
            
            # Spectral contrast
            contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
            features.spectral_contrast_mean = float(np.mean(contrast))
            
            # ============================================
            # 🥁 Beat Grid - beat positions
            # ============================================
            try:
                tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
                beat_times = librosa.frames_to_time(beat_frames, sr=sr)
                features.beat_positions = beat_times.tolist()
                
                # Downbeats (strong beats - every 4th in 4/4)
                # Assuming 4/4 as default time signature
                features.time_signature = "4/4"
                if len(beat_times) >= 4:
                    features.downbeat_positions = beat_times[::4].tolist()
                else:
                    features.downbeat_positions = beat_times.tolist()
            except Exception as e:
                print(f"  ⚠️ Beat detection error: {e}")
                features.beat_positions = []
                features.downbeat_positions = []
            
            # ============================================
            # 🎸 Chord Detection
            # ============================================
            try:
                # Using chroma for chord detection
                # Simple detector: map chroma to major/minor chords
                hop_length = 512
                chroma_for_chords = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=hop_length)
                
                # Chord templates (major and minor)
                chord_templates = {
                    'C': [1,0,0,0,1,0,0,1,0,0,0,0],
                    'Cm': [1,0,0,1,0,0,0,1,0,0,0,0],
                    'C#': [0,1,0,0,0,1,0,0,1,0,0,0],
                    'C#m': [0,1,0,0,1,0,0,0,1,0,0,0],
                    'D': [0,0,1,0,0,0,1,0,0,1,0,0],
                    'Dm': [0,0,1,0,0,1,0,0,0,1,0,0],
                    'D#': [0,0,0,1,0,0,0,1,0,0,1,0],
                    'D#m': [0,0,0,1,0,0,1,0,0,0,1,0],
                    'E': [0,0,0,0,1,0,0,0,1,0,0,1],
                    'Em': [0,0,0,0,1,0,0,1,0,0,0,1],
                    'F': [1,0,0,0,0,1,0,0,0,1,0,0],
                    'Fm': [1,0,0,0,0,1,0,0,1,0,0,0],
                    'F#': [0,1,0,0,0,0,1,0,0,0,1,0],
                    'F#m': [0,1,0,0,0,0,1,0,0,1,0,0],
                    'G': [0,0,1,0,0,0,0,1,0,0,0,1],
                    'Gm': [0,0,1,0,0,0,0,1,0,0,1,0],
                    'G#': [1,0,0,1,0,0,0,0,1,0,0,0],
                    'G#m': [0,0,0,1,0,0,0,0,1,0,0,1],
                    'A': [0,1,0,0,1,0,0,0,0,1,0,0],
                    'Am': [1,0,0,0,1,0,0,0,0,1,0,0],
                    'A#': [0,0,1,0,0,1,0,0,0,0,1,0],
                    'A#m': [0,1,0,0,0,1,0,0,0,0,1,0],
                    'B': [0,0,0,1,0,0,1,0,0,0,0,1],
                    'Bm': [0,0,1,0,0,0,1,0,0,0,0,1],
                }
                
                # Chord detection every ~0.5 seconds
                frames_per_chord = int(0.5 * sr / hop_length)
                chords_detected = []
                
                for i in range(0, chroma_for_chords.shape[1], frames_per_chord):
                    chunk = chroma_for_chords[:, i:i+frames_per_chord]
                    if chunk.shape[1] == 0:
                        continue
                    avg_chroma = np.mean(chunk, axis=1)
                    
                    # Find closest chord
                    best_chord = 'N'  # No chord
                    best_score = 0
                    for chord_name, template in chord_templates.items():
                        score = np.dot(avg_chroma, template)
                        if score > best_score:
                            best_score = score
                            best_chord = chord_name
                    
                    time_sec = i * hop_length / sr
                    if best_score > 0.5:  # Threshold
                        chords_detected.append({'time': round(time_sec, 2), 'chord': best_chord})
                
                features.chords = chords_detected
                
                # Unique chord sequence (without consecutive repetitions)
                chord_sequence = []
                prev_chord = None
                for c in chords_detected:
                    if c['chord'] != prev_chord:
                        chord_sequence.append(c['chord'])
                        prev_chord = c['chord']
                features.chord_sequence = chord_sequence[:20]  # Max 20 chords
                
            except Exception as e:
                print(f"  ⚠️ Chord detection error: {e}")
                features.chords = []
                features.chord_sequence = []
            
            # ============================================
            # Categorization (for prompts)
            # ============================================
            
            # Tempo category
            if features.tempo < 70:
                features.tempo_category = "very slow"
            elif features.tempo < 100:
                features.tempo_category = "slow"
            elif features.tempo < 120:
                features.tempo_category = "moderate"
            elif features.tempo < 150:
                features.tempo_category = "fast"
            else:
                features.tempo_category = "very fast"
            
            # Energy category
            if features.energy < 0.02:
                features.energy_category = "very quiet"
            elif features.energy < 0.05:
                features.energy_category = "quiet"
            elif features.energy < 0.1:
                features.energy_category = "moderate"
            elif features.energy < 0.2:
                features.energy_category = "loud"
            else:
                features.energy_category = "very loud"
            
            # Brightness category (spectral centroid)
            if features.spectral_centroid < 1500:
                features.brightness_category = "dark"
            elif features.spectral_centroid < 2500:
                features.brightness_category = "warm"
            elif features.spectral_centroid < 4000:
                features.brightness_category = "balanced"
            elif features.spectral_centroid < 6000:
                features.brightness_category = "bright"
            else:
                features.brightness_category = "very bright"
            
        except Exception as e:
            print(f"  ⚠️ Feature extraction error: {e}")
        
        return features
    
    def _extract_segment_features(
        self,
        y: np.ndarray,
        sr: int,
        start_time: float,
        end_time: float,
    ) -> Dict[str, Any]:
        """Extracts features for a specific segment"""
        start_sample = int(start_time * sr)
        end_sample = int(end_time * sr)
        
        segment_y = y[start_sample:end_sample]
        
        if len(segment_y) < sr:  # Min 1 sekunda
            return {}
        
        features = {}
        
        try:
            # Energy
            rms = librosa.feature.rms(y=segment_y)[0]
            features['energy'] = float(np.mean(rms))
            
            # Spectral centroid
            cent = librosa.feature.spectral_centroid(y=segment_y, sr=sr)[0]
            features['spectral_centroid'] = float(np.mean(cent))
            
            # Tempo (can be different per segment)
            tempo, _ = librosa.beat.beat_track(y=segment_y, sr=sr)
            features['tempo'] = float(tempo) if isinstance(tempo, (int, float)) else float(tempo[0])
            
            # Key
            chroma = librosa.feature.chroma_cens(y=segment_y, sr=sr)
            chroma_means = np.mean(chroma, axis=1)
            notes = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
            features['dominant_key'] = notes[np.argmax(chroma_means)]
            
            # 🎵 F0/Pitch extraction (v3)
            if self.f0_extractor is not None:
                try:
                    f0, voiced_flag = self.f0_extractor.extract(segment_y, sr=sr)
                    
                    # Store continuous F0
                    features['f0'] = f0.tolist()
                    features['f0_voiced_mask'] = voiced_flag.tolist()
                    
                    # Convert to coarse (MIDI-like bins)
                    f0_coarse = self.f0_extractor.hz_to_coarse(f0)
                    features['f0_coarse'] = f0_coarse.tolist()
                    
                    # Statistics (only for voiced frames)
                    voiced_f0 = f0[voiced_flag]
                    if len(voiced_f0) > 0:
                        features['f0_statistics'] = {
                            'mean': float(np.mean(voiced_f0)),
                            'std': float(np.std(voiced_f0)),
                            'min': float(np.min(voiced_f0)),
                            'max': float(np.max(voiced_f0)),
                            'voiced_ratio': float(np.mean(voiced_flag)),
                        }
                except Exception as e:
                    pass  # F0 extraction failed, skip
            
            # 🔊 Loudness (LUFS) extraction (v3)
            if HAS_PYLOUDNORM:
                try:
                    meter = pyln.Meter(sr)
                    loudness = meter.integrated_loudness(segment_y)
                    # Handle -inf for silent segments
                    if np.isfinite(loudness):
                        features['loudness'] = float(loudness)
                    else:
                        features['loudness'] = -70.0  # Very quiet
                except Exception as e:
                    pass  # Loudness extraction failed, skip
            
            # 🎤 Vibrato analysis (v3) - only if we have F0
            if features.get('f0') and features.get('f0_voiced_mask'):
                try:
                    vibrato = self._analyze_vibrato(
                        np.array(features['f0']),
                        np.array(features['f0_voiced_mask']),
                        sr=sr
                    )
                    if vibrato:
                        features.update(vibrato)
                except Exception as e:
                    pass  # Vibrato analysis failed
            
            # 😤 Breath detection (v3)
            try:
                breath_positions = self._detect_breaths(segment_y, sr)
                if breath_positions:
                    features['breath_positions'] = breath_positions
            except Exception as e:
                pass  # Breath detection failed
            
        except Exception as e:
            pass
        
        return features
    
    def _analyze_vibrato(
        self,
        f0: np.ndarray,
        voiced_mask: np.ndarray,
        sr: int,
        hop_length: int = 256,
    ) -> Optional[Dict[str, float]]:
        """
        Analyzes vibrato in F0 contour.
        
        Vibrato is periodic pitch modulation, typically:
        - Rate: 4-8 Hz
        - Depth: 20-100 cents
        
        Returns:
            Dict with vibrato_rate, vibrato_depth, vibrato_extent
        """
        # Need enough voiced frames
        if np.sum(voiced_mask) < 20:
            return None
        
        # Get voiced F0 segments
        voiced_f0 = f0.copy()
        voiced_f0[~voiced_mask] = 0
        
        # Convert to cents (log scale)
        f0_safe = np.maximum(f0, 1.0)
        f0_cents = 1200 * np.log2(f0_safe / 440.0)  # Cents relative to A4
        f0_cents[~voiced_mask] = 0
        
        # Frame rate
        frame_rate = sr / hop_length
        
        # Find continuous voiced segments
        vibrato_rates = []
        vibrato_depths = []
        
        # Simple approach: look for periodic variations in F0
        # Use autocorrelation on F0 cents
        voiced_indices = np.where(voiced_mask)[0]
        if len(voiced_indices) < 30:
            return None
        
        # Get continuous segments
        segments = []
        seg_start = voiced_indices[0]
        for i in range(1, len(voiced_indices)):
            if voiced_indices[i] - voiced_indices[i-1] > 2:
                if voiced_indices[i-1] - seg_start >= 20:
                    segments.append((seg_start, voiced_indices[i-1]))
                seg_start = voiced_indices[i]
        if voiced_indices[-1] - seg_start >= 20:
            segments.append((seg_start, voiced_indices[-1]))
        
        for seg_start, seg_end in segments:
            seg_f0 = f0_cents[seg_start:seg_end+1]
            
            # Remove trend (detrend)
            seg_f0_detrended = seg_f0 - np.linspace(seg_f0[0], seg_f0[-1], len(seg_f0))
            
            # Autocorrelation
            if len(seg_f0_detrended) > 10:
                autocorr = np.correlate(seg_f0_detrended, seg_f0_detrended, mode='full')
                autocorr = autocorr[len(autocorr)//2:]
                autocorr = autocorr / (autocorr[0] + 1e-10)
                
                # Find first peak after lag 0 (vibrato period)
                # Look for peaks in 4-8 Hz range
                min_lag = int(frame_rate / 8)  # 8 Hz
                max_lag = int(frame_rate / 4)  # 4 Hz
                
                if max_lag < len(autocorr) and min_lag < max_lag:
                    search_range = autocorr[min_lag:max_lag]
                    if len(search_range) > 0:
                        peak_idx = np.argmax(search_range) + min_lag
                        peak_val = autocorr[peak_idx]
                        
                        if peak_val > 0.3:  # Significant periodicity
                            rate = frame_rate / peak_idx
                            depth = np.std(seg_f0_detrended) * 2  # ~2 std = peak-to-peak
                            
                            if 3.5 < rate < 9.0 and 10 < depth < 200:
                                vibrato_rates.append(rate)
                                vibrato_depths.append(depth)
        
        if not vibrato_rates:
            return {
                'vibrato_rate': 0.0,
                'vibrato_depth': 0.0,
                'vibrato_extent': 0.0,
            }
        
        return {
            'vibrato_rate': float(np.mean(vibrato_rates)),
            'vibrato_depth': float(np.mean(vibrato_depths)),
            'vibrato_extent': float(len(vibrato_rates) / max(len(segments), 1)),
        }
    
    def _detect_breaths(
        self,
        audio: np.ndarray,
        sr: int,
        min_breath_duration: float = 0.1,
        max_breath_duration: float = 0.8,
    ) -> List[float]:
        """
        Detects breath positions in audio.
        
        Breaths are characterized by:
        - Sudden energy drop (pause)
        - High-frequency noise (inhalation)
        - Short duration (0.1-0.8s)
        
        Returns:
            List of breath times in seconds
        """
        # Compute RMS energy
        hop_length = 512
        frame_length = 2048
        rms = librosa.feature.rms(y=audio, frame_length=frame_length, hop_length=hop_length)[0]
        
        # Compute spectral centroid (breaths have high frequency content)
        cent = librosa.feature.spectral_centroid(y=audio, sr=sr, hop_length=hop_length)[0]
        
        # Normalize
        rms_norm = (rms - rms.min()) / (rms.max() - rms.min() + 1e-10)
        cent_norm = (cent - cent.min()) / (cent.max() - cent.min() + 1e-10)
        
        # Frame times
        frame_times = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=hop_length)
        
        # Find potential breath regions:
        # - Low energy (below 30th percentile)
        # - High spectral centroid (above 50th percentile) - breath noise
        energy_threshold = np.percentile(rms_norm, 30)
        centroid_threshold = np.percentile(cent_norm, 50)
        
        breath_candidates = (rms_norm < energy_threshold) & (cent_norm > centroid_threshold)
        
        # Find contiguous regions
        breath_positions = []
        in_breath = False
        breath_start = 0
        
        for i, is_breath in enumerate(breath_candidates):
            if is_breath and not in_breath:
                breath_start = frame_times[i]
                in_breath = True
            elif not is_breath and in_breath:
                breath_end = frame_times[i]
                breath_duration = breath_end - breath_start
                
                if min_breath_duration <= breath_duration <= max_breath_duration:
                    breath_positions.append(float(breath_start))
                
                in_breath = False
        
        return breath_positions
    
    def _extract_id3_tags(self, file_path: Path) -> Dict[str, Any]:
        """
        Extracts metadata from MP3 file ID3 tags.
        Fallback when no CSV or file is not in CSV.
        """
        metadata = {
            'artist': None,
            'genres': [],
            'language': None,
            'source': 'none',
        }
        
        if not HAS_MUTAGEN:
            return metadata
        
        try:
            # Try to extract ID3 tags
            tags = ID3(str(file_path))
            metadata['source'] = 'id3'
            
            # Artist (TPE1 = Lead performer)
            if 'TPE1' in tags:
                metadata['artist'] = str(tags['TPE1'].text[0])
            
            # Genre (TCON)
            if 'TCON' in tags:
                genre = str(tags['TCON'].text[0])
                if genre and genre.lower() not in ['unknown', 'other', '']:
                    metadata['genres'].append(genre)
            
            # Language (TLAN)
            if 'TLAN' in tags:
                metadata['language'] = str(tags['TLAN'].text[0])
                
        except Exception:
            pass
        
        # Fallback to folder name for artist if no ID3
        if not metadata['artist']:
            parent = file_path.parent.name
            skip_folders = ['music', 'mp3', 'audio', 'songs', 'own', 'downloads',
                           '000', '001', '002', 'CD1', 'CD2', 'Disc 1', 'Disc 2']
            if parent and parent not in skip_folders:
                metadata['artist'] = parent.replace('_', ' ')
                metadata['source'] = 'folder'
        
        return metadata
    
    def _get_metadata(self, file_path: Path) -> Dict[str, Any]:
        """
        Gets metadata for track - tries in order:
        1. Manual mapping from file (metadata_mapping_file) - first by path, then by name
        2. CSV (for FMA with numeric track_id)
        3. ID3 tags (for MP3)
        4. Fallback to folder name for artist
        
        Only fields needed for training: artist, genres, language
        """
        metadata = {
            'artist': None,
            'genres': [],
            'language': None,
            'source': 'none',  # mapping, csv, id3, folder
        }
        
        filename = file_path.name
        
        # 0. First check manual mapping (highest priority!)
        # Try in order: full path, relative path, filename only
        mapping = None
        
        # Attempt 1: Full absolute path
        full_path = str(file_path)
        if full_path in self.metadata_mapping:
            mapping = self.metadata_mapping[full_path]
        
        # Attempt 2: Relative path to audio_dir
        if mapping is None:
            try:
                relative_path = str(file_path.relative_to(self.audio_dir))
                if relative_path in self.metadata_mapping:
                    mapping = self.metadata_mapping[relative_path]
            except ValueError:
                pass  # file_path is not in audio_dir
        
        # Attempt 3: Filename only (fallback for backwards compatibility)
        if mapping is None and filename in self.metadata_mapping:
            mapping = self.metadata_mapping[filename]
        
        if mapping is not None:
            metadata['source'] = 'mapping'
            
            if mapping.get('artist'):
                metadata['artist'] = mapping['artist']
            if mapping.get('genres'):
                metadata['genres'] = mapping['genres']
            elif mapping.get('genre'):
                metadata['genres'] = [mapping['genre']]
            if mapping.get('language'):
                metadata['language'] = mapping['language']
            
            # DON'T return here! Continue to fill missing fields from other sources
        
        # 1. Try to get from CSV (FMA) - fill missing fields
        if self.tracks_df is not None:
            track_id = self._get_track_id_from_path(file_path)
            if track_id is not None and track_id in self.tracks_df.index:
                row = self.tracks_df.loc[track_id]
                if metadata['source'] == 'none':
                    metadata['source'] = 'csv'
                
                # Artist - only if missing from mapping
                if not metadata['artist']:
                    if 'artist_name' in row and pd.notna(row['artist_name']):
                        metadata['artist'] = row['artist_name']
                
                # Language - only if missing
                if not metadata['language']:
                    if 'track_language_code' in row and pd.notna(row['track_language_code']):
                        metadata['language'] = row['track_language_code']
                
                # Genres - only if missing from mapping
                if not metadata['genres']:
                    for col in ['genre_top', 'track_genres', 'track_genres_all', 'genres']:
                        if col not in row or pd.isna(row[col]):
                            continue
                        try:
                            val = row[col]
                            if isinstance(val, str) and val.startswith('['):
                                genre_ids = ast.literal_eval(val)
                            elif isinstance(val, list):
                                genre_ids = val
                            else:
                                if col == 'genre_top':
                                    metadata['genres'].append(str(val))
                                continue
                            
                            for gid in genre_ids:
                                if gid in self.genre_map:
                                    genre_name = self.genre_map[gid]
                                    if genre_name not in metadata['genres']:
                                        metadata['genres'].append(genre_name)
                        except:
                            pass
        
        # 2. Fallback: ID3 tags - fill remaining missing fields
        id3_data = self._extract_id3_tags(file_path)
        
        # Fill missing fields from ID3
        if not metadata['artist'] and id3_data.get('artist'):
            metadata['artist'] = id3_data['artist']
        if not metadata['genres'] and id3_data.get('genres'):
            metadata['genres'] = id3_data['genres']
        if not metadata['language'] and id3_data.get('language'):
            metadata['language'] = id3_data['language']
        
        # Update source if ID3 provided data
        if id3_data.get('source', 'none') != 'none' and metadata['source'] == 'none':
            metadata['source'] = id3_data['source']
        
        # 3. Track missing fields (for later completion)
        missing_fields = []
        if not metadata['artist']:
            missing_fields.append('artist')
            # Fallback: nazwa folderu (ale NIE dla typowych nazw)
            parent = file_path.parent.name
            skip_folders = ['music', 'mp3', 'audio', 'songs', 'own', 'downloads', 
                           '000', '001', '002', 'CD1', 'CD2', 'Disc 1', 'Disc 2']
            if parent and parent not in skip_folders:
                metadata['artist'] = parent.replace('_', ' ')
                metadata['source'] = 'folder'
        if not metadata['genres']:
            missing_fields.append('genres')
        # language is optional, we don't track as "missing"
        
        metadata['missing_fields'] = missing_fields
        
        return metadata
    
    # Keep old name for compatibility
    def _get_metadata_from_csv(self, file_path: Path) -> Dict[str, Any]:
        """DEPRECATED: Use _get_metadata() instead."""
        return self._get_metadata(file_path)
    
    def _infer_moods(self, genres: List[str], energy: float, centroid: float) -> List[str]:
        """Infers moods from genres and features"""
        moods = set()
        
        # From genres
        for genre in genres:
            genre_lower = genre.lower()
            for key, genre_moods in self.genre_moods.items():
                if key in genre_lower:
                    moods.update(genre_moods[:2])
        
        # From energy
        if energy > 0.15:
            moods.add('energetic')
            moods.add('powerful')
        elif energy < 0.03:
            moods.add('calm')
            moods.add('peaceful')
        
        # From brightness (spectral centroid)
        if centroid > 4000:
            moods.add('bright')
        elif centroid < 2000:
            moods.add('dark')
            moods.add('warm')
        
        return list(moods)[:4]
    
    def _get_instruments(self, genres: List[str]) -> List[str]:
        """Returns typical instruments for genres"""
        instruments = set()
        
        for genre in genres:
            genre_lower = genre.lower()
            for key, genre_instruments in self.genre_instruments.items():
                if key in genre_lower:
                    instruments.update(genre_instruments[:2])
        
        return list(instruments)[:4]
    
    def _generate_global_prompt(self, track: TrackData) -> str:
        """
        Generates richer global prompt for track.
        
        Uses ALL available data:
        - Genre, artist, language
        - BPM, key, chords
        - Mood from lyrics analysis
        - Themes from lyrics
        """
        parts = []
        
        # 1. ARTIST (if known and not generic)
        if track.artist and track.artist.lower() not in ['unknown', 'various', 'va', 'unknown artist']:
            parts.append(f"by {track.artist}")
        
        # 2. GENRE + STYLE
        if track.genres:
            genre_str = track.genres[0].lower()
            # Mapowanie na lepsze opisy
            genre_styles = {
                'hip hop': 'hip-hop track with hard-hitting beats',
                'hip-hop': 'hip-hop track with hard-hitting beats',
                'rap': 'rap track with rhythmic flow',
                'trap': 'trap banger with heavy 808s',
                'pop': 'catchy pop song',
                'rock': 'rock track with guitar riffs',
                'electronic': 'electronic production',
                'edm': 'EDM track with synth drops',
                'r&b': 'smooth R&B groove',
                'jazz': 'jazz piece with improvisation',
                'classical': 'classical composition',
                'metal': 'heavy metal track',
            }
            style = genre_styles.get(genre_str, f"{genre_str} music")
            parts.append(style)
        else:
            parts.append("music track")
        
        # 3. MOOD (from energy + lyrics sentiment)
        moods = self._infer_moods(
            track.genres,
            track.features.energy,
            track.features.spectral_centroid,
        )
        
        # Add mood from lyrics sentiment
        if track.vocals.has_vocals and track.vocals.sentiment_label:
            sentiment_moods = {
                'positive': ['uplifting', 'energetic'],
                'negative': ['dark', 'emotional'],
                'neutral': ['chill'],
            }
            moods.extend(sentiment_moods.get(track.vocals.sentiment_label, []))
        
        # Add mood keywords from lyrics
        if track.vocals.mood_keywords:
            moods.extend(track.vocals.mood_keywords[:2])
        
        moods = list(set(moods))[:2]
        if moods:
            parts.append(f"with {' and '.join(moods)} vibe")
        
        # 4. BPM + KEY (technical but useful)
        tempo = track.features.tempo
        key = track.features.dominant_key
        
        tempo_desc = ""
        if tempo < 80:
            tempo_desc = "slow"
        elif tempo < 100:
            tempo_desc = "mid-tempo"
        elif tempo < 130:
            tempo_desc = "upbeat"
        else:
            tempo_desc = "fast"
        
        parts.append(f"at {int(tempo)} BPM ({tempo_desc})")
        
        if key:
            # Add minor/major if detected
            parts.append(f"in the key of {key}")
        
        # 5. CHORD PROGRESSION (if interesting)
        if hasattr(track.features, 'chord_sequence') and track.features.chord_sequence:
            unique_chords = list(dict.fromkeys(track.features.chord_sequence[:8]))  # First unique ones
            if len(unique_chords) >= 3:
                chord_str = " → ".join(unique_chords[:4])
                parts.append(f"chord progression: {chord_str}")
        
        # 6. VOCALS INFO
        if track.vocals.has_vocals:
            if track.vocals.vocal_confidence > 0.7:
                parts.append("featuring prominent vocals")
            else:
                parts.append("with vocals")
            
            # Language (important for non-English!)
            lang = track.vocals.lyrics_language
            if lang and lang not in ['en', 'english']:
                lang_names = {
                    'pl': 'Polish', 'es': 'Spanish', 'fr': 'French', 
                    'de': 'German', 'it': 'Italian', 'pt': 'Portuguese',
                    'ja': 'Japanese', 'ko': 'Korean', 'zh': 'Chinese',
                    'ru': 'Russian', 'ar': 'Arabic',
                }
                lang_name = lang_names.get(lang, lang)
                parts.append(f"sung in {lang_name}")
        
        # 7. THEMES from lyrics (if meaningful)
        if track.vocals.themes:
            theme_str = ' and '.join(track.vocals.themes[:2])
            parts.append(f"about {theme_str}")
        
        # 8. INSTRUMENTS (from genre inference)
        instruments = self._get_instruments(track.genres)
        if instruments:
            parts.append(f"featuring {', '.join(instruments[:2])}")
        
        # Build final prompt
        prompt = ', '.join(parts)
        
        # Clean up and capitalize
        prompt = prompt.replace('  ', ' ').strip()
        return prompt[0].upper() + prompt[1:] if prompt else "Instrumental music track"
    
    def _generate_segment_prompt(
        self,
        segment: SegmentData,
        track: TrackData,
    ) -> str:
        """
        Generates richer prompt for segment.
        
        Contains:
        - Section type (verse, chorus, etc.)
        - Segment key
        - Vocals info
        - Energy level
        - Lyrics snippet (if available)
        """
        parts = []
        
        # 1. SECTION TYPE - more descriptive prefixes
        section_prefixes = {
            'intro': 'Atmospheric intro',
            'verse': 'Verse',
            'chorus': 'Catchy chorus',
            'bridge': 'Bridge section',
            'outro': 'Fading outro',
            'drop': 'Heavy drop',
            'buildup': 'Tension buildup',
            'breakdown': 'Stripped-down breakdown',
            'instrumental': 'Instrumental break',
            'pre_chorus': 'Pre-chorus',
            'post_chorus': 'Post-chorus hook',
        }
        
        prefix = section_prefixes.get(segment.section_type, segment.section_type.replace('_', ' ').title())
        parts.append(prefix)
        
        # 2. KEY for this segment (if different from global)
        if segment.dominant_key:
            if segment.dominant_key != track.features.dominant_key:
                parts.append(f"modulating to {segment.dominant_key}")
            else:
                parts.append(f"in {segment.dominant_key}")
        
        # 3. ENERGY LEVEL
        if segment.energy > 0.25:
            parts.append("high energy")
        elif segment.energy > 0.15:
            parts.append("moderate energy")
        elif segment.energy < 0.05:
            parts.append("quiet and calm")
        
        # 4. TEMPO (if significantly different)
        tempo_diff = abs(segment.tempo - track.features.tempo)
        if tempo_diff > 15:
            if segment.tempo > track.features.tempo:
                parts.append(f"faster ({int(segment.tempo)} BPM)")
            else:
                parts.append(f"slower ({int(segment.tempo)} BPM)")
        
        # 5. VOCALS INFO
        if segment.has_vocals:
            if segment.vocal_confidence > 0.7:
                parts.append("prominent vocals")
            elif segment.vocal_confidence > 0.4:
                parts.append("background vocals")
            
            # Sentiment
            if segment.lyrics_sentiment == 'positive':
                parts.append("uplifting mood")
            elif segment.lyrics_sentiment == 'negative':
                parts.append("emotional mood")
        else:
            parts.append("instrumental")
        
        # 6. LYRICS SNIPPET (short fragment for context)
        if segment.lyrics_text and len(segment.lyrics_text) > 10:
            # Take first ~50 characters
            snippet = segment.lyrics_text[:50].strip()
            if len(segment.lyrics_text) > 50:
                snippet = snippet.rsplit(' ', 1)[0] + '...'  # Don't cut in middle of word
            parts.append(f'"{snippet}"')
        
        # 7. POSITION in track
        if segment.position < 0.05:
            parts.append("opening")
        elif segment.position > 0.95:
            parts.append("finale")
        
        # Build prompt - join with commas for readability
        prompt = ', '.join(parts)
        
        # Clean up double spaces
        prompt = prompt.replace('  ', ' ').strip()
        
        return prompt[0].upper() + prompt[1:] if prompt else "Music segment"
    
    def process_track(
        self,
        file_path: Path,
        extract_features: bool = True,
        with_segments: bool = True,
    ) -> Optional[TrackData]:
        """Processes single track"""
        try:
            # Load audio
            y, sr = librosa.load(str(file_path), sr=self.sample_rate)
            duration = len(y) / sr
            
            if duration < 10:  # Min 10 sekund
                return None
            
            # Create track data
            track = TrackData(
                track_id=self._generate_track_id(str(file_path)),
                file_path=str(file_path),
                duration=duration,
                sample_rate=sr,
            )
            
            # Get metadata (CSV → ID3 → filename fallback)
            metadata = self._get_metadata(file_path)
            track.artist = metadata['artist']
            track.genres = metadata['genres']
            track.language = metadata.get('language')  # From CSV/ID3 if available
            track.metadata_source = metadata.get('source', 'none')
            track.missing_fields = metadata.get('missing_fields', [])
            
            # Extract global features
            if extract_features:
                track.features = self._extract_audio_features(y, sr)
            else:
                # Basic features only
                track.features.duration = duration
                tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
                track.features.tempo = float(tempo) if isinstance(tempo, (int, float)) else float(tempo[0])
            
            # ====================================
            # 🎤 Vocal Processing
            # ====================================
            if self.vocal_processor and self.extract_vocals:
                # Detect vocals in whole track
                has_vocals, vocal_conf = self.vocal_processor.detect_vocals(y, sr)
                track.vocals.has_vocals = has_vocals
                track.vocals.vocal_confidence = vocal_conf
                
                if has_vocals and vocal_conf > 0.3:
                    # Extract BOTH embeddings (dla style_of i voice_clone)
                    emb_result = self.vocal_processor.extract_all_embeddings(y, sr)
                    
                    # Embedding z miksu (dla "w stylu X")
                    if emb_result['embedding_mix'] is not None:
                        track.vocals.voice_embedding = emb_result['embedding_mix'].tolist()
                    
                    # Embedding z separowanych wokali (dla "jak X" - voice clone)
                    if emb_result['embedding_separated'] is not None:
                        track.vocals.voice_embedding_separated = emb_result['embedding_separated'].tolist()
                        track.vocals.separation_method = emb_result['separation_method']
                    
                    track.vocals.embedding_backend = emb_result['backend']
                    track.vocals.embedding_dim = emb_result['embedding_dim']
                    
                    # 🎤 ZAPISZ SEPAROWANE WOKALE DO FOLDERU ARTYSTY
                    if self.save_separated_vocals and emb_result['separated_vocals'] is not None:
                        vocals_path = self._save_separated_vocals(
                            vocals=emb_result['separated_vocals'],
                            sr=sr,
                            artist=track.artist,
                            track_id=track.track_id,
                        )
                        if vocals_path:
                            track.vocals.vocals_path = vocals_path
                    
                    # 🎤 Aktualizuj embeddings.json per-artysta (przyrostowo)
                    self._update_artist_embeddings(
                        artist=track.artist,
                        track_id=track.track_id,
                        voice_embedding=track.vocals.voice_embedding,
                        voice_embedding_separated=track.vocals.voice_embedding_separated,
                        vocal_confidence=vocal_conf,
                        genres=track.genres,
                    )
                    
                    # Extract lyrics with artist/genre context for better Whisper accuracy
                    if self.extract_lyrics:
                        lyrics_data = self.vocal_processor.transcribe_lyrics(
                            y, sr,
                            artist=track.artist,
                            genres=track.genres,
                        )
                        track.vocals.lyrics_full = lyrics_data['text']
                        track.vocals.lyrics_language = lyrics_data['language']
                        track.vocals.lyrics_segments = lyrics_data['segments']
                        
                        # 🔤 G2P: Konwersja lyrics na fonemy IPA
                        if lyrics_data['text']:
                            phoneme_data = self.vocal_processor._phoneme_processor.text_to_phonemes(
                                lyrics_data['text'],
                                language=lyrics_data['language'] or 'en',
                            )
                            track.vocals.phonemes_ipa = phoneme_data.get('phonemes_ipa', '')
                            track.vocals.phonemes_words = phoneme_data.get('words', [])
                            track.vocals.phoneme_backend = phoneme_data.get('backend')
                        
                        # Detect language name for prompts
                        lang_names = {
                            'en': 'English', 'pl': 'Polish', 'es': 'Spanish', 'fr': 'French',
                            'de': 'German', 'it': 'Italian', 'pt': 'Portuguese', 'ja': 'Japanese',
                            'ko': 'Korean', 'zh': 'Chinese', 'ru': 'Russian', 'ar': 'Arabic',
                        }
                        track.vocals.lyrics_language_name = lang_names.get(
                            lyrics_data['language'], 
                            lyrics_data['language'] or 'Unknown'
                        )
                        
                        # Analyze lyrics content
                        if lyrics_data['text']:
                            # Sentiment
                            sentiment = self.vocal_processor.analyze_sentiment(lyrics_data['text'])
                            track.vocals.sentiment_label = sentiment['label']
                            track.vocals.sentiment_score = sentiment['score']
                            
                            # Content analysis
                            content = self.vocal_processor.analyze_lyrics_content(lyrics_data['text'])
                            track.vocals.themes = content['themes']
                            track.vocals.mood_keywords = content['mood_keywords']
                            track.vocals.is_repetitive = content['is_repetitive']
                            track.vocals.word_count = content['word_count']
                            
                            # Explicit content check
                            explicit_words = ['fuck', 'shit', 'bitch', 'nigga', 'ass', 'damn', 'hell']
                            text_lower = lyrics_data['text'].lower()
                            track.vocals.explicit = any(w in text_lower for w in explicit_words)
                            
                        # Set language at track level (Whisper overrides CSV if detected)
                        if lyrics_data['language']:
                            track.language = lyrics_data['language']
                    
                    # Copy explicit and language to track level for easy access
                    track.explicit = track.vocals.explicit
                    if not track.language and track.vocals.lyrics_language:
                        track.language = track.vocals.lyrics_language
                
                # Always set explicit and language at track level (even if no vocals)
                if track.explicit is None:
                    track.explicit = track.vocals.explicit
                if track.language is None and track.vocals.lyrics_language:
                    track.language = track.vocals.lyrics_language
            
            # Annotate segments
            if with_segments:
                annotated = self.annotator.annotate_audio(
                    y, sr, 
                    track_id=track.track_id,
                    file_path=str(file_path),
                )
                
                for i, seg in enumerate(annotated.segments):
                    # Extract segment-specific features
                    seg_features = {}
                    if extract_features:
                        seg_features = self._extract_segment_features(
                            y, sr, seg.start_time, seg.end_time
                        )
                    
                    # Handle section_type - can be enum or string
                    section_type_str = seg.section_type.value if hasattr(seg.section_type, 'value') else str(seg.section_type)
                    
                    segment_data = SegmentData(
                        segment_id=f"{track.track_id}_{i:03d}",
                        section_type=section_type_str,
                        start_time=seg.start_time,
                        end_time=seg.end_time,
                        duration=seg.end_time - seg.start_time,
                        position=seg.start_time / duration,
                        tempo=seg_features.get('tempo', track.features.tempo),
                        energy=seg_features.get('energy', 0.5),
                        spectral_centroid=seg_features.get('spectral_centroid', 2500),
                        dominant_key=seg_features.get('dominant_key', track.features.dominant_key),
                        # 🎵 F0/Pitch data (v3)
                        f0=seg_features.get('f0'),
                        f0_coarse=seg_features.get('f0_coarse'),
                        f0_voiced_mask=seg_features.get('f0_voiced_mask'),
                        f0_statistics=seg_features.get('f0_statistics'),
                        # 🔊 Loudness (LUFS)
                        loudness=seg_features.get('loudness'),
                        # 🎤 Vibrato (v3)
                        vibrato_rate=seg_features.get('vibrato_rate'),
                        vibrato_depth=seg_features.get('vibrato_depth'),
                        vibrato_extent=seg_features.get('vibrato_extent'),
                        # 😤 Breath positions (v3)
                        breath_positions=seg_features.get('breath_positions'),
                    )
                    
                    # 🥁 Beat positions per segment (filter from track-level)
                    if track.features.beat_positions:
                        segment_data.beat_positions = [
                            float(b - seg.start_time)
                            for b in track.features.beat_positions
                            if seg.start_time <= b < seg.end_time
                        ]
                        segment_data.num_beats = len(segment_data.beat_positions)
                    
                    if track.features.downbeat_positions:
                        segment_data.downbeat_positions = [
                            float(b - seg.start_time)
                            for b in track.features.downbeat_positions
                            if seg.start_time <= b < seg.end_time
                        ]
                    
                    # Find similar segments (using is_repetition_of if available)
                    if hasattr(seg, 'is_repetition_of') and seg.is_repetition_of is not None:
                        segment_data.similar_to = f"{track.track_id}_{seg.is_repetition_of:03d}"
                    elif hasattr(seg, 'cluster_id') and seg.cluster_id is not None:
                        for j, other_seg in enumerate(annotated.segments[:i]):
                            if hasattr(other_seg, 'cluster_id') and other_seg.cluster_id == seg.cluster_id:
                                segment_data.similar_to = f"{track.track_id}_{j:03d}"
                                break
                    
                    # 🎤 Per-segment vocal analysis
                    if self.vocal_processor and self.extract_vocals:
                        start_sample = int(seg.start_time * sr)
                        end_sample = int(seg.end_time * sr)
                        segment_y = y[start_sample:end_sample]
                        
                        if len(segment_y) > sr:  # Min 1 sec
                            seg_has_vocals, seg_vocal_conf = self.vocal_processor.detect_vocals(segment_y, sr)
                            segment_data.has_vocals = seg_has_vocals
                            segment_data.vocal_confidence = seg_vocal_conf
                            
                            # Get lyrics for this segment from full lyrics
                            if track.vocals.lyrics_segments:
                                segment_lyrics = []
                                segment_phoneme_timestamps = []
                                for lseg in track.vocals.lyrics_segments:
                                    # Check overlap
                                    if lseg['end'] > seg.start_time and lseg['start'] < seg.end_time:
                                        segment_lyrics.append(lseg['text'])
                                        # 📝 Add phoneme timestamp (relative to segment)
                                        segment_phoneme_timestamps.append({
                                            'start': max(0.0, lseg['start'] - seg.start_time),
                                            'end': min(seg.end_time - seg.start_time, lseg['end'] - seg.start_time),
                                            'text': lseg['text'],
                                            'confidence': lseg.get('confidence', 1.0)
                                        })
                                
                                segment_data.lyrics_text = ' '.join(segment_lyrics)
                                segment_data.phoneme_timestamps = segment_phoneme_timestamps if segment_phoneme_timestamps else None
                                
                                if segment_data.lyrics_text:
                                    seg_sentiment = self.vocal_processor.analyze_sentiment(segment_data.lyrics_text)
                                    segment_data.lyrics_sentiment = seg_sentiment['label']
                                    segment_data.sentiment_score = seg_sentiment['score']
                    
                    track.segments.append(segment_data)
            
            # Generate prompts (now with vocal info)
            track.global_prompt = self._generate_global_prompt(track)
            
            # 🤖 LLM Enhancement for global prompt
            if self.llm_enhancer:
                # ALWAYS generate prompts in English!
                # Text encoder (CLAP/T5) is trained on EN.
                # Lyrics remain in original language - vocals should sing in that language.
                
                features_dict = {
                    'tempo': track.features.tempo,
                    'energy': track.features.energy,
                    'dominant_key': track.features.dominant_key,
                }
                
                track.global_prompt = self.llm_enhancer.enhance_prompt(
                    base_prompt=track.global_prompt,
                    features=features_dict,
                    artist=track.artist,
                    genre=track.genres[0] if track.genres else None,
                    language='en',  # ALWAYS English for prompts!
                )
            
            # 🎵 CLAP Embeddings (audio-text multimodal)
            if self.clap_processor:
                # Audio embedding
                clap_audio = self.clap_processor.get_audio_embedding(y, sr)
                if clap_audio:
                    track.vocals.clap_audio_embedding = clap_audio
                
                # Text embedding (z promptu)
                if track.global_prompt:
                    clap_text = self.clap_processor.get_text_embedding(track.global_prompt)
                    if clap_text:
                        track.vocals.clap_text_embedding = clap_text
            
            for segment in track.segments:
                segment.prompt = self._generate_segment_prompt(segment, track)
                
                # 🤖 LLM Enhancement for segment prompts (optional, less important)
                # Skip for now - global prompt is most important
            
            return track
            
        except Exception as e:
            print(f"  ❌ Error processing {file_path}: {e}")
            # Oznacz jako failed w checkpoint manager
            if self.checkpoint_manager:
                self.checkpoint_manager.mark_failed(str(file_path), str(e))
            return None
    
    def process_track_with_gpu_data(
        self,
        file_path: Path,
        gpu_data: Dict[str, Any],
        extract_features: bool = True,
        with_segments: bool = True,
    ) -> Optional[TrackData]:
        """
        Processes track using already processed GPU batch data.
        
        gpu_data contains (from BatchGPUProcessor.process_batch):
        - audio: np.ndarray (original audio, already loaded)
        - vocals: np.ndarray (separated vocals from Demucs)
        - lyrics: Dict{'text': str, 'language': str, 'segments': List}
        - clap_audio: np.ndarray (512-dim) or None
        - clap_text: np.ndarray (512-dim) or None
        
        Skips Demucs and Whisper - uses data from batch processor.
        """
        try:
            # Use audio from gpu_data (already loaded in batch)
            y = gpu_data.get('audio')
            if y is None:
                # Fallback - load if not in gpu_data
                y, sr = librosa.load(str(file_path), sr=self.sample_rate)
            else:
                sr = self.sample_rate
            
            duration = len(y) / sr
            
            if duration < 10:  # Min 10 sekund
                return None
            
            # Create track data
            track = TrackData(
                track_id=self._generate_track_id(str(file_path)),
                file_path=str(file_path),
                duration=duration,
                sample_rate=sr,
            )
            
            # Get metadata (CSV → ID3 → filename fallback)
            metadata = self._get_metadata(file_path)
            track.artist = metadata['artist']
            track.genres = metadata['genres']
            track.language = metadata.get('language')
            track.metadata_source = metadata.get('source', 'none')
            track.missing_fields = metadata.get('missing_fields', [])
            
            # Extract global features (librosa - CPU)
            if extract_features:
                track.features = self._extract_audio_features(y, sr)
            else:
                track.features.duration = duration
                tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
                track.features.tempo = float(tempo) if isinstance(tempo, (int, float)) else float(tempo[0])
            
            # ====================================
            # 🎤 Vocal Processing (z GPU batch danych!)
            # ====================================
            vocals = gpu_data.get('vocals')
            vocals_sr = gpu_data.get('vocals_sr', sr)
            
            # Detect vocals using original audio
            if self.vocal_processor:
                has_vocals, vocal_conf = self.vocal_processor.detect_vocals(y, sr)
                track.vocals.has_vocals = has_vocals
                track.vocals.vocal_confidence = vocal_conf
                
                if has_vocals and vocal_conf > 0.3:
                    # Embedding z miksu (dla "w stylu X")
                    if hasattr(self.vocal_processor, 'voice_extractor') and self.vocal_processor.voice_extractor:
                        emb_mix = self.vocal_processor.voice_extractor.extract_embedding(y, sr)
                        if emb_mix is not None:
                            track.vocals.voice_embedding = emb_mix.tolist()
                    
                    # Embedding z separowanych wokali (z GPU batch)
                    if vocals is not None and len(vocals) > 0:
                        # Use vocals from batch processor
                        if hasattr(self.vocal_processor, 'voice_extractor') and self.vocal_processor.voice_extractor:
                            emb_sep = self.vocal_processor.voice_extractor.extract_embedding(vocals, vocals_sr)
                            if emb_sep is not None:
                                track.vocals.voice_embedding_separated = emb_sep.tolist()
                                track.vocals.separation_method = 'demucs_batch'
                        
                        track.vocals.embedding_backend = 'resemblyzer'
                        track.vocals.embedding_dim = 256
                        
                        # Zapisz separowane wokale
                        if self.save_separated_vocals:
                            vocals_path = self._save_separated_vocals(
                                vocals=vocals,
                                sr=vocals_sr,
                                artist=track.artist,
                                track_id=track.track_id,
                            )
                            if vocals_path:
                                track.vocals.vocals_path = vocals_path
                        
                        # 🎤 Aktualizuj embeddings.json per-artysta (przyrostowo)
                        self._update_artist_embeddings(
                            artist=track.artist,
                            track_id=track.track_id,
                            voice_embedding=track.vocals.voice_embedding,
                            voice_embedding_separated=track.vocals.voice_embedding_separated,
                            vocal_confidence=vocal_conf,
                            genres=track.genres,
                        )
                    
                    # Use lyrics from GPU batch (instead of Whisper call)
                    if self.extract_lyrics:
                        # gpu_data['lyrics'] to dict {'text': ..., 'language': ..., 'segments': [...]}
                        lyrics_data = gpu_data.get('lyrics', {})
                        if isinstance(lyrics_data, dict):
                            lyrics_text = lyrics_data.get('text', '')
                            lyrics_lang = lyrics_data.get('language', 'en')
                            lyrics_segments = lyrics_data.get('segments', [])
                        else:
                            # Fallback dla starego formatu
                            lyrics_text = lyrics_data if isinstance(lyrics_data, str) else ''
                            lyrics_lang = gpu_data.get('lyrics_language', 'en')
                            lyrics_segments = gpu_data.get('lyrics_segments', [])
                        
                        track.vocals.lyrics_full = lyrics_text
                        track.vocals.lyrics_language = lyrics_lang
                        track.vocals.lyrics_segments = lyrics_segments
                        
                        # G2P: Konwersja lyrics na fonemy
                        if lyrics_text and hasattr(self.vocal_processor, '_phoneme_processor'):
                            phoneme_data = self.vocal_processor._phoneme_processor.text_to_phonemes(
                                lyrics_text,
                                language=lyrics_lang or 'en',
                            )
                            track.vocals.phonemes_ipa = phoneme_data.get('phonemes_ipa', '')
                            track.vocals.phonemes_words = phoneme_data.get('words', [])
                            track.vocals.phoneme_backend = phoneme_data.get('backend')
                        
                        # Analiza tekstu
                        if lyrics_text:
                            # Sentiment
                            sentiment = self.vocal_processor.analyze_sentiment(lyrics_text)
                            track.vocals.sentiment_label = sentiment['label']
                            track.vocals.sentiment_score = sentiment['score']
                            
                            # Content analysis
                            content = self.vocal_processor.analyze_lyrics_content(lyrics_text)
                            track.vocals.themes = content['themes']
                            track.vocals.mood_keywords = content['mood_keywords']
                            track.vocals.is_repetitive = content['is_repetitive']
                            track.vocals.word_count = content['word_count']
                            
                            # Explicit check
                            explicit_words = ['fuck', 'shit', 'bitch', 'nigga', 'ass', 'damn', 'hell']
                            track.vocals.explicit = any(w in lyrics_text.lower() for w in explicit_words)
                        
                        if lyrics_lang:
                            track.language = lyrics_lang
                    
                    track.explicit = track.vocals.explicit
                    if not track.language and track.vocals.lyrics_language:
                        track.language = track.vocals.lyrics_language
            
            # Ustaw explicit/language na poziomie tracka
            if track.explicit is None:
                track.explicit = track.vocals.explicit
            if track.language is None and track.vocals.lyrics_language:
                track.language = track.vocals.lyrics_language
            
            # Annotate segments
            if with_segments:
                annotated = self.annotator.annotate_audio(
                    y, sr, 
                    track_id=track.track_id,
                    file_path=str(file_path),
                )
                
                for i, seg in enumerate(annotated.segments):
                    seg_features = {}
                    if extract_features:
                        seg_features = self._extract_segment_features(
                            y, sr, seg.start_time, seg.end_time
                        )
                    
                    section_type_str = seg.section_type.value if hasattr(seg.section_type, 'value') else str(seg.section_type)
                    
                    # 🥁 Filter beat_positions for this segment
                    segment_beat_positions = []
                    if track.features.beat_positions:
                        segment_beat_positions = [
                            b - seg.start_time 
                            for b in track.features.beat_positions 
                            if seg.start_time <= b < seg.end_time
                        ]
                    
                    # 📝 Filter phoneme_timestamps for this segment
                    segment_phoneme_timestamps = []
                    if track.vocals.lyrics_segments:
                        for lseg in track.vocals.lyrics_segments:
                            # Check if this lyric segment overlaps with current segment
                            if lseg['end'] > seg.start_time and lseg['start'] < seg.end_time:
                                # Adjust timestamps relative to segment start
                                segment_phoneme_timestamps.append({
                                    'start': max(0, lseg['start'] - seg.start_time),
                                    'end': min(seg.end_time - seg.start_time, lseg['end'] - seg.start_time),
                                    'text': lseg['text'],
                                    'confidence': lseg.get('confidence', 1.0)
                                })
                    
                    segment_data = SegmentData(
                        segment_id=f"{track.track_id}_{i:03d}",
                        section_type=section_type_str,
                        start_time=seg.start_time,
                        end_time=seg.end_time,
                        duration=seg.end_time - seg.start_time,
                        position=seg.start_time / duration,
                        tempo=seg_features.get('tempo', track.features.tempo),
                        energy=seg_features.get('energy', 0.5),
                        spectral_centroid=seg_features.get('spectral_centroid', 2500),
                        dominant_key=seg_features.get('dominant_key', track.features.dominant_key),
                        # 🎵 F0/Pitch data (v3)
                        f0=seg_features.get('f0'),
                        f0_coarse=seg_features.get('f0_coarse'),
                        f0_voiced_mask=seg_features.get('f0_voiced_mask'),
                        f0_statistics=seg_features.get('f0_statistics'),
                        # 🔊 Loudness (LUFS)
                        loudness=seg_features.get('loudness'),
                        # 🥁 Beat positions (relative to segment)
                        beat_positions=segment_beat_positions,
                        # 📝 Phoneme timestamps (relative to segment)
                        phoneme_timestamps=segment_phoneme_timestamps,
                        # 🎤 Vibrato analysis (v3)
                        vibrato_rate=seg_features.get('vibrato_rate'),
                        vibrato_depth=seg_features.get('vibrato_depth'),
                        vibrato_extent=seg_features.get('vibrato_extent'),
                        # 💨 Breath detection (v3)
                        breath_positions=seg_features.get('breath_positions'),
                    )
                    
                    # Similar segments
                    if hasattr(seg, 'is_repetition_of') and seg.is_repetition_of is not None:
                        segment_data.similar_to = f"{track.track_id}_{seg.is_repetition_of:03d}"
                    elif hasattr(seg, 'cluster_id') and seg.cluster_id is not None:
                        for j, other_seg in enumerate(annotated.segments[:i]):
                            if hasattr(other_seg, 'cluster_id') and other_seg.cluster_id == seg.cluster_id:
                                segment_data.similar_to = f"{track.track_id}_{j:03d}"
                                break
                    
                    # Per-segment vocal analysis
                    if self.vocal_processor and self.extract_vocals:
                        start_sample = int(seg.start_time * sr)
                        end_sample = int(seg.end_time * sr)
                        segment_y = y[start_sample:end_sample]
                        
                        if len(segment_y) > sr:
                            seg_has_vocals, seg_vocal_conf = self.vocal_processor.detect_vocals(segment_y, sr)
                            segment_data.has_vocals = seg_has_vocals
                            segment_data.vocal_confidence = seg_vocal_conf
                            
                            # Lyrics dla segmentu
                            if track.vocals.lyrics_segments:
                                segment_lyrics = []
                                for lseg in track.vocals.lyrics_segments:
                                    if lseg['end'] > seg.start_time and lseg['start'] < seg.end_time:
                                        segment_lyrics.append(lseg['text'])
                                
                                segment_data.lyrics_text = ' '.join(segment_lyrics)
                                
                                if segment_data.lyrics_text:
                                    seg_sentiment = self.vocal_processor.analyze_sentiment(segment_data.lyrics_text)
                                    segment_data.lyrics_sentiment = seg_sentiment['label']
                                    segment_data.sentiment_score = seg_sentiment['score']
                    
                    track.segments.append(segment_data)
            
            # Generate prompts
            track.global_prompt = self._generate_global_prompt(track)
            
            # LLM Enhancement
            if self.llm_enhancer:
                features_dict = {
                    'tempo': track.features.tempo,
                    'energy': track.features.energy,
                    'dominant_key': track.features.dominant_key,
                }
                track.global_prompt = self.llm_enhancer.enhance_prompt(
                    base_prompt=track.global_prompt,
                    features=features_dict,
                    artist=track.artist,
                    genre=track.genres[0] if track.genres else None,
                    language='en',
                )
            
            # CLAP Embeddings - use from batch if available
            clap_audio_batch = gpu_data.get('clap_audio')
            clap_text_batch = gpu_data.get('clap_text')
            
            if clap_audio_batch is not None:
                # Use from batch (already computed)
                track.vocals.clap_audio_embedding = clap_audio_batch.tolist() if hasattr(clap_audio_batch, 'tolist') else clap_audio_batch
            elif self.clap_processor:
                # Fallback - compute if not in batch
                clap_audio = self.clap_processor.get_audio_embedding(y, sr)
                if clap_audio:
                    track.vocals.clap_audio_embedding = clap_audio
            
            # Text embedding - compute with new prompt (after LLM enhancement)
            # Not using batch because prompt may have changed
            if track.global_prompt:
                if clap_text_batch is not None:
                    track.vocals.clap_text_embedding = clap_text_batch.tolist() if hasattr(clap_text_batch, 'tolist') else clap_text_batch
                elif self.clap_processor:
                    clap_text = self.clap_processor.get_text_embedding(track.global_prompt)
                    if clap_text:
                        track.vocals.clap_text_embedding = clap_text
            
            # Segment prompts
            for segment in track.segments:
                segment.prompt = self._generate_segment_prompt(segment, track)
            
            return track
            
        except Exception as e:
            print(f"  ❌ Error processing {file_path} (with GPU data): {e}")
            if self.checkpoint_manager:
                self.checkpoint_manager.mark_failed(str(file_path), str(e))
            return None
    
    def _serialize_track(self, track: TrackData) -> Dict[str, Any]:
        """
        Serializuje pojedynczy track do dict (dla JSON).
        Wyekstrahowane z build_dataset dla checkpointów.
        """
        return {
            'track_id': track.track_id,
            'file_path': track.file_path,
            'duration': track.duration,
            'sample_rate': track.sample_rate,
            'artist': track.artist,
            'genres': track.genres,
            # At track level for easy access
            'language': track.language or track.vocals.lyrics_language,
            'explicit': track.explicit if track.explicit is not None else track.vocals.explicit,
            'global_prompt': track.global_prompt,
            # Full audio features (all from v1 + new ones)
            'features': {
                'energy': track.features.energy,
                'energy_std': track.features.energy_std,
                'tempo': track.features.tempo,
                'dominant_key': track.features.dominant_key,
                'key_strength': track.features.key_strength,
                'spectral_centroid': track.features.spectral_centroid,
                'spectral_bandwidth': track.features.spectral_bandwidth,
                'spectral_rolloff': track.features.spectral_rolloff,
                'spectral_contrast_mean': track.features.spectral_contrast_mean,
                'zcr': track.features.zcr,
                'mfcc_1_mean': track.features.mfcc_1_mean,
                'mfcc_2_mean': track.features.mfcc_2_mean,
                'mfcc_3_mean': track.features.mfcc_3_mean,
                'duration': track.features.duration,
                'tempo_category': track.features.tempo_category,
                'energy_category': track.features.energy_category,
                'brightness_category': track.features.brightness_category,
                # 🥁 Beat grid
                'beat_positions': track.features.beat_positions,
                'downbeat_positions': track.features.downbeat_positions,
                'time_signature': track.features.time_signature,
                # 🎸 Chord progression
                'chords': track.features.chords,
                'chord_sequence': track.features.chord_sequence,
            },
            # 🎤 Vocal data - BOTH embeddings for different inference modes
            'vocals': {
                'has_vocals': bool(track.vocals.has_vocals),
                'vocal_confidence': float(track.vocals.vocal_confidence),
                
                # Embedding z miksu - dla "w stylu X" (style transfer)
                'voice_embedding': track.vocals.voice_embedding,  # 256-dim Resemblyzer
                
                # Embedding z separowanych wokali - dla "jak X" (voice cloning)
                'voice_embedding_separated': track.vocals.voice_embedding_separated,  # 192-dim SpeechBrain
                'separation_method': track.vocals.separation_method,  # 'demucs', 'none'
                
                # 🎤 Path to saved clean vocals (per artist)
                'vocals_path': track.vocals.vocals_path,  # e.g., "data_v2/vocals/zeus/abc123.wav"
                
                # Embedding metadata
                'embedding_backend': track.vocals.embedding_backend,
                'embedding_dim': int(track.vocals.embedding_dim),
                
                # 🎵 CLAP embeddings (audio-text multimodal)
                'clap_audio_embedding': track.vocals.clap_audio_embedding,  # 512-dim
                'clap_text_embedding': track.vocals.clap_text_embedding,    # 512-dim
                
                # Lyrics
                'lyrics_full': track.vocals.lyrics_full,
                'lyrics_language': track.vocals.lyrics_language,
                'lyrics_language_name': track.vocals.lyrics_language_name,
                'lyrics_segments': track.vocals.lyrics_segments,
                
                # 🔤 G2P: Phoneme representation (IPA)
                'phonemes_ipa': track.vocals.phonemes_ipa,
                'phonemes_words': track.vocals.phonemes_words,  # [{'word': str, 'phonemes': [...]}]
                'phoneme_backend': track.vocals.phoneme_backend,  # 'gruut' lub 'espeak'
                
                # Sentiment & content
                'sentiment_label': track.vocals.sentiment_label,
                'sentiment_score': float(track.vocals.sentiment_score),
                'themes': track.vocals.themes,
                'mood_keywords': track.vocals.mood_keywords,
                'text_energy': track.vocals.text_energy,
                'is_repetitive': bool(track.vocals.is_repetitive),
                'word_count': int(track.vocals.word_count),
                'explicit': bool(track.vocals.explicit),
            },
            'segments': [
                {
                    'segment_id': seg.segment_id,
                    'section_type': seg.section_type,
                    'start_time': float(seg.start_time),
                    'end_time': float(seg.end_time),
                    'duration': float(seg.duration),
                    'position': float(seg.position),
                    'tempo': float(seg.tempo),
                    'energy': float(seg.energy),
                    'dominant_key': seg.dominant_key,
                    'prompt': seg.prompt,
                    'similar_to': seg.similar_to,
                    # 🎤 Per-segment vocal data
                    'has_vocals': bool(seg.has_vocals),
                    'vocal_confidence': float(seg.vocal_confidence),
                    'lyrics_text': seg.lyrics_text,
                    'lyrics_sentiment': seg.lyrics_sentiment,
                    'sentiment_score': float(seg.sentiment_score),
                    # 🎵 F0/Pitch data (v3)
                    'f0': seg.f0,
                    'f0_coarse': seg.f0_coarse,
                    'f0_voiced_mask': seg.f0_voiced_mask,
                    'f0_statistics': seg.f0_statistics,
                    # 🔊 Loudness (LUFS)
                    'loudness': seg.loudness,
                    # 🥁 Beat positions per segment (v3) - timestamps relative to segment start
                    'beat_positions': seg.beat_positions,
                    # 📝 Phoneme timestamps per segment (v3) - aligned with lyrics
                    'phoneme_timestamps': seg.phoneme_timestamps,
                    # 🎤 Vibrato analysis (v3)
                    'vibrato_rate': seg.vibrato_rate,
                    'vibrato_depth': seg.vibrato_depth,
                    'vibrato_extent': seg.vibrato_extent,
                    # 💨 Breath detection points (v3) - relative timestamps
                    'breath_positions': seg.breath_positions,
                }
                for seg in track.segments
            ],
        }
    
    def build_dataset(
        self,
        output_path: str,
        max_tracks: Optional[int] = None,
        extract_features: bool = True,
        with_segments: bool = True,
        file_extensions: List[str] = ['.mp3', '.wav', '.flac', '.ogg'],
        # 💾 Checkpoint options
        auto_merge: bool = True,  # Automatycznie merguj na koniec
        # 📦 Sharding options
        shard_index: Optional[int] = None,
        total_shards: Optional[int] = None,
        shard_by: str = 'hash',
    ) -> Dict[str, Any]:
        """
        Buduje pełny dataset
        
        Args:
            shard_index: Indeks shardu (0-based) - jeśli podane, przetworzy tylko ten shard
            total_shards: Całkowita liczba shardów
            shard_by: Strategia shardingu ('hash', 'alphabetical', 'directory')
        
        Returns:
            Statystyki datasetu
        """
        print("\n" + "="*60)
        print("🎵 Building Dataset v2")
        if shard_index is not None:
            print(f"📦 SHARD {shard_index + 1}/{total_shards} (strategy: {shard_by})")
        print("="*60)
        
        # Find audio files
        print(f"\n📂 Scanning {self.audio_dir}...")
        audio_files = []
        for ext in file_extensions:
            audio_files.extend(self.audio_dir.rglob(f"*{ext}"))
        
        print(f"   Found {len(audio_files)} audio files")
        
        # 📦 SHARDING: Filtruj pliki do tego shardu
        if shard_index is not None and total_shards is not None:
            original_count = len(audio_files)
            audio_files = _apply_sharding(audio_files, shard_index, total_shards, shard_by)
            print(f"   📦 Shard {shard_index}: {len(audio_files)} files (z {original_count})")
        
        # 📋 METADATA VALIDATION (if required)
        if self.require_metadata_check:
            print("\n🔍 Checking metadata (require_metadata_check=True)...")
            missing_genre = []
            missing_artist = []
            
            for file_path in audio_files:
                metadata = self._get_metadata(file_path)
                if not metadata.get('genres'):
                    missing_genre.append(str(file_path))
                if not metadata.get('artist'):
                    missing_artist.append(str(file_path))
            
            if missing_genre or missing_artist:
                print("\n" + "="*60)
                print("❌ WALIDACJA METADANYCH NIE POWIODŁA SIĘ!")
                print("="*60)
                
                if missing_genre:
                    print(f"\n⚠️  {len(missing_genre)} plików bez gatunku:")
                    for f in missing_genre[:10]:
                        print(f"   • {Path(f).name}")
                    if len(missing_genre) > 10:
                        print(f"   ... i {len(missing_genre) - 10} więcej")
                
                if missing_artist:
                    print(f"\n⚠️  {len(missing_artist)} plików bez artysty:")
                    for f in missing_artist[:10]:
                        print(f"   • {Path(f).name}")
                    if len(missing_artist) > 10:
                        print(f"   ... i {len(missing_artist) - 10} więcej")
                
                print("\n💡 Rozwiązania:")
                print("   1. Uruchom: python tools/analyze_metadata.py --music-dir <dir> --export-missing")
                print("   2. Uzupełnij plik output/missing_metadata.csv")
                print("   3. Przekaż jako: metadata_mapping_file='output/missing_metadata.csv'")
                print("\n   Lub wyłącz walidację: require_metadata_check=False")
                
                raise ValueError(f"Brakujące metadane: {len(missing_genre)} bez genre, {len(missing_artist)} bez artist")
            
            print("   ✅ Wszystkie pliki mają wymagane metadane!")
        
        if max_tracks:
            audio_files = audio_files[:max_tracks]
            print(f"   Limited to {max_tracks} tracks")
        
        # 💾 CHECKPOINT: Filter already processed files
        if self.checkpoint_manager:
            audio_files = self.checkpoint_manager.get_files_to_process(audio_files)
            if not audio_files:
                print("\n✅ All files already processed!")
                print("   Użyj --auto_merge aby zmergować do finalnego datasetu")
                if auto_merge:
                    return self.checkpoint_manager.merge_to_final(output_path)
                return {'total_tracks': 0, 'message': 'All files already processed'}
        
        # 🚀 TIME ESTIMATION
        parallel = ParallelProcessor(
            device=self.device,
            gpu_batch_size=self.batch_size,
        )
        parallel.print_estimate(len(audio_files))
        
        # Process tracks
        print(f"\n🔄 Processing {len(audio_files)} tracks...")
        tracks = []
        processed_count = 0
        
        # 🚀 BATCH MODE (gdy batch_size > 1 i GPU)
        if self.batch_size > 1 and self.device == "cuda":
            print(f"   🚀 Using BATCH GPU mode (batch_size={self.batch_size})")
            
            # Przetwarzaj w batchach
            for batch_start in range(0, len(audio_files), self.batch_size):
                batch_end = min(batch_start + self.batch_size, len(audio_files))
                batch_files = audio_files[batch_start:batch_end]
                
                try:
                    # Pobierz metadane dla batcha
                    batch_metadata = []
                    for fp in batch_files:
                        metadata = self._get_metadata(fp)
                        batch_metadata.append({
                            'artist': metadata.get('artist'),
                            'genres': metadata.get('genres', []),
                        })
                    
                    # GPU batch processing
                    if self.batch_gpu_processor:
                        batch_results = self.batch_gpu_processor.process_batch(
                            batch_files,
                            batch_metadata,
                        )
                        
                        # Process each result from batch
                        for i, (file_path, gpu_data) in enumerate(zip(batch_files, batch_results)):
                            try:
                                track = self.process_track_with_gpu_data(
                                    file_path=file_path,
                                    gpu_data=gpu_data,
                                    extract_features=extract_features,
                                    with_segments=with_segments,
                                )
                                
                                if track:
                                    if self.checkpoint_manager:
                                        track_dict = self._serialize_track(track)
                                        self.checkpoint_manager.save_track(track_dict, str(file_path))
                                    else:
                                        tracks.append(track)
                                    processed_count += 1
                                    
                            except Exception as e:
                                print(f"\n  ❌ Error processing {file_path}: {e}")
                                if self.checkpoint_manager:
                                    self.checkpoint_manager.mark_failed(str(file_path), str(e))
                    
                except KeyboardInterrupt:
                    print("\n\n⚠️ Interrupted by user (Ctrl+C)")
                    if self.checkpoint_manager:
                        self.checkpoint_manager._save_progress()
                        print(f"💾 Saved checkpoint: {processed_count} tracks")
                    raise
                except Exception as e:
                    print(f"\n  ❌ Batch processing error: {e}")
                    # Fallback: process individually
                    for file_path in batch_files:
                        try:
                            track = self.process_track(file_path, extract_features, with_segments)
                            if track:
                                if self.checkpoint_manager:
                                    self.checkpoint_manager.save_track(self._serialize_track(track), str(file_path))
                                else:
                                    tracks.append(track)
                                processed_count += 1
                        except Exception as e2:
                            if self.checkpoint_manager:
                                self.checkpoint_manager.mark_failed(str(file_path), str(e2))
                
                # Progress
                print(f"   Processed {min(batch_end, len(audio_files))}/{len(audio_files)} tracks")
        
        else:
            # 🐢 SEQUENTIAL MODE (default or when batch_size=1)
            for file_path in tqdm(audio_files, desc="Processing"):
                try:
                    track = self.process_track(
                        file_path,
                        extract_features=extract_features,
                        with_segments=with_segments,
                    )
                    if track:
                        # 💾 CHECKPOINT: Zapisz od razu po przetworzeniu
                        if self.checkpoint_manager:
                            track_dict = self._serialize_track(track)
                            self.checkpoint_manager.save_track(track_dict, str(file_path))
                        else:
                            tracks.append(track)
                        processed_count += 1
                        
                except KeyboardInterrupt:
                    print("\n\n⚠️ Przerwano przez użytkownika (Ctrl+C)")
                    if self.checkpoint_manager:
                        self.checkpoint_manager._save_progress()
                        print(f"💾 Zapisano checkpoint: {processed_count} tracków")
                        print(f"   Wznów: --resume_run_id {self.checkpoint_manager.run_id}")
                    raise
                except Exception as e:
                    print(f"\n  ❌ Błąd przetwarzania {file_path}: {e}")
                    if self.checkpoint_manager:
                        self.checkpoint_manager.mark_failed(str(file_path), str(e))
        
        # 💾 CHECKPOINT: Merge or standard save
        if self.checkpoint_manager:
            # Save final progress
            self.checkpoint_manager._save_progress()
            
            if auto_merge:
                print(f"\n💾 Merging checkpoints...")
                stats = self.checkpoint_manager.merge_to_final(output_path)
            else:
                print(f"\n✅ Processed {processed_count} tracks")
                print(f"   Checkpoints in: {self.checkpoint_manager.run_dir}")
                print(f"   Merge manually: python build_dataset_v2.py --merge --output {output_path}")
                return self.checkpoint_manager.stats
        else:
            # Standard save (without checkpoints)
            print(f"\n✅ Processed {len(tracks)} tracks successfully")
            
            # Calculate statistics
            total_segments = sum(len(t.segments) for t in tracks)
            total_duration = sum(t.duration for t in tracks)
            
            # Section type distribution
            section_counts = {}
            for track in tracks:
                for seg in track.segments:
                    section_counts[seg.section_type] = section_counts.get(seg.section_type, 0) + 1
            
            # Genre distribution
            genre_counts = {}
            for track in tracks:
                for genre in track.genres:
                    genre_counts[genre] = genre_counts.get(genre, 0) + 1
            
            # Vocal stats
            tracks_with_vocals = sum(1 for t in tracks if t.vocals.has_vocals)
            tracks_with_lyrics = sum(1 for t in tracks if t.vocals.lyrics_full)
            total_voice_embeddings = sum(1 for t in tracks if t.vocals.voice_embedding)
            
            # Theme distribution (from lyrics)
            theme_counts = {}
            for track in tracks:
                for theme in track.vocals.themes:
                    theme_counts[theme] = theme_counts.get(theme, 0) + 1
            
            stats = {
                'total_tracks': len(tracks),
                'total_segments': total_segments,
                'total_duration_hours': total_duration / 3600,
                'avg_segments_per_track': total_segments / len(tracks) if tracks else 0,
                'section_distribution': section_counts,
                'genre_distribution': dict(sorted(genre_counts.items(), key=lambda x: -x[1])[:20]),
                # Vocal stats
                'tracks_with_vocals': tracks_with_vocals,
                'tracks_with_lyrics': tracks_with_lyrics,
                'tracks_with_voice_embedding': total_voice_embeddings,
                'theme_distribution': dict(sorted(theme_counts.items(), key=lambda x: -x[1])[:10]),
            }
            
            # Serialize tracks
            print(f"\n💾 Saving to {output_path}...")
            
            output_data = {
                'version': '2.0',
                'stats': stats,
                'tracks': [self._serialize_track(t) for t in tracks],
            }
            
            # Write JSON
            output_path_obj = Path(output_path)
            output_path_obj.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_path_obj, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, indent=2, ensure_ascii=False)
            
            # 📋 Generate missing metadata report
            missing_metadata = []
            for track in tracks:
                if track.missing_fields:
                    missing_metadata.append({
                        'track_id': track.track_id,
                        'file_path': track.file_path,
                        'missing': track.missing_fields,
                    })
            
            if missing_metadata:
                missing_path = output_path_obj.with_name(output_path_obj.stem + '_missing_metadata.json')
                with open(missing_path, 'w', encoding='utf-8') as f:
                    json.dump(missing_metadata, f, indent=2, ensure_ascii=False)
                print(f"\n📋 Zapisano {len(missing_metadata)} tracków z brakującymi danymi do: {missing_path}")
        
        # Print stats
        print("\n" + "="*60)
        print("📊 Dataset Statistics")
        print("="*60)
        print(f"   Total tracks: {stats['total_tracks']}")
        print(f"   Total segments: {stats.get('total_segments', 'N/A')}")
        print(f"   Total duration: {stats.get('total_duration_hours', 0):.1f} hours")
        print(f"   Avg segments/track: {stats.get('avg_segments_per_track', 0):.1f}")
        
        section_counts = stats.get('section_distribution', {})
        if section_counts:
            print("\n   Section distribution:")
            for section, count in sorted(section_counts.items(), key=lambda x: -x[1]):
                print(f"     {section}: {count}")
        
        genre_counts = stats.get('genre_distribution', {})
        if genre_counts:
            print("\n   Top genres:")
            for genre, count in list(genre_counts.items())[:10]:
                print(f"     {genre}: {count}")
        
        # Vocal stats
        if stats.get('total_tracks', 0) > 0:
            print("\n   🎤 Vocal Statistics:")
            print(f"     Tracks with vocals: {stats.get('tracks_with_vocals', 0)}")
            print(f"     Tracks with lyrics: {stats.get('tracks_with_lyrics', 0)}")
            print(f"     Tracks with voice embedding: {stats.get('tracks_with_voice_embedding', 0)}")
            
            theme_dist = stats.get('theme_distribution', {})
            if theme_dist:
                print("\n   📝 Lyrics themes:")
                for theme, count in list(theme_dist.items())[:7]:
                    print(f"     {theme}: {count}")
        
        print(f"\n✅ Dataset saved to {output_path}")
        
        return stats


# =============================================================================
# 📦 SHARDING HELPERS
# =============================================================================

def _apply_sharding(
    files: List[Path],
    shard_index: int,
    total_shards: int,
    strategy: str = 'hash',
) -> List[Path]:
    """
    Dzieli listę plików na shardy.
    
    Strategie:
    - hash: stabilne (ten sam plik zawsze w tym samym shardzie)
    - alphabetical: alfabetycznie (łatwe do debugowania)
    - directory: po katalogu nadrzędnym (dla struktury artystów)
    
    Args:
        files: Lista plików do podzielenia
        shard_index: Indeks shardu (0-based)
        total_shards: Całkowita liczba shardów
        strategy: Strategia shardowania
        
    Returns:
        Lista plików dla tego shardu
    """
    if strategy == 'hash':
        # Hash MD5 nazwy pliku -> stabilne przypisanie
        def get_shard(f: Path) -> int:
            h = hashlib.md5(str(f).encode()).hexdigest()
            return int(h, 16) % total_shards
        
        return [f for f in files if get_shard(f) == shard_index]
    
    elif strategy == 'alphabetical':
        # Sort alphabetically and divide evenly
        sorted_files = sorted(files, key=lambda f: str(f).lower())
        chunk_size = len(sorted_files) // total_shards
        remainder = len(sorted_files) % total_shards
        
        start = shard_index * chunk_size + min(shard_index, remainder)
        end = start + chunk_size + (1 if shard_index < remainder else 0)
        
        return sorted_files[start:end]
    
    elif strategy == 'directory':
        # Group by parent directory
        from collections import defaultdict
        dir_files = defaultdict(list)
        for f in files:
            dir_files[f.parent].append(f)
        
        # Sort directories and assign to shards
        sorted_dirs = sorted(dir_files.keys(), key=lambda d: str(d).lower())
        result = []
        for i, d in enumerate(sorted_dirs):
            if i % total_shards == shard_index:
                result.extend(dir_files[d])
        
        return result
    
    else:
        raise ValueError(f"Unknown sharding strategy: {strategy}")


def _merge_sharded_datasets(
    shard_paths: List[str],
    output_path: str,
) -> Dict[str, Any]:
    """
    Merguje shardy datasetu w jeden plik.
    
    Args:
        shard_paths: Lista ścieżek do shardów (dataset_shard_0.json, ...)
        output_path: Ścieżka do wyjściowego zmergowanego datasetu
        
    Returns:
        Statystyki zmergowanego datasetu
    """
    print(f"\n📦 Merging {len(shard_paths)} shards...")
    
    all_tracks = []
    combined_stats = {
        'total_tracks': 0,
        'total_segments': 0,
        'total_duration_hours': 0,
        'tracks_with_vocals': 0,
        'tracks_with_lyrics': 0,
        'section_distribution': {},
        'genre_distribution': {},
    }
    
    for i, shard_path in enumerate(shard_paths):
        if not Path(shard_path).exists():
            print(f"   ⚠️ Shard nie istnieje: {shard_path}")
            continue
        
        print(f"   Loading shard {i+1}/{len(shard_paths)}: {shard_path}")
        
        with open(shard_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        tracks = data.get('tracks', [])
        stats = data.get('stats', {})
        
        all_tracks.extend(tracks)
        
        # Merge stats
        combined_stats['total_tracks'] += stats.get('total_tracks', len(tracks))
        combined_stats['total_segments'] += stats.get('total_segments', 0)
        combined_stats['total_duration_hours'] += stats.get('total_duration_hours', 0)
        combined_stats['tracks_with_vocals'] += stats.get('tracks_with_vocals', 0)
        combined_stats['tracks_with_lyrics'] += stats.get('tracks_with_lyrics', 0)
        
        # Merge distributions
        for genre, count in stats.get('genre_distribution', {}).items():
            combined_stats['genre_distribution'][genre] = \
                combined_stats['genre_distribution'].get(genre, 0) + count
        
        for section, count in stats.get('section_distribution', {}).items():
            combined_stats['section_distribution'][section] = \
                combined_stats['section_distribution'].get(section, 0) + count
    
    # Sort distributions
    combined_stats['genre_distribution'] = dict(
        sorted(combined_stats['genre_distribution'].items(), key=lambda x: -x[1])[:50]
    )
    combined_stats['section_distribution'] = dict(
        sorted(combined_stats['section_distribution'].items(), key=lambda x: -x[1])
    )
    
    # Recalculate averages
    if combined_stats['total_tracks'] > 0:
        combined_stats['avg_segments_per_track'] = \
            combined_stats['total_segments'] / combined_stats['total_tracks']
    
    # Save merged dataset
    output_data = {
        'version': '2.0',
        'merged_from_shards': len(shard_paths),
        'stats': combined_stats,
        'tracks': all_tracks,
    }
    
    output_path_obj = Path(output_path)
    output_path_obj.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"\n💾 Saving merged dataset to {output_path}...")
    with open(output_path_obj, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ Merged {combined_stats['total_tracks']} tracks from {len(shard_paths)} shards")
    
    return combined_stats


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description='🎵 Build Dataset v2 - Full pipeline with segments, vocals & lyrics'
    )
    
    # Input/Output
    parser.add_argument('--audio_dir', type=str, required=True,
                        help='Directory with audio files')
    parser.add_argument('--output', type=str, default='./data_v2/training_dataset_v2.json',
                        help='Output JSON path')
    
    # CSV metadata (optional) - artist info is already in tracks CSV
    parser.add_argument('--tracks_csv', type=str, default=None,
                        help='Tracks metadata CSV (includes artist_name)')
    parser.add_argument('--genres_csv', type=str, default=None,
                        help='Genres metadata CSV (for genre_id to name mapping)')
    
    # 📋 METADATA MAPPING (for files without ID3 tags or for completion)
    parser.add_argument('--metadata_mapping', type=str, default=None,
                        help='JSON/CSV file with manual metadata mapping (highest priority)')
    parser.add_argument('--require_metadata_check', action='store_true',
                        help='Require all tracks to have artist+genre before building')
    
    # 🎤 VOCALS OUTPUT
    parser.add_argument('--vocals_output_dir', type=str, default='./data_v2/vocals',
                        help='Directory for separated vocals per artist')
    
    # 💾 CHECKPOINT OPTIONS
    parser.add_argument('--checkpoint_dir', type=str, default=None,
                        help='Directory for checkpoints (enables resume on crash)')
    parser.add_argument('--run_name', type=str, default=None,
                        help='Human-readable run name (e.g. "server1_hiphop", "gpu2_rock")')
    parser.add_argument('--resume_run_id', type=str, default=None,
                        help='Run ID to resume (uses existing checkpoints)')
    parser.add_argument('--no_auto_merge', action='store_true',
                        help='Do not auto-merge checkpoints at the end')
    parser.add_argument('--merge', action='store_true',
                        help='Only merge existing checkpoints (no processing)')
    parser.add_argument('--merge_runs', type=str, nargs='+',
                        help='List of run IDs to merge together')
    parser.add_argument('--list_runs', action='store_true',
                        help='List all checkpoint runs')
    
    # Processing options
    parser.add_argument('--with_segments', action='store_true', default=True,
                        help='Include segment annotations (default: True)')
    parser.add_argument('--no_segments', action='store_true',
                        help='Disable segment annotations')
    parser.add_argument('--extract_features', action='store_true', default=True,
                        help='Extract detailed audio features (default: True)')
    parser.add_argument('--no_features', action='store_true',
                        help='Disable detailed feature extraction')
    parser.add_argument('--segments_only', action='store_true',
                        help='Only segment annotations, minimal features')
    
    # 🎤 Vocal processing - ALWAYS ENABLED (no flags to disable)
    parser.add_argument('--whisper_model', type=str, default='large-v3',
                        choices=['tiny', 'base', 'small', 'medium', 'large', 'large-v2', 'large-v3'],
                        help='Whisper model size for lyrics extraction (large-v3 best for PL)')
    parser.add_argument('--device', type=str, default='cpu',
                        help='Device for vocal processing (cpu/cuda)')
    parser.add_argument('--pitch_method', type=str, default='crepe',
                        choices=['pyin', 'crepe'],
                        help='F0/pitch extraction method: crepe (accurate, default) or pyin (fast fallback)')
    
    # 🤖 LLM prompt enhancement - ALWAYS ENABLED (no flag to disable)
    parser.add_argument('--llm_model', type=str, default='gpt-4o-mini',
                        help='OpenAI model to use (default: gpt-4o-mini)')
    parser.add_argument('--llm_cache', type=str, default='./data_v2/.prompt_cache.json',
                        help='Path to LLM prompt cache file')
    
    # Limits
    parser.add_argument('--max_tracks', type=int, default=None,
                        help='Maximum number of tracks to process')
    parser.add_argument('--min_segment', type=float, default=4.0,
                        help='Minimum segment duration (seconds)')
    
    # Audio
    parser.add_argument('--sample_rate', type=int, default=22050,
                        help='Sample rate for processing')
    
    # 🚀 PARALLEL PROCESSING
    parser.add_argument('--workers', type=int, default=None,
                        help='Number of CPU workers for librosa features (default: cpu_count-2)')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='GPU batch size for Demucs/Whisper (more = faster but needs more VRAM)')
    parser.add_argument('--estimate_time', action='store_true',
                        help='Only estimate processing time, do not build')
    
    # 📦 SHARDING (for 1M+ files)
    parser.add_argument('--shard_index', type=int, default=None,
                        help='Shard index (0-based). Use with --total_shards')
    parser.add_argument('--total_shards', type=int, default=None,
                        help='Total number of shards (e.g. 10 = 10 parts)')
    parser.add_argument('--shard_by', type=str, default='hash',
                        choices=['hash', 'alphabetical', 'directory'],
                        help='Sharding strategy: hash (stable), alphabetical, directory')
    parser.add_argument('--merge_shards', type=str, nargs='+', default=None,
                        help='Merge shard files into one dataset (e.g. dataset_shard_*.json)')
    
    args = parser.parse_args()
    
    # 📦 MERGE SHARDS MODE
    if args.merge_shards:
        stats = _merge_sharded_datasets(args.merge_shards, args.output)
        return
    
    # 📋 LIST RUNS
    if args.list_runs:
        if not args.checkpoint_dir:
            print("❌ Provide --checkpoint_dir")
            return
        
        runs = CheckpointManager.list_runs(args.checkpoint_dir)
        if not runs:
            print("No checkpoints")
            return
        
        print(f"\n📂 Checkpoints in {args.checkpoint_dir}:")
        print("-" * 70)
        for run in runs:
            name_str = f" ({run['run_name']})" if run.get('run_name') else ""
            print(f"  📁 {run['run_id']}{name_str}")
            print(f"     Processed: {run['processed']}, Failed: {run['failed']}")
            print(f"     Last update: {run.get('last_update', 'N/A')}")
        return
    
    # 🔄 MERGE RUNS
    if args.merge or args.merge_runs:
        if not args.checkpoint_dir:
            print("❌ Provide --checkpoint_dir")
            return
        
        if args.merge_runs:
            run_ids = args.merge_runs
        else:
            # Merge latest run
            runs = CheckpointManager.list_runs(args.checkpoint_dir)
            if not runs:
                print("❌ No checkpoints to merge")
                return
            run_ids = [runs[0]['run_id']]
        
        print(f"🔄 Merging runs: {run_ids}")
        stats = CheckpointManager.merge_runs(
            checkpoint_dir=args.checkpoint_dir,
            run_ids=run_ids,
            output_path=args.output,
        )
        print(f"✅ Merged {stats['total_tracks']} tracks")
        return
    
    # 📦 SHARDING - argument validation
    if (args.shard_index is not None) != (args.total_shards is not None):
        print("❌ Use --shard_index TOGETHER with --total_shards")
        return
    
    if args.shard_index is not None:
        if args.shard_index < 0 or args.shard_index >= args.total_shards:
            print(f"❌ --shard_index must be in range 0 to {args.total_shards - 1}")
            return
        print(f"📦 SHARD MODE: shard {args.shard_index + 1}/{args.total_shards} (strategy: {args.shard_by})")
    
    # Handle flags
    with_segments = not args.no_segments
    extract_features = not args.no_features
    
    if args.segments_only:
        extract_features = False
    
    # EVERYTHING ALWAYS ENABLED - no flags to disable
    extract_vocals = True
    extract_lyrics = True
    use_demucs = True
    use_llm = True
    
    # 🚀 PARALLEL PROCESSOR (do estymacji czasu i info o hardware)
    parallel = ParallelProcessor(
        device=args.device,
        cpu_workers=args.workers,
        gpu_batch_size=args.batch_size,
    )
    
    # ⏱️  ESTIMATE TIME MODE
    if args.estimate_time:
        # Policz pliki audio
        from pathlib import Path
        audio_dir = Path(args.audio_dir)
        audio_files = []
        for ext in ['.mp3', '.wav', '.flac', '.ogg']:
            audio_files.extend(audio_dir.rglob(f'*{ext}'))
        
        # Apply sharding if enabled
        if args.shard_index is not None:
            audio_files = _apply_sharding(
                audio_files, args.shard_index, args.total_shards, args.shard_by
            )
        
        if args.max_tracks:
            audio_files = audio_files[:args.max_tracks]
        
        parallel.print_estimate(len(audio_files))
        return
    
    # Build dataset
    builder = DatasetBuilderV2(
        audio_dir=args.audio_dir,
        sample_rate=args.sample_rate,
        tracks_csv=args.tracks_csv,
        genres_csv=args.genres_csv,
        metadata_mapping_file=args.metadata_mapping,
        require_metadata_check=args.require_metadata_check,
        min_segment_duration=args.min_segment,
        vocals_output_dir=args.vocals_output_dir,
        # Vocal options - ENABLED BY DEFAULT
        extract_vocals=extract_vocals,
        extract_lyrics=extract_lyrics,
        use_demucs=use_demucs,
        whisper_model=args.whisper_model,
        device=args.device,
        # F0/Pitch extraction
        pitch_method=args.pitch_method,
        # LLM options - ENABLED BY DEFAULT (fixed API key)
        use_llm_prompts=use_llm,
        llm_model=args.llm_model,
        llm_cache_file=args.llm_cache,
        # 💾 Checkpoint options
        checkpoint_dir=args.checkpoint_dir,
        resume_run_id=args.resume_run_id,
        run_name=args.run_name,
        # 🚀 Batch GPU
        batch_size=args.batch_size,
    )
    
    # 📦 SHARDING: Modify output path if this is a shard
    output_path = args.output
    if args.shard_index is not None:
        # Dodaj suffix _shard_X do nazwy pliku
        output_path_obj = Path(args.output)
        output_path = str(output_path_obj.with_name(
            f"{output_path_obj.stem}_shard_{args.shard_index}{output_path_obj.suffix}"
        ))
        print(f"📦 Output for shard {args.shard_index}: {output_path}")
    
    stats = builder.build_dataset(
        output_path=output_path,
        max_tracks=args.max_tracks,
        extract_features=extract_features,
        with_segments=with_segments,
        auto_merge=not args.no_auto_merge,
        # 📦 Sharding
        shard_index=args.shard_index,
        total_shards=args.total_shards,
        shard_by=args.shard_by,
    )
    
    # Save LLM cache at the end
    if builder.llm_enhancer:
        builder.llm_enhancer.save_cache()


if __name__ == "__main__":
    main()