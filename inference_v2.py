"""
🎵 Inference v2 - Generacja utworów z section-aware pipeline

Generuje pełne 2-4 minutowe utwory z koherentną strukturą sekcji.

Usage:
    # Basic usage
    python inference_v2.py \
        --prompt "Energiczny pop z chwytliwym refrenem, żeński wokal" \
        --duration 180 \
        --output ./output/my_song.wav
    
    # Z szablonem struktury
    python inference_v2.py \
        --prompt "Spokojny jazz z fortepianem" \
        --template verse_chorus \
        --duration 120
    
    # 🎤 STYLE TRANSFER: "In the style of artist X" (general sound)
    python inference_v2.py \
        --prompt "Electronic dance track" \
        --style_of "Metallica" \
        --duration 180
    
    # 🎤 VOICE CLONING: "Like artist X" (exact voice)
    python inference_v2.py \
        --prompt "Rock ballad with vocals" \
        --voice_clone "Metallica" \
        --duration 180
    
    # 🎤 VOICE CLONING from custom samples (folder)
    python inference_v2.py \
        --prompt "Pop song" \
        --voice_clone_samples ./my_voice_samples/ \
        --duration 120
    
    # 🎤 VOICE CLONING z pojedynczego pliku (szybki test)
    python inference_v2.py \
        --prompt "Test song" \
        --voice_clone_samples ./sample.wav \
        --duration 60

    # 📝 LYRICS: Generation with vocals for given text
    python inference_v2.py \
        --prompt "Emotional ballad with piano" \
        --lyrics "I walk alone through empty streets, searching for your light" \
        --voice_clone "Adele" \
        --duration 120

    # 📝 LYRICS in Polish (automatic language detection or --language)
    python inference_v2.py \
        --prompt "Polska ballada rockowa" \
        --lyrics "Idę sam przez puste ulice, szukam twego światła" \
        --language pl \
        --voice_clone "Doda" \
        --duration 120

    # 🎤 SING LYRICS: Generate instrumental + synthesized vocals (GPT-SoVITS)
    # Flow: LDM → Demucs (strip any vocals) → GPT-SoVITS vocals → Mix
    python inference_v2.py \\
        --prompt "Epic orchestral ballad with dramatic strings" \\
        --lyrics "I walk alone through empty streets, searching for your light" \\
        --sing_lyrics \\
        --singing_voice_ref ./my_voice_5sec.wav \\
        --singing_backend gpt_sovits \\
        --duration 120
    
    # 🐟 Fish Speech (best quality, #1 TTS-Arena2, supports emotions)
    python inference_v2.py \\
        --prompt "Pop song with emotional vocals" \\
        --lyrics "(excited) Every moment I think of you! (sighing)" \\
        --sing_lyrics \\
        --singing_voice_ref ./singer_sample.wav \\
        --singing_backend fish_speech \\
        --fish_speech_url http://localhost:8080 \\
        --mix_vocals 0.8 \\
        --duration 180
    
    # ⚡ Fast mode (skip Demucs vocal stripping - use if LDM is instrumental-only)
    python inference_v2.py \\
        --prompt "Instrumental electronic track" \\
        --lyrics "Dancing through the night" \\
        --sing_lyrics \\
        --singing_voice_ref ./voice.wav \\
        --no_strip_ldm_vocals \\
        --duration 120
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Optional, List, Dict, Any, Union, Tuple

import torch
import torchaudio
import numpy as np
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))


# ============================================
# 📝 Phoneme Processing (G2P)
# ============================================

class PhonemeProcessor:
    """
    Text to phoneme (IPA) conversion for inference.
    
    Supports:
    - Gruut: en, de, es, fr, it, ru, cs, nl, sv (better for Western languages)
    - eSpeak: pl, uk, pt, ja, ko, zh, tr, vi, hi (for the rest of the world)
    
    Auto-fallback: if Gruut doesn't support the language → eSpeak
    """
    
    # Languages supported by Gruut (preferred)
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
    
    # Languages for eSpeak (including Polish!)
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
        """Lazy check for Phonemizer/eSpeak"""
        if self._phonemizer_available is None:
            try:
                from phonemizer import phonemize
                from phonemizer.backend import EspeakBackend
                EspeakBackend.version()
                self._phonemizer_available = True
            except Exception:
                self._phonemizer_available = False
        return self._phonemizer_available
    
    def detect_language(self, text: str) -> str:
        """
        Prosta detekcja języka na podstawie charakterystycznych znaków.
        Fallback do 'en' jeśli nie rozpoznany.
        """
        # Polish characters
        if any(c in text for c in 'ąćęłńóśźżĄĆĘŁŃÓŚŹŻ'):
            return 'pl'
        # German
        if any(c in text for c in 'äöüßÄÖÜ'):
            return 'de'
        # French
        if any(c in text for c in 'àâçéèêëîïôùûüÿœæ'):
            return 'fr'
        # Spanish
        if any(c in text for c in 'áéíóúñ¿¡'):
            return 'es'
        # Russian/Cyrillic
        if any('\u0400' <= c <= '\u04FF' for c in text):
            return 'ru'
        # Japanese
        if any('\u3040' <= c <= '\u30FF' or '\u4E00' <= c <= '\u9FFF' for c in text):
            return 'ja'
        # Korean
        if any('\uAC00' <= c <= '\uD7AF' for c in text):
            return 'ko'
        # Chinese (if mostly Chinese chars)
        if any('\u4E00' <= c <= '\u9FFF' for c in text):
            return 'zh'
        
        return 'en'  # Default to English
    
    def text_to_phonemes(
        self,
        text: str,
        language: str = None,
    ) -> Dict[str, Any]:
        """
        Konwertuje tekst (lyrics) na fonemy IPA.
        
        Args:
            text: Tekst do konwersji (lyrics)
            language: Kod języka (np. 'pl', 'en', 'de'). None = auto-detect
            
        Returns:
            {
                'phonemes_ipa': str,  # Full phoneme string
                'words': [{'word': str, 'phonemes': List[str]}],
                'language': str,
                'backend': str,  # 'gruut', 'espeak', lub None
            }
        """
        if not text or not text.strip():
            return {
                'phonemes_ipa': '',
                'words': [],
                'language': language or 'en',
                'backend': None,
            }
        
        # Auto-detect language if not specified
        if language is None:
            language = self.detect_language(text)
            print(f"   🌍 Auto-detected language: {language}")
        
        lang_lower = language.lower() if language else 'en'
        
        # Select backend based on language
        if lang_lower in self.GRUUT_LANGUAGES and self._check_gruut():
            return self._phonemize_gruut(text, lang_lower)
        elif self._check_phonemizer():
            return self._phonemize_espeak(text, lang_lower)
        else:
            print("   ⚠️ No G2P backend available (install gruut or phonemizer)")
            return {
                'phonemes_ipa': '',
                'words': [],
                'language': language,
                'backend': None,
                'error': 'No G2P backend available',
            }
    
    def _phonemize_gruut(self, text: str, language: str) -> Dict[str, Any]:
        """Use Gruut for supported languages (en, de, es, fr, it, ru, cs, nl, sv)"""
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
                        all_phonemes.append(' ')  # Separator
            
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
                print(f"   ⚠️ Gruut error, falling back to eSpeak: {e}")
                return self._phonemize_espeak(text, language)
            return {
                'phonemes_ipa': '',
                'words': [],
                'language': language,
                'backend': 'gruut',
                'error': str(e),
            }
    
    def _phonemize_espeak(self, text: str, language: str) -> Dict[str, Any]:
        """Use Phonemizer/eSpeak (especially for Polish!)"""
        from phonemizer import phonemize
        
        # Map language to espeak code
        espeak_lang = self.ESPEAK_LANGUAGES.get(language, language)
        
        # Check if language is supported
        try:
            from phonemizer.backend import EspeakBackend
            supported = EspeakBackend.supported_languages()
            if espeak_lang not in supported:
                espeak_lang = espeak_lang.split('-')[0]
                if espeak_lang not in supported:
                    print(f"   ⚠️ Language '{language}' not in eSpeak, using 'en'")
                    espeak_lang = 'en-us'
        except:
            pass
        
        try:
            phonemes_ipa = phonemize(
                text,
                language=espeak_lang,
                backend='espeak',
                strip=True,
                preserve_punctuation=False,
                language_switch='remove-flags',
            )
            
            # Per-word phonemization
            words = text.split()
            words_data = []
            
            for word in words:
                if word.strip():
                    try:
                        word_phonemes = phonemize(
                            word,
                            language=espeak_lang,
                            backend='espeak',
                            strip=True,
                            preserve_punctuation=False,
                        )
                        phoneme_list = list(word_phonemes.replace(' ', ''))
                        words_data.append({
                            'word': word,
                            'phonemes': phoneme_list,
                        })
                    except:
                        pass
            
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


# Globalny processor (lazy init)
_phoneme_processor = None

def get_phoneme_processor() -> PhonemeProcessor:
    """Zwraca singleton PhonemeProcessor"""
    global _phoneme_processor
    if _phoneme_processor is None:
        _phoneme_processor = PhonemeProcessor()
    return _phoneme_processor


# ============================================
# 🎤 Voice Embedding Loading Functions
# ============================================

def load_artist_embeddings(embeddings_path: str) -> Dict[str, Any]:
    """Loads artist_embeddings.json file"""
    path = Path(embeddings_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Artist embeddings not found: {embeddings_path}\n"
            f"Generate them with: python tools_v2/generate_artist_embeddings.py"
        )
    
    with open(path, 'r') as f:
        return json.load(f)


def get_artist_embedding(
    artist_name: str,
    embeddings: Dict[str, Any],
    mode: str = "style",  # "style", "voice", or "voice_separated"
) -> Optional[np.ndarray]:
    """
    Pobiera embedding artysty.
    
    Args:
        artist_name: Nazwa artysty
        embeddings: Załadowane artist_embeddings
        mode: 
            - "style" (dla style_of) - 256-dim resemblyzer
            - "voice" (dla voice_clone) - 256-dim resemblyzer z mixu
            - "voice_separated" (dla voice_as) - 192-dim ECAPA-TDNN z separated vocals
    
    Returns:
        numpy array z embeddingiem lub None
    """
    # Case-insensitive search
    artist_lower = artist_name.lower()
    matched_artist = None
    
    for name in embeddings.keys():
        if name.lower() == artist_lower:
            matched_artist = name
            break
    
    if not matched_artist:
        # Partial match
        for name in embeddings.keys():
            if artist_lower in name.lower() or name.lower() in artist_lower:
                matched_artist = name
                print(f"   🔍 Partial match: '{artist_name}' → '{matched_artist}'")
                break
    
    if not matched_artist:
        print(f"   ⚠️ Artist '{artist_name}' not found in embeddings")
        print(f"   Available artists: {list(embeddings.keys())[:10]}...")
        return None
    
    data = embeddings[matched_artist]
    
    if mode == "voice_separated":
        # 192-dim ECAPA-TDNN from Demucs separated vocals - best quality
        emb = data.get('voice_embedding_separated')
        if emb is None:
            print(f"   ⚠️ Artist '{matched_artist}' has no voice_embedding_separated")
            print(f"   This requires Demucs separation during dataset build")
            print(f"   Falling back to voice_embedding")
            emb = data.get('voice_embedding')
    elif mode == "voice":
        emb = data.get('voice_embedding')
        if emb is None:
            print(f"   ⚠️ Artist '{matched_artist}' has no voice_embedding (no Demucs separation)")
            print(f"   Falling back to style_embedding")
            emb = data.get('style_embedding')
    else:
        emb = data.get('style_embedding')
    
    if emb:
        return np.array(emb)
    return None


def extract_embedding_from_samples(
    samples_path: Union[str, Path],
    device: str = "cpu",
) -> Optional[np.ndarray]:
    """
    Ekstrahuje embedding z własnych sampli użytkownika.
    
    Args:
        samples_path: Ścieżka do pliku WAV lub folderu z plikami
        device: Device dla modelu
    
    Returns:
        Uśredniony embedding
    """
    import librosa
    
    samples_path = Path(samples_path)
    
    # Zbierz pliki audio
    audio_files = []
    if samples_path.is_file():
        audio_files = [samples_path]
        print(f"   ⚠️ Single sample - voice cloning quality may be limited")
    elif samples_path.is_dir():
        for ext in ['.wav', '.mp3', '.flac', '.ogg']:
            audio_files.extend(samples_path.glob(f'*{ext}'))
        print(f"   📁 Found {len(audio_files)} audio files in folder")
    else:
        print(f"   ❌ Path not found: {samples_path}")
        return None
    
    if not audio_files:
        print(f"   ❌ No audio files found")
        return None
    
    # Lazy load VocalProcessor
    try:
        from build_dataset_v2 import VocalProcessor
        processor = VocalProcessor(
            sample_rate=16000,
            use_demucs=True,  # Separate vocals for better quality
        )
    except ImportError:
        print("   ⚠️ VocalProcessor not available, using basic extraction")
        processor = None
    
    embeddings = []
    
    for audio_file in tqdm(audio_files, desc="   Extracting embeddings"):
        try:
            # Load audio
            y, sr = librosa.load(str(audio_file), sr=16000)
            
            if len(y) < sr:  # Min 1 second
                continue
            
            if processor:
                result = processor.extract_all_embeddings(y, sr)
                # Prefer separated embedding (don't use 'or' with numpy arrays!)
                emb = result.get('embedding_separated')
                if emb is None:
                    emb = result.get('embedding_mix')
            else:
                # Fallback - basic resemblyzer
                try:
                    from resemblyzer import VoiceEncoder, preprocess_wav
                    encoder = VoiceEncoder(device=device)
                    processed = preprocess_wav(y)
                    emb = encoder.embed_utterance(processed)
                except ImportError:
                    print("   ❌ Resemblyzer not installed")
                    return None
            
            if emb is not None:
                embeddings.append(emb)
                
        except Exception as e:
            print(f"   ⚠️ Error processing {audio_file.name}: {e}")
    
    if not embeddings:
        print(f"   ❌ No embeddings extracted")
        return None
    
    # Average
    avg_embedding = np.mean(embeddings, axis=0)
    print(f"   ✅ Extracted and averaged {len(embeddings)} embeddings")
    
    return avg_embedding


def load_voice_conditioning(
    args,
    artist_embeddings: Optional[Dict[str, Any]] = None,
    device: str = "cpu",
) -> Optional[torch.Tensor]:
    """
    Ładuje voice conditioning na podstawie argumentów.
    
    Priorytety:
    1. --voice_clone_samples (własne pliki użytkownika)
    2. --voice_as (artysta z vocals/artist/embeddings.json lub artist_embeddings.json)
    3. --voice_clone (artysta z datasetu, 256-dim resemblyzer)
    4. --style_of (artysta z datasetu, style transfer)
    5. --voice_ref (legacy, pojedynczy plik)
    
    Returns:
        Tensor z embeddingiem lub None
    """
    embedding = None
    mode_str = ""
    
    # 1. User's custom samples
    if args.voice_clone_samples:
        print(f"\n🎤 Extracting voice embedding from: {args.voice_clone_samples}")
        embedding = extract_embedding_from_samples(args.voice_clone_samples, device)
        mode_str = "voice_clone (custom samples)"
    
    # 2. Voice as - pre-computed separated embedding (192-dim ECAPA-TDNN)
    elif args.voice_as:
        print(f"\n🎤 Loading separated voice embedding for: {args.voice_as}")
        
        # Najpierw szukaj w vocals/artist/embeddings.json
        artist_name_clean = "".join(
            c if c.isalnum() or c in (' ', '-', '_') else '_' 
            for c in args.voice_as
        ).strip().replace(' ', '_').lower()
        
        vocals_dir = Path(args.vocals_dir)
        artist_emb_path = vocals_dir / artist_name_clean / "embeddings.json"
        
        if artist_emb_path.exists():
            print(f"   📂 Found in vocals/{artist_name_clean}/")
            with open(artist_emb_path, 'r') as f:
                data = json.load(f)
            emb = data.get('voice_embedding_separated')
            if emb is not None:
                embedding = np.array(emb)
                mode_str = "voice_as (192-dim ECAPA-TDNN from vocals/)"
            else:
                emb = data.get('voice_embedding') or data.get('style_embedding')
                if emb is not None:
                    embedding = np.array(emb)
                    mode_str = "voice_as (256-dim fallback from vocals/)"
        else:
            # Fallback to artist_embeddings.json
            if artist_embeddings is None:
                artist_embeddings = load_artist_embeddings(args.artist_embeddings_path)
            embedding = get_artist_embedding(args.voice_as, artist_embeddings, mode="voice_separated")
            mode_str = "voice_as (192-dim ECAPA-TDNN separated)"
    
    # 3. Voice clone z datasetu (256-dim resemblyzer)
    elif args.voice_clone:
        print(f"\n🎤 Loading voice clone embedding for: {args.voice_clone}")
        if artist_embeddings is None:
            artist_embeddings = load_artist_embeddings(args.artist_embeddings_path)
        embedding = get_artist_embedding(args.voice_clone, artist_embeddings, mode="voice")
        mode_str = "voice_clone (256-dim resemblyzer)"
    
    # 4. Style transfer z datasetu
    elif args.style_of:
        print(f"\n🎤 Loading style embedding for: {args.style_of}")
        if artist_embeddings is None:
            artist_embeddings = load_artist_embeddings(args.artist_embeddings_path)
        embedding = get_artist_embedding(args.style_of, artist_embeddings, mode="style")
        mode_str = "style_of (256-dim style transfer)"
    
    # 5. Legacy single file
    elif args.voice_ref:
        print(f"\n🎤 Loading voice reference from: {args.voice_ref}")
        embedding = extract_embedding_from_samples(args.voice_ref, device)
        mode_str = "voice_ref (legacy)"
    
    if embedding is not None:
        print(f"   Mode: {mode_str}")
        print(f"   Embedding shape: {embedding.shape}")
        return torch.from_numpy(embedding).float().to(device)
    
    return None


# ============================================
# Model Loading
# ============================================

def load_models(args, device):
    """Loading all models with automatic configuration detection"""
    from models.audio_vae import AudioVAE
    from models.vocoder import HiFiGAN
    from models_v2.composition_planner import CompositionPlanner
    from models_v2.latent_diffusion import UNetV2, LatentDiffusionV2
    from models_v2.text_encoder import EnhancedMusicEncoder
    
    models = {}
    use_fp16 = getattr(args, 'fp16', False) and device.type == 'cuda'
    dtype = torch.float16 if use_fp16 else torch.float32
    
    if use_fp16:
        print("🚀 Using FP16 inference (half precision)")
    
    # VAE - load checkpoint to determine latent_dim
    print("📦 Loading VAE...")
    vae_ckpt = torch.load(args.vae_checkpoint, map_location=device)
    
    # Wykryj latent_dim z checkpointu
    latent_dim = 128  # default
    latent_scale = 1.0  # default (brak skalowania)
    if 'config' in vae_ckpt:
        latent_dim = vae_ckpt['config'].get('latent_dim', 128)
        latent_scale = vae_ckpt['config'].get('latent_scale', 1.0)
    elif 'latent_dim' in vae_ckpt:
        latent_dim = vae_ckpt['latent_dim']
    else:
        # Try to detect from weights
        state_dict = vae_ckpt.get('model_state_dict', vae_ckpt)
        for key in state_dict:
            if 'conv_out.weight' in key:
                # conv_out output = latent_dim * 2
                latent_dim = state_dict[key].shape[0] // 2
                break
    
    print(f"   Detected latent_dim: {latent_dim}")
    print(f"   Using latent_scale: {latent_scale:.4f}")
    
    vae = AudioVAE(latent_dim=latent_dim).to(device)
    vae.load_state_dict(vae_ckpt.get('model_state_dict', vae_ckpt))
    vae.eval()
    if use_fp16:
        vae = vae.half()
    models['vae'] = vae
    models['latent_dim'] = latent_dim
    models['latent_scale'] = latent_scale  # ✅ Save for use during decode
    models['dtype'] = dtype
    
    # Composition Planner
    print("📦 Loading Composition Planner...")
    if Path(args.planner_checkpoint).exists():
        planner = CompositionPlanner.from_pretrained(
            args.planner_checkpoint,
            device=str(device),
        )
    else:
        print(f"   ⚠️ Checkpoint not found: {args.planner_checkpoint}")
        print(f"   Using default CompositionPlanner with template generation")
        from models_v2.composition_planner import CompositionTransformer
        model = CompositionTransformer()
        planner = CompositionPlanner(model, device=str(device))
    models['planner'] = planner
    
    # Text Encoder
    print("📦 Loading Text Encoder...")
    text_encoder = EnhancedMusicEncoder(
        use_clap=args.use_clap,
        use_t5_fallback=True,
        device=device,
    )
    models['text_encoder'] = text_encoder
    
    # U-Net v2 + LDM - automatyczna detekcja konfiguracji z checkpointu
    print("📦 Loading Latent Diffusion v2...")
    ldm_ckpt = torch.load(args.ldm_checkpoint, map_location=device, weights_only=False)
    
    # Detect configuration from checkpoint
    ldm_config = ldm_ckpt.get('config', {})
    state_dict = ldm_ckpt.get('model_state_dict', ldm_ckpt)
    
    # model_channels - priorytet: checkpoint > argument > default
    if 'model_channels' in ldm_config:
        model_channels = ldm_config['model_channels']
        print(f"   Detected model_channels from checkpoint: {model_channels}")
    elif hasattr(args, 'model_channels') and args.model_channels:
        model_channels = args.model_channels
        print(f"   Using model_channels from argument: {model_channels}")
    else:
        # Try to detect from weights (conv_in.weight shape)
        model_channels = 320  # default
        for key in state_dict:
            if 'unet.conv_in.weight' in key or 'conv_in.weight' in key:
                # conv_in.weight shape: [model_channels, in_channels, 3, 3]
                model_channels = state_dict[key].shape[0]
                print(f"   Detected model_channels from weights: {model_channels}")
                break
    
    # num_timesteps - wykryj z betas/alphas w checkpoincie
    num_timesteps = ldm_config.get('num_timesteps', 1000)
    for key in state_dict:
        if key in ['betas', 'alphas', 'alphas_cumprod']:
            num_timesteps = state_dict[key].shape[0]
            print(f"   Detected num_timesteps from checkpoint: {num_timesteps}")
            break
    
    # channel_mult - z checkpointu lub default
    channel_mult = ldm_config.get('channel_mult', [1, 2, 4, 4])
    
    # use_voice_stream, use_dual_voice - z checkpointu lub default
    use_voice_stream = ldm_config.get('use_voice_stream', True)
    use_dual_voice = ldm_config.get('use_dual_voice', True)
    
    print(f"   Config: latent_dim={latent_dim}, model_channels={model_channels}, timesteps={num_timesteps}")
    print(f"   Voice: stream={use_voice_stream}, dual={use_dual_voice}")
    
    unet = UNetV2(
        in_channels=latent_dim,
        out_channels=latent_dim,
        model_channels=model_channels,
        channel_mult=channel_mult,
        context_dim=768,
        use_context_fusion=True,
        use_voice_stream=use_voice_stream,
        use_dual_voice=use_dual_voice,
    ).to(device)
    
    ldm = LatentDiffusionV2(unet, num_timesteps=num_timesteps).to(device)
    ldm.load_state_dict(state_dict)
    ldm.eval()
    if use_fp16:
        ldm = ldm.half()
    models['ldm'] = ldm
    
    # Vocoder (HiFi-GAN) - always FP32 for audio quality
    print("📦 Loading HiFi-GAN Vocoder...")
    vocoder = HiFiGAN().to(device)
    if args.vocoder_checkpoint and Path(args.vocoder_checkpoint).exists():
        vocoder_ckpt = torch.load(args.vocoder_checkpoint, map_location=device, weights_only=False)
        # HiFiGAN checkpoint may have 'generator' as key
        state_dict = vocoder_ckpt.get('generator', vocoder_ckpt.get('model_state_dict', vocoder_ckpt))
        vocoder.load_state_dict(state_dict)
    models['vocoder'] = vocoder
    
    return models


def generate_composition_plan(
    prompt: str,
    duration: float,
    template: Optional[str],
    planner,
    device,
):
    """Generacja planu kompozycji"""
    print("\n📝 Generating composition plan...")
    
    if template:
        # Use template
        plan = planner.generate_from_template(
            template_name=template,
            target_duration=duration,
            tempo=120,
        )
    else:
        # Generuj automatycznie na podstawie promptu
        plan = planner.generate(
            prompt=prompt,
            target_duration=duration,
            genre='pop',  # TODO: ekstrakcja z promptu
            mood='energetic',  # TODO: ekstrakcja z promptu
        )
    
    print(f"   Generated {len(plan.sections)} sections:")
    for i, section in enumerate(plan.sections):
        print(f"     {i+1}. {section.section_type} - {section.duration:.1f}s "
              f"(tempo={section.tempo:.0f}, energy={section.energy:.2f})")
    
    total_dur = sum(s.duration for s in plan.sections)
    print(f"   Total duration: {total_dur:.1f}s")
    
    return plan


def generate_section_audio(
    section,
    prompt: str,
    position: float,
    context_latent: Optional[torch.Tensor],
    models: dict,
    device: torch.device,
    sample_rate: int = 22050,
    voice_embedding: Optional[torch.Tensor] = None,
    phonemes_ipa: Optional[str] = None,
) -> tuple:
    """
    Generuje audio dla jednej sekcji.
    
    Args:
        section: SectionPlan z danymi sekcji
        prompt: Text prompt
        position: Pozycja w utworze (0-1)
        context_latent: Latent z poprzedniej sekcji (dla ciągłości)
        models: Załadowane modele
        device: torch device
        sample_rate: Sample rate
        voice_embedding: Voice conditioning tensor
        phonemes_ipa: Fonemy IPA dla śpiewu (opcjonalne)
    """
    ldm = models['ldm']
    vae = models['vae']
    vocoder = models['vocoder']
    text_encoder = models['text_encoder']
    latent_dim = models.get('latent_dim', 128)
    dtype = models.get('dtype', torch.float32)
    
    # Text embedding z kontekstem sekcji
    section_prompt = f"{prompt}, {section.section_type} section"
    
    text_embed = text_encoder(
        [section_prompt],
        [section.section_type],
        torch.tensor([position]).to(device),
        torch.tensor([section.tempo]).to(device),
        torch.tensor([section.energy]).to(device),
    )
    
    # Conversion to proper dtype for FP16
    if dtype == torch.float16:
        text_embed = text_embed.half()
    
    # Sample latent
    segment_duration = min(section.duration, 10.0)  # Max 10s per segment
    latent_time = int(segment_duration * sample_rate / 256 / 8)  # VAE 8x time downsampling
    latent_height = 16  # VAE spatial compression: 128 mels -> 16
    
    # Prepare voice embeddings (distinguish resemblyzer vs ecapa)
    voice_emb = None
    voice_emb_separated = None
    if voice_embedding is not None:
        ve = voice_embedding.to(dtype)  # Conversion to proper dtype
        if voice_embedding.shape[-1] == 256:
            # Resemblyzer embedding (mix-based)
            voice_emb = ve.unsqueeze(0) if ve.dim() == 1 else ve
        elif voice_embedding.shape[-1] == 192:
            # ECAPA-TDNN embedding (separated vocals)
            voice_emb_separated = ve.unsqueeze(0) if ve.dim() == 1 else ve
        else:
            # Unknown dimension, try as resemblyzer
            voice_emb = ve.unsqueeze(0) if ve.dim() == 1 else ve
    
    # Sample latent with full v3 conditioning
    latent = ldm.sample_section(
        shape=(1, latent_dim, latent_height, latent_time),  # 4D: [B, C, H, W]
        text_embed=text_embed,
        section_type=section.section_type,
        position=position,
        energy=section.energy,
        tempo=section.tempo,
        key=section.key,
        has_vocals=section.has_vocals,
        voice_emb=voice_emb,
        voice_emb_separated=voice_emb_separated,
        context_latent=context_latent,
        phonemes_ipa=[phonemes_ipa] if phonemes_ipa else None,  # 📝 Lyrics support
        cfg_scale=7.5,
        device=str(device),  # Konwertuj na string
        verbose=False,
    )
    
    # ✅ Reverse latent scaling before decode (undo training scale)
    latent_scale = models.get('latent_scale', 1.0)
    if latent_scale != 1.0:
        latent = latent / latent_scale
    
    # Decode latent to mel (konwersja do FP32 dla vocodera)
    with torch.no_grad():
        latent_fp32 = latent.float() if dtype == torch.float16 else latent
        mel = vae.float().decode(latent_fp32) if dtype == torch.float16 else vae.decode(latent)
    
    # Vocoder oczekuje 3D [B, mels, time], VAE zwraca 4D [B, 1, mels, time]
    if mel.dim() == 4:
        mel = mel.squeeze(1)  # [B, mels, time]
    
    # Vocoder to waveform (inference mode) - zawsze FP32
    with torch.no_grad():
        audio = vocoder.inference(mel.float())
    
    return audio, latent


def generate_full_song(
    prompt: str,
    plan,
    models: dict,
    device: torch.device,
    voice_embedding: Optional[torch.Tensor] = None,  # Renamed from voice_ref
    phonemes_ipa: Optional[str] = None,  # 📝 Lyrics phonemes
    sample_rate: int = 22050,
) -> torch.Tensor:
    """
    Generuje pełny utwór sekcja po sekcji.
    
    Args:
        prompt: Text prompt
        plan: CompositionPlan z sekcjami
        models: Załadowane modele
        device: torch device
        voice_embedding: Voice embedding tensor (256 lub 192 dim)
        phonemes_ipa: Fonemy IPA dla śpiewu (opcjonalne)
        sample_rate: Sample rate
    """
    print("\n🎶 Generating audio sections...")
    
    if voice_embedding is not None:
        print(f"   🎤 Using voice conditioning (dim={voice_embedding.shape[-1]})")
    
    if phonemes_ipa:
        print(f"   📝 Using lyrics phonemes ({len(phonemes_ipa)} chars)")
    
    all_audio = []
    context_latent = None
    
    total_duration = sum(s.duration for s in plan.sections)
    
    for i, section in enumerate(tqdm(plan.sections, desc="Generating")):
        position = sum(s.duration for s in plan.sections[:i]) / total_duration
        
        # Generate section
        audio, latent = generate_section_audio(
            section=section,
            prompt=prompt,
            position=position,
            context_latent=context_latent,
            models=models,
            device=device,
            sample_rate=sample_rate,
            voice_embedding=voice_embedding,  # Pass to section generator
            phonemes_ipa=phonemes_ipa,  # 📝 Pass lyrics
        )
        
        # Handle longer sections (>10s) by generating multiple segments
        remaining_duration = section.duration - 10.0
        segment_count = 1
        
        while remaining_duration > 0:
            segment_count += 1
            segment_position = position + (10.0 * (segment_count - 1)) / total_duration
            
            seg_audio, latent = generate_section_audio(
                section=section,
                prompt=prompt,
                position=segment_position,
                context_latent=latent,  # Use previous segment as context
                models=models,
                device=device,
                sample_rate=sample_rate,
                voice_embedding=voice_embedding,  # Pass voice embedding
                phonemes_ipa=phonemes_ipa,  # 📝 Pass lyrics
            )
            
            # Crossfade
            crossfade_samples = int(0.5 * sample_rate)  # 500ms crossfade
            if audio.shape[-1] > crossfade_samples and seg_audio.shape[-1] > crossfade_samples:
                fade_out = torch.linspace(1, 0, crossfade_samples, device=device)
                fade_in = torch.linspace(0, 1, crossfade_samples, device=device)
                
                audio[..., -crossfade_samples:] *= fade_out
                seg_audio[..., :crossfade_samples] *= fade_in
                
                # Overlap-add
                overlap = audio[..., -crossfade_samples:] + seg_audio[..., :crossfade_samples]
                audio = torch.cat([audio[..., :-crossfade_samples], overlap, seg_audio[..., crossfade_samples:]], dim=-1)
            else:
                audio = torch.cat([audio, seg_audio], dim=-1)
            
            remaining_duration -= 10.0
        
        all_audio.append(audio)
        context_latent = latent
    
    # Concatenate all sections with crossfade
    print("\n🔗 Concatenating sections...")
    final_audio = all_audio[0]
    
    for i, audio in enumerate(all_audio[1:], 1):
        crossfade_samples = int(0.3 * sample_rate)  # 300ms between sections
        
        if final_audio.shape[-1] > crossfade_samples and audio.shape[-1] > crossfade_samples:
            fade_out = torch.linspace(1, 0, crossfade_samples, device=device)
            fade_in = torch.linspace(0, 1, crossfade_samples, device=device)
            
            final_audio[..., -crossfade_samples:] *= fade_out
            audio[..., :crossfade_samples] *= fade_in
            
            overlap = final_audio[..., -crossfade_samples:] + audio[..., :crossfade_samples]
            final_audio = torch.cat([final_audio[..., :-crossfade_samples], overlap, audio[..., crossfade_samples:]], dim=-1)
        else:
            final_audio = torch.cat([final_audio, audio], dim=-1)
    
    
    return final_audio


# ============================================
# 🎤 Singing Voice Synthesis (GPT-SoVITS / Fish Speech)
# ============================================

def synthesize_vocals(
    lyrics_text: str,
    reference_audio: str,
    language: str = None,
    backend: str = "gpt_sovits",
    gpt_sovits_url: str = "http://localhost:9880",
    fish_speech_url: str = "http://localhost:8080",
    fish_speech_api_key: str = None,
    device: str = "cpu",
) -> Tuple[torch.Tensor, int]:
    """
    Synthesize singing vocals from lyrics using GPT-SoVITS, Fish Speech, or other backends.
    
    Backends:
    - gpt_sovits: GPT-SoVITS (MIT, 5s sample, EN/JA/KO/ZH)
    - fish_speech: Fish Speech / OpenAudio S1 (#1 TTS-Arena2, Apache 2.0)
    - coqui: XTTS v2 (Apache 2.0, multilingual)
    - elevenlabs: ElevenLabs API (best quality, paid)
    
    Fish Speech Features:
    - #1 on TTS-Arena2 benchmark
    - Zero-shot: 10-30s reference audio
    - Emotion control: (angry), (excited), (sad), etc.
    - Multilingual: EN, JA, KO, ZH, FR, DE, AR, ES
    
    Args:
        lyrics_text: Text to sing (supports emotion markers for fish_speech)
        reference_audio: Path to voice reference (5-30s WAV/MP3)
        language: Language code (auto-detected if None)
        backend: "gpt_sovits", "fish_speech", "coqui", or "elevenlabs"
        gpt_sovits_url: GPT-SoVITS API server URL
        fish_speech_url: Fish Speech API server URL
        fish_speech_api_key: Fish Audio cloud API key (optional)
        device: cpu/cuda/mps
        
    Returns:
        Tuple[audio_tensor, sample_rate]
        
    Example:
        vocals, sr = synthesize_vocals(
            lyrics_text="I walk alone through empty streets",
            reference_audio="./voice_sample.wav",
            backend="gpt_sovits"
        )
    """
    from models.voice_synthesis import VoiceSynthesizer
    
    print(f"\n🎤 Synthesizing vocals with {backend}...")
    print(f"   Lyrics: {lyrics_text[:80]}{'...' if len(lyrics_text) > 80 else ''}")
    print(f"   Reference: {Path(reference_audio).name}")
    
    # Initialize synthesizer
    synth_kwargs = {
        "backend": backend,
        "device": device,
    }
    
    if backend == "gpt_sovits":
        synth_kwargs["gpt_sovits_url"] = gpt_sovits_url
    elif backend == "fish_speech":
        synth_kwargs["fish_speech_url"] = fish_speech_url
        if fish_speech_api_key:
            synth_kwargs["api_key"] = fish_speech_api_key
    
    synth = VoiceSynthesizer(**synth_kwargs)
    
    # Register the reference voice
    synth.register_voice(
        name="singing_voice",
        reference_audio=reference_audio,
        description="Voice for singing synthesis",
        source_type="recording",
    )
    
    # Auto-detect language if not specified
    if language is None:
        processor = get_phoneme_processor()
        language = processor.detect_language(lyrics_text)
        print(f"   Auto-detected language: {language}")
    
    # Synthesize
    audio = synth.synthesize(
        text=lyrics_text,
        voice="singing_voice",
        language=language,
        speed=1.0,
    )
    
    # Get sample rate from model
    if backend == "gpt_sovits":
        sample_rate = 32000  # GPT-SoVITS v2/v3 default
    elif backend == "fish_speech":
        sample_rate = 44100  # Fish Speech default
    elif backend == "coqui":
        sample_rate = 22050  # XTTS default
    else:
        sample_rate = 22050  # Default
    
    print(f"   ✅ Synthesized {audio.shape[-1] / sample_rate:.1f}s of vocals")
    
    return audio, sample_rate


def extract_clean_instrumental(
    audio: torch.Tensor,
    sample_rate: int = 22050,
    device: str = "cpu",
) -> Tuple[torch.Tensor, bool]:
    """
    Extract clean instrumental from LDM output using Demucs.
    
    If LDM accidentally generated vocal-like sounds, this removes them
    to prevent double-vocals when mixing with TTS.
    
    Args:
        audio: Audio tensor from LDM
        sample_rate: Sample rate of audio
        device: cpu/cuda/mps
        
    Returns:
        Tuple[instrumental_audio, had_vocals]
        - instrumental_audio: Clean instrumental (vocals removed if any)
        - had_vocals: True if vocals were detected and removed
    """
    print(f"\n🔍 Checking LDM output for accidental vocals...")
    
    try:
        from models.voice_synthesis import VoiceExtractorFromSong
        import tempfile
        
        # Save audio to temp file for Demucs
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
            temp_path = f.name
        
        # Ensure proper shape for saving
        audio_save = audio.squeeze()
        if audio_save.dim() == 1:
            audio_save = audio_save.unsqueeze(0)
        torchaudio.save(temp_path, audio_save.cpu(), sample_rate)
        
        # Initialize Demucs
        extractor = VoiceExtractorFromSong(
            separation_model="demucs",
            device=device,
        )
        extractor._init_separator()
        
        # Load and prepare audio for Demucs (needs 44100 Hz)
        audio_44k, sr = torchaudio.load(temp_path)
        if sr != 44100:
            audio_44k = torchaudio.functional.resample(audio_44k, sr, 44100)
        
        # Ensure stereo
        if audio_44k.shape[0] == 1:
            audio_44k = audio_44k.repeat(2, 1)
        
        # Run Demucs separation
        audio_44k = audio_44k.unsqueeze(0).to(device)
        
        with torch.no_grad():
            sources = extractor._separator["apply"](
                extractor._separator["model"],
                audio_44k,
                device=device,
            )
        
        # Sources order: drums, bass, other, vocals
        # Get everything EXCEPT vocals (sum drums + bass + other)
        instrumental = sources[0, 0] + sources[0, 1] + sources[0, 2]  # [2, samples]
        vocals = sources[0, 3]  # [2, samples]
        
        # Check if there were actual vocals
        vocals_rms = torch.sqrt(torch.mean(vocals ** 2)).item()
        instrumental_rms = torch.sqrt(torch.mean(instrumental ** 2)).item()
        
        # Vocal detection threshold
        vocal_ratio = vocals_rms / (instrumental_rms + 1e-8)
        had_vocals = vocal_ratio > 0.05  # If vocals are >5% of instrumental energy
        
        if had_vocals:
            print(f"   ⚠️ Vocals detected in LDM output! (ratio: {vocal_ratio:.2%})")
            print(f"   🧹 Removing LDM vocals to prevent double-voice...")
            
            # Resample instrumental back to original sample rate
            instrumental = instrumental.cpu()
            if sample_rate != 44100:
                instrumental = torchaudio.functional.resample(
                    instrumental, 44100, sample_rate
                )
            
            # Convert to mono if original was mono
            if audio.squeeze().dim() == 1:
                instrumental = instrumental.mean(dim=0)
            
            print(f"   ✅ Clean instrumental extracted")
        else:
            print(f"   ✅ No vocals detected (ratio: {vocal_ratio:.2%}) - using original")
            instrumental = audio
        
        # Cleanup
        import os
        os.unlink(temp_path)
        
        return instrumental, had_vocals
        
    except Exception as e:
        print(f"   ⚠️ Demucs separation failed: {e}")
        print(f"   Using original audio (risk of double-vocals)")
        return audio, False


def detect_vocal_regions(
    audio: torch.Tensor,
    sample_rate: int = 22050,
    hop_length: int = 512,
    energy_threshold: float = 0.02,
    min_silence_duration: float = 0.3,
    composition_plan = None,  # NEW: Use plan if available
) -> List[Tuple[float, float]]:
    """
    Detect regions in instrumental where vocals should be placed.
    
    Priority order:
    1. If composition_plan provided → use sections with has_vocals=True
    2. Fallback: analyze energy envelope
    
    Args:
        audio: Instrumental audio tensor
        sample_rate: Sample rate
        hop_length: Analysis hop length
        energy_threshold: Threshold for detecting low-energy regions
        min_silence_duration: Minimum duration of silence to consider
        composition_plan: Optional CompositionPlan with section info
        
    Returns:
        List of (start_time, end_time) tuples for vocal regions
    """
    import numpy as np
    
    # =========================================
    # BEST: Use composition plan (knows structure!)
    # =========================================
    if composition_plan is not None and hasattr(composition_plan, 'sections'):
        regions = []
        current_time = 0.0
        
        for section in composition_plan.sections:
            section_start = current_time
            section_end = current_time + section.duration
            
            # Only include sections marked for vocals
            if getattr(section, 'has_vocals', False):
                # Skip first 0.5s of each section (let music establish)
                vocal_start = section_start + 0.5
                # End 0.5s before section ends (outro transition)
                vocal_end = section_end - 0.5
                
                if vocal_end > vocal_start:
                    regions.append((vocal_start, vocal_end))
                    print(f"   📍 {section.section_type}: {vocal_start:.1f}s - {vocal_end:.1f}s")
            
            current_time = section_end
        
        if regions:
            print(f"   ✅ Using CompositionPlan: {len(regions)} vocal sections")
            return regions
        else:
            print(f"   ⚠️ No vocal sections in plan, falling back to energy analysis")
    
    # =========================================
    # FALLBACK: Energy-based detection
    # =========================================
    audio_np = audio.squeeze().cpu().numpy()
    
    # Calculate RMS energy in frames
    frame_length = hop_length * 4
    num_frames = len(audio_np) // hop_length
    
    energy = []
    for i in range(num_frames):
        start = i * hop_length
        end = min(start + frame_length, len(audio_np))
        frame = audio_np[start:end]
        rms = np.sqrt(np.mean(frame ** 2))
        energy.append(rms)
    
    energy = np.array(energy)
    
    # Normalize energy
    max_energy = np.max(energy) + 1e-8
    energy_norm = energy / max_energy
    
    # Find regions with moderate energy (good for vocals)
    # Not too quiet (no music), not too loud (would clash)
    good_for_vocals = (energy_norm > 0.1) & (energy_norm < 0.8)
    
    # Convert frames to time
    frame_duration = hop_length / sample_rate
    
    # Find continuous regions
    regions = []
    in_region = False
    region_start = 0
    
    for i, good in enumerate(good_for_vocals):
        if good and not in_region:
            in_region = True
            region_start = i * frame_duration
        elif not good and in_region:
            in_region = False
            region_end = i * frame_duration
            if region_end - region_start >= min_silence_duration:
                regions.append((region_start, region_end))
    
    # Handle last region
    if in_region:
        region_end = len(good_for_vocals) * frame_duration
        if region_end - region_start >= min_silence_duration:
            regions.append((region_start, region_end))
    
    # =========================================
    # ENHANCEMENT: Snap to beats
    # =========================================
    try:
        import librosa
        
        # Detect beats
        tempo, beat_frames = librosa.beat.beat_track(
            y=audio_np, 
            sr=sample_rate,
            hop_length=hop_length
        )
        beat_times = librosa.frames_to_time(beat_frames, sr=sample_rate, hop_length=hop_length)
        
        if len(beat_times) > 0:
            # Snap region starts/ends to nearest beat
            snapped_regions = []
            for start, end in regions:
                # Find nearest beat to start
                start_idx = np.argmin(np.abs(beat_times - start))
                snapped_start = beat_times[start_idx]
                
                # Find nearest beat to end
                end_idx = np.argmin(np.abs(beat_times - end))
                snapped_end = beat_times[end_idx]
                
                if snapped_end > snapped_start + 1.0:  # At least 1 second
                    snapped_regions.append((snapped_start, snapped_end))
            
            if snapped_regions:
                print(f"   🎵 Beat-aligned {len(snapped_regions)} regions (tempo: {tempo:.0f} BPM)")
                return snapped_regions
    except Exception as e:
        print(f"   ⚠️ Beat detection failed: {e}, using raw regions")
    
    return regions


def align_vocals_to_instrumental(
    vocals: torch.Tensor,
    vocals_sr: int,
    instrumental_duration: float,
    vocal_regions: List[Tuple[float, float]],
    target_sr: int = 22050,
) -> torch.Tensor:
    """
    Align TTS vocals to detected vocal regions in instrumental.
    
    Stretches/compresses vocals to fit into detected regions,
    with silence padding for intro/outro.
    
    Args:
        vocals: TTS vocal audio tensor
        vocals_sr: Sample rate of vocals
        instrumental_duration: Total duration of instrumental
        vocal_regions: List of (start, end) times for vocals
        target_sr: Target sample rate
        
    Returns:
        Aligned vocals tensor matching instrumental duration
    """
    # Resample vocals to target SR
    if vocals_sr != target_sr:
        vocals = torchaudio.functional.resample(vocals, vocals_sr, target_sr)
    
    vocals = vocals.squeeze()
    if vocals.dim() == 0:
        vocals = vocals.unsqueeze(0)
    
    total_samples = int(instrumental_duration * target_sr)
    aligned = torch.zeros(total_samples)
    
    if not vocal_regions:
        # No regions detected - start vocals from beginning with 2s offset
        offset_samples = int(2.0 * target_sr)
        vocals_len = min(len(vocals), total_samples - offset_samples)
        aligned[offset_samples:offset_samples + vocals_len] = vocals[:vocals_len]
        print(f"   ⚠️ No vocal regions - placing vocals at 2s offset")
        return aligned
    
    # Calculate total available time for vocals
    total_region_duration = sum(end - start for start, end in vocal_regions)
    vocals_duration = len(vocals) / target_sr
    
    print(f"   🎯 Vocal alignment:")
    print(f"      Detected {len(vocal_regions)} vocal regions")
    print(f"      Total region time: {total_region_duration:.1f}s")
    print(f"      TTS vocals duration: {vocals_duration:.1f}s")
    
    # Strategy: distribute vocals across regions proportionally
    # If vocals shorter → pad with silence between sections
    # If vocals longer → speed up slightly (max 1.3x) or truncate
    
    if vocals_duration > total_region_duration * 1.3:
        # Vocals way too long - truncate
        print(f"      ⚠️ Vocals too long, truncating to fit")
        max_samples = int(total_region_duration * 1.3 * target_sr)
        vocals = vocals[:max_samples]
        vocals_duration = len(vocals) / target_sr
    
    # Time stretch factor (gentle - max 1.3x speed)
    stretch_factor = total_region_duration / vocals_duration if vocals_duration > 0 else 1.0
    stretch_factor = max(0.77, min(1.3, stretch_factor))  # 0.77x-1.3x range
    
    if abs(stretch_factor - 1.0) > 0.05:
        print(f"      Time stretch: {stretch_factor:.2f}x")
        # Apply time stretch via resampling
        intermediate_sr = int(target_sr * stretch_factor)
        vocals = torchaudio.functional.resample(
            vocals.unsqueeze(0), target_sr, intermediate_sr
        ).squeeze()
        vocals = torchaudio.functional.resample(
            vocals.unsqueeze(0), intermediate_sr, target_sr  
        ).squeeze()
    
    # Place vocals in regions sequentially
    vocals_pos = 0
    for i, (start_time, end_time) in enumerate(vocal_regions):
        start_sample = int(start_time * target_sr)
        end_sample = int(end_time * target_sr)
        region_samples = end_sample - start_sample
        
        # How much vocals to place in this region
        remaining_vocals = len(vocals) - vocals_pos
        remaining_regions = len(vocal_regions) - i
        
        # Distribute remaining vocals across remaining regions
        vocals_for_region = min(
            region_samples,
            remaining_vocals // remaining_regions if remaining_regions > 0 else remaining_vocals
        )
        
        if vocals_for_region <= 0:
            break
        
        # Get segment with fades
        segment = vocals[vocals_pos:vocals_pos + vocals_for_region].clone()
        
        # Apply fades for smooth transitions
        fade_samples = min(int(0.03 * target_sr), len(segment) // 4)  # 30ms fade
        if fade_samples > 0 and len(segment) > fade_samples * 2:
            fade_in = torch.linspace(0, 1, fade_samples)
            fade_out = torch.linspace(1, 0, fade_samples)
            segment[:fade_samples] *= fade_in
            segment[-fade_samples:] *= fade_out
        
        # Place in aligned output
        place_end = min(start_sample + len(segment), total_samples)
        actual_len = place_end - start_sample
        aligned[start_sample:place_end] = segment[:actual_len]
        
        vocals_pos += vocals_for_region
    
    placed_duration = vocals_pos / target_sr
    print(f"      ✅ Placed {placed_duration:.1f}s of vocals across {len(vocal_regions)} regions")
    
    return aligned


def mix_vocals_with_instrumental(
    instrumental: torch.Tensor,
    vocals: torch.Tensor,
    instrumental_sr: int,
    vocals_sr: int,
    mix_level: float = 0.7,
    target_sr: int = 22050,
) -> torch.Tensor:
    """
    Mix synthesized vocals with generated instrumental track.
    
    Args:
        instrumental: Instrumental audio tensor
        vocals: Vocals audio tensor
        instrumental_sr: Sample rate of instrumental
        vocals_sr: Sample rate of vocals
        mix_level: Vocals volume (0.0-1.0), default 0.7
        target_sr: Output sample rate
        
    Returns:
        Mixed audio tensor at target_sr
    """
    print(f"\n🎚️ Mixing vocals with instrumental...")
    print(f"   Vocals level: {mix_level:.0%}")
    
    # Ensure same sample rate
    if instrumental_sr != target_sr:
        instrumental = torchaudio.functional.resample(
            instrumental, instrumental_sr, target_sr
        )
    
    if vocals_sr != target_sr:
        vocals = torchaudio.functional.resample(
            vocals, vocals_sr, target_sr
        )
    
    # Ensure proper dimensions
    if instrumental.dim() == 1:
        instrumental = instrumental.unsqueeze(0)
    if vocals.dim() == 1:
        vocals = vocals.unsqueeze(0)
    
    # Match lengths - pad shorter with silence
    inst_len = instrumental.shape[-1]
    voc_len = vocals.shape[-1]
    
    if voc_len < inst_len:
        # Pad vocals to match instrumental
        padding = torch.zeros(vocals.shape[0], inst_len - voc_len, device=vocals.device)
        vocals = torch.cat([vocals, padding], dim=-1)
        print(f"   Padded vocals: {voc_len/target_sr:.1f}s → {inst_len/target_sr:.1f}s")
    elif voc_len > inst_len:
        # Truncate vocals
        vocals = vocals[..., :inst_len]
        print(f"   Truncated vocals: {voc_len/target_sr:.1f}s → {inst_len/target_sr:.1f}s")
    
    # Mix: instrumental at full level, vocals at mix_level
    # This gives vocals presence while preserving the instrumental
    mixed = instrumental * (1 - mix_level * 0.3) + vocals * mix_level
    
    # Normalize to prevent clipping
    max_val = mixed.abs().max()
    if max_val > 0.95:
        mixed = mixed / max_val * 0.95
    
    print(f"   ✅ Mixed audio: {mixed.shape[-1] / target_sr:.1f}s")
    
    return mixed.squeeze()


def main():
    parser = argparse.ArgumentParser(
        description='🎵 Muzible Muze AI v2 - Section-Aware Music Generation'
    )
    
    # Main params
    parser.add_argument('--prompt', type=str, required=True,
                        help='Text prompt describing the song')
    parser.add_argument('--duration', type=float, default=120.0,
                        help='Target duration in seconds (default: 120)')
    parser.add_argument('--output', type=str, default='./output/generated_v2.wav',
                        help='Output file path')
    
    # Generation options
    parser.add_argument('--template', type=str, default=None,
                        choices=['verse_chorus', 'edm', 'ballad', 'progressive'],
                        help='Use a structure template')
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed for reproducibility')
    
    # ============================================
    # 🎤 Voice Conditioning Options
    # ============================================
    parser.add_argument('--style_of', type=str, default=None,
                        help='Generate "in style of" artist (style transfer, general sound) [256-dim]')
    parser.add_argument('--voice_clone', type=str, default=None,
                        help='Generate "like" artist (voice cloning, exact voice) [256-dim from mix]')
    parser.add_argument('--voice_as', type=str, default=None,
                        help='Use pre-computed separated voice embedding for artist [192-dim ECAPA-TDNN, best quality]')
    parser.add_argument('--vocals_dir', type=str, default='./data_v2/vocals',
                        help='Directory with vocals/{artist}/embeddings.json (default: ./data_v2/vocals)')
    parser.add_argument('--voice_clone_samples', type=str, default=None,
                        help='Path to WAV file or folder with voice samples for custom voice cloning')
    parser.add_argument('--voice_ref', type=str, default=None,
                        help='[LEGACY] Reference audio for voice style (use --voice_clone_samples instead)')
    parser.add_argument('--artist_embeddings_path', type=str,
                        default='./data_v2/artist_embeddings.json',
                        help='Path to artist_embeddings.json')
    
    # ============================================
    # 📝 Lyrics Options
    # ============================================
    parser.add_argument('--lyrics', type=str, default=None,
                        help='Lyrics text for singing (will be converted to phonemes)')
    parser.add_argument('--lyrics_file', type=str, default=None,
                        help='Path to text file with lyrics')
    parser.add_argument('--language', type=str, default=None,
                        help='Language code for lyrics (en, pl, de, fr, es, etc). Auto-detected if not specified')
    
    # ============================================
    # 🎤 Singing Voice Synthesis Options (GPT-SoVITS / Fish Speech)
    # ============================================
    parser.add_argument('--sing_lyrics', action='store_true',
                        help='Actually SING the lyrics (synthesize vocals)')
    parser.add_argument('--singing_backend', type=str, default='gpt_sovits',
                        choices=['gpt_sovits', 'fish_speech', 'coqui', 'elevenlabs'],
                        help='Backend for singing synthesis (default: gpt_sovits). '
                             'fish_speech is #1 on TTS-Arena2')
    parser.add_argument('--gpt_sovits_url', type=str, default='http://localhost:9880',
                        help='GPT-SoVITS API server URL')
    parser.add_argument('--fish_speech_url', type=str, default='http://localhost:8080',
                        help='Fish Speech API server URL')
    parser.add_argument('--fish_speech_api_key', type=str, default=None,
                        help='Fish Audio cloud API key (get from fish.audio)')
    parser.add_argument('--gpt_sovits_model', type=str, default=None,
                        help='Path to fine-tuned GPT-SoVITS model (optional)')
    parser.add_argument('--singing_voice_ref', type=str, default=None,
                        help='Reference audio for singing voice (5-30s sample)')
    parser.add_argument('--mix_vocals', type=float, default=0.7,
                        help='Mix level for vocals vs instrumental (0.0-1.0, default: 0.7)')
    parser.add_argument('--strip_ldm_vocals', action='store_true', default=True,
                        help='Use Demucs to strip any accidental vocals from LDM output '
                             'before mixing with TTS vocals (prevents double-voice). Default: True')
    parser.add_argument('--no_strip_ldm_vocals', action='store_false', dest='strip_ldm_vocals',
                        help='Disable stripping LDM vocals (faster but may cause double-voice)')
    
    # Model paths
    parser.add_argument('--vae_checkpoint', type=str, 
                        default='./checkpoints/vae_best.pt')
    parser.add_argument('--planner_checkpoint', type=str,
                        default='./checkpoints_v2/composition_planner_best.pt')
    parser.add_argument('--ldm_checkpoint', type=str,
                        default='./checkpoints_v2/ldm_v2_best.pt')
    parser.add_argument('--vocoder_checkpoint', type=str,
                        default='./checkpoints/vocoder_best.pt')
    
    # Model configuration
    parser.add_argument('--model_channels', type=int, default=320,
                        help='UNet model_channels (256=test, 320=default, 512=large)')
    
    # Device
    parser.add_argument('--device', type=str, 
                        default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--use_clap', action='store_true',
                        help='Use CLAP encoder instead of T5')
    parser.add_argument('--fp16', action='store_true',
                        help='Use FP16 for inference (reduces VRAM by half)')
    
    args = parser.parse_args()
    
    # Validate voice conditioning args (mutually exclusive)
    voice_args = [args.style_of, args.voice_clone, args.voice_clone_samples, args.voice_ref]
    voice_args_set = sum(1 for a in voice_args if a is not None)
    if voice_args_set > 1:
        parser.error("Only one of --style_of, --voice_clone, --voice_clone_samples, --voice_ref can be used")
    
    # Validate singing synthesis args
    if args.sing_lyrics and not args.lyrics and not args.lyrics_file:
        parser.error("--sing_lyrics requires --lyrics or --lyrics_file")
    
    # Set seed
    if args.seed is not None:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
    
    device = torch.device(args.device)
    print(f"\n🎵 Muzible Muze AI v2 - Section-Aware Generation")
    print(f"   Device: {device}")
    if args.fp16 and device.type == 'cuda':
        print(f"   Precision: FP16 (half)")
    else:
        print(f"   Precision: FP32")
    print(f"   Prompt: {args.prompt}")
    print(f"   Duration: {args.duration}s")
    
    # Load artist embeddings if needed
    artist_embeddings = None
    if args.style_of or args.voice_clone:
        try:
            artist_embeddings = load_artist_embeddings(args.artist_embeddings_path)
            print(f"   Loaded {len(artist_embeddings)} artist embeddings")
        except FileNotFoundError as e:
            print(f"   ⚠️ {e}")
            if args.voice_clone or args.style_of:
                print("   Cannot use --style_of or --voice_clone without artist embeddings")
                return
    
    # Load voice conditioning
    voice_embedding = load_voice_conditioning(args, artist_embeddings, args.device)
    
    # ============================================
    # 📝 Process Lyrics → Phonemes
    # ============================================
    phonemes_ipa = None
    lyrics_text = None
    
    # Load lyrics from file or argument
    if args.lyrics_file:
        lyrics_path = Path(args.lyrics_file)
        if lyrics_path.exists():
            lyrics_text = lyrics_path.read_text(encoding='utf-8').strip()
            print(f"\n📝 Loaded lyrics from: {args.lyrics_file}")
        else:
            print(f"   ⚠️ Lyrics file not found: {args.lyrics_file}")
    elif args.lyrics:
        lyrics_text = args.lyrics.strip()
    
    # Convert lyrics to phonemes
    if lyrics_text:
        print(f"\n📝 Processing lyrics ({len(lyrics_text)} chars)...")
        print(f"   Text: {lyrics_text[:100]}{'...' if len(lyrics_text) > 100 else ''}")
        
        processor = get_phoneme_processor()
        result = processor.text_to_phonemes(lyrics_text, language=args.language)
        
        if result.get('phonemes_ipa'):
            phonemes_ipa = result['phonemes_ipa']
            backend = result.get('backend', 'unknown')
            lang = result.get('language', 'unknown')
            print(f"   ✅ Converted to {len(phonemes_ipa)} phonemes (backend: {backend}, lang: {lang})")
            print(f"   IPA: {phonemes_ipa[:80]}{'...' if len(phonemes_ipa) > 80 else ''}")
        else:
            error = result.get('error', 'Unknown error')
            print(f"   ⚠️ Failed to convert lyrics: {error}")
            print("   Continuing without lyrics conditioning...")
    
    # Load models
    models = load_models(args, device)
    
    # Generate composition plan
    plan = generate_composition_plan(
        prompt=args.prompt,
        duration=args.duration,
        template=args.template,
        planner=models['planner'],
        device=device,
    )
    
    # Generate full song
    audio = generate_full_song(
        prompt=args.prompt,
        plan=plan,
        models=models,
        device=device,
        voice_embedding=voice_embedding,  # Pass embedding
        phonemes_ipa=phonemes_ipa,  # 📝 Pass phonemes
    )
    
    # Output path (needed for vocals saving)
    output_path = Path(args.output)
    output_path.parent.mkdir(exist_ok=True)
    
    # ============================================
    # 🎤 Singing Voice Synthesis (GPT-SoVITS / Fish Speech)
    # ============================================
    if args.sing_lyrics and lyrics_text:
        if not args.singing_voice_ref:
            print("\n⚠️ --sing_lyrics requires --singing_voice_ref (path to voice sample)")
            print("   Skipping vocal synthesis...")
        else:
            voice_ref_path = Path(args.singing_voice_ref)
            if not voice_ref_path.exists():
                print(f"\n⚠️ Voice reference not found: {voice_ref_path}")
                print("   Skipping vocal synthesis...")
            else:
                try:
                    # ============================================
                    # STEP 1: Strip accidental vocals from LDM output
                    # (prevents double-voice when mixing with TTS)
                    # ============================================
                    instrumental = audio
                    if args.strip_ldm_vocals:
                        instrumental, had_ldm_vocals = extract_clean_instrumental(
                            audio=audio,
                            sample_rate=22050,
                            device=args.device,
                        )
                        if had_ldm_vocals:
                            # Save LDM vocals for debugging
                            ldm_vocals_path = output_path.parent / f"{output_path.stem}_ldm_vocals_removed.wav"
                            # Note: we don't save them by default, just log
                            print(f"   (LDM vocals were stripped to prevent double-voice)")
                    
                    # ============================================
                    # STEP 2: Detect vocal regions using CompositionPlan
                    # (knows exactly where verse/chorus are!)
                    # ============================================
                    instrumental_duration = instrumental.shape[-1] / 22050
                    vocal_regions = detect_vocal_regions(
                        audio=instrumental,
                        sample_rate=22050,
                        composition_plan=plan,  # Use plan for precise section timing!
                    )
                    
                    # ============================================
                    # STEP 3: Synthesize TTS vocals from lyrics
                    # ============================================
                    vocals_audio, vocals_sr = synthesize_vocals(
                        lyrics_text=lyrics_text,
                        reference_audio=str(voice_ref_path),
                        language=args.language,
                        backend=args.singing_backend,
                        gpt_sovits_url=args.gpt_sovits_url,
                        fish_speech_url=args.fish_speech_url,
                        fish_speech_api_key=args.fish_speech_api_key,
                        device=args.device,
                    )
                    
                    # ============================================
                    # STEP 4: Align vocals to detected regions
                    # (time-stretch and place vocals where they fit)
                    # ============================================
                    aligned_vocals = align_vocals_to_instrumental(
                        vocals=vocals_audio,
                        vocals_sr=vocals_sr,
                        instrumental_duration=instrumental_duration,
                        vocal_regions=vocal_regions,
                        target_sr=22050,
                    )
                    
                    # ============================================
                    # STEP 5: Mix aligned vocals with instrumental
                    # ============================================
                    audio = mix_vocals_with_instrumental(
                        instrumental=instrumental,  # Clean instrumental (LDM vocals stripped)
                        vocals=aligned_vocals,      # Aligned TTS vocals
                        instrumental_sr=22050,
                        vocals_sr=22050,  # Already resampled in align function
                        mix_level=args.mix_vocals,
                        target_sr=22050,
                    )
                    
                    # Also save vocals-only version
                    vocals_output = output_path.parent / f"{output_path.stem}_vocals.wav"
                    vocals_cpu = aligned_vocals.squeeze().cpu()
                    if vocals_cpu.dim() == 1:
                        vocals_cpu = vocals_cpu.unsqueeze(0)
                    torchaudio.save(str(vocals_output), vocals_cpu, 22050)
                    print(f"   💾 Vocals saved to: {vocals_output}")
                    
                except Exception as e:
                    print(f"\n❌ Vocal synthesis failed: {e}")
                    print("   Saving instrumental only...")
                    import traceback
                    traceback.print_exc()
    
    # Save final output
    audio_cpu = audio.squeeze().cpu()
    torchaudio.save(str(output_path), audio_cpu.unsqueeze(0), 22050)
    
    print(f"\n✅ Saved to {output_path}")
    print(f"   Duration: {audio_cpu.shape[-1] / 22050:.1f}s")
    
    # Save composition plan with voice info
    plan_path = output_path.with_suffix('.json')
    plan_data = {
        'prompt': args.prompt,
        'duration': args.duration,
        'template': args.template,
        'seed': args.seed,
        'voice_conditioning': {
            'style_of': args.style_of,
            'voice_clone': args.voice_clone,
            'voice_clone_samples': args.voice_clone_samples,
            'has_embedding': voice_embedding is not None,
        },
        'lyrics': {
            'text': lyrics_text,
            'language': args.language,
            'phonemes_ipa': phonemes_ipa,
            'has_lyrics': phonemes_ipa is not None,
        },
        'singing': {
            'enabled': args.sing_lyrics,
            'backend': args.singing_backend if args.sing_lyrics else None,
            'voice_ref': args.singing_voice_ref,
            'mix_level': args.mix_vocals if args.sing_lyrics else None,
        },
        'sections': [
            {
                'type': s.section_type,
                'duration': s.duration,
                'tempo': s.tempo,
                'energy': s.energy,
                'key': s.key,
                'has_vocals': s.has_vocals,
            }
            for s in plan.sections
        ]
    }
    with open(plan_path, 'w') as f:
        json.dump(plan_data, f, indent=2, ensure_ascii=False)
    print(f"   Plan saved to {plan_path}")


if __name__ == "__main__":
    main()
