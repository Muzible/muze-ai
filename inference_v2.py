"""
🎵 Inference v2 - Generacja utworów z section-aware pipeline

Generuje pełne 2-4 minutowe utwory z koherentną strukturą sekcji.

Użycie:
    # Podstawowe użycie
    python inference_v2.py \
        --prompt "Energiczny pop z chwytliwym refrenem, żeński wokal" \
        --duration 180 \
        --output ./output/my_song.wav
    
    # Z szablonem struktury
    python inference_v2.py \
        --prompt "Spokojny jazz z fortepianem" \
        --template verse_chorus \
        --duration 120
    
    # 🎤 STYLE TRANSFER: "W stylu artysty X" (ogólne brzmienie)
    python inference_v2.py \
        --prompt "Electronic dance track" \
        --style_of "Metallica" \
        --duration 180
    
    # 🎤 VOICE CLONING: "Jak artysta X" (dokładny głos)
    python inference_v2.py \
        --prompt "Rock ballad with vocals" \
        --voice_clone "Metallica" \
        --duration 180
    
    # 🎤 VOICE CLONING z własnych sampli (folder)
    python inference_v2.py \
        --prompt "Pop song" \
        --voice_clone_samples ./my_voice_samples/ \
        --duration 120
    
    # 🎤 VOICE CLONING z pojedynczego pliku (szybki test)
    python inference_v2.py \
        --prompt "Test song" \
        --voice_clone_samples ./sample.wav \
        --duration 60

    # 📝 LYRICS: Generacja ze śpiewem do podanego tekstu
    python inference_v2.py \
        --prompt "Emotional ballad with piano" \
        --lyrics "I walk alone through empty streets, searching for your light" \
        --voice_clone "Adele" \
        --duration 120

    # 📝 LYRICS po polsku (automatyczna detekcja języka lub --language)
    python inference_v2.py \
        --prompt "Polska ballada rockowa" \
        --lyrics "Idę sam przez puste ulice, szukam twego światła" \
        --language pl \
        --voice_clone "Doda" \
        --duration 120
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Optional, List, Dict, Any, Union

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
    Konwersja tekstu na fonemy (IPA) dla inference.
    
    Wspiera:
    - Gruut: en, de, es, fr, it, ru, cs, nl, sv (lepszy dla zachodnich)
    - eSpeak: pl, uk, pt, ja, ko, zh, tr, vi, hi (dla reszty świata)
    
    Auto-fallback: jeśli Gruut nie obsługuje języka → eSpeak
    """
    
    # Języki obsługiwane przez Gruut (preferowane)
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
    
    # Języki dla eSpeak (w tym polski!)
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
                'phonemes_ipa': str,  # Pełny ciąg fonemów
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
        
        # Wybierz backend na podstawie języka
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
        """Użyj Gruut dla obsługiwanych języków (en, de, es, fr, it, ru, cs, nl, sv)"""
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
        """Użyj Phonemizer/eSpeak (szczególnie dla polskiego!)"""
        from phonemizer import phonemize
        
        # Mapuj język na kod espeak
        espeak_lang = self.ESPEAK_LANGUAGES.get(language, language)
        
        # Sprawdź czy język jest obsługiwany
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
    """Ładuje plik artist_embeddings.json"""
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
        # 192-dim ECAPA-TDNN z Demucs separated vocals - najlepsza jakość
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
            use_demucs=True,  # Separuj wokale dla lepszej jakości
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
                # Preferuj separated embedding
                emb = result.get('embedding_separated') or result.get('embedding_mix')
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
    
    # Uśrednij
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
    
    # 1. Własne sample użytkownika
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
    """Ładowanie wszystkich modeli z automatyczną detekcją konfiguracji"""
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
    
    # VAE - ładuj checkpoint żeby określić latent_dim
    print("📦 Loading VAE...")
    vae_ckpt = torch.load(args.vae_checkpoint, map_location=device)
    
    # Wykryj latent_dim z checkpointu
    latent_dim = 128  # default
    if 'config' in vae_ckpt:
        latent_dim = vae_ckpt['config'].get('latent_dim', 128)
    elif 'latent_dim' in vae_ckpt:
        latent_dim = vae_ckpt['latent_dim']
    else:
        # Spróbuj wykryć z wag
        state_dict = vae_ckpt.get('model_state_dict', vae_ckpt)
        for key in state_dict:
            if 'conv_out.weight' in key:
                # conv_out output = latent_dim * 2
                latent_dim = state_dict[key].shape[0] // 2
                break
    
    print(f"   Detected latent_dim: {latent_dim}")
    
    vae = AudioVAE(latent_dim=latent_dim).to(device)
    vae.load_state_dict(vae_ckpt.get('model_state_dict', vae_ckpt))
    vae.eval()
    if use_fp16:
        vae = vae.half()
    models['vae'] = vae
    models['latent_dim'] = latent_dim
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
    ldm_ckpt = torch.load(args.ldm_checkpoint, map_location=device)
    
    # Wykryj konfigurację z checkpointu
    ldm_config = ldm_ckpt.get('config', {})
    
    # model_channels - priorytet: checkpoint > argument > default
    if 'model_channels' in ldm_config:
        model_channels = ldm_config['model_channels']
        print(f"   Detected model_channels from checkpoint: {model_channels}")
    elif hasattr(args, 'model_channels') and args.model_channels:
        model_channels = args.model_channels
        print(f"   Using model_channels from argument: {model_channels}")
    else:
        # Spróbuj wykryć z wag (conv_in.weight shape)
        state_dict = ldm_ckpt.get('model_state_dict', ldm_ckpt)
        model_channels = 320  # default
        for key in state_dict:
            if 'unet.conv_in.weight' in key or 'conv_in.weight' in key:
                # conv_in.weight shape: [model_channels, in_channels, 3, 3]
                model_channels = state_dict[key].shape[0]
                print(f"   Detected model_channels from weights: {model_channels}")
                break
    
    # channel_mult - z checkpointu lub default
    channel_mult = ldm_config.get('channel_mult', [1, 2, 4, 4])
    
    # use_voice_stream, use_dual_voice - z checkpointu lub default
    use_voice_stream = ldm_config.get('use_voice_stream', True)
    use_dual_voice = ldm_config.get('use_dual_voice', True)
    
    print(f"   Config: latent_dim={latent_dim}, model_channels={model_channels}")
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
    
    ldm = LatentDiffusionV2(unet, num_timesteps=1000).to(device)
    ldm.load_state_dict(ldm_ckpt.get('model_state_dict', ldm_ckpt))
    ldm.eval()
    if use_fp16:
        ldm = ldm.half()
    models['ldm'] = ldm
    
    # Vocoder (HiFi-GAN) - zawsze FP32 dla jakości audio
    print("📦 Loading HiFi-GAN Vocoder...")
    vocoder = HiFiGAN().to(device)
    if args.vocoder_checkpoint and Path(args.vocoder_checkpoint).exists():
        vocoder_ckpt = torch.load(args.vocoder_checkpoint, map_location=device)
        # HiFiGAN checkpoint może mieć 'generator' jako klucz
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
        # Użyj szablonu
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
    
    # Konwersja do właściwego dtype dla FP16
    if dtype == torch.float16:
        text_embed = text_embed.half()
    
    # Sample latent
    segment_duration = min(section.duration, 10.0)  # Max 10s per segment
    latent_length = int(segment_duration * sample_rate / 256)  # VAE downsampling
    
    # Przygotuj voice embeddings (rozróżnienie resemblyzer vs ecapa)
    voice_emb = None
    voice_emb_separated = None
    if voice_embedding is not None:
        ve = voice_embedding.to(dtype)  # Konwersja do właściwego dtype
        if voice_embedding.shape[-1] == 256:
            # Resemblyzer embedding (mix-based)
            voice_emb = ve.unsqueeze(0) if ve.dim() == 1 else ve
        elif voice_embedding.shape[-1] == 192:
            # ECAPA-TDNN embedding (separated vocals)
            voice_emb_separated = ve.unsqueeze(0) if ve.dim() == 1 else ve
        else:
            # Unknown dimension, try as resemblyzer
            voice_emb = ve.unsqueeze(0) if ve.dim() == 1 else ve
    
    # Sample latent z pełnym kondycjonowaniem v3
    latent = ldm.sample_section(
        shape=(1, latent_dim, latent_length),  # Używaj latent_dim z VAE
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
    
    # Decode latent to mel (konwersja do FP32 dla vocodera)
    with torch.no_grad():
        latent_fp32 = latent.float() if dtype == torch.float16 else latent
        mel = vae.float().decode(latent_fp32) if dtype == torch.float16 else vae.decode(latent)
    
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
        voice_embedding=voice_embedding,  # Przekaż embedding
        phonemes_ipa=phonemes_ipa,  # 📝 Przekaż fonemy
    )
    
    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(exist_ok=True)
    
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
