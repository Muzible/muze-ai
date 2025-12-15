"""
Voice Synthesis & Cloning Module

Dwa tryby użycia głosu z utworu:

1. "W STYLU artysty X" (--artist_style)
   - Ekstrahuje embedding głosu (SpeechBrain ECAPA-TDNN)
   - Wpływa na GENEROWANĄ MUZYKĘ (vibe, energia, styl)
   - NIE klonuje głosu, tylko "inspiruje" model
   - Legalne (styl nie jest chroniony)

2. "ŚPIEWAJ JAK artysta X" (--clone_voice_from)
   - Ekstrahuje głos z utworu (separacja Demucs)
   - Klonuje FAKTYCZNY GŁOS (ElevenLabs/RVC/XTTS)
   - Generuje śpiew głosem artysty
   - ⚠️ Prawnie problematyczne (IP głosu)

Pipeline dla trybu 2:
1. Demucs separuje wokal z utworu
2. Voice cloning model (ElevenLabs/XTTS) uczy się głosu
3. Generujemy śpiew nowym tekstem używając sklonowanego głosu

Obsługiwane backendy:
- Coqui TTS (XTTS v2) - open source, lokalne
- ElevenLabs - API, very high quality
- Bark - open source, Facebook
- RVC (Retrieval Voice Conversion) - voice conversion

Usage example:
    from models.voice_synthesis import VoiceSynthesizer, VoiceExtractorFromSong
    
    # Mode 1: Your own voice
    synth = VoiceSynthesizer(backend="coqui")
    synth.register_voice("my_voice", "recording.wav")
    
    # Mode 2: Extract voice from song and clone
    extractor = VoiceExtractorFromSong()
    vocals_path = extractor.extract_vocals("artist_song.mp3")
    synth.register_voice("artist_x", vocals_path)
    
    # Generate singing
    audio = synth.synthesize(text="New text", voice="artist_x")
"""

import torch
import torchaudio
import numpy as np
from pathlib import Path
from typing import Optional, Dict, List, Union, Any, Tuple
from dataclasses import dataclass
import json
import tempfile
import os


@dataclass
class RegisteredVoice:
    """Registered user voice."""
    name: str
    reference_audio_path: str
    embedding: Optional[List[float]] = None
    speaker_id: Optional[str] = None  # dla API jak ElevenLabs
    backend: str = "coqui"
    metadata: Optional[Dict] = None
    source_type: str = "recording"  # "recording", "extracted_from_song"


class VoiceExtractorFromSong:
    """
    Extracts vocals from a music track for voice cloning.
    
    Pipeline:
    1. Demucs/Spleeter separates vocals from instruments
    2. Optionally: diarization if multiple vocalists
    3. Returns clean vocals for voice cloning
    
    Usage:
        extractor = VoiceExtractorFromSong()
        vocals = extractor.extract_vocals("song.mp3")
        # vocals is a path to file with vocals only
    """
    
    def __init__(
        self,
        separation_model: str = "htdemucs",  # htdemucs, demucs, spleeter
        device: str = "cpu",
        output_dir: Optional[str] = None,
    ):
        """
        Args:
            separation_model: "htdemucs" (best), "demucs", "spleeter"
            device: cpu/cuda
            output_dir: where to save extracted vocals
        """
        self.separation_model = separation_model
        self.device = device
        self.output_dir = output_dir or tempfile.gettempdir()
        self._separator = None
    
    def _init_separator(self):
        """Lazy load separation model."""
        if self._separator is not None:
            return
        
        if self.separation_model in ["htdemucs", "demucs"]:
            try:
                import demucs.separate
                from demucs.pretrained import get_model
                from demucs.apply import apply_model
                
                print(f"🎵 Loading {self.separation_model} separation model...")
                self._separator = {
                    "type": "demucs",
                    "model": get_model(self.separation_model),
                    "apply": apply_model,
                }
                self._separator["model"].to(self.device)
                print("   ✓ Demucs loaded")
                
            except ImportError:
                raise ImportError(
                    "Demucs not installed. Install with:\n"
                    "  pip install demucs"
                )
        
        elif self.separation_model == "spleeter":
            try:
                from spleeter.separator import Separator
                
                print("🎵 Loading Spleeter separation model...")
                self._separator = {
                    "type": "spleeter",
                    "model": Separator('spleeter:2stems'),
                }
                print("   ✓ Spleeter loaded")
                
            except ImportError:
                raise ImportError(
                    "Spleeter not installed. Install with:\n"
                    "  pip install spleeter"
                )
        else:
            raise ValueError(f"Unknown separation model: {self.separation_model}")
    
    def extract_vocals(
        self,
        audio_path: str,
        output_path: Optional[str] = None,
        min_vocal_duration: float = 10.0,
    ) -> str:
        """
        Ekstrahuje wokal z utworu muzycznego.
        
        Args:
            audio_path: ścieżka do utworu (MP3/WAV/FLAC)
            output_path: gdzie zapisać wokal (opcjonalne)
            min_vocal_duration: min. długość wokalu w sekundach
            
        Returns:
            ścieżka do pliku z wyekstrahowanym wokalem
        """
        self._init_separator()
        
        audio_path = Path(audio_path)
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        
        print(f"🎤 Extracting vocals from: {audio_path.name}")
        
        # Generate output path
        if output_path is None:
            output_path = Path(self.output_dir) / f"{audio_path.stem}_vocals.wav"
        else:
            output_path = Path(output_path)
        
        # Run separation
        if self._separator["type"] == "demucs":
            vocals = self._extract_with_demucs(str(audio_path))
        else:
            vocals = self._extract_with_spleeter(str(audio_path))
        
        # Check if vocals actually contain voice (not silent/instrumental)
        rms_energy = torch.sqrt(torch.mean(vocals ** 2)).item()
        has_voice = rms_energy > 0.01  # Threshold for actual voice content
        
        if not has_voice:
            print(f"   ⚠️  No voice detected (RMS: {rms_energy:.4f}) - instrumental track?")
            return None  # Don't save empty vocals
        
        # Save vocals
        torchaudio.save(str(output_path), vocals, 44100)
        
        # Check duration
        duration = vocals.shape[-1] / 44100
        if duration < min_vocal_duration:
            print(f"   ⚠️  Warning: Extracted vocals only {duration:.1f}s (min: {min_vocal_duration}s)")
        
        print(f"   ✓ Vocals saved to: {output_path} (RMS: {rms_energy:.3f})")
        print(f"   Duration: {duration:.1f}s")
        
        return str(output_path)
    
    def _extract_with_demucs(self, audio_path: str) -> torch.Tensor:
        """Extract vocals using Demucs."""
        # Load audio
        audio, sr = torchaudio.load(audio_path)
        
        # Resample to 44100 if needed
        if sr != 44100:
            audio = torchaudio.functional.resample(audio, sr, 44100)
        
        # Ensure stereo
        if audio.shape[0] == 1:
            audio = audio.repeat(2, 1)
        
        # Add batch dimension
        audio = audio.unsqueeze(0).to(self.device)
        
        # Apply separation
        with torch.no_grad():
            sources = self._separator["apply"](
                self._separator["model"],
                audio,
                device=self.device,
            )
        
        # Sources order: drums, bass, other, vocals
        # Get vocals (last source)
        vocals = sources[0, -1]  # [2, samples]
        
        return vocals.cpu()
    
    def _extract_with_spleeter(self, audio_path: str) -> torch.Tensor:
        """Extract vocals using Spleeter."""
        import tempfile
        
        # Spleeter outputs to directory
        with tempfile.TemporaryDirectory() as tmpdir:
            self._separator["model"].separate_to_file(
                audio_path,
                tmpdir,
            )
            
            # Find vocals file
            stem = Path(audio_path).stem
            vocals_path = Path(tmpdir) / stem / "vocals.wav"
            
            if vocals_path.exists():
                vocals, sr = torchaudio.load(str(vocals_path))
                if sr != 44100:
                    vocals = torchaudio.functional.resample(vocals, sr, 44100)
                return vocals
            else:
                raise RuntimeError("Spleeter failed to extract vocals")
    
    def extract_and_analyze(
        self,
        audio_path: str,
    ) -> Dict[str, Any]:
        """
        Ekstrahuje wokal i analizuje charakterystyki.
        
        Returns dict z:
            - vocals_path: ścieżka do wokalu
            - duration: długość
            - has_multiple_singers: czy wielu wokalistów
            - dominant_pitch_range: zakres głosu
            - quality_score: jakość do voice cloningu
        """
        vocals_path = self.extract_vocals(audio_path)
        
        # Load vocals for analysis
        vocals, sr = torchaudio.load(vocals_path)
        duration = vocals.shape[-1] / sr
        
        # Basic analysis
        analysis = {
            "vocals_path": vocals_path,
            "duration": duration,
            "sample_rate": sr,
            "channels": vocals.shape[0],
        }
        
        # Analyze pitch range
        try:
            import librosa
            
            # Convert to mono numpy
            vocals_mono = vocals.mean(dim=0).numpy()
            
            # Extract pitch
            pitches, magnitudes = librosa.piptrack(
                y=vocals_mono,
                sr=sr,
                fmin=50,
                fmax=2000,
            )
            
            # Get valid pitches
            valid_pitches = []
            for t in range(pitches.shape[1]):
                idx = magnitudes[:, t].argmax()
                pitch = pitches[idx, t]
                if pitch > 0:
                    valid_pitches.append(pitch)
            
            if valid_pitches:
                analysis["pitch_min"] = min(valid_pitches)
                analysis["pitch_max"] = max(valid_pitches)
                analysis["pitch_mean"] = np.mean(valid_pitches)
                
                # Estimate voice type
                mean_pitch = analysis["pitch_mean"]
                if mean_pitch < 165:
                    analysis["voice_type"] = "bass/baritone"
                elif mean_pitch < 262:
                    analysis["voice_type"] = "tenor/alto"
                else:
                    analysis["voice_type"] = "soprano/high"
                    
        except ImportError:
            pass
        
        # Quality score for cloning
        quality_score = 1.0
        if duration < 10:
            quality_score *= 0.5
        if duration < 5:
            quality_score *= 0.5
        if duration > 60:
            quality_score = min(quality_score, 0.9)  # Too long might have variety
        
        analysis["quality_score"] = quality_score
        analysis["recommended_for_cloning"] = quality_score >= 0.7
        
        return analysis


class VoiceSynthesizer:
    """
    Synteza mowy/śpiewu z klonowaniem głosu.
    
    Wspiera wiele backendów:
    - coqui: XTTS v2 (lokalne, open source)
    - elevenlabs: ElevenLabs API (najlepsza jakość)
    - bark: Bark by Facebook (lokalne)
    - rvc: RVC voice conversion
    - gpt_sovits: GPT-SoVITS (SOTA zero/few-shot TTS, MIT license)
    - fish_speech: Fish Speech / OpenAudio S1 (#1 TTS-Arena2, Apache 2.0)
    
    GPT-SoVITS Features:
    - Zero-shot TTS: 5s sample → instant voice cloning
    - Few-shot TTS: 1min training → perfect voice match
    - Cross-lingual: EN, JA, KO, ZH, Cantonese
    - 48kHz output (v4), fast RTF (~0.028 on 4060Ti)
    - MIT License (compatible with GPL-3.0)
    
    Fish Speech / OpenAudio Features:
    - #1 ranking on TTS-Arena2 benchmark
    - Zero-shot TTS: 10-30s sample
    - Emotion control: (angry), (excited), (sad), etc.
    - Multilingual: EN, JA, KO, ZH, FR, DE, AR, ES
    - S1-mini: 0.5B params, open source on HuggingFace
    - Apache 2.0 License
    """
    
    SUPPORTED_BACKENDS = ["coqui", "elevenlabs", "bark", "rvc", "gpt_sovits", "fish_speech"]
    
    def __init__(
        self,
        backend: str = "coqui",
        api_key: Optional[str] = None,
        device: str = "cpu",
        model_path: Optional[str] = None,
        gpt_sovits_url: Optional[str] = None,
        gpt_model_path: Optional[str] = None,
        sovits_model_path: Optional[str] = None,
        fish_speech_url: Optional[str] = None,
    ):
        """
        Inicjalizuje syntezator głosu.
        
        Args:
            backend: "coqui", "elevenlabs", "bark", "rvc", "gpt_sovits", "fish_speech"
            api_key: klucz API (dla elevenlabs, fish_speech cloud)
            device: cpu/cuda/mps
            model_path: ścieżka do modelu (dla RVC)
            gpt_sovits_url: URL do GPT-SoVITS API (np. http://localhost:9880)
            gpt_model_path: ścieżka do GPT modelu (.ckpt) dla GPT-SoVITS
            sovits_model_path: ścieżka do SoVITS modelu (.pth) dla GPT-SoVITS
            fish_speech_url: URL do Fish Speech API (np. http://localhost:8080)
        """
        if backend not in self.SUPPORTED_BACKENDS:
            raise ValueError(f"Unknown backend: {backend}. Supported: {self.SUPPORTED_BACKENDS}")
        
        self.backend = backend
        self.api_key = api_key
        self.device = device
        self.model_path = model_path
        
        # GPT-SoVITS specific
        self.gpt_sovits_url = gpt_sovits_url or "http://localhost:9880"
        self.gpt_model_path = gpt_model_path
        self.sovits_model_path = sovits_model_path
        
        # Fish Speech specific
        self.fish_speech_url = fish_speech_url or "http://localhost:8080"
        
        # Registered voices
        self.voices: Dict[str, RegisteredVoice] = {}
        
        # Lazy load backend
        self._model = None
        self._initialized = False
    
    def _init_backend(self):
        """Lazy initialization of backend."""
        if self._initialized:
            return
        
        if self.backend == "coqui":
            self._init_coqui()
        elif self.backend == "elevenlabs":
            self._init_elevenlabs()
        elif self.backend == "bark":
            self._init_bark()
        elif self.backend == "rvc":
            self._init_rvc()
        elif self.backend == "gpt_sovits":
            self._init_gpt_sovits()
        elif self.backend == "fish_speech":
            self._init_fish_speech()
        
        self._initialized = True
    
    def _init_coqui(self):
        """Initialize Coqui XTTS v2."""
        try:
            from TTS.api import TTS
            print("🎤 Loading Coqui XTTS v2...")
            self._model = TTS("tts_models/multilingual/multi-dataset/xtts_v2")
            if self.device != "cpu":
                self._model = self._model.to(self.device)
            print("   ✓ XTTS v2 loaded")
        except ImportError:
            raise ImportError(
                "Coqui TTS not installed. Install with:\n"
                "  pip install TTS\n"
                "Or for GPU support:\n"
                "  pip install TTS[all]"
            )
    
    def _init_elevenlabs(self):
        """Initialize ElevenLabs client."""
        if not self.api_key:
            import os
            self.api_key = os.environ.get("ELEVENLABS_API_KEY")
            if not self.api_key:
                raise ValueError(
                    "ElevenLabs API key required. Set ELEVENLABS_API_KEY env var "
                    "or pass api_key parameter."
                )
        
        try:
            from elevenlabs import ElevenLabs
            print("🎤 Initializing ElevenLabs...")
            self._model = ElevenLabs(api_key=self.api_key)
            print("   ✓ ElevenLabs ready")
        except ImportError:
            raise ImportError(
                "ElevenLabs not installed. Install with:\n"
                "  pip install elevenlabs"
            )
    
    def _init_bark(self):
        """Initialize Bark."""
        try:
            from bark import SAMPLE_RATE, generate_audio, preload_models
            print("🎤 Loading Bark models...")
            preload_models()
            self._model = {
                "generate": generate_audio,
                "sample_rate": SAMPLE_RATE,
            }
            print("   ✓ Bark loaded")
        except ImportError:
            raise ImportError(
                "Bark not installed. Install with:\n"
                "  pip install git+https://github.com/suno-ai/bark.git"
            )
    
    def _init_rvc(self):
        """Initialize RVC (Retrieval Voice Conversion)."""
        if not self.model_path:
            raise ValueError("RVC requires model_path to .pth file")
        
        try:
            # RVC has various implementations, we use the popular one
            print(f"🎤 Loading RVC model from {self.model_path}...")
            # Placeholder - RVC wymaga specyficznej konfiguracji
            self._model = {"model_path": self.model_path}
            print("   ✓ RVC loaded")
        except Exception as e:
            raise RuntimeError(f"Failed to load RVC model: {e}")
    
    def _init_gpt_sovits(self):
        """
        Initialize GPT-SoVITS backend.
        
        GPT-SoVITS can run in two modes:
        1. API mode: Connect to running GPT-SoVITS server (api_v2.py)
        2. Local mode: Load models directly (requires GPT-SoVITS installation)
        
        For API mode, start the server first:
            cd GPT-SoVITS
            python api_v2.py -a 0.0.0.0 -p 9880
        """
        import requests
        
        print(f"🎤 Initializing GPT-SoVITS...")
        print(f"   API URL: {self.gpt_sovits_url}")
        
        # Check if API is available
        try:
            response = requests.get(f"{self.gpt_sovits_url}/", timeout=5)
            api_available = response.status_code == 200
        except Exception:
            api_available = False
        
        if api_available:
            # API mode - use running server
            self._model = {
                "mode": "api",
                "url": self.gpt_sovits_url,
                "sample_rate": 32000,  # GPT-SoVITS v2/v3 default
            }
            print("   ✓ GPT-SoVITS API connected")
            
            # Try to get model info
            try:
                info_resp = requests.get(f"{self.gpt_sovits_url}/info")
                if info_resp.status_code == 200:
                    info = info_resp.json()
                    print(f"   Model: {info.get('version', 'unknown')}")
            except Exception:
                pass
        else:
            # Local mode - try to load models directly
            print("   ⚠️ API not available, attempting local mode...")
            
            try:
                # GPT-SoVITS local inference
                # This requires GPT-SoVITS to be installed
                import sys
                gpt_sovits_path = os.environ.get("GPT_SOVITS_PATH", "./GPT-SoVITS")
                if os.path.exists(gpt_sovits_path):
                    sys.path.insert(0, gpt_sovits_path)
                
                # Import GPT-SoVITS inference modules
                from GPT_SoVITS.inference_webui import get_tts_wav
                
                self._model = {
                    "mode": "local",
                    "inference_fn": get_tts_wav,
                    "gpt_path": self.gpt_model_path,
                    "sovits_path": self.sovits_model_path,
                    "sample_rate": 32000,
                }
                print("   ✓ GPT-SoVITS local mode initialized")
                
            except ImportError as e:
                raise ImportError(
                    f"GPT-SoVITS not available. Options:\n"
                    f"1. Start GPT-SoVITS API server:\n"
                    f"   cd GPT-SoVITS && python api_v2.py -a 0.0.0.0 -p 9880\n\n"
                    f"2. Install GPT-SoVITS locally:\n"
                    f"   git clone https://github.com/RVC-Boss/GPT-SoVITS.git\n"
                    f"   cd GPT-SoVITS && pip install -r requirements.txt\n"
                    f"   Set GPT_SOVITS_PATH environment variable\n\n"
                    f"Error: {e}"
                )
    
    def _init_fish_speech(self):
        """
        Initialize Fish Speech / OpenAudio backend.
        
        Fish Speech is the #1 ranked TTS on TTS-Arena2 benchmark.
        
        Features:
        - Zero-shot voice cloning (10-30s sample)
        - Emotion markers: (angry), (excited), (sad), etc.
        - Multilingual: EN, JA, KO, ZH, FR, DE, AR, ES
        - S1-mini: 0.5B params, available on HuggingFace
        
        Two modes:
        1. API mode: Connect to Fish Audio cloud or local server
        2. Local mode: Run fish-speech locally
        
        For local server:
            pip install fish-speech
            python -m fish_speech.webui.api --listen 0.0.0.0:8080
        """
        import requests
        
        print(f"🐟 Initializing Fish Speech...")
        print(f"   API URL: {self.fish_speech_url}")
        
        # Check if using Fish Audio cloud API
        is_cloud = "fish.audio" in self.fish_speech_url or self.api_key
        
        if is_cloud and self.api_key:
            # Fish Audio cloud API
            self._model = {
                "mode": "cloud",
                "url": "https://api.fish.audio",
                "api_key": self.api_key,
                "sample_rate": 44100,
            }
            print("   ✓ Fish Audio cloud API configured")
            return
        
        # Check if local API is available
        try:
            response = requests.get(f"{self.fish_speech_url}/", timeout=5)
            api_available = response.status_code in [200, 404]  # 404 is ok, means server running
        except Exception:
            api_available = False
        
        if api_available:
            # Local API mode
            self._model = {
                "mode": "api",
                "url": self.fish_speech_url,
                "sample_rate": 44100,
            }
            print("   ✓ Fish Speech local API connected")
        else:
            # Try to load locally
            print("   ⚠️ API not available, attempting local mode...")
            
            try:
                # Check if fish_speech package is installed
                import fish_speech
                from fish_speech.inference import inference
                
                self._model = {
                    "mode": "local",
                    "inference": inference,
                    "sample_rate": 44100,
                }
                print("   ✓ Fish Speech local mode initialized")
                
            except ImportError as e:
                raise ImportError(
                    f"Fish Speech not available. Options:\n\n"
                    f"1. Use Fish Audio cloud API (best quality):\n"
                    f"   Get API key from https://fish.audio\n"
                    f"   VoiceSynthesizer(backend='fish_speech', api_key='your_key')\n\n"
                    f"2. Run Fish Speech locally:\n"
                    f"   pip install fish-speech\n"
                    f"   python -m fish_speech.webui.api --listen 0.0.0.0:8080\n\n"
                    f"3. Use HuggingFace Spaces (free demo):\n"
                    f"   https://huggingface.co/spaces/fishaudio/fish-speech-1\n\n"
                    f"Error: {e}"
                )
    
    def register_voice(
        self,
        name: str,
        reference_audio: str,
        description: Optional[str] = None,
        source_type: str = "recording",
    ) -> RegisteredVoice:
        """
        Registers user voice for later use.
        
        Args:
            name: unique name for this voice (e.g. "my_voice")
            reference_audio: path to recording (10-30s, clean audio)
            description: optional description
            source_type: "recording" (own recording) or "extracted_from_song"
            
        Returns:
            RegisteredVoice object
            
        Example:
            synth.register_voice("adam", "adam_recording.wav")
            synth.register_voice("kate", "kate_recording.mp3", "Kate - alto")
        """
        self._init_backend()
        
        audio_path = Path(reference_audio)
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {reference_audio}")
        
        print(f"🎙️  Registering voice '{name}' from {audio_path.name}...")
        
        # Backend-specific registration
        voice = RegisteredVoice(
            name=name,
            reference_audio_path=str(audio_path.absolute()),
            backend=self.backend,
            metadata={"description": description} if description else None,
            source_type=source_type,
        )
        
        if self.backend == "coqui":
            # XTTS uses reference audio directly
            # We can also extract embedding for cache
            voice.embedding = self._extract_voice_embedding(reference_audio)
        
        elif self.backend == "elevenlabs":
            # Upload voice do ElevenLabs (voice cloning)
            voice.speaker_id = self._upload_to_elevenlabs(name, reference_audio, description)
        
        elif self.backend == "bark":
            # Bark uses speaker embeddings
            voice.embedding = self._extract_bark_embedding(reference_audio)
        
        elif self.backend == "rvc":
            # RVC needs feature extraction
            voice.embedding = self._extract_rvc_features(reference_audio)
        
        elif self.backend == "gpt_sovits":
            # GPT-SoVITS uses reference audio directly (zero-shot)
            # No embedding extraction needed - just store the path
            pass
        
        elif self.backend == "fish_speech":
            # Fish Speech uses reference audio directly (zero-shot)
            # No embedding extraction needed - just store the path
            pass
        
        self.voices[name] = voice
        print(f"   ✓ Voice '{name}' registered (source: {source_type})")
        
        return voice
    
    def register_voice_from_song(
        self,
        name: str,
        song_path: str,
        separation_model: str = "htdemucs",
        description: Optional[str] = None,
    ) -> RegisteredVoice:
        """
        🎵 Ekstrahuje głos z utworu muzycznego i rejestruje do klonowania.
        
        To jest tryb "ŚPIEWAJ JAK artysta X" - klonuje faktyczny głos.
        
        ⚠️ UWAGA PRAWNA: Klonowanie głosu artysty bez zgody może
        naruszać prawa osobiste i IP. Używaj odpowiedzialnie!
        
        Pipeline:
        1. Demucs separuje wokal od instrumentów
        2. Czysty wokal jest używany do voice cloningu
        3. Możesz generować nowy śpiew tym głosem
        
        Args:
            name: nazwa dla tego głosu (np. "freddie_mercury")
            song_path: ścieżka do utworu z wokalem
            separation_model: "htdemucs" (najlepszy), "demucs", "spleeter"
            description: optional description
            
        Returns:
            RegisteredVoice object
            
        Example:
            # Extract voice from Queen song
            synth.register_voice_from_song(
                "freddie",
                "queen_bohemian_rhapsody.mp3",
                description="Freddie Mercury vocal"
            )
            
            # Now you can sing "like Freddie"
            audio = synth.synthesize(
                text="New text to sing",
                voice="freddie"
            )
        """
        print("="*60)
        print(f"🎵 Extracting voice from song: {Path(song_path).name}")
        print("="*60)
        print("⚠️  Legal notice: Voice cloning may have IP implications")
        print("-"*60)
        
        # Step 1: Extract vocals
        extractor = VoiceExtractorFromSong(
            separation_model=separation_model,
            device=self.device,
        )
        
        analysis = extractor.extract_and_analyze(song_path)
        vocals_path = analysis["vocals_path"]
        
        print(f"\n📊 Vocal analysis:")
        print(f"   Duration: {analysis['duration']:.1f}s")
        if "voice_type" in analysis:
            print(f"   Voice type: {analysis['voice_type']}")
        if "pitch_mean" in analysis:
            print(f"   Pitch range: {analysis['pitch_min']:.0f}-{analysis['pitch_max']:.0f} Hz")
        print(f"   Quality score: {analysis['quality_score']:.2f}")
        print(f"   Recommended: {'✅ Yes' if analysis['recommended_for_cloning'] else '⚠️ Maybe not ideal'}")
        
        # Step 2: Register extracted voice
        if description is None:
            description = f"Extracted from: {Path(song_path).name}"
        
        voice = self.register_voice(
            name=name,
            reference_audio=vocals_path,
            description=description,
            source_type="extracted_from_song",
        )
        
        # Add analysis to metadata
        voice.metadata = voice.metadata or {}
        voice.metadata["source_song"] = str(Path(song_path).absolute())
        voice.metadata["vocal_analysis"] = analysis
        
        print("="*60)
        print(f"✅ Voice '{name}' ready for cloning!")
        print(f"   Use: synth.synthesize(text='...', voice='{name}')")
        print("="*60)
        
        return voice
    
    def _extract_voice_embedding(self, audio_path: str) -> List[float]:
        """Extract voice embedding using ECAPA-TDNN."""
        try:
            from speechbrain.inference import EncoderClassifier
            
            encoder = EncoderClassifier.from_hparams(
                source="speechbrain/spkrec-ecapa-voxceleb",
                savedir="./models/pretrained/ecapa",
                run_opts={"device": self.device}
            )
            
            signal, sr = torchaudio.load(audio_path)
            if sr != 16000:
                signal = torchaudio.functional.resample(signal, sr, 16000)
            
            if signal.shape[0] > 1:
                signal = signal.mean(dim=0, keepdim=True)
            
            embedding = encoder.encode_batch(signal)
            return embedding.squeeze().cpu().numpy().tolist()
            
        except Exception as e:
            print(f"   ⚠️  Could not extract embedding: {e}")
            return []
    
    def _upload_to_elevenlabs(
        self,
        name: str,
        audio_path: str,
        description: Optional[str]
    ) -> str:
        """Upload voice to ElevenLabs for cloning."""
        with open(audio_path, "rb") as f:
            response = self._model.clone(
                name=name,
                description=description or f"Cloned voice: {name}",
                files=[f],
            )
        return response.voice_id
    
    def _extract_bark_embedding(self, audio_path: str) -> List[float]:
        """Extract Bark speaker embedding."""
        # Bark uses its own format - placeholder
        return []
    
    def _extract_rvc_features(self, audio_path: str) -> List[float]:
        """Extract RVC voice features."""
        # RVC feature extraction - placeholder
        return []
    
    def synthesize(
        self,
        text: str,
        voice: str = "default",
        language: str = "pl",
        speed: float = 1.0,
        pitch_shift: int = 0,
        output_path: Optional[str] = None,
    ) -> torch.Tensor:
        """
        Synthesizes speech/singing using registered voice.
        
        Args:
            text: text to speak/sing
            voice: name of registered voice
            language: language ("pl", "en", "es", etc.)
            speed: speaking speed (0.5-2.0)
            pitch_shift: pitch shift in semitones
            output_path: optional path to save
            
        Returns:
            audio tensor [samples] @ sample_rate
        """
        self._init_backend()
        
        if voice != "default" and voice not in self.voices:
            raise ValueError(
                f"Voice '{voice}' not registered. "
                f"Available: {list(self.voices.keys())}"
            )
        
        print(f"🎤 Synthesizing with voice '{voice}'...")
        print(f"   Text: {text[:50]}..." if len(text) > 50 else f"   Text: {text}")
        
        if self.backend == "coqui":
            audio = self._synthesize_coqui(text, voice, language, speed)
        elif self.backend == "elevenlabs":
            audio = self._synthesize_elevenlabs(text, voice, speed)
        elif self.backend == "bark":
            audio = self._synthesize_bark(text, voice)
        elif self.backend == "rvc":
            audio = self._synthesize_rvc(text, voice, pitch_shift)
        elif self.backend == "gpt_sovits":
            audio = self._synthesize_gpt_sovits(text, voice, language, speed)
        elif self.backend == "fish_speech":
            audio = self._synthesize_fish_speech(text, voice, language, speed)
        
        # Apply pitch shift if needed
        if pitch_shift != 0 and self.backend != "rvc":
            audio = self._apply_pitch_shift(audio, pitch_shift)
        
        if output_path:
            self.save_audio(audio, output_path)
        
        return audio
    
    def _synthesize_coqui(
        self,
        text: str,
        voice: str,
        language: str,
        speed: float
    ) -> torch.Tensor:
        """Synthesize with Coqui XTTS."""
        reference_audio = None
        if voice != "default" and voice in self.voices:
            reference_audio = self.voices[voice].reference_audio_path
        
        # XTTS wymaga reference audio dla voice cloningu
        if reference_audio:
            audio_np = self._model.tts(
                text=text,
                speaker_wav=reference_audio,
                language=language,
                speed=speed,
            )
        else:
            # Default voice
            audio_np = self._model.tts(
                text=text,
                language=language,
                speed=speed,
            )
        
        # Coqui TTS may return list or np.ndarray
        import numpy as np
        if isinstance(audio_np, list):
            audio_np = np.array(audio_np, dtype=np.float32)
        elif not isinstance(audio_np, np.ndarray):
            audio_np = np.array(audio_np, dtype=np.float32)
        
        return torch.from_numpy(audio_np).float()
    
    def _synthesize_elevenlabs(
        self,
        text: str,
        voice: str,
        speed: float
    ) -> torch.Tensor:
        """Synthesize with ElevenLabs."""
        voice_id = None
        if voice != "default" and voice in self.voices:
            voice_id = self.voices[voice].speaker_id
        
        audio_bytes = self._model.generate(
            text=text,
            voice=voice_id or "Adam",  # default ElevenLabs voice
            model="eleven_multilingual_v2",
        )
        
        # Convert bytes to tensor
        import io
        audio, sr = torchaudio.load(io.BytesIO(audio_bytes))
        return audio.squeeze()
    
    def _synthesize_bark(self, text: str, voice: str) -> torch.Tensor:
        """Synthesize with Bark."""
        # Bark speaker prompt
        speaker = None
        if voice != "default" and voice in self.voices:
            # Use saved embedding as speaker prompt
            speaker = f"speaker_{voice}"
        
        audio_np = self._model["generate"](text, history_prompt=speaker)
        return torch.from_numpy(audio_np).float()
    
    def _synthesize_rvc(
        self,
        text: str,
        voice: str,
        pitch_shift: int
    ) -> torch.Tensor:
        """Synthesize with RVC (voice conversion)."""
        # RVC wymaga source audio - najpierw generujemy TTS, potem konwertujemy
        # Placeholder implementation
        raise NotImplementedError("RVC synthesis requires additional setup")
    
    def _synthesize_gpt_sovits(
        self,
        text: str,
        voice: str,
        language: str,
        speed: float
    ) -> torch.Tensor:
        """
        Synthesize with GPT-SoVITS.
        
        GPT-SoVITS supports two modes:
        1. Zero-shot: Just provide 5s reference audio
        2. Few-shot: Fine-tuned model for specific voice
        
        Language codes:
        - 'zh': Chinese
        - 'en': English  
        - 'ja': Japanese
        - 'ko': Korean
        - 'yue': Cantonese
        - 'all_zh'/'all_ja'/etc: Force all text as one language
        """
        import requests
        import io
        
        # Get reference audio
        reference_audio = None
        if voice != "default" and voice in self.voices:
            reference_audio = self.voices[voice].reference_audio_path
        
        if not reference_audio:
            raise ValueError(
                "GPT-SoVITS requires reference audio. "
                "Register a voice first with synth.register_voice(name, audio_path)"
            )
        
        # Map language code
        lang_map = {
            'en': 'en', 'english': 'en',
            'zh': 'zh', 'chinese': 'zh',
            'ja': 'ja', 'japanese': 'ja',
            'ko': 'ko', 'korean': 'ko',
            'yue': 'yue', 'cantonese': 'yue',
            # For non-supported languages, try English
            'pl': 'en', 'de': 'en', 'fr': 'en', 'es': 'en', 'ru': 'en',
        }
        gpt_sovits_lang = lang_map.get(language.lower(), 'en')
        
        print(f"   GPT-SoVITS language: {gpt_sovits_lang}")
        print(f"   Reference audio: {Path(reference_audio).name}")
        
        if self._model["mode"] == "api":
            return self._synthesize_gpt_sovits_api(
                text, reference_audio, gpt_sovits_lang, speed
            )
        else:
            return self._synthesize_gpt_sovits_local(
                text, reference_audio, gpt_sovits_lang, speed
            )
    
    def _synthesize_gpt_sovits_api(
        self,
        text: str,
        reference_audio: str,
        language: str,
        speed: float
    ) -> torch.Tensor:
        """Synthesize using GPT-SoVITS HTTP API."""
        import requests
        import io
        
        # Load reference audio text (prompt text)
        # GPT-SoVITS needs to know what's said in the reference audio
        # If not available, we'll try without it
        prompt_text = ""
        
        # Check if we have metadata with transcript
        voice_name = None
        for name, voice in self.voices.items():
            if voice.reference_audio_path == reference_audio:
                voice_name = name
                if voice.metadata and 'transcript' in voice.metadata:
                    prompt_text = voice.metadata['transcript']
                break
        
        # Read reference audio file
        with open(reference_audio, 'rb') as f:
            ref_audio_bytes = f.read()
        
        # API v2 endpoint (recommended)
        url = f"{self.gpt_sovits_url}/tts"
        
        # Prepare multipart form data
        files = {
            'ref_audio': (Path(reference_audio).name, ref_audio_bytes, 'audio/wav'),
        }
        
        data = {
            'text': text,
            'text_lang': language,
            'ref_audio_text': prompt_text,
            'prompt_lang': language,
            'speed_factor': speed,
            'media_type': 'wav',
            'streaming_mode': False,
        }
        
        print(f"   Calling GPT-SoVITS API: {url}")
        
        try:
            response = requests.post(url, files=files, data=data, timeout=120)
            response.raise_for_status()
        except requests.exceptions.Timeout:
            raise RuntimeError("GPT-SoVITS API timeout - text may be too long")
        except requests.exceptions.RequestException as e:
            # Try alternative API format
            print(f"   Primary API failed, trying alternative...")
            return self._synthesize_gpt_sovits_api_v1(
                text, reference_audio, language, speed, prompt_text
            )
        
        # Parse response
        content_type = response.headers.get('Content-Type', '')
        
        if 'audio' in content_type or len(response.content) > 1000:
            # Direct audio response
            audio, sr = torchaudio.load(io.BytesIO(response.content))
            if sr != self._model["sample_rate"]:
                audio = torchaudio.functional.resample(
                    audio, sr, self._model["sample_rate"]
                )
            return audio.squeeze()
        else:
            # JSON response with error or URL
            try:
                result = response.json()
                if 'error' in result:
                    raise RuntimeError(f"GPT-SoVITS error: {result['error']}")
                elif 'audio_url' in result:
                    # Download audio from URL
                    audio_response = requests.get(result['audio_url'])
                    audio, sr = torchaudio.load(io.BytesIO(audio_response.content))
                    return audio.squeeze()
            except Exception as e:
                raise RuntimeError(f"Failed to parse GPT-SoVITS response: {e}")
    
    def _synthesize_gpt_sovits_api_v1(
        self,
        text: str,
        reference_audio: str,
        language: str,
        speed: float,
        prompt_text: str
    ) -> torch.Tensor:
        """Alternative API format for older GPT-SoVITS versions."""
        import requests
        import io
        import base64
        
        # Read and encode reference audio
        with open(reference_audio, 'rb') as f:
            ref_audio_b64 = base64.b64encode(f.read()).decode('utf-8')
        
        # API v1 format (JSON body)
        url = f"{self.gpt_sovits_url}/"
        
        payload = {
            "text": text,
            "text_language": language,
            "refer_wav_path": reference_audio,  # Some versions accept path
            "prompt_text": prompt_text,
            "prompt_language": language,
            "speed": speed,
        }
        
        try:
            response = requests.post(
                url, 
                json=payload, 
                timeout=120,
                headers={'Content-Type': 'application/json'}
            )
            response.raise_for_status()
            
            audio, sr = torchaudio.load(io.BytesIO(response.content))
            return audio.squeeze()
            
        except Exception as e:
            raise RuntimeError(
                f"GPT-SoVITS API failed. Make sure the server is running:\n"
                f"  cd GPT-SoVITS && python api_v2.py -a 0.0.0.0 -p 9880\n"
                f"Error: {e}"
            )
    
    def _synthesize_gpt_sovits_local(
        self,
        text: str,
        reference_audio: str,
        language: str,
        speed: float
    ) -> torch.Tensor:
        """Synthesize using local GPT-SoVITS installation."""
        # Local inference function loaded in _init_gpt_sovits
        inference_fn = self._model["inference_fn"]
        
        # Get prompt text from voice metadata if available
        prompt_text = ""
        for voice in self.voices.values():
            if voice.reference_audio_path == reference_audio:
                if voice.metadata and 'transcript' in voice.metadata:
                    prompt_text = voice.metadata['transcript']
                break
        
        # Call GPT-SoVITS inference
        # Returns tuple: (sample_rate, audio_numpy)
        sr, audio_np = inference_fn(
            ref_wav_path=reference_audio,
            prompt_text=prompt_text,
            prompt_language=language,
            text=text,
            text_language=language,
            speed=speed,
        )
        
        # Convert to tensor
        audio = torch.from_numpy(audio_np).float()
        
        # Ensure proper shape
        if audio.dim() == 1:
            pass  # Already [samples]
        elif audio.dim() == 2:
            audio = audio.squeeze(0)
        
        # Resample if needed
        target_sr = self._model["sample_rate"]
        if sr != target_sr:
            audio = torchaudio.functional.resample(audio, sr, target_sr)
        
        return audio
    
    def _synthesize_fish_speech(
        self,
        text: str,
        voice: str,
        language: str,
        speed: float
    ) -> torch.Tensor:
        """
        Synthesize with Fish Speech / OpenAudio.
        
        Fish Speech supports emotion markers in text:
        - Emotions: (angry), (sad), (excited), (surprised), etc.
        - Tones: (whispering), (shouting), (soft tone)
        - Effects: (laughing), (sighing), (crying)
        
        Example:
            text = "(excited) I can't believe we won! (laughing)"
        """
        import requests
        import io
        
        # Get reference audio
        reference_audio = None
        if voice != "default" and voice in self.voices:
            reference_audio = self.voices[voice].reference_audio_path
        
        if not reference_audio:
            raise ValueError(
                "Fish Speech requires reference audio. "
                "Register a voice first with synth.register_voice(name, audio_path)"
            )
        
        print(f"   🐟 Fish Speech synthesis")
        print(f"   Reference: {Path(reference_audio).name}")
        
        mode = self._model["mode"]
        
        if mode == "cloud":
            return self._synthesize_fish_speech_cloud(text, reference_audio, language, speed)
        elif mode == "api":
            return self._synthesize_fish_speech_api(text, reference_audio, language, speed)
        else:
            return self._synthesize_fish_speech_local(text, reference_audio, language, speed)
    
    def _synthesize_fish_speech_cloud(
        self,
        text: str,
        reference_audio: str,
        language: str,
        speed: float
    ) -> torch.Tensor:
        """Synthesize using Fish Audio cloud API."""
        import requests
        import io
        
        # Fish Audio API endpoint
        url = "https://api.fish.audio/v1/tts"
        
        # Read reference audio
        with open(reference_audio, 'rb') as f:
            ref_audio_bytes = f.read()
        
        # Prepare multipart request
        files = {
            'reference_audio': (Path(reference_audio).name, ref_audio_bytes, 'audio/wav'),
        }
        
        data = {
            'text': text,
            'language': language,
            'speed': speed,
        }
        
        headers = {
            'Authorization': f"Bearer {self._model['api_key']}",
        }
        
        print(f"   Calling Fish Audio cloud API...")
        
        response = requests.post(url, files=files, data=data, headers=headers, timeout=120)
        response.raise_for_status()
        
        # Parse audio response
        audio, sr = torchaudio.load(io.BytesIO(response.content))
        
        target_sr = self._model["sample_rate"]
        if sr != target_sr:
            audio = torchaudio.functional.resample(audio, sr, target_sr)
        
        return audio.squeeze()
    
    def _synthesize_fish_speech_api(
        self,
        text: str,
        reference_audio: str,
        language: str,
        speed: float
    ) -> torch.Tensor:
        """Synthesize using local Fish Speech API server."""
        import requests
        import io
        
        url = f"{self.fish_speech_url}/v1/tts"
        
        # Read reference audio
        with open(reference_audio, 'rb') as f:
            ref_audio_bytes = f.read()
        
        # Try different API formats
        # Format 1: Multipart form
        files = {
            'reference_audio': (Path(reference_audio).name, ref_audio_bytes, 'audio/wav'),
        }
        
        data = {
            'text': text,
            'chunk_length': 200,
            'format': 'wav',
            'streaming': False,
        }
        
        try:
            response = requests.post(url, files=files, data=data, timeout=120)
            response.raise_for_status()
            
            audio, sr = torchaudio.load(io.BytesIO(response.content))
            target_sr = self._model["sample_rate"]
            if sr != target_sr:
                audio = torchaudio.functional.resample(audio, sr, target_sr)
            return audio.squeeze()
            
        except requests.exceptions.RequestException:
            # Try format 2: JSON body with base64 audio
            import base64
            
            ref_audio_b64 = base64.b64encode(ref_audio_bytes).decode('utf-8')
            
            json_data = {
                "text": text,
                "reference_audio": ref_audio_b64,
                "reference_text": "",  # Optional transcript
                "max_new_tokens": 1024,
                "chunk_length": 200,
                "top_p": 0.7,
                "repetition_penalty": 1.2,
                "temperature": 0.7,
            }
            
            response = requests.post(
                f"{self.fish_speech_url}/v1/tts",
                json=json_data,
                timeout=120
            )
            response.raise_for_status()
            
            audio, sr = torchaudio.load(io.BytesIO(response.content))
            target_sr = self._model["sample_rate"]
            if sr != target_sr:
                audio = torchaudio.functional.resample(audio, sr, target_sr)
            return audio.squeeze()
    
    def _synthesize_fish_speech_local(
        self,
        text: str,
        reference_audio: str,
        language: str,
        speed: float
    ) -> torch.Tensor:
        """Synthesize using local Fish Speech installation."""
        # Local inference loaded in _init_fish_speech
        inference_fn = self._model["inference"]
        
        # Call fish_speech inference
        audio_np = inference_fn(
            text=text,
            reference_audio=reference_audio,
            max_new_tokens=1024,
            chunk_length=200,
            top_p=0.7,
            repetition_penalty=1.2,
            temperature=0.7,
        )
        
        # Convert to tensor
        audio = torch.from_numpy(audio_np).float()
        
        if audio.dim() == 2:
            audio = audio.squeeze(0)
        
        return audio
    
    def _apply_pitch_shift(
        self,
        audio: torch.Tensor,
        semitones: int
    ) -> torch.Tensor:
        """Apply pitch shift to audio."""
        try:
            import librosa
            
            # Convert to numpy
            audio_np = audio.numpy()
            
            # Pitch shift
            shifted = librosa.effects.pitch_shift(
                audio_np,
                sr=22050,  # assume 22kHz
                n_steps=semitones
            )
            
            return torch.from_numpy(shifted).float()
        except ImportError:
            print("   ⚠️  librosa not installed, skipping pitch shift")
            return audio
    
    def save_audio(
        self,
        audio: torch.Tensor,
        path: str,
        sample_rate: int = 22050
    ):
        """Save audio to file."""
        if audio.dim() == 1:
            audio = audio.unsqueeze(0)
        
        # Normalize
        max_val = audio.abs().max()
        if max_val > 0:
            audio = audio / max_val * 0.95
        
        torchaudio.save(path, audio, sample_rate)
        print(f"   ✓ Saved to {path}")
    
    def list_voices(self) -> List[str]:
        """List of registered voices."""
        return list(self.voices.keys())
    
    def save_voices(self, path: str):
        """Saves registered voices to file."""
        voices_data = {}
        for name, voice in self.voices.items():
            voices_data[name] = {
                "reference_audio_path": voice.reference_audio_path,
                "embedding": voice.embedding,
                "speaker_id": voice.speaker_id,
                "backend": voice.backend,
                "metadata": voice.metadata,
            }
        
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(voices_data, f, indent=2, ensure_ascii=False)
        print(f"💾 Saved {len(voices_data)} voices to {path}")
    
    def load_voices(self, path: str):
        """Loads registered voices from file."""
        with open(path, 'r', encoding='utf-8') as f:
            voices_data = json.load(f)
        
        for name, data in voices_data.items():
            self.voices[name] = RegisteredVoice(
                name=name,
                reference_audio_path=data["reference_audio_path"],
                embedding=data.get("embedding"),
                speaker_id=data.get("speaker_id"),
                backend=data.get("backend", self.backend),
                metadata=data.get("metadata"),
            )
        
        print(f"📂 Loaded {len(voices_data)} voices from {path}")


class SingingVoiceSynthesizer(VoiceSynthesizer):
    """
    Specialized SINGING synthesis (not just speech).
    
    Uses models better adapted for singing:
    - Pitch/melody control
    - Timing synchronized with music
    - Better note holding
    
    Backends:
    - diff_singer: DiffSinger (SOTA for singing)
    - so_vits_svc: So-VITS-SVC (singing voice conversion)
    - coqui: XTTS (more speech than singing)
    """
    
    def __init__(
        self,
        backend: str = "coqui",  # diff_singer or so_vits_svc for better singing
        **kwargs
    ):
        super().__init__(backend=backend, **kwargs)
        self.melody_extractor = None
    
    def synthesize_singing(
        self,
        lyrics: str,
        melody_audio: Optional[torch.Tensor] = None,
        melody_midi: Optional[str] = None,
        voice: str = "default",
        tempo_bpm: float = 120.0,
        key: str = "C",
        output_path: Optional[str] = None,
    ) -> torch.Tensor:
        """
        Synthesizes SINGING with text and melody.
        
        Args:
            lyrics: tekst do zaśpiewania (z timing markers opcjonalnie)
            melody_audio: audio z melodią do naśladowania
            melody_midi: MIDI file z melodią
            voice: zarejestrowany głos
            tempo_bpm: tempo w BPM
            key: tonacja
            output_path: gdzie zapisać
            
        Returns:
            audio tensor ze śpiewem
            
        Przykład:
            singing = synth.synthesize_singing(
                lyrics="La la la, to jest moja piosenka",
                melody_audio=generated_instrumental,
                voice="moj_glos",
                tempo_bpm=120
            )
        """
        print(f"🎵 Synthesizing singing with voice '{voice}'...")
        
        # Extract melody info if audio provided
        melody_info = None
        if melody_audio is not None:
            melody_info = self._extract_melody(melody_audio)
            print(f"   Extracted melody: {melody_info.get('notes', 'N/A')} notes")
        elif melody_midi:
            melody_info = self._load_midi_melody(melody_midi)
        
        # For now, use regular synthesis + pitch adjustment
        # Full singing synthesis would require DiffSinger or similar
        audio = self.synthesize(
            text=lyrics,
            voice=voice,
            language="pl",
            speed=1.0,
            output_path=None,
        )
        
        # TODO: Add melody following with pitch extraction
        # This would require:
        # 1. Extract pitch contour from melody_audio
        # 2. Apply pitch contour to synthesized speech
        # 3. Time-stretch to match rhythm
        
        if output_path:
            self.save_audio(audio, output_path)
        
        return audio
    
    def _extract_melody(self, audio: torch.Tensor) -> Dict:
        """Extract melody information from audio."""
        try:
            import librosa
            
            audio_np = audio.numpy()
            
            # Extract pitch
            pitches, magnitudes = librosa.piptrack(
                y=audio_np,
                sr=22050,
                fmin=80,
                fmax=1000,
            )
            
            # Get dominant pitch per frame
            pitch_track = []
            for t in range(pitches.shape[1]):
                index = magnitudes[:, t].argmax()
                pitch = pitches[index, t]
                if pitch > 0:
                    pitch_track.append(pitch)
            
            return {
                "pitch_track": pitch_track,
                "notes": len(pitch_track),
                "avg_pitch": np.mean(pitch_track) if pitch_track else 0,
            }
        except Exception as e:
            print(f"   ⚠️  Melody extraction failed: {e}")
            return {}
    
    def _load_midi_melody(self, midi_path: str) -> Dict:
        """Load melody from MIDI file."""
        try:
            import pretty_midi
            
            midi = pretty_midi.PrettyMIDI(midi_path)
            notes = []
            
            for instrument in midi.instruments:
                if not instrument.is_drum:
                    for note in instrument.notes:
                        notes.append({
                            "pitch": note.pitch,
                            "start": note.start,
                            "end": note.end,
                            "velocity": note.velocity,
                        })
            
            return {
                "notes": sorted(notes, key=lambda x: x["start"]),
                "tempo": midi.estimate_tempo(),
            }
        except ImportError:
            print("   ⚠️  pretty_midi not installed")
            return {}


# ============================================================================
# Convenience functions
# ============================================================================

def create_voice_synthesizer(
    backend: str = "gpt_sovits",
    **kwargs
) -> VoiceSynthesizer:
    """Factory function for VoiceSynthesizer."""
    return VoiceSynthesizer(backend=backend, **kwargs)


def create_singing_synthesizer(
    backend: str = "gpt_sovits",
    **kwargs
) -> SingingVoiceSynthesizer:
    """Factory function for SingingVoiceSynthesizer."""
    return SingingVoiceSynthesizer(backend=backend, **kwargs)


# ============================================================================
# Example usage
# ============================================================================

if __name__ == "__main__":
    print("""
╔══════════════════════════════════════════════════════════════╗
║           Voice Synthesis & Cloning Module                   ║
╠══════════════════════════════════════════════════════════════╣
║                                                              ║
║  🎤 GPT-SoVITS (RECOMMENDED - MIT License, SOTA quality)    ║
║                                                              ║
║  Zero-shot TTS: 5s voice sample → instant cloning           ║
║  Few-shot TTS: 1min training → perfect voice match          ║
║  Languages: EN, JA, KO, ZH, Cantonese                        ║
║  Output: 48kHz (v4), fast RTF ~0.028                         ║
║                                                              ║
║  Setup:                                                      ║
║    1. Clone: git clone https://github.com/RVC-Boss/GPT-SoVITS║
║    2. Install: cd GPT-SoVITS && pip install -r requirements  ║
║    3. Start API: python api_v2.py -a 0.0.0.0 -p 9880        ║
║                                                              ║
║  Usage:                                                      ║
║    synth = VoiceSynthesizer(backend="gpt_sovits")           ║
║    synth.register_voice("singer", "voice_5s.wav")           ║
║    audio = synth.synthesize(                                 ║
║        text="I walk alone through empty streets",            ║
║        voice="singer",                                       ║
║        language="en"                                         ║
║    )                                                        ║
║                                                              ║
║  ──────────────────────────────────────────────────          ║
║                                                              ║
║  Other Backends:                                             ║
║  - coqui: XTTS v2 (local, open source)                      ║
║  - elevenlabs: API (best quality, paid)                     ║
║  - bark: Meta (local, experimental)                         ║
║  - rvc: RVC voice conversion                                ║
║                                                              ║
║  Installation (other backends):                              ║
║    pip install TTS              # Coqui XTTS                ║
║    pip install elevenlabs       # ElevenLabs                ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
""")
