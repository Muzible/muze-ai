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
- ElevenLabs - API, bardzo wysoka jakość
- Bark - open source, Facebook
- RVC (Retrieval Voice Conversion) - konwersja głosu

Przykład użycia:
    from models.voice_synthesis import VoiceSynthesizer, VoiceExtractorFromSong
    
    # Tryb 1: Własny głos
    synth = VoiceSynthesizer(backend="coqui")
    synth.register_voice("moj_glos", "nagranie.wav")
    
    # Tryb 2: Wyekstrahuj głos z utworu i sklonuj
    extractor = VoiceExtractorFromSong()
    vocals_path = extractor.extract_vocals("piosenka_artysty.mp3")
    synth.register_voice("artysta_x", vocals_path)
    
    # Wygeneruj śpiew
    audio = synth.synthesize(text="Nowy tekst", voice="artysta_x")
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
    """Zarejestrowany głos użytkownika."""
    name: str
    reference_audio_path: str
    embedding: Optional[List[float]] = None
    speaker_id: Optional[str] = None  # dla API jak ElevenLabs
    backend: str = "coqui"
    metadata: Optional[Dict] = None
    source_type: str = "recording"  # "recording", "extracted_from_song"


class VoiceExtractorFromSong:
    """
    Ekstrahuje wokal z utworu muzycznego do klonowania głosu.
    
    Pipeline:
    1. Demucs/Spleeter separuje wokal od instrumentów
    2. Opcjonalnie: diarization jeśli wielu wokalistów
    3. Zwraca czysty wokal do voice cloningu
    
    Użycie:
        extractor = VoiceExtractorFromSong()
        vocals = extractor.extract_vocals("song.mp3")
        # vocals to ścieżka do pliku z samym wokalem
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
            output_dir: gdzie zapisać wyekstrahowany wokal
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
    """
    
    SUPPORTED_BACKENDS = ["coqui", "elevenlabs", "bark", "rvc"]
    
    def __init__(
        self,
        backend: str = "coqui",
        api_key: Optional[str] = None,
        device: str = "cpu",
        model_path: Optional[str] = None,
    ):
        """
        Inicjalizuje syntezator głosu.
        
        Args:
            backend: "coqui", "elevenlabs", "bark", "rvc"
            api_key: klucz API (dla elevenlabs)
            device: cpu/cuda/mps
            model_path: ścieżka do modelu (dla RVC)
        """
        if backend not in self.SUPPORTED_BACKENDS:
            raise ValueError(f"Unknown backend: {backend}. Supported: {self.SUPPORTED_BACKENDS}")
        
        self.backend = backend
        self.api_key = api_key
        self.device = device
        self.model_path = model_path
        
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
            # RVC ma różne implementacje, używamy popularnej
            print(f"🎤 Loading RVC model from {self.model_path}...")
            # Placeholder - RVC wymaga specyficznej konfiguracji
            self._model = {"model_path": self.model_path}
            print("   ✓ RVC loaded")
        except Exception as e:
            raise RuntimeError(f"Failed to load RVC model: {e}")
    
    def register_voice(
        self,
        name: str,
        reference_audio: str,
        description: Optional[str] = None,
        source_type: str = "recording",
    ) -> RegisteredVoice:
        """
        Rejestruje głos użytkownika do późniejszego użycia.
        
        Args:
            name: unikalna nazwa dla tego głosu (np. "moj_glos")
            reference_audio: ścieżka do nagrania (10-30s, czyste audio)
            description: opcjonalny opis
            source_type: "recording" (własne nagranie) lub "extracted_from_song"
            
        Returns:
            RegisteredVoice object
            
        Przykład:
            synth.register_voice("adam", "nagranie_adama.wav")
            synth.register_voice("kasia", "nagranie_kasi.mp3", "Kasia - alto")
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
            # XTTS używa reference audio bezpośrednio
            # Możemy też wyekstrahować embedding dla cache'a
            voice.embedding = self._extract_voice_embedding(reference_audio)
        
        elif self.backend == "elevenlabs":
            # Upload voice do ElevenLabs (voice cloning)
            voice.speaker_id = self._upload_to_elevenlabs(name, reference_audio, description)
        
        elif self.backend == "bark":
            # Bark używa speaker embeddings
            voice.embedding = self._extract_bark_embedding(reference_audio)
        
        elif self.backend == "rvc":
            # RVC potrzebuje feature extraction
            voice.embedding = self._extract_rvc_features(reference_audio)
        
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
            description: opcjonalny opis
            
        Returns:
            RegisteredVoice object
            
        Przykład:
            # Wyekstrahuj głos z piosenki Queen
            synth.register_voice_from_song(
                "freddie",
                "queen_bohemian_rhapsody.mp3",
                description="Freddie Mercury vocal"
            )
            
            # Teraz możesz śpiewać "jak Freddie"
            audio = synth.synthesize(
                text="Nowy tekst do zaśpiewania",
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
        # Bark używa własnego formatu - placeholder
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
        Syntezuje mowę/śpiew używając zarejestrowanego głosu.
        
        Args:
            text: tekst do wypowiedzenia/zaśpiewania
            voice: nazwa zarejestrowanego głosu
            language: język ("pl", "en", "es", etc.)
            speed: prędkość mówienia (0.5-2.0)
            pitch_shift: przesunięcie pitch w półtonach
            output_path: opcjonalna ścieżka do zapisania
            
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
        
        # Coqui TTS może zwrócić listę lub np.ndarray
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
        """Lista zarejestrowanych głosów."""
        return list(self.voices.keys())
    
    def save_voices(self, path: str):
        """Zapisuje zarejestrowane głosy do pliku."""
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
        """Wczytuje zarejestrowane głosy z pliku."""
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
    Wyspecjalizowana synteza ŚPIEWU (nie tylko mowy).
    
    Używa modeli lepiej przystosowanych do śpiewania:
    - Kontola pitch/melodii
    - Timing synchronizowany z muzyką
    - Lepsze trzymanie nut
    
    Backendy:
    - diff_singer: DiffSinger (SOTA dla singing)
    - so_vits_svc: So-VITS-SVC (singing voice conversion)
    - coqui: XTTS (bardziej mówienie niż śpiew)
    """
    
    def __init__(
        self,
        backend: str = "coqui",  # diff_singer lub so_vits_svc dla lepszego śpiewu
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
        Syntezuje ŚPIEW z tekstem i melodią.
        
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
    backend: str = "coqui",
    **kwargs
) -> VoiceSynthesizer:
    """Factory function for VoiceSynthesizer."""
    return VoiceSynthesizer(backend=backend, **kwargs)


def create_singing_synthesizer(
    backend: str = "coqui",
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
║  Użycie:                                                     ║
║                                                              ║
║  1. Nagraj próbkę swojego głosu (10-30 sekund)              ║
║     - Mów wyraźnie                                           ║
║     - Cisza w tle                                            ║
║     - Różnorodne słowa/dźwięki                              ║
║                                                              ║
║  2. Zarejestruj głos:                                        ║
║     synth = VoiceSynthesizer(backend="coqui")               ║
║     synth.register_voice("moj_glos", "moje_nagranie.wav")   ║
║                                                              ║
║  3. Generuj:                                                 ║
║     audio = synth.synthesize(                                ║
║         text="Cześć, to jest mój sklonowany głos!",         ║
║         voice="moj_glos",                                    ║
║         language="pl"                                        ║
║     )                                                        ║
║                                                              ║
║  Backendy:                                                   ║
║  - coqui: XTTS v2 (lokalne, dobre jakościowo)              ║
║  - elevenlabs: API (najlepsza jakość, płatne)              ║
║  - bark: Meta (lokalne, eksperymentalne)                    ║
║                                                              ║
║  Instalacja:                                                 ║
║    pip install TTS              # dla Coqui                 ║
║    pip install elevenlabs       # dla ElevenLabs            ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
""")
