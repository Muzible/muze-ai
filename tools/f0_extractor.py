"""
🎵 F0 Extractor - Ekstrakcja konturu melodycznego (F0)

Wspiera różne metody ekstrakcji:
1. CREPE - najdokładniejsza sieć neuronowa (wymaga crepe)
2. PYIN - probabilistyczny YIN (librosa)
3. WORLD - szybki WORLD vocoder (wymaga pyworld)
4. librosa YIN - klasyczny algorytm

Użycie:
    from tools.f0_extractor import F0Extractor
    
    extractor = F0Extractor(method='pyin', sr=22050)
    f0, voiced = extractor.extract(audio_path)
    # f0: [T] array of frequencies in Hz (0 = unvoiced)
    # voiced: [T] boolean array
"""

import numpy as np
from pathlib import Path
from typing import Optional, Tuple, Union, List
import warnings


class F0Extractor:
    """
    Ekstraktor F0 (fundamental frequency) dla wokali i instrumentów.
    """
    
    SUPPORTED_METHODS = ['pyin', 'yin', 'crepe', 'world', 'parselmouth']
    
    def __init__(
        self,
        method: str = 'pyin',
        sr: int = 22050,
        hop_length: int = 256,
        fmin: float = 65.0,   # C2 - niska nuta wokalna
        fmax: float = 2000.0,  # B6 - wysoka nuta + harmoniczne
        center: bool = True,
        frame_length: int = 2048,
    ):
        """
        Args:
            method: Metoda ekstrakcji ('pyin', 'yin', 'crepe', 'world')
            sr: Sample rate
            hop_length: Hop między ramkami (= frame rate)
            fmin: Minimalna częstotliwość F0 w Hz
            fmax: Maksymalna częstotliwość F0 w Hz
            center: Czy centrować okna
            frame_length: Długość okna analizy
        """
        if method not in self.SUPPORTED_METHODS:
            warnings.warn(f"Method '{method}' unknown, using 'pyin'")
            method = 'pyin'
        
        self.method = method
        self.sr = sr
        self.hop_length = hop_length
        self.fmin = fmin
        self.fmax = fmax
        self.center = center
        self.frame_length = frame_length
        
        # Frame rate (ramki na sekundę)
        self.frame_rate = sr / hop_length
        
        # Lazy load heavy dependencies
        self._librosa = None
        self._crepe = None
        self._world = None
        self._parselmouth = None
    
    @property
    def librosa(self):
        if self._librosa is None:
            import librosa
            self._librosa = librosa
        return self._librosa
    
    def extract(
        self,
        audio: Union[str, Path, np.ndarray],
        sr: Optional[int] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Ekstraktuj F0 z audio.
        
        Args:
            audio: Ścieżka do pliku lub array audio
            sr: Sample rate (jeśli array)
            
        Returns:
            f0: [T] array częstotliwości w Hz (0 = unvoiced)
            voiced_flag: [T] boolean array (True = voiced)
        """
        # Load audio if path
        if isinstance(audio, (str, Path)):
            audio, sr = self.librosa.load(str(audio), sr=self.sr, mono=True)
        elif sr is None:
            sr = self.sr
        
        # Resample if needed
        if sr != self.sr:
            audio = self.librosa.resample(audio, orig_sr=sr, target_sr=self.sr)
        
        # Extract F0 using chosen method
        if self.method == 'pyin':
            return self._extract_pyin(audio)
        elif self.method == 'yin':
            return self._extract_yin(audio)
        elif self.method == 'crepe':
            return self._extract_crepe(audio)
        elif self.method == 'world':
            return self._extract_world(audio)
        elif self.method == 'parselmouth':
            return self._extract_parselmouth(audio)
        else:
            return self._extract_pyin(audio)
    
    def _extract_pyin(self, audio: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        PYIN - Probabilistyczny YIN (librosa).
        
        Dobra dokładność, rozsądna szybkość.
        """
        f0, voiced_flag, voiced_probs = self.librosa.pyin(
            audio,
            fmin=self.fmin,
            fmax=self.fmax,
            sr=self.sr,
            hop_length=self.hop_length,
            frame_length=self.frame_length,
            center=self.center,
        )
        
        # Replace NaN with 0 (unvoiced)
        f0 = np.nan_to_num(f0, nan=0.0)
        
        return f0, voiced_flag
    
    def _extract_yin(self, audio: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        YIN - klasyczny algorytm (librosa).
        
        Szybszy ale mniej dokładny niż PYIN.
        """
        f0 = self.librosa.yin(
            audio,
            fmin=self.fmin,
            fmax=self.fmax,
            sr=self.sr,
            hop_length=self.hop_length,
            frame_length=self.frame_length,
            center=self.center,
        )
        
        # Estimate voiced from f0 validity
        voiced_flag = (f0 > self.fmin) & (f0 < self.fmax)
        f0 = np.where(voiced_flag, f0, 0.0)
        
        return f0, voiced_flag
    
    def _extract_crepe(self, audio: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        CREPE - sieć neuronowa (najdokładniejsza).
        
        Wymaga: pip install crepe tensorflow
        """
        try:
            import crepe
        except ImportError:
            warnings.warn("CREPE not installed. Using PYIN instead. Run: pip install crepe")
            return self._extract_pyin(audio)
        
        # CREPE wymaga 16kHz
        if self.sr != 16000:
            audio_16k = self.librosa.resample(audio, orig_sr=self.sr, target_sr=16000)
        else:
            audio_16k = audio
        
        # CREPE step size in ms (hop_length w ms)
        step_size = (self.hop_length / self.sr) * 1000
        
        time, frequency, confidence, _ = crepe.predict(
            audio_16k,
            16000,
            step_size=step_size,
            viterbi=True,  # Smooth trajectory
            center=self.center,
        )
        
        # Voiced if confidence > threshold
        voiced_flag = confidence > 0.5
        f0 = np.where(voiced_flag, frequency, 0.0)
        
        return f0, voiced_flag
    
    def _extract_world(self, audio: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        WORLD vocoder - bardzo szybki.
        
        Wymaga: pip install pyworld
        """
        try:
            import pyworld as pw
        except ImportError:
            warnings.warn("PyWORLD not installed. Using PYIN instead. Run: pip install pyworld")
            return self._extract_pyin(audio)
        
        # WORLD wymaga float64
        audio = audio.astype(np.float64)
        
        # DIO + StoneMask (refined F0)
        frame_period = (self.hop_length / self.sr) * 1000  # ms
        
        f0, t = pw.dio(
            audio,
            self.sr,
            f0_floor=self.fmin,
            f0_ceil=self.fmax,
            frame_period=frame_period,
        )
        
        # Refine with StoneMask
        f0 = pw.stonemask(audio, f0, t, self.sr)
        
        # Voiced flag
        voiced_flag = f0 > 0
        
        return f0.astype(np.float32), voiced_flag
    
    def _extract_parselmouth(self, audio: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Parselmouth (Praat) - klasyczne narzędzie fonetyczne.
        
        Wymaga: pip install praat-parselmouth
        """
        try:
            import parselmouth
        except ImportError:
            warnings.warn("Parselmouth not installed. Using PYIN instead. Run: pip install praat-parselmouth")
            return self._extract_pyin(audio)
        
        # Create Praat Sound object
        snd = parselmouth.Sound(audio, sampling_frequency=self.sr)
        
        # Extract pitch
        pitch = snd.to_pitch_ac(
            time_step=self.hop_length / self.sr,
            pitch_floor=self.fmin,
            pitch_ceiling=self.fmax,
        )
        
        # Get F0 values
        f0 = pitch.selected_array['frequency']
        
        # Replace 0 with NaN then back
        voiced_flag = f0 > 0
        
        return f0.astype(np.float32), voiced_flag
    
    def extract_coarse(
        self,
        audio: Union[str, Path, np.ndarray],
        sr: Optional[int] = None,
        num_bins: int = 128,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Ekstraktuj zdyskretyzowany F0 (MIDI-like bins).
        
        Args:
            audio: Audio input
            sr: Sample rate
            num_bins: Liczba binów (128 = pełny zakres MIDI)
            
        Returns:
            f0_coarse: [T] array of bin indices (num_bins = unvoiced)
            voiced_flag: [T] boolean array
        """
        f0, voiced = self.extract(audio, sr)
        
        # Convert Hz to MIDI then to bins
        # MIDI = 12 * log2(f/440) + 69
        f0_safe = np.maximum(f0, 1.0)  # Avoid log(0)
        midi = 12 * np.log2(f0_safe / 440.0) + 69
        
        # Clamp to MIDI range (21-108 typical piano range)
        midi = np.clip(midi, 0, num_bins - 1)
        
        # Quantize to bins
        f0_coarse = midi.astype(np.int64)
        
        # Set unvoiced to special bin
        f0_coarse[~voiced] = num_bins
        
        return f0_coarse, voiced
    
    def interpolate_unvoiced(
        self,
        f0: np.ndarray,
        voiced: np.ndarray,
        method: str = 'linear',
    ) -> np.ndarray:
        """
        Interpoluj unvoiced segmenty (dla gładszego konturu).
        
        Args:
            f0: F0 contour
            voiced: Voiced flag
            method: 'linear', 'nearest', 'zero'
            
        Returns:
            f0_interp: Interpolated F0
        """
        if method == 'zero':
            return np.where(voiced, f0, 0.0)
        
        if not voiced.any():
            return f0
        
        from scipy.interpolate import interp1d
        
        indices = np.arange(len(f0))
        voiced_indices = indices[voiced]
        voiced_f0 = f0[voiced]
        
        if len(voiced_indices) < 2:
            return f0
        
        # Interpolate
        interp_func = interp1d(
            voiced_indices,
            voiced_f0,
            kind=method,
            bounds_error=False,
            fill_value=(voiced_f0[0], voiced_f0[-1]),
        )
        
        f0_interp = interp_func(indices)
        
        return f0_interp.astype(np.float32)
    
    def hz_to_coarse(
        self,
        f0: np.ndarray,
        num_bins: int = 128,
    ) -> np.ndarray:
        """
        Konwertuj F0 w Hz na zdyskretyzowane biny MIDI-like.
        
        Args:
            f0: [T] array częstotliwości w Hz (0 = unvoiced)
            num_bins: Liczba binów (128 = pełny zakres MIDI)
            
        Returns:
            f0_coarse: [T] array of bin indices (num_bins = unvoiced)
        """
        # Voiced mask (f0 > 0)
        voiced = f0 > 0
        
        # Convert Hz to MIDI: MIDI = 12 * log2(f/440) + 69
        f0_safe = np.maximum(f0, 1.0)  # Avoid log(0)
        midi = 12 * np.log2(f0_safe / 440.0) + 69
        
        # Clamp to MIDI range
        midi = np.clip(midi, 0, num_bins - 1)
        
        # Quantize to bins
        f0_coarse = midi.astype(np.int64)
        
        # Set unvoiced to special bin
        f0_coarse[~voiced] = num_bins
        
        return f0_coarse
    
    def get_statistics(
        self,
        f0: np.ndarray,
        voiced: np.ndarray,
    ) -> dict:
        """
        Oblicz statystyki F0.
        
        Returns:
            dict z: mean, std, min, max, range, voiced_ratio
        """
        voiced_f0 = f0[voiced]
        
        if len(voiced_f0) == 0:
            return {
                'f0_mean': 0.0,
                'f0_std': 0.0,
                'f0_min': 0.0,
                'f0_max': 0.0,
                'f0_range': 0.0,
                'voiced_ratio': 0.0,
            }
        
        return {
            'f0_mean': float(np.mean(voiced_f0)),
            'f0_std': float(np.std(voiced_f0)),
            'f0_min': float(np.min(voiced_f0)),
            'f0_max': float(np.max(voiced_f0)),
            'f0_range': float(np.max(voiced_f0) - np.min(voiced_f0)),
            'voiced_ratio': float(np.mean(voiced)),
        }


def extract_f0_from_vocals(
    vocals_path: str,
    method: str = 'pyin',
    sr: int = 22050,
    hop_length: int = 256,
) -> dict:
    """
    Convenience function: Ekstraktuj F0 z pliku wokali.
    
    Returns dict z:
        - f0: lista wartości Hz
        - f0_coarse: lista binów MIDI-like
        - voiced: lista bool
        - statistics: dict ze statystykami
    """
    extractor = F0Extractor(method=method, sr=sr, hop_length=hop_length)
    
    f0, voiced = extractor.extract(vocals_path)
    f0_coarse, _ = extractor.extract_coarse(vocals_path)
    stats = extractor.get_statistics(f0, voiced)
    
    return {
        'f0': f0.tolist(),
        'f0_coarse': f0_coarse.tolist(),
        'voiced': voiced.tolist(),
        'f0_statistics': stats,
        'f0_frame_rate': extractor.frame_rate,
        'f0_hop_length': hop_length,
    }


if __name__ == '__main__':
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python f0_extractor.py <audio_file> [method]")
        print("Methods: pyin, yin, crepe, world, parselmouth")
        sys.exit(1)
    
    audio_path = sys.argv[1]
    method = sys.argv[2] if len(sys.argv) > 2 else 'pyin'
    
    print(f"Extracting F0 from {audio_path} using {method}...")
    
    result = extract_f0_from_vocals(audio_path, method=method)
    
    print(f"\nStatistics:")
    for key, value in result['f0_statistics'].items():
        print(f"  {key}: {value:.2f}")
    
    print(f"\nF0 frames: {len(result['f0'])}")
    print(f"Frame rate: {result['f0_frame_rate']:.1f} fps")
