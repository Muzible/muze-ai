"""
🎵 Segment Annotator - Automatyczna/semi-automatyczna anotacja sekcji utworów muzycznych

Wykrywa strukturę utworu (Intro, Verse, Chorus, Bridge, Outro, etc.) używając:
1. Analizy zmian energii (RMS, spectral flux)
2. Analizy struktury harmonicznej (chroma similarity)
3. Detekcji powtórzeń (self-similarity matrix)
4. Analizy rytmicznej (beat tracking, tempo changes)
5. Detekcji wokali (vocal activity detection)

Użycie:
    python tools_v2/segment_annotator.py --input ./music/track.mp3 --output ./annotations/
    python tools_v2/segment_annotator.py --input_dir ./music/fma_small --output ./data_v2/segments.json
"""

import os
import json
import argparse
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from enum import Enum
import warnings
warnings.filterwarnings('ignore')

try:
    import librosa
    import librosa.display
except ImportError:
    raise ImportError("librosa required: pip install librosa")

try:
    from scipy import ndimage, signal
    from scipy.spatial.distance import cdist
except ImportError:
    raise ImportError("scipy required: pip install scipy")

try:
    from sklearn.cluster import AgglomerativeClustering
except ImportError:
    raise ImportError("sklearn required: pip install scikit-learn")


class SectionType(Enum):
    """Typy sekcji w utworze muzycznym"""
    INTRO = "intro"
    VERSE = "verse"
    PRE_CHORUS = "pre_chorus"
    CHORUS = "chorus"
    POST_CHORUS = "post_chorus"
    BRIDGE = "bridge"
    INSTRUMENTAL = "instrumental"
    SOLO = "solo"
    BREAKDOWN = "breakdown"
    BUILDUP = "buildup"
    DROP = "drop"
    OUTRO = "outro"
    UNKNOWN = "unknown"


@dataclass
class MusicSegment:
    """Single track segment"""
    start_time: float           # Start time (seconds)
    end_time: float             # End time (seconds)
    section_type: str           # Section type (from SectionType)
    confidence: float           # Detection confidence (0-1)
    
    # Musical features of segment
    tempo: float                # Average tempo in BPM
    key: str                    # Dominant key
    energy: float               # Average energy (0-1)
    loudness: float             # Average loudness (dB)
    spectral_centroid: float    # Sound brightness
    
    # Dodatkowe informacje
    has_vocals: bool            # Czy segment zawiera wokale
    is_repetition_of: Optional[int] = None  # Indeks powtarzanego segmentu
    label_source: str = "auto"  # "auto", "manual", "corrected"
    
    # Prompt do generacji (wygenerowany automatycznie)
    auto_prompt: str = ""


@dataclass 
class AnnotatedTrack:
    """Full track annotation"""
    track_id: str
    file_path: str
    duration: float
    sample_rate: int
    
    # Globalne cechy
    global_tempo: float
    global_key: str
    global_energy: float
    
    # Segmenty
    segments: List[MusicSegment]
    
    # Metadane
    annotation_version: str = "2.0"
    annotator: str = "auto"


class SegmentAnnotator:
    """
    Główna klasa do anotacji segmentów muzycznych.
    
    Algorytm:
    1. Wczytaj audio i oblicz cechy (chroma, MFCC, RMS, spectral)
    2. Zbuduj self-similarity matrix
    3. Znajdź granice segmentów (novelty function)
    4. Klasteruj podobne segmenty
    5. Przypisz etykiety sekcji na podstawie cech i pozycji
    6. Wygeneruj automatyczne prompty
    """
    
    def __init__(
        self,
        sample_rate: int = 22050,
        hop_length: int = 512,
        min_segment_duration: float = 4.0,  # Minimum segment length (s)
        max_segment_duration: float = 60.0,  # Maximum segment length (s)
        n_mfcc: int = 20,
        n_chroma: int = 12,
        vocal_detection: bool = True,
        device: str = "cpu",
    ):
        self.sr = sample_rate
        self.hop_length = hop_length
        self.min_segment_duration = min_segment_duration
        self.max_segment_duration = max_segment_duration
        self.n_mfcc = n_mfcc
        self.n_chroma = n_chroma
        self.vocal_detection = vocal_detection
        self.device = device
        
        # Lazy loaded models
        self._vocal_detector = None
    
    def annotate(self, audio_path: str, track_id: Optional[str] = None) -> AnnotatedTrack:
        """
        Główna metoda - anotuje cały utwór.
        
        Args:
            audio_path: Ścieżka do pliku audio
            track_id: Opcjonalny identyfikator (domyślnie nazwa pliku)
            
        Returns:
            AnnotatedTrack z pełną anotacją
        """
        audio_path = Path(audio_path)
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        
        track_id = track_id or audio_path.stem
        print(f"\n🎵 Annotating: {audio_path.name}")
        
        # 1. Wczytaj audio
        print("  📂 Loading audio...")
        y, sr = librosa.load(str(audio_path), sr=self.sr)
        duration = len(y) / sr
        print(f"     Duration: {duration:.1f}s")
        
        # 2. Ekstrahuj cechy
        print("  🔍 Extracting features...")
        features = self._extract_features(y, sr)
        
        # 3. Find segment boundaries
        print("  ✂️  Detecting boundaries...")
        boundaries = self._detect_boundaries(features, duration)
        print(f"     Found {len(boundaries)-1} segments")
        
        # 4. Analyze each segment
        print("  📊 Analyzing segments...")
        raw_segments = self._analyze_segments(y, sr, boundaries, features)
        
        # 5. Klasteruj podobne segmenty
        print("  🔗 Clustering similar segments...")
        clustered_segments = self._cluster_segments(raw_segments, features, boundaries)
        
        # 6. Przypisz etykiety sekcji
        print("  🏷️  Labeling sections...")
        labeled_segments = self._label_sections(clustered_segments, duration)
        
        # 7. Wygeneruj prompty
        print("  📝 Generating prompts...")
        final_segments = self._generate_prompts(labeled_segments)
        
        # 8. Oblicz globalne cechy
        global_tempo = float(np.median([s.tempo for s in final_segments]))
        global_key = self._get_dominant_key([s.key for s in final_segments])
        global_energy = float(np.mean([s.energy for s in final_segments]))
        
        print(f"  ✅ Done! Global tempo: {global_tempo:.0f} BPM, Key: {global_key}")
        
        return AnnotatedTrack(
            track_id=track_id,
            file_path=str(audio_path),
            duration=duration,
            sample_rate=sr,
            global_tempo=global_tempo,
            global_key=global_key,
            global_energy=global_energy,
            segments=final_segments,
        )
    
    def annotate_audio(
        self, 
        y: np.ndarray, 
        sr: int, 
        track_id: str = "unknown",
        file_path: str = "",
        verbose: bool = False,
    ) -> AnnotatedTrack:
        """
        Anotuje utwór bezpośrednio z numpy array (bez wczytywania pliku).
        
        Args:
            y: Audio signal jako numpy array
            sr: Sample rate
            track_id: Identyfikator utworu
            file_path: Ścieżka do pliku (dla metadanych)
            verbose: Czy wyświetlać logi
            
        Returns:
            AnnotatedTrack z pełną anotacją
        """
        duration = len(y) / sr
        
        if verbose:
            print(f"\n🎵 Annotating: {track_id}")
            print(f"     Duration: {duration:.1f}s")
        
        # 1. Ekstrahuj cechy
        features = self._extract_features(y, sr)
        
        # 2. Find segment boundaries
        boundaries = self._detect_boundaries(features, duration)
        
        # 3. Analyze each segment
        raw_segments = self._analyze_segments(y, sr, boundaries, features)
        
        # 4. Klasteruj podobne segmenty
        clustered_segments = self._cluster_segments(raw_segments, features, boundaries)
        
        # 5. Przypisz etykiety sekcji
        labeled_segments = self._label_sections(clustered_segments, duration)
        
        # 6. Wygeneruj prompty
        final_segments = self._generate_prompts(labeled_segments)
        
        # 7. Oblicz globalne cechy
        if final_segments:
            global_tempo = float(np.median([s.tempo for s in final_segments]))
            global_key = self._get_dominant_key([s.key for s in final_segments])
            global_energy = float(np.mean([s.energy for s in final_segments]))
        else:
            global_tempo = 120.0
            global_key = "C"
            global_energy = 0.5
        
        return AnnotatedTrack(
            track_id=track_id,
            file_path=file_path,
            duration=duration,
            sample_rate=sr,
            global_tempo=global_tempo,
            global_key=global_key,
            global_energy=global_energy,
            segments=final_segments,
        )
    
    def _extract_features(self, y: np.ndarray, sr: int) -> Dict[str, np.ndarray]:
        """Ekstrahuje wszystkie potrzebne cechy audio"""
        
        features = {}
        
        # Chroma (harmony) - using chroma_stft instead of chroma_cqt for compatibility
        try:
            features['chroma'] = librosa.feature.chroma_cqt(
                y=y, sr=sr, hop_length=self.hop_length
            )
        except Exception:
            # Fallback dla starszych wersji numpy/librosa
            features['chroma'] = librosa.feature.chroma_stft(
                y=y, sr=sr, hop_length=self.hop_length
            )
        
        # MFCC (timbr)
        features['mfcc'] = librosa.feature.mfcc(
            y=y, sr=sr, n_mfcc=self.n_mfcc, hop_length=self.hop_length
        )
        
        # RMS energy
        features['rms'] = librosa.feature.rms(y=y, hop_length=self.hop_length)[0]
        
        # Spectral centroid (brightness)
        features['spectral_centroid'] = librosa.feature.spectral_centroid(
            y=y, sr=sr, hop_length=self.hop_length
        )[0]
        
        # Spectral contrast
        features['spectral_contrast'] = librosa.feature.spectral_contrast(
            y=y, sr=sr, hop_length=self.hop_length
        )
        
        # Tempo i beaty - z fallback dla numpy/librosa compatibility
        try:
            tempo, beats = librosa.beat.beat_track(y=y, sr=sr, hop_length=self.hop_length)
            features['tempo'] = float(tempo) if isinstance(tempo, (int, float)) else float(tempo[0])
            features['beats'] = beats
        except Exception:
            # Fallback: estimate tempo using onset strength
            try:
                onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=self.hop_length)
                tempogram = librosa.feature.tempogram(onset_envelope=onset_env, sr=sr, hop_length=self.hop_length)
                tempo_freqs = librosa.tempo_frequencies(tempogram.shape[0], sr=sr, hop_length=self.hop_length)
                mean_tempogram = np.mean(tempogram, axis=1)
                tempo_idx = np.argmax(mean_tempogram)
                features['tempo'] = float(tempo_freqs[tempo_idx])
                # Generate pseudo-beats based on tempo
                beat_interval = sr / self.hop_length * 60 / features['tempo']
                num_frames = features['chroma'].shape[1]
                features['beats'] = np.arange(0, num_frames, beat_interval).astype(int)
            except Exception:
                features['tempo'] = 120.0
                features['beats'] = np.array([0])
        
        # Beat-synchronous features (dla similarity matrix)
        features['chroma_sync'] = librosa.util.sync(
            features['chroma'], features['beats'], aggregate=np.median
        )
        features['mfcc_sync'] = librosa.util.sync(
            features['mfcc'], features['beats'], aggregate=np.median
        )
        
        # Self-similarity matrix (chroma-based)
        features['ssm'] = self._compute_ssm(features['chroma_sync'])
        
        # Frames per second
        features['fps'] = sr / self.hop_length
        
        return features
    
    def _compute_ssm(self, features: np.ndarray) -> np.ndarray:
        """Oblicza Self-Similarity Matrix"""
        # Normalize features
        features_norm = features / (np.linalg.norm(features, axis=0, keepdims=True) + 1e-8)
        
        # Cosine similarity
        ssm = np.dot(features_norm.T, features_norm)
        
        # Enhance diagonals (for repetitions)
        ssm = ndimage.median_filter(ssm, size=3)
        
        return ssm
    
    def _detect_boundaries(self, features: Dict, duration: float) -> List[float]:
        """
        Wykrywa granice segmentów używając novelty function.
        
        Novelty function mierzy jak bardzo zmienia się muzyka w danym momencie.
        """
        # Combine multiple features for boundary detection
        chroma = features['chroma']
        mfcc = features['mfcc']
        rms = features['rms']
        fps = features['fps']
        
        # Normalize features
        chroma_norm = (chroma - chroma.mean()) / (chroma.std() + 1e-8)
        mfcc_norm = (mfcc - mfcc.mean()) / (mfcc.std() + 1e-8)
        
        # Compute novelty for each feature type
        novelty_chroma = self._compute_novelty(chroma_norm)
        novelty_mfcc = self._compute_novelty(mfcc_norm)
        novelty_rms = self._compute_novelty(rms.reshape(1, -1))
        
        # Combine novelties
        min_len = min(len(novelty_chroma), len(novelty_mfcc), len(novelty_rms))
        novelty_combined = (
            0.4 * novelty_chroma[:min_len] + 
            0.4 * novelty_mfcc[:min_len] + 
            0.2 * novelty_rms[:min_len]
        )
        
        # Smooth
        novelty_smooth = ndimage.gaussian_filter1d(novelty_combined, sigma=5)
        
        # Find peaks (potential boundaries)
        min_frames = int(self.min_segment_duration * fps)
        peaks, _ = signal.find_peaks(
            novelty_smooth, 
            distance=min_frames,
            height=np.percentile(novelty_smooth, 60),  # Only significant peaks
            prominence=0.1
        )
        
        # Convert to time
        boundaries = [0.0]  # Start
        for peak in peaks:
            time = peak / fps
            if time < duration - 1:  # Not too close to end
                boundaries.append(time)
        boundaries.append(duration)  # End
        
        # Ensure no segment is too long
        max_frames = int(self.max_segment_duration * fps)
        final_boundaries = [0.0]
        for i in range(1, len(boundaries)):
            segment_duration = boundaries[i] - final_boundaries[-1]
            if segment_duration > self.max_segment_duration:
                # Split long segments
                n_splits = int(np.ceil(segment_duration / self.max_segment_duration))
                split_duration = segment_duration / n_splits
                for j in range(1, n_splits):
                    final_boundaries.append(final_boundaries[-1] + split_duration)
            final_boundaries.append(boundaries[i])
        
        return sorted(list(set(final_boundaries)))
    
    def _compute_novelty(self, features: np.ndarray) -> np.ndarray:
        """Oblicza novelty function z cech"""
        # Compute difference between consecutive frames
        diff = np.diff(features, axis=1)
        
        # Euclidean norm of differences
        novelty = np.sqrt(np.sum(diff ** 2, axis=0))
        
        return novelty
    
    def _analyze_segments(
        self, 
        y: np.ndarray, 
        sr: int, 
        boundaries: List[float],
        features: Dict
    ) -> List[Dict]:
        """Analyzes each segment and computes its features"""
        
        segments = []
        fps = features['fps']
        
        for i in range(len(boundaries) - 1):
            start = boundaries[i]
            end = boundaries[i + 1]
            
            # Wytnij fragment audio
            start_sample = int(start * sr)
            end_sample = int(end * sr)
            y_segment = y[start_sample:end_sample]
            
            if len(y_segment) < sr * 0.5:  # Too short
                continue
            
            # Oblicz cechy segmentu
            start_frame = int(start * fps)
            end_frame = int(end * fps)
            
            # Tempo (lokalne)
            try:
                tempo, _ = librosa.beat.beat_track(y=y_segment, sr=sr)
                tempo = float(tempo)
            except:
                tempo = features['tempo']
            
            # Tonacja (z chroma)
            chroma_segment = features['chroma'][:, start_frame:end_frame]
            if chroma_segment.size > 0:
                key = self._detect_key(chroma_segment)
            else:
                key = "C"
            
            # Energia
            rms_segment = features['rms'][start_frame:end_frame]
            energy = float(np.mean(rms_segment)) if len(rms_segment) > 0 else 0.5
            
            # Loudness (dB) - convert scalar to array for numpy 1.24+ compatibility
            mean_amplitude = np.mean(np.abs(y_segment)) + 1e-8
            loudness = float(20 * np.log10(mean_amplitude))  # Manual dB conversion
            
            # Spectral centroid (brightness)
            sc_segment = features['spectral_centroid'][start_frame:end_frame]
            spectral_centroid = float(np.mean(sc_segment)) if len(sc_segment) > 0 else 2000
            
            # Detekcja wokali (uproszczona)
            has_vocals = self._detect_vocals_simple(y_segment, sr)
            
            segments.append({
                'start_time': start,
                'end_time': end,
                'tempo': tempo,
                'key': key,
                'energy': min(1.0, energy * 10),  # Normalize to 0-1
                'loudness': loudness,
                'spectral_centroid': spectral_centroid,
                'has_vocals': has_vocals,
                'chroma_mean': np.mean(chroma_segment, axis=1) if chroma_segment.size > 0 else np.zeros(12),
                'mfcc_mean': np.mean(features['mfcc'][:, start_frame:end_frame], axis=1) if end_frame > start_frame else np.zeros(self.n_mfcc),
            })
        
        return segments
    
    def _detect_key(self, chroma: np.ndarray) -> str:
        """Detects key from chroma features"""
        keys = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        
        # Average chroma
        chroma_mean = np.mean(chroma, axis=1)
        
        # Profiles dla major i minor
        major_profile = np.array([1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1])
        minor_profile = np.array([1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0])
        
        best_corr = -1
        best_key = "C"
        best_mode = "major"
        
        for i, key in enumerate(keys):
            # Rotate profiles
            major_rot = np.roll(major_profile, i)
            minor_rot = np.roll(minor_profile, i)
            
            # Correlation
            corr_major = np.corrcoef(chroma_mean, major_rot)[0, 1]
            corr_minor = np.corrcoef(chroma_mean, minor_rot)[0, 1]
            
            if corr_major > best_corr:
                best_corr = corr_major
                best_key = key
                best_mode = "major"
            if corr_minor > best_corr:
                best_corr = corr_minor
                best_key = key
                best_mode = "minor"
        
        return f"{best_key}{' minor' if best_mode == 'minor' else ''}"
    
    def _detect_vocals_simple(self, y: np.ndarray, sr: int) -> bool:
        """Uproszczona detekcja wokali (na podstawie cech spektralnych)"""
        # Spectral flatness - vocals have lower flatness than instruments
        flatness = librosa.feature.spectral_flatness(y=y)[0]
        mean_flatness = np.mean(flatness)
        
        # Spectral bandwidth - wokal ma charakterystyczny zakres
        bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
        mean_bandwidth = np.mean(bandwidth)
        
        # Zero crossing rate
        zcr = librosa.feature.zero_crossing_rate(y)[0]
        mean_zcr = np.mean(zcr)
        
        # Heuristic: vocals have low flatness, medium bandwidth, high ZCR
        vocal_score = (
            (1 - min(1, mean_flatness * 5)) * 0.4 +
            (1 if 1500 < mean_bandwidth < 4000 else 0.5) * 0.3 +
            (min(1, mean_zcr * 10)) * 0.3
        )
        
        return vocal_score > 0.55
    
    def _cluster_segments(
        self, 
        segments: List[Dict], 
        features: Dict,
        boundaries: List[float]
    ) -> List[Dict]:
        """Clusters similar segments (detects repetitions)"""
        
        if len(segments) < 2:
            for s in segments:
                s['cluster'] = 0
            return segments
        
        # Zbuduj feature matrix dla klasteryzacji
        feature_matrix = []
        for seg in segments:
            feat = np.concatenate([
                seg['chroma_mean'],  # 12
                seg['mfcc_mean'][:10],  # 10
                [seg['energy'], seg['spectral_centroid'] / 5000]  # 2
            ])
            feature_matrix.append(feat)
        
        feature_matrix = np.array(feature_matrix)
        
        # Normalize
        feature_matrix = (feature_matrix - feature_matrix.mean(axis=0)) / (feature_matrix.std(axis=0) + 1e-8)
        
        # Hierarchical clustering
        n_clusters = min(len(segments), max(2, len(segments) // 2))
        clustering = AgglomerativeClustering(
            n_clusters=n_clusters,
            metric='euclidean',
            linkage='ward'
        )
        
        labels = clustering.fit_predict(feature_matrix)
        
        # Przypisz klastry
        for i, seg in enumerate(segments):
            seg['cluster'] = int(labels[i])
        
        # Find repetitions (same cluster)
        cluster_first = {}
        for i, seg in enumerate(segments):
            cluster = seg['cluster']
            if cluster in cluster_first:
                seg['is_repetition_of'] = cluster_first[cluster]
            else:
                cluster_first[cluster] = i
        
        return segments
    
    def _label_sections(self, segments: List[Dict], duration: float) -> List[MusicSegment]:
        """
        Przypisuje etykiety sekcji na podstawie cech i pozycji w utworze.
        
        Heurystyki:
        - Pierwszy segment o niskiej energii = Intro
        - Ostatni segment o niskiej energii = Outro
        - Wysokoenergetyczne, powtarzające się = Chorus
        - Średnioenergetyczne z wokalem = Verse
        - Różniący się od reszty, wysoka energia = Bridge
        - Bez wokalu, wysoka energia = Instrumental/Solo
        """
        
        labeled = []
        
        # Oblicz statystyki globalne
        energies = [s['energy'] for s in segments]
        mean_energy = np.mean(energies)
        std_energy = np.std(energies)
        
        # Find most common cluster (probably Verse or Chorus)
        clusters = [s['cluster'] for s in segments]
        cluster_counts = {}
        for c in clusters:
            cluster_counts[c] = cluster_counts.get(c, 0) + 1
        most_common_cluster = max(cluster_counts, key=cluster_counts.get)
        
        # Average energy of most common cluster
        most_common_energy = np.mean([
            s['energy'] for s in segments if s['cluster'] == most_common_cluster
        ])
        
        for i, seg in enumerate(segments):
            position = (seg['start_time'] + seg['end_time']) / 2 / duration  # 0-1
            relative_energy = seg['energy'] - mean_energy
            is_repetition = seg.get('is_repetition_of') is not None
            
            # Default unknown
            section_type = SectionType.UNKNOWN
            confidence = 0.5
            
            # ===== INTRO =====
            if i == 0 and position < 0.15:
                if seg['energy'] < mean_energy:
                    section_type = SectionType.INTRO
                    confidence = 0.8
                elif not seg['has_vocals']:
                    section_type = SectionType.INTRO
                    confidence = 0.7
            
            # ===== OUTRO =====
            elif i == len(segments) - 1 and position > 0.85:
                if seg['energy'] < mean_energy:
                    section_type = SectionType.OUTRO
                    confidence = 0.8
                elif not seg['has_vocals']:
                    section_type = SectionType.OUTRO
                    confidence = 0.7
            
            # ===== CHORUS (high energy, repetitions) =====
            elif relative_energy > std_energy * 0.5 and is_repetition:
                section_type = SectionType.CHORUS
                confidence = 0.75
            
            # ===== HIGH ENERGY WITHOUT VOCALS (Instrumental/Drop) =====
            elif relative_energy > std_energy and not seg['has_vocals']:
                if seg['spectral_centroid'] > 3000:
                    section_type = SectionType.DROP
                    confidence = 0.65
                else:
                    section_type = SectionType.INSTRUMENTAL
                    confidence = 0.7
            
            # ===== VERSE (most common, with vocals) =====
            elif seg['cluster'] == most_common_cluster and seg['has_vocals']:
                section_type = SectionType.VERSE
                confidence = 0.7
            
            # ===== BRIDGE (different from rest) =====
            elif cluster_counts.get(seg['cluster'], 0) == 1:
                if 0.4 < position < 0.8:  # Typically in the middle
                    section_type = SectionType.BRIDGE
                    confidence = 0.6
            
            # ===== PRE-CHORUS (before high energy) =====
            elif i < len(segments) - 1:
                next_seg = segments[i + 1]
                if next_seg['energy'] > seg['energy'] + std_energy * 0.5:
                    section_type = SectionType.PRE_CHORUS
                    confidence = 0.55
            
            # ===== BUILDUP (increasing energy) =====
            if section_type == SectionType.UNKNOWN:
                if i < len(segments) - 1:
                    if segments[i + 1]['energy'] > seg['energy'] * 1.3:
                        section_type = SectionType.BUILDUP
                        confidence = 0.5
            
            # ===== DEFAULT: VERSE =====
            if section_type == SectionType.UNKNOWN:
                section_type = SectionType.VERSE if seg['has_vocals'] else SectionType.INSTRUMENTAL
                confidence = 0.4
            
            labeled.append(MusicSegment(
                start_time=seg['start_time'],
                end_time=seg['end_time'],
                section_type=section_type.value,
                confidence=confidence,
                tempo=seg['tempo'],
                key=seg['key'],
                energy=seg['energy'],
                loudness=seg['loudness'],
                spectral_centroid=seg['spectral_centroid'],
                has_vocals=seg['has_vocals'],
                is_repetition_of=seg.get('is_repetition_of'),
            ))
        
        return labeled
    
    def _generate_prompts(self, segments: List[MusicSegment]) -> List[MusicSegment]:
        """Generates automatic prompts for each segment"""
        
        for seg in segments:
            parts = []
            
            # Section type
            section_descriptors = {
                'intro': 'atmospheric intro',
                'verse': 'melodic verse',
                'pre_chorus': 'building pre-chorus',
                'chorus': 'powerful chorus',
                'post_chorus': 'energetic post-chorus',
                'bridge': 'contrasting bridge',
                'instrumental': 'instrumental section',
                'solo': 'solo section',
                'breakdown': 'sparse breakdown',
                'buildup': 'building tension',
                'drop': 'heavy drop',
                'outro': 'fading outro',
            }
            parts.append(section_descriptors.get(seg.section_type, 'music'))
            
            # Energy description
            if seg.energy > 0.7:
                parts.append('high energy')
            elif seg.energy < 0.3:
                parts.append('calm')
            
            # Key
            if 'minor' in seg.key.lower():
                parts.append('melancholic')
            else:
                parts.append('uplifting')
            
            # Tempo description
            if seg.tempo > 140:
                parts.append('fast tempo')
            elif seg.tempo < 80:
                parts.append('slow tempo')
            
            # Vocals
            if seg.has_vocals:
                parts.append('with vocals')
            else:
                parts.append('instrumental')
            
            # Brightness
            if seg.spectral_centroid > 3000:
                parts.append('bright')
            elif seg.spectral_centroid < 1500:
                parts.append('warm')
            
            seg.auto_prompt = ', '.join(parts)
        
        return segments
    
    def _get_dominant_key(self, keys: List[str]) -> str:
        """Finds the dominant key"""
        key_counts = {}
        for key in keys:
            key_counts[key] = key_counts.get(key, 0) + 1
        return max(key_counts, key=key_counts.get) if key_counts else "C"


class BatchAnnotator:
    """Annotates multiple files in parallel"""
    
    def __init__(self, annotator: SegmentAnnotator, n_jobs: int = 4):
        self.annotator = annotator
        self.n_jobs = n_jobs
    
    def annotate_directory(
        self, 
        input_dir: str, 
        output_file: str,
        extensions: List[str] = ['.mp3', '.wav', '.flac'],
        max_files: Optional[int] = None,
    ) -> List[AnnotatedTrack]:
        """Anotuje wszystkie pliki w katalogu"""
        
        input_dir = Path(input_dir)
        
        # Find all audio files
        audio_files = []
        for ext in extensions:
            audio_files.extend(input_dir.rglob(f'*{ext}'))
        
        if max_files:
            audio_files = audio_files[:max_files]
        
        print(f"Found {len(audio_files)} audio files")
        
        # Annotate each file
        results = []
        errors = []
        
        from tqdm import tqdm
        
        for audio_file in tqdm(audio_files, desc="Annotating"):
            try:
                # Generate track_id from path
                track_id = audio_file.stem
                
                annotation = self.annotator.annotate(str(audio_file), track_id)
                results.append(annotation)
                
            except Exception as e:
                print(f"  ❌ Error processing {audio_file.name}: {e}")
                errors.append({'file': str(audio_file), 'error': str(e)})
        
        # Save results
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert to serializable format
        output_data = {
            'version': '2.0',
            'total_tracks': len(results),
            'errors': errors,
            'tracks': [asdict(r) for r in results]
        }
        
        # Convert numpy arrays to lists
        def convert_numpy(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_numpy(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(i) for i in obj]
            return obj
        
        output_data = convert_numpy(output_data)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        print(f"\n✅ Saved {len(results)} annotations to {output_path}")
        print(f"   Errors: {len(errors)}")
        
        return results


def main():
    parser = argparse.ArgumentParser(
        description='🎵 Segment Annotator - Automatyczna anotacja sekcji muzycznych'
    )
    
    parser.add_argument(
        '--input', '-i',
        type=str,
        help='Ścieżka do pojedynczego pliku audio'
    )
    
    parser.add_argument(
        '--input_dir', '-d',
        type=str,
        help='Ścieżka do katalogu z plikami audio'
    )
    
    parser.add_argument(
        '--output', '-o',
        type=str,
        default='./data_v2/segments.json',
        help='Ścieżka do pliku wyjściowego JSON'
    )
    
    parser.add_argument(
        '--max_files',
        type=int,
        default=None,
        help='Maksymalna liczba plików do przetworzenia'
    )
    
    parser.add_argument(
        '--min_segment',
        type=float,
        default=4.0,
        help='Minimalna długość segmentu (sekundy)'
    )
    
    parser.add_argument(
        '--sample_rate',
        type=int,
        default=22050,
        help='Sample rate'
    )
    
    args = parser.parse_args()
    
    if not args.input and not args.input_dir:
        parser.error("Wymagane jest --input lub --input_dir")
    
    # Create annotator
    annotator = SegmentAnnotator(
        sample_rate=args.sample_rate,
        min_segment_duration=args.min_segment,
    )
    
    if args.input:
        # Single file
        result = annotator.annotate(args.input)
        
        # Print results
        print("\n" + "="*60)
        print(f"📊 Annotation Results: {result.track_id}")
        print("="*60)
        print(f"Duration: {result.duration:.1f}s")
        print(f"Global tempo: {result.global_tempo:.0f} BPM")
        print(f"Global key: {result.global_key}")
        print(f"\nSegments ({len(result.segments)}):")
        
        for i, seg in enumerate(result.segments):
            print(f"\n  [{i+1}] {seg.section_type.upper()}")
            print(f"      Time: {seg.start_time:.1f}s - {seg.end_time:.1f}s ({seg.end_time - seg.start_time:.1f}s)")
            print(f"      Tempo: {seg.tempo:.0f} BPM | Key: {seg.key}")
            print(f"      Energy: {seg.energy:.2f} | Vocals: {'Yes' if seg.has_vocals else 'No'}")
            print(f"      Confidence: {seg.confidence:.1%}")
            print(f"      Prompt: {seg.auto_prompt}")
        
        # Save to file
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(asdict(result), f, indent=2, ensure_ascii=False, default=str)
        
        print(f"\n✅ Saved to {output_path}")
        
    else:
        # Directory
        batch = BatchAnnotator(annotator)
        batch.annotate_directory(
            args.input_dir,
            args.output,
            max_files=args.max_files,
        )


if __name__ == "__main__":
    main()
