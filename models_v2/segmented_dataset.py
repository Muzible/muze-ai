"""
🎵 Segmented Music Dataset - Dataset ze świadomością sekcji utworu

Główne różnice od oryginalnego MusicDataset:
1. Zamiast random crop - zwraca konkretne segmenty
2. Każdy segment ma informacje o: section_type, position, tempo, energy
3. Obsługuje sekwencje segmentów dla treningu z kontekstem
4. Obsługuje CLAP embeddings per sekcja

Użycie:
    dataset = SegmentedMusicDataset(
        annotations_json='./data_v2/segments.json',
        audio_dir='./music/fma_small',
        segment_duration=10.0,
    )
    
    sample = dataset[0]
    # sample['audio'] - waveform segmentu
    # sample['section_type'] - typ sekcji (one-hot lub embedding)
    # sample['position'] - pozycja w utworze (0-1)
    # sample['context_audio'] - opcjonalnie poprzedni segment
"""

import os
import json
import torch
import torchaudio
import numpy as np
from torch.utils.data import Dataset, DataLoader
from typing import Optional, List, Dict, Tuple, Any
from pathlib import Path
import random


class SectionEmbedding:
    """Creates embeddings for section types"""
    
    SECTION_TYPES = [
        'intro', 'verse', 'pre_chorus', 'chorus', 'post_chorus',
        'bridge', 'instrumental', 'solo', 'breakdown', 'buildup',
        'drop', 'outro', 'unknown'
    ]
    
    def __init__(self, embed_dim: int = 64):
        self.embed_dim = embed_dim
        self.num_sections = len(self.SECTION_TYPES)
        self.section_to_idx = {s: i for i, s in enumerate(self.SECTION_TYPES)}
        
        # Pre-computed embeddings (can also use nn.Embedding)
        self._embeddings = self._create_embeddings()
    
    def _create_embeddings(self) -> Dict[str, np.ndarray]:
        """Creates semantic embeddings for sections"""
        # Based on section characteristics:
        # [energy, has_vocals, is_repetitive, position_bias, brightness, complexity]
        
        base_features = {
            'intro':        [0.3, 0.0, 0.0, 0.0, 0.5, 0.4],
            'verse':        [0.5, 1.0, 0.8, 0.3, 0.5, 0.5],
            'pre_chorus':   [0.6, 1.0, 0.5, 0.4, 0.6, 0.6],
            'chorus':       [0.9, 1.0, 1.0, 0.5, 0.8, 0.7],
            'post_chorus':  [0.7, 0.5, 0.7, 0.6, 0.7, 0.5],
            'bridge':       [0.4, 0.8, 0.2, 0.7, 0.5, 0.8],
            'instrumental': [0.6, 0.0, 0.5, 0.5, 0.6, 0.7],
            'solo':         [0.7, 0.0, 0.3, 0.6, 0.8, 0.9],
            'breakdown':    [0.2, 0.3, 0.4, 0.6, 0.3, 0.3],
            'buildup':      [0.5, 0.2, 0.6, 0.5, 0.6, 0.5],
            'drop':         [1.0, 0.2, 0.8, 0.5, 0.9, 0.6],
            'outro':        [0.2, 0.3, 0.5, 1.0, 0.4, 0.3],
            'unknown':      [0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
        }
        
        embeddings = {}
        for section, features in base_features.items():
            # Expand to embed_dim using random projection (deterministic seed)
            np.random.seed(hash(section) % 2**32)
            projection = np.random.randn(len(features), self.embed_dim) * 0.1
            embedding = np.array(features) @ projection
            embedding = embedding / (np.linalg.norm(embedding) + 1e-8)
            embeddings[section] = embedding.astype(np.float32)
        
        return embeddings
    
    def get_embedding(self, section_type: str) -> np.ndarray:
        """Zwraca embedding dla typu sekcji"""
        section_type = section_type.lower()
        return self._embeddings.get(section_type, self._embeddings['unknown'])
    
    def get_index(self, section_type: str) -> int:
        """Zwraca indeks dla typu sekcji"""
        return self.section_to_idx.get(section_type.lower(), self.section_to_idx['unknown'])
    
    def get_one_hot(self, section_type: str) -> np.ndarray:
        """Zwraca one-hot encoding"""
        idx = self.get_index(section_type)
        one_hot = np.zeros(self.num_sections, dtype=np.float32)
        one_hot[idx] = 1.0
        return one_hot


class SegmentedMusicDataset(Dataset):
    """
    Dataset ze świadomością sekcji utworu.
    
    Każdy sample to segment utworu z pełną informacją o kontekście:
    - Typ sekcji (verse, chorus, etc.)
    - Pozycja w utworze (0-1)
    - Poprzedni segment (dla kontekstu)
    - Cechy muzyczne (tempo, energy, key)
    """
    
    def __init__(
        self,
        annotations_json: str,
        audio_dir: str,
        sample_rate: int = 22050,
        segment_duration: float = 10.0,
        include_context: bool = True,      # Whether to include previous segment
        context_overlap: float = 0.5,      # Overlap with previous segment (0-1)
        max_tracks: Optional[int] = None,
        filter_short_segments: bool = True,
        min_segment_duration: float = 4.0,
        section_embedding_dim: int = 64,
    ):
        """
        Args:
            annotations_json: Ścieżka do pliku z anotacjami segmentów
            audio_dir: Folder z plikami audio
            sample_rate: Docelowy sample rate
            segment_duration: Długość segmentu do wczytania
            include_context: Czy dołączać poprzedni segment jako kontekst
            context_overlap: Ile poprzedniego segmentu nakłada się na obecny
            max_tracks: Limit utworów
            filter_short_segments: Czy pomijać za krótkie segmenty
            min_segment_duration: Minimalna długość segmentu
            section_embedding_dim: Wymiar embeddingu sekcji
        """
        self.audio_dir = Path(audio_dir)
        self.sr = sample_rate
        self.segment_duration = segment_duration
        self.num_samples = int(sample_rate * segment_duration)
        self.include_context = include_context
        self.context_overlap = context_overlap
        self.min_segment_duration = min_segment_duration
        
        # Section embedding
        self.section_embedder = SectionEmbedding(section_embedding_dim)
        
        # Wczytaj anotacje
        print(f"Loading annotations from {annotations_json}...")
        with open(annotations_json, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Handle both formats (list or dict with 'tracks')
        if isinstance(data, dict):
            tracks = data.get('tracks', [])
        else:
            tracks = data
        
        print(f"  Loaded {len(tracks)} tracks")
        
        if max_tracks:
            tracks = tracks[:max_tracks]
        
        # Flatten: each segment is a separate sample
        self.samples = []
        missing_audio = 0
        short_segments = 0
        
        for track in tracks:
            track_id = track.get('track_id', '')
            file_path = track.get('file_path', '')
            
            # Find audio file
            audio_path = self._find_audio_file(track_id, file_path)
            if audio_path is None:
                missing_audio += 1
                continue
            
            segments = track.get('segments', [])
            
            for i, segment in enumerate(segments):
                seg_duration = segment.get('end_time', 0) - segment.get('start_time', 0)
                
                if filter_short_segments and seg_duration < min_segment_duration:
                    short_segments += 1
                    continue
                
                # Previous segment (for context)
                prev_segment = segments[i-1] if i > 0 else None
                
                # Pobierz vocals data
                vocals_data = track.get('vocals', {})
                features_data = track.get('features', {})
                
                self.samples.append({
                    'track_id': track_id,
                    'audio_path': str(audio_path),
                    'segment_idx': i,
                    'segment': segment,
                    'prev_segment': prev_segment,
                    'total_duration': track.get('duration', 0),
                    'global_tempo': features_data.get('tempo', 120),
                    'global_key': features_data.get('dominant_key', 'C'),
                    
                    # Nowe pola z build_dataset_v2
                    'global_prompt': track.get('global_prompt', ''),
                    'artist': track.get('artist', ''),
                    'genres': track.get('genres', []),
                    
                    # 🎤 Voice embeddings dla voice cloning
                    'voice_embedding': vocals_data.get('voice_embedding', []),
                    'voice_embedding_separated': vocals_data.get('voice_embedding_separated', []),
                    
                    # 📝 Lyrics
                    'lyrics_full': vocals_data.get('lyrics_full', ''),
                    'lyrics_language': vocals_data.get('lyrics_language', ''),
                    
                    # 🔊 CLAP embeddings (pre-computed)
                    'clap_audio_embedding': vocals_data.get('clap_audio_embedding', []),
                    'clap_text_embedding': vocals_data.get('clap_text_embedding', []),
                    
                    # 🗣️ Phonemes dla voice synthesis
                    'phonemes_ipa': vocals_data.get('phonemes_ipa', ''),
                    'phonemes_words': vocals_data.get('phonemes_words', []),
                    
                    # 🎵 Beat and chord data
                    'beat_positions': features_data.get('beat_positions', []),
                    'downbeat_positions': features_data.get('downbeat_positions', []),
                    'chord_sequence': features_data.get('chord_sequence', []),
                    'chords': features_data.get('chords', {}),
                    'time_signature': features_data.get('time_signature', '4/4'),
                    
                    # 💭 Sentiment
                    'sentiment_label': vocals_data.get('sentiment_label', 'neutral'),
                    'sentiment_score': vocals_data.get('sentiment_score', 0.5),
                })
        
        print(f"  Total samples: {len(self.samples)}")
        print(f"  Missing audio: {missing_audio}")
        print(f"  Short segments filtered: {short_segments}")
    
    def _find_audio_file(self, track_id: str, file_path: str) -> Optional[Path]:
        """Finds audio file"""
        # First try file_path (may be relative or absolute)
        if file_path:
            p = Path(file_path)
            if p.exists():
                return p
            # Try relative to working directory
            if not p.is_absolute():
                # file_path may be relative like "music/own/artist/track.mp3"
                cwd_path = Path.cwd() / p
                if cwd_path.exists():
                    return cwd_path
            # Try relative to audio_dir (filename only)
            p = self.audio_dir / p.name
            if p.exists():
                return p
        
        # FMA format: XXX/XXXXXX.mp3
        if track_id.isdigit():
            tid_str = f'{int(track_id):06d}'
            p = self.audio_dir / tid_str[:3] / f'{tid_str}.mp3'
            if p.exists():
                return p
        
        # Try different extensions
        for ext in ['.mp3', '.wav', '.flac']:
            p = self.audio_dir / f'{track_id}{ext}'
            if p.exists():
                return p
            # Z subfolderem
            for subfolder in self.audio_dir.iterdir():
                if subfolder.is_dir():
                    p = subfolder / f'{track_id}{ext}'
                    if p.exists():
                        return p
        
        return None
    
    def _load_audio_segment(
        self, 
        audio_path: str, 
        start_time: float, 
        duration: float
    ) -> torch.Tensor:
        """Loads audio segment"""
        try:
            # Calculate offsets
            info = torchaudio.info(audio_path)
            sr_orig = info.sample_rate
            
            # Wczytaj fragment
            frame_offset = int(start_time * sr_orig)
            num_frames = int(duration * sr_orig)
            
            waveform, sr = torchaudio.load(
                audio_path,
                frame_offset=frame_offset,
                num_frames=num_frames,
            )
        except Exception as e:
            print(f"Error loading {audio_path}: {e}")
            return torch.zeros(self.num_samples)
        
        # Convert to mono
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        
        # Resample
        if sr != self.sr:
            resampler = torchaudio.transforms.Resample(sr, self.sr)
            waveform = resampler(waveform)
        
        waveform = waveform.squeeze(0)
        
        # Pad/crop to target length
        target_samples = int(self.segment_duration * self.sr)
        
        if waveform.shape[0] > target_samples:
            waveform = waveform[:target_samples]
        elif waveform.shape[0] < target_samples:
            pad_size = target_samples - waveform.shape[0]
            waveform = torch.nn.functional.pad(waveform, (0, pad_size))
        
        return waveform
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self.samples[idx]
        segment = sample['segment']
        
        # Wczytaj audio segmentu
        start_time = segment.get('start_time', 0)
        end_time = segment.get('end_time', start_time + self.segment_duration)
        duration = min(end_time - start_time, self.segment_duration)
        
        audio = self._load_audio_segment(
            sample['audio_path'],
            start_time,
            duration,
        )
        
        # Section embedding
        section_type = segment.get('section_type', 'unknown')
        section_embedding = self.section_embedder.get_embedding(section_type)
        section_index = self.section_embedder.get_index(section_type)
        section_one_hot = self.section_embedder.get_one_hot(section_type)
        
        # Pozycja w utworze (0-1)
        total_duration = sample['total_duration']
        if total_duration > 0:
            position = (start_time + end_time) / 2 / total_duration
        else:
            position = 0.5
        
        # Cechy muzyczne
        tempo = segment.get('tempo', sample['global_tempo'])
        energy = segment.get('energy', 0.5)
        loudness = segment.get('loudness', -20.0)
        has_vocals = segment.get('has_vocals', False)
        key = segment.get('key', sample['global_key'])
        
        # Prompt - segment lub global
        prompt = segment.get('auto_prompt', '') or sample.get('global_prompt', '')
        
        # Voice embeddings (jako tensory)
        voice_emb = sample.get('voice_embedding', [])
        voice_emb_sep = sample.get('voice_embedding_separated', [])
        
        # CLAP embeddings (pre-computed)
        clap_audio = sample.get('clap_audio_embedding', [])
        clap_text = sample.get('clap_text_embedding', [])
        
        # Lyrics dla tego segmentu
        segment_lyrics = segment.get('lyrics_text', '')
        segment_sentiment = segment.get('lyrics_sentiment', 'neutral')
        segment_sentiment_score = segment.get('sentiment_score', 0.5)
        
        # Beat positions within this segment
        all_beats = sample.get('beat_positions', [])
        segment_beats = [b - start_time for b in all_beats 
                        if start_time <= b < end_time]
        
        # Chord in this segment (find current chord)
        chord_seq = sample.get('chord_sequence', [])
        current_chord = 'C'  # default
        for chord_info in chord_seq:
            if isinstance(chord_info, dict):
                chord_time = chord_info.get('time', 0)
                if chord_time <= start_time:
                    current_chord = chord_info.get('chord', 'C')
        
        result = {
            # Audio
            'audio': audio,
            
            # Segment info
            'section_type': section_type,
            'section_index': section_index,
            'section_embedding': torch.tensor(section_embedding, dtype=torch.float32),
            'section_one_hot': torch.tensor(section_one_hot, dtype=torch.float32),
            
            # Position in song
            'position': position,
            'start_time': start_time,
            'end_time': end_time,
            
            # Musical features
            'tempo': tempo,
            'energy': energy,
            'loudness': loudness,
            'has_vocals': has_vocals,
            'key': key,
            'current_chord': current_chord,
            'time_signature': sample.get('time_signature', '4/4'),
            
            # Beat sync data
            'beat_positions': segment_beats,  # Beats in this segment (relative)
            'num_beats': len(segment_beats),
            
            # Prompts
            'prompt': prompt,  # Segment prompt or global
            'global_prompt': sample.get('global_prompt', ''),
            
            # Track info
            'track_id': sample['track_id'],
            'segment_idx': sample['segment_idx'],
            'artist': sample.get('artist', ''),
            'genres': sample.get('genres', []),
            
            # 🎙️ Voice embeddings for voice cloning
            'voice_embedding': torch.tensor(voice_emb, dtype=torch.float32) if voice_emb else torch.zeros(256),
            'voice_embedding_separated': torch.tensor(voice_emb_sep, dtype=torch.float32) if voice_emb_sep else torch.zeros(192),
            
            # 🔊 CLAP embeddings (pre-computed) - 512 dim
            'clap_audio_embedding': torch.tensor(clap_audio, dtype=torch.float32) if clap_audio else torch.zeros(512),
            'clap_text_embedding': torch.tensor(clap_text, dtype=torch.float32) if clap_text else torch.zeros(512),
            
            # 📝 Lyrics dla tego segmentu
            'lyrics_text': segment_lyrics,
            'lyrics_full': sample.get('lyrics_full', ''),
            'lyrics_language': sample.get('lyrics_language', ''),
            'lyrics_sentiment': segment_sentiment,
            'sentiment_score': segment_sentiment_score,
            
            # 🗣️ Phonemes dla voice synthesis (track-level)
            'phonemes_ipa': sample.get('phonemes_ipa', ''),
            'phonemes_words': sample.get('phonemes_words', []),
            
            # 📝 Phoneme timestamps per segment (v3) - dla lip-sync/alignment
            'phoneme_timestamps': segment.get('phoneme_timestamps', []),
            
            # 🎵 F0/Pitch contour (v3) - dla singing synthesis (from SEGMENT, not sample!)
            'f0': torch.tensor(segment.get('f0', []), dtype=torch.float32) if segment.get('f0') else None,
            'f0_coarse': torch.tensor(segment.get('f0_coarse', []), dtype=torch.long) if segment.get('f0_coarse') else None,
            'f0_voiced_mask': torch.tensor(segment.get('f0_voiced_mask', []), dtype=torch.bool) if segment.get('f0_voiced_mask') else None,
            'f0_statistics': segment.get('f0_statistics', {}),
            
            # 🎙️ Vibrato analysis (v3) - for vocal expression
            'vibrato_rate': segment.get('vibrato_rate'),      # Hz (typically 4-8 Hz)
            'vibrato_depth': segment.get('vibrato_depth'),    # semitones
            'vibrato_extent': segment.get('vibrato_extent'),  # % of voiced frames with vibrato
            
            # 💨 Breath positions (v3) - for natural breaths
            'breath_positions': segment.get('breath_positions', []),
        }
        
        # Context (previous segment)
        if self.include_context and sample['prev_segment'] is not None:
            prev_seg = sample['prev_segment']
            prev_start = prev_seg.get('start_time', 0)
            prev_end = prev_seg.get('end_time', prev_start + self.segment_duration)
            
            # Load end of previous segment
            overlap_duration = self.segment_duration * self.context_overlap
            context_start = max(prev_start, prev_end - overlap_duration)
            
            context_audio = self._load_audio_segment(
                sample['audio_path'],
                context_start,
                overlap_duration,
            )
            
            # Ensure fixed size for context audio
            context_samples = int(self.sr * overlap_duration)
            if context_audio.shape[0] > context_samples:
                context_audio = context_audio[:context_samples]
            elif context_audio.shape[0] < context_samples:
                pad_size = context_samples - context_audio.shape[0]
                context_audio = torch.nn.functional.pad(context_audio, (0, pad_size))
            
            result['context_audio'] = context_audio
            result['context_section_type'] = prev_seg.get('section_type', 'unknown')
            result['context_section_embedding'] = torch.tensor(
                self.section_embedder.get_embedding(prev_seg.get('section_type', 'unknown')),
                dtype=torch.float32
            )
        else:
            # No context - zeros (same fixed size)
            context_samples = int(self.sr * self.segment_duration * self.context_overlap)
            result['context_audio'] = torch.zeros(context_samples)
            result['context_section_type'] = 'none'
            result['context_section_embedding'] = torch.zeros(self.section_embedder.embed_dim)
        
        return result


class CompositionDataset(Dataset):
    """
    Dataset do treningu Composition Plannera.
    
    v2 Updates:
    - Dodane prompty tekstowe (global_prompt, genre, mood)
    - Używane do treningu z prawdziwymi embeddingami
    
    Dla każdego utworu zwraca:
    - Globalny opis (prompt)
    - Sekwencję sekcji z ich cechami
    - Target duration
    - Genre i mood indices
    """
    
    # v2: Predefiniowane listy genres i moods
    GENRES = [
        'pop', 'rock', 'electronic', 'hip-hop', 'r&b', 'jazz', 'classical',
        'folk', 'country', 'metal', 'punk', 'indie', 'ambient', 'soul',
        'blues', 'reggae', 'latin', 'world', 'disco', 'funk', 'house',
        'techno', 'trance', 'dubstep', 'drum and bass', 'lo-fi', 'chillout',
        'acoustic', 'orchestral', 'soundtrack', 'experimental', 'alternative',
        'grunge', 'shoegaze', 'post-rock', 'progressive', 'psychedelic',
        'synth-pop', 'new wave', 'emo', 'hardcore', 'trap', 'k-pop',
        'j-pop', 'edm', 'ballad', 'other', 'unknown'
    ]
    
    MOODS = [
        'happy', 'sad', 'energetic', 'calm', 'angry', 'romantic', 'melancholic',
        'uplifting', 'dark', 'peaceful', 'intense', 'dreamy', 'nostalgic',
        'powerful', 'gentle', 'aggressive', 'hopeful', 'mysterious', 'playful',
        'serious', 'emotional', 'epic', 'chill', 'groovy', 'atmospheric',
        'euphoric', 'somber', 'triumphant', 'other', 'unknown'
    ]
    
    def __init__(
        self,
        annotations_json: str,
        max_sections: int = 20,
        max_tracks: Optional[int] = None,
    ):
        print(f"Loading composition data from {annotations_json}...")
        
        with open(annotations_json, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if isinstance(data, dict):
            tracks = data.get('tracks', [])
        else:
            tracks = data
        
        if max_tracks:
            tracks = tracks[:max_tracks]
        
        self.max_sections = max_sections
        self.samples = []
        
        # Section type mapping
        self.section_to_idx = {
            'pad': 0, 'bos': 1, 'eos': 2,
            'intro': 3, 'verse': 4, 'pre_chorus': 5, 'chorus': 6,
            'post_chorus': 7, 'bridge': 8, 'instrumental': 9, 'solo': 10,
            'breakdown': 11, 'buildup': 12, 'drop': 13, 'outro': 14, 'unknown': 4,
        }
        
        # Key mapping
        keys = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        self.key_to_idx = {}
        for i, k in enumerate(keys):
            self.key_to_idx[k] = i
            self.key_to_idx[k + ' minor'] = i + 12
            self.key_to_idx[k + 'm'] = i + 12
        
        # v2: Genre i mood mapping
        self.genre_to_idx = {g: i for i, g in enumerate(self.GENRES)}
        self.mood_to_idx = {m: i for i, m in enumerate(self.MOODS)}
        
        for track in tracks:
            segments = track.get('segments', [])
            if len(segments) < 2:  # Need at least 2 segments
                continue
            
            # v2: Extract prompts and metadata
            global_prompt = track.get('prompt', track.get('global_prompt', ''))
            if not global_prompt:
                # Fallback: generate simple prompt from available data
                genre = track.get('genre', track.get('top_genre', 'unknown'))
                mood = track.get('mood', 'unknown')
                tempo = track.get('global_tempo', track.get('features', {}).get('tempo', 120))
                key = track.get('global_key', track.get('features', {}).get('dominant_key', 'C'))
                global_prompt = f"{genre} track, {mood} mood, {int(tempo)} BPM, key of {key}"
            
            # v2: Extract genre and mood
            genre = track.get('genre', track.get('top_genre', 'unknown'))
            if isinstance(genre, list):
                genre = genre[0] if genre else 'unknown'
            genre = genre.lower() if genre else 'unknown'
            
            mood = track.get('mood', 'unknown')
            if isinstance(mood, list):
                mood = mood[0] if mood else 'unknown'
            mood = mood.lower() if mood else 'unknown'
            
            self.samples.append({
                'duration': track.get('duration', 0),
                'global_tempo': track.get('global_tempo', track.get('features', {}).get('tempo', 120)),
                'global_key': track.get('global_key', track.get('features', {}).get('dominant_key', 'C')),
                'segments': segments[:max_sections],
                # v2: Nowe pola
                'prompt': global_prompt,
                'genre': genre,
                'mood': mood,
            })
        
        print(f"  Loaded {len(self.samples)} tracks for composition training")
        print(f"  Genres: {len(self.GENRES)}, Moods: {len(self.MOODS)}")
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]
        segments = sample['segments']
        
        # Przygotuj sekwencje
        n_seg = len(segments)
        
        # Section types: [BOS, seg1, seg2, ..., EOS, PAD, PAD, ...]
        section_tokens = [1]  # BOS
        for seg in segments:
            section_type = seg.get('section_type', 'verse').lower()
            token = self.section_to_idx.get(section_type, 4)  # default: verse
            section_tokens.append(token)
        section_tokens.append(2)  # EOS
        
        # Pad to max_sections + 2 (BOS + segments + EOS)
        while len(section_tokens) < self.max_sections + 2:
            section_tokens.append(0)  # PAD
        
        section_tokens = section_tokens[:self.max_sections + 2]
        
        # Attributes: [tempo_norm, energy, duration_norm, has_vocals]
        attrs = [[0, 0, 0, 0]]  # BOS
        for seg in segments:
            tempo = seg.get('tempo', sample['global_tempo'])
            tempo_norm = (tempo - 60) / 140  # Normalize to ~0-1
            
            energy = seg.get('energy', 0.5)
            
            start = seg.get('start_time', 0)
            end = seg.get('end_time', start + 20)
            duration = end - start
            duration_norm = (duration - 8) / 52  # Normalize to ~0-1
            
            has_vocals = 1.0 if seg.get('has_vocals', False) else 0.0
            
            attrs.append([tempo_norm, energy, duration_norm, has_vocals])
        
        attrs.append([0, 0, 0, 0])  # EOS
        
        while len(attrs) < self.max_sections + 2:
            attrs.append([0, 0, 0, 0])  # PAD
        
        attrs = attrs[:self.max_sections + 2]
        
        # Keys
        keys = [0]  # BOS
        for seg in segments:
            key = seg.get('key', sample['global_key'])
            key_idx = self.key_to_idx.get(key, 0)
            keys.append(key_idx)
        keys.append(0)  # EOS
        
        while len(keys) < self.max_sections + 2:
            keys.append(0)
        
        keys = keys[:self.max_sections + 2]
        
        # Target duration (normalized)
        duration = sample['duration']
        duration_norm = duration / 300.0  # Normalize assuming max 5 min
        
        return {
            'sections': torch.tensor(section_tokens, dtype=torch.long),
            'attrs': torch.tensor(attrs, dtype=torch.float32),
            'keys': torch.tensor(keys, dtype=torch.long),
            'vocals': torch.tensor([a[3] for a in attrs], dtype=torch.float32),
            'duration': torch.tensor([duration_norm], dtype=torch.float32),
            'global_tempo': sample['global_tempo'],
            'global_key': sample['global_key'],
            # v2: Nowe pola dla Composition Planner
            'prompt': sample.get('prompt', ''),
            'genre': sample.get('genre', 'unknown'),
            'mood': sample.get('mood', 'unknown'),
            'genre_idx': self.genre_to_idx.get(sample.get('genre', 'unknown').lower(), len(self.GENRES) - 1),
            'mood_idx': self.mood_to_idx.get(sample.get('mood', 'unknown').lower(), len(self.MOODS) - 1),
        }


def collate_segmented(batch: List[Dict]) -> Dict[str, Any]:
    """Collate function dla SegmentedMusicDataset
    
    Obsługuje wszystkie dane z datasetu:
    - Audio i section info
    - CLAP embeddings (pre-computed)
    - Voice embeddings  
    - Lyrics per segment
    - Phonemes
    - Beat positions
    """
    
    result = {
        # === AUDIO ===
        'audio': torch.stack([b['audio'] for b in batch]),
        
        # === SECTION INFO ===
        'section_index': torch.tensor([b['section_index'] for b in batch], dtype=torch.long),
        'section_embedding': torch.stack([b['section_embedding'] for b in batch]),
        'section_one_hot': torch.stack([b['section_one_hot'] for b in batch]),
        'section_type': [b['section_type'] for b in batch],
        
        # === POSITION & TIME ===
        'position': torch.tensor([b['position'] for b in batch], dtype=torch.float32),
        'start_time': torch.tensor([b['start_time'] for b in batch], dtype=torch.float32),
        'end_time': torch.tensor([b['end_time'] for b in batch], dtype=torch.float32),
        
        # === MUSICAL FEATURES ===
        'tempo': torch.tensor([b['tempo'] for b in batch], dtype=torch.float32),
        'energy': torch.tensor([b['energy'] for b in batch], dtype=torch.float32),
        'loudness': torch.tensor([b['loudness'] for b in batch], dtype=torch.float32),  # v3: Loudness in dB
        'has_vocals': torch.tensor([b['has_vocals'] for b in batch], dtype=torch.bool),
        'key': [b['key'] for b in batch],
        'current_chord': [b['current_chord'] for b in batch],
        'time_signature': [b['time_signature'] for b in batch],
        'num_beats': torch.tensor([b['num_beats'] for b in batch], dtype=torch.long),
        
        # === PROMPTS ===
        'prompt': [b['prompt'] for b in batch],
        'global_prompt': [b['global_prompt'] for b in batch],
        
        # === TRACK INFO ===
        'track_id': [b['track_id'] for b in batch],
        'segment_idx': torch.tensor([b['segment_idx'] for b in batch], dtype=torch.long),
        'artist': [b['artist'] for b in batch],
        'genres': [b['genres'] for b in batch],  # v3: Genre list per sample
        
        # === 🎤 VOICE EMBEDDINGS ===
        'voice_embedding': torch.stack([b['voice_embedding'] for b in batch]),
        'voice_embedding_separated': torch.stack([b['voice_embedding_separated'] for b in batch]),
        
        # === 🔊 CLAP EMBEDDINGS (pre-computed) ===
        'clap_audio_embedding': torch.stack([b['clap_audio_embedding'] for b in batch]),
        'clap_text_embedding': torch.stack([b['clap_text_embedding'] for b in batch]),
        
        # === 📝 LYRICS PER SEGMENT ===
        'lyrics_text': [b['lyrics_text'] for b in batch],
        'lyrics_language': [b['lyrics_language'] for b in batch],
        'lyrics_sentiment': [b['lyrics_sentiment'] for b in batch],
        'sentiment_score': torch.tensor([b['sentiment_score'] for b in batch], dtype=torch.float32),
        
        # === 🗣️ PHONEMES ===
        'phonemes_ipa': [b['phonemes_ipa'] for b in batch],
        # phonemes_words is a list of dicts - keep as list
        'phonemes_words': [b['phonemes_words'] for b in batch],
        
        # === 📝 PHONEME TIMESTAMPS PER SEGMENT (v3) ===
        # List of dicts with timestamps - variable length
        'phoneme_timestamps': [b.get('phoneme_timestamps', []) for b in batch],
        
        # === 🎵 BEAT DATA ===
        # beat_positions has different length per sample - keep as list
        'beat_positions': [b['beat_positions'] for b in batch],
        
        # === 💨 BREATH POSITIONS (v3) ===
        # Variable length - list of lists
        'breath_positions': [b.get('breath_positions', []) for b in batch],
    }
    
    # === 🎙️ VIBRATO ANALYSIS (v3) ===
    # Scalar values per segment - may be None
    vibrato_rates = [b.get('vibrato_rate') for b in batch]
    vibrato_depths = [b.get('vibrato_depth') for b in batch]
    vibrato_extents = [b.get('vibrato_extent') for b in batch]
    
    # Convert to tensors with NaN for None values
    result['vibrato_rate'] = torch.tensor(
        [v if v is not None else float('nan') for v in vibrato_rates], 
        dtype=torch.float32
    )
    result['vibrato_depth'] = torch.tensor(
        [v if v is not None else float('nan') for v in vibrato_depths],
        dtype=torch.float32
    )
    result['vibrato_extent'] = torch.tensor(
        [v if v is not None else float('nan') for v in vibrato_extents],
        dtype=torch.float32
    )
    
    # === 🎵 F0/PITCH CONTOUR (v3) ===
    # F0 has variable length - need padding or list
    f0_list = [b.get('f0') for b in batch]
    if any(f0 is not None for f0 in f0_list):
        # Pad to max length
        max_len = max(len(f0) if f0 is not None else 0 for f0 in f0_list)
        if max_len > 0:
            padded_f0 = []
            padded_f0_coarse = []
            padded_f0_voiced_mask = []
            for b in batch:
                f0 = b.get('f0')
                f0_coarse = b.get('f0_coarse')
                f0_voiced_mask = b.get('f0_voiced_mask')
                if f0 is not None and len(f0) > 0:
                    # Pad with zeros
                    pad_len = max_len - len(f0)
                    padded_f0.append(torch.cat([f0, torch.zeros(pad_len)]))
                    if f0_coarse is not None:
                        padded_f0_coarse.append(torch.cat([f0_coarse, torch.full((pad_len,), 128, dtype=torch.long)]))
                    else:
                        padded_f0_coarse.append(torch.full((max_len,), 128, dtype=torch.long))
                    # Voiced mask - pad with False (unvoiced)
                    if f0_voiced_mask is not None:
                        padded_f0_voiced_mask.append(torch.cat([f0_voiced_mask, torch.zeros(pad_len, dtype=torch.bool)]))
                    else:
                        padded_f0_voiced_mask.append(torch.zeros(max_len, dtype=torch.bool))
                else:
                    padded_f0.append(torch.zeros(max_len))
                    padded_f0_coarse.append(torch.full((max_len,), 128, dtype=torch.long))
                    padded_f0_voiced_mask.append(torch.zeros(max_len, dtype=torch.bool))
            
            result['f0'] = torch.stack(padded_f0)
            result['f0_coarse'] = torch.stack(padded_f0_coarse)
            result['f0_voiced_mask'] = torch.stack(padded_f0_voiced_mask)
    
    # Context (poprzedni segment)
    if 'context_audio' in batch[0]:
        result['context_audio'] = torch.stack([b['context_audio'] for b in batch])
        result['context_section_embedding'] = torch.stack([b['context_section_embedding'] for b in batch])
        result['context_section_type'] = [b['context_section_type'] for b in batch]
    
    return result


def create_segmented_dataloader(
    annotations_json: str,
    audio_dir: str,
    batch_size: int = 8,
    num_workers: int = 4,
    **kwargs,
) -> DataLoader:
    """Helper do tworzenia DataLoader"""
    
    dataset = SegmentedMusicDataset(
        annotations_json=annotations_json,
        audio_dir=audio_dir,
        **kwargs,
    )
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_segmented,
        pin_memory=True,
    )


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Test Segmented Dataset')
    parser.add_argument('--annotations', type=str, default='./data_v2/segments.json')
    parser.add_argument('--audio_dir', type=str, required=True)
    parser.add_argument('--max_tracks', type=int, default=10)
    
    args = parser.parse_args()
    
    print("="*60)
    print("🎵 Testing Segmented Music Dataset")
    print("="*60)
    
    # Test section embedding
    print("\n📊 Section Embeddings:")
    embedder = SectionEmbedding(embed_dim=64)
    for section in ['intro', 'verse', 'chorus', 'bridge', 'outro']:
        emb = embedder.get_embedding(section)
        idx = embedder.get_index(section)
        print(f"  {section}: idx={idx}, emb_norm={np.linalg.norm(emb):.3f}")
    
    # Test dataset
    print(f"\n📂 Loading dataset...")
    
    if Path(args.annotations).exists():
        dataset = SegmentedMusicDataset(
            annotations_json=args.annotations,
            audio_dir=args.audio_dir,
            segment_duration=10.0,
            include_context=True,
            max_tracks=args.max_tracks,
        )
        
        print(f"\nDataset size: {len(dataset)}")
        
        if len(dataset) > 0:
            sample = dataset[0]
            print(f"\n📝 Sample:")
            print(f"  Track ID: {sample['track_id']}")
            print(f"  Section: {sample['section_type']}")
            print(f"  Position: {sample['position']:.2%}")
            print(f"  Tempo: {sample['tempo']:.0f} BPM")
            print(f"  Energy: {sample['energy']:.2f}")
            print(f"  Has vocals: {sample['has_vocals']}")
            print(f"  Audio shape: {sample['audio'].shape}")
            print(f"  Section embedding shape: {sample['section_embedding'].shape}")
            print(f"  Context audio shape: {sample['context_audio'].shape}")
            print(f"  Prompt: {sample['prompt']}")
    else:
        print(f"  ⚠️ Annotations file not found: {args.annotations}")
        print("  Run segment_annotator.py first to create annotations.")
