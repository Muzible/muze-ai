#!/usr/bin/env python3
"""
Pre-analysis script for metadata validation before dataset generation.
Checks genre, artist, title from ID3 tags and CSV files.
Reports missing metadata and generates file for manual completion.
"""

import os
import sys
import json
import csv
import re
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Optional, Set, Tuple
from dataclasses import dataclass, field, asdict
import argparse

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from mutagen import File as MutagenFile
    from mutagen.id3 import ID3
    from mutagen.mp3 import MP3
    MUTAGEN_AVAILABLE = True
except ImportError:
    MUTAGEN_AVAILABLE = False
    print("⚠️  mutagen nie jest zainstalowany. Uruchom: pip install mutagen")


@dataclass
class TrackMetadata:
    """Metadata extracted from a track"""
    file_path: str
    filename: str
    
    # Core metadata
    title: Optional[str] = None
    artist: Optional[str] = None
    album: Optional[str] = None
    genre: Optional[str] = None
    
    # Additional metadata
    year: Optional[str] = None
    bpm: Optional[float] = None
    duration: Optional[float] = None
    
    # Validation status
    genre_mapped: Optional[str] = None  # Mapped to known genre
    genre_valid: bool = False
    missing_fields: List[str] = field(default_factory=list)
    
    # Source
    source: str = "id3"  # "id3", "csv", or "manual"


class GenreTaxonomy:
    """Genre taxonomy from CSV with hierarchy support"""
    
    def __init__(self, csv_path: str):
        self.genres: Dict[int, dict] = {}
        self.genre_names: Set[str] = set()
        self.name_to_id: Dict[str, int] = {}
        self.parent_map: Dict[int, int] = {}
        self._load_csv(csv_path)
        
    def _load_csv(self, csv_path: str):
        """Load genre taxonomy from CSV"""
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                genre_id = int(row['genre_id'])
                parent_id = int(row['parent']) if row['parent'] else 0
                title = row['title'].strip()
                
                self.genres[genre_id] = {
                    'id': genre_id,
                    'parent': parent_id,
                    'title': title
                }
                self.genre_names.add(title.lower())
                self.name_to_id[title.lower()] = genre_id
                self.parent_map[genre_id] = parent_id
                
        print(f"📚 Załadowano {len(self.genres)} gatunków z taksonomii")
        
    def get_parent_chain(self, genre_id: int) -> List[str]:
        """Get full parent chain for a genre"""
        chain = []
        current = genre_id
        while current in self.genres and current != 0:
            chain.append(self.genres[current]['title'])
            current = self.parent_map.get(current, 0)
        return chain
    
    def match_genre(self, genre_str: str) -> Optional[Tuple[str, int]]:
        """
        Match genre string to taxonomy.
        Returns (matched_genre_name, genre_id) or None
        """
        if not genre_str:
            return None
            
        genre_lower = genre_str.lower().strip()
        
        # Direct match
        if genre_lower in self.name_to_id:
            genre_id = self.name_to_id[genre_lower]
            return (self.genres[genre_id]['title'], genre_id)
        
        # Partial match - check if genre contains any known genre
        for known_genre in self.genre_names:
            if known_genre in genre_lower or genre_lower in known_genre:
                genre_id = self.name_to_id[known_genre]
                return (self.genres[genre_id]['title'], genre_id)
        
        # Common mappings
        common_mappings = {
            'hip hop': 'Hip-Hop',
            'hiphop': 'Hip-Hop',
            'r&b': 'Soul-RnB',
            'rnb': 'Soul-RnB',
            'rhythm and blues': 'Soul-RnB',
            'electro': 'Electronic',
            'electronica': 'Electronic',
            'edm': 'Electronic',
            'indie': 'Indie-Rock',
            'alternative': 'Alternative',
            'alt rock': 'Alternative',
            'metal': 'Metal',
            'heavy metal': 'Metal',
            'punk': 'Punk',
            'punk rock': 'Punk',
            'classical': 'Classical',
            'country': 'Country & Western',
            'folk': 'Folk',
            'jazz': 'Jazz',
            'blues': 'Blues',
            'reggae': 'Reggae - Dub',
            'soul': 'Soul-RnB',
            'funk': 'Funk',
            'disco': 'Disco',
            'pop': 'Pop',
            'rock': 'Rock',
            'rap': 'Rap',
            'trap': 'Rap',
            'drill': 'Rap',
            'grime': 'Rap',
            'lo-fi': 'Lo-Fi',
            'lofi': 'Lo-Fi',
            'ambient': 'Ambient',
            'house': 'House',
            'techno': 'Techno',
            'trance': 'Techno',
            'drum and bass': 'Drum & Bass',
            'dnb': 'Drum & Bass',
            'dubstep': 'Dubstep',
            'experimental': 'Experimental',
            'noise': 'Noise',
            'world': 'International',
            'latin': 'Latin',
            'brazilian': 'Brazilian',
            'african': 'African',
            'revival': 'Rock',  # Rock revival, folk revival, etc.
            'singer-songwriter': 'Singer-Songwriter',
            'acoustic': 'Folk',
            'instrumental': 'Instrumental',
        }
        
        for pattern, mapped in common_mappings.items():
            if pattern in genre_lower:
                if mapped.lower() in self.name_to_id:
                    genre_id = self.name_to_id[mapped.lower()]
                    return (self.genres[genre_id]['title'], genre_id)
        
        return None
    
    def get_all_genres(self) -> List[str]:
        """Get list of all genre names"""
        return sorted([g['title'] for g in self.genres.values()])


class MetadataAnalyzer:
    """Analyzes audio files for metadata completeness"""
    
    def __init__(self, taxonomy: GenreTaxonomy):
        self.taxonomy = taxonomy
        self.tracks: List[TrackMetadata] = []
        self.csv_metadata: Dict[str, dict] = {}
        self.music_dir: Optional[Path] = None  # Ustawiane przez analyze_directory()
        
    def load_csv_metadata(self, 
                          tracks_csv: str,
                          artists_csv: Optional[str] = None,
                          genres_csv: Optional[str] = None):
        """Load metadata from FMA-style CSV files"""
        
        # Load tracks
        if os.path.exists(tracks_csv):
            with open(tracks_csv, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    track_id = row.get('track_id', '')
                    if track_id:
                        self.csv_metadata[track_id] = {
                            'title': row.get('track_title', ''),
                            'artist': row.get('artist_name', ''),
                            'album': row.get('album_title', ''),
                            'genre': row.get('track_genre_top', '') or row.get('genre', ''),
                            'duration': row.get('track_duration', ''),
                        }
            print(f"📄 Załadowano {len(self.csv_metadata)} rekordów z CSV")
    
    def extract_track_id_from_path(self, file_path: str) -> Optional[str]:
        """Extract track ID from FMA-style path (e.g., 000/000002.mp3 -> 2)"""
        filename = os.path.basename(file_path)
        match = re.match(r'^(\d+)\.mp3$', filename)
        if match:
            return str(int(match.group(1)))  # Remove leading zeros
        return None
    
    def extract_id3_metadata(self, file_path: str) -> TrackMetadata:
        """Extract metadata from ID3 tags"""
        metadata = TrackMetadata(
            file_path=file_path,
            filename=os.path.basename(file_path)
        )
        
        if not MUTAGEN_AVAILABLE:
            metadata.missing_fields = ['all - mutagen not installed']
            return metadata
        
        try:
            audio = MutagenFile(file_path, easy=True)
            if audio is None:
                metadata.missing_fields = ['all - file not readable']
                return metadata
            
            # Extract tags
            if audio.tags:
                metadata.title = audio.tags.get('title', [None])[0]
                metadata.artist = audio.tags.get('artist', [None])[0]
                metadata.album = audio.tags.get('album', [None])[0]
                metadata.genre = audio.tags.get('genre', [None])[0]
                metadata.year = audio.tags.get('date', [None])[0]
                
                # Try to get BPM
                bpm = audio.tags.get('bpm', [None])[0]
                if bpm:
                    try:
                        metadata.bpm = float(bpm)
                    except ValueError:
                        pass
            
            # Get duration
            if hasattr(audio, 'info') and audio.info:
                metadata.duration = audio.info.length
                
        except Exception as e:
            metadata.missing_fields.append(f'error: {str(e)}')
            
        return metadata
    
    def analyze_file(self, file_path: str) -> TrackMetadata:
        """Analyze a single file for metadata"""
        
        # First try ID3 tags
        metadata = self.extract_id3_metadata(file_path)
        
        # Then try CSV lookup
        track_id = self.extract_track_id_from_path(file_path)
        if track_id and track_id in self.csv_metadata:
            csv_data = self.csv_metadata[track_id]
            
            # Fill in missing fields from CSV
            if not metadata.title and csv_data.get('title'):
                metadata.title = csv_data['title']
                metadata.source = 'csv'
            if not metadata.artist and csv_data.get('artist'):
                metadata.artist = csv_data['artist']
            if not metadata.album and csv_data.get('album'):
                metadata.album = csv_data['album']
            if not metadata.genre and csv_data.get('genre'):
                metadata.genre = csv_data['genre']
        
        # Check for missing required fields
        metadata.missing_fields = []
        if not metadata.title:
            metadata.missing_fields.append('title')
        if not metadata.artist:
            metadata.missing_fields.append('artist')
        if not metadata.genre:
            metadata.missing_fields.append('genre')
        
        # Validate genre against taxonomy
        if metadata.genre:
            match = self.taxonomy.match_genre(metadata.genre)
            if match:
                metadata.genre_mapped = match[0]
                metadata.genre_valid = True
            else:
                metadata.genre_valid = False
                if 'genre' not in metadata.missing_fields:
                    metadata.missing_fields.append('genre_unmapped')
        
        return metadata
    
    def analyze_directory(self, 
                          directory: str, 
                          extensions: List[str] = ['.mp3', '.wav', '.flac', '.ogg', '.m4a'],
                          recursive: bool = True) -> List[TrackMetadata]:
        """Analyze all audio files in a directory"""
        
        audio_files = []
        dir_path = Path(directory)
        self.music_dir = dir_path  # Zapisz katalog główny do obliczania ścieżek względnych
        
        if recursive:
            for ext in extensions:
                audio_files.extend(dir_path.rglob(f'*{ext}'))
        else:
            for ext in extensions:
                audio_files.extend(dir_path.glob(f'*{ext}'))
        
        print(f"🔍 Znaleziono {len(audio_files)} plików audio")
        
        self.tracks = []
        for i, file_path in enumerate(sorted(audio_files)):
            if (i + 1) % 100 == 0:
                print(f"  Analizuję: {i + 1}/{len(audio_files)}")
            metadata = self.analyze_file(str(file_path))
            self.tracks.append(metadata)
        
        return self.tracks
    
    def get_statistics(self) -> dict:
        """Get statistics about analyzed tracks"""
        stats = {
            'total_tracks': len(self.tracks),
            'complete': 0,
            'missing_genre': 0,
            'missing_artist': 0,
            'missing_title': 0,
            'genre_unmapped': 0,
            'genre_distribution': defaultdict(int),
            'unmapped_genres': defaultdict(int),
        }
        
        for track in self.tracks:
            if not track.missing_fields:
                stats['complete'] += 1
            if 'genre' in track.missing_fields:
                stats['missing_genre'] += 1
            if 'artist' in track.missing_fields:
                stats['missing_artist'] += 1
            if 'title' in track.missing_fields:
                stats['missing_title'] += 1
            if 'genre_unmapped' in track.missing_fields:
                stats['genre_unmapped'] += 1
                if track.genre:
                    stats['unmapped_genres'][track.genre] += 1
            
            if track.genre_mapped:
                stats['genre_distribution'][track.genre_mapped] += 1
        
        return stats
    
    def generate_report(self, output_path: Optional[str] = None) -> str:
        """Generate a text report of metadata analysis"""
        stats = self.get_statistics()
        
        lines = [
            "=" * 70,
            "📊 RAPORT ANALIZY METADANYCH",
            "=" * 70,
            "",
            f"📁 Łącznie plików: {stats['total_tracks']}",
            f"✅ Kompletne metadane: {stats['complete']} ({100*stats['complete']/max(1,stats['total_tracks']):.1f}%)",
            "",
            "📝 Brakujące pola:",
            f"   • Genre: {stats['missing_genre']} plików",
            f"   • Artist: {stats['missing_artist']} plików",
            f"   • Title: {stats['missing_title']} plików",
            f"   • Genre niezmapowane: {stats['genre_unmapped']} plików",
            "",
        ]
        
        if stats['genre_distribution']:
            lines.append("🎵 Rozkład gatunków (zmapowane):")
            for genre, count in sorted(stats['genre_distribution'].items(), key=lambda x: -x[1])[:20]:
                lines.append(f"   • {genre}: {count}")
        
        if stats['unmapped_genres']:
            lines.append("")
            lines.append("⚠️  Niezmapowane gatunki (wymagają ręcznego mapowania):")
            for genre, count in sorted(stats['unmapped_genres'].items(), key=lambda x: -x[1]):
                lines.append(f"   • '{genre}': {count} plików")
        
        report = "\n".join(lines)
        
        if output_path:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(report)
            print(f"📄 Raport zapisany do: {output_path}")
        
        return report
    
    def export_for_manual_completion(self, output_path: str, only_missing_genre: bool = True):
        """Export tracks with missing metadata for manual completion
        
        Eksportuje ścieżki względne do katalogu skanowania, co pozwala na
        jednoznaczną identyfikację plików nawet przy powtarzających się nazwach.
        """
        
        tracks_to_export = []
        for track in self.tracks:
            if only_missing_genre and 'genre' not in track.missing_fields and 'genre_unmapped' not in track.missing_fields:
                continue
            
            # Oblicz ścieżkę względną do katalogu skanowania
            try:
                relative_path = str(Path(track.file_path).relative_to(self.music_dir))
            except ValueError:
                relative_path = track.file_path  # Fallback do pełnej ścieżki
            
            tracks_to_export.append({
                'file_path': relative_path,  # Ścieżka względna
                'filename': track.filename,
                'title': track.title or '',
                'artist': track.artist or '',
                'genre_original': track.genre or '',
                'genre_mapped': track.genre_mapped or '',
                'genre_to_assign': '',  # Field for manual entry
                'missing_fields': track.missing_fields,
            })
        
        # Sort by file_path for better organization
        tracks_to_export.sort(key=lambda x: x['file_path'])
        
        # Export as JSON (format kompatybilny z build_dataset_v2.py metadata_mapping_file)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump({
                'instructions': 'Uzupełnij pole "genre" dla każdego pliku. Użyj gatunków z listy "available_genres". Plik można przekazać do build_dataset_v2.py jako metadata_mapping_file.',
                'note': 'Kolumna "file_path" zawiera ścieżkę względną do katalogu skanowania. Można używać ścieżek lub samych nazw plików w metadata_mapping_file.',
                'available_genres': self.taxonomy.get_all_genres(),
                'tracks': tracks_to_export
            }, f, ensure_ascii=False, indent=2)
        
        print(f"📝 Wyeksportowano {len(tracks_to_export)} plików do uzupełnienia: {output_path}")
        
        # Also export as CSV for easier editing (format kompatybilny z metadata_mapping_file!)
        # Kolumny: filename,artist,genre - język wykrywany automatycznie przez Whisper!
        csv_path = output_path.replace('.json', '.csv')
        with open(csv_path, 'w', encoding='utf-8', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['filename', 'artist', 'genre'])
            writer.writeheader()
            for track in tracks_to_export:
                writer.writerow({
                    'filename': track['file_path'],  # Ścieżka względna (nie tylko nazwa!)
                    'artist': track['artist'],
                    'genre': track['genre_mapped'] or track['genre_original'] or '',  # Użyj zmapowanego jeśli jest
                })
        
        print(f"📝 CSV do edycji (kompatybilny z metadata_mapping_file): {csv_path}")
        print(f"   💡 Ścieżki są względne do: {self.music_dir}")
        print(f"   💡 Język wykrywany automatycznie przez Whisper - nie trzeba wypełniać!")
        print(f"   💡 Po edycji użyj: metadata_mapping_file='{csv_path}'")
        
        return tracks_to_export
    
    def validate_ready_for_dataset(self) -> Tuple[bool, str]:
        """Check if all tracks are ready for dataset generation"""
        stats = self.get_statistics()
        
        issues = []
        if stats['missing_genre'] > 0:
            issues.append(f"{stats['missing_genre']} plików bez gatunku")
        if stats['genre_unmapped'] > 0:
            issues.append(f"{stats['genre_unmapped']} plików z niezmapowanym gatunkiem")
        
        if issues:
            return False, "❌ NIE GOTOWE do generowania datasetu:\n   • " + "\n   • ".join(issues)
        else:
            return True, f"✅ GOTOWE do generowania datasetu! ({stats['total_tracks']} plików)"


def main():
    parser = argparse.ArgumentParser(
        description='Analiza metadanych plików audio przed generowaniem datasetu'
    )
    parser.add_argument('--music-dir', '-m', type=str, required=True,
                        help='Katalog z plikami muzycznymi')
    parser.add_argument('--genre-csv', '-g', type=str, 
                        default='data/muz_raw_genres_mod.csv',
                        help='CSV z taksonomią gatunków')
    parser.add_argument('--tracks-csv', '-t', type=str,
                        default='data/muz_raw_tracks_mod.csv',
                        help='CSV z metadanymi utworów (opcjonalnie)')
    parser.add_argument('--output-dir', '-o', type=str, default='output',
                        help='Katalog na raporty')
    parser.add_argument('--export-missing', '-e', action='store_true',
                        help='Eksportuj pliki z brakującymi metadanymi')
    parser.add_argument('--no-recursive', action='store_true',
                        help='Nie skanuj podkatalogów')
    
    args = parser.parse_args()
    
    # Resolve paths
    base_dir = Path(__file__).parent.parent
    music_dir = Path(args.music_dir)
    if not music_dir.is_absolute():
        music_dir = base_dir / music_dir
    
    genre_csv = Path(args.genre_csv)
    if not genre_csv.is_absolute():
        genre_csv = base_dir / genre_csv
    
    tracks_csv = Path(args.tracks_csv)
    if not tracks_csv.is_absolute():
        tracks_csv = base_dir / tracks_csv
    
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = base_dir / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 70)
    print("🎵 MUZE-AI: Analiza metadanych")
    print("=" * 70)
    print(f"📁 Katalog muzyki: {music_dir}")
    print(f"📚 Taksonomia gatunków: {genre_csv}")
    print()
    
    # Load genre taxonomy
    if not genre_csv.exists():
        print(f"❌ Nie znaleziono pliku z gatunkami: {genre_csv}")
        sys.exit(1)
    
    taxonomy = GenreTaxonomy(str(genre_csv))
    
    # Create analyzer
    analyzer = MetadataAnalyzer(taxonomy)
    
    # Load CSV metadata if available
    if tracks_csv.exists():
        analyzer.load_csv_metadata(str(tracks_csv))
    
    # Analyze directory
    if not music_dir.exists():
        print(f"❌ Nie znaleziono katalogu: {music_dir}")
        sys.exit(1)
    
    print()
    analyzer.analyze_directory(str(music_dir), recursive=not args.no_recursive)
    
    # Generate report
    print()
    report = analyzer.generate_report(str(output_dir / 'metadata_report.txt'))
    print(report)
    
    # Export missing metadata for manual completion
    if args.export_missing:
        print()
        analyzer.export_for_manual_completion(str(output_dir / 'missing_metadata.json'))
    
    # Final validation
    print()
    print("=" * 70)
    is_ready, message = analyzer.validate_ready_for_dataset()
    print(message)
    print("=" * 70)
    
    if not is_ready and not args.export_missing:
        print()
        print("💡 Wskazówka: Użyj --export-missing aby wyeksportować pliki do uzupełnienia")
    
    return 0 if is_ready else 1


if __name__ == '__main__':
    sys.exit(main())
