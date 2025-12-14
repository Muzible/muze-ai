#!/usr/bin/env python3
"""
🔍 MP3 Folder Scanner - Generuje CSV metadanych z tagów ID3

Skanuje folder z MP3 i wyciąga:
- artist (z ID3 lub nazwy folderu)
- title (z ID3 lub nazwy pliku)
- genre (z ID3 lub "unknown")
- album (z ID3)
- duration (z audio)
- language (opcjonalnie z tagu lub auto-detect)

Użycie:
    # Podstawowy skan
    python tools_v2/scan_mp3_folder.py \
        --input_dir /path/to/mp3s \
        --output ./data_v2/my_music_metadata.csv
    
    # With language detection (slower)
    python tools_v2/scan_mp3_folder.py \
        --input_dir /path/to/mp3s \
        --output ./data_v2/my_music_metadata.csv \
        --detect_language
    
    # Tylko hip-hop subfolder
    python tools_v2/scan_mp3_folder.py \
        --input_dir /path/to/hiphop \
        --output ./data_v2/hiphop_metadata.csv \
        --default_genre "hip-hop"
"""

import os
import sys
import csv
import argparse
from pathlib import Path
from typing import Optional, Dict, Any, List
from dataclasses import dataclass
from tqdm import tqdm

# Mutagen for ID3 tags
from mutagen.mp3 import MP3
from mutagen.id3 import ID3
from mutagen import MutagenError


@dataclass
class TrackMetadata:
    """Metadane pojedynczego utworu"""
    file_path: str
    artist: str = "Unknown Artist"
    title: str = ""
    album: str = ""
    genre: str = "unknown"
    language: str = ""
    duration: float = 0.0
    year: str = ""
    
    # Metadata source
    metadata_source: str = "none"  # id3, filename, folder


def extract_id3_tags(file_path: Path) -> Dict[str, Any]:
    """
    Wyciąga tagi ID3 z pliku MP3.
    
    Returns:
        Dict z kluczami: artist, title, album, genre, year, duration
    """
    result = {
        'artist': None,
        'title': None,
        'album': None,
        'genre': None,
        'year': None,
        'duration': 0.0,
        'has_id3': False,
    }
    
    try:
        audio = MP3(str(file_path))
        result['duration'] = audio.info.length if audio.info else 0.0
        
        # Try to get ID3 tags
        try:
            tags = ID3(str(file_path))
            result['has_id3'] = True
            
            # Artist - try multiple tags
            for tag in ['TPE1', 'TPE2', 'TOPE']:  # Artist, Album Artist, Original Artist
                if tag in tags:
                    result['artist'] = str(tags[tag].text[0]).strip()
                    break
            
            # Title
            if 'TIT2' in tags:
                result['title'] = str(tags['TIT2'].text[0]).strip()
            
            # Album
            if 'TALB' in tags:
                result['album'] = str(tags['TALB'].text[0]).strip()
            
            # Genre
            if 'TCON' in tags:
                genre_raw = str(tags['TCON'].text[0]).strip()
                # Czasem genre jest jako "(13)" = Pop, normalizuj
                result['genre'] = normalize_genre(genre_raw)
            
            # Year
            for tag in ['TDRC', 'TYER', 'TDAT']:
                if tag in tags:
                    result['year'] = str(tags[tag].text[0])[:4]
                    break
                    
        except Exception:
            # No ID3 tags, but we have duration from MP3 header
            pass
            
    except MutagenError as e:
        print(f"  ⚠️ Mutagen error: {file_path.name} - {e}")
    except Exception as e:
        print(f"  ⚠️ Error reading: {file_path.name} - {e}")
    
    return result


def normalize_genre(genre: str) -> str:
    """
    Normalizuje genre string.
    
    Przykłady:
    - "(13)" -> "Pop"
    - "Hip-Hop/Rap" -> "hip-hop"
    - "ELECTRONIC" -> "electronic"
    """
    if not genre:
        return "unknown"
    
    # ID3v1 numeric genres
    id3v1_genres = {
        "0": "blues", "1": "classic rock", "2": "country", "3": "dance",
        "4": "disco", "5": "funk", "6": "grunge", "7": "hip-hop",
        "8": "jazz", "9": "metal", "10": "new age", "11": "oldies",
        "12": "other", "13": "pop", "14": "r&b", "15": "rap",
        "16": "reggae", "17": "rock", "18": "techno", "19": "industrial",
        "20": "alternative", "21": "ska", "22": "death metal", "23": "pranks",
        "24": "soundtrack", "25": "euro-techno", "26": "ambient",
        "27": "trip-hop", "28": "vocal", "29": "jazz+funk", "30": "fusion",
        "31": "trance", "32": "classical", "33": "instrumental", "34": "acid",
        "35": "house", "36": "game", "37": "sound clip", "38": "gospel",
        "39": "noise", "40": "alternative rock", "41": "bass",
        "42": "soul", "43": "punk", "44": "space", "45": "meditative",
        "46": "instrumental pop", "47": "instrumental rock", "48": "ethnic",
        "49": "gothic", "50": "darkwave", "51": "techno-industrial",
        "52": "electronic", "53": "pop-folk", "54": "eurodance",
    }
    
    # Check for numeric format "(13)" or "13"
    genre_clean = genre.strip("()[] ")
    if genre_clean.isdigit():
        return id3v1_genres.get(genre_clean, "unknown")
    
    # Normalize common variations
    genre_lower = genre.lower().strip()
    
    # Common mappings
    genre_map = {
        'hip-hop/rap': 'hip-hop',
        'hiphop': 'hip-hop',
        'hip hop': 'hip-hop',
        'rap': 'hip-hop',
        'r&b/soul': 'r&b',
        'rhythm and blues': 'r&b',
        'rock & roll': 'rock',
        'heavy metal': 'metal',
        'edm': 'electronic',
        'electronica': 'electronic',
        'classical music': 'classical',
        'polish hip-hop': 'hip-hop',
        'polski hip-hop': 'hip-hop',
        'polski rap': 'hip-hop',
    }
    
    return genre_map.get(genre_lower, genre_lower)


def extract_from_filename(file_path: Path) -> Dict[str, str]:
    """
    Próbuje wyciągnąć artist i title z nazwy pliku.
    
    Obsługiwane formaty:
    - "Artist - Title.mp3"
    - "01 - Artist - Title.mp3"
    - "01. Title.mp3" (tylko title)
    - "Title.mp3" (tylko title)
    """
    result = {'artist': None, 'title': None}
    
    stem = file_path.stem  # Nazwa bez rozszerzenia
    
    # Remove track number from beginning
    # "01 - Something" -> "Something"
    # "01. Something" -> "Something"
    import re
    stem = re.sub(r'^[\d]+[\.\-\s]+', '', stem)
    
    # Try to split by " - "
    if ' - ' in stem:
        parts = stem.split(' - ', 1)
        if len(parts) == 2:
            result['artist'] = parts[0].strip()
            result['title'] = parts[1].strip()
    else:
        # Tylko title
        result['title'] = stem.strip()
    
    return result


def extract_from_folder(file_path: Path) -> Dict[str, str]:
    """
    Próbuje wyciągnąć artist z nazwy folderu nadrzędnego.
    
    Struktura: /muzyka/Artist Name/album/track.mp3
             lub: /muzyka/Artist Name/track.mp3
    """
    result = {'artist': None, 'album': None}
    
    parts = file_path.parts
    
    # Go up the folder structure
    if len(parts) >= 2:
        parent = parts[-2]  # Direct parent folder
        
        # Check if it looks like an artist name (not a number, not "mp3", etc)
        if parent and not parent.isdigit() and len(parent) > 2:
            # Might be album
            if len(parts) >= 3:
                grandparent = parts[-3]
                if grandparent and not grandparent.isdigit() and len(grandparent) > 2:
                    # grandparent = artist, parent = album
                    result['artist'] = grandparent
                    result['album'] = parent
                else:
                    result['artist'] = parent
            else:
                result['artist'] = parent
    
    return result


def detect_language_from_text(text: str) -> str:
    """
    Prosta detekcja języka na podstawie znaków i słów.
    
    Returns: 'pl', 'en', lub ''
    """
    if not text:
        return ''
    
    text_lower = text.lower()
    
    # Polskie znaki
    polish_chars = set('ąćęłńóśźżĄĆĘŁŃÓŚŹŻ')
    has_polish_chars = any(c in polish_chars for c in text)
    
    if has_polish_chars:
        return 'pl'
    
    # Polish keywords (in titles/artists)
    polish_words = ['i', 'w', 'nie', 'na', 'się', 'do', 'jest', 'to', 
                   'feat', 'feat.', 'ft', 'ft.']  # feat is universal
    
    # Check Polish words in context
    polish_artist_indicators = ['quebonafide', 'taco', 'hemingway', 'sobel', 
                                'bedoes', 'mata', 'szpaku', 'paluch', 'pezet',
                                'ostr', 'kękę', 'keke', 'young', 'lull']
    
    for word in polish_artist_indicators:
        if word in text_lower:
            return 'pl'
    
    return ''  # Unknown/default to English


def scan_folder(
    input_dir: Path,
    default_genre: str = "unknown",
    detect_language: bool = False,
    show_progress: bool = True,
) -> List[TrackMetadata]:
    """
    Skanuje folder rekurencyjnie i zbiera metadane MP3.
    """
    tracks = []
    
    # Find all MP3
    mp3_files = list(input_dir.rglob("*.mp3"))
    mp3_files.extend(input_dir.rglob("*.MP3"))
    
    # Deduplikacja (case-insensitive na macOS)
    mp3_files = list({str(f).lower(): f for f in mp3_files}.values())
    
    print(f"📂 Znaleziono {len(mp3_files)} plików MP3 w {input_dir}")
    
    if not mp3_files:
        return tracks
    
    iterator = tqdm(mp3_files, desc="Skanowanie") if show_progress else mp3_files
    
    stats = {'id3': 0, 'filename': 0, 'folder': 0, 'none': 0}
    
    for mp3_path in iterator:
        metadata = TrackMetadata(file_path=str(mp3_path))
        
        # 1. First try ID3
        id3_data = extract_id3_tags(mp3_path)
        metadata.duration = id3_data['duration']
        
        if id3_data['has_id3'] and id3_data['artist']:
            metadata.artist = id3_data['artist']
            metadata.title = id3_data['title'] or mp3_path.stem
            metadata.album = id3_data['album'] or ""
            metadata.genre = id3_data['genre'] or default_genre
            metadata.year = id3_data['year'] or ""
            metadata.metadata_source = "id3"
            stats['id3'] += 1
            
        else:
            # 2. Try from filename
            filename_data = extract_from_filename(mp3_path)
            folder_data = extract_from_folder(mp3_path)
            
            if filename_data['artist']:
                metadata.artist = filename_data['artist']
                metadata.title = filename_data['title'] or mp3_path.stem
                metadata.metadata_source = "filename"
                stats['filename'] += 1
                
            elif folder_data['artist']:
                metadata.artist = folder_data['artist']
                metadata.title = filename_data['title'] or mp3_path.stem
                metadata.album = folder_data['album'] or ""
                metadata.metadata_source = "folder"
                stats['folder'] += 1
                
            else:
                # 3. Last resort - Unknown Artist
                metadata.artist = "Unknown Artist"
                metadata.title = mp3_path.stem
                metadata.metadata_source = "none"
                stats['none'] += 1
            
            # Genre z ID3 lub default
            metadata.genre = id3_data['genre'] or default_genre
        
        # Language detection (optional)
        if detect_language:
            # Check based on artist + title
            text_to_check = f"{metadata.artist} {metadata.title}"
            metadata.language = detect_language_from_text(text_to_check)
        
        tracks.append(metadata)
    
    # Stats
    print(f"\n📊 Źródła metadanych:")
    print(f"   ID3 tags:     {stats['id3']:>5} ({100*stats['id3']/len(tracks):.1f}%)")
    print(f"   Filename:     {stats['filename']:>5} ({100*stats['filename']/len(tracks):.1f}%)")
    print(f"   Folder name:  {stats['folder']:>5} ({100*stats['folder']/len(tracks):.1f}%)")
    print(f"   Unknown:      {stats['none']:>5} ({100*stats['none']/len(tracks):.1f}%)")
    
    return tracks


def save_csv(tracks: List[TrackMetadata], output_path: Path):
    """Zapisuje metadane do CSV"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    fieldnames = [
        'file_path', 'artist', 'title', 'album', 'genre', 
        'language', 'duration', 'year', 'metadata_source'
    ]
    
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        
        for track in tracks:
            writer.writerow({
                'file_path': track.file_path,
                'artist': track.artist,
                'title': track.title,
                'album': track.album,
                'genre': track.genre,
                'language': track.language,
                'duration': f"{track.duration:.2f}",
                'year': track.year,
                'metadata_source': track.metadata_source,
            })
    
    print(f"\n✅ Zapisano {len(tracks)} utworów do {output_path}")


def export_missing_for_completion(tracks: List[TrackMetadata], output_path: Path):
    """
    Eksportuje CSV z utworami które mają brakujące genre/artist do ręcznego uzupełnienia.
    
    Format CSV gotowy do użycia przez build_dataset_v2.py --tracks_csv
    Po uzupełnieniu kolumny 'genre' można użyć tego pliku jako metadane.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Filter tracks with missing data
    missing_tracks = [
        t for t in tracks 
        if t.genre in ('unknown', '', None) or t.artist in ('Unknown Artist', '', None)
    ]
    
    if not missing_tracks:
        print("\n✅ Wszystkie pliki mają kompletne metadane! Nie trzeba nic uzupełniać.")
        return 0
    
    # Format kompatybilny z build_dataset_v2.py --metadata_mapping
    # Kolumny: file_path, artist, genre (wymagane przez _load_metadata_mapping)
    fieldnames = [
        'file_path',       # Path to file (required - full path!)
        'artist',          # Artist (to complete if missing)
        'genre',           # Genre (TO COMPLETE!)
        'title',           # Title (optional)
        'album',           # Album (optional)
        '_current_genre',  # Current genre (for info - ignored by builder)
        '_current_artist', # Current artist (for info - ignored by builder)
        '_metadata_source' # Where current comes from (for info - ignored by builder)
    ]
    
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        
        for track in missing_tracks:
            # Columns starting with _ are info only (ignored by builder)
            # build_dataset_v2.py will use file_path, artist, genre
            writer.writerow({
                'file_path': track.file_path,
                'artist': track.artist if track.artist != 'Unknown Artist' else '',
                'genre': '',  # <-- TO COMPLETE!
                'title': track.title,
                'album': track.album,
                '_current_genre': track.genre,
                '_current_artist': track.artist,
                '_metadata_source': track.metadata_source,
            })
    
    print(f"\n📝 WYEKSPORTOWANO DO UZUPEŁNIENIA: {output_path}")
    print(f"   Plików do uzupełnienia: {len(missing_tracks)}")
    print(f"")
    print(f"   📋 INSTRUKCJA:")
    print(f"   1. Otwórz {output_path} w Excel/LibreOffice")
    print(f"   2. Uzupełnij kolumnę 'genre' (np. 'rock', 'pop', 'hip-hop')")
    print(f"   3. Opcjonalnie uzupełnij 'artist' jeśli pusty")
    print(f"   4. Zapisz plik (zachowaj format CSV UTF-8)")
    print(f"   5. Uruchom builder z tym plikiem:")
    print(f"      METADATA_MAPPING={output_path} ./run_multi_gpu.sh 8")
    print(f"")
    
    return len(missing_tracks)


def print_summary(tracks: List[TrackMetadata]):
    """Displays dataset summary"""
    from collections import Counter
    
    # Unique artists
    artists = Counter(t.artist for t in tracks)
    genres = Counter(t.genre for t in tracks if t.genre != "unknown")
    languages = Counter(t.language for t in tracks if t.language)
    
    total_duration = sum(t.duration for t in tracks)
    
    print("\n" + "="*60)
    print("📊 PODSUMOWANIE DATASETU")
    print("="*60)
    print(f"   Łączna liczba utworów: {len(tracks)}")
    print(f"   Łączny czas: {total_duration/3600:.1f} godzin")
    print(f"   Unikalnych artystów: {len(artists)}")
    
    print(f"\n   Top 10 artystów:")
    for artist, count in artists.most_common(10):
        print(f"      {artist}: {count}")
    
    if genres:
        print(f"\n   Top gatunki:")
        for genre, count in genres.most_common(10):
            print(f"      {genre}: {count}")
    
    if languages:
        print(f"\n   Języki:")
        for lang, count in languages.most_common():
            lang_name = {'pl': 'Polish', 'en': 'English'}.get(lang, lang)
            print(f"      {lang_name}: {count}")
    
    # Artists with "Unknown"
    unknown_count = artists.get("Unknown Artist", 0)
    if unknown_count > 0:
        print(f"\n   ⚠️ Utwory bez artysty: {unknown_count}")
        print("      Rozważ ręczne uzupełnienie CSV lub organizację folderów")


def main():
    parser = argparse.ArgumentParser(
        description='🔍 Skanuj folder MP3 i generuj CSV metadanych'
    )
    
    parser.add_argument('--input_dir', type=str, required=True,
                        help='Folder z plikami MP3')
    parser.add_argument('--output', type=str, default='./data_v2/scanned_metadata.csv',
                        help='Ścieżka do wyjściowego CSV')
    parser.add_argument('--default_genre', type=str, default='unknown',
                        help='Domyślny gatunek jeśli brak w tagach (np. "hip-hop")')
    parser.add_argument('--detect_language', action='store_true',
                        help='Próbuj wykryć język z nazw artystów/tytułów')
    parser.add_argument('--export_missing', type=str, default=None,
                        help='Eksportuj pliki z brakującymi genre do osobnego CSV do uzupełnienia')
    parser.add_argument('--quiet', action='store_true',
                        help='Bez progress bar')
    
    args = parser.parse_args()
    
    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        print(f"❌ Folder nie istnieje: {input_dir}")
        sys.exit(1)
    
    # Skanuj
    tracks = scan_folder(
        input_dir=input_dir,
        default_genre=args.default_genre,
        detect_language=args.detect_language,
        show_progress=not args.quiet,
    )
    
    if not tracks:
        print("❌ Nie znaleziono plików MP3")
        sys.exit(1)
    
    # Podsumowanie
    print_summary(tracks)
    
    # Save full CSV
    save_csv(tracks, Path(args.output))
    
    # Export for completion if --export_missing provided
    missing_count = 0
    if args.export_missing:
        missing_count = export_missing_for_completion(tracks, Path(args.export_missing))
    else:
        # Check if there are missing and suggest export
        missing_tracks = [
            t for t in tracks 
            if t.genre in ('unknown', '', None) or t.artist in ('Unknown Artist', '', None)
        ]
        if missing_tracks:
            missing_path = Path(args.output).with_suffix('.to_complete.csv')
            print(f"\n⚠️  {len(missing_tracks)} plików ma brakujące metadane!")
            print(f"   Użyj --export_missing {missing_path} aby wyeksportować do uzupełnienia")
    
    # Next step
    if missing_count > 0:
        print("\n💡 Po uzupełnieniu CSV uruchom:")
        print(f"   METADATA_MAPPING={args.export_missing} ./run_multi_gpu.sh 8 {args.input_dir}")
        print(f"")
        print(f"   Lub bezpośrednio:")
        print(f"   python build_dataset_v2.py \\")
        print(f"       --audio_dir {args.input_dir} \\")
        print(f"       --metadata_mapping {args.export_missing} \\")
        print(f"       --output ./data_v2/training_dataset.json \\")
        print(f"       --with_segments --extract_features")
    else:
        print("\n💡 Następny krok:")
        print(f"   python build_dataset_v2.py \\")
        print(f"       --audio_dir {args.input_dir} \\")
        print(f"       --metadata_mapping {args.output} \\")
        print(f"       --output ./data_v2/training_dataset.json \\")
        print(f"       --with_segments --extract_features")


if __name__ == "__main__":
    main()
