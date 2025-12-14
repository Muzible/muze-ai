#!/usr/bin/env python3
"""
🎤 Generate Artist Embeddings

Generuje plik artist_embeddings.json z:
1. Datasetu v2 (--dataset)
2. Folderów vocals/ z embeddings.json (--from_vocals_dir)

Usage:
    # Z datasetu
    python tools_v2/generate_artist_embeddings.py \
        --dataset ./data_v2/training_dataset_v2.json \
        --output ./data_v2/artist_embeddings.json \
        --min_tracks 3

    # Z folderów vocals/ (zbiera istniejące embeddings.json)
    python tools_v2/generate_artist_embeddings.py \
        --from_vocals_dir ./data_v2/vocals/ \
        --output ./data_v2/artist_embeddings.json

Output format:
{
    "Metallica": {
        "style_embedding": [...],               # 256-dim, z miksu, dla "w stylu X" (--style_of)
        "voice_embedding": [...],               # 256-dim, alias = style_embedding
        "voice_embedding_separated": [...],     # 192-dim, z Demucs, dla "jak X" (--voice_as)
        "track_count": 15,
        "tracks_with_separated": 12,
        "avg_vocal_confidence": 0.85,
        "genres": ["metal", "rock"],
        "sample_tracks": ["track_001", "track_015", "track_023"],
        "style_embedding_dim": 256,
        "voice_embedding_separated_dim": 192
    },
    ...
}

Voice options in inference:
    --style_of "Metallica"    → uses style_embedding [256-dim]
    --voice_clone "Metallica" → uses voice_embedding [256-dim] 
    --voice_as "Metallica"    → uses voice_embedding_separated [192-dim] (best quality)
    --voice_clone_samples ./my_voice.wav → extracts from user file
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Any, Optional
from collections import defaultdict
import numpy as np


def load_dataset(dataset_path: str) -> Dict[str, Any]:
    """Ładuje dataset JSON"""
    with open(dataset_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def extract_artist_data(dataset: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """
    Grupuje dane per-artysta.
    
    Returns:
        {
            "Artist Name": {
                "style_embeddings": [[...], [...], ...],  # z miksu
                "voice_embeddings": [[...], [...], ...],  # z separowanych
                "vocal_confidences": [0.8, 0.9, ...],
                "track_ids": ["t1", "t2", ...],
                "genres": ["rock", "metal", ...],
            }
        }
    """
    artists = defaultdict(lambda: {
        'style_embeddings': [],
        'voice_embeddings': [],
        'vocal_confidences': [],
        'track_ids': [],
        'genres': set(),
    })
    
    tracks = dataset.get('tracks', [])
    
    for track in tracks:
        artist = track.get('artist')
        if not artist:
            continue
        
        vocals = track.get('vocals', {})
        
        # Skip tracks without vocals
        if not vocals.get('has_vocals', False):
            continue
        
        # Style embedding (z miksu) - zawsze dostępny jeśli są wokale
        style_emb = vocals.get('voice_embedding')
        if style_emb:
            artists[artist]['style_embeddings'].append(style_emb)
        
        # Voice embedding (z separowanych wokali) - tylko z --use_demucs
        voice_emb = vocals.get('voice_embedding_separated')
        if voice_emb:
            artists[artist]['voice_embeddings'].append(voice_emb)
        
        # Metadata
        artists[artist]['vocal_confidences'].append(
            vocals.get('vocal_confidence', 0.0)
        )
        artists[artist]['track_ids'].append(track.get('track_id', ''))
        
        # Genres
        for genre in track.get('genres', []):
            artists[artist]['genres'].add(genre)
    
    return dict(artists)


def compute_artist_embeddings(
    artist_data: Dict[str, Dict[str, Any]],
    min_tracks: int = 1,
    weighting: str = "uniform",  # "uniform" or "confidence"
) -> Dict[str, Dict[str, Any]]:
    """
    Oblicza uśrednione embeddingi dla każdego artysty.
    
    Args:
        artist_data: Dane z extract_artist_data()
        min_tracks: Minimalna liczba utworów z wokalami
        weighting: Metoda ważenia:
            - "uniform": Zwykła średnia
            - "confidence": Ważona vocal_confidence
    
    Returns:
        Słownik artist_embeddings gotowy do zapisu
    """
    result = {}
    
    for artist, data in artist_data.items():
        style_embs = data['style_embeddings']
        voice_embs = data['voice_embeddings']
        confidences = data['vocal_confidences']
        
        # Skip artists with too few tracks
        if len(style_embs) < min_tracks:
            continue
        
        # Compute style embedding (średnia z miksu)
        style_embedding = None
        if style_embs:
            if weighting == "confidence" and confidences:
                # Ważona średnia
                weights = np.array(confidences[:len(style_embs)])
                weights = weights / weights.sum()
                style_embedding = np.average(style_embs, axis=0, weights=weights).tolist()
            else:
                # Zwykła średnia
                style_embedding = np.mean(style_embs, axis=0).tolist()
        
        # Compute voice embedding (średnia z separowanych)
        voice_embedding = None
        if voice_embs:
            if weighting == "confidence" and confidences:
                weights = np.array(confidences[:len(voice_embs)])
                weights = weights / weights.sum()
                voice_embedding = np.average(voice_embs, axis=0, weights=weights).tolist()
            else:
                voice_embedding = np.mean(voice_embs, axis=0).tolist()
        
        # Build result
        result[artist] = {
            # Embeddingi
            'style_embedding': style_embedding,  # 256-dim, dla "w stylu X" (z miksu)
            'voice_embedding': style_embedding,  # 256-dim, alias dla kompatybilności
            'voice_embedding_separated': voice_embedding,  # 192-dim, dla "jak X" (z Demucs)
            
            # Metadata
            'track_count': len(style_embs),
            'tracks_with_separated': len(voice_embs),
            'avg_vocal_confidence': float(np.mean(confidences)) if confidences else 0.0,
            'genres': sorted(list(data['genres'])),
            
            # Sample tracks (dla debugowania)
            'sample_tracks': data['track_ids'][:5],
            
            # Embedding dimensions
            'style_embedding_dim': len(style_embedding) if style_embedding else 0,
            'voice_embedding_separated_dim': len(voice_embedding) if voice_embedding else 0,
        }
    
    return result


def print_statistics(artist_embeddings: Dict[str, Any]):
    """Drukuje statystyki"""
    print("\n" + "="*60)
    print("📊 Artist Embeddings Statistics")
    print("="*60)
    
    total_artists = len(artist_embeddings)
    artists_with_separated = sum(
        1 for a in artist_embeddings.values() 
        if a['voice_embedding_separated'] is not None
    )
    
    print(f"\n   Total artists: {total_artists}")
    print(f"   Artists with style_embedding (256-dim): {total_artists}")
    print(f"   Artists with voice_embedding_separated (192-dim): {artists_with_separated}")
    
    # Top artists by track count
    sorted_artists = sorted(
        artist_embeddings.items(),
        key=lambda x: x[1]['track_count'],
        reverse=True
    )
    
    print(f"\n   Top 10 artists by track count:")
    for artist, data in sorted_artists[:10]:
        voice_str = "✅" if data['voice_embedding_separated'] else "❌"
        print(f"     {artist}: {data['track_count']} tracks, "
              f"voice_separated={voice_str}, "
              f"conf={data['avg_vocal_confidence']:.2f}")
    
    # Genre distribution
    genre_counts = defaultdict(int)
    for data in artist_embeddings.values():
        for genre in data['genres']:
            genre_counts[genre] += 1
    
    print(f"\n   Top genres:")
    for genre, count in sorted(genre_counts.items(), key=lambda x: -x[1])[:10]:
        print(f"     {genre}: {count} artists")


def collect_from_vocals_dir(vocals_dir: str) -> Dict[str, Dict[str, Any]]:
    """
    Zbiera embeddingi z folderów vocals/artist/embeddings.json
    
    Args:
        vocals_dir: Ścieżka do folderu vocals/
        
    Returns:
        Słownik artist_embeddings
    """
    vocals_path = Path(vocals_dir)
    if not vocals_path.exists():
        raise FileNotFoundError(f"Vocals directory not found: {vocals_dir}")
    
    result = {}
    
    for artist_dir in vocals_path.iterdir():
        if not artist_dir.is_dir():
            continue
        
        embeddings_file = artist_dir / "embeddings.json"
        if not embeddings_file.exists():
            continue
        
        try:
            with open(embeddings_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            artist_name = data.get('artist', artist_dir.name)
            
            # Kopiuj dane, usuwając _raw_embeddings (nie potrzebne w zbiorczym pliku)
            result[artist_name] = {
                'style_embedding': data.get('style_embedding'),
                'voice_embedding': data.get('voice_embedding'),
                'voice_embedding_separated': data.get('voice_embedding_separated'),
                'track_count': data.get('track_count', 0),
                'tracks_with_separated': data.get('tracks_with_separated', 0),
                'avg_vocal_confidence': data.get('avg_vocal_confidence', 0.0),
                'genres': data.get('genres', []),
                'sample_tracks': list(data.get('tracks', {}).keys())[:5] if 'tracks' in data else [],
                'style_embedding_dim': len(data['style_embedding']) if data.get('style_embedding') else 0,
                'voice_embedding_separated_dim': len(data['voice_embedding_separated']) if data.get('voice_embedding_separated') else 0,
                'source': f"vocals/{artist_dir.name}/embeddings.json",
            }
            
        except Exception as e:
            print(f"   ⚠️ Error loading {embeddings_file}: {e}")
    
    return result


def main():
    parser = argparse.ArgumentParser(
        description='🎤 Generate Artist Embeddings from Dataset v2 or vocals/ folders'
    )
    
    # Źródła danych (jedno z dwóch)
    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument('--dataset', type=str,
                              help='Path to training_dataset_v2.json')
    source_group.add_argument('--from_vocals_dir', type=str,
                              help='Path to vocals/ directory (collects embeddings.json from subfolders)')
    
    parser.add_argument('--output', type=str, 
                        default='./data_v2/artist_embeddings.json',
                        help='Output path for artist_embeddings.json')
    parser.add_argument('--min_tracks', type=int, default=1,
                        help='Minimum tracks per artist (default: 1, only for --dataset)')
    parser.add_argument('--weighting', type=str, default='confidence',
                        choices=['uniform', 'confidence'],
                        help='Embedding weighting method (only for --dataset)')
    
    args = parser.parse_args()
    
    artist_embeddings = {}
    
    if args.from_vocals_dir:
        # Zbierz z folderów vocals/
        print(f"\n📂 Collecting embeddings from: {args.from_vocals_dir}")
        artist_embeddings = collect_from_vocals_dir(args.from_vocals_dir)
        print(f"   Found {len(artist_embeddings)} artists with embeddings")
    else:
        # Z datasetu
        print(f"\n📂 Loading dataset: {args.dataset}")
        dataset = load_dataset(args.dataset)
        
        print(f"   Found {len(dataset.get('tracks', []))} tracks")
        
        # Extract artist data
        print(f"\n🔍 Extracting artist data...")
        artist_data = extract_artist_data(dataset)
        print(f"   Found {len(artist_data)} unique artists with vocals")
        
        # Compute embeddings
        print(f"\n🧮 Computing embeddings (min_tracks={args.min_tracks}, weighting={args.weighting})...")
        artist_embeddings = compute_artist_embeddings(
            artist_data,
            min_tracks=args.min_tracks,
            weighting=args.weighting,
        )
    
    if not artist_embeddings:
        print("❌ No artist embeddings found")
        return
    
    # Print stats
    print_statistics(artist_embeddings)
    
    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(artist_embeddings, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ Saved to {output_path}")
    print(f"   File size: {output_path.stat().st_size / 1024:.1f} KB")
    print(f"\n💡 Usage in inference:")
    print(f"   python inference_v2.py --voice_as 'Artist Name' ...")


if __name__ == "__main__":
    main()
