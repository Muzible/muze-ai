#!/bin/bash
# =============================================================================
# 🚀 PRODUCTION Multi-GPU Dataset Builder
# =============================================================================
# Uruchamia równolegle N procesów na N GPU dla PEŁNEJ ekstrakcji datasetu.
#
# ⚠️  TO JEST WERSJA PRODUKCYJNA - wszystkie opcje ekstrakcji są WŁĄCZONE!
#
# 🎵 CO JEST EKSTRAHOWANE:
#   ✅ Segmenty (verse/chorus/bridge/intro/outro)
#   ✅ Audio features (tempo, key, energy, MFCC, chroma, spektralne)
#   ✅ Beat grid (pozycje uderzeń, downbeaty, metrum)
#   ✅ Chord progression
#   ✅ Vocal separation (Demucs → zapis do osobnych plików WAV per artysta)
#   ✅ Voice embeddings:
#       - Resemblyzer 256-dim (dla style transfer "w stylu X")
#       - ECAPA-TDNN 192-dim (dla voice cloning "jak X")
#   ✅ Lyrics transcription (Whisper large-v3, timestamps per word)
#   ✅ G2P phonemes (IPA, timestamps)
#   ✅ F0/pitch contour
#   ✅ Vibrato analysis (rate, depth, extent)
#   ✅ Breath detection
#   ✅ Sentiment analysis
#   ✅ CLAP embeddings (512-dim audio-text)
#   ✅ LLM-enhanced prompts (GPT-4o-mini)
#
# 📦 WYMAGANIA:
#   - NVIDIA GPU(s) z CUDA
#   - ~16GB VRAM per GPU (dla batch_size=4)
#   - ~50GB RAM per 8 GPU
#   - Pakiety: torch, torchaudio, demucs, whisper, resemblyzer, speechbrain,
#              transformers, phonemizer, librosa, soundfile
#   - OpenAI API key (dla LLM promptów): export OPENAI_API_KEY=...
#
# 🔧 UŻYCIE:
#   ./run_multi_gpu.sh              # Auto-detect GPU count
#   ./run_multi_gpu.sh 8            # Użyj 8 GPU
#   ./run_multi_gpu.sh 8 /data/music  # 8 GPU, własny katalog audio
#
# 📋 PRZYKŁADY DLA 8 GPU:
#
#   # Podstawowe (z domyślnym ./music):
#   ./run_multi_gpu.sh 8
#
#   # Z własnym katalogiem muzyki:
#   ./run_multi_gpu.sh 8 /data/fma_large
#
#   # Z metadanymi FMA:
#   TRACKS_CSV=/data/fma_metadata/tracks.csv \
#   GENRES_CSV=/data/fma_metadata/genres.csv \
#   ./run_multi_gpu.sh 8 /data/fma_large
#
#   # Z własnym plikiem metadanych:
#   METADATA_MAPPING=./my_metadata.json ./run_multi_gpu.sh 8 /data/music
#
#   # Większy batch size (wymaga więcej VRAM):
#   BATCH_SIZE=8 ./run_multi_gpu.sh 8 /data/music
#
#   # Szybki start (bez sprawdzania pakietów):
#   SKIP_CHECKS=1 ./run_multi_gpu.sh 8
#
#   # Pre-scan metadanych (sprawdź tagi ID3 przed budowaniem):
#   PRE_SCAN=1 ./run_multi_gpu.sh 8 /data/music
#
# 🔧 ZMIENNE ŚRODOWISKOWE:
#   BATCH_SIZE=4           # Batch size per GPU (domyślnie 4, więcej = więcej VRAM)
#   WHISPER_MODEL=large-v3 # Model Whisper (large-v3 najlepszy dla PL!)
#   LLM_MODEL=gpt-4o-mini  # Model LLM do generowania promptów
#   SAMPLE_RATE=22050      # Sample rate
#   METADATA_MAPPING=...   # Plik JSON/CSV z metadanymi (artysta, gatunek)
#   TRACKS_CSV=...         # FMA tracks.csv (domyślnie: data/muz_raw_tracks_mod.csv)
#   GENRES_CSV=...         # FMA genres.csv (domyślnie: data/muz_raw_genres_mod.csv)
#   OUTPUT_DIR=./data_v2   # Katalog wyjściowy
#   VOCALS_DIR=./vocals    # Katalog na separowane wokale
#   SKIP_CHECKS=1          # Pomiń sprawdzanie pakietów
#   PRE_SCAN=1             # Pre-scan tagów ID3 przed budowaniem
#
# 📋 NASZE DANE (domyślnie używane jeśli istnieją):
#   data/muz_raw_tracks_mod.csv  - metadane tracków FMA
#   data/muz_raw_genres_mod.csv  - taksonomia gatunków
#
# 📊 SZACOWANY CZAS (8x RTX 4090):
#   1000 tracków: ~20 minut
#   10000 tracków: ~3.5 godziny
#   100000 tracków: ~35 godzin
#
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ============================================================================
# SPRAWDZENIE WYMAGANYCH PAKIETÓW
# ============================================================================
check_requirements() {
    echo ""
    echo "🔍 Sprawdzam wymagane pakiety..."
    
    MISSING_SYSTEM=()
    MISSING_PYTHON=()
    
    # Sprawdź zależności systemowe
    echo "   Sprawdzam zależności systemowe..."
    
    # FFmpeg
    if ! command -v ffmpeg &> /dev/null; then
        MISSING_SYSTEM+=("ffmpeg")
    else
        echo "   ✅ ffmpeg: $(ffmpeg -version 2>&1 | head -1)"
    fi
    
    # espeak-ng (dla G2P - polskie fonemy!)
    if ! command -v espeak-ng &> /dev/null; then
        MISSING_SYSTEM+=("espeak-ng")
    else
        echo "   ✅ espeak-ng: $(espeak-ng --version 2>&1 | head -1)"
    fi
    
    # nvidia-smi (dla GPU)
    if ! command -v nvidia-smi &> /dev/null; then
        echo "   ⚠️  nvidia-smi: nie znaleziono (wymagane dla GPU)"
    else
        echo "   ✅ nvidia-smi: $(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)"
    fi
    
    # Sprawdź pakiety Python
    echo ""
    echo "   Sprawdzam pakiety Python..."
    
    REQUIRED_PACKAGES=(
        "torch:PyTorch"
        "torchaudio:TorchAudio"
        "librosa:Librosa"
        "whisper:OpenAI Whisper"
        "demucs:Facebook Demucs"
        "resemblyzer:Resemblyzer"
        "phonemizer:Phonemizer (G2P)"
        "transformers:Transformers (CLAP)"
        "pandas:Pandas"
        "mutagen:Mutagen (ID3)"
        "openai:OpenAI API"
        "soundfile:SoundFile"
    )
    
    for pkg_info in "${REQUIRED_PACKAGES[@]}"; do
        IFS=':' read -r pkg_name pkg_desc <<< "$pkg_info"
        if python3 -c "import $pkg_name" 2>/dev/null; then
            VERSION=$(python3 -c "import $pkg_name; print(getattr($pkg_name, '__version__', 'ok'))" 2>/dev/null || echo "ok")
            echo "   ✅ $pkg_desc ($pkg_name): $VERSION"
        else
            MISSING_PYTHON+=("$pkg_name")
            echo "   ❌ $pkg_desc ($pkg_name): NIE ZNALEZIONO"
        fi
    done
    
    # Sprawdź CUDA
    echo ""
    echo "   Sprawdzam CUDA..."
    CUDA_AVAILABLE=$(python3 -c "import torch; print(torch.cuda.is_available())" 2>/dev/null || echo "False")
    if [ "$CUDA_AVAILABLE" = "True" ]; then
        CUDA_VERSION=$(python3 -c "import torch; print(torch.version.cuda)" 2>/dev/null || echo "?")
        GPU_COUNT=$(python3 -c "import torch; print(torch.cuda.device_count())" 2>/dev/null || echo "0")
        echo "   ✅ CUDA: $CUDA_VERSION ($GPU_COUNT GPU dostępnych)"
    else
        echo "   ⚠️  CUDA: niedostępne (będzie WOLNO!)"
    fi
    
    # Raport końcowy
    echo ""
    
    if [ ${#MISSING_SYSTEM[@]} -gt 0 ]; then
        echo "❌ Brakujące zależności systemowe:"
        for pkg in "${MISSING_SYSTEM[@]}"; do
            echo "   - $pkg"
        done
        echo ""
        echo "   Instalacja (macOS):  brew install ${MISSING_SYSTEM[*]}"
        echo "   Instalacja (Ubuntu): sudo apt install ${MISSING_SYSTEM[*]}"
        echo ""
    fi
    
    if [ ${#MISSING_PYTHON[@]} -gt 0 ]; then
        echo "❌ Brakujące pakiety Python:"
        for pkg in "${MISSING_PYTHON[@]}"; do
            echo "   - $pkg"
        done
        echo ""
        echo "   Instalacja: pip install -r requirements.txt"
        echo ""
    fi
    
    if [ ${#MISSING_SYSTEM[@]} -gt 0 ] || [ ${#MISSING_PYTHON[@]} -gt 0 ]; then
        echo "=============================================="
        echo "⚠️  BRAKUJĄ WYMAGANE PAKIETY!"
        echo "=============================================="
        read -p "   Kontynuować mimo to? [y/N]: " CONTINUE
        if [[ ! "$CONTINUE" =~ ^[Yy]$ ]]; then
            echo "   Przerwano. Zainstaluj brakujące pakiety."
            exit 1
        fi
    else
        echo "✅ Wszystkie wymagane pakiety są zainstalowane!"
    fi
}

# Sprawdź pakiety (pomiń jeśli SKIP_CHECKS=1)
if [ "${SKIP_CHECKS:-0}" != "1" ]; then
    check_requirements
fi

# ============================================================================
# KONFIGURACJA PRODUKCYJNA - PEŁNA EKSTRAKCJA
# ============================================================================
# 🎵 Ścieżki
AUDIO_DIR="${2:-./music}"                          # Katalog z muzyką
OUTPUT_DIR="${OUTPUT_DIR:-./data_v2}"              # Katalog wyjściowy
CHECKPOINT_DIR="${CHECKPOINT_DIR:-./checkpoints}"  # Katalog na checkpointy
VOCALS_DIR="${VOCALS_DIR:-./vocals}"               # Katalog na separowane wokale (per artysta)
LLM_CACHE="${LLM_CACHE:-./data_v2/.prompt_cache.json}"  # Cache dla promptów LLM

# 🔧 Parametry GPU
BATCH_SIZE="${BATCH_SIZE:-4}"                      # Batch size per GPU (4 dla 24GB VRAM)
WHISPER_MODEL="${WHISPER_MODEL:-large-v3}"         # Najlepszy model Whisper (dla PL!)
LLM_MODEL="${LLM_MODEL:-gpt-4o-mini}"              # Model LLM do generowania promptów
SAMPLE_RATE="${SAMPLE_RATE:-22050}"                # Sample rate (22050 = standard)

# 📋 NASZE METADANE FMA (domyślnie włączone jeśli istnieją!)
# Priorytet: argumenty użytkownika > nasze CSV > auto-detect
OUR_TRACKS_CSV="./data/muz_raw_tracks_mod.csv"
OUR_GENRES_CSV="./data/muz_raw_genres_mod.csv"

# Użyj naszych CSV jeśli nie podano innych i istnieją
if [ -z "$TRACKS_CSV" ] && [ -f "$OUR_TRACKS_CSV" ]; then
    TRACKS_CSV="$OUR_TRACKS_CSV"
    echo "📋 Używam naszych metadanych: $TRACKS_CSV"
fi
if [ -z "$GENRES_CSV" ] && [ -f "$OUR_GENRES_CSV" ]; then
    GENRES_CSV="$OUR_GENRES_CSV"
fi

# Opcjonalne zewnętrzne metadane
METADATA_MAPPING="${METADATA_MAPPING:-}"           # Plik JSON/CSV z metadanymi (nadpisuje CSV)

# 🎤 Opcje ekstrakcji (WSZYSTKIE WŁĄCZONE dla produkcji)
# Te flagi są domyślnie TRUE w build_dataset_v2.py:
#   --with_segments        ✅ Detekcja sekcji (verse/chorus/bridge)
#   --extract_features     ✅ Pełna ekstrakcja cech audio
# Te są zawsze uruchamiane:
#   - Demucs vocal separation
#   - Whisper lyrics extraction  
#   - G2P phoneme conversion
#   - Voice embeddings (Resemblyzer 256-dim + ECAPA-TDNN 192-dim)
#   - CLAP embeddings (512-dim audio-text)
#   - F0/pitch extraction
#   - Vibrato analysis
#   - Breath detection
#   - Sentiment analysis
#   - LLM prompt generation

# Auto-detect liczby GPU (lub użyj argumentu)
if [ -n "$1" ]; then
    NUM_GPUS=$1
else
    NUM_GPUS=$(nvidia-smi -L 2>/dev/null | wc -l)
    if [ "$NUM_GPUS" -eq 0 ]; then
        echo "❌ Nie wykryto GPU! Użyj: ./run_multi_gpu.sh <num_gpus>"
        exit 1
    fi
fi

echo "============================================================================" 
echo "🚀 PRODUCTION Multi-GPU Dataset Builder"
echo "============================================================================" 
echo ""
echo "📁 PATHS:"
echo "   Audio dir:      $AUDIO_DIR"
echo "   Output dir:     $OUTPUT_DIR"
echo "   Checkpoint dir: $CHECKPOINT_DIR"
echo "   Vocals dir:     $VOCALS_DIR"
echo "   LLM cache:      $LLM_CACHE"
echo ""
echo "🔧 GPU CONFIG:"
echo "   GPUs:              $NUM_GPUS"
echo "   Batch size/GPU:    $BATCH_SIZE"
echo "   Sample rate:       $SAMPLE_RATE Hz"
echo ""
echo "🎤 MODELS:"
echo "   Whisper:           $WHISPER_MODEL (lyrics + timestamps)"
echo "   LLM:               $LLM_MODEL (prompt generation)"
echo "   Demucs:            htdemucs (vocal separation)"
echo "   Voice Encoder 1:   Resemblyzer (256-dim, style transfer)"
echo "   Voice Encoder 2:   ECAPA-TDNN (192-dim, voice cloning)"
echo "   CLAP:              LAION (512-dim, audio-text)"
echo ""
echo "📋 METADATA:"
echo "   metadata_mapping:  ${METADATA_MAPPING:-<none>}"
echo "   tracks_csv:        ${TRACKS_CSV:-<none>}"
echo "   genres_csv:        ${GENRES_CSV:-<none>}"
echo ""
echo "🎵 EXTRACTION (ALL ENABLED):"
echo "   ✅ Segment detection (verse/chorus/bridge/intro/outro)"
echo "   ✅ Audio features (tempo, key, energy, MFCC, chroma, beats)"
echo "   ✅ Vocal separation (Demucs → per-artist WAV export)"
echo "   ✅ Voice embeddings (style_of + voice_clone)"
echo "   ✅ Lyrics transcription (Whisper large-v3)"
echo "   ✅ G2P phonemes (IPA + timestamps)"
echo "   ✅ F0/pitch contour extraction"
echo "   ✅ Vibrato analysis (rate, depth, extent)"
echo "   ✅ Breath detection"
echo "   ✅ Sentiment analysis"
echo "   ✅ CLAP embeddings (audio-text)"
echo "   ✅ LLM prompt enhancement (GPT-4o-mini)"
echo ""
echo "============================================================================"

# ============================================================================
# PRE-SCAN METADANYCH (opcjonalny)
# ============================================================================
# Uruchom z PRE_SCAN=1 żeby sprawdzić ID3 tagi PRZED budowaniem
# Generuje CSV do ręcznego uzupełnienia genre/artist

if [ "${PRE_SCAN:-0}" = "1" ]; then
    echo ""
    echo "🔍 PRE-SCAN: Sprawdzam tagi ID3 w plikach audio..."
    echo ""
    
    PRE_SCAN_OUTPUT="./output/pre_scan_metadata.csv"
    PRE_SCAN_MISSING="./output/to_complete_metadata.csv"
    
    mkdir -p ./output
    
    python tools_v2/scan_mp3_folder.py \
        --input_dir "$AUDIO_DIR" \
        --output "$PRE_SCAN_OUTPUT" \
        --export_missing "$PRE_SCAN_MISSING"
    
    # Sprawdź czy są pliki do uzupełnienia
    if [ -f "$PRE_SCAN_MISSING" ]; then
        MISSING_COUNT=$(tail -n +2 "$PRE_SCAN_MISSING" | wc -l | tr -d ' ')
        if [ "$MISSING_COUNT" -gt 0 ]; then
            echo ""
            echo "============================================================================"
            echo "⚠️  BRAKUJĄCE METADANE: $MISSING_COUNT plików bez genre/artist"
            echo "============================================================================"
            echo ""
            echo "📝 Plik do uzupełnienia: $PRE_SCAN_MISSING"
            echo ""
            echo "   OPCJE:"
            echo "   1. Uzupełnij CSV ręcznie i uruchom:"
            echo "      METADATA_MAPPING=$PRE_SCAN_MISSING ./run_multi_gpu.sh $NUM_GPUS $AUDIO_DIR"
            echo ""
            echo "   2. Kontynuuj bez uzupełniania (LLM spróbuje zgadnąć genre)"
            echo ""
            read -p "Kontynuować budowanie bez uzupełnienia? [y/N]: " CONTINUE
            if [ "$CONTINUE" != "y" ] && [ "$CONTINUE" != "Y" ]; then
                echo ""
                echo "📋 Uzupełnij plik: $PRE_SCAN_MISSING"
                echo "   Następnie uruchom: METADATA_MAPPING=$PRE_SCAN_MISSING ./run_multi_gpu.sh $NUM_GPUS"
                exit 0
            fi
        fi
    fi
    echo ""
fi

if [ ! -d "$AUDIO_DIR" ]; then
    echo "❌ Katalog audio nie istnieje: $AUDIO_DIR"
    exit 1
fi

# Policz pliki audio
TOTAL_FILES=$(find "$AUDIO_DIR" -type f \( -name "*.mp3" -o -name "*.wav" -o -name "*.flac" -o -name "*.ogg" \) | wc -l)
echo "   Total audio files: $TOTAL_FILES"

if [ "$TOTAL_FILES" -eq 0 ]; then
    echo "❌ Brak plików audio w $AUDIO_DIR"
    exit 1
fi

# ============================================================================
# INFO O METADANYCH
# ============================================================================
# Metadane są automatycznie wykrywane przez build_dataset_v2.py:
#   1. ID3 tags z plików MP3 (artist, genre, title)
#   2. FMA CSV jeśli podano --tracks_csv i --genres_csv
#   3. Nazwy folderów jako fallback dla artysty
#   4. LLM (GPT-4o-mini) generuje prompty z kontekstem
#
# Jeśli chcesz podać własne metadane, użyj zmiennych środowiskowych:
#   METADATA_MAPPING=./my_metadata.json ./run_multi_gpu.sh 8
#   TRACKS_CSV=./tracks.csv GENRES_CSV=./genres.csv ./run_multi_gpu.sh 8

if [ -z "$METADATA_MAPPING" ] && [ -z "$TRACKS_CSV" ]; then
    echo ""
    echo "📋 Metadane będą wykrywane automatycznie:"
    echo "   1. ID3 tags z plików audio"
    echo "   2. Nazwy folderów (jako fallback dla artysty)"
    echo "   3. LLM prompt generation (uzupełnia brakujące info)"
    echo ""
fi

# Utwórz katalogi
mkdir -p "$OUTPUT_DIR"
mkdir -p "$CHECKPOINT_DIR"
mkdir -p "$VOCALS_DIR"

# ============================================================================
# URUCHOM PROCESY NA KAŻDYM GPU (PEŁNA PRODUKCYJNA EKSTRAKCJA)
# ============================================================================
echo ""
echo "🔄 Starting $NUM_GPUS parallel PRODUCTION processes..."
echo ""

PIDS=()
RUN_NAMES=()

for i in $(seq 0 $((NUM_GPUS - 1))); do
    RUN_NAME="gpu${i}_$(date +%Y%m%d_%H%M%S)"
    RUN_NAMES+=("$RUN_NAME")
    
    LOG_FILE="$CHECKPOINT_DIR/${RUN_NAME}.log"
    
    echo "   🚀 GPU $i -> $RUN_NAME (log: $LOG_FILE)"
    
    # Buduj argumenty dla metadanych (opcjonalne)
    METADATA_ARGS=""
    if [ -n "$METADATA_MAPPING" ]; then
        METADATA_ARGS="$METADATA_ARGS --metadata_mapping $METADATA_MAPPING"
    fi
    if [ -n "$TRACKS_CSV" ]; then
        METADATA_ARGS="$METADATA_ARGS --tracks_csv $TRACKS_CSV"
    fi
    if [ -n "$GENRES_CSV" ]; then
        METADATA_ARGS="$METADATA_ARGS --genres_csv $GENRES_CSV"
    fi
    
    # ========================================
    # 🎵 PEŁNE PRODUKCYJNE WYWOŁANIE
    # ========================================
    # Wszystkie flagi ekstrakcji włączone:
    #   --with_segments    = detekcja sekcji (verse/chorus/bridge)
    #   --extract_features = pełna ekstrakcja audio features
    # Wokale, lyrics, G2P, CLAP, F0 są ZAWSZE włączone w build_dataset_v2.py
    # 
    CUDA_VISIBLE_DEVICES=$i python build_dataset_v2.py \
        --audio_dir "$AUDIO_DIR" \
        --output "$OUTPUT_DIR/dataset_${RUN_NAME}.json" \
        --checkpoint_dir "$CHECKPOINT_DIR" \
        --run_name "$RUN_NAME" \
        --vocals_output_dir "$VOCALS_DIR" \
        --device cuda \
        --batch_size "$BATCH_SIZE" \
        --sample_rate "$SAMPLE_RATE" \
        --whisper_model "$WHISPER_MODEL" \
        --llm_model "$LLM_MODEL" \
        --llm_cache "$LLM_CACHE" \
        --with_segments \
        --extract_features \
        $METADATA_ARGS \
        > "$LOG_FILE" 2>&1 &
    
    PIDS+=($!)
    
    # Krótka pauza żeby nie załadować wszystkich modeli naraz (10s dla bezpieczeństwa)
    sleep 10
done

echo ""
echo "✅ All $NUM_GPUS PRODUCTION processes started!"
echo "   PIDs: ${PIDS[*]}"
echo ""

# ============================================================================
# MONITORUJ POSTĘP
# ============================================================================
echo "📊 Monitoring progress (Ctrl+C to stop monitoring, processes continue)..."
echo ""

monitor_progress() {
    while true; do
        clear
        echo "=============================================="
        echo "🚀 Multi-GPU Progress - $(date)"
        echo "=============================================="
        
        ALL_DONE=true
        
        for i in $(seq 0 $((NUM_GPUS - 1))); do
            RUN_NAME="${RUN_NAMES[$i]}"
            PID="${PIDS[$i]}"
            LOG_FILE="$CHECKPOINT_DIR/${RUN_NAME}.log"
            
            # Sprawdź czy proces żyje
            if kill -0 "$PID" 2>/dev/null; then
                STATUS="🔄 Running"
                ALL_DONE=false
            else
                wait "$PID" 2>/dev/null
                EXIT_CODE=$?
                if [ "$EXIT_CODE" -eq 0 ]; then
                    STATUS="✅ Done"
                else
                    STATUS="❌ Failed (exit $EXIT_CODE)"
                fi
            fi
            
            # Policz przetworzone tracki z logu
            if [ -f "$LOG_FILE" ]; then
                PROCESSED=$(grep -c "Processing\|Processed" "$LOG_FILE" 2>/dev/null || echo "0")
            else
                PROCESSED="0"
            fi
            
            echo "   GPU $i [$RUN_NAME]: $STATUS (tracks: ~$PROCESSED)"
        done
        
        echo ""
        echo "   Logs: $CHECKPOINT_DIR/*.log"
        echo "   Press Ctrl+C to stop monitoring"
        
        if $ALL_DONE; then
            echo ""
            echo "🎉 All processes completed!"
            break
        fi
        
        sleep 10
    done
}

# Uruchom monitoring (można przerwać Ctrl+C)
trap "echo ''; echo 'Monitoring stopped. Processes continue in background.'; exit 0" INT
monitor_progress

# ============================================================================
# MERGE WYNIKÓW
# ============================================================================
echo ""
echo "=============================================="
echo "🔗 Merging results from all GPUs..."
echo "=============================================="

# Buduj listę run_names do merge
MERGE_ARGS=""
for RUN_NAME in "${RUN_NAMES[@]}"; do
    MERGE_ARGS="$MERGE_ARGS $RUN_NAME"
done

# Merge wszystkich runów
python build_dataset_v2.py \
    --merge \
    --checkpoint_dir "$CHECKPOINT_DIR" \
    --merge_runs $MERGE_ARGS \
    --output "$OUTPUT_DIR/dataset_merged.json"

echo ""
echo "=============================================="
echo "🎉 DONE!"
echo "=============================================="
echo "   Final dataset: $OUTPUT_DIR/dataset_merged.json"
echo "   Individual runs: $OUTPUT_DIR/dataset_gpu*.json"
echo "   Checkpoints: $CHECKPOINT_DIR/"
echo "=============================================="

# Pokaż statystyki końcowe
if [ -f "$OUTPUT_DIR/dataset_merged.json" ]; then
    TRACK_COUNT=$(python -c "import json; d=json.load(open('$OUTPUT_DIR/dataset_merged.json')); print(len(d.get('tracks', [])))" 2>/dev/null || echo "?")
    echo ""
    echo "📊 Final statistics:"
    echo "   Total tracks: $TRACK_COUNT"
fi
