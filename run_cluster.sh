#!/bin/bash
# =============================================================================
# 🌐 Multi-Server Cluster Dataset Builder
# =============================================================================
# Uruchamia przetwarzanie datasetu na wielu serwerach z automatycznym 
# shardingiem i mergeowaniem wyników.
#
# ARCHITEKTURA:
#   Server 1 (8 GPU) → shardy 0-9   → dataset_shard_0.json ... dataset_shard_9.json
#   Server 2 (8 GPU) → shardy 10-19 → dataset_shard_10.json ... dataset_shard_19.json
#   Server 3 (8 GPU) → shardy 20-29 → ...
#   ...
#   Merge node       → dataset_merged.json
#
# UŻYCIE:
#   1. Na każdym serwerze sklonuj repo i skopiuj dane
#   2. Uruchom z odpowiednimi parametrami:
#
#      # Server 1 (shardy 0-9):
#      ./run_cluster.sh --server_id 0 --total_servers 4 --total_shards 40
#
#      # Server 2 (shardy 10-19):
#      ./run_cluster.sh --server_id 1 --total_servers 4 --total_shards 40
#
#   3. Po zakończeniu wszystkich serwerów, merguj na jednym:
#      ./run_cluster.sh --merge_only --total_shards 40
#
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ============================================================================
# PARSOWANIE ARGUMENTÓW
# ============================================================================
SERVER_ID=""
TOTAL_SERVERS=""
TOTAL_SHARDS=10
AUDIO_DIR="./music"
OUTPUT_DIR="./data_v2"
CHECKPOINT_DIR="./checkpoints"
VOCALS_DIR="./vocals"
BATCH_SIZE="${BATCH_SIZE:-4}"
WHISPER_MODEL="${WHISPER_MODEL:-large-v3}"
LLM_MODEL="${LLM_MODEL:-gpt-4o-mini}"
LLM_CACHE="${LLM_CACHE:-./data_v2/.prompt_cache.json}"
SAMPLE_RATE="${SAMPLE_RATE:-22050}"
METADATA_MAPPING=""
TRACKS_CSV=""
GENRES_CSV=""
MERGE_ONLY=false

# 📋 NASZE METADANE FMA (domyślnie włączone jeśli istnieją!)
OUR_TRACKS_CSV="./data/muz_raw_tracks_mod.csv"
OUR_GENRES_CSV="./data/muz_raw_genres_mod.csv"
SHARD_BY="hash"
NUM_GPUS=""

print_usage() {
    echo "Usage: $0 [options]"
    echo ""
    echo "Required options:"
    echo "  --server_id N          ID tego serwera (0-based)"
    echo "  --total_servers N      Całkowita liczba serwerów"
    echo ""
    echo "Path options:"
    echo "  --audio_dir PATH       Katalog z muzyką (default: ./music)"
    echo "  --output_dir PATH      Katalog wyjściowy (default: ./data_v2)"
    echo "  --checkpoint_dir PATH  Katalog na checkpointy (default: ./checkpoints)"
    echo "  --vocals_dir PATH      Katalog na wokale (default: ./vocals)"
    echo ""
    echo "Metadata options:"
    echo "  --metadata_mapping FILE  Plik JSON/CSV z metadanymi genres"
    echo "  --tracks_csv FILE        FMA tracks.csv (opcjonalnie)"
    echo "  --genres_csv FILE        FMA genres.csv (opcjonalnie)"
    echo ""
    echo "Processing options:"
    echo "  --total_shards N       Całkowita liczba shardów (default: 10)"
    echo "  --num_gpus N           Liczba GPU na tym serwerze (auto-detect)"
    echo "  --shard_by STRATEGY    Strategia shardingu: hash, alphabetical, directory"
    echo "  --batch_size N         Batch size per GPU (default: 4)"
    echo "  --whisper_model MODEL  Model Whisper (default: large-v3)"
    echo ""
    echo "Other options:"
    echo "  --merge_only           Tylko merguj shardy (po zakończeniu wszystkich serwerów)"
    echo "  -h, --help             Pokaż tę pomoc"
    echo ""
    echo "Examples:"
    echo "  # Server 1 of 4 (shardy 0-9):"
    echo "  $0 --server_id 0 --total_servers 4 --total_shards 40 \\"
    echo "     --audio_dir /data/music --metadata_mapping /data/metadata.json"
    echo ""
    echo "  # Server 2 of 4 z FMA metadata:"
    echo "  $0 --server_id 1 --total_servers 4 --total_shards 40 \\"
    echo "     --audio_dir /data/fma_large --tracks_csv /data/fma_metadata/tracks.csv \\"
    echo "     --genres_csv /data/fma_metadata/genres.csv"
    echo ""
    echo "  # Merge all shards after completion:"
    echo "  $0 --merge_only --total_shards 40 --output_dir /data/output"
}

while [[ $# -gt 0 ]]; do
    case $1 in
        --server_id)
            SERVER_ID="$2"
            shift 2
            ;;
        --total_servers)
            TOTAL_SERVERS="$2"
            shift 2
            ;;
        --total_shards)
            TOTAL_SHARDS="$2"
            shift 2
            ;;
        --audio_dir)
            AUDIO_DIR="$2"
            shift 2
            ;;
        --output_dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --checkpoint_dir)
            CHECKPOINT_DIR="$2"
            shift 2
            ;;
        --vocals_dir)
            VOCALS_DIR="$2"
            shift 2
            ;;
        --metadata_mapping)
            METADATA_MAPPING="$2"
            shift 2
            ;;
        --tracks_csv)
            TRACKS_CSV="$2"
            shift 2
            ;;
        --genres_csv)
            GENRES_CSV="$2"
            shift 2
            ;;
        --num_gpus)
            NUM_GPUS="$2"
            shift 2
            ;;
        --batch_size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --whisper_model)
            WHISPER_MODEL="$2"
            shift 2
            ;;
        --merge_only)
            MERGE_ONLY=true
            shift
            ;;
        --shard_by)
            SHARD_BY="$2"
            shift 2
            ;;
        -h|--help)
            print_usage
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            print_usage
            exit 1
            ;;
    esac
done

# ============================================================================
# MERGE ONLY MODE
# ============================================================================
if [ "$MERGE_ONLY" = true ]; then
    echo "=============================================="
    echo "🔗 MERGE MODE - Merging $TOTAL_SHARDS shards"
    echo "=============================================="
    
    # Znajdź wszystkie shardy
    SHARD_FILES=""
    MISSING_SHARDS=()
    
    for i in $(seq 0 $((TOTAL_SHARDS - 1))); do
        SHARD_FILE="$OUTPUT_DIR/dataset_shard_${i}.json"
        if [ -f "$SHARD_FILE" ]; then
            SHARD_FILES="$SHARD_FILES $SHARD_FILE"
        else
            MISSING_SHARDS+=($i)
        fi
    done
    
    if [ ${#MISSING_SHARDS[@]} -gt 0 ]; then
        echo "⚠️  Brakujące shardy: ${MISSING_SHARDS[*]}"
        echo "   Upewnij się że wszystkie serwery zakończyły przetwarzanie"
        read -p "   Kontynuować merge bez brakujących shardów? [y/N]: " CONTINUE
        if [[ ! "$CONTINUE" =~ ^[Yy]$ ]]; then
            exit 1
        fi
    fi
    
    # Merge wszystkich shardów
    python build_dataset_v2.py \
        --merge_shards $SHARD_FILES \
        --output "$OUTPUT_DIR/dataset_merged.json"
    
    echo ""
    echo "=============================================="
    echo "🎉 MERGE COMPLETE!"
    echo "=============================================="
    echo "   Final dataset: $OUTPUT_DIR/dataset_merged.json"
    
    # Statystyki
    if [ -f "$OUTPUT_DIR/dataset_merged.json" ]; then
        TRACK_COUNT=$(python3 -c "import json; d=json.load(open('$OUTPUT_DIR/dataset_merged.json')); print(len(d.get('tracks', [])))" 2>/dev/null || echo "?")
        echo "   Total tracks: $TRACK_COUNT"
    fi
    
    exit 0
fi

# ============================================================================
# WALIDACJA ARGUMENTÓW
# ============================================================================
if [ -z "$SERVER_ID" ] || [ -z "$TOTAL_SERVERS" ]; then
    echo "❌ Wymagane: --server_id i --total_servers"
    print_usage
    exit 1
fi

if [ "$SERVER_ID" -ge "$TOTAL_SERVERS" ]; then
    echo "❌ server_id ($SERVER_ID) musi być mniejszy niż total_servers ($TOTAL_SERVERS)"
    exit 1
fi

# Auto-detect GPU count
if [ -z "$NUM_GPUS" ]; then
    NUM_GPUS=$(nvidia-smi -L 2>/dev/null | wc -l)
    if [ "$NUM_GPUS" -eq 0 ]; then
        echo "❌ Nie wykryto GPU!"
        exit 1
    fi
fi

# Użyj naszych CSV jeśli nie podano innych i istnieją
if [ -z "$TRACKS_CSV" ] && [ -f "$OUR_TRACKS_CSV" ]; then
    TRACKS_CSV="$OUR_TRACKS_CSV"
    echo "📋 Używam naszych metadanych: $TRACKS_CSV"
fi
if [ -z "$GENRES_CSV" ] && [ -f "$OUR_GENRES_CSV" ]; then
    GENRES_CSV="$OUR_GENRES_CSV"
fi

# ============================================================================
# WALIDACJA METADANYCH
# ============================================================================
echo ""
echo "🔍 Sprawdzam metadane..."

METADATA_OK=true

# Sprawdź metadata_mapping
if [ -n "$METADATA_MAPPING" ]; then
    if [ -f "$METADATA_MAPPING" ]; then
        MAPPING_SIZE=$(wc -l < "$METADATA_MAPPING" 2>/dev/null || echo "?")
        echo "   ✅ metadata_mapping: $METADATA_MAPPING ($MAPPING_SIZE lines)"
    else
        echo "   ❌ metadata_mapping NIE ISTNIEJE: $METADATA_MAPPING"
        METADATA_OK=false
    fi
else
    echo "   ⚠️  metadata_mapping: nie podano (użyje ID3 tags lub nazw folderów)"
fi

# Sprawdź FMA CSV
if [ -n "$TRACKS_CSV" ]; then
    if [ -f "$TRACKS_CSV" ]; then
        echo "   ✅ tracks_csv: $TRACKS_CSV"
    else
        echo "   ❌ tracks_csv NIE ISTNIEJE: $TRACKS_CSV"
        METADATA_OK=false
    fi
fi

if [ -n "$GENRES_CSV" ]; then
    if [ -f "$GENRES_CSV" ]; then
        echo "   ✅ genres_csv: $GENRES_CSV"
    else
        echo "   ❌ genres_csv NIE ISTNIEJE: $GENRES_CSV"
        METADATA_OK=false
    fi
fi

# Jeśli brak jakichkolwiek metadanych, ostrzeż
if [ -z "$METADATA_MAPPING" ] && [ -z "$TRACKS_CSV" ]; then
    echo ""
    echo "   ⚠️  UWAGA: Brak plików metadanych!"
    echo "   Skrypt spróbuje wyciągnąć metadane z:"
    echo "   1. ID3 tags (jeśli MP3 mają wypełnione tagi)"
    echo "   2. Nazw folderów (jako fallback dla artysty)"
    echo ""
    read -p "   Kontynuować bez zewnętrznych metadanych? [y/N]: " CONTINUE
    if [[ ! "$CONTINUE" =~ ^[Yy]$ ]]; then
        echo ""
        echo "   Podaj metadane używając:"
        echo "   --metadata_mapping /path/to/metadata.json"
        echo "   lub"
        echo "   --tracks_csv /path/to/tracks.csv --genres_csv /path/to/genres.csv"
        exit 1
    fi
fi

if [ "$METADATA_OK" = false ]; then
    echo ""
    echo "❌ Niektóre pliki metadanych nie istnieją!"
    exit 1
fi

# Oblicz zakres shardów dla tego serwera
SHARDS_PER_SERVER=$((TOTAL_SHARDS / TOTAL_SERVERS))
REMAINDER=$((TOTAL_SHARDS % TOTAL_SERVERS))

# Dodaj extra shard dla pierwszych serwerów jeśli nie dzieli się równo
if [ "$SERVER_ID" -lt "$REMAINDER" ]; then
    SHARDS_PER_SERVER=$((SHARDS_PER_SERVER + 1))
    START_SHARD=$((SERVER_ID * SHARDS_PER_SERVER))
else
    START_SHARD=$((SERVER_ID * SHARDS_PER_SERVER + REMAINDER))
fi

END_SHARD=$((START_SHARD + SHARDS_PER_SERVER - 1))

echo ""
echo "=============================================="
echo "🌐 Cluster Dataset Builder"
echo "=============================================="
echo "   Server: $SERVER_ID of $TOTAL_SERVERS"
echo "   GPUs on this server: $NUM_GPUS"
echo "   Total shards: $TOTAL_SHARDS"
echo "   This server's shards: $START_SHARD - $END_SHARD ($SHARDS_PER_SERVER shards)"
echo "   Shard strategy: $SHARD_BY"
echo ""
echo "   Paths:"
echo "     Audio dir: $AUDIO_DIR"
echo "     Output dir: $OUTPUT_DIR"
echo "     Checkpoint dir: $CHECKPOINT_DIR"
echo "     Vocals dir: $VOCALS_DIR"
echo ""
echo "   Metadata:"
echo "     metadata_mapping: ${METADATA_MAPPING:-<none>}"
echo "     tracks_csv: ${TRACKS_CSV:-<none>}"
echo "     genres_csv: ${GENRES_CSV:-<none>}"
echo ""
echo "   Processing:"
echo "     Batch size: $BATCH_SIZE"
echo "     Whisper model: $WHISPER_MODEL"
echo "=============================================="

# Sprawdź katalog audio
if [ ! -d "$AUDIO_DIR" ]; then
    echo "❌ Katalog audio nie istnieje: $AUDIO_DIR"
    exit 1
fi

# Policz pliki
TOTAL_FILES=$(find "$AUDIO_DIR" -type f \( -name "*.mp3" -o -name "*.wav" -o -name "*.flac" -o -name "*.ogg" \) | wc -l)
echo "   Total audio files in dir: $TOTAL_FILES"
echo "   Estimated files per shard: ~$((TOTAL_FILES / TOTAL_SHARDS))"

# Utwórz katalogi
mkdir -p "$OUTPUT_DIR"
mkdir -p "$CHECKPOINT_DIR"
mkdir -p "$VOCALS_DIR"

# ============================================================================
# URUCHOM SHARDY RÓWNOLEGLE NA GPU
# ============================================================================
echo ""
echo "🚀 Starting $SHARDS_PER_SERVER shards on $NUM_GPUS GPUs..."

PIDS=()
SHARD_IDS=()

# Przypisz shardy do GPU (round-robin)
GPU_IDX=0
for SHARD_IDX in $(seq $START_SHARD $END_SHARD); do
    SHARD_IDS+=($SHARD_IDX)
    
    RUN_NAME="server${SERVER_ID}_shard${SHARD_IDX}"
    LOG_FILE="$CHECKPOINT_DIR/${RUN_NAME}.log"
    
    echo "   🚀 Shard $SHARD_IDX on GPU $GPU_IDX -> $RUN_NAME"
    
    # Buduj extra args dla metadanych
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
    # Wokale, lyrics, G2P, CLAP, F0 są ZAWSZE włączone
    #
    CUDA_VISIBLE_DEVICES=$GPU_IDX python build_dataset_v2.py \
        --audio_dir "$AUDIO_DIR" \
        --output "$OUTPUT_DIR/dataset_shard_${SHARD_IDX}.json" \
        --checkpoint_dir "$CHECKPOINT_DIR" \
        --run_name "$RUN_NAME" \
        --vocals_output_dir "$VOCALS_DIR" \
        --device cuda \
        --batch_size "$BATCH_SIZE" \
        --sample_rate "$SAMPLE_RATE" \
        --whisper_model "$WHISPER_MODEL" \
        --llm_model "$LLM_MODEL" \
        --llm_cache "$LLM_CACHE" \
        --shard_index "$SHARD_IDX" \
        --total_shards "$TOTAL_SHARDS" \
        --shard_by "$SHARD_BY" \
        --with_segments \
        --extract_features \
        $METADATA_ARGS \
        > "$LOG_FILE" 2>&1 &
    
    PIDS+=($!)
    
    # Następne GPU (round-robin)
    GPU_IDX=$(( (GPU_IDX + 1) % NUM_GPUS ))
    
    # Pauza między startami żeby nie załadować wszystkich modeli naraz
    sleep 10
done

echo ""
echo "✅ Started ${#PIDS[@]} PRODUCTION shard processes"
echo "   PIDs: ${PIDS[*]}"
echo "   Shards: ${SHARD_IDS[*]}"

# ============================================================================
# MONITORING
# ============================================================================
echo ""
echo "📊 Monitoring progress..."

monitor_progress() {
    while true; do
        clear
        echo "=============================================="
        echo "🌐 Server $SERVER_ID Progress - $(date)"
        echo "=============================================="
        
        ALL_DONE=true
        COMPLETED=0
        FAILED=0
        
        for i in "${!SHARD_IDS[@]}"; do
            SHARD_IDX="${SHARD_IDS[$i]}"
            PID="${PIDS[$i]}"
            LOG_FILE="$CHECKPOINT_DIR/server${SERVER_ID}_shard${SHARD_IDX}.log"
            
            if kill -0 "$PID" 2>/dev/null; then
                STATUS="🔄 Running"
                ALL_DONE=false
            else
                wait "$PID" 2>/dev/null
                EXIT_CODE=$?
                if [ "$EXIT_CODE" -eq 0 ]; then
                    STATUS="✅ Done"
                    ((COMPLETED++))
                else
                    STATUS="❌ Failed ($EXIT_CODE)"
                    ((FAILED++))
                fi
            fi
            
            # Progress z logu
            if [ -f "$LOG_FILE" ]; then
                PROCESSED=$(grep -c "Processed\|Processing" "$LOG_FILE" 2>/dev/null || echo "0")
            else
                PROCESSED="0"
            fi
            
            echo "   Shard $SHARD_IDX: $STATUS (tracks: ~$PROCESSED)"
        done
        
        echo ""
        echo "   Completed: $COMPLETED / ${#SHARD_IDS[@]}"
        if [ $FAILED -gt 0 ]; then
            echo "   ⚠️ Failed: $FAILED"
        fi
        echo ""
        echo "   Logs: $CHECKPOINT_DIR/server${SERVER_ID}_shard*.log"
        
        if $ALL_DONE; then
            echo ""
            if [ $FAILED -eq 0 ]; then
                echo "🎉 All shards completed successfully!"
            else
                echo "⚠️ Some shards failed. Check logs."
            fi
            break
        fi
        
        sleep 15
    done
}

trap "echo ''; echo 'Monitoring stopped.'; exit 0" INT
monitor_progress

# ============================================================================
# PODSUMOWANIE
# ============================================================================
echo ""
echo "=============================================="
echo "📊 Server $SERVER_ID Complete"
echo "=============================================="
echo "   Shards processed: $START_SHARD - $END_SHARD"
echo "   Output files:"
for SHARD_IDX in $(seq $START_SHARD $END_SHARD); do
    SHARD_FILE="$OUTPUT_DIR/dataset_shard_${SHARD_IDX}.json"
    if [ -f "$SHARD_FILE" ]; then
        SIZE=$(du -h "$SHARD_FILE" | cut -f1)
        echo "     ✅ dataset_shard_${SHARD_IDX}.json ($SIZE)"
    else
        echo "     ❌ dataset_shard_${SHARD_IDX}.json (missing!)"
    fi
done
echo ""
echo "📋 Next steps:"
echo "   1. Skopiuj pliki dataset_shard_*.json na merge node"
echo "   2. Po zakończeniu WSZYSTKICH serwerów, uruchom:"
echo "      ./run_cluster.sh --merge_only --total_shards $TOTAL_SHARDS"
