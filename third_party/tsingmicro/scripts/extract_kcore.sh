#!/bin/bash
# Extract the most recent KCORE log append for a given chip/tile.

set -euo pipefail

print_help() {
    cat <<EOF
用法: $(basename "$0") [options] [tile_list]
      $(basename "$0") -h|--help

从 KCORE 日志中提取指定 tile 最近一次 append 的日志信息。

参数:
  tile_list   逗号分隔的 tile ID，支持范围和 "all"。
              示例: "0" "0,1,2" "0-5,9,11" "all" (等价于 0-15)。
              未指定时默认值为 0。

选项:
  -s <file>   提取完成后，将 kcore_log 中的 t* 文件复制到
              kcore_log/{id:06d}_<file> 目录下，{id} 顺次递增。

环境变量:
  TXDA_VISIBLE_DEVICES   芯片号 (默认 10)。

输出:
  kcore_log/chip{芯片}_tile{tile}   二进制日志片段。
  kcore_log/t{tile}                 strings 可读输出。
EOF
}

SAVE_SUFFIX=""
while [ $# -gt 0 ]; do
    case "$1" in
        -h|--help) print_help; exit 0 ;;
        -s)
            if [ $# -lt 2 ]; then
                echo "Error: -s requires an argument" >&2
                exit 1
            fi
            SAVE_SUFFIX="$2"
            shift 2
            ;;
        *)
            if [ -z "${TILE_ARG:-}" ]; then
                TILE_ARG="$1"
                shift
            else
                print_help >&2
                exit 1
            fi
            ;;
    esac
done

expand_tiles() {
    local raw="$1"
    local result=()
    IFS=',' read -ra parts <<< "$raw"
    for part in "${parts[@]}"; do
        if [[ "$part" =~ ^([0-9]+)-([0-9]+)$ ]]; then
            local start="${BASH_REMATCH[1]}"
            local end="${BASH_REMATCH[2]}"
            for ((i = start; i <= end; i++)); do
                result+=("$i")
            done
        else
            result+=("$part")
        fi
    done
    echo "${result[@]}"
}

CHIP="${TXDA_VISIBLE_DEVICES:-10}"

if [ -n "${TILE_ARG:-}" ]; then
    if [ "$TILE_ARG" = "all" ]; then
        TILES=($(expand_tiles "0-15"))
    else
        TILES=($(expand_tiles "$TILE_ARG"))
    fi
else
    TILES=($(expand_tiles "0"))
fi

OUTPUT_DIR=/login_home/yinle/workspace/triton/kcore_log
mkdir -p "$OUTPUT_DIR"

# Clean up old output files before extraction
if ! ( cd "$OUTPUT_DIR" && rm -f c* t* ); then
    echo "Error: failed to clean up $OUTPUT_DIR" >&2
    exit 1
fi

if [ ${#TILES[@]} -gt 0 ]; then
    # Map chip -> bus-id from tsm_smi
    BUS_ID=$(tsm_smi 2>/dev/null \
        | grep -oP '\b\d+\s+[0-9a-f]{4}:[0-9a-f]{2}:[0-9a-f]{2}\.[0-9a-f]' \
        | awk -v c="$CHIP" '$1==c{print $2; exit}')

    if [ -z "$BUS_ID" ]; then
        echo "Error: cannot find bus-id for chip $CHIP from tsm_smi"
        exit 1
    fi

    LOG_DIR="/var/npu_ep_log/${BUS_ID}/KCORE"

    echo "chip=$CHIP  bus=$BUS_ID  tiles=${TILES[*]}"
    echo
fi

for TILE in "${TILES[@]}"; do
    # Tiles 10-15 are named KCOREA_* ... KCOREF_* in the filesystem
    TILE_HEX=$(printf '%X' "$TILE")
    LOG_FILE=$(ls -t "$LOG_DIR"/KCORE${TILE_HEX}_* 2>/dev/null | grep -v '\.tar\.gz$' | head -1) || true

    if [ -z "$LOG_FILE" ]; then
        echo "  tile=$TILE: no active log file, skip"
        continue
    fi

    CHIP_FILE="chip${CHIP}_tile${TILE}"
    echo "  tile=$TILE  source: $LOG_FILE"

    cd "$OUTPUT_DIR"

    # Step 1: strings the LOG_FILE
    strings "$LOG_FILE" > "$CHIP_FILE"
    echo "    strings -> $CHIP_FILE  ($(wc -c < "$CHIP_FILE") bytes)"

    # Step 2: extract the most recent append: everything after the second-to-last
    # [DONE] marker. If there are 0 or 1 [DONE] markers, take the whole file.
    # Write header (source file name + total lines) followed by the extracted chunk.
    CHIP_LINES=$(wc -l < "$CHIP_FILE")
    python3 -c "
import sys

with open('$CHIP_FILE') as f:
    data = f.read()

marker = '[DONE]'
offsets = []
pos = 0
while True:
    idx = data.find(marker, pos)
    if idx == -1:
        break
    offsets.append(idx)
    pos = idx + 1

if len(offsets) >= 2:
    start = offsets[-2]
    nl = data.find('\n', start)
    if nl != -1:
        start = nl + 1
    chunk = data[start:]
else:
    chunk = data

sys.stdout.write(f'source: $CHIP_FILE\n')
sys.stdout.write(f'total_lines: $CHIP_LINES\n')
sys.stdout.write(chunk)
" > "t${TILE}"

    echo "    -> t${TILE}  ($(wc -c < t${TILE}) bytes)"
done

# Save: copy t* to kcore_log/<suffix>/
if [ -n "$SAVE_SUFFIX" ]; then
    SAVE_PATH="${OUTPUT_DIR}/${SAVE_SUFFIX}"
    mkdir -p "$SAVE_PATH"

    copied=0
    while IFS= read -r src; do
        cp "$src" "$SAVE_PATH/"
        copied=$((copied + 1))
    done < <(find "$OUTPUT_DIR" -maxdepth 1 -type f -name 't*' 2>/dev/null)

    rm -f "$OUTPUT_DIR"/t* "$OUTPUT_DIR"/chip*

    echo
    echo "Saved $copied files -> $SAVE_PATH"
fi
