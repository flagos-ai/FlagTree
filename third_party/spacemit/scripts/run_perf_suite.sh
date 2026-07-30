#!/usr/bin/env bash
set -euo pipefail

# Run perf scripts for spine-triton.
# Usage:
#   ./run_perf_suite.sh            # uses default python
#   PYTHON=/path/to/python ./run_perf_suite.sh
#   ./run_perf_suite.sh --logdir /path/to/logs
#
# Notes:
# - This script runs each perf script in isolation and records stdout/stderr.

PYTHON_BIN="${PYTHON:-python3}"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

LOGDIR="${LOGDIR:-${REPO_ROOT}/perf_logs}"
#!/usr/bin/env bash
set -euo pipefail

# Run spine-triton perf scripts and append all outputs into ONE log.
#
# Usage:
#   ./run_perf_suite.sh
#   PYTHON=/path/to/python ./run_perf_suite.sh
#   ./run_perf_suite.sh --python /path/to/python --logdir /path/to/logs

PYTHON_BIN="${PYTHON:-python3}"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

LOGDIR="${LOGDIR:-${REPO_ROOT}/perf_logs}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --python)
      PYTHON_BIN="$2"; shift 2;;
    --logdir)
      LOGDIR="$2"; shift 2;;
    -h|--help)
      sed -n '1,80p' "$0"; exit 0;;
    *)
      echo "Unknown arg: $1" >&2
      exit 2;;
  esac
done

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "ERROR: Python executable not found: $PYTHON_BIN" >&2
  exit 1
fi

mkdir -p "$LOGDIR"
ts="$(date +"%Y%m%d_%H%M%S")"
LOGFILE="${LOGDIR}/perf_suite.${ts}.log"

PERF_SCRIPTS=(
  "${REPO_ROOT}/python/perf/perf_mm_col.py"
  "${REPO_ROOT}/python/perf/test_mm_512_fp16.py"
  "${REPO_ROOT}/python/perf/perf_custom_matrix.py"
  "${REPO_ROOT}/python/perf/perf_elementwise.py"
  "${REPO_ROOT}/python/perf/test_softmax.py"
  "${REPO_ROOT}/python/perf/perf_attention.py"
)

{
  echo "# perf suite"
  echo "date: $(date -Iseconds)"
  echo "repo_root: $REPO_ROOT"
  echo "python: $PYTHON_BIN"
  echo "logfile: $LOGFILE"
  echo

  cd "$REPO_ROOT"
  for s in "${PERF_SCRIPTS[@]}"; do
    echo "============================================================"
    echo "[RUN] $s"
    if [[ ! -f "$s" ]]; then
      echo "[SKIP] missing: $s"
      continue
    fi
    "$PYTHON_BIN" "$s"
    echo "[DONE] $s"
    echo
  done
} 2>&1 | tee "$LOGFILE"

echo "All outputs saved to: $LOGFILE"
  if ! run_one "$s"; then
