#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TEST_FILE="${ROOT_DIR}/python/test/unit/integrations/vllm/punica_lora/test_punica_ops.py"

usage() {
  cat <<'EOF'
Usage: run_punica_lora_tests.sh <mode> [args] [-- pytest_args...]

Modes:
  full        Run the full parameter sweep (large hidden sizes list)
  quick       Run a small, fast smoke test
  expand      Run only lora_expand tests
  shrink      Run only lora_shrink tests
  nonconsec   Run only non-consecutive mapping tests
  case        Run a single case via PUNICA_CASE (see example below)

Examples:
  ./run_punica_lora_tests.sh full
  ./run_punica_lora_tests.sh quick
  ./run_punica_lora_tests.sh expand
  ./run_punica_lora_tests.sh shrink
  ./run_punica_lora_tests.sh nonconsec
  ./run_punica_lora_tests.sh case "batches=4,num_loras=4,rank=32,hidden_size=2048,nslices=1,dtype=bf16,op=expand"
  ./run_punica_lora_tests.sh expand -- -k test_kernels_hidden_size -vv

Env knobs:
  PUNICA_TEST_LEVEL=default|quick|full
  PUNICA_OP=expand|shrink
  PUNICA_DEVICE=cuda:0
  PUNICA_DISABLE_STORE_STP=0|1
EOF
}

MODE="${1:-}"
shift || true

PYTEST_ARGS=()
if [[ "${1:-}" == "--" ]]; then
  shift
  PYTEST_ARGS=("$@")
fi

run_pytest() {
  local cmd=(pytest -q "${TEST_FILE}")
  if (( ${#PYTEST_ARGS[@]} )); then
    cmd+=("${PYTEST_ARGS[@]}")
  fi
  if (( $# )); then
    env "$@" "${cmd[@]}"
  else
    "${cmd[@]}"
  fi
}

case "${MODE}" in
  full)
    run_pytest PUNICA_TEST_LEVEL=full
    ;;
  quick)
    run_pytest PUNICA_TEST_LEVEL=quick
    ;;
  expand)
    run_pytest PUNICA_OP=expand
    ;;
  shrink)
    run_pytest PUNICA_OP=shrink
    ;;
  nonconsec)
    PYTEST_ARGS+=("-k" "nonconsecutive_mapping")
    run_pytest
    ;;
  case)
    if [[ -z "${1:-}" ]]; then
      echo "case mode requires a PUNICA_CASE string" >&2
      exit 2
    fi
    run_pytest PUNICA_CASE="$1"
    ;;
  ""|-h|--help|help)
    usage
    ;;
  *)
    echo "Unknown mode: ${MODE}" >&2
    usage
    exit 2
    ;;
esac
