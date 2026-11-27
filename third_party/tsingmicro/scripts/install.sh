#!/bin/bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=lib/env.sh
source "$script_dir/lib/env.sh"

use_venv=ON
[ "${1:-}" = "no_venv" ] && use_venv=OFF

# Skip the repo-local venv when the caller is already inside one.  Activation
# normally sets VIRTUAL_ENV, but a PATH-only setup (common in containers) does
# not, so also ask the interpreter itself.
_in_venv() {
    "$PYTHON" -c 'import sys; raise SystemExit(0 if sys.prefix != sys.base_prefix else 1)' 2>/dev/null
}
if [ "$use_venv" = "ON" ] && [ -z "${VIRTUAL_ENV:-}" ] && ! _in_venv; then
    "$PYTHON" -m venv "$PROJECT_DIR/.venv" --prompt triton
    # shellcheck disable=SC1091
    source "$PROJECT_DIR/.venv/bin/activate"
    PYTHON="$PROJECT_DIR/.venv/bin/python3"
    export PYTHON
fi

tsm_setup_proxy

for _tool in git ccache; do
    command -v "$_tool" >/dev/null 2>&1 || \
        echo "warning: $_tool not found; install it if the build needs it" >&2
done
command -v ld.lld >/dev/null 2>&1 || \
    echo "warning: ld.lld not found; point LLVM_SYSPATH at an LLVM that provides lld" >&2

triton_origin_req="$PROJECT_DIR/python/requirements.txt"
[ -f "$triton_origin_req" ] || triton_origin_req="$script_dir/requirements.txt"
[ -f "$triton_origin_req" ] || { echo "error: cannot find requirements.txt" >&2; exit 1; }

"$PYTHON" -m pip install -r "$triton_origin_req"
"$PYTHON" -m pip install -r "$script_dir/requirements_ts.txt"

# torch_txda is built against a specific torch ABI; validate the pair when it is
# present instead of relying on a hardcoded version alone.
if "$PYTHON" -c "import importlib.util, sys; sys.exit(0 if importlib.util.find_spec('torch_txda') else 1)"; then
    if ! "$PYTHON" -c "import torch, torch_txda, torch.txda; assert torch.txda.is_available()"; then
        echo "error: torch_txda is not compatible with the installed torch" >&2
        "$PYTHON" -c "import torch; print('installed torch', torch.__version__)" >&2 || true
        exit 1
    fi
else
    echo "note: torch_txda is not installed; install it before running Triton" >&2
fi
