#!/bin/bash

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=lib/env.sh
source "$script_dir/lib/env.sh" || { echo "error: cannot load $script_dir/lib/env.sh" >&2; exit 1; }

exec bash "$TSM_SCRIPTS_DIR/base/base_run.sh" "$WORKSPACE" "$@"
