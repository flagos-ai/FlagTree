#!/bin/bash

if [ $# -lt 3 ]; then
    echo "Error: usage: $(basename "$0") <workspace> <pytest|python> <args...>" >&2
    exit 1
fi

WORKSPACE=$1
[ -d "$WORKSPACE" ] || { echo "Error: $WORKSPACE not exist!" >&2; exit 1; }
export WORKSPACE

run_model=$2
case "$run_model" in
    pytest|python) : ;;
    *) echo "Error: first argument must be 'pytest' or 'python'" >&2; exit 1 ;;
esac
shift 2

# shellcheck source=../lib/env.sh
_tsm_env_sh="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/../lib/env.sh"
# shellcheck disable=SC1090
source "$_tsm_env_sh" || { echo "error: cannot load $_tsm_env_sh" >&2; exit 1; }
tsm_env_check || exit 1
tsm_check_runtime_abi || exit 1
tsm_activate_venv

txda_skip_ops="repeat_interleave.self_int,pad,uniform_,sort.values_stable,resolve_conj"
txda_fallback_cpu_ops="random_,quantile,_local_scalar_dense,arange,unfold,index,le,all,ge,pad,to,gather_backward,zero_,view_as_real,resolve_neg,embedding_backward,sort,repeat_interleave,rsub,hstack,vstack,min,uniform_,abs,ne,eq,mul,bitwise_and,masked_select,max,ceil,div,gt,lt,sum,scatter,where,resolve_conj,isclose,isfinite,tile,equal,gather,_index_put_impl_,sub,to_dtype,isneginf,tril,count_nonzero,exp,exp_out,exp.out,fill_,flip,diag,view_as_complex,cat,log_sigmoid,kron,add"

export TXDA_SKIP_OPS=$txda_skip_ops
export TXDA_FALLBACK_CPU_OPS=$txda_fallback_cpu_ops

# Test-run defaults: keep the fast settings but let the caller override them.
export TRITON_ALWAYS_COMPILE="${TRITON_ALWAYS_COMPILE:-1}"
export TRITON_QUICK_MODE="${TRITON_QUICK_MODE:-1}"
export TRITON_PRINT_AUTOTUNING="${TRITON_PRINT_AUTOTUNING:-1}"
export PRECISION_MODE="${PRECISION_MODE:-2}"
export TRITON_ALLOW_NON_CONSTEXPR_GLOBALS="${TRITON_ALLOW_NON_CONSTEXPR_GLOBALS:-1}"

# flagtree discovers backends from the source tree; triton uses the flaggems
# custom ops instead.
if [ -d "$TRITON/python/flagtree" ]; then
    export TRITON_BACKENDS_IN_TREE=1
else
    export FLAG_GEMS_CUSTOM_OPS=1
fi

# Only the JIT cache is disposable: env.sh keeps json/llvm/tx8_deps fallbacks
# under ~/.triton as well.
if [ "${ENABLE_PROFILING:-}" = "1" ] || [ -n "${CUSTOMIZED_IR:-}" ]; then
    cache_dir="${TRITON_CACHE_DIR:-$HOME/.triton/cache}"
    rm -rf "$cache_dir"
    echo "run cmd:rm -rf $cache_dir"
fi

tsm_dump_env TXDA_VISIBLE_DEVICES TX8_DEPS_ROOT LLVM_SYSPATH LLVM_BINARY_DIR \
    PYTHONPATH LD_LIBRARY_PATH PRECISION_MODE FLAG_GEMS_CUSTOM_OPS \
    TX_LAUNCH_LOG_LEVEL TRITON_DUMP_PATH TRITON_ALWAYS_COMPILE \
    TXDA_SKIP_OPS TXDA_FALLBACK_CPU_OPS ENABLE_PROFILING CUSTOMIZED_IR

"$PYTHON" -c 'import torch, torch_txda' >/dev/null 2>&1 || \
    echo "[env] WARN: torch/torch_txda not importable; see READMD_DEV.md 用户安装" >&2

pytest_args=()
if [ "$run_model" == "pytest" ]; then
    pytest_args=(-m pytest -v -s)
    # Point pytest at the inifile next to the test so a foreign pyproject.toml
    # in the CWD does not become the rootdir.  Skip option values (-k, -m, ...).
    _skip=0
    for _arg in "$@"; do
        if [ "$_skip" = "1" ]; then
            _skip=0
            continue
        fi
        case "$_arg" in
            -k|-m|-p|-n|-o|-W|--deselect|--rootdir) _skip=1; continue ;;
            -*) continue ;;
        esac
        _path=${_arg%%::*}
        if [ -d "$_path" ]; then
            _dir=$(cd "$_path" 2>/dev/null && pwd)
        else
            _dir=$(cd "$(dirname "$_path")" 2>/dev/null && pwd)
        fi
        while [ -n "$_dir" ] && [ "$_dir" != "/" ]; do
            if [ -f "$_dir/pytest.ini" ] || [ -f "$_dir/setup.cfg" ] || [ -f "$_dir/tox.ini" ]; then
                pytest_args+=("--rootdir=$_dir")
                break
            fi
            _dir=$(dirname "$_dir")
        done
        break
    done
fi

echo "run cmd: $PYTHON ${pytest_args[*]} $*"
USE_SIM_MODE="${USE_SIM_MODE:-}" "$PYTHON" "${pytest_args[@]}" "$@"
