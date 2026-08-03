#!/bin/bash
# ---------------------------------------------------------------------------
# Common config and functions shared by CI scripts.
# Source this file from run_triton_flaggems_ci_*.sh.
#
# Usage:
#   source "$(dirname "$(realpath "$0")")/ci_common.sh"
# ---------------------------------------------------------------------------

set -e

# ===========================================================================
# 1. Path setup — computed from the calling script's location
# ===========================================================================

script_path=$(realpath "$0")
script_dir=$(dirname "$script_path")
project_dir=$(realpath "$script_dir/../../../../../")
export TRITON_WORKSPACE=$project_dir
TRITON_DEPENDS_SRC=/login_home/jenkins_tc/triton

# ===========================================================================
# 2. Config variables
# ===========================================================================

precision_mode=2
tx8_depends_name=tx8_depends_dev_20260507_104051
torch_txda_name=torch_txda-0.1.0+20260416.b8f53e8a.noprofile-cp310-cp310-linux_x86_64
txops_name=txops-0.1.0+20260508.60287151-py3-none-any
txda_skip_ops="repeat_interleave.self_int,pad,uniform_,sort.values_stable,resolve_conj"
txda_fallback_cpu_ops="random_,quantile,_local_scalar_dense,arange,unfold,index,le,all,ge,pad,to,gather_backward,zero_,view_as_real,resolve_neg,embedding_backward,sort,repeat_interleave,rsub,hstack,vstack,min,uniform_,abs,ne,eq,mul,bitwise_and,masked_select,max,ceil,div,gt,lt,sum,scatter,where,resolve_conj,isclose,isfinite,tile,equal,gather,_index_put_impl_,sub,to_dtype,isneginf,tril,count_nonzero,exp,exp_out,exp.out,fill_,flip,diag,view_as_complex,cat,log_sigmoid,kron,add"

# ===========================================================================
# 3. Download dependencies
# ===========================================================================

download_deps() {
    cd "$project_dir"

    ### download llvm (rarely changes)
    if [ ! -d "./llvm-a66376b0-ubuntu-x64" ]; then
        if [ -d "$TRITON_DEPENDS_SRC/llvm-a66376b0-ubuntu-x64" ]; then
            cp -r "$TRITON_DEPENDS_SRC/llvm-a66376b0-ubuntu-x64/" ./
            echo "cp $TRITON_DEPENDS_SRC/llvm-a66376b0-ubuntu-x64 complete!"
        else
            echo "warning：$TRITON_DEPENDS_SRC/llvm-a66376b0-ubuntu-x64 not exist， use wget to download, maybe very slowly!"
        fi
    fi

    if [ ! -d "./llvm-a66376b0-ubuntu-x64" ]; then
        wget -P "$TRITON_DEPENDS_SRC" https://toolchain-jfrog.tsingmicro.xyz:443/artifactory/tx8-generic-dev/triton/tools/llvm-a66376b0-ubuntu-x64.tar.gz
        if [ $? -eq 0 ]; then
            echo "Download llvm complete!"
        else
            echo "Download llvm fail!!!"
            exit -1
        fi
        cp "$TRITON_DEPENDS_SRC/llvm-a66376b0-ubuntu-x64.tar.gz" ./
        tar -xzvf llvm-a66376b0-ubuntu-x64.tar.gz
        rm llvm-a66376b0-ubuntu-x64.tar.gz
    fi

    if [ ! -d "./llvm-a66376b0-ubuntu-x64" ]; then
        echo "fail: not find llvm!!!"
        exit -1
    fi

    ### download torch2.7 wheels for offline install (rarely changes)
    if [ ! -d "./offline_pkgs" ]; then
        if [ -d "$TRITON_DEPENDS_SRC/offline_pkgs" ]; then
            cp -r "$TRITON_DEPENDS_SRC/offline_pkgs/" ./
            echo "cp $TRITON_DEPENDS_SRC/offline_pkgs complete!"
        else
            echo "warning：$TRITON_DEPENDS_SRC/offline_pkgs not exist， use wget to download, maybe very slowly!"
        fi
    fi

    if [ ! -d "./offline_pkgs" ]; then
        wget -P "$TRITON_DEPENDS_SRC" https://toolchain-jfrog.tsingmicro.xyz:443/artifactory/tx8-generic-dev/triton/offline_pkgs/offline_pkgs_v5.3.0.tar.gz
        if [ $? -eq 0 ]; then
            echo "Download offline package complete!"
        else
            echo "Download offline package fail!!!"
            exit -1
        fi
        cp "$TRITON_DEPENDS_SRC/offline_pkgs_v5.3.0.tar.gz" ./
        tar -xzvf offline_pkgs_v5.3.0.tar.gz
        rm offline_pkgs_v5.3.0.tar.gz
    fi

    if [ ! -d "./offline_pkgs" ]; then
        echo "fail: not find offline_pkgs!!!"
        exit -1
    fi

    ### download tx8_deps (changes frequently)
    if [ ! -e "$tx8_depends_name.tar.gz" ]; then
        if [ ! -e "$TRITON_DEPENDS_SRC/$tx8_depends_name.tar.gz" ]; then
            echo "warning：$TRITON_DEPENDS_SRC/$tx8_depends_name.tar.gz not exist， use wget to download, maybe very slowly!"
            wget -P "$TRITON_DEPENDS_SRC" "https://toolchain-jfrog.tsingmicro.xyz:443/artifactory/tx8-generic-dev/triton/tx8_depends/$tx8_depends_name.tar.gz"
            if [ $? -eq 0 ]; then
                echo "Download tx8_deps complete!"
            else
                echo "Download tx8_dpes fail!!!"
                exit -1
            fi
        fi
        if [ -e "$TRITON_DEPENDS_SRC/$tx8_depends_name.tar.gz" ]; then
            cp "$TRITON_DEPENDS_SRC/$tx8_depends_name.tar.gz" ./
            if [ -d "./tx8_deps" ]; then
                rm -rf tx8_deps
            fi
            tar -xzvf "$tx8_depends_name.tar.gz"
            echo "cp $TRITON_DEPENDS_SRC/$tx8_depends_name.tar.gz complete!"
        else
            echo "fail: not find tx8_deps!!!"
            exit -1
        fi
    fi

    if [ ! -d "./pack" ]; then
        mkdir pack
    fi

    ### download torch_txda (changes frequently)
    if [ ! -e "./pack/$torch_txda_name.whl" ]; then
        if [ ! -e "$TRITON_DEPENDS_SRC/$torch_txda_name.whl" ]; then
            echo "warning：$TRITON_DEPENDS_SRC/$torch_txda_name.tar.gz not exist， use wget to download, maybe very slowly!"
            wget -P "$TRITON_DEPENDS_SRC" "https://toolchain-jfrog.tsingmicro.xyz:443/artifactory/tx8-generic-dev/torch_txda/$torch_txda_name.whl"
            if [ $? -eq 0 ]; then
                echo "Download torch_txda complete!"
            else
                echo "Download torch_txda fail!!!"
                exit -1
            fi
        fi

        if [ -e "$TRITON_DEPENDS_SRC/$torch_txda_name.whl" ]; then
            cp "$TRITON_DEPENDS_SRC/$torch_txda_name.whl" ./pack
            echo "cp $TRITON_DEPENDS_SRC/$torch_txda_name.whl complete!"
        else
            echo "fail: not find torch_txda pack!!!"
            exit -1
        fi
    fi

    ### download txops
    if [ ! -e "./pack/$txops_name.whl" ]; then
        if [ ! -e "$TRITON_DEPENDS_SRC/$txops_name.whl" ]; then
            echo "warning：$TRITON_DEPENDS_SRC/$txops_name.tar.gz not exist， use wget to download, maybe very slowly!"
            wget -P "$TRITON_DEPENDS_SRC" "https://toolchain-jfrog.tsingmicro.xyz:443/artifactory/tx8-generic-dev/torch_txda/$txops_name.whl"
            if [ $? -eq 0 ]; then
                echo "Download txops complete!"
            else
                echo "Download txops fail!!!"
                exit -1
            fi
        fi

        if [ -e "$TRITON_DEPENDS_SRC/$txops_name.whl" ]; then
            cp "$TRITON_DEPENDS_SRC/$txops_name.whl" ./pack
            echo "cp $TRITON_DEPENDS_SRC/$txops_name.whl complete!"
        else
            echo "fail: not find txops pack!!!"
            exit -1
        fi
    fi
}

# ===========================================================================
# 4. Install dependencies
# ===========================================================================

install_deps() {
    cd "$project_dir/triton"

    if [ -d "./.venv" ]; then
        rm -rf .venv
    fi
    python3 -m venv .venv --prompt triton
    source .venv/bin/activate

    # check python version
    python3 --version

    bash third_party/tsingmicro/scripts/tools/offline_python_deps.sh -i -r python/requirements.txt -d ../offline_pkgs
    if [ $? -eq 0 ]; then
        echo "Install compile tool package complete!"
    else
        echo "Install compile tool package fail!!!"
        exit -1
    fi

    bash third_party/tsingmicro/scripts/tools/offline_python_deps.sh -i -r third_party/tsingmicro/scripts/requirements_ts.txt -d ../offline_pkgs
    if [ $? -eq 0 ]; then
        echo "Install torch package complete!"
    else
        echo "Install torch package fail!!!"
        exit -1
    fi

    # check torch version
    python3 -c "import torch; print(torch.__version__)"

    PROXY=http://192.168.100.225:8889
    export https_proxy=$PROXY http_proxy=$PROXY all_proxy=$PROXY
    apt update
    apt install -y ccache
    pip install loguru
    pip install scipy
    unset https_proxy
    unset http_proxy
    unset all_proxy

    ### install torch_txda and txops
    txops_wheel=$(find ../pack/ -maxdepth 1 -name $txops_name.whl -print -quit)
    torch_txda_wheel=$(find ../pack/ -maxdepth 1 -name $torch_txda_name.whl -print -quit)
    pip install "$txops_wheel"
    pip install "$torch_txda_wheel"
}

# ===========================================================================
# 5. Build triton
# ===========================================================================

build_triton() {
    cd "$project_dir/triton"
    rm -rf triton/python/build
    bash ./third_party/tsingmicro/scripts/build_tsingmicro.sh
    if [ $? -eq 0 ]; then
        echo "Build triton complete!"
    else
        echo "Build triton fail!!!"
        exit -1
    fi
}

# ===========================================================================
# 6. Setup environment variables (export + print)
# ===========================================================================

setup_env_vars() {
    TX8_DEPS_ROOT=$TRITON_WORKSPACE/tx8_deps
    LLVM=$TRITON_WORKSPACE/llvm-a66376b0-ubuntu-x64
    export TX8_DEPS_ROOT=$TX8_DEPS_ROOT
    export LLVM_SYSPATH=$LLVM
    export LLVM_BINARY_DIR=$LLVM/bin
    export PYTHONPATH=$LLVM/python_packages/mlir_core:$PYTHONPATH
    export LD_LIBRARY_PATH=$TX8_DEPS_ROOT/lib:$LD_LIBRARY_PATH
    export TRITON_ALWAYS_COMPILE=1
    export TRITON_QUICK_MODE=1
    export TRITON_PRINT_AUTOTUNING=1
    export PRECISION_MODE=$precision_mode
    export TRITON_ALLOW_NON_CONSTEXPR_GLOBALS=1
    export TXDA_SKIP_OPS=$txda_skip_ops
    export TXDA_FALLBACK_CPU_OPS=$txda_fallback_cpu_ops
    # flaggemm 的tsingmicro 后端优化
    export FLAG_GEMS_CUSTOM_OPS=${FLAG_GEMS_CUSTOM_OPS:-1}

    echo "TX8_DEPS_ROOT="$TX8_DEPS_ROOT
    echo "LLVM_SYSPATH="$LLVM_SYSPATH
    echo "LLVM_BINARY_DIR="$LLVM_BINARY_DIR
    echo "PYTHONPATH="$PYTHONPATH
    echo "LD_LIBRARY_PATH="$LD_LIBRARY_PATH
    echo "TRITON_ALWAYS_COMPILE="$TRITON_ALWAYS_COMPILE
    echo "PRECISION_MODE="$PRECISION_MODE
    echo "TRITON_ALLOW_NON_CONSTEXPR_GLOBALS="$TRITON_ALLOW_NON_CONSTEXPR_GLOBALS
    echo "TXDA_SKIP_OPS="$TXDA_SKIP_OPS
    echo "TXDA_FALLBACK_CPU_OPS="$TXDA_FALLBACK_CPU_OPS
    echo "export FLAG_GEMS_CUSTOM_OPS=$FLAG_GEMS_CUSTOM_OPS"
}

# ===========================================================================
# 7. Profiler environment (kernel device time via tsm profiler)
# ===========================================================================

setup_kernel_profiler_env() {
    local profiler_lib=/usr/local/kuiper/tsm8-profiler/lib
    export TSM_PROFILER_EN=1
    export TRITON_QUICK_MODE=0
    export LD_LIBRARY_PATH=$profiler_lib:$LD_LIBRARY_PATH
    export LD_PRELOAD=$profiler_lib/libtsmprofiler-register.so:$profiler_lib/libtsmprofiler-sdk.so
    export ROCP_TOOL_LIBRARIES=$profiler_lib/libtsm-api-log-tracing.so

    echo "TSM_PROFILER_EN=$TSM_PROFILER_EN"
    echo "LD_PRELOAD=$LD_PRELOAD"
    echo "ROCP_TOOL_LIBRARIES=$ROCP_TOOL_LIBRARIES"
}

# ===========================================================================
# 8. Common cleanup before running tests
# ===========================================================================

cleanup_before_run() {
    cd "$project_dir"
    rm -rf ~/.triton/
    rm -rf ~/.flaggems/
    rm -rf triton/dump/
    rm -rf /tmp/triton_*
    rm -rf /tmp/flaggems_*
    rm -rf log/
    rm -f result.json
    rm -f tsingmicro_launch.log
}

# ===========================================================================
# 8. Activate venv if needed (after skipping install)
# ===========================================================================

activate_venv_if_needed() {
    if [ $skip_install -eq 1 ]; then
        source "$project_dir/triton/.venv/bin/activate"
    fi
}
