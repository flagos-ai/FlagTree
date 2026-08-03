#!/bin/bash

if [ $# -le 2 ]; then
    echo "Error: At least two parameters need to be passed!"
    exit 1
fi

WORKSPACE=$1
if [ ! -d $WORKSPACE ]; then
    echo "Error: $WORKSPACE not exist!" 1>&2
    exit 1
fi

args1=$2
if [ "$args1" != "pytest" ] && [ "$args1" != "python" ]; then
    echo "Error: first args is 'pytest' or 'python'"
    exit 1
fi
run_model=$args1

shift
shift

TRITON=$WORKSPACE/triton
TX8_DEPS_ROOT=$WORKSPACE/tx8_deps
LLVM=$WORKSPACE/llvm-a66376b0-ubuntu-x64

if [ ! -d $TX8_DEPS_ROOT ] || [ ! -d $LLVM ]; then
    WORKSPACE="${HOME}/.triton/tsingmicro/"
    TX8_DEPS_ROOT=$WORKSPACE/tx8_deps
    LLVM=$WORKSPACE/llvm-a66376b0-ubuntu-x64
fi

if [ ! -d $TX8_DEPS_ROOT ]; then
    echo "Error: $TX8_DEPS_ROOT not exist!" 1>&2
    exit 1
fi

if [ ! -d $LLVM ]; then
    echo "Error: $LLVM not exist!" 1>&2
    exit 1
fi

if [ -f $TRITON/.venv/bin/activate ]; then
    source $TRITON/.venv/bin/activate
fi

txda_skip_ops="repeat_interleave.self_int,pad,uniform_,sort.values_stable,resolve_conj"
txda_fallback_cpu_ops="random_,quantile,_local_scalar_dense,arange,unfold,index,le,all,ge,pad,to,gather_backward,zero_,view_as_real,resolve_neg,embedding_backward,sort,repeat_interleave,rsub,hstack,vstack,min,uniform_,abs,ne,eq,mul,bitwise_and,masked_select,max,ceil,div,gt,lt,sum,scatter,where,resolve_conj,isclose,isfinite,tile,equal,gather,_index_put_impl_,sub,to_dtype,isneginf,tril,count_nonzero,exp,exp_out,exp.out,fill_,flip,diag,view_as_complex,cat,log_sigmoid,kron,add"

# 必须的
export TX8_DEPS_ROOT=$TX8_DEPS_ROOT
export LLVM_SYSPATH=$LLVM
export LLVM_BINARY_DIR=$LLVM/bin

# 后续需要优化删除的
export PYTHONPATH=$LLVM/python_packages/mlir_core:$PYTHONPATH
export LD_LIBRARY_PATH=$TX8_DEPS_ROOT/lib:$LD_LIBRARY_PATH
export TXDA_SKIP_OPS=$txda_skip_ops
export TXDA_FALLBACK_CPU_OPS=$txda_fallback_cpu_ops

export TRITON_ALWAYS_COMPILE=1
#autotune不走do_bench函数,每个config kernel只会运行一次,减少运行耗时
export TRITON_QUICK_MODE=1
export TRITON_PRINT_AUTOTUNING=1

# 高精度模式
export PRECISION_MODE=2
#multinomial算子编译需要
export TRITON_ALLOW_NON_CONSTEXPR_GLOBALS=1
# flaggemm 的tsingmicro 后端优化
export FLAG_GEMS_CUSTOM_OPS=1

# 非必须的 调试相关， 不配置不生成dump文件
# export TRITON_DUMP_PATH=$TRITON/dump
# export ENABLE_PROFILING=1
# export TX_LAUNCH_LOG_LEVEL=debug
# export TX_LOG_LEVEL=debug
# export CUSTOMIZED_IR=test_0.mlir,test_1.mlir
# 选择板卡，默认0卡
# export TXDA_VISIBLE_DEVICES=30

# 指令profile
# export ENALBE_PROFILING=1
# export EXPORT_PROFILING_PATH=$TRITON/instruct_profile

# kernel profile
export TSM_PROFILER_EN=${TSM_PROFILER_EN:-0}
if [ "$TSM_PROFILER_EN" == "1" ]; then
    profiler_lib=/usr/local/kuiper/tsm8-profiler/lib
    export LD_LIBRARY_PATH=$profiler_lib:$LD_LIBRARY_PATH
    export LD_PRELOAD=$profiler_lib/libtsmprofiler-register.so:$profiler_lib/libtsmprofiler-sdk.so
    export ROCP_TOOL_LIBRARIES=$profiler_lib/libtsm-api-log-tracing.so
fi

if [ "$ENABLE_PROFILING" == "1" ] || [ -n "$CUSTOMIZED_IR" ]; then
    # 调试中，删除临时文件，必须重新生成
    rm -rf ~/.triton
    echo "run cmd:rm -rf ~/.triton"
fi

echo "export TXDA_VISIBLE_DEVICES=$TXDA_VISIBLE_DEVICES"
echo "export TX8_DEPS_ROOT=$TX8_DEPS_ROOT"
echo "export LLVM_SYSPATH=$LLVM_SYSPATH"
echo "export LLVM_BINARY_DIR=$LLVM_BINARY_DIR"
echo "export PYTHONPATH=$PYTHONPATH"
echo "export LD_LIBRARY_PATH=$LD_LIBRARY_PATH"
echo "export PRECISION_MODE=$PRECISION_MODE"
echo "export FLAG_GEMS_CUSTOM_OPS=$FLAG_GEMS_CUSTOM_OPS"

echo "export TX_LAUNCH_LOG_LEVEL=$TX_LAUNCH_LOG_LEVEL"
echo "export TRITON_DUMP_PATH=$TRITON_DUMP_PATH"
echo "export TRITON_ALWAYS_COMPILE=$TRITON_ALWAYS_COMPILE"
echo "export TXDA_SKIP_OPS=$TXDA_SKIP_OPS"
echo "export TXDA_FALLBACK_CPU_OPS=$TXDA_FALLBACK_CPU_OPS"

echo "export ENALBE_PROFILING=$ENALBE_PROFILING"
echo "export EXPORT_PROFILING_PATH=$EXPORT_PROFILING_PATH"
echo "export TSM_PROFILER_EN=$TSM_PROFILER_EN"
echo "export LD_PRELOAD=$LD_PRELOAD"
echo "export ROCP_TOOL_LIBRARIES=$ROCP_TOOL_LIBRARIES"

echo "export ENABLE_PROFILING=$ENABLE_PROFILING"
echo "export CUSTOMIZED_IR=$CUSTOMIZED_IR"

# 检查指定卡是否被占用
_TARGET_CARDS=${TXDA_VISIBLE_DEVICES:-0}
_OCCUPIED_CARDS=$(tsm_smi 2>/dev/null | awk '
    /^[|] Processes:/ { in_proc=1; next }
    !in_proc { next }
    /^[|] *No running/ { exit }
    /^[|]/ {
        gsub(/^[|] */,"")
        split($0, a, /[ \t]+/)
        if (a[1] ~ /^[0-9]+$/) print a[1]
    }
')
for _card in ${_TARGET_CARDS//,/ }; do
    for _occ in $_OCCUPIED_CARDS; do
        if [ "$_card" == "$_occ" ]; then
            echo "Error: compute card $_card is occupied, abort!" >&2
            echo "Occupied cards: $(echo $_OCCUPIED_CARDS | tr '\n' ' ')" >&2
            exit 1
        fi
    done
done

pytest_cmd=""
if [ "$args1" == "pytest" ]; then
    pytest_cmd="-m pytest -v -s"
fi

echo "run cmd:python3 $pytest_cmd $@"

USE_SIM_MODE=${USE_SIM_MODE} python3 $pytest_cmd $@
