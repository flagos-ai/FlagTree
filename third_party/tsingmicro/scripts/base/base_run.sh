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

txda_skip_ops="repeat_interleave.self_int,pad,to.dtype,uniform_,sort.values_stable,contiguous,resolve_conj"
txda_fallback_cpu_ops="random_,quantile,_local_scalar_dense,arange,unfold,index,le,all,ge,pad,to,gather_backward,zero_,view_as_real,resolve_neg,embedding_backward,sort,repeat_interleave,rsub,hstack,vstack,min,uniform_,abs,ne,eq,mul,bitwise_and,masked_select,max,ceil,div,gt,lt,sum,scatter,where,resolve_conj,isclose,isfinite,tile,equal,gather,contiguous"

# 必须的
export TX8_DEPS_ROOT=$TX8_DEPS_ROOT
export LLVM_SYSPATH=$LLVM
export LLVM_BINARY_DIR=$LLVM/bin

# 后续需要优化删除的
export PYTHONPATH=$LLVM/python_packages/mlir_core:$PYTHONPATH
export LD_LIBRARY_PATH=$TX8_DEPS_ROOT/lib:$LD_LIBRARY_PATH
export TXDA_SKIP_OPS=$txda_skip_ops
export TXDA_FALLBACK_CPU_OPS=$txda_fallback_cpu_ops

# 非必须的 调试相关
export TRITON_DUMP_PATH=$TRITON/dump
export TRITON_ALWAYS_COMPILE=1
# dump launch调用的所有参数，包括kernel func调用的参数
export DUMP_KERNEL_ARGS=1

# 高精度模式
export PRECISION_PRIORITY=1
#multinomial算子编译需要
export TRITON_ALLOW_NON_CONSTEXPR_GLOBALS=1

# export DEBUG=ON
# export USE_PROFILE=1
# export USE_HOST_PROFILE=1
# export CUSTOMIZED_IR=test_0.mlir,test_1.mlir
# export TRACE_POINTS="__Rdma,__Wdma"

echo "export TX8_DEPS_ROOT=$TX8_DEPS_ROOT"
echo "export LLVM_SYSPATH=$LLVM_SYSPATH"
echo "export LLVM_BINARY_DIR=$LLVM_BINARY_DIR"
echo "export PYTHONPATH=$PYTHONPATH"
echo "export LD_LIBRARY_PATH=$LD_LIBRARY_PATH"
echo "export PRECISION_PRIORITY=$PRECISION_PRIORITY"

echo "export DUMP_KERNEL_ARGS=$DUMP_KERNEL_ARGS"
echo "export TRITON_DUMP_PATH=$TRITON_DUMP_PATH"
echo "export TRITON_ALWAYS_COMPILE=$TRITON_ALWAYS_COMPILE"
echo "export TXDA_SKIP_OPS=$TXDA_SKIP_OPS"
echo "export TXDA_FALLBACK_CPU_OPS=$TXDA_FALLBACK_CPU_OPS"

echo "export USE_PROFILE=$USE_PROFILE"
echo "export USE_HOST_PROFILE=$USE_HOST_PROFILE"
echo "export CUSTOMIZED_IR=$CUSTOMIZED_IR"
echo "export TRACE_POINTS=$TRACE_POINTS"

pytest_cmd=""
if [ "$args1" == "pytest" ]; then
    pytest_cmd="-m pytest -v -s"
fi

echo "run cmd:python3 $pytest_cmd $@"

USE_SIM_MODE=${USE_SIM_MODE} python3 $pytest_cmd $@
