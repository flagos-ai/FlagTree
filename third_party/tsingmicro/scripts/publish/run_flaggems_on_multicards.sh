#!/bin/bash
set -e
##.在docker容器内版本包路径下执行
#bash scripts/run_flaggems_on_multicards.sh ci_ops 1


##########################################################################################################################
##                                                                                                                      ##
##  在多卡上并行运行Triton算子测试脚本                                                                                  ##
##     param1: test_set,     set test set name, default 'ci_ops'.                                                       ##
##     param2: device_count, set device count number,   default 1.                                                      ##
##     ##param: precision_priority, set 1-triton compiler use high precision mode for special ops, default 1.           ##
##     param3: quick_mode,   set 1-quick mode to run flaggems, set 0-normal mode, default 0.                            ##
##     param4: skip_device,  set devices that need to be skipped, when they are unavailable, default [].                ##
##                                                                                                                      ##
##########################################################################################################################

script_path=$(realpath "$0")
echo $script_path
script_dir=$(dirname "$script_path")
echo $script_dir
project_dir=$(realpath "$script_dir/../")
echo $project_dir
export TRITON_WORKSPACE=$project_dir
test_set=ci_ops
device_count=1
quick_mode=0
skip_device=
precision_priority=1
txda_skip_ops="repeat_interleave.self_int,pad,to.dtype,uniform_,sort.values_stable,contiguous,resolve_conj"
txda_fallback_cpu_ops="random_,quantile,_local_scalar_dense,arange,unfold,index,le,all,ge,pad,to,gather_backward,zero_,view_as_real,resolve_neg,embedding_backward,sort,repeat_interleave,rsub,hstack,vstack,min,uniform_,abs,ne,eq,mul,bitwise_and,masked_select,max,ceil,div,gt,lt,sum,scatter,where,resolve_conj,isclose,isfinite,tile,equal,gather,contiguous"

if [ $# -ge 1 ]; then
	test_set=$1
fi
if [ $# -ge 2 ]; then
	device_count=$2
fi
if [ $# -ge 3 ]; then
	quick_mode=$3
fi
if [ $# -ge 4 ]; then
	skip_device=$(echo $4 | tr ',' ' ')
fi
echo "param count:"$#
echo "test_set:"$test_set
echo "device_count:"$device_count
echo "quick_mode:"$quick_mode
echo "skip_device:"$skip_device
echo "precision_priority:"$precision_priority
echo "txda_skip_ops:"$txda_skip_ops
echo "txda_fallback_cpu_ops:"$txda_fallback_cpu_ops

#triton系统相关环境变量
TX8_DEPS_ROOT=$project_dir/tx8_deps
LLVM=$project_dir/llvm-a66376b0-ubuntu-x64
export TX8_DEPS_ROOT=$TX8_DEPS_ROOT
export LLVM_SYSPATH=$LLVM
export LLVM_BINARY_DIR=$LLVM/bin
export PYTHONPATH=$LLVM/python_packages/mlir_core:$PYTHONPATH
export LD_LIBRARY_PATH=$TX8_DEPS_ROOT/lib:$LD_LIBRARY_PATH
export TRITON_ALWAYS_COMPILE=1
#测试任务相关环境变量
export JSON_FILE_PATH=$project_dir/flaggems_tests
export PRECISION_PRIORITY=$precision_priority
export TRITON_ALLOW_NON_CONSTEXPR_GLOBALS=1
export TXDA_SKIP_OPS=$txda_skip_ops
export TXDA_FALLBACK_CPU_OPS=$txda_fallback_cpu_ops

echo "TX8_DEPS_ROOT="$TX8_DEPS_ROOT
echo "LLVM_SYSPATH="$LLVM_SYSPATH
echo "LLVM_BINARY_DIR="$LLVM_BINARY_DIR
echo "PYTHONPATH="$PYTHONPATH
echo "LD_LIBRARY_PATH="$LD_LIBRARY_PATH
echo "TRITON_ALWAYS_COMPILE="$TRITON_ALWAYS_COMPILE
echo "JSON_FILE_PATH="$JSON_FILE_PATH
echo "PRECISION_PRIORITY="$PRECISION_PRIORITY
echo "TRITON_ALLOW_NON_CONSTEXPR_GLOBALS="$TRITON_ALLOW_NON_CONSTEXPR_GLOBALS
echo "TXDA_SKIP_OPS="$TXDA_SKIP_OPS
echo "TXDA_FALLBACK_CPU_OPS="$TXDA_FALLBACK_CPU_OPS

source $project_dir/triton/.venv/bin/activate
if [ $quick_mode -eq 1 ]; then
	python3 $project_dir/flaggems_tests/test_flag_gems_ci.py --test_set $test_set --device_count $device_count --skip_device $skip_device --quick
else
	python3 $project_dir/flaggems_tests/test_flag_gems_ci.py --test_set $test_set --device_count $device_count --skip_device $skip_device
fi

if [ $? -eq 0 ]; then
    echo "Run test complete!"
else
    echo "Run test fail!!!"
    exit -1
fi
