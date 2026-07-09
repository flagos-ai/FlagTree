#!/bin/bash
set -e

CUR_PATH="$( cd $( dirname ${BASH_SOURCE} );pwd )"

# CUR_PATH为脚本所在路径，进一步获取triton项目根路径
export SRC_HOME=$(realpath "${CUR_PATH}/../../../")

echo $SRC_HOME

hcu_file_path=${SRC_HOME}/python/test/unit
hcu_test_cmds=(
  "pytest ${hcu_file_path}/matmul.py"
  "pytest ${hcu_file_path}/rmsnorm.py"
  "python ${hcu_file_path}/vector-add.py --no_benchmark"
  "python ${hcu_file_path}/fused-softmax.py --no_benchmark"
  "python ${hcu_file_path}/matrix-multiplication.py --no_benchmark"
  "python ${hcu_file_path}/low-memory-dropout.py"
  #"python ${hcu_file_path}/layer-norm.py --no_benchmark" TODO: need check on bmz
  "pytest ${hcu_file_path}/fused-attention.py"
  "python ${hcu_file_path}/extern-functions.py"
  "python ${hcu_file_path}/grouped-gemm.py --no_benchmark"
  "python ${hcu_file_path}/persistent-matmul.py --no_benchmark"
)

function run_pytest() {
  for cmd in "${hcu_test_cmds[@]}"; do
    echo "$cmd .........."
    eval "$cmd"
    ret=$?
    if [ $ret -ne 0 ]; then
      echo "test hcu triton case ${cmd} failed!!"
      return $ret
    fi
  done
  echo -e "\n================="
  echo    "Run all passed!!!"
  echo -e "=================\n"
}

# pytest
run_pytest
