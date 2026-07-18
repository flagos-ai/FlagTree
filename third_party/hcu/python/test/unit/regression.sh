#!/bin/bash

# Copyright 2026 FlagOS Contributors
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

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
