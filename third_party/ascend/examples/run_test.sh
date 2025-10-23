#!/bin/bash

set -ex

script=$(readlink -f "$0")
script_dir=$(dirname "$script")

# skiped script
skip_script=("bench_utils.py" "11-rab_time.py")

function uninstall_triton_ascend() {
  set +e
  while true; do
    pip3 uninstall triton_ascend -y | grep "Found existing installation"
    if [ $? -eq 1 ]; then
        echo "All triton_ascend versions are uninstalled"
        break
    fi
  done
  set -e
}

function build_triton() {

  cd ${WORKSPACE}
  # Run uninstall once because the while-loop does not stop. No idea why.
  # uninstall_triton_ascend
  pip3 uninstall triton_ascend -y

  git submodule set-url third_party/triton https://gitee.com/shijingchang/triton.git
  git submodule sync && git submodule update --init --recursive

  bash scripts/build.sh ${WORKSPACE}/ascend ${LLVM_BUILD_DIR} 3.2.0 install 0
}

function run_pytestcases() {
  if [ -d ${HOME}/.triton/dump ]; then
    rm -rf ${HOME}/.triton/dump
  fi
  if [ -d ${HOME}/.triton/cache ]; then
    rm -rf ${HOME}/.triton/cache
  fi

  cd ${script_dir}
  TARGET_DIR="$1"
  cd ${TARGET_DIR}
  pytest -n 16 --dist=load . || { exit 1 ; }

}

function run_pythoncases() {
  if [ -d ${HOME}/.triton/dump ]; then
    rm -rf ${HOME}/.triton/dump
  fi
  if [ -d ${HOME}/.triton/cache ]; then
    rm -rf ${HOME}/.triton/cache
  fi

  cd ${script_dir}
  TARGET_DIR="$1"
  cd ${TARGET_DIR}

  declare -a pids
  declare -A status_map
  has_failure=0

  # 查找并运行所有.py文件
  for test_script in *.py; do
    for skip_item in "${skip_script[@]}"; do
        if [ "$test_script" == "$skip_item" ]; then
            break
        fi
    done

    if [ -f "$test_script" ]; then
        echo "启动测试: $test_script"
        python "./$test_script" &
        pid=$!
        pids+=($pid)
        status_map[$pid]=$test_script
    fi
  done

  # 等待所有后台进程完成并检查状态
  for pid in "${pids[@]}"; do
      wait "$pid"
      exit_status=$?
      script_name=${status_map[$pid]}

      if [ $exit_status -ne 0 ]; then
          echo "[失败] $script_name - 退出码 $exit_status"
          has_failure=1
      else
          echo "[成功] $script_name"
      fi
  done

  echo "--------------------------------"

  # 根据测试结果退出
  if [ $has_failure -eq 1 ]; then
      echo "部分测试失败！"
      exit 1
  else
      echo "所有测试通过！"
      exit 0
  fi
}

function validate_git_commit_title() {
  if [ $# -lt 1 ]; then
    echo "Usage: $0 <commit_title>"
    exit 1
  fi
  commit_title=$1
  if ! echo "${commit_title}" | grep -qE "^(feat|fix|docs|style|refactor|test|chore|revert)(\(.*\))?: .+"; then
    echo "❌ The git commit title does not comply with the specifications!"
    echo "Format Requirements: <type>(<scope>): <subject>"
    echo "e.g.: feat(user): The login function is added."
    echo "Allowed Types: feat | fix | docs | style | refactor | test | chore | revert"
    exit 1
  fi
  echo "✅ The submitted information complies with the specifications."
}

function validate_pr_all_commits_title() {
  commit_titles=$(git log master..HEAD --oneline | sed 's/^[^ ]* //')
  if [ -z "$commit_titles" ]; then
    echo "No commits found between HEAD and master."
    exit 1
  fi
  echo "Validating commit titles..."
  echo "----------------------------"
  while IFS= read -r title; do
    echo "Checking: $title"
    if ! validate_git_commit_title "$title" 2>/dev/null; then
      echo "Error in commit: $title" >&2
      HAS_ERROR=true
    fi
  done <<< "$commit_titles"
  if [ "$HAS_ERROR" = true ]; then
    echo "----------------------------"
    echo "❌ Some commit titles do not meet the specifications." >&2
    exit 1
  else
    echo "----------------------------"
    echo "✅ All commit titles meet the specifications."
  fi
}

# if ! validate_pr_all_commits_title 2>/dev/null; then
#   exit 1
# fi

source /usr/local/CANN_8.2.RC1.alpha002/ascend-toolkit/set_env.sh
export LLVM_BUILD_DIR=/opt/llvm-b5cc222

# FIXME: 20250508 the bishengir-compile in the CANN 8.0.T115 fails lots of cases
#        So we need to use another version of compiler.
COMPILER_ROOT=/home/shared/bisheng_toolkit_20250922
BSIR_COMPILE_PATH=$(find "$COMPILER_ROOT" -name "bishengir-compile" | xargs dirname)
export PATH=${COMPILER_ROOT}:${BSIR_COMPILE_PATH}:$PATH
# FIXME: the 20250812 bishengir-compile requires the pairing bisheng compiler
export BISHENG_INSTALL_PATH=/home/shared/cann_compiler_20250812/compiler/ccec_compiler/bin

# build in torch 2.6.0
source /opt/miniconda3/bin/activate torch_260
build_triton

echo "Run ttir to linalg tests..."
cd ${WORKSPACE}/build/cmake.linux-aarch64-cpython-3.11
ninja check-triton-adapter-lit-tests
if [ $? -eq 0 ]; then
    echo "All ttir to linalg tests passed"
else
    echo "Some ttir to linalg tests failed"
    exit 1
fi

pytestcase_dir=("pytest_ut")
for test_dir in "${pytestcase_dir[@]}"; do
    echo "run pytestcase in ${test_dir}"
    run_pytestcases ${test_dir}
done

pythoncase_dir=("autotune_cases" "benchmark_cases" "tutorials")
for test_dir in "${pythoncase_dir[@]}"; do
    echo "run pythoncase in ${test_dir}"
    run_pythoncases ${test_dir}
done


