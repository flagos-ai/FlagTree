#!/bin/bash

set -ex

script=$(readlink -f "$0")
script_dir=$(dirname "$script")

source /usr/local/CANN_8.2.RC1.alpha002/ascend-toolkit/set_env.sh
export LLVM_BUILD_DIR=/opt/llvm-b5cc222

COMPILER_ROOT=/home/shared/bisheng_toolkit_20250922
BSIR_COMPILE_PATH=$(find "$COMPILER_ROOT" -name "bishengir-compile" | xargs dirname)
export PATH=${COMPILER_ROOT}:${BSIR_COMPILE_PATH}:$PATH
# FIXME: the 20250812 bishengir-compile requires the pairing bisheng compiler
export BISHENG_INSTALL_PATH=/home/shared/cann_compiler_20250812/compiler/ccec_compiler/bin

# 新增：定义统计文件路径
SUMMARY_FILE="${WORKSPACE}/ascend/examples/summary.txt"

function build_triton() {
  cd ${WORKSPACE}
  pip3 uninstall triton_ascend -y

  git submodule set-url third_party/triton https://gitee.com/shijingchang/triton.git
  git submodule sync && git submodule update --init --recursive

  bash scripts/build.sh ${WORKSPACE}/ascend ${LLVM_BUILD_DIR} 3.2.0 install 0
}

function run_case_by_multi_card() {
    NPU_DEVICES=$(ls /dev/davinci? 2>/dev/null | wc -l)
    [ $NPU_DEVICES -eq 0 ] && {
        echo "No Ascend devices found!"
        exit 1
    }

    echo "Detected $NPU_DEVICES Ascend devices"

    if [ -d ${WORKSPACE}triton ];then
      rm -rf ${WORKSPACE}triton
    fi

    if [ -d ~/.triton/dump ];then
      rm -rf ~/.triton/dump
    fi

    if [ -d ~/.triton/cache ];then
      rm -rf ~/.triton/cache
    fi

    test_dir=$1
    cd ${test_dir}

    # 清理旧日志
    rm -rf logs && mkdir logs

    # 记录测试开始时间
    start_time=$(date +"%Y-%m-%d %H:%M:%S")
    echo "===== 测试开始时间: ${start_time} ====="

    # 运行测试并捕获退出状态
    set +e
    pytest ${test_dir} -n auto --dist=loadfile -v --junitxml=logs/results.xml | tee logs/raw_output.log
    pytest_exit=$?
    set -e

    # 处理日志（添加设备标签）
    awk '
      />> Worker gw[0-9]+ using NPU device/ {
        split($0, parts, / /)
        dev_id = parts[NF]
        worker = parts[3]
        print "[" strftime("%Y-%m-%d %H:%M:%S") "| DEV-" dev_id "] " $0
        next
      }
      { print "[" strftime("%Y-%m-%d %H:%M:%S") "| DEV-" dev_id "] " $0 }
    ' logs/raw_output.log > logs/combined.log

    # 新增：解析测试结果统计
    total_tests=0
    passed_tests=0
    failed_tests=0
    skipped_tests=0
    error_tests=0

    # 使用Python解析JUnit XML报告
    python3 -c "
import xml.etree.ElementTree as ET
import os

xml_file = os.path.join('logs', 'results.xml')
if not os.path.exists(xml_file):
    print('JUnitXML report not found:', xml_file)
    exit(1)

tree = ET.parse(xml_file)
root = tree.getroot()

total = 0
passed = 0
failed = 0
skipped = 0
errors = 0

# 遍历所有testsuite
for testsuite in root.findall('testsuite'):
    total += int(testsuite.get('tests', 0))
    passed += int(testsuite.get('tests', 0)) - int(testsuite.get('errors', 0)) - int(testsuite.get('failures', 0)) - int(testsuite.get('skipped', 0))
    failed += int(testsuite.get('failures', 0))
    skipped += int(testsuite.get('skipped', 0))
    errors += int(testsuite.get('errors', 0))

print(f'total_tests={total}')
print(f'passed_tests={passed}')
print(f'failed_tests={failed}')
print(f'skipped_tests={skipped}')
print(f'error_tests={errors}')
" > logs/stats.tmp

    # 加载统计结果
    source logs/stats.tmp
    rm logs/stats.tmp

    # 记录测试结束时间
    end_time=$(date +"%Y-%m-%d %H:%M:%S")
    duration=$(( $(date -d "$end_time" +%s) - $(date -d "$start_time" +%s) ))
    duration_str=$(printf "%02dh %02dm %02ds" $((duration/3600)) $(((duration%3600)/60)) $((duration%60)))

    # 新增：生成统计摘要
    stats_summary="
===== generalization_cases测试统计摘要 =====
测试目录:       $(basename ${test_dir})
测试开始时间:   ${start_time}
测试结束时间:   ${end_time}
总耗时:         ${duration_str}
------------------------
总用例数:       ${total_tests}
成功用例:       ${passed_tests}
失败用例:       ${failed_tests}
跳过用例:       ${skipped_tests}
错误用例:       ${error_tests}
成功率:         $(( passed_tests * 100 / total_tests ))% (成功/总数)
设备数量:       ${NPU_DEVICES}
========================
"

    # 输出统计信息到控制台
    echo "${stats_summary}"

    # 追加统计信息到summary.txt
    echo "${stats_summary}" >> ${SUMMARY_FILE}

    echo "========================================"
    echo "All tests completed!"
    echo "JUnit Report: logs/results.xml"
    echo "Combined Log: logs/combined.log"
    echo "统计摘要已追加到: ${SUMMARY_FILE}"
    echo "========================================"

    zip_file=$2
    cd ${test_dir}/logs
    zip ${zip_file} combined.log
    cp ${zip_file} "/home/daily_log"

    # 返回pytest的退出状态
    return $pytest_exit
}

# build in torch 2.6.0
source /opt/miniconda3/bin/activate torch_260
build_triton

cd ${WORKSPACE}

# 初始化统计文件
echo "生成时间: $(date +"%Y-%m-%d %H:%M:%S")" >> ${SUMMARY_FILE}
echo "========================================" >> ${SUMMARY_FILE}

# run inductor cases
TEST_inductor_cases="${WORKSPACE}/ascend/examples/inductor_cases"
cd ${TEST_inductor_cases}
bash run_inductor_test.sh

# run gene case
zip_file="test_generalizetion_case_$(date +%Y%m%d).zip"
TEST_generalization="${WORKSPACE}/ascend/examples/generalization_cases"
run_case_by_multi_card ${TEST_generalization} ${zip_file}

echo "========================================" >> ${SUMMARY_FILE}

# run flaggems cases
TEST_flaggems_cases="${WORKSPACE}/ascend/examples/flaggems_cases"
cd ${TEST_flaggems_cases}
bash run_flaggems_test.sh

# copy summary.txt to /home/daily_log
cp ${SUMMARY_FILE} /home/daily_log