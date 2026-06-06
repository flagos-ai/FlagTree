#!/bin/bash
# notice: this script supports python3.11.x

set -ex

script=$(readlink -f "$0")
script_dir=$(dirname "$script")

# clean old logs
mkdir -p /home/pr_test_log
UNITTEST_DIR="triton-ascend/third_party/ascend/unittest"

# define summary file path
SUMMARY_FILE="${WORKSPACE}/${UNITTEST_DIR}/summary.txt"

function clean_cache() {
    if [ -d /tmp/torchinductor_* ];then
      rm -rf /tmp/torchinductor_*
    fi
    if [ -d ${HOME}/.triton/dump ];then
      rm -rf ${HOME}/.triton/dump
    fi

    if [ -d ${HOME}/.triton/cache ];then
      rm -rf ${HOME}/.triton/cache
    fi
}


function run_case_by_multi_card() {
    NPU_DEVICES=$(ls /dev/davinci[0-9]* 2>/dev/null | wc -l)
    [ $NPU_DEVICES -eq 0 ] && {
        echo "No Ascend devices found!"
        exit 1
    }

    echo "Detected $NPU_DEVICES Ascend devices"

    test_dir=$1
    cd ${test_dir}

    # clean logs
    rm -rf logs && mkdir logs

    # record start time
    start_time=$(date +"%Y-%m-%d %H:%M:%S")
    echo "===== Test Start Time: ${start_time} ====="

    # run tests and capture exit status
    set +e
    python -m pytest ${test_dir} -n auto --dist=loadfile -v --junitxml=logs/results.xml | tee logs/raw_output.log
    pytest_exit=$?
    set -e

    # process logs (add device tags)
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

    # parse test result statistics
    total_tests=0
    passed_tests=0
    failed_tests=0
    skipped_tests=0
    error_tests=0

    # use Python to parse JUnit XML report
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

    # load stats
    source logs/stats.tmp
    rm logs/stats.tmp

    # record end time
    end_time=$(date +"%Y-%m-%d %H:%M:%S")
    duration=$(( $(date -d "$end_time" +%s) - $(date -d "$start_time" +%s) ))
    duration_str=$(printf "%02dh %02dm %02ds" $((duration/3600)) $(((duration%3600)/60)) $((duration%60)))

    # generate summary
    stats_summary="
===== Test Summary - [generalization_cases] =====
Test Directory:       $(basename ${test_dir})
Test Start Time:   ${start_time}
Test End Time:   ${end_time}
Total Duration:         ${duration_str}
------------------------
Total Tests:       ${total_tests}
Passed Tests:       ${passed_tests}
Failed Tests:       ${failed_tests}
Skipped Tests:       ${skipped_tests}
Error Tests:       ${error_tests}
Success Rate:         $(( passed_tests * 100 / total_tests ))% (Passed/Total)
NPU Devices:       ${NPU_DEVICES}
========================
"

    # output stats summary to console
    echo "${stats_summary}"

    # append stats summary to summary.txt
    echo "${stats_summary}" >> ${SUMMARY_FILE}

    echo "========================================"
    echo "All tests completed!"
    echo "JUnit Report: logs/results.xml"
    echo "Combined Log: logs/combined.log"
    echo "Stats Summary has been appended to: ${SUMMARY_FILE}"
    echo "========================================"

    zip_file=$2
    cd ${test_dir}/logs
    zip ${zip_file} combined.log
    cp ${zip_file} "/home/pr_test_log"

    # return pytest exit status
    return $pytest_exit
}

# initialize stats file
echo "Generate Time: $(date +"%Y-%m-%d %H:%M:%S")" >> ${SUMMARY_FILE}
echo "========================================" >> ${SUMMARY_FILE}

# run gene case
zip_file="test_generalization_case_$(date +%Y%m%d).zip"
TEST_generalization="${WORKSPACE}/${UNITTEST_DIR}/generalization_cases"
clean_cache
run_case_by_multi_card ${TEST_generalization} ${zip_file}

echo "========================================" >> ${SUMMARY_FILE}

# copy summary.txt to /home/pr_test_log
cp ${SUMMARY_FILE} /home/pr_test_log
