#!/bin/bash
# 从批跑日志提取所有测试用例，单步执行并保存独立日志
#
# 用法:
#   ./run_perf_single.sh <source_log> <output_dir> [custom_ops_value]
#
# 示例:
#   ./run_perf_single.sh /path/to/batch.log /path/to/out_dir_0 0
#   ./run_perf_single.sh /path/to/batch.log /path/to/out_dir_1 1

if [ $# -lt 2 ]; then
    echo "Usage: $0 <source_log> <output_dir> [custom_ops_value]"
    exit 1
fi

SOURCE_LOG="$1"
OUTPUT_DIR="$2"
CUSTOM_OPS="${3:-1}"

TRITON_DIR="/login_home/linyanfeng/triton_ws/triton"
DOCKER_CONTAINER="linyanfeng"

mkdir -p "$OUTPUT_DIR"

# Extract test case names from source log
# Lines look like: ../flaggems/tests/perf/perf_test.py::test_accuracy_abs[dtype0-shape0]
# Also support the old path: ../flaggems/tests/perf_test.py::...
grep -oP 'perf_test\.py::test_accuracy_\S+' "$SOURCE_LOG" | sed 's/\x1b\[[0-9;]*m//g' | sort -u > "$OUTPUT_DIR/test_list.txt"

total=$(wc -l < "$OUTPUT_DIR/test_list.txt")
echo "Total test cases: $total"
echo "Output dir: $OUTPUT_DIR"
echo "FLAG_GEMS_CUSTOM_OPS=$CUSTOM_OPS"

count=0
while IFS= read -r test_case; do
    count=$((count + 1))
    # Sanitize test name for filename
    safe_name=$(echo "$test_case" | sed 's/::/_/g' | sed 's/[\[\],]/_/g' | sed 's/__*/_/g' | sed 's/_$//')
    log_file="$OUTPUT_DIR/${safe_name}.log"

    if [ -f "$log_file" ]; then
        echo "[$count/$total] SKIP (exists): $test_case"
        continue
    fi

    # Strip "perf_test.py::" prefix if present (test_list has it from log extraction)
    test_name=$(echo "$test_case" | sed 's/^perf_test.py:://')

    echo "[$count/$total] RUN: $test_name"

    sudo docker start "$DOCKER_CONTAINER" > /dev/null 2>&1 || true
    sudo docker exec -w "$TRITON_DIR" "$DOCKER_CONTAINER" \
        bash -c "PYTHONPATH=/login_home/linyanfeng/triton_ws/flaggems/src:\$PYTHONPATH \
        FLAG_GEMS_CUSTOM_OPS=$CUSTOM_OPS \
        TX_LAUNCH_LOG_LEVEL=info \
        TX_LOG_LEVEL=info \
        ./third_party/tsingmicro/scripts/run_tsingmicro.sh \
        pytest ../flaggems/tests/perf/perf_test.py::$test_name \
        --ref cpu --mode quick -v" \
        > "$log_file" 2>&1

    # Check result
    if grep -q "PASSED" "$log_file" 2>/dev/null; then
        echo "       OK"
    elif grep -q "FAILED" "$log_file" 2>/dev/null; then
        echo "       FAILED"
    else
        echo "       ?"
    fi
done < "$OUTPUT_DIR/test_list.txt"

echo "Done. $count test cases. Logs in $OUTPUT_DIR"
