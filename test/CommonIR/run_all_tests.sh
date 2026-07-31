#!/bin/bash
# ==============================================================================
# run_all_tests.sh — 批量运行当前目录下所有 .py 测试用例
#
# 用法:
#   ./run_all_tests.sh              # 运行所有 .py（排除自身和 bench_*）
#   ./run_all_tests.sh --include-bench  # 同时运行 bench_* 文件
#   ./run_all_tests.sh fa_cv.py native_matmul.py  # 只运行指定文件
# ==============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# 结果统计
PASSED=0
FAILED=0
SKIPPED=0
TOTAL=0

FAILED_LIST=()
INCLUDE_BENCH=false
SPECIFIED_FILES=()

# ---------- 参数解析 ----------
for arg in "$@"; do
    case "$arg" in
        --include-bench)
            INCLUDE_BENCH=true
            ;;
        *.py)
            SPECIFIED_FILES+=("$arg")
            ;;
        *)
            echo "未知参数: $arg"
            exit 1
            ;;
    esac
done

# ---------- 收集测试文件 ----------
collect_test_files() {
    if [ ${#SPECIFIED_FILES[@]} -gt 0 ]; then
        printf '%s\n' "${SPECIFIED_FILES[@]}"
        return
    fi

    for f in *.py; do
        # 跳过本脚本同名 py（如有）
        [[ "$f" == "run_all_tests.py" ]] && continue
        # 默认跳过 bench 文件（除非 --include-bench）
        if [[ "$f" == bench_* ]] && [ "$INCLUDE_BENCH" = false ]; then
            continue
        fi
        echo "$f"
    done
}

# ---------- 运行单个测试 ----------
run_one() {
    local py_file="$1"
    local log_file="/tmp/test_${py_file%.py}.log"

    TOTAL=$((TOTAL + 1))

    if [ ! -f "$py_file" ]; then
        echo -e "  ${YELLOW}[SKIP]${NC} $py_file (文件不存在)"
        SKIPPED=$((SKIPPED + 1))
        return
    fi

    printf "  [%2d] %-40s ... " "$TOTAL" "$py_file"

    # 运行测试，超时 300 秒
    if timeout 300 python "$py_file" > "$log_file" 2>&1; then
        echo -e "${GREEN}PASS${NC}"
        PASSED=$((PASSED + 1))
    else
        local exit_code=$?
        if [ $exit_code -eq 124 ]; then
            echo -e "${YELLOW}TIMEOUT${NC}"
        else
            echo -e "${RED}FAIL${NC} (exit=$exit_code)"
        fi
        FAILED=$((FAILED + 1))
        FAILED_LIST+=("$py_file")
        # 打印最后 10 行日志方便排查
        echo -e "       ${CYAN}--- last 10 lines of log ---${NC}"
        tail -10 "$log_file" | sed 's/^/       /'
        echo ""
    fi
}

# ---------- 主流程 ----------
echo "============================================================"
echo " CommonIR 批量测试"
echo " 目录: $SCRIPT_DIR"
echo " 时间: $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================================"
echo ""

mapfile -t TEST_FILES < <(collect_test_files)

if [ ${#TEST_FILES[@]} -eq 0 ]; then
    echo "没有找到可运行的 .py 测试文件。"
    exit 0
fi

echo "共发现 ${#TEST_FILES[@]} 个测试文件:"
echo ""

for f in "${TEST_FILES[@]}"; do
    run_one "$f"
done

# ---------- 汇总 ----------
echo ""
echo "============================================================"
echo -e " 结果汇总: 总计=${TOTAL}  ${GREEN}通过=${PASSED}${NC}  ${RED}失败=${FAILED}${NC}  ${YELLOW}跳过=${SKIPPED}${NC}"
echo "============================================================"

if [ ${#FAILED_LIST[@]} -gt 0 ]; then
    echo ""
    echo -e " ${RED}失败用例:${NC}"
    for f in "${FAILED_LIST[@]}"; do
        echo "   - $f"
    done
    echo ""
    exit 1
fi

exit 0
