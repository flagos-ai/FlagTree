#!/bin/bash

set -e

# define color
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT_DIR="${GITHUB_WORKSPACE}/result"
TEST_FILE_PATH="${SCRIPT_DIR}"

# test files are named with pattern 'test_*.py'
TEST_FILES=$(find "${TEST_FILE_PATH}" -maxdepth 1 -type f -name "test_*.py")

function install_triton_ascend_build_output() {
    WHL_DIR=$1
    readarray -d '' WHL_FILES < <(find "$WHL_DIR" -maxdepth 1 -type f -name "*.whl" -print0)

    WHL_COUNT=${#WHL_FILES[@]}

    # ensure only one .whl file built
    if [ "$WHL_COUNT" -eq 0 ]; then
        echo -e "${RED}ERROR: no .whl file found.${NC}"
        return 1

    elif [ "$WHL_COUNT" -gt 1 ]; then
        echo -e "${RED}ERROR: expected 1 .whl file, find $WHL_COUNT files.${NC}"
        for file in "${WHL_FILES[@]}"; do
            echo -e "${RED}  - $(basename "$file")${NC}"
        done
        return 1

    else
        TARGET_WHL="${WHL_FILES[0]}"
        pip install --force-reinstall --no-deps "$TARGET_WHL"

        if [ $? -eq 0 ]; then
            echo -e "${GREEN}install $TARGET_WHL success!${NC}"
            return 0
        else
            echo -e "${RED}install $TARGET_WHL failed!${NC}"
            return 1
        fi
    fi
}

function run_test_cases() {
    echo "Start running dynamic-cv-pipeline test cases..."
    echo "-----------------------------------------------"

    SUCCESS_COUNT=0
    FAILURE_COUNT=0

    for file in $TEST_FILES; do
        filename=$(basename "$file")
        pytest -sv --confcutdir="${TEST_FILE_PATH}" "$file"

        if [ $? -eq 0 ]; then
            echo -e "${GREEN}test case $filename success!${NC}"
            SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
        else
            echo -e "${RED}test case $filename failed!${NC}"
            FAILURE_COUNT=$((FAILURE_COUNT + 1))
        fi
        echo "-----------------------------------------------"
    done

    echo -e "${RED}Reports:${NC}"
    echo -e "${RED}  Success cases count: $SUCCESS_COUNT${NC}"
    echo -e "${RED}  Failed cases count: $FAILURE_COUNT${NC}"

    if [ $FAILURE_COUNT -gt 0 ]; then
        return 1
    fi
}

function compare_mlir() {
    arg_num="$#"
    if [ $arg_num -lt 2 ]; then
        echo -e "${RED}ERROR: invalid argument number: $arg_num < 2${NC}"
        return 1
    fi

    BASE_DIR="$1"
    PR_DIR="$2"
    DIFF_OUTPUT_DIR=${3:-"${OUTPUT_DIR}/diff"}

    if ! [ -d "$BASE_DIR" ]; then
        echo -e "${RED}ERROR: base directory does not exist -> $BASE_DIR${NC}"
        return 1
    fi

    if ! [ -d "$PR_DIR" ]; then
        echo -e "${RED}ERROR: pr directory does not exist -> $PR_DIR${NC}"
        return 1
    fi

    BASE_DIR="${BASE_DIR%/}"
    PR_DIR="${PR_DIR%/}"

    python3 "$SCRIPT_DIR/mlir_diff.py" "$BASE_DIR" "$PR_DIR" "$DIFF_OUTPUT_DIR"

    echo -e "\n${GREEN}==================================================${NC}"
    echo -e "${GREEN}  Compare and diff finished${NC}"
    echo -e "${GREEN}==================================================${NC}"
}

function main() {
    rm -rf "${OUTPUT_DIR}"

    echo " Creating output dir ${OUTPUT_DIR}"
    mkdir -p "${OUTPUT_DIR}"

    # End if there are no test cases
    if [ -z "$TEST_FILES" ]; then
        echo -e "${YELLOW}No test cases found in $TEST_FILE_PATH${NC}"
        return 0
    fi

    # mlir output dir created by test cases
    TESTCASE_MLIR_OUTPUT=${TEST_FILE_PATH}/mlir_output

    PR_MLIR_DIR=${OUTPUT_DIR}/mlir/pr
    BASE_MLIR_DIR=${OUTPUT_DIR}/mlir/base
    rm -rf "${PR_MLIR_DIR}"
    rm -rf "${BASE_MLIR_DIR}"

    PR_PKG=${GITHUB_WORKSPACE}/pr/wheelhouse
    install_triton_ascend_build_output "$PR_PKG"

    if ! run_test_cases; then
        echo -e "${RED} Run test cases on pr package failed!${NC}"
        return 1
    fi

    mkdir -p "${PR_MLIR_DIR}"
    cp -r "${TESTCASE_MLIR_OUTPUT}/." "${PR_MLIR_DIR}/"

    BASE_PKG=${GITHUB_WORKSPACE}/base/wheelhouse
    install_triton_ascend_build_output "$BASE_PKG"

    # re-run test cases after installing base package
    # Will not return on failure
    if ! run_test_cases; then
        echo -e "${YELLOW} Run test cases on base package failed!${NC}"
    fi

    mkdir -p "${BASE_MLIR_DIR}"
    cp -r "${TESTCASE_MLIR_OUTPUT}/." "${BASE_MLIR_DIR}/"

    cd "${OUTPUT_DIR}"
    compare_mlir "${BASE_MLIR_DIR}" "${PR_MLIR_DIR}"
}

main "$@"
