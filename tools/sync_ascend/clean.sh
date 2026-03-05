#!/bin/bash

# 清理同步代码
# 警告：脚本中涉及路径的所有修改均会撤销！
#
# 用法:
#   bash tools/sync_ascend/clean.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

_find_git_root() {
    local _dir="$1"
    while [[ "${_dir}" != "/" ]]; do
        [[ -e "${_dir}/.git" ]] && { echo "${_dir}"; return 0; }
        _dir="$(dirname "${_dir}")"
    done
    return 1
}

FLAGTREE_DIR="$(_find_git_root "${SCRIPT_DIR}")" || {
    echo "[ERROR] 无法从 '${SCRIPT_DIR}' 向上找到 .git 目录，请在 flagtree 项目内执行脚本。" >&2
    exit 1
}

cd "${FLAGTREE_DIR}"

git stash
git clean -xdf bin/
git clean -xdf include/
git clean -xdf lib/
git clean -xdf python/
git clean -xdf test/
git clean -xdf third_party/ascend/

pushd third_party/flir/
git stash
git clean -xdf include/
git clean -xdf lib/
popd
