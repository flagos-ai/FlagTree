#!/usr/bin/env bash
#
# 将 triton-ascend 仓库从 BASE_COMMIT 到 HEAD 的新 commits 同步到 FlagTree。
# 使用 git merge-file 做三路合并，保留 FlagTree 的特化修改，冲突处留标记供人工处理。
#
# 用法:
#   export TRITON_ASCEND_DIR=<path>   # 必填：源仓库 triton-ascend
#   export TRITON_DIR=<path>          # 可选：官方 triton，用于生成与 openai triton 的对比 diff
#   export FLIR_DIR=<path>            # 可选：flir 子仓库，默认 <flagtree根目录>/third_party/flir
#   bash tools/sync_ascend/sync_from_triton_ascend.sh [BASE_COMMIT]
#
# BASE_COMMIT 默认为 29d243e（上次同步点）。
# 脚本不会执行 git commit，所有变更保留为工作区修改。

set -euo pipefail

# ============================================================
# 定位 flagtree 项目根目录（从脚本所在目录向上查找 .git）
# ============================================================
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

# ============================================================
# 配置（优先读取环境变量）
# ============================================================
if [[ -z "${TRITON_ASCEND_DIR:-}" ]]; then
    echo "[ERROR] 请设置环境变量 TRITON_ASCEND_DIR，指向 triton-ascend 仓库路径。" >&2
    exit 1
fi

TRITON_DIR="${TRITON_DIR:-}"          # 可选，置空则跳过 openai triton 对比
FLIR_DIR="${FLIR_DIR:-${FLAGTREE_DIR}/third_party/flir}"

BASE_COMMIT="${1:-29d243e}"

MAPPER_PY="${SCRIPT_DIR}/path_mapper.py"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
WORK_DIR="/tmp/flagtree_sync_from_triton_ascend_${TIMESTAMP}"
REPORT_DIR="${FLAGTREE_DIR}/sync_report_${TIMESTAMP}"

mkdir -p "${WORK_DIR}" "${REPORT_DIR}"

MODIFIED_LIST="${REPORT_DIR}/modified.txt"
CREATED_LIST="${REPORT_DIR}/created.txt"
CONFLICT_LIST="${REPORT_DIR}/conflict.txt"
DELETED_LIST="${REPORT_DIR}/deleted.txt"
SKIPPED_LIST="${REPORT_DIR}/skipped.txt"

> "${MODIFIED_LIST}"
> "${CREATED_LIST}"
> "${CONFLICT_LIST}"
> "${DELETED_LIST}"
> "${SKIPPED_LIST}"

# 路径映射逻辑见 tools/path_mapper.py


# ============================================================
# 工具函数
# ============================================================

log_info()    { echo -e "\033[0;34m[INFO]\033[0m  $*"; }
log_ok()      { echo -e "\033[0;32m[OK]\033[0m    $*"; }
log_warn()    { echo -e "\033[0;33m[WARN]\033[0m  $*"; }
log_error()   { echo -e "\033[0;31m[ERROR]\033[0m $*"; }
log_conflict(){ echo -e "\033[0;35m[CONFLICT]\033[0m $*"; }

# 生成列表条目：路径相同时只展示一次，不同时展示同步方向
# 用法：fmt_entry <repo> <triton_file> <mapped_path>
fmt_entry() {
    local repo="$1" triton_file="$2" mapped_path="$3"
    if [[ "$repo" == "flagtree" && "$triton_file" == "$mapped_path" ]]; then
        echo "[flagtree] ${mapped_path}"
    else
        echo "[${repo}] ${mapped_path}  <-  [triton-ascend] ${triton_file}"
    fi
}

# 根据 repo 类型返回仓库根目录
repo_dir() {
    local repo="$1"
    if [[ "$repo" == "flir" ]]; then
        echo "${FLIR_DIR}"
    else
        echo "${FLAGTREE_DIR}"
    fi
}

# 解析 Python 输出：返回 status
get_status() {
    python3 -c "import sys,json; d=json.loads(sys.stdin.read()); print(d['status'])"
}

get_targets() {
    # 输出每行 "<repo>|<path>"
    python3 -c "
import sys, json
d = json.loads(sys.stdin.read())
for t in d['targets']:
    print(t['repo'] + '|' + t['path'])
"
}

get_manual_review() {
    python3 -c "
import sys, json
d = json.loads(sys.stdin.read())
print('1' if d.get('manual_review') else '0')
"
}

# 写入指定报告文件，若 MAP_JSON 标记了 manual_review 则追加 (需人工处理)
# 用法：_write_entry <list_file> <repo> <triton_file> <mapped_path> <map_json> [extra_suffix]
_write_entry() {
    local list_file="$1" repo="$2" triton_file="$3" mapped_path="$4" map_json="$5"
    local extra_suffix="${6:-}"
    local entry
    entry=$(fmt_entry "${repo}" "${triton_file}" "${mapped_path}")
    [[ -n "${extra_suffix}" ]] && entry="${entry}  ${extra_suffix}"
    if [[ "$(echo "${map_json}" | get_manual_review)" == "1" ]]; then
        echo "${entry}  (需人工处理)" >> "${list_file}"
    else
        echo "${entry}" >> "${list_file}"
    fi
}

# ============================================================
# Step 1: 获取变更文件列表
# ============================================================
log_info "=== Triton-Ascend → FlagTree 同步脚本 ==="
log_info "基准 commit: ${BASE_COMMIT}"
log_info "triton-ascend: ${TRITON_ASCEND_DIR}"
log_info "FlagTree:      ${FLAGTREE_DIR}"
log_info "FLIR:          ${FLIR_DIR}"
echo ""

cd "${TRITON_ASCEND_DIR}"

CURRENT_COMMIT=$(git rev-parse HEAD)
log_info "triton-ascend 当前 HEAD: ${CURRENT_COMMIT}"

# 确认 BASE_COMMIT 存在
if ! git rev-parse --verify "${BASE_COMMIT}" > /dev/null 2>&1; then
    log_error "BASE_COMMIT '${BASE_COMMIT}' 在 triton-ascend 仓库中不存在，请检查。"
    exit 1
fi

if [[ "${BASE_COMMIT}" == "$(git rev-parse HEAD)" ]]; then
    log_warn "BASE_COMMIT 等于当前 HEAD，没有新的 commits 需要同步。"
    exit 0
fi

log_info "新 commits 列表："
git log --oneline "${BASE_COMMIT}..${CURRENT_COMMIT}" | sed 's/^/  /'
echo ""

# 获取所有变更文件（含状态：A/M/D/R 等）
# 格式：<status>\t<文件路径>（R 类型为 <status>\t<旧路径>\t<新路径>）
CHANGED_FILES_RAW="${WORK_DIR}/changed_files_raw.txt"
git diff --name-status "${BASE_COMMIT}" "${CURRENT_COMMIT}" > "${CHANGED_FILES_RAW}"

TOTAL=$(wc -l < "${CHANGED_FILES_RAW}")
log_info "共 ${TOTAL} 个文件变更"
echo ""

# 生成 triton-ascend 基线到 HEAD 的完整 diff，存入报告目录
TRITON_ASCEND_DIFF="${REPORT_DIR}/triton_ascend_changes.diff"
git diff "${BASE_COMMIT}" "${CURRENT_COMMIT}" > "${TRITON_ASCEND_DIFF}"
log_info "triton-ascend diff 已生成: ${TRITON_ASCEND_DIFF}"
echo ""

# ============================================================
# Step 2: 同步前，生成 flagtree 与官方 Triton 的对比 diff
# ============================================================
TRITON_DIFF_DIR="${REPORT_DIR}/diff_with_openai_triton"
TRITON_DIFF_COUNT=0

if [[ ! -d "${TRITON_DIR}" ]]; then
    log_warn "TRITON_DIR '${TRITON_DIR}' 不存在，跳过 Triton 代码比较"
else
    log_info "[2/4] 生成同步前 flagtree 与官方 Triton 的对比..."
    mkdir -p "${TRITON_DIFF_DIR}"

    while IFS=$'\t' read -r _st _f1 _f2; do
        case "${_st:0:1}" in
            D) continue ;;          # 删除文件在 flagtree 侧无意义，跳过
            R) _tf="${_f2}" ;;      # 重命名取新路径
            *) _tf="${_f1}" ;;
        esac

        _mj=$(python3 "${MAPPER_PY}" "${_tf}")
        [[ "$(echo "${_mj}" | get_status)" != "ok" ]] && continue

        while IFS='|' read -r _repo _mpath; do
            [[ "$_repo" != "flagtree" ]] && continue
            [[ "$_mpath" == third_party/* ]] && continue

            _ft_file="${FLAGTREE_DIR}/${_mpath}"
            _triton_file="${TRITON_DIR}/${_mpath}"

            [[ ! -f "${_triton_file}" ]] && continue
            [[ ! -f "${_ft_file}" ]] && continue

            mkdir -p "${TRITON_DIFF_DIR}/$(dirname "${_mpath}")"
            _diff_file="${TRITON_DIFF_DIR}/${_mpath}.diff"

            if ! diff -u \
                --label "triton/${_mpath}" \
                --label "flagtree/${_mpath}" \
                "${_triton_file}" "${_ft_file}" > "${_diff_file}" 2>/dev/null; then
                log_info "  diff 已生成: ${_mpath}"
                (( TRITON_DIFF_COUNT++ )) || true
            else
                # 无差异：保留文件，写入仅含标题行的空 diff
                printf -- "--- triton/%s\n+++ flagtree/%s\n" "${_mpath}" "${_mpath}" > "${_diff_file}"
            fi
        done < <(echo "${_mj}" | get_targets)
    done < "${CHANGED_FILES_RAW}"

    if [[ ${TRITON_DIFF_COUNT} -gt 0 ]]; then
        log_ok "已生成 ${TRITON_DIFF_COUNT} 个与 openai triton 的对比 diff，目录: ${TRITON_DIFF_DIR}"
    else
        log_info "所有涉及文件与 openai triton 完全一致（或无对应文件）"
    fi
fi
echo ""

# ============================================================
# Step 3: 逐文件处理（应用同步）
# ============================================================
log_info "开始处理文件..."
echo ""

MODIFIED_COUNT=0
CREATED_COUNT=0
CONFLICT_COUNT=0
DELETED_COUNT=0
SKIPPED_COUNT=0

while IFS=$'\t' read -r status file1 file2; do
    # 处理 Rename（R）的情况：file1=旧路径，file2=新路径
    # 我们把重命名视为：删除 file1，添加 file2
    if [[ "${status}" == R* ]]; then
        old_file="${file1}"
        new_file="${file2}"

        # 处理旧路径（标记为已删除）
        MAP_JSON=$(python3 "${MAPPER_PY}" "${old_file}")
        MAP_STATUS=$(echo "${MAP_JSON}" | get_status)
        if [[ "${MAP_STATUS}" == "ok" ]]; then
            while IFS='|' read -r repo mapped_path; do
                [[ -z "${mapped_path}" ]] && continue
                _write_entry "${DELETED_LIST}" "${repo}" "${old_file}" "${mapped_path}" "${MAP_JSON}" "(RENAME_DEL)"
            done < <(echo "${MAP_JSON}" | get_targets)
        fi

        # 处理新路径（视为新增）
        status="A"
        file1="${new_file}"
    fi

    triton_file="${file1}"

    # 查询路径映射
    MAP_JSON=$(python3 "${MAPPER_PY}" "${triton_file}")
    MAP_STATUS=$(echo "${MAP_JSON}" | get_status)

    if [[ "${MAP_STATUS}" == "unknown" ]]; then
        log_warn "未知路径（跳过）: [triton-ascend] ${triton_file}"
        echo "[triton-ascend] ${triton_file}  (未配置忽略路径)" >> "${SKIPPED_LIST}"
        (( SKIPPED_COUNT++ )) || true
        continue
    fi

    if [[ "${MAP_STATUS}" == "skip" ]]; then
        # 不在同步范围（如 test/ 目录），静默记录到 skipped
        echo "[triton-ascend] ${triton_file}" >> "${SKIPPED_LIST}"
        (( SKIPPED_COUNT++ )) || true
        continue
    fi

    TARGETS=$(echo "${MAP_JSON}" | get_targets)
    if [[ -z "${TARGETS}" ]]; then
        # 明确跳过（如 .gitignore），不记录
        log_info "  跳过（不同步）: ${triton_file}"
        continue
    fi

    # 遍历每个映射目标
    while IFS='|' read -r repo mapped_path; do
        [[ -z "${mapped_path}" ]] && continue

        TARGET_REPO_DIR=$(repo_dir "${repo}")
        TARGET_ABS="${TARGET_REPO_DIR}/${mapped_path}"

        # ---------- 文件删除 ----------
        if [[ "${status}" == "D" ]]; then
            log_warn "  已删除（需人工确认是否同步删除）: ${triton_file}"
            _write_entry "${DELETED_LIST}" "${repo}" "${triton_file}" "${mapped_path}" "${MAP_JSON}"
            (( DELETED_COUNT++ )) || true
            continue
        fi

        # ---------- 文件新增 ----------
        if [[ "${status}" == "A" ]]; then
            # 确保目标目录存在
            mkdir -p "$(dirname "${TARGET_ABS}")"

            # 从 triton-ascend 复制文件内容
            TRITON_NEW_CONTENT="${WORK_DIR}/triton_new_$(echo "${triton_file}" | tr '/' '_')"
            git show "${CURRENT_COMMIT}:${triton_file}" > "${TRITON_NEW_CONTENT}" 2>/dev/null || {
                log_warn "  无法获取新文件内容: ${triton_file}"
                echo "[${repo}] ${mapped_path}  <-  [triton-ascend] ${triton_file}  (无法读取)" >> "${SKIPPED_LIST}"
                (( SKIPPED_COUNT++ )) || true
                continue
            }

            if [[ -f "${TARGET_ABS}" ]]; then
                # 目标已存在，做差异报告
                log_warn "  新增但目标已存在（做 merge）: [${repo}] ${mapped_path}"
            else
                cp "${TRITON_NEW_CONTENT}" "${TARGET_ABS}"
                log_ok "  新增: [${repo}] ${mapped_path}"
                _write_entry "${CREATED_LIST}" "${repo}" "${triton_file}" "${mapped_path}" "${MAP_JSON}"
                (( CREATED_COUNT++ )) || true
                continue
            fi
        fi

        # ---------- 文件修改（包含上面"已存在"的新增走到这里的情况）----------
        # 获取 triton-ascend 中的 base 版本（BASE_COMMIT 处的内容）
        TRITON_BASE="${WORK_DIR}/base_$(echo "${mapped_path}" | tr '/' '_')"
        TRITON_NEW="${WORK_DIR}/new_$(echo "${mapped_path}" | tr '/' '_')"

        if ! git show "${BASE_COMMIT}:${triton_file}" > "${TRITON_BASE}" 2>/dev/null; then
            # BASE_COMMIT 时该文件不存在（说明是新增文件但 status 不是 A，异常情况）
            # 退而用空文件作为 base
            touch "${TRITON_BASE}"
        fi

        if ! git show "${CURRENT_COMMIT}:${triton_file}" > "${TRITON_NEW}" 2>/dev/null; then
            log_warn "  无法读取 triton-ascend 新版本: ${triton_file}"
            echo "[${repo}] ${mapped_path}  <-  [triton-ascend] ${triton_file}  (无法读取新版本)" >> "${SKIPPED_LIST}"
            (( SKIPPED_COUNT++ )) || true
            continue
        fi

        # 目标文件不存在时，直接创建
        if [[ ! -f "${TARGET_ABS}" ]]; then
            mkdir -p "$(dirname "${TARGET_ABS}")"
            cp "${TRITON_NEW}" "${TARGET_ABS}"
            log_ok "  创建（目标不存在）: [${repo}] ${mapped_path}"
            _write_entry "${CREATED_LIST}" "${repo}" "${triton_file}" "${mapped_path}" "${MAP_JSON}"
            (( CREATED_COUNT++ )) || true
            continue
        fi

        # 三路合并：
        #   current = FlagTree 当前版本（含 FlagTree 特化改动）
        #   base    = triton-ascend @ BASE_COMMIT（上次同步点）
        #   other   = triton-ascend @ HEAD（新版本）
        # git merge-file 会将结果写回 current 文件
        #   退出码 0  → 无冲突
        #   退出码 >0 → 有冲突（文件中已写入 <<<<<<< 标记）
        #   退出码 <0 → 错误

        MERGE_EXIT=0
        git merge-file \
            -L "flagtree/${mapped_path}" \
            -L "triton-ascend@${BASE_COMMIT}/${triton_file}" \
            -L "triton-ascend@HEAD/${triton_file}" \
            "${TARGET_ABS}" \
            "${TRITON_BASE}" \
            "${TRITON_NEW}" || MERGE_EXIT=$?

        if [[ ${MERGE_EXIT} -eq 0 ]]; then
            log_ok "  已合并: [${repo}] ${mapped_path}"
            _write_entry "${MODIFIED_LIST}" "${repo}" "${triton_file}" "${mapped_path}" "${MAP_JSON}"
            (( MODIFIED_COUNT++ )) || true
        elif [[ ${MERGE_EXIT} -gt 0 ]]; then
            log_conflict "  合并冲突（请人工处理）: [${repo}] ${mapped_path}"
            _write_entry "${CONFLICT_LIST}" "${repo}" "${triton_file}" "${mapped_path}" "${MAP_JSON}" "(${MERGE_EXIT} 处冲突)"
            (( CONFLICT_COUNT++ )) || true
        else
            log_error "  merge-file 错误: [${repo}] ${mapped_path}"
            echo "[${repo}] ${mapped_path}  (merge-file error)" >> "${SKIPPED_LIST}"
            (( SKIPPED_COUNT++ )) || true
        fi

    done <<< "${TARGETS}"

done < "${CHANGED_FILES_RAW}"

# ============================================================
# Step 4: 对无需人工处理的新增/删除文件执行 git 暂存操作
# ============================================================
log_info ""
log_info "[4/5] 自动 git add / git rm..."

GIT_ADD_COUNT=0
GIT_RM_COUNT=0

# 从列表行中提取 [repo] path，跳过含 (需人工处理) 的行
# 返回 "<repo> <path>" 或空
_parse_entry_no_manual() {
    local _ln="$1"
    [[ "$_ln" == *"(需人工处理)"* ]] && return
    if [[ "$_ln" =~ ^\[([a-zA-Z]+)\][[:space:]]+([^[:space:]]+) ]]; then
        echo "${BASH_REMATCH[1]} ${BASH_REMATCH[2]}"
    fi
}

# git add：遍历 created.txt
while IFS= read -r _line || [[ -n "$_line" ]]; do
    [[ -z "$_line" ]] && continue
    _parsed=$(_parse_entry_no_manual "$_line")
    [[ -z "$_parsed" ]] && continue
    _repo=$(echo "$_parsed" | cut -d' ' -f1)
    _path=$(echo "$_parsed" | cut -d' ' -f2-)
    _repo_dir=$(repo_dir "${_repo}")
    _abs="${_repo_dir}/${_path}"
    if [[ -f "$_abs" ]]; then
        (cd "${_repo_dir}" && git add "${_path}") \
            && { log_ok "  git add: [${_repo}] ${_path}"; (( GIT_ADD_COUNT++ )) || true; } \
            || log_warn "  git add 失败: [${_repo}] ${_path}"
    else
        log_warn "  git add 跳过（文件不存在）: [${_repo}] ${_path}"
    fi
done < "${CREATED_LIST}"

# git rm：遍历 deleted.txt，跳过 RENAME_DEL 条目（已由新增侧处理）
while IFS= read -r _line || [[ -n "$_line" ]]; do
    [[ -z "$_line" ]] && continue
    [[ "$_line" == *"(RENAME_DEL)"* ]] && continue
    _parsed=$(_parse_entry_no_manual "$_line")
    [[ -z "$_parsed" ]] && continue
    _repo=$(echo "$_parsed" | cut -d' ' -f1)
    _path=$(echo "$_parsed" | cut -d' ' -f2-)
    _repo_dir=$(repo_dir "${_repo}")
    _abs="${_repo_dir}/${_path}"
    if [[ -f "$_abs" ]]; then
        (cd "${_repo_dir}" && git rm "${_path}") \
            && { log_ok "  git rm: [${_repo}] ${_path}"; (( GIT_RM_COUNT++ )) || true; } \
            || log_warn "  git rm 失败: [${_repo}] ${_path}"
    else
        log_warn "  git rm 跳过（文件不存在）: [${_repo}] ${_path}"
    fi
done < "${DELETED_LIST}"

log_info "自动暂存完成：git add ${GIT_ADD_COUNT} 个，git rm ${GIT_RM_COUNT} 个"

# ============================================================
# Step 5: 生成汇总报告
# ============================================================
SUMMARY="${REPORT_DIR}/summary.txt"
{
    echo "========================================="
    echo "  Triton-Ascend → FlagTree 同步报告"
    echo "========================================="
    echo "同步时间:     $(date '+%Y-%m-%d %H:%M:%S')"
    echo "triton-ascend: ${TRITON_ASCEND_DIR}"
    echo "BASE_COMMIT:  ${BASE_COMMIT}"
    echo "HEAD_COMMIT:  ${CURRENT_COMMIT}"
    echo ""
    _rdir=$(basename "${REPORT_DIR}")
    echo "统计:"
    echo "  已成功合并:       ${MODIFIED_COUNT} 个文件"
    echo "  新增创建:         ${CREATED_COUNT} 个文件"
    echo "  存在合并冲突:     ${CONFLICT_COUNT} 个文件"
    echo "  已删除:           ${DELETED_COUNT} 个文件  ← 需人工处理"
    echo "  跳过/未知路径:    ${SKIPPED_COUNT} 个文件"
    echo ""
    echo "详细清单:"
    echo "  ${_rdir}/modified.txt"
    echo "  ${_rdir}/created.txt"
    echo "  ${_rdir}/conflict.txt"
    echo "  ${_rdir}/deleted.txt"
    echo "  ${_rdir}/skipped.txt"
    echo ""
    echo "flagtree 与 openai triton 原有 diff（${TRITON_DIFF_COUNT} 个文件）:"
    echo "  ${_rdir}/diff_with_openai_triton/"
    echo ""
    echo "triton-ascend 变更 diff:"
    echo "  ${_rdir}/triton_ascend_changes.diff"
    echo "========================================="
} | tee "${SUMMARY}"

echo ""
log_info "报告目录: ${REPORT_DIR}"
echo ""

if [[ ${CONFLICT_COUNT} -gt 0 ]]; then
    echo ""
    log_warn "以下文件存在合并冲突，请人工检查并解决（文件中已含 <<<<<<< 标记）："
    cat "${CONFLICT_LIST}"
    echo ""
fi

if [[ ${DELETED_COUNT} -gt 0 ]]; then
    echo ""
    log_warn "以下 triton-ascend 文件已被删除，请人工确认 FlagTree 侧是否也需同步删除："
    cat "${DELETED_LIST}"
    echo ""
fi

if [[ ${SKIPPED_COUNT} -gt 0 ]]; then
    echo ""
    log_warn "以下文件已跳过（不在同步范围内或路径未匹配），如有需要请人工处理："
    cat "${SKIPPED_LIST}"
    echo ""
fi

# 临时工作目录保留供调试，可手动删除
log_info "临时工作目录（可手动删除）: ${WORK_DIR}"
log_info "报告目录: ${REPORT_DIR}"
log_info "摘要：${SUMMARY}"
log_info "完成"
