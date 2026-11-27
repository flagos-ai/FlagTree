#!/bin/bash
# Shared environment resolution for the tsingmicro Triton scripts.
#
# This file is meant to be sourced, not executed.  It only fills in defaults
# and never changes the command line interface of the callers.
#
# Resolution rules (no machine-specific paths):
#   TRITON          repository root containing this file
#   WORKSPACE       parent of the repository root (override with WORKSPACE)
#   LLVM_SYSPATH    $LLVM_SYSPATH, then $WORKSPACE/llvm-<hash> or llvm-<major>,
#                   then ~/.tsingmicro and ~/.triton/tsingmicro fallbacks
#   TX8_DEPS_ROOT   $TX8_DEPS_ROOT, then $WORKSPACE/tx8_deps, then fallbacks
#   PYTHON          $PYTHON, then $TRITON/.venv/bin/python3, then python3
#
# The expected LLVM major version comes from the Triton version in setup.py
# (3.6 -> 22, 3.7 -> 23) and the expected commit hash from cmake/llvm-hash.txt.

_tsm_lib_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TSM_SCRIPTS_DIR="$(cd "$_tsm_lib_dir/.." && pwd)"
PROJECT_DIR="$(cd "$_tsm_lib_dir/../../../.." && pwd)"
if [ ! -f "$PROJECT_DIR/setup.py" ]; then
    # Installed bundle layout: <package>/lib/env.sh, no source tree.
    PROJECT_DIR="$(cd "$_tsm_lib_dir/.." && pwd)"
fi
export TSM_SCRIPTS_DIR PROJECT_DIR

TRITON="${TRITON:-$PROJECT_DIR}"
if [ -z "${WORKSPACE:-}" ]; then
    # The workspace holds tx8_deps/ and llvm-*/.  A checkout may live directly
    # under it or inside a worktree subdirectory, so probe nearby levels.
    WORKSPACE="$PROJECT_DIR"
    for _tsm_cand in "$PROJECT_DIR" "$PROJECT_DIR/.." "$PROJECT_DIR/../.."; do
        if [ -d "$_tsm_cand/tx8_deps" ] || ls -d "$_tsm_cand"/llvm-* >/dev/null 2>&1; then
            WORKSPACE="$(cd "$_tsm_cand" && pwd)"
            break
        fi
    done
fi
export TRITON WORKSPACE

_tsm_version="$(sed -n 's/^TRITON_VERSION = "\([0-9][0-9]*\.[0-9][0-9]*\)\..*/\1/p' \
    "$PROJECT_DIR/setup.py" 2>/dev/null | head -1)"
case "$_tsm_version" in
    3.6) _tsm_llvm_major=22 ;;
    3.7) _tsm_llvm_major=23 ;;
    *)   _tsm_llvm_major="" ;;
esac
_tsm_llvm_hash="$(head -c 8 "$PROJECT_DIR/cmake/llvm-hash.txt" 2>/dev/null || true)"

_tsm_first_dir() {
    local dir
    for dir in "$@"; do
        [ -n "$dir" ] || continue
        [ -d "$dir" ] && { printf '%s\n' "$dir"; return 0; }
    done
    return 1
}

# A usable LLVM prefix must have both a compiler and the MLIR tools.
_tsm_first_llvm() {
    local dir
    for dir in "$@"; do
        [ -n "$dir" ] || continue
        if [ -x "$dir/bin/clang" ] && [ -x "$dir/bin/mlir-opt" ]; then
            printf '%s\n' "$dir"
            return 0
        fi
    done
    return 1
}

if [ -n "${LLVM_SYSPATH:-}" ] && [ ! -d "$LLVM_SYSPATH" ]; then
    echo "[env] WARN: LLVM_SYSPATH=$LLVM_SYSPATH does not exist; trying auto-detection" >&2
    LLVM_SYSPATH=""
fi
if [ -z "${LLVM_SYSPATH:-}" ]; then
    _tsm_llvm_candidates=()
    if [ -n "$_tsm_llvm_hash" ]; then
        _tsm_llvm_candidates+=("$WORKSPACE/llvm-$_tsm_llvm_hash")
        for _tsm_dir in "$WORKSPACE"/llvm-"$_tsm_llvm_hash"*; do
            _tsm_llvm_candidates+=("$_tsm_dir")
        done
    fi
    if [ -n "$_tsm_llvm_major" ]; then
        for _tsm_dir in "$WORKSPACE"/llvm-"$_tsm_llvm_major"*; do
            _tsm_llvm_candidates+=("$_tsm_dir")
        done
        _tsm_llvm_candidates+=("$HOME/.tsingmicro/llvm-$_tsm_llvm_major")
        _tsm_llvm_candidates+=("$HOME/.triton/tsingmicro/llvm-$_tsm_llvm_major")
    else
        # Installed bundle without setup.py: accept any llvm-* next to it.
        for _tsm_dir in "$WORKSPACE"/llvm-*; do
            _tsm_llvm_candidates+=("$_tsm_dir")
        done
    fi
    LLVM_SYSPATH="$(_tsm_first_llvm "${_tsm_llvm_candidates[@]}" || true)"
fi

if [ -z "${TX8_DEPS_ROOT:-}" ] || [ ! -d "${TX8_DEPS_ROOT:-}" ]; then
    TX8_DEPS_ROOT="$(_tsm_first_dir \
        "$WORKSPACE/tx8_deps" \
        "$HOME/.tsingmicro/tx8_deps" \
        "$HOME/.triton/tsingmicro/tx8_deps" || true)"
fi

if [ -z "${PYTHON:-}" ]; then
    if [ -x "$TRITON/.venv/bin/python3" ]; then
        PYTHON="$TRITON/.venv/bin/python3"
    else
        PYTHON="$(command -v python3 || echo python3)"
    fi
fi

# nlohmann/json headers, required when TRITON_OFFLINE_BUILD is enabled.
if [ -z "${JSON_SYSPATH:-}" ]; then
    for _tsm_json in "$HOME/.triton/json" "$HOME/.tsingmicro/json" "$WORKSPACE/json"; do
        if [ -d "$_tsm_json/include" ] || [ -d "$_tsm_json/single_include" ]; then
            JSON_SYSPATH="$_tsm_json"
            break
        fi
    done
fi
export JSON_SYSPATH

export LLVM_SYSPATH
[ -n "$LLVM_SYSPATH" ] && export LLVM_BINARY_DIR="$LLVM_SYSPATH/bin"
[ -n "$LLVM_SYSPATH" ] && LLD_DIR="${LLD_DIR:-$LLVM_SYSPATH/lib/cmake/lld}"
export LLD_DIR
export TX8_DEPS_ROOT
export PYTHON

# Wheel version stamp: "<base>.post<YYYYMMDDHH>+<backend>". Set here so the
# packaging code stays upstream-identical; other backends never source this file.
export FLAGTREE_WHEEL_VERSION="${FLAGTREE_WHEEL_VERSION:-0.7.0.post$(date +%Y%m%d%H)+${FLAGTREE_BACKEND:-tsingmicro}}"

# Prepend directories without duplicating existing entries.  Safe to source
# more than once.
_tsm_add_paths() {
    local var="$1"; shift
    local current="${!var:-}" dir add=""
    for dir in "$@"; do
        [ -n "$dir" ] && [ -d "$dir" ] || continue
        case ":$current:$add:" in
            *":$dir:"*) continue ;;
        esac
        add="${add:+$add:}$dir"
    done
    if [ -n "$add" ]; then
        printf -v "$var" '%s%s' "$add" "${current:+:$current}"
        export "$var"
    fi
}

[ -n "$LLVM_SYSPATH" ] && _tsm_add_paths PYTHONPATH "$LLVM_SYSPATH/python_packages/mlir_core"
# CMake selects clang/clang++ by name for the tsingmicro backend.
[ -n "$LLVM_SYSPATH" ] && _tsm_add_paths PATH "$LLVM_SYSPATH/bin"
_tsm_add_paths LD_LIBRARY_PATH \
    "$TX8_DEPS_ROOT/lib" \
    "${KUIPER_ROOT:-/usr/local/kuiper}/lib" \
    "${KUIPER_ROOT:-/usr/local/kuiper}/tsm8-profiler/lib"

# Strict validation used by build scripts.
tsm_env_check() {
    local err=0
    if [ -z "$LLVM_SYSPATH" ] || [ ! -d "$LLVM_SYSPATH" ]; then
        echo "[env] ERROR: LLVM not found; set LLVM_SYSPATH or place llvm-<hash>/llvm-<major> under $WORKSPACE" >&2
        err=1
    else
        local bin
        for bin in clang clang++ mlir-opt mlir-translate FileCheck; do
            [ -x "$LLVM_SYSPATH/bin/$bin" ] || {
                echo "[env] ERROR: missing $LLVM_SYSPATH/bin/$bin" >&2
                err=1
            }
        done
        [ -x "$LLVM_SYSPATH/bin/ld.lld" ] || \
            echo "[env] WARN: missing $LLVM_SYSPATH/bin/ld.lld (required by TRITON_BUILD_WITH_CLANG_LLD)" >&2
        [ -d "$LLVM_SYSPATH/python_packages/mlir_core" ] || {
            echo "[env] ERROR: missing $LLVM_SYSPATH/python_packages/mlir_core" >&2
            err=1
        }
        local actual_major
        actual_major="$("$LLVM_SYSPATH/bin/mlir-opt" --version 2>/dev/null \
            | sed -n 's/.*version \([0-9][0-9]*\)\..*/\1/p' | head -1)"
        if [ -n "$_tsm_llvm_major" ] && [ -n "$actual_major" ] && [ "$actual_major" != "$_tsm_llvm_major" ]; then
            echo "[env] ERROR: LLVM major $actual_major does not match Triton $_tsm_version (expected $_tsm_llvm_major)" >&2
            err=1
        fi
    fi
    if [ -z "$TX8_DEPS_ROOT" ] || [ ! -d "$TX8_DEPS_ROOT" ]; then
        echo "[env] ERROR: tx8_deps not found; set TX8_DEPS_ROOT or place it under $WORKSPACE" >&2
        err=1
    fi
    if [ -n "${LLD_DIR:-}" ] && [ ! -d "$LLD_DIR" ]; then
        echo "[env] WARN: LLD_DIR does not exist: $LLD_DIR" >&2
    fi
    if [ -n "${TRITON_OFFLINE_BUILD:-}" ] && [ ! "${TRITON_OFFLINE_BUILD}" = "0" ] \
        && [ -z "${JSON_SYSPATH:-}" ]; then
        echo "[env] ERROR: offline build needs nlohmann/json; set JSON_SYSPATH or unpack it to $WORKSPACE/json" >&2
        err=1
    fi
    "$PYTHON" -c 'import sys' >/dev/null 2>&1 || {
        echo "[env] ERROR: PYTHON is not runnable: $PYTHON" >&2
        err=1
    }
    local want_py="" got_py
    case "$_tsm_version" in
        3.6) want_py=3.10 ;;
        3.7) want_py=3.12 ;;
    esac
    got_py="$("$PYTHON" -c 'import sys; print("%d.%d" % sys.version_info[:2])' 2>/dev/null)"
    if [ -n "$want_py" ] && [ -n "$got_py" ] && [ "$got_py" != "$want_py" ]; then
        echo "[env] WARN: Triton $_tsm_version expects Python $want_py, PYTHON resolves to $got_py" >&2
    fi
    [ -d "${KUIPER_ROOT:-/usr/local/kuiper}/lib" ] || \
        echo "[env] WARN: kuiper lib not found (KUIPER_ROOT=${KUIPER_ROOT:-/usr/local/kuiper})" >&2
    command -v readelf >/dev/null 2>&1 || \
        echo "[env] WARN: readelf not found; glibc ABI checks are skipped" >&2
    command -v unzip >/dev/null 2>&1 || \
        echo "[env] WARN: unzip not found; the wheel ABI report will fail" >&2
    return $err
}

# --- ABI helpers ---------------------------------------------------------

# Local glibc version, e.g. "2.35".
tsm_glibc_version() {
    local v
    v="$(getconf GNU_LIBC_VERSION 2>/dev/null | awk '{print $2}')"
    if [ -z "$v" ]; then
        v="$(ldd --version 2>/dev/null | head -1 | grep -oE '[0-9]+\.[0-9]+$')"
    fi
    printf '%s\n' "$v"
}

# Highest GLIBC_x.y required by the given ELF files.
tsm_glibc_max() {
    local f
    for f in "$@"; do
        [ -f "$f" ] || continue
        readelf -V --wide "$f" 2>/dev/null \
            | sed -n 's/.*Name: GLIBC_\([0-9][0-9.]*\).*/\1/p'
    done | sort -Vu | tail -1
}

_tsm_abi_cache_dir() {
    local d="${XDG_CACHE_HOME:-$HOME/.cache}/tsm_abi"
    mkdir -p "$d" 2>/dev/null || return 1
    printf '%s\n' "$d"
}

# Cached glibc_max for a set of files; the key covers path, size and mtime.
tsm_cached_glibc_max() {
    local key dir sum
    [ $# -gt 0 ] || return 0
    dir="$(_tsm_abi_cache_dir)" || { tsm_glibc_max "$@"; return; }
    key="$(for f in "$@"; do [ -f "$f" ] && stat -c '%n:%s:%Y' "$f"; done \
        | sort | sha256sum | cut -d' ' -f1)"
    if [ -s "$dir/$key" ]; then
        cat "$dir/$key"
        return 0
    fi
    sum="$(tsm_glibc_max "$@")"
    if [ -n "$sum" ]; then
        printf '%s\n' "$sum" > "$dir/$key.$$" && mv -f "$dir/$key.$$" "$dir/$key"
    fi
    printf '%s\n' "$sum"
}

# Fail when an artifact needs a newer glibc than the local one.
_tsm_abi_require() {
    local need="$1" what="$2" local_v
    [ -n "$need" ] || return 0
    local_v="$(tsm_glibc_version)"
    [ -n "$local_v" ] || return 0
    if [ "$need" != "$local_v" ] && \
        [ "$(printf '%s\n%s\n' "$local_v" "$need" | sort -V | tail -1)" = "$need" ]; then
        echo "[abi] ERROR: $what requires glibc >= $need, local glibc is $local_v" >&2
        return 1
    fi
    return 0
}

# Mandatory runtime check for LLVM and the installed Triton extension.
tsm_check_runtime_abi() {
    local rc=0 need

    if [ -n "${LLVM_SYSPATH:-}" ] && [ -d "$LLVM_SYSPATH" ] \
        && [ -f "$LLVM_SYSPATH/.tsm_abi" ]; then
        need="$(sed -n 's/^glibc_max=//p' "$LLVM_SYSPATH/.tsm_abi" | head -1)"
        _tsm_abi_require "$need" "LLVM ($LLVM_SYSPATH)" || rc=1
    fi

    local triton_c="$TRITON/python/triton/_C"
    if [ ! -d "$triton_c" ]; then
        local site
        site="$("$PYTHON" -c 'import site; print(site.getsitepackages()[0])' 2>/dev/null)"
        [ -n "$site" ] && [ -d "$site/triton/_C" ] && triton_c="$site/triton/_C"
    fi
    if [ -d "$triton_c" ]; then
        local files=() f
        while IFS= read -r -d '' f; do files+=("$f"); done \
            < <(find "$triton_c" -maxdepth 2 -type f -name '*.so*' -print0)
        if [ ${#files[@]} -gt 0 ]; then
            need="$(tsm_cached_glibc_max "${files[@]}")"
            _tsm_abi_require "$need" "triton ($triton_c)" || rc=1
        fi
    fi
    return $rc
}

# glibc_max of a wheel's shared objects.
tsm_wheel_glibc_max() {
    local wheel="$1" tmp max
    [ -f "$wheel" ] || return 1
    command -v unzip >/dev/null 2>&1 || { echo "[abi] ERROR: unzip not found" >&2; return 1; }
    tmp="$(mktemp -d)" || return 1
    if ! unzip -qq -o "$wheel" '*.so*' -d "$tmp" 2>/dev/null; then
        rm -rf "$tmp"
        return 1
    fi
    local files=() f
    while IFS= read -r -d '' f; do files+=("$f"); done \
        < <(find "$tmp" -type f -name '*.so*' -print0)
    max="$(tsm_glibc_max "${files[@]}")"
    rm -rf "$tmp"
    printf '%s\n' "$max"
}

# Package an installed LLVM prefix into a portable tarball.
# Absolute symlinks are materialized and the result is ABI-tagged.
tsm_package_llvm() {
    local prefix="$1" out="$2" name="$3"
    local stage l target n=0 required max
    local files=() f

    if [ ! -d "$prefix" ]; then
        echo "[package] ERROR: prefix not found: $prefix" >&2
        return 1
    fi
    stage="$(mktemp -d)" || return 1
    cp -a "$prefix" "$stage/$name" || { rm -rf "$stage"; return 1; }

    while IFS= read -r -d '' l; do
        target="$(readlink "$l")"
        case "$target" in /*) ;; *) continue ;; esac
        if [ ! -e "$target" ]; then
            echo "[package] ERROR: dangling absolute symlink: $l -> $target" >&2
            rm -rf "$stage"
            return 1
        fi
        if [ -d "$target" ]; then
            cp -rL --remove-destination "$target" "$l"
        else
            cp -Lp --remove-destination "$target" "$l"
        fi
        n=$((n + 1))
    done < <(find "$stage/$name" -type l -print0)
    echo "[package] dereferenced $n absolute symlinks"

    for required in bin/clang bin/mlir-opt bin/mlir-translate bin/FileCheck \
        lib/cmake/lld/LLDConfig.cmake python_packages/mlir_core; do
        if [ ! -e "$stage/$name/$required" ]; then
            echo "[package] ERROR: missing required component: $required" >&2
            rm -rf "$stage"
            return 1
        fi
    done
    echo "[package] required components: ok"

    while IFS= read -r -d '' f; do files+=("$f"); done < <(
        find "$stage/$name" -type f \( -path '*/bin/*' -o -name '*.so*' \) -print0)
    max="$(tsm_glibc_max "${files[@]}")"
    printf 'glibc_max=%s\n' "$max" > "$stage/$name/.tsm_abi"
    echo "[package] glibc max ${max:-unknown}"

    tar -czf "$out" -C "$stage" "$name" || { rm -rf "$stage"; return 1; }
    if tar tzvf "$out" | grep -q ' -> /'; then
        echo "[package] ERROR: package contains absolute symlinks" >&2
        rm -rf "$stage"
        return 1
    fi
    rm -rf "$stage"
    echo "[package] wrote $out"
    return 0
}

# --- shared helpers used by the scripts -----------------------------------

# Activate a repository-local venv when one exists (3.6-era layout).
tsm_activate_venv() {
    if [ -f "$TRITON/.venv/bin/activate" ]; then
        # shellcheck disable=SC1091
        source "$TRITON/.venv/bin/activate"
    fi
}

# Print the named variables as export lines.
tsm_dump_env() {
    local v
    for v in "$@"; do
        printf 'export %s=%s\n' "$v" "${!v-}"
    done
}

# Export proxy settings for the pip-based scripts.
tsm_setup_proxy() {
    [ -n "${PROXY:-}" ] || return 0
    export https_proxy=$PROXY http_proxy=$PROXY all_proxy=$PROXY
    export HTTPS_PROXY=$PROXY HTTP_PROXY=$PROXY ALL_PROXY=$PROXY
}
