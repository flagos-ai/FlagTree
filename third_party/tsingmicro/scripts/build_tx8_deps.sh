#!/bin/bash
set -e
# set -x

# 显示帮助信息
show_help() {
    echo "usage: $0 mode [tx81fw url] [tx_profiler url] [tx8-yoc-rt-thread-smp url]"
    echo "support mode:"
    echo "  build_flagtree_tx8_deps"
    echo "  build_tx8_deps"
    echo "  build_dev"
    echo "  now build_tx8_deps=build_flagtree_tx8_deps, both not include profile module, build_dev include profile module."
    echo ""
    echo "tx81fw url:"
    echo "  eg: http://172.50.1.66:8082/artifactory/tx8-generic-dev/tx81fw/tx81fw_202512041135_b731cf.tar.gz"
    echo "  default: http://172.50.1.66:8082/artifactory/tx8-generic-dev/tx81fw/tx81fw_202512041135_b731cf.tar.gz"
    echo "  ..."
    echo ""
    echo "tx_profiler url:"
    echo "  eg: http://172.50.1.66:8082/artifactory/tx8-generic-dev/tx81-profiling/master/profiling_tool_v5.5.0_release_2025-1124_.tar.gz"
    echo "  default: http://172.50.1.66:8082/artifactory/tx8-generic-dev/tx81-profiling/master/profiling_tool_v5.5.0_release_2025-1124_.tar.gz"
    echo "  ..."
    echo ""
    echo "tx8-yoc-rt-thread-smp url:"
    echo "  eg: http://172.50.1.66:8082/artifactory/tx8-generic-dev/tx81fw/tx8-yoc-rt-thread-smp-202603031631-88bfb9.tar.gz"
    echo "  default: http://172.50.1.66:8082/artifactory/tx8-generic-dev/tx81fw/tx8-yoc-rt-thread-smp-202603031631-88bfb9.tar.gz"
    echo "  ..."
    echo ""
    echo "example: bash triton/third_party/tsingmicro/scripts/build_tx8_deps.sh build_dev http://172.50.1.66:8082/artifactory/tx8-generic-dev/tx81fw/tx81fw_202602261758_b72af3.tar.gz http://172.50.1.66:8082/artifactory/tx8-generic-dev/tx81-profiling/master/profiling_tool_v5.6.0_release_2026-0228_.tar.gz http://172.50.1.66:8082/artifactory/tx8-generic-dev/tx81fw/tx8-yoc-rt-thread-smp-202603031631-88bfb9.tar.gz"
}

# 检查参数数量
if [ $# -lt 1 ]; then
    show_help
    exit 1
fi

MODE=$1
tx81fw=http://172.50.1.66:8082/artifactory/tx8-generic-dev/tx81fw/tx81fw_202512041135_b731cf.tar.gz
if [ $# -ge 2 ]; then
	tx81fw=$2
fi
echo "tx81fw: "$tx81fw

profile_url=http://172.50.1.66:8082/artifactory/tx8-generic-dev/tx81-profiling/master/profiling_tool_v5.5.0_release_2025-1124_.tar.gz
if [ $# -ge 3 ]; then
	profile_url=$3
fi
echo "profile_url: "$profile_url

tx8_yoc_rt_thread_smp=http://172.50.1.66:8082/artifactory/tx8-generic-dev/tx81fw/tx8-yoc-rt-thread-smp-202603031631-88bfb9.tar.gz
if [ $# -ge 4 ]; then
	tx8_yoc_rt_thread_smp=$4
fi
echo "tx8_yoc_rt_thread_smp: "$tx8_yoc_rt_thread_smp

script_path=$(realpath "$0")
script_dir=$(dirname "$script_path")
project_dir=$(realpath "$script_dir/../../..")

WORKSPACE="${WORKSPACE:-$(realpath "$project_dir/..")}"
case "$WORKSPACE" in
    ""|"/"|"$HOME")
        echo "error: refusing to use WORKSPACE='$WORKSPACE'" >&2
        exit 1
        ;;
esac
[ -d "$WORKSPACE" ] || { echo "error: WORKSPACE not found: $WORKSPACE" >&2; exit 1; }

declare -A PROJECT_INFO

# 定义一个函数，用于克隆 Git 仓库并切换到指定分支或指定 commit

download_and_extract() {
    local url=$1 target=$2 temp_root=$3 tag=$4
    local temp=$temp_root/tar_path file
    mkdir -p "$target" "$temp"
    file=$temp/$(basename "$url")

    if [ ! -f "$file" ]; then
        echo "downloading $url"
        wget -O "$file.part" "$url" || { echo "error: download failed: $url" >&2; return 1; }
        tar -tzf "$file.part" >/dev/null 2>&1 || {
            echo "error: corrupt archive: $url" >&2
            rm -f "$file.part"
            return 1
        }
        mv "$file.part" "$file"
    else
        echo "using cached $file"
    fi

    PROJECT_INFO+="mode:$tag \nfile:$(basename "$url") \nmd5:$(md5sum "$file" | awk '{print $1}') \nurl:$url\n==============================\n"

    if [ -f "$target/.extracted-$tag" ]; then
        echo "already extracted: $tag"
        return 0
    fi
    echo "extracting $file"
    tar -xzf "$file" -C "$target"
    touch "$target/.extracted-$tag"
}

if [ "x$MODE" == "xbuild_flagtree_tx8_deps" ] || [ "x$MODE" == "xbuild_tx8_deps" ] || [ "x$MODE" == "xbuild_dev" ]; then
    download_dir=$WORKSPACE/download
    ########################################################################################
    tx8fw_dir=$download_dir/triton-tx8fw
    download_and_extract $tx81fw \
        "$tx8fw_dir" "$download_dir" "tx8fw"

    ########################################################################################
    xuantie_dir=$tx8fw_dir/rcs1fw-rtt/tool/rcsfw-xuantie-sdk
    if [ ! -d "$xuantie_dir" ]; then
        echo "error: xuantie sdk not found: $xuantie_dir" >&2
        exit 1
    fi

    pushd "$xuantie_dir"
        bash setup.sh -t all
    popd

    ########################################################################################
    # doc
    file1=$download_dir/"最终用户许可协议.pdf"
    # if [ ! -f $file1 ]; then
    if [ -f "$file1" ]; then
        echo "$file1 exist."
    else
        wget -P "$download_dir" http://172.50.1.66:8082/artifactory/tx8-generic-dev/triton/tx8_depends/%E6%9C%80%E7%BB%88%E7%94%A8%E6%88%B7%E8%AE%B8%E5%8F%AF%E5%8D%8F%E8%AE%AE.pdf
    fi

    ########################################################################################

    tx_profiler_dir=$download_dir/tx_profiler
    download_and_extract $profile_url \
        "$tx_profiler_dir" "$download_dir" "tx_profiler"

    ########################################################################################

    tx8_yoc_rt_thread_smp_dir=$download_dir/tx8-yoc-rt-thread-smp
    download_and_extract $tx8_yoc_rt_thread_smp \
        "$tx8_yoc_rt_thread_smp_dir" "$download_dir" "tx8-yoc-rt-thread-smp"

    ########################################################################################
    tx8_depends_dir=$WORKSPACE/tx8_deps
    rm -rf "$tx8_depends_dir"
    mkdir -p "$tx8_depends_dir"

    printf '%b\n' "$PROJECT_INFO" > "$tx8_depends_dir/version.txt"

    mkdir -p "$tx8_depends_dir/tx8-yoc-rt-thread-smp"
    cp -r "$download_dir/triton-tx8fw/rcs1fw-rtt" "$tx8_depends_dir/"
    cp "$download_dir"/*.pdf "$tx8_depends_dir/" 2>/dev/null || true
    cp -r "$download_dir"/tx8-yoc-rt-thread-smp/tx8-yoc-rt-thread-smp*/* \
          "$tx8_depends_dir/tx8-yoc-rt-thread-smp/"
    if [ "$MODE" == "build_dev" ]; then
        mkdir -p "$tx8_depends_dir/profiling_tool"
        cp -r "$download_dir"/tx_profiler/profiling_tool*/* "$tx8_depends_dir/profiling_tool/"
    fi

    current_time=$(date +%Y%m%d_%H%M%S)
    case "$MODE" in
        build_tx8_deps|build_dev) tar_flag=dev ;;
        *) tar_flag=release ;;
    esac

    pkg_file=$WORKSPACE/tx8_depends_${tar_flag}_${current_time}.tar.gz
    rm -f "$pkg_file"
    tar -zcf "$pkg_file" -C "$WORKSPACE" tx8_deps
    echo "========================================================================================================================="
    echo "tx8_deps generated at: $pkg_file"
    echo "tx8_deps tree kept at: $tx8_depends_dir"
    echo "========================================================================================================================="
else
    echo "error: unknown mode '$MODE'" >&2
    show_help >&2
    exit 1
fi
