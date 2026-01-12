#!/bin/bash

declare -A PROJECT_INFO

set -e
# set -x

# # 定义一个函数，用于克隆 Git 仓库并切换到指定分支或指定 commit
clone_and_checkout() {
    local git_url="$1"
    local base_dir="$2"
    local ref_type="$3"  # "branch" 或 "commit"
    local ref_value="$4" # 分支名称或 commit ID
    local tag_name="$5" # 标签

    target_dir=$base_dir/$tag_name

    PROJECT_INFO+="模块:$tag_name \n分支信息:$ref_value \nurl:$git_url\n==============================\n"

    # 检查目标目录是否存在，如果不存在则创建
    if [ ! -d "$target_dir" ]; then
        mkdir -p "$target_dir"
        if [ $? -ne 0 ]; then
            echo "Error: Failed to create target directory: $target_dir"
            return 1
        fi
    fi

    if [ "$(ls -A $target_dir)" ]; then
        echo "jump clone $target_dir"
        return 0
    fi

    # 使用 pushd 进入目标目录
    pushd "$target_dir" >/dev/null || return 1

    # 克隆仓库
    git clone "$git_url" .
    if [ $? -ne 0 ]; then
        echo "Error: Failed to clone repository: $git_url"
        popd >/dev/null
        return 1
    fi

    # 根据 ref_type 切换到分支或 commit
    if [ "$ref_type" == "branch" ]; then
        # 检查分支是否存在
        if ! git branch -r | grep -q "origin/$ref_value"; then
            echo "Error: Branch '$ref_value' does not exist in the repository."
            popd >/dev/null
            return 1
        fi
        # 切换到指定分支
        git checkout "$ref_value"
        if [ $? -ne 0 ]; then
            echo "Error: Failed to switch to branch: $ref_value"
            popd >/dev/null
            return 1
        fi
        echo "Successfully cloned and switched to branch '$ref_value' for repository: $git_url"
    elif [ "$ref_type" == "commit" ]; then
        # 检查 commit 是否存在
        if ! git rev-parse "$ref_value" >/dev/null 2>&1; then
            echo "Error: Commit '$ref_value' does not exist in the repository."
            popd >/dev/null
            return 1
        fi
        # 切换到指定 commit
        git checkout "$ref_value"
        if [ $? -ne 0 ]; then
            echo "Error: Failed to switch to commit: $ref_value"
            popd >/dev/null
            return 1
        fi
        echo "Successfully cloned and switched to commit '$ref_value' for repository: $git_url"
    else
        echo "Error: Invalid ref_type. Use 'branch' or 'commit'."
        popd >/dev/null
        return 1
    fi

    # 使用 popd 退出目录
    popd >/dev/null
    return 0
}

download_and_extract() {
    local url="$1"       # 下载链接
    local target_dir="$2" # 目标目录
    local temp_dir="$3"   # 临时目录
    local tag_name="$4" # 标签

    temp_dir=$temp_dir/tar_path

    # 确保目标目录和临时目录存在
    mkdir -p "$target_dir"
    mkdir -p "$temp_dir"

    # 下载文件到临时目录
    local file_name="$(basename "$url")"
    local temp_file="$temp_dir/$file_name"
    echo "temp_file:$temp_file"
    is_download=0

    if [[ ! -f $temp_file ]]; then
        wget -O "$temp_file" "$url"
        is_download=1
    else
        echo "file $temp_file exist."
    fi

    local md5_value
    md5_value=$(md5sum "$temp_file" | awk '{print $1}')
    PROJECT_INFO+="mode:$tag_name \nfile:$file_name \nmd5:$md5_value \nurl:$url\n==============================\n"

    if [ $is_download -eq 1 ]; then
        echo "start unzip..."

        # 检查下载是否成功
        if [ $? -eq 0 ]; then
            # 解压到临时目录
            unzip_dir=$temp_dir/${tag_name}_$(date +"%y%m%d_%H%M%S")
            mkdir -p $unzip_dir
            tar -xz -C "$unzip_dir" -f "$temp_file"
            echo "unzip to:$unzip_dir"

            cp -r $unzip_dir/* $target_dir
        else
            echo "download error, exit."
            exit 1
        fi
    else
        echo "dir $target_dir not empty."
    fi
}

script_path=$(realpath "$0")
script_dir=$(dirname "$script_path")
project_dir=$(realpath "$script_dir/../../..")

if [ -z "${WORKSPACE+x}" ]; then
    WORKSPACE=$(realpath "$project_dir/..")
fi

build_tx8_deps=OFF
build_dev=OFF
build_flagtree_tx8_deps=OFF

# 配置文件路径
CONFIG_FILE="$script_dir/copy_config.conf"

# 显示帮助信息
show_help() {
    echo "usage: $0 mode [tx81fw url]"
    echo "support mode:"
    echo "  build_flagtree_tx8_deps"
    echo "  build_tx8_deps"
    echo "  build_dev"
    echo ""
    echo "tx81fw url:"
    echo "  eg: http://172.50.1.66:8082/artifactory/tx8-generic-dev/tx81fw/tx81fw_202512041135_b731cf.tar.gz"
    echo "  default: http://172.50.1.66:8082/artifactory/tx8-generic-dev/tx81fw/tx81fw_202512041135_b731cf.tar.gz"
    echo "  ..."
    echo ""
    echo "example: $0 build_tx8_deps"
}

# 检查参数数量
if [ $# -le 1 ]; then
    show_help
    exit 1
fi

# 收集所有适用于当前模式的路径
declare -A CONFIG_BLOCKS
declare -A CONFIG_ITEMS
declare -A EXECUTED_BLOCKS

load_copy_cfg() {
    local current_block=""

    while IFS= read -r line; do
        # 跳过注释和空行
        [[ "$line" =~ ^[[:space:]]*# ]] && continue

        [[ -z "$line" ]] && continue

        # 解析块定义
        if [[ "$line" =~ ^\[([^:]+):([^]]+)\] ]]; then
            current_block="${BASH_REMATCH[1]}"
            CONFIG_BLOCKS["$current_block"]="${BASH_REMATCH[2]}"
            continue
        fi

        # 存储配置项
        if [[ -n "$current_block" ]]; then
            CONFIG_ITEMS["$current_block"]+="$line"$'\n'
        fi
    done < "$CONFIG_FILE"
}

copy_files() {
    local mode="$1"
    local destination="$2"

    # 检查目标路径
    if [ ! -d "$destination" ]; then
        echo "create dir: $destination"
        mkdir -p "$destination" || { echo "error, can't mkdir"; return 1; }
    fi

    echo "start '$mode' copy to '$destination'..."

    # 遍历所有配置块
    for block in "${!CONFIG_BLOCKS[@]}"; do
        # 检查模式是否匹配
        if [[ ",${CONFIG_BLOCKS[$block]}," =~ ",$mode," ]]; then
            [[ -n "${EXECUTED_BLOCKS[$block]}" ]] && continue
            EXECUTED_BLOCKS["$block"]=1

            # 处理块内的每个配置项
            while IFS= read -r line; do
                [[ -z "$line" ]] && continue

                # 解析配置行
                if [[ "$line" =~ ^([^:]+):([^,]+),?(.*)$ ]]; then
                    local copy_type="${BASH_REMATCH[1]}"
                    local source_pattern="$WORKSPACE/${BASH_REMATCH[2]}"
                    local target_subdir="${BASH_REMATCH[3]:-.}"

                    local full_target="$destination/${target_subdir}"
                    [[ "$target_subdir" == "." ]] && full_target="$destination"

                    mkdir -p "$full_target"
                    echo "copy [$copy_type] '$source_pattern' to '$full_target'"
                    if [[ "$copy_type" == "dir" ]]; then
                        cp -r $source_pattern $full_target
                    else
                        cp $source_pattern $full_target
                    fi
                fi
            done <<< "${CONFIG_ITEMS[$block]}"
        fi
    done

    echo "copy finish"
    return 0
}

MODE=$1
tx81fw=http://172.50.1.66:8082/artifactory/tx8-generic-dev/tx81fw/tx81fw_202512041135_b731cf.tar.gz
if [ $# -ge 2 ]; then
	tx81fw=$2
fi
echo "tx81fw: "$tx81fw
if [ "x$MODE" == "xbuild_flagtree_tx8_deps" ] || [ "x$MODE" == "xbuild_tx8_deps" ] || [ "x$MODE" == "xbuild_dev" ]; then
    download_dir=$WORKSPACE/download
    ########################################################################################
    tx8fw_dir=$download_dir/triton-tx8fw
    download_and_extract $tx81fw \
        "$tx8fw_dir" "$download_dir" "tx8fw"

    ########################################################################################
    xuantie_sdk_dir=$download_dir/tx8fw-xuantie-sdk
    clone_and_checkout "git@gitlab.tsingmicro.com:tx8_developers/tx8fw-xuantie-sdk.git" \
        "$download_dir" "commit" "9f96c2b235ca15e90907205b6738065f8a8edd65" "tx8fw-xuantie-sdk"

    # instr_tx81_lib=$WORKSPACE/download/libinstr_tx81.a
    xuantie_dir=$xuantie_sdk_dir/Xuantie-900-gcc-elf-newlib-x86_64-V2.10.2
    if [ ! -d $xuantie_dir ]; then
        echo "error can't find:$xuantie_dir"
        exit
    fi

    ########################################################################################
    # doc
    file1=$download_dir/"最终用户许可协议.pdf"
    # if [ ! -f $file1 ]; then
    if ls $file1 >/dev/null 2>&1; then
        echo "$file1 exist."
    else
        wget -P $download_dir http://172.50.1.66:8082/artifactory/tx8-generic-dev/triton/tx8_depends/%E6%9C%80%E7%BB%88%E7%94%A8%E6%88%B7%E8%AE%B8%E5%8F%AF%E5%8D%8F%E8%AE%AE.pdf
    fi

    ########################################################################################
    triton_profiler_dir=$WORKSPACE/triton_profiler
    # 因为要编译，不能放到download 里面，否则找到一些一些tx8_deps的路径
    clone_and_checkout "git@gitlab.tsingmicro.com:triton-based-projects/triton_profiler.git" \
            "$WORKSPACE" "branch" "master" "triton_profiler"

    profiler_head=$triton_profiler_dir/profiler/include/profiler.h
    if [ ! -f $profiler_head ]; then
        echo "error can't find:$profiler_head"
        exit
    fi

    ########################################################################################
    tx8_depends_dir=$WORKSPACE/tx8_deps
    if [ -d $tx8_depends_dir ]; then
        rm -rf $tx8_depends_dir
    fi
    mkdir $tx8_depends_dir

    load_copy_cfg
    echo -e $PROJECT_INFO > $tx8_depends_dir/version.txt

    if [ "x$MODE" == "xbuild_dev" ]; then
        copy_files profile_need $tx8_depends_dir
        triton_profiler_pkg=$triton_profiler_dir/profile
        if [ ! -f $triton_profiler_dir/profile/profiler_commit_id.txt ]; then
            bash $triton_profiler_dir/build.sh rebuild
        fi
    fi
    copy_files $MODE $tx8_depends_dir

    pushd $WORKSPACE
        current_time=$(date +%Y%m%d_%H%M%S)
        tar_flag="release"
        if [ "x$MODE" == "xbuild_tx8_deps" ] || [ "x$MODE" == "xbuild_dev" ]; then
            tar_flag="dev"
        fi

        pkg_file=download/tx8_depends_${tar_flag}_${current_time}.tar.gz
        if [ ! -d download ]; then
            mkdir download
        fi
        if [ -f $pkg_file ]; then
            rm -f $pkg_file
        fi
        tar -zcf $pkg_file tx8_deps
    popd
else
    echo abc
    # tx8_deps_base=$WORKSPACE/tx8_deps
    # # clone_and_checkout "git@gitlab.tsingmicro.com:triton-based-projects/llvm-project.git" "$WORKSPACE/llvm-project-for-ztc" "branch" "ztc"
    # clone_and_checkout "git@gitlab.tsingmicro.com:triton-based-projects/llvm-project.git" "$WORKSPACE/llvm-project" "commit" "a66376b0dc3b2ea8a84fda26faca287980986f78"

    # download_and_extract "http://172.50.1.66:8082/artifactory/tx8-generic-dev/tx8/triton/tx8_depends_20250512_145415.tar.gz" \
    #         "$tx8_deps_base" "$WORKSPACE/download"
    # clone_and_checkout "ssh://192.168.100.107:29418/tx8_toolchain/tx8be-oplib" "third_party/tx8be-oplib" "commit" "b5651a734f1a6a8943765c83bee1e80d6a2c6a37"
fi
