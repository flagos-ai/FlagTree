#!/bin/bash
set -e

##########################################################################################################################
##                                                                                                                      ##
##  Triton版本包制作脚本                                                                                                ##
##      在docker容器外triton代码目录上一级目录下执行, 制作triton完整版本包.                                             ##
##      执行该脚本前先执行build_wheel.sh生成triton、flaggems wheel包.                                                   ##
##                                                                                                                      ##
##########################################################################################################################

script_path=$(realpath "$0")
script_dir=$(dirname "$script_path")
project_dir=$(realpath "$script_dir/../../../..")

if [ -z "${WORKSPACE+x}" ]; then
    WORKSPACE=$(realpath "$project_dir/..")
fi

TX8_SDK_Triton_version=1.9.0
sdk_name=TX8_SDK_${TX8_SDK_Triton_version}_Triton
# TX8_DEPS_ROOT=$WORKSPACE/tx8_deps
# LLVM=$WORKSPACE/llvm-a66376b0-ubuntu-x64
triton_dir=$project_dir
flaggems_dir=$WORKSPACE/flaggems
publish_dir=$WORKSPACE/$sdk_name
version_file=$publish_dir/code_version.txt

if [ ! -d $publish_dir ]; then
    mkdir $publish_dir
fi

if [ ! -d $publish_dir/scripts ]; then
    mkdir $publish_dir/scripts
fi

if [ ! -d $triton_dir ]; then
    echo "未找到triton项目目录"
    exit 1
else
    pushd $triton_dir
        triton_wheel=$(find python -maxdepth 1 -name "triton*.whl" -print -quit)
        if [ -z "$triton_wheel" ]; then
            echo "错误：未找到 triton*.whl 文件, build cmd: python3 -m pip wheel $triton_dir"
            exit 1
        fi

        commit_id=$(git rev-parse HEAD 2>/dev/null)
        if [ $? -eq 0 ]; then
            # 写入文件：目录名和commit id
            echo "triton: $commit_id" > "$version_file"
        else
            echo "Warning: Failed to get commit id in $dir"
        fi
        cp $triton_wheel $publish_dir
        cp -r third_party/tsingmicro/examples/ $publish_dir/triton_examples
    popd
fi

# pip install build
# python -m build --wheel
if [ ! -d $flaggems_dir ]; then
    echo "未找到flaggem项目目录"
    exit 1
else
    pushd $flaggems_dir
        flaggems_wheel=$(find . -maxdepth 2 -name "flag_gems*.whl" -print -quit)
        if [ -z "$flaggems_wheel" ]; then
            echo "错误：未找到 flag_gems*.whl 文件"
            exit 1
        fi

        commit_id=$(git rev-parse HEAD 2>/dev/null)
        if [ $? -eq 0 ]; then
            # 写入文件：目录名和commit id
            echo "flaggems: $commit_id" >> "$version_file"
        else
            echo "Warning: Failed to get commit id in $dir"
        fi
        cp $flaggems_wheel $publish_dir
        cp -r tests $publish_dir/flaggems_tests
    popd
fi

# cp $WORKSPACE/download/tx8_depends*.tar.gz $publish_dir
# cp $WORKSPACE/download/LLVM

cp $script_dir/README.md $publish_dir
cp $script_dir/install.sh $publish_dir/scripts
cp $script_dir/run_tsingmicro.sh $publish_dir/scripts
cp $script_dir/../base/base_run.sh $publish_dir/scripts
cp $script_dir/../requirements_ts.txt $publish_dir/scripts
cp $script_dir/../tools/offline_python_deps.sh $publish_dir/scripts
# cp $project_dir/python/requirements.txt $publish_dir

pushd $WORKSPACE
    current_time=$(date +%Y%m%d_%H%M%S)
    pkg_file=${sdk_name}_${current_time}.tar.gz
    if [ -f $pkg_file ]; then
        rm -f $pkg_file
    fi
    tar -zcf $pkg_file $sdk_name
popd
