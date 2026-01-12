#!/bin/bash
set -e

##########################################################################################################################
##                                                                                                                      ##
##  Triton wheel包编译脚本                                                                                              ##
##      在docker容器内triton代码目录上一级目录下执行, 将triton、flaggems编译出wheel包进行转测、发布.                    ##
##                                                                                                                      ##
##########################################################################################################################

script_path=$(realpath "$0")
script_dir=$(dirname "$script_path")
project_dir=$(realpath "$script_dir/../../../..")
triton_dir=$project_dir
flaggems_dir=$project_dir/../flaggems

if [ -z "${WORKSPACE+x}" ]; then
    WORKSPACE=$(realpath "$project_dir/..")
fi

TX8_DEPS_ROOT=$WORKSPACE/tx8_deps
LLVM=$WORKSPACE/llvm-a66376b0-ubuntu-x64
BUILD_TYPE="release"

#1.检测llvm
llvm_tar=$(find $WORKSPACE -maxdepth 1 -name "llvm-*.tar.gz" -print -quit)
if [ -f "$llvm_tar" ]; then
    if find $WORKSPACE -maxdepth 1 -type d -name "llvm-*" | grep -q .; then
        echo "find llvm"
    else
        tar -zxvf $llvm_tar
    fi
fi

if ! find $WORKSPACE -maxdepth 1 -type d -name "llvm-a66376b0-ubuntu-x64" | grep -q .; then
    echo "error: not find llvm dir!"
    exit -1
fi

#2.检测torch2.7等whl包
offline_tar=$(find $WORKSPACE -maxdepth 1 -name "offline_pkgs*.tar.gz" -print -quit)
if [ -f "$offline_tar" ]; then
    if find $WORKSPACE -maxdepth 1 -type d -name "offline_pkgs" | grep -q .; then
        echo "find offline_pkgs"
    else
        tar -zxvf $offline_tar
    fi
fi

if ! find $WORKSPACE -maxdepth 1 -type d -name "offline_pkgs" | grep -q .; then
    echo "error: not find offline_pkgs!"
    exit -1
fi

#3.检测tx8_deps
tx8_deps_tar=$(find $WORKSPACE -maxdepth 1 -name "tx8_depends_*.tar.gz" -print -quit)
if [ -f "$tx8_deps_tar" ]; then
    if find $WORKSPACE -maxdepth 1 -type d -name "tx8_deps" | grep -q .; then
       echo "find tx8_deps"
    else
        tar -zxvf $tx8_deps_tar
    fi
fi

if ! find $WORKSPACE -maxdepth 1 -type d -name "tx8_deps" | grep -q .; then
    echo "error: not find tx8_deps dir!"
    exit -1
fi


#4.创建虚拟python环境
cd $triton_dir
if [ -d "./.venv" ]; then
    rm -rf .venv
fi
python3 -m venv .venv --prompt triton
source .venv/bin/activate
#check python version
python3 --version

#5.安装编译依赖的python环境
pip3 uninstall setuptools -y
bash third_party/tsingmicro/scripts/tools/offline_python_deps.sh -i -r python/requirements.txt -d ../offline_pkgs
bash third_party/tsingmicro/scripts/tools/offline_python_deps.sh -i -r third_party/tsingmicro/scripts/requirements_ts.txt -d ../offline_pkgs
if [ $? -eq 0 ]; then
    echo "Install compile tool package completed!"
else
    echo "Install compile tool package failed!!!"
    exit -1
fi
PROXY=http://192.168.100.225:8889
export https_proxy=$PROXY http_proxy=$PROXY all_proxy=$PROXY
apt install -y lld
apt install ccache
pip3 install scikit_build_core #flaggems需要,也需要torch
#unset https_proxy
unset http_proxy
unset all_proxy

#6.编译tritoon wheel包
triton_wheel=$(find python -maxdepth 1 -name "triton*.whl" -print -quit)
if [ -f "$triton_wheel" ]; then
    rm -rf  $triton_wheel
fi

if [ "$BUILD_TYPE" == "debug" ]; then
    export DEBUG=ON
else
    export REL_WITH_DBG_INFO=ON
fi

export TRITON_BUILD_WITH_CLANG_LLD=true
export TRITON_BUILD_WITH_CCACHE=true
export TRITON_OFFLINE_BUILD=ON
export TRITON_BUILD_PROTON=OFF
export LLVM_SYSPATH=$LLVM
export TX8_DEPS_ROOT=$TX8_DEPS_ROOT

cd python
python3 -m pip wheel . --no-build-isolation -v --verbos

#7.编译flaggems wheel包
cd $flaggems_dir
flaggems_wheel=$(find . -maxdepth 2 -name "flag_gems*.whl" -print -quit)
if [ -f "$flaggems_wheel" ]; then
    rm -rf  $flaggems_wheel
fi
python3 -m pip wheel . --no-deps -v --verbos
