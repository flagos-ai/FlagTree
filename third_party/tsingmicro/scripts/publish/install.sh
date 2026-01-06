#!/bin/bash
set -e


#设置工作目录
script_path=$(realpath "$0")
script_dir=$(dirname "$script_path")
project_dir=$(realpath "$script_dir/../")
cd $project_dir

#是否使用虚拟python环境
use_venv=OFF
if [ $# -gt 0 ]; then
    if [[ "${1,,}" == "venv" ]]; then
        use_venv=ON
    fi
fi

#1.检测llvm
llvm_tar=$(find $project_dir -maxdepth 1 -name "llvm-*.tar.gz" -print -quit)
if [ -f "$llvm_tar" ]; then
    if find $project_dir -maxdepth 1 -type d -name "llvm-*" | grep -q .; then
        echo "find llvm"
    else
        tar -zxvf $llvm_tar
    fi
fi

if ! find $project_dir -maxdepth 1 -type d -name "llvm-*" | grep -q .; then
    echo "error: not find llvm dir!"
    exit -1
fi

#2.检测torch2.7等whl包
offline_tar=$(find $project_dir -maxdepth 1 -name "offline_pkgs*.tar.gz" -print -quit)
if [ -f "$offline_tar" ]; then
    if find $project_dir -maxdepth 1 -type d -name "offline_pkgs" | grep -q .; then
        echo "find offline_pkgs"
    else
        tar -zxvf $offline_tar
    fi
fi

if ! find $project_dir -maxdepth 1 -type d -name "offline_pkgs" | grep -q .; then
    echo "error: not find offline_pkgs!"
    exit -1
fi

#3.检测tx8_deps
tx8_deps_tar=$(find $project_dir -maxdepth 1 -name "tx8_depends_*.tar.gz" -print -quit)
if [ -f "$tx8_deps_tar" ]; then
    if find $project_dir -maxdepth 1 -type d -name "tx8_deps" | grep -q .; then
        rm -rf $project_dir/tx8_deps
    fi
    tar -zxvf $tx8_deps_tar
fi

if ! find $project_dir -maxdepth 1 -type d -name "tx8_deps" | grep -q .; then
    echo "error: not find tx8_deps dir!"
    exit -1
fi

#4.检测torch_txda
torch_txda_tar=$(find $project_dir -maxdepth 1 -name "torch_txda*.tar.gz" -print -quit)
if [ -f "$torch_txda_tar" ]; then
    if find $project_dir -maxdepth 1 -type d -name "pack" | grep -q .; then
        rm -rf $project_dir/pack
    fi
    tar -zxvf $torch_txda_tar
fi

if ! find $project_dir -maxdepth 1 -type d -name "pack" | grep -q .; then
    echo "error: not find torch_txda pack!"
    exit -1
fi

#5.激活python虚拟环境
if [ "x$use_venv" == "xON" ]; then
    if [ -d $project_dir/triton/.venv ]; then
        rm -rf $project_dir/triton/.venv
    fi
	if [ ! -d $project_dir/triton ]; then
        mkdir $project_dir/triton
    fi
    python3 -m venv $project_dir/triton/.venv --prompt triton
    source $project_dir/triton/.venv/bin/activate
fi
#check python version
python3 --version

#6.安装torch2.7等whl包
bash $script_dir/offline_python_deps.sh -i -r $script_dir/requirements_ts.txt -d $project_dir/offline_pkgs
if [ $? -eq 0 ]; then
    echo "Install torch package completed!"
else
    echo "Install torch package failed!"
    exit -1
fi
#check torch version
python3 -c "import torch; print(torch.__version__)"

PROXY=http://192.168.100.225:8889
export https_proxy=$PROXY http_proxy=$PROXY all_proxy=$PROXY
apt install ccache
pip install loguru
pip install scipy
unset https_proxy
unset http_proxy
unset all_proxy

#7.安装torch_txda
txops_wheel=$(find $project_dir/pack -maxdepth 1 -name "txops*.whl" -print -quit)
torch_txda_wheel=$(find $project_dir/pack -maxdepth 1 -name "torch_txda*.whl" -print -quit)
pip3 install $txops_wheel
pip3 install $torch_txda_wheel
#check torch_txda
python3 -c "import txops;print(txops.__dict__['__path__'])"
python3 -c "import torch_txda;print(torch_txda.__dict__['__path__'])"

#8.安装triton
triton_wheel=$(find $project_dir -maxdepth 1 -name "triton*.whl" -print -quit)
if [ -z "$triton_wheel" ]; then
    echo "error：not find triton*.whl!"
    exit -1
fi
pip3 uninstall triton -y
pip3 install $triton_wheel

#9.安装flaggems
flaggems_wheel=$(find $project_dir -maxdepth 1 -name "flag_gems*.whl" -print -quit)
if [ -z "$flaggems_wheel" ]; then
    echo "error：not find flag_gems*.whl!"
    exit -1
fi
pip3 install $flaggems_wheel
#check flaggems
#python3 -c "import flag_gems;print(flag_gems.__dict__['__path__'])"

#10.运行flaggems一个简单case
bash $script_dir/run_tsingmicro.sh pytest $project_dir/flaggems_tests/test_unary_pointwise_ops.py::test_accuracy_abs --ref cpu --mode quick
