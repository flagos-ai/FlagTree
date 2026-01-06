#!/bin/bash
set -e
##1.下载triton、flaggems代码到/login_home/jenkins_tc/triton目录(CI负责,必须是这个目录,在容器外以root账号)
#sudo -s
#mkdir triton
#cd triton
#git clone "http://192.168.100.107/triton-based-projects/triton" && (cd "triton" && mkdir -p .git/hooks && curl -Lo `git rev-parse --git-dir`/hooks/commit-msg http://192.168.100.107/tools/hooks/commit-msg; chmod +x `git rev-parse --git-dir`/hooks/commit-msg)
#git clone "http://gitlab.tsingmicro.com/triton-based-projects/flaggems.git" -b board-test-base

##2.创建容器(CI负责)
#docker run -d --name jenkins_tc_triton_ci --network=host --ipc=host --privileged -v /dev:/dev -v /tmp:/tmp -v /lib/modules:/lib/modules -v /sys:/sys -v /login_home/:/login_home/ -w /login_home/jenkins_tc/ hub.tsingmicro.com/tx8/ubuntu/v5.5.0.1030:tsingmicro_release

##3.进入容器(CI负责)
#docker exec -it jenkins_tc_triton_ci /bin/bash

##4.在/login_home/jenkins_tc/triton执行业务的ci运行脚本
#cd triton
#bash triton/third_party/tsingmicro/scripts/ci/run_triton_flaggems_ci_test.sh 0 0 0 ci_ops 1


##########################################################################################################################
##                                                                                                                      ##
##  业务的ci运行脚本                                                                                                    ##
##     param1: skip_install, set 1-skip depends install,set 0-install depends, default 0.                               ##
##     param2: skip_build,   set 1-skip triton build,   set 0-build triton,    default 0.                               ##
##     param3: skip_run,     set 1-skip run ci test,    set 0-run ci test,     default 0.                               ##
##     param4: test_set,     set test set name, default 'ci_ops'.                                                       ##
##     param5: device_count, set device count number,   default 1.                                                      ##
##     ##param: precision_priority, set 1-triton compiler use high precision mode for special ops, default 1.           ##
##     param6: quick_mode,   set 1-quick mode to run flaggems, set 0-normal mode, default 0.                            ##
##     param7: skip_device,  set devices that need to be skipped, when they are unavailable, default [].                ##
##                                                                                                                      ##
##########################################################################################################################

script_path=$(realpath "$0")
echo $script_path
script_dir=$(dirname "$script_path")
echo $script_dir
project_dir=$(realpath "$script_dir/../../../../../")
echo $project_dir
export TRITON_WORKSPACE=$project_dir
skip_install=0
skip_build=0
skip_run=0
test_set=ci_ops
device_count=1
quick_mode=0
skip_device=

precision_priority=1
tx8_depends_name=tx8_depends_dev_20251205_164530
torch_txda_name=torch_txda+txops-20251212-eb33daf9+8d7d6e7
txda_skip_ops="repeat_interleave.self_int,pad,to.dtype,uniform_,sort.values_stable,contiguous,resolve_conj"
txda_fallback_cpu_ops="random_,quantile,_local_scalar_dense,arange,unfold,index,le,all,ge,pad,to,gather_backward,zero_,view_as_real,resolve_neg,embedding_backward,sort,repeat_interleave,rsub,hstack,vstack,min,uniform_,abs,ne,eq,mul,bitwise_and,masked_select,max,ceil,div,gt,lt,sum,scatter,where,resolve_conj,isclose,isfinite,tile,equal,gather,contiguous"

if [ $# -ge 1 ]; then
	skip_install=$1
fi
if [ $# -ge 2 ]; then
	skip_build=$2
fi
if [ $# -ge 3 ]; then
	skip_run=$3
fi
if [ $# -ge 4 ]; then
	test_set=$4
fi
if [ $# -ge 5 ]; then
	device_count=$5
fi
if [ $# -ge 6 ]; then
	quick_mode=$6
fi
if [ $# -ge 7 ]; then
	skip_device=$(echo $7 | tr ',' ' ')
fi
echo "param count:"$#
echo "skip_install:"$skip_install
echo "skip_build:"$skip_build
echo "skip_run:"$skip_run
echo "test_set:"$test_set
echo "device_count:"$device_count
echo "quick_mode:"$quick_mode
echo "skip_device:"$skip_device
echo "precision_priority:"$precision_priority
echo "tx8_depends_name:"$tx8_depends_name
echo "torch_txda_name:"$torch_txda_name
echo "txda_skip_ops:"$txda_skip_ops
echo "txda_fallback_cpu_ops:"$txda_fallback_cpu_ops
# ##1.下载依赖(triton业务负责)
# cd $project_dir
# #为了加快ci速度,从提前下载好的位置cp. src位置变后此处要更新
# TRITON_DEPENDS_SRC=/login_home/jenkins_tc/triton
# ###download llvm(很少变化)
# if [ ! -d "./llvm-a66376b0-ubuntu-x64" ]; then
# 	if [ -d $TRITON_DEPENDS_SRC/llvm-a66376b0-ubuntu-x64 ]; then
# 		cp -r $TRITON_DEPENDS_SRC/llvm-a66376b0-ubuntu-x64/ ./
# 		echo "cp $TRITON_DEPENDS_SRC/llvm-a66376b0-ubuntu-x64 complete!"
# 	else
# 		echo "warning：$TRITON_DEPENDS_SRC/llvm-a66376b0-ubuntu-x64 not exist， use wget to download, maybe very slowly!"
# 	fi
# fi
# 
# if [ ! -d "./llvm-a66376b0-ubuntu-x64" ]; then
# 	wget https://toolchain-jfrog.tsingmicro.xyz:443/artifactory/tx8-generic-dev/triton/tools/llvm-a66376b0-ubuntu-x64.tar.gz
# 	if [ $? -eq 0 ]; then
# 		echo "Download llvm complete!"
# 	else
# 		echo "Download llvm fail!!!"
# 		exit -1
# 	fi
# 	tar -xzvf llvm-a66376b0-ubuntu-x64.tar.gz
# 	rm llvm-a66376b0-ubuntu-x64.tar.gz
# fi
# 
# if [ ! -d "./llvm-a66376b0-ubuntu-x64" ]; then
# 	echo "fail: not find llvm!!!"
# 	exit -1
# fi
# 
# ###download torch2.7 wheels for offline install(很少变化)
# if [ ! -d "./offline_pkgs" ]; then
# 	if [ -d $TRITON_DEPENDS_SRC/offline_pkgs ]; then
# 		cp -r $TRITON_DEPENDS_SRC/offline_pkgs/ ./
# 		echo "cp $TRITON_DEPENDS_SRC/offline_pkgs complete!"
# 	else
# 		echo "warning：$TRITON_DEPENDS_SRC/offline_pkgs not exist， use wget to download, maybe very slowly!"
# 	fi
# fi
# 
# if [ ! -d "./offline_pkgs" ]; then
# 	wget https://toolchain-jfrog.tsingmicro.xyz:443/artifactory/tx8-generic-dev/triton/offline_pkgs/offline_pkgs_v5.3.0.tar.gz
# 	if [ $? -eq 0 ]; then
# 		echo "Download offline package complete!"
# 	else
# 		echo "Download offline package fail!!!"
# 		exit -1
# 	fi
# 	tar -xzvf offline_pkgs_v5.3.0.tar.gz
# 	rm offline_pkgs_v5.3.0.tar.gz
# fi
# 
# if [ ! -d "./offline_pkgs" ]; then
# 	echo "fail: not find offline_pkgs!!!"
# 	exit -1
# fi
# 
# ###download tx8_deps(变化频率较高)
# if [ ! -e $tx8_depends_name.tar.gz ]; then
# 	if [ -e $TRITON_DEPENDS_SRC/$tx8_depends_name.tar.gz ]; then
# 		cp $TRITON_DEPENDS_SRC/$tx8_depends_name.tar.gz ./
# 		if [ -d "./tx8_deps" ]; then
# 			rm -rf tx8_deps
# 		fi
# 		tar -xzvf $tx8_depends_name.tar.gz
# 		echo "cp $TRITON_DEPENDS_SRC/$tx8_depends_name.tar.gz complete!"
# 	else
# 		echo "warning：$TRITON_DEPENDS_SRC/$tx8_depends_name.tar.gz not exist， use wget to download, maybe very slowly!"
# 	fi
# fi
# 
# if [ ! -e $tx8_depends_name.tar.gz ]; then
# 	if [ -d "./tx8_deps" ]; then
# 		rm -rf tx8_deps
# 	fi
# 	
# 	wget https://toolchain-jfrog.tsingmicro.xyz:443/artifactory/tx8-generic-dev/triton/tx8_depends/$tx8_depends_name.tar.gz
# 	if [ $? -eq 0 ]; then
# 		echo "Download tx8_deps complete!"
# 	else
# 		echo "Download tx8_dpes fail!!!"
# 		exit -1
# 	fi
# 	tar -xzvf $tx8_depends_name.tar.gz
# fi
# 
# if [ ! -d "./tx8_deps" ]; then
# 	echo "fail: not find tx8_deps!!!"
# 	exit -1
# fi
# 
# ###download torch_txda(变化频率较高)
# if [ ! -e $torch_txda_name.tar.gz ]; then
# 	if [ -e $TRITON_DEPENDS_SRC/$torch_txda_name.tar.gz ]; then
# 		cp $TRITON_DEPENDS_SRC/$torch_txda_name.tar.gz ./
# 		if [ -d "./pack" ]; then
# 			rm -rf pack
# 		fi
# 		tar -xzvf $torch_txda_name.tar.gz
# 		echo "cp $TRITON_DEPENDS_SRC/$torch_txda_name.tar.gz complete!"
# 	else
# 		echo "warning：$TRITON_DEPENDS_SRC/$torch_txda_name.tar.gz not exist， use wget to download, maybe very slowly!"
# 	fi
# fi
# 
# if [ ! -e $torch_txda_name.tar.gz ]; then
# 	if [ -d "./pack" ]; then
# 		rm -rf pack
# 	fi
# 	
# 	wget https://toolchain-jfrog.tsingmicro.xyz:443/artifactory/tx8-generic-dev/torch_txda/$torch_txda_name.tar.gz
# 	if [ $? -eq 0 ]; then
# 		echo "Download torch_txda complete!"
# 	else
# 		echo "Download torch_txda fail!!!"
# 		exit -1
# 	fi
# 	tar -xzvf $torch_txda_name.tar.gz
# fi
# 
# if [ ! -d "./pack" ]; then
# 	echo "fail: not find torch_txda pack!!!"
# 	exit -1
# fi
# 
# ##2.安装依赖(triton业务负责)
# cd triton
# if [ $skip_install -ne 1 ]; then
# 	if [ -d "./.venv" ]; then
# 		rm -rf .venv
# 	fi
# 	python3 -m venv .venv --prompt triton
# 	source .venv/bin/activate
# 	#check python version
# 	python3 --version
# 	bash third_party/tsingmicro/scripts/tools/offline_python_deps.sh -i -r python/requirements.txt -d ../offline_pkgs
# 	if [ $? -eq 0 ]; then
# 		echo "Install compile tool package complete!"
# 	else
# 		echo "Install compile tool package fail!!!"
# 		exit -1
# 	fi
# 
# 	bash third_party/tsingmicro/scripts/tools/offline_python_deps.sh -i -r third_party/tsingmicro/scripts/requirements_ts.txt -d ../offline_pkgs
# 	if [ $? -eq 0 ]; then
# 		echo "Install torch package complete!"
# 	else
# 		echo "Install torch package fail!!!"
# 		exit -1
# 	fi
# 	#check torch version
# 	python3 -c "import torch; print(torch.__version__)"
# 
# 	PROXY=http://192.168.100.225:8889
# 	export https_proxy=$PROXY http_proxy=$PROXY all_proxy=$PROXY
# 	apt install -y lld
# 	apt install ccache
# 	pip install loguru
# 	pip install scipy
# 	unset https_proxy
# 	unset http_proxy
# 	unset all_proxy
# 
# 	###install torch_txda(变化频率较高,须随着上述下载名字变化而变化)
# 	txops_wheel=$(find ../pack/ -maxdepth 1 -name "txops*.whl" -print -quit)
# 	torch_txda_wheel=$(find ../pack/ -maxdepth 1 -name "torch_txda*.whl" -print -quit)
# 	pip install $txops_wheel
# 	pip install $torch_txda_wheel
# fi
# 
# ##3.编译triton(triton业务负责)
# if [ $skip_build -ne 1 ]; then
# 	bash ./third_party/tsingmicro/scripts/build_tsingmicro.sh
# 	if [ $? -eq 0 ]; then
# 		echo "Build triton complete!"
# 	else
# 		echo "Build triton fail!!!"
# 		exit -1
# 	fi
# fi
##4.运行测试(triton业务负责)
#triton系统相关环境变量
TX8_DEPS_ROOT=$TRITON_WORKSPACE/tx8_deps
LLVM=$TRITON_WORKSPACE/llvm-a66376b0-ubuntu-x64
export TX8_DEPS_ROOT=$TX8_DEPS_ROOT
export LLVM_SYSPATH=$LLVM
export LLVM_BINARY_DIR=$LLVM/bin
export PYTHONPATH=$LLVM/python_packages/mlir_core:$PYTHONPATH
export LD_LIBRARY_PATH=$TX8_DEPS_ROOT/lib:$LD_LIBRARY_PATH
#export TRITON_DUMP_PATH=$TRITON_WORKSPACE/dump
export TRITON_ALWAYS_COMPILE=1
#测试任务相关环境变量
export JSON_FILE_PATH=$project_dir/flaggems/tests
#export TX8_DEVICES_COUNT=$device_count
export PRECISION_PRIORITY=$precision_priority
export TRITON_ALLOW_NON_CONSTEXPR_GLOBALS=1
export TXDA_SKIP_OPS=$txda_skip_ops
export TXDA_FALLBACK_CPU_OPS=$txda_fallback_cpu_ops

echo "TX8_DEPS_ROOT="$TX8_DEPS_ROOT
echo "LLVM_SYSPATH="$LLVM_SYSPATH
echo "LLVM_BINARY_DIR="$LLVM_BINARY_DIR
echo "PYTHONPATH="$PYTHONPATH
echo "LD_LIBRARY_PATH="$LD_LIBRARY_PATH
#echo "TRITON_DUMP_PATH="$TRITON_DUMP_PATH
echo "TRITON_ALWAYS_COMPILE="$TRITON_ALWAYS_COMPILE
echo "JSON_FILE_PATH="$JSON_FILE_PATH
#echo "TX8_DEVICES_COUNT="$TX8_DEVICES_COUNT
echo "PRECISION_PRIORITY="$PRECISION_PRIORITY
echo "TRITON_ALLOW_NON_CONSTEXPR_GLOBALS="$TRITON_ALLOW_NON_CONSTEXPR_GLOBALS
echo "TXDA_SKIP_OPS="$TXDA_SKIP_OPS
echo "TXDA_FALLBACK_CPU_OPS="$TXDA_FALLBACK_CPU_OPS

if [ $skip_run -ne 1 ]; then
	if [ $quick_mode -eq 1 ]; then
		python ./flaggems/tests/test_flag_gems_ci.py --test_set $test_set --device_count $device_count --skip_device $skip_device --quick
	else
		python ./flaggems/tests/test_flag_gems_ci.py --test_set $test_set --device_count $device_count --skip_device $skip_device --quick
	fi

	if [ $? -eq 0 ]; then
    		echo "Run test complete!"
	else
    		echo "Run test fail!!!"
    		exit -1
	fi
fi
