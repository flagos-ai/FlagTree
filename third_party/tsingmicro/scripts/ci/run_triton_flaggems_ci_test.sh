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
##  业务的ci运行脚本(功能正确性测试)                                                                                    ##
##     param1: skip_install, set 1-skip depends install,set 0-install depends, default 0.                               ##
##     param2: skip_build,   set 1-skip triton build,   set 0-build triton,    default 0.                               ##
##     param3: skip_run,     set 1-skip run ci test,    set 0-run ci test,     default 0.                               ##
##     param4: test_set,     set test set name, default 'ci_ops'.                                                       ##
##     param5: device_count, set device count number,   default 1.                                                      ##
##     ##param: precision_mode, set 1-triton compiler use high precision mode for special ops, default 2.               ##
##     param6: quick_mode,   set 1-quick mode to run flaggems, set 0-normal mode, default 0.                            ##
##     param7: skip_device,  set devices that need to be skipped, when they are unavailable, default [].                ##
##     param8: flaggems_path, set flaggems root path, default '$TRITON_WORKSPACE/flaggems'.                             ##
##                                                                                                                      ##
##########################################################################################################################

## Source common config and functions
source "$(dirname "$(realpath "$0")")/ci_common.sh"

skip_install=0
skip_build=0
skip_run=0
test_set=ci_ops
device_count=1
quick_mode=0
skip_device=
flaggems_path=flaggems

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
if [ $# -ge 8 ]; then
    flaggems_path=$8
fi

echo "param count:"$#
echo "skip_install:"$skip_install
echo "skip_build:"$skip_build
echo "skip_run:"$skip_run
echo "test_set:"$test_set
echo "device_count:"$device_count
echo "quick_mode:"$quick_mode
echo "skip_device:"$skip_device
echo "flaggems_path:"$flaggems_path
echo "precision_mode:"$precision_mode
echo "tx8_depends_name:"$tx8_depends_name
echo "torch_txda_name:"$torch_txda_name
echo "txops_name:"$txops_name
echo "txda_skip_ops:"$txda_skip_ops
echo "txda_fallback_cpu_ops:"$txda_fallback_cpu_ops

##1.下载依赖
if [ $skip_install -ne 1 ]; then
    download_deps
fi

##2.安装依赖
if [ $skip_install -ne 1 ]; then
    install_deps
fi

##3.编译triton
if [ $skip_build -ne 1 ]; then
    build_triton
fi

##4.运行测试
setup_env_vars

if [ $skip_run -ne 1 ]; then
    activate_venv_if_needed
    cleanup_before_run

    if [ $quick_mode -eq 1 ]; then
        python $flaggems_path/tests/test_flag_gems_ci.py --test_set $test_set --device_count $device_count --skip_device $skip_device --quick
    else
        python $flaggems_path/tests/test_flag_gems_ci.py --test_set $test_set --device_count $device_count --skip_device $skip_device
    fi

    if [ $? -eq 0 ]; then
        echo "Run test complete!"
    else
        echo "Run test fail!!!"
        exit -1
    fi
fi
