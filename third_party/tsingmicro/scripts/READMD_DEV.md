# READMD DEV 版本

## 制作offline python离线包

- 制作：./offline_python_deps.sh -p -r requirements_ts.txt -o offline_pkgs
- 安装：
  - ./offline_python_deps.sh -i -r requirements_ts.txt -d offline_pkgs
  - 虚拟环境: 先手动创建，激活虚拟环境
  <!-- - 虚拟环境：./offline_python_deps.sh -i -r requirements_ts.txt -d offline_pkgs -v my_venv -->
- 增量制作：./supplement_packages.sh requirements.txt existing_packages_dir

## 常用环境搭建命令

### 设置代理

```shell
export https_proxy=http://192.168.100.225:8889
export http_proxy=http://192.168.100.225:8889
```

### 虚拟环境构建

```shell
python3 -m venv .venv --prompt triton
source .venv/bin/activate
./third_party/tsingmicro/scripts/tools/offline_python_deps.sh -i -r third_party/tsingmicro/scripts/requirements_ts.txt -d ../offline_pkgs/
./third_party/tsingmicro/scripts/tools/offline_python_deps.sh -i -r python/requirements.txt -d ../offline_pkgs/
apt install lld
```

### 宿主机依赖包安装

某个docker版本后就会需要

```shell
apt install openmpi-bin openmpi-doc libopenmpi-dev
```

## 安装torch_txda

- 下载http://gitlab.tsingmicro.com/tx8_toolchain/torch_txda
- python setup.py bdist_wheel
- 安装whl

## 安装triton

- ./third_party/tsingmicro/scripts/build_tsingmicro.sh

## 运行

- ./third_party/tsingmicro/scripts/run_tsingmicro.sh python third_party/tsingmicro/examples/profile_matmul.py
- ./third_party/tsingmicro/scripts/run_tsingmicro.sh pytest ../flaggems/tests/test_unary_pointwise_ops.py::test_accuracy_abs --ref cpu

## profile使用

```shell
# 仅开启host侧profile，准确的获取launch的时间，需要重新编译
export USE_HOST_PROFILE=1
# 开启全部profile，包括device的，立即生效，注意因为插桩和打印的影响，此时host侧的launch时间已经不准了
export USE_PROFILE=1
#自己定义需要对什么指令做profile，必填
export TRACE_POINTS="__Rdma,__Wdma"

./third_party/tsingmicro/scripts/run_tsingmicro.sh python xxxxxx
```