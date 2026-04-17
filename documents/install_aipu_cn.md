[中文版|[English](./install_aipu.md)]

## 💫 ARM China（安谋科技）[aipu](https://github.com/flagos-ai/FlagTree/tree/triton_v3.3.x/third_party/aipu/)

- 对应的 Triton 版本为 3.3，基于 x64/arm64 平台

### 1. 构建及运行环境

#### 1.1 使用预装镜像（x64 cpu 模拟环境）

使用该预装镜像，则不必执行后续步骤 1.x。
如果网络环境畅通，也不必执行后续步骤 1.x，依赖库会在构建时自动拉取。

```shell
IMAGE=flagtree-aipu-py310-torch2.6.0-clang16-lld16-ubuntu22.04:202603
# Plan A: docker pull (36.9GB)
docker pull harbor.baai.ac.cn/flagtree/${IMAGE}
# Plan B: docker load (17GB)
wget https://baai-cp-web.ks3-cn-beijing.ksyuncs.com/trans/flagtree-aipu-py310-torch2.6.0-clang16-lld16-ubuntu22.04.202603.tar.gz
docker load -i flagtree-aipu-py310-torch2.6.0-clang16-lld16-ubuntu22.04.202603.tar.gz
```

```shell
CONTAINER=flagtree-dev-xxx
docker run -dit \
    --net=host --uts=host --ipc=host --privileged \
    --shm-size 100gb --ulimit memlock=-1 \
    --security-opt seccomp=unconfined --security-opt apparmor=unconfined \
    -v /etc/localtime:/etc/localtime:ro \
    -v /data:/data -v /home:/home -v /tmp:/tmp \
    -w /root --name ${CONTAINER} ${IMAGE} bash
docker exec -it ${CONTAINER} /bin/bash
```

#### 1.2 手动下载 FlagTree 依赖库

```shell
mkdir -p ~/.flagtree/aipu; cd ~/.flagtree/aipu
wget https://baai-cp-web.ks3-cn-beijing.ksyuncs.com/trans/llvm-a66376b0-ubuntu-x64-clang16-lld16_v0.4.0.tar.gz
tar zxvf llvm-a66376b0-ubuntu-x64-clang16-lld16_v0.4.0.tar.gz
```

#### 1.3 手动下载 Triton 依赖库

预装镜像中已下载安装 Triton 依赖库。
如果无需从源码构建 FlagTree 或 Triton，那么无需下载 Triton 依赖库。

```shell
cd ${YOUR_CODE_DIR}/FlagTree
# For Triton 3.3 (x64)
wget https://baai-cp-web.ks3-cn-beijing.ksyuncs.com/trans/build-deps-triton_3.3.x-linux-x64.tar.gz
sh python/scripts/unpack_triton_build_deps.sh ./build-deps-triton_3.3.x-linux-x64.tar.gz
```

执行完上述脚本后，原有的 ~/.triton 目录将被重命名，新的 ~/.triton 目录会被创建并存放预下载包。
注意执行脚本过程中会提示手动确认。

### 2. 安装命令

#### 2.1 免源码安装

```shell
# Note: First install PyTorch, then execute the following commands
python3 -m pip uninstall -y triton  # Repeat the cmd until fully uninstalled
RES="--index-url=https://resource.flagos.net/repository/flagos-pypi-hosted/simple"
python3 -m pip install flagtree===0.5.0+aipu3.3 $RES
```

预装镜像中已安装 `flagtree`，可通过下列命令查看：

```shell
python3 -m pip show flagtree
```

#### 2.2 从源码构建

构建前需执行 `source ~/env_setup.sh`，该脚本内容如下：

```bash
SDK_PATH=~/AI610-SDK-dev-4.0.7
export PATH=${PATH}:${SDK_PATH}/AI610-SDK-1002/simulator/bin:${SDK_PATH}/AI100-SDK-0006/opencl-tool-chain/compiler/aipu_opencl_toolchain/bin
export LD_LIBRARY_PATH=${SDK_PATH}/AI610-SDK-1002/simulator/lib:${SDK_PATH}/AI610-SDK-1012/Linux-driver/bin/sim/release:${LD_LIBRARY_PATH}
export ZHOUYI_LINUX_DRIVER_HOME=${SDK_PATH}/AI610-SDK-1012/Linux-driver
export PYTHONPATH=~/.flagtree/aipu/llvm-a66376b0-ubuntu-x64-clang16-lld16/python_packages/mlir_core:${PYTHONPATH}
```

```shell
apt update; apt install zlib1g zlib1g-dev libxml2 libxml2-dev
cd ${YOUR_CODE_DIR}/FlagTree/python
git checkout -b triton_v3.3.x origin/triton_v3.3.x
python3 -m pip install -r requirements.txt
export FLAGTREE_BACKEND=aipu
MAX_JOBS=32 python3 -m pip install . --no-build-isolation -v
```

### 3. 测试验证

测试前需执行 `source ~/env_setup.sh`，该脚本内容见上文。

参考 [Tests of aipu backend](https://github.com/flagos-ai/FlagTree/blob/triton_v3.3.x/.github/workflows/aipu-build-and-test.yml)
