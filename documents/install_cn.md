[中文版|[English](./install.md)]

## 💫 NVIDIA & AMD [nvidia](/third_party/nvidia/) & [amd](/third_party/amd/)

- 对应的 Triton 版本为 3.1/3.2/3.3/3.4/3.5/3.6，基于 x64 平台

### 1. 构建及运行环境

#### 1.1 使用预装镜像（对应 Triton 3.6）

使用该预装镜像，则不必执行后续步骤 1.x，除非需要安装 FlagTree 的非 Triton 3.6 对应版本。
如果网络环境畅通，也不必执行后续步骤 1.x，依赖库会在构建时自动拉取。

```shell
IMAGE=flagtree-3.6.x-py312-torch2.8.0a0_5228986c39.nv25.05-ubuntu24.04:202603
# Plan A: docker pull (37GB)
docker pull harbor.baai.ac.cn/flagtree/${IMAGE}
# Plan B: docker load (17GB)
wget https://baai-cp-web.ks3-cn-beijing.ksyuncs.com/trans/flagtree-3.6.x-py312-torch2.8.0a0_5228986c39.nv25.05-ubuntu24.04.202603.tar.gz
docker load -i flagtree-3.6.x-py312-torch2.8.0a0_5228986c39.nv25.05-ubuntu24.04.202603.tar.gz
```

```shell
CONTAINER=flagtree-dev-xxx
docker run -dit \
    --net=host --uts=host --ipc=host --privileged \
    --ulimit stack=67108864 --ulimit memlock=-1 \
    --security-opt seccomp=unconfined \
    --gpus=all \
    -v /etc/localtime:/etc/localtime:ro \
    -v /data:/data -v /home:/home -v /tmp:/tmp \
    -w /root --name ${CONTAINER} ${IMAGE} bash
docker exec -it ${CONTAINER} /bin/bash
```

#### 1.2 手动下载 LLVM 依赖包

```shell
cd ${YOUR_LLVM_DOWNLOAD_DIR}
# For Triton 3.1
wget https://oaitriton.blob.core.windows.net/public/llvm-builds/llvm-10dc3a8e-ubuntu-x64.tar.gz
tar zxvf llvm-10dc3a8e-ubuntu-x64.tar.gz
export LLVM_SYSPATH=${YOUR_LLVM_DOWNLOAD_DIR}/llvm-10dc3a8e-ubuntu-x64
# For Triton 3.2
wget https://oaitriton.blob.core.windows.net/public/llvm-builds/llvm-86b69c31-ubuntu-x64.tar.gz
tar zxvf llvm-86b69c31-ubuntu-x64.tar.gz
export LLVM_SYSPATH=${YOUR_LLVM_DOWNLOAD_DIR}/llvm-86b69c31-ubuntu-x64
# For Triton 3.3
wget https://oaitriton.blob.core.windows.net/public/llvm-builds/llvm-a66376b0-ubuntu-x64.tar.gz
tar zxvf llvm-a66376b0-ubuntu-x64.tar.gz
export LLVM_SYSPATH=${YOUR_LLVM_DOWNLOAD_DIR}/llvm-a66376b0-ubuntu-x64
# For Triton 3.4
wget https://oaitriton.blob.core.windows.net/public/llvm-builds/llvm-8957e64a-ubuntu-x64.tar.gz
tar zxvf llvm-8957e64a-ubuntu-x64.tar.gz
export LLVM_SYSPATH=${YOUR_LLVM_DOWNLOAD_DIR}/llvm-8957e64a-ubuntu-x64
# For Triton 3.5
wget https://oaitriton.blob.core.windows.net/public/llvm-builds/llvm-7d5de303-ubuntu-x64.tar.gz
tar zxvf llvm-7d5de303-ubuntu-x64.tar.gz
export LLVM_SYSPATH=${YOUR_LLVM_DOWNLOAD_DIR}/llvm-7d5de303-ubuntu-x64
# For Triton 3.6 (Plan A)
wget https://oaitriton.blob.core.windows.net/public/llvm-builds/llvm-f6ded0be-ubuntu-x64.tar.gz
tar zxvf llvm-f6ded0be-ubuntu-x64.tar.gz
export LLVM_SYSPATH=${YOUR_LLVM_DOWNLOAD_DIR}/llvm-f6ded0be-ubuntu-x64
# For Triton 3.6 (Plan B for TLE-Raw)
RES="--index-url=https://resource.flagos.net/repository/flagos-pypi-hosted/simple"
python3.12 -m pip install mlir $RES
python3.12 -m pip show mlir
export LLVM_SYSPATH=${MLIR_INSTALL_DIR}/llvm_artifact
# For all versions
export LLVM_INCLUDE_DIRS=$LLVM_SYSPATH/include
export LLVM_LIBRARY_DIR=$LLVM_SYSPATH/lib
```

#### 1.3 手动下载 Triton 依赖库

预装镜像中已下载安装 Triton 依赖库。
如果无需从源码构建 FlagTree 或 Triton，那么无需下载 Triton 依赖库。

```shell
cd ${YOUR_CODE_DIR}/FlagTree
# For Triton 3.1 (x64)
wget https://baai-cp-web.ks3-cn-beijing.ksyuncs.com/trans/build-deps-triton_3.1.x-linux-x64.tar.gz
sh python/scripts/unpack_triton_build_deps.sh ./build-deps-triton_3.1.x-linux-x64.tar.gz
# For Triton 3.2 (x64)
wget https://baai-cp-web.ks3-cn-beijing.ksyuncs.com/trans/build-deps-triton_3.2.x-linux-x64.tar.gz
sh python/scripts/unpack_triton_build_deps.sh ./build-deps-triton_3.2.x-linux-x64.tar.gz
# For Triton 3.2 (aarch64)
wget https://baai-cp-web.ks3-cn-beijing.ksyuncs.com/trans/build-deps-triton_3.2.x-linux-aarch64.tar.gz
sh python/scripts/unpack_triton_build_deps.sh ./build-deps-triton_3.2.x-linux-aarch64.tar.gz
# For Triton 3.3 (x64)
wget https://baai-cp-web.ks3-cn-beijing.ksyuncs.com/trans/build-deps-triton_3.3.x-linux-x64.tar.gz
sh python/scripts/unpack_triton_build_deps.sh ./build-deps-triton_3.3.x-linux-x64.tar.gz
# For Triton 3.4 (x64)
wget https://baai-cp-web.ks3-cn-beijing.ksyuncs.com/trans/build-deps-triton_3.4.x-linux-x64.tar.gz
sh python/scripts/unpack_triton_build_deps.sh ./build-deps-triton_3.4.x-linux-x64.tar.gz
# For Triton 3.5 (x64)
wget https://baai-cp-web.ks3-cn-beijing.ksyuncs.com/trans/build-deps-triton_3.5.x-linux-x64.tar.gz
sh python/scripts/unpack_triton_build_deps.sh ./build-deps-triton_3.5.x-linux-x64.tar.gz
# For Triton 3.6 (x64)
wget https://baai-cp-web.ks3-cn-beijing.ksyuncs.com/trans/build-deps-triton_3.6.x-linux-x64.tar.gz
sh python/scripts/unpack_triton_build_deps.sh ./build-deps-triton_3.6.x-linux-x64.tar.gz
```

执行完上述脚本后，原有的 ~/.triton 目录将被重命名，新的 ~/.triton 目录会被创建并存放预下载包。
注意执行脚本过程中会提示手动确认。

### 2. 安装命令

#### 2.1 免源码安装

```shell
# Note: First install PyTorch, then execute the following commands
python3 -m pip uninstall -y triton  # Repeat the cmd until fully uninstalled
RES="--index-url=https://resource.flagos.net/repository/flagos-pypi-hosted/simple"
python3 -m pip install flagtree===0.5.0 $RES
```

预装镜像中已安装 `flagtree`，可通过下列命令查看：

```shell
python3 -m pip show flagtree
```

#### 2.2 从源码构建

```shell
apt update; apt install zlib1g zlib1g-dev libxml2 libxml2-dev
cd ${YOUR_CODE_DIR}/FlagTree
python3 -m pip install -r python/requirements.txt
cd python  # For Triton 3.1, 3.2, 3.3, you need to enter the python directory to build
git checkout main                                   # For Triton 3.1
git checkout -b triton_v3.2.x origin/triton_v3.2.x  # For Triton 3.2
git checkout -b triton_v3.3.x origin/triton_v3.3.x  # For Triton 3.3
git checkout -b triton_v3.4.x origin/triton_v3.4.x  # For Triton 3.4
git checkout -b triton_v3.5.x origin/triton_v3.5.x  # For Triton 3.5
git checkout -b triton_v3.6.x origin/triton_v3.6.x  # For Triton 3.6
unset FLAGTREE_BACKEND
MAX_JOBS=32 python3 -m pip install . --no-build-isolation -v
# If you need to build other backends afterward, you should clear LLVM-related environment variables
unset LLVM_SYSPATH LLVM_INCLUDE_DIRS LLVM_LIBRARY_DIR
```

### 3. 测试验证

参考 [Tests of nvidia backend](https://github.com/flagos-ai/FlagTree/blob/triton_v3.6.x/.github/workflows/hopper-build-and-test.yml)

## Q&A

### Q: 安装完成后，运行时报错：version GLIBC or GLIBCXX not found

A: 查询环境中的 libc.so.6、libstdc++.so.6.0.30 支持的 GLIBC / GLIBCXX 版本：

```shell
strings /lib/x86_64-linux-gnu/libc.so.6 |grep GLIBC
strings /usr/lib/x86_64-linux-gnu/libstdc++.so.6.0.30 | grep GLIBCXX
```

若支持该版本 GLIBC / GLIBCXX，那么也可尝试：

```shell
export LD_PRELOAD="/lib/x86_64-linux-gnu/libc.so.6"  # If GLIBC cannot be found
export LD_PRELOAD="/usr/lib/x86_64-linux-gnu/libstdc++.so.6.0.30"  # If GLIBCXX cannot be found
export LD_PRELOAD="/lib/x86_64-linux-gnu/libc.so.6 \
    /usr/lib/x86_64-linux-gnu/libstdc++.so.6.0.30"  # If neither GLIBC nor GLIBCXX can be found
```
