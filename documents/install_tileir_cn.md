[中文版|[English](./install_tileir.md)]

## 💫 NVIDIA TileIR [tileir](/third_party/tileir/) (Triton 3.6)

- 对应的 Triton 版本为 3.6，基于 x64 平台
- 可用于 Hopper/Blackwell

### 1. 构建及运行环境

#### 1.1 使用镜像（Hopper/Blackwell）

如果网络环境畅通，不必执行后续步骤 1.x，依赖库会在构建时自动拉取。

```shell
# Plan A: docker pull (24.9GB)
IMAGE=harbor.baai.ac.cn/flagtree/flagtree-py312-2.13.0a0_8145d630e8.nv26.06-cuda13.3-ubuntu24.04:202607-3.6-base
docker pull ${IMAGE}
# Plan B: docker load (10GB)
IMAGE=flagtree-py312-2.13.0a0_8145d630e8.nv26.06-cuda13.3-ubuntu24.04:202607-3.6-base
wget https://baai-cp-web.ks3-cn-beijing.ksyuncs.com/trans/flagtree-py312-2.13.0a0_8145d630e8.nv26.06-cuda13.3-ubuntu24.04.202607-3.6-base.tar.gz
docker load -i flagtree-py312-2.13.0a0_8145d630e8.nv26.06-cuda13.3-ubuntu24.04.202607-3.6-base.tar.gz
```

本镜像亦可用于 `nvidia` 后端。

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

#### 1.2 安装 FlagTree 依赖库

```shell
# For Triton 3.6
RES="--index-url=https://resource.flagos.net/repository/flagos-pypi-hosted/simple"
python3.12 -m pip install mlir $RES
```

#### 1.3 手动下载 Triton 依赖库

镜像中已下载安装 Triton 依赖库。
如果无需从源码构建 FlagTree 或 Triton，那么无需下载 Triton 依赖库。

```shell
cd ${YOUR_CODE_DIR}/FlagTree
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
python3.12 -m pip install flagtree===0.6.0+tileir3.6 $RES
```

安装 `flagtree` 后，可通过下列命令查看：

```shell
python3 -m pip show flagtree
```

#### 2.2 从源码构建

```shell
cd ${YOUR_CODE_DIR}/FlagTree
git checkout -b triton_v3.6.x origin/triton_v3.6.x
export FLAGTREE_BACKEND=tileir
export CMAKE_BUILD_PARALLEL_LEVEL=32
export TRITON_PTXAS_PATH="$PWD/third_party/nvidia/backend/bin/ptxas"
export TRITON_PTXAS_BLACKWELL_PATH="$PWD/third_party/nvidia/backend/bin/ptxas-blackwell"
MAX_JOBS=32 python3 -m pip install . --no-build-isolation -v
```

### 3. 测试验证

参考 [Tests of tileir3.6 backend](/.github/workflows/tileir3.6-build-and-test.yml)
