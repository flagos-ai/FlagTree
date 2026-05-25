[中文版|[English](./install_hcu.md)]

## 💫 HYGON（海光信息）[hcu](https://github.com/flagos-ai/FlagTree/tree/triton_v3.6.x/third_party/hcu/) (Triton 3.6)

- 对应的 Triton 版本为 3.6，基于 x64 平台
- 可用于 K100/BW1000

### 1. 构建及运行环境

#### 1.1 使用镜像（BW1000）

如果网络环境畅通，不必执行后续步骤 1.x，依赖库会在构建时自动拉取。

```shell
# Plan A: docker pull (28.4GB)
IMAGE=harbor.baai.ac.cn/flagtree/flagtree-hcu3.6-py310-torch2.9.0-ubuntu22.04:202604-base
docker pull ${IMAGE}
# Plan B: docker load (7.3GB)
IMAGE=flagtree-hcu3.6-py310-torch2.9.0-ubuntu22.04:202604-base
wget https://baai-cp-web.ks3-cn-beijing.ksyuncs.com/trans/flagtree-hcu3.6-py310-torch2.9.0-ubuntu22.04.202604-base.tar.gz
docker load -i flagtree-hcu3.6-py310-torch2.9.0-ubuntu22.04.202604-base.tar.gz
```

```shell
CONTAINER=flagtree-dev-xxx
docker run -dit \
    --network=host --ipc=host --privileged=true \
    --group-add video --cap-add=SYS_PTRACE \
    --security-opt seccomp=unconfined \
    --device=/dev/kfd --device=/dev/mkfd --device=/dev/dri \
    -v /opt/hyhal:/opt/hyhal \
    -v /etc/localtime:/etc/localtime:ro \
    -v /data:/data -v /home:/home -v /tmp:/tmp \
    -w /root --name ${CONTAINER} ${IMAGE}
docker exec -it ${CONTAINER} /bin/bash
```

#### 1.2 手动下载 FlagTree 依赖库

```shell
mkdir -p ~/.flagtree/hcu; cd ~/.flagtree/hcu
wget https://baai-cp-web.ks3-cn-beijing.ksyuncs.com/trans/hcu-llvm22-b0ca808-glibc2.35-glibcxx3.4.30-ubuntu-x86_64_v0.5.0.tar.gz
tar zxvf hcu-llvm22-b0ca808-glibc2.35-glibcxx3.4.30-ubuntu-x86_64_v0.5.0.tar.gz
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
python3.10 -m pip install flagtree===0.5.1+hcu3.6 $RES
```

安装 `flagtree` 后，可通过下列命令查看：

```shell
python3 -m pip show flagtree
```

#### 2.2 从源码构建

```shell
cd ${YOUR_CODE_DIR}/FlagTree
git checkout -b triton_v3.6.x origin/triton_v3.6.x
export FLAGTREE_BACKEND=hcu
MAX_JOBS=32 python3 -m pip install . --no-build-isolation -v
```

### 3. 测试验证

参考 [Tests of hcu3.6 backend](https://github.com/flagos-ai/FlagTree/blob/triton_v3.6.x/.github/workflows/hcu-build-and-test.yml)

## 💫 HYGON（海光信息）[hcu](/third_party/hcu/) (Triton 3.1)

- 对应的 Triton 版本为 3.1，基于 x64 平台
- 可用于 K100/BW1000

### 1. 构建及运行环境

#### 1.1 使用预装镜像（BW1000）

使用该预装镜像，则不必执行后续步骤 1.x。
如果网络环境畅通，也不必执行后续步骤 1.x，依赖库会在构建时自动拉取。

```shell
# Plan A: docker pull (22.7GB)
IMAGE=harbor.baai.ac.cn/flagtree/flagtree-hcu-py310-torch2.9.0-ubuntu22.04:202603
docker pull ${IMAGE}
# Plan B: docker load (5.7GB)
IMAGE=flagtree-hcu-py310-torch2.9.0-ubuntu22.04:202603
wget https://baai-cp-web.ks3-cn-beijing.ksyuncs.com/trans/flagtree-hcu-py310-torch2.9.0-ubuntu22.04.202603.tar.gz
docker load -i flagtree-hcu-py310-torch2.9.0-ubuntu22.04.202603.tar.gz
```

```shell
CONTAINER=flagtree-dev-xxx
docker run -dit \
    --network=host --ipc=host --privileged=true \
    --group-add video --cap-add=SYS_PTRACE \
    --security-opt seccomp=unconfined \
    --device=/dev/kfd --device=/dev/mkfd --device=/dev/dri \
    -v /opt/hyhal:/opt/hyhal \
    -v /etc/localtime:/etc/localtime:ro \
    -v /data:/data -v /home:/home -v /tmp:/tmp \
    -w /root --name ${CONTAINER} ${IMAGE}
docker exec -it ${CONTAINER} /bin/bash
```

#### 1.2 手动下载 FlagTree 依赖库

```shell
mkdir -p ~/.flagtree/hcu; cd ~/.flagtree/hcu
wget https://baai-cp-web.ks3-cn-beijing.ksyuncs.com/trans/hcu-llvm20-df0864e-glibc2.35-glibcxx3.4.30-ubuntu-x86_64_v0.3.0.tar.gz
tar zxvf hcu-llvm20-df0864e-glibc2.35-glibcxx3.4.30-ubuntu-x86_64_v0.3.0.tar.gz
```

#### 1.3 手动下载 Triton 依赖库

预装镜像中已下载安装 Triton 依赖库。
如果无需从源码构建 FlagTree 或 Triton，那么无需下载 Triton 依赖库。

```shell
cd ${YOUR_CODE_DIR}/FlagTree
# For Triton 3.1 (x64)
wget https://baai-cp-web.ks3-cn-beijing.ksyuncs.com/trans/build-deps-triton_3.1.x-linux-x64.tar.gz
sh python/scripts/unpack_triton_build_deps.sh ./build-deps-triton_3.1.x-linux-x64.tar.gz
```

执行完上述脚本后，原有的 ~/.triton 目录将被重命名，新的 ~/.triton 目录会被创建并存放预下载包。
注意执行脚本过程中会提示手动确认。

### 2. 安装命令

#### 2.1 免源码安装

```shell
# Note: First install PyTorch, then execute the following commands
python3 -m pip uninstall -y triton  # Repeat the cmd until fully uninstalled
RES="--index-url=https://resource.flagos.net/repository/flagos-pypi-hosted/simple"
python3.10 -m pip install flagtree===0.5.0+hcu3.0 $RES
```

预装镜像中已安装 `flagtree`，可通过下列命令查看：

```shell
python3 -m pip show flagtree
```

#### 2.2 从源码构建

```shell
cd ${YOUR_CODE_DIR}/FlagTree/python
export FLAGTREE_BACKEND=hcu
MAX_JOBS=32 python3 -m pip install . --no-build-isolation -v
```

### 3. 测试验证

参考 [Tests of hcu backend](/.github/workflows/hcu-build-and-test.yml)
