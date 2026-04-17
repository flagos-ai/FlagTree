[中文版|[English](./install_enflame.md)]

## 💫 Enflame（燧原）[enflame](https://github.com/flagos-ai/FlagTree/tree/triton_v3.5.x/third_party/enflame/)（triton 3.5）

- 对应的 Triton 版本为 3.5，基于 x64 平台
- 可用于 GCU300/GCU400

### 1. 构建及运行环境

#### 1.1 使用镜像（triton 3.5）

如果网络环境畅通，不必执行后续步骤 1.x，依赖库会在构建时自动拉取。

```shell
IMAGE=flagtree-enflame3.5-py312-torch2.9.1-ubuntu24.04:202603
# Plan A: docker pull (13.3GB)
docker pull harbor.baai.ac.cn/flagtree/${IMAGE}
# Plan B: docker load (2.8GB)
wget https://baai-cp-web.ks3-cn-beijing.ksyuncs.com/trans/flagtree-enflame3.5-py312-torch2.9.1-ubuntu24.04.202603.tar.gz
docker load -i flagtree-enflame3.5-py312-torch2.9.1-ubuntu24.04.202603.tar.gz
```

```shell
CONTAINER=flagtree-dev-xxx
docker run -dit \
    --privileged \
    -v /etc/localtime:/etc/localtime:ro \
    -v /home:/home \
    -w /root --name ${CONTAINER} ${IMAGE} bash
docker cp ${CONTAINER}:/enflame enflame    # Will create ./enflame dir
bash enflame/driver/enflame-x86_64-gcc-1.7.2.14-20260302150833.run
efsmi
docker stop ${CONTAINER}
docker start ${CONTAINER}
docker exec -it ${CONTAINER} /bin/bash
```

#### 1.2 手动下载 FlagTree 依赖库

```shell
mkdir -p ~/.flagtree/enflame; cd ~/.flagtree/enflame
wget https://baai-cp-web.ks3-cn-beijing.ksyuncs.com/trans/enflame-llvm22-189e06b-gcc9-x64_v0.4.0.tar.gz
tar zxvf enflame-llvm22-189e06b-gcc9-x64_v0.4.0.tar.gz
```

#### 1.3 手动下载 Triton 依赖库

预装镜像中已下载安装 Triton 依赖库。
如果无需从源码构建 FlagTree 或 Triton，那么无需下载 Triton 依赖库。

```shell
cd ${YOUR_CODE_DIR}/FlagTree
# For Triton 3.5 (x64)
wget https://baai-cp-web.ks3-cn-beijing.ksyuncs.com/trans/build-deps-triton_3.5.x-linux-x64.tar.gz
sh python/scripts/unpack_triton_build_deps.sh ./build-deps-triton_3.5.x-linux-x64.tar.gz
```

执行完上述脚本后，原有的 ~/.triton 目录将被重命名，新的 ~/.triton 目录会被创建并存放预下载包。
注意执行脚本过程中会提示手动确认。

### 2. 安装命令

#### 2.1 免源码安装

```shell
# Note: First install PyTorch, then execute the following commands
python3 -m pip uninstall -y triton --break-system-packages  # Repeat the cmd until fully uninstalled
RES="--index-url=https://resource.flagos.net/repository/flagos-pypi-hosted/simple"
python3 -m pip install flagtree===0.5.0+enflame3.5 --break-system-packages $RES
```

安装 `flagtree` 后，可通过下列命令查看：

```shell
python3 -m pip show flagtree
```

#### 2.2 从源码构建

```shell
apt update; apt install zlib1g zlib1g-dev libxml2 libxml2-dev
cd ${YOUR_CODE_DIR}/FlagTree
git checkout -b triton_v3.5.x origin/triton_v3.5.x
python3 -m pip install -r python/requirements.txt --break-system-packages
export FLAGTREE_BACKEND=enflame
MAX_JOBS=8 python3 -m pip install . --no-build-isolation -v --break-system-packages
```

### 3. 测试验证

参考 [Tests of enflame-gcu400 backend (triton 3.5)](https://github.com/flagos-ai/FlagTree/blob/triton_v3.5.x/.github/workflows/enflame-gcu400-build-and-test.yml)、[Tests of enflame-gcu300 backend (triton 3.5)](https://github.com/flagos-ai/FlagTree/blob/triton_v3.5.x/.github/workflows/enflame-gcu300-build-and-test.yml)

---

## 💫 Enflame（燧原）[enflame](https://github.com/flagos-ai/FlagTree/tree/triton_v3.3.x/third_party/enflame/)（triton 3.3 旧版本）

- 对应的 Triton 版本为 3.3，基于 x64 平台
- 可用于 GCU300

### 1. 构建及运行环境

#### 1.1 使用镜像（triton 3.3）

如果网络环境畅通，不必执行后续步骤 1.x，依赖库会在构建时自动拉取。

```shell
IMAGE=flagtree-enflame3.3-py310-torch2.7.0-ubuntu22.04:202603
# Plan A: docker pull (12.5GB)
docker pull harbor.baai.ac.cn/flagtree/${IMAGE}
# Plan B: docker load (5.7GB)
wget https://baai-cp-web.ks3-cn-beijing.ksyuncs.com/trans/flagtree-hcu-py310-torch2.9.0-ubuntu22.04.202603.tar.gz
docker load -i flagtree-hcu-py310-torch2.9.0-ubuntu22.04.202603.tar.gz
```

```shell
CONTAINER=flagtree-dev-xxx
docker run -dit \
    --privileged \
    -v /etc/localtime:/etc/localtime:ro \
    -v /home:/home \
    -w /root --name ${CONTAINER} ${IMAGE} bash
docker cp ${CONTAINER}:/enflame enflame    # Will create ./enflame dir
bash enflame/driver/enflame-x86_64-gcc-1.6.3.12-20251115104629.run
efsmi
docker stop ${CONTAINER}
docker start ${CONTAINER}
docker exec -it ${CONTAINER} /bin/bash
```

#### 1.2 手动下载 FlagTree 依赖库

```shell
mkdir -p ~/.flagtree/enflame; cd ~/.flagtree/enflame
wget https://baai-cp-web.ks3-cn-beijing.ksyuncs.com/trans/enflame-llvm21-d752c5b-gcc9-x64_v0.3.0.tar.gz
tar zxvf enflame-llvm21-d752c5b-gcc9-x64_v0.3.0.tar.gz
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
python3 -m pip install flagtree===0.4.0+enflame3.3 $RES
```

安装 `flagtree` 后，可通过下列命令查看：

```shell
python3 -m pip show flagtree
```

#### 2.2 从源码构建

```shell
apt update; apt install zlib1g zlib1g-dev libxml2 libxml2-dev
cd ${YOUR_CODE_DIR}/FlagTree/python
git checkout -b triton_v3.3.x origin/triton_v3.3.x
python3 -m pip install -r requirements.txt
export FLAGTREE_BACKEND=enflame
MAX_JOBS=8 python3 -m pip install . --no-build-isolation -v
```

### 3. 测试验证

参考 [Tests of enflame-gcu300 backend (triton 3.3)](https://github.com/flagos-ai/FlagTree/blob/triton_v3.3.x/.github/workflows/enflame-gcu300-3.3-build-and-test.yml)
