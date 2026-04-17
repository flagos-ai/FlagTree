[中文版|[English](./install_xpu.md)]

## 💫 KLX [xpu](/third_party/xpu/)

- 对应的 Triton 版本为 3.0，基于 x64 平台
- 可用于 P800

### 1. 构建及运行环境

#### 1.1 使用预装镜像（P800）

如果网络环境畅通，也不必执行后续步骤 1.x，依赖库会在构建时自动拉取。

```shell
IMAGE=flagtree-xpu-py310-torch2.5.1-ubuntu20.04:202604-base
# Plan A: docker pull (44.9GB)
docker pull harbor.baai.ac.cn/flagtree/${IMAGE}
# Plan B: docker load (21GB)
wget https://baai-cp-web.ks3-cn-beijing.ksyuncs.com/trans/flagtree-xpu-py310-torch2.5.1-ubuntu20.04.202604-base.tar.gz
docker load -i flagtree-xpu-py310-torch2.5.1-ubuntu20.04.202604-base.tar.gz
```

```shell
CONTAINER=flagtree-dev-xxx
docker run -dit \
    --net=host --privileged \
    --ulimit stack=67108864 --ulimit memlock=-1 \
    --ulimit nofile=120000 --shm-size=256g \
    --group-add video --cap-add=SYS_PTRACE --cap-add=SYS_ADMIN \
    --security-opt seccomp=unconfined \
    --device=/dev/xpu0:/dev/xpu0 --device=/dev/xpu1:/dev/xpu1 \
    --device=/dev/xpu2:/dev/xpu2 --device=/dev/xpu3:/dev/xpu3 \
    --device=/dev/xpu4:/dev/xpu4 --device=/dev/xpu5:/dev/xpu5 \
    --device=/dev/xpu6:/dev/xpu6 --device=/dev/xpu7:/dev/xpu7 \
    --device=/dev/xpuctrl:/dev/xpuctrl --device /dev/fuse \
    -v /etc/localtime:/etc/localtime:ro \
    -v /data:/data -v /home:/home -v /tmp:/tmp \
    -w /root --name ${CONTAINER} ${IMAGE} bash
docker exec -it ${CONTAINER} /bin/bash
```

#### 1.2 手动下载 FlagTree 依赖库

```shell
mkdir -p ~/.flagtree/xpu; cd ~/.flagtree/xpu
wget https://klx-sdk-release-public.su.bcebos.com/v1/triton/flaggems/2025_4_season/llvm/20260304/XTDK-llvm19-ubuntu2004_x86_64.tar.gz
tar zxvf XTDK-llvm19-ubuntu2004_x86_64.tar.gz
wget https://baai-cp-web.ks3-cn-beijing.ksyuncs.com/trans/xre-Linux-x86_64_v0.3.0.tar.gz
tar zxvf xre-Linux-x86_64_v0.3.0.tar.gz
wget https://klx-sdk-release-public.su.bcebos.com/XTriton/xpu-device-libs-ubuntu-x64_v0.3.6.1.1.tar.gz
tar zxvf xpu-device-libs-ubuntu-x64_v0.3.6.1.1.tar.gz
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
python3 -m pip install flagtree===0.5.1+xpu3.0 $RES
```

安装 `flagtree` 后，可通过下列命令查看：

```shell
python3 -m pip show flagtree
```

#### 2.2 从源码构建

```shell
apt update; apt install zlib1g zlib1g-dev libxml2 libxml2-dev
cd ${YOUR_CODE_DIR}/FlagTree/python
python3 -m pip install -r requirements.txt
export FLAGTREE_BACKEND=xpu
MAX_JOBS=32 python3 -m pip install . --no-build-isolation -v
```

### 3. 测试验证

参考 [Tests of xpu backend](/.github/workflows/xpu-build-and-test.yml)
