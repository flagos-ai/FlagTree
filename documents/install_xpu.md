[[中文版](./install_xpu_cn.md)|English]

## 💫 KLX [xpu](/third_party/xpu/)

- Based on Triton 3.0, x64
- Available for P800

### 1. Build and run environment

#### 1.1 Use the preinstalled image (P800)

If your network connection is available, you also do not need to perform the later step 1.x, because dependencies will be fetched automatically during the build.

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

#### 1.2 Manually download the FlagTree dependencies

```shell
mkdir -p ~/.flagtree/xpu; cd ~/.flagtree/xpu
wget https://klx-sdk-release-public.su.bcebos.com/v1/triton/flaggems/2025_4_season/llvm/20260304/XTDK-llvm19-ubuntu2004_x86_64.tar.gz
tar zxvf XTDK-llvm19-ubuntu2004_x86_64.tar.gz
wget https://baai-cp-web.ks3-cn-beijing.ksyuncs.com/trans/xre-Linux-x86_64_v0.3.0.tar.gz
tar zxvf xre-Linux-x86_64_v0.3.0.tar.gz
wget https://klx-sdk-release-public.su.bcebos.com/XTriton/xpu-device-libs-ubuntu-x64_v0.3.6.1.1.tar.gz
tar zxvf xpu-device-libs-ubuntu-x64_v0.3.6.1.1.tar.gz
```

#### 1.3 Manually download the Triton dependencies

The Triton dependencies are already downloaded and installed in the preinstalled image.
If you do not need to build FlagTree or Triton from source, you do not need to download the Triton dependencies.

```shell
cd ${YOUR_CODE_DIR}/FlagTree
# For Triton 3.1 (x64)
wget https://baai-cp-web.ks3-cn-beijing.ksyuncs.com/trans/build-deps-triton_3.1.x-linux-x64.tar.gz
sh python/scripts/unpack_triton_build_deps.sh ./build-deps-triton_3.1.x-linux-x64.tar.gz
```

After executing the above script, the original ~/.triton directory will be renamed, and a new ~/.triton directory will be created to store the pre-downloaded packages.
Note that the script will prompt for manual confirmation during execution.

### 2. Installation Commands

#### 2.1 Source-free Installation

```shell
# Note: First install PyTorch, then execute the following commands
python3 -m pip uninstall -y triton  # Repeat the cmd until fully uninstalled
RES="--index-url=https://resource.flagos.net/repository/flagos-pypi-hosted/simple"
python3 -m pip install flagtree===0.5.1+xpu3.0 $RES
```

After installing `flagtree`, you can check it with:

```shell
python3 -m pip show flagtree
```

#### 2.2 Build from Source

```shell
apt update; apt install zlib1g zlib1g-dev libxml2 libxml2-dev
cd ${YOUR_CODE_DIR}/FlagTree/python
python3 -m pip install -r requirements.txt
export FLAGTREE_BACKEND=xpu
MAX_JOBS=32 python3 -m pip install . --no-build-isolation -v
```

### 3. Testing and validation

Refer to [Tests of xpu backend](/.github/workflows/xpu-build-and-test.yml)
