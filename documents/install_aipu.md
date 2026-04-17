[[中文版](./install_aipu_cn.md)|English]

## 💫 ARM China（安谋科技）[aipu](https://github.com/flagos-ai/FlagTree/tree/triton_v3.3.x/third_party/aipu/)

- Based on Triton 3.3, x64/arm64

### 1. Build and run environment

#### 1.1 Use the preinstalled image (for the x64 CPU simulation environment)

If you use this preinstalled image, you do not need to perform the later step 1.x.
If your network connection is available, you also do not need to perform the later step 1.x, because dependencies will be fetched automatically during the build.

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

#### 1.2 Manually download the FlagTree dependencies

```shell
mkdir -p ~/.flagtree/aipu; cd ~/.flagtree/aipu
wget https://baai-cp-web.ks3-cn-beijing.ksyuncs.com/trans/llvm-a66376b0-ubuntu-x64-clang16-lld16_v0.4.0.tar.gz
tar zxvf llvm-a66376b0-ubuntu-x64-clang16-lld16_v0.4.0.tar.gz
```

#### 1.3 Manually download the Triton dependencies

The Triton dependencies are already downloaded and installed in the preinstalled image.
If you do not need to build FlagTree or Triton from source, you do not need to download the Triton dependencies.

```shell
cd ${YOUR_CODE_DIR}/FlagTree
# For Triton 3.3 (x64)
wget https://baai-cp-web.ks3-cn-beijing.ksyuncs.com/trans/build-deps-triton_3.3.x-linux-x64.tar.gz
sh python/scripts/unpack_triton_build_deps.sh ./build-deps-triton_3.3.x-linux-x64.tar.gz
```

After executing the above script, the original ~/.triton directory will be renamed, and a new ~/.triton directory will be created to store the pre-downloaded packages.
Note that the script will prompt for manual confirmation during execution.

### 2. Installation Commands

#### 2.1 Source-free Installation

```shell
# Note: First install PyTorch, then execute the following commands
python3 -m pip uninstall -y triton  # Repeat the cmd until fully uninstalled
RES="--index-url=https://resource.flagos.net/repository/flagos-pypi-hosted/simple"
python3 -m pip install flagtree===0.5.0+aipu3.3 $RES
```

`flagtree` is already installed in the preinstalled image. You can check it with:

```shell
python3 -m pip show flagtree
```

#### 2.2 Build from Source

Before building, you need to execute `source ~/env_setup.sh`. The content of this script is as follows:

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

### 3. Testing and validation

Before testing, you need to execute `source ~/env_setup.sh`. The content of this script is shown above.

Refer to [Tests of aipu backend](https://github.com/flagos-ai/FlagTree/blob/triton_v3.3.x/.github/workflows/aipu-build-and-test.yml)
