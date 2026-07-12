[[中文版](./install_tileir_cn.md)|English]

## 💫 NVIDIA TileIR [tileir](/third_party/tileir/) (Triton 3.6)

- Based on Triton 3.6, x64
- Available for Hopper/Blackwell

### 1. Build and run environment

#### 1.1 Use the image (Hopper/Blackwell)

If your network connection is available, you do not need to perform the later step 1.x, because dependencies will be fetched automatically during the build.

```shell
# Plan A: docker pull (24.9GB)
IMAGE=harbor.baai.ac.cn/flagtree/flagtree-py312-2.13.0a0_8145d630e8.nv26.06-cuda13.3-ubuntu24.04:202607-3.6-base
docker pull ${IMAGE}
# Plan B: docker load (10GB)
IMAGE=flagtree-py312-2.13.0a0_8145d630e8.nv26.06-cuda13.3-ubuntu24.04:202607-3.6-base
wget https://baai-cp-web.ks3-cn-beijing.ksyuncs.com/trans/flagtree-py312-2.13.0a0_8145d630e8.nv26.06-cuda13.3-ubuntu24.04.202607-3.6-base.tar.gz
docker load -i flagtree-py312-2.13.0a0_8145d630e8.nv26.06-cuda13.3-ubuntu24.04.202607-3.6-base.tar.gz
```

This image can also be used for the `nvidia` backend.

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

#### 1.2 Install the FlagTree dependencies

```shell
# For Triton 3.6
RES="--index-url=https://resource.flagos.net/repository/flagos-pypi-hosted/simple"
python3.12 -m pip install mlir $RES
```

#### 1.3 Manually download the Triton dependencies

The Triton dependencies are already downloaded and installed in the image.
If you do not need to build FlagTree or Triton from source, you do not need to download the Triton dependencies.

```shell
cd ${YOUR_CODE_DIR}/FlagTree
# For Triton 3.6 (x64)
wget https://baai-cp-web.ks3-cn-beijing.ksyuncs.com/trans/build-deps-triton_3.6.x-linux-x64.tar.gz
sh python/scripts/unpack_triton_build_deps.sh ./build-deps-triton_3.6.x-linux-x64.tar.gz
```

After executing the above script, the original ~/.triton directory will be renamed, and a new ~/.triton directory will be created to store the pre-downloaded packages.
Note that the script will prompt for manual confirmation during execution.

### 2. Installation Commands

#### 2.1 Source-free Installation

```shell
# Note: First install PyTorch, then execute the following commands
python3 -m pip uninstall -y triton  # Repeat the cmd until fully uninstalled
RES="--index-url=https://resource.flagos.net/repository/flagos-pypi-hosted/simple"
python3.12 -m pip install flagtree===0.6.0+tileir3.6 $RES
```

After installing `flagtree`, you can check it with:

```shell
python3 -m pip show flagtree
```

#### 2.2 Build from Source

```shell
cd ${YOUR_CODE_DIR}/FlagTree
git checkout -b triton_v3.6.x origin/triton_v3.6.x
export FLAGTREE_BACKEND=tileir
export CMAKE_BUILD_PARALLEL_LEVEL=32
export TRITON_PTXAS_PATH="$PWD/third_party/nvidia/backend/bin/ptxas"
export TRITON_PTXAS_BLACKWELL_PATH="$PWD/third_party/nvidia/backend/bin/ptxas-blackwell"
MAX_JOBS=32 python3 -m pip install . --no-build-isolation -v
```

### 3. Testing and validation

Refer to [Tests of tileir3.6 backend](/.github/workflows/tileir3.6-build-and-test.yml)
