[[中文版](./install_tsingmicro_cn.md)|English]

## 💫 Tsingmicro（清微智能）[tsingmicro](https://github.com/flagos-ai/FlagTree/tree/triton_v3.3.x/third_party/tsingmicro/) (Triton 3.3)

- Based on Triton 3.3, x64
- Available for TX81

### 1. Build and run environment

#### 1.1 Use the image (TX81)

If your network connection is available, you do not need to perform the later step 1.x, because dependencies will be fetched automatically during the build.

```shell
# Plan A: docker pull (13.2GB)
IMAGE=harbor.baai.ac.cn/flagtree/flagtree-tsingmicro3.3-py310-torch2.7.0-ubuntu22.04:202606-clean
docker pull ${IMAGE}
# Plan B: docker load (5.5GB)
IMAGE=flagtree-tsingmicro3.3-py310-torch2.7.0-ubuntu22.04:202606-clean
wget https://baai-cp-web.ks3-cn-beijing.ksyuncs.com/trans/flagtree-tsingmicro3.3-py310-torch2.7.0-ubuntu22.04.202606-clean.tar.gz
docker load -i flagtree-tsingmicro3.3-py310-torch2.7.0-ubuntu22.04.202606-clean.tar.gz
```

```shell
CONTAINER=flagtree-dev-xxx
docker run -dit \
    --network=host --ipc=host --privileged \
    --security-opt seccomp=unconfined \
    -v /dev:/dev -v /lib/modules:/lib/modules -v /sys:/sys \
    -v /etc/localtime:/etc/localtime:ro \
    -v /data:/data -v /home:/home -v /tmp:/tmp \
    -w /root --name ${CONTAINER} ${IMAGE} bash
docker exec -it ${CONTAINER} /bin/bash
```

#### 1.2 Manually download the FlagTree dependencies

For the tsingmicro backend, the FlagTree dependency library is also required at runtime.

```shell
mkdir -p ~/.flagtree/tsingmicro; cd ~/.flagtree/tsingmicro
wget https://baai-cp-web.ks3-cn-beijing.ksyuncs.com/trans/tsingmicro-llvm21-glibc2.30-glibcxx3.4.28-python3.10-x64_v0.6.0.tar.gz
tar zxvf tsingmicro-llvm21-glibc2.30-glibcxx3.4.28-python3.10-x64_v0.6.0.tar.gz
wget https://baai-cp-web.ks3-cn-beijing.ksyuncs.com/trans/tx8_depends_dev_20260507_104051_v0.6.0.tar.gz
tar zxvf tx8_depends_dev_20260507_104051_v0.6.0.tar.gz
```

#### 1.3 Manually download the Triton dependencies

The Triton dependencies are already downloaded and installed in the image.
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
python3.10 -m pip install flagtree===0.6.0+tsingmicro3.3 $RES
```

After installing `flagtree`, you can check it with:

```shell
python3 -m pip show flagtree
```

#### 2.2 Build from Source

Before building and testing, you need to set the following environment variables:

```bash
export TX8_DEPS_ROOT=~/.flagtree/tsingmicro/tx8_deps
export LLVM_SYSPATH=~/.flagtree/tsingmicro/tsingmicro-llvm21-glibc2.30-glibcxx3.4.28-python3.10-x64
export LLVM_BINARY_DIR=${LLVM_SYSPATH}/bin
export PYTHONPATH=${LLVM_SYSPATH}/python_packages/mlir_core:$PYTHONPATH
export LD_LIBRARY_PATH=$TX8_DEPS_ROOT/lib:$LD_LIBRARY_PATH
export PATH=${LLVM_BINARY_DIR}:${PATH}
```

```shell
cd ${YOUR_CODE_DIR}/FlagTree/python
git checkout -b triton_v3.3.x origin/triton_v3.3.x
export FLAGTREE_BACKEND=tsingmicro
MAX_JOBS=32 python3 -m pip install . --no-build-isolation -v
```

### 3. Testing and validation

Before testing, you need to set the environment variables as described above.

Refer to [Tests of tsingmicro3.3 backend](https://github.com/flagos-ai/FlagTree/blob/triton_v3.3.x/.github/workflows/tsingmicro3.3-build-and-test.yml)
