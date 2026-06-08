[[中文版](./install_ascend_cn.md)|English]

## 💫 Huawei Ascend（华为昇腾）[ascend](https://github.com/flagos-ai/FlagTree/blob/triton_v3.5.x/third_party/ascend) (Triton 3.5)

- Based on Triton 3.5, aarch64
- Available for 910B, 910C

### 1. Build and run environment

#### 1.1 Use the image (910B)

If your network connection is available, you do not need to perform the later step 1.x, because dependencies will be fetched automatically during the build.

```shell
# Plan A: docker pull (13.2GB)
IMAGE=harbor.baai.ac.cn/flagtree/flagtree-ascend3.5-910b-py311-cann9.0.0-ubuntu22.04-aarch64:202605-torch2.9.0-base
docker pull ${IMAGE}
# Plan B: docker load (4.7GB)
IMAGE=flagtree-ascend3.5-910b-py311-cann9.0.0-ubuntu22.04-aarch64:202605-torch2.9.0-base
wget https://baai-cp-web.ks3-cn-beijing.ksyuncs.com/trans/flagtree-ascend3.5-910b-py311-cann9.0.0-ubuntu22.04-aarch64.202605-torch2.9.0-base.tar.gz
docker load -i flagtree-ascend3.5-910b-py311-cann9.0.0-ubuntu22.04-aarch64.202605-torch2.9.0-base.tar.gz
```

```shell
CONTAINER=flagtree-dev-xxx
docker run -dit -u 0 --user=root \
    --network=host --pid=host --ipc=host --privileged \
    -v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
    -v /usr/local/Ascend/add-ons/:/usr/local/Ascend/add-ons/ \
    -v /usr/local/sbin/:/usr/local/sbin/ \
    -v /etc/ascend_install.info:/etc/ascend_install.info \
    --device=/dev/davinci0 --device=/dev/davinci1 \
    --device=/dev/davinci2 --device=/dev/davinci3 \
    --device=/dev/davinci4 --device=/dev/davinci5 \
    --device=/dev/davinci6 --device=/dev/davinci7 \
    --device=/dev/davinci_manager --device=/dev/devmm_svm --device=/dev/hisi_hdc \
    -v /etc/localtime:/etc/localtime:ro \
    -v /data:/data -v /home:/home -v /tmp:/tmp \
    -w /root --name ${CONTAINER} ${IMAGE} bash
docker exec -it ${CONTAINER} /bin/bash
```

#### 1.2 Install CANN

- A2 CANN is already installed in the 910B image. For 910C, register an account at https://www.hiascend.com/developer/download/community/result?module=cann and download the corresponding `cann-ops` for your platform.

```shell
# cann-toolkit (A2|A3)
chmod +x Ascend-cann-toolkit_9.0.0_linux-aarch64.run
./Ascend-cann-toolkit_9.0.0_linux-aarch64.run --install
# cann-ops for 910B (A2)
chmod +x Ascend-cann-910b-ops_9.0.0_linux-aarch64.run
./Ascend-cann-910b-ops_9.0.0_linux-aarch64.run --install
# cann-ops for 910C (A3)
chmod +x Ascend-cann-A3-ops_9.0.0_linux-aarch64.run
./Ascend-cann-A3-ops_9.0.0_linux-aarch64.run --install
```

#### 1.3 Manually download the FlagTree dependencies

```shell
mkdir -p ~/.flagtree/ascend; cd ~/.flagtree/ascend
wget https://baai-cp-web.ks3-cn-beijing.ksyuncs.com/trans/llvm-7d5de303-ubuntu-aarch64-python311-compat_v0.6.0.tar.gz
tar zxvf llvm-7d5de303-ubuntu-aarch64-python311-compat_v0.6.0.tar.gz
```

```shell
cd ${YOUR_CODE_DIR}/FlagTree/third_party
git clone https://github.com/flagos-ai/flir.git
```

```shell
cd ${YOUR_CODE_DIR}/FlagTree/third_party/ascend
git clone https://gitcode.com/Ascend/AscendNPU-IR.git
git checkout 4c304921
```

#### 1.4 Manually download the Triton dependencies

The Triton dependencies are already downloaded and installed in the image.
If you do not need to build FlagTree or Triton from source, you do not need to download the Triton dependencies.

```shell
cd ${YOUR_CODE_DIR}/FlagTree
# For Triton 3.5 (aarch64)
wget https://baai-cp-web.ks3-cn-beijing.ksyuncs.com/trans/build-deps-triton_3.5.x-linux-aarch64.tar.gz
sh python/scripts/unpack_triton_build_deps.sh ./build-deps-triton_3.5.x-linux-aarch64.tar.gz
```

After executing the above script, the original ~/.triton directory will be renamed, and a new ~/.triton directory will be created to store the pre-downloaded packages.
Note that the script will prompt for manual confirmation during execution.

### 2. Installation Commands

#### 2.1 Source-free Installation

```shell
# Note: First install PyTorch, then execute the following commands
python3 -m pip uninstall -y triton  # Repeat the cmd until fully uninstalled
RES="--index-url=https://resource.flagos.net/repository/flagos-pypi-hosted/simple"
python3.11 -m pip install flagtree===0.6.0rc1+ascend3.5 $RES
```

After installing `flagtree`, you can check it with:

```shell
python3 -m pip show flagtree
```

#### 2.2 Build from Source

```shell
cd ${YOUR_CODE_DIR}/FlagTree
git checkout -b triton_v3.5.x origin/triton_v3.5.x
export PATH=~/.flagtree/ascend/llvm-7d5de303-ubuntu-aarch64-python311-compat/bin/:${PATH}  # clang
export FLAGTREE_BACKEND=ascend
MAX_JOBS=32 python3 -m pip install . --no-build-isolation -v
```

### 3. Testing and validation

Before testing, you need to execute `source /usr/local/Ascend/ascend-toolkit/set_env.sh`

Refer to [Tests of ascend backend](https://github.com/flagos-ai/FlagTree/blob/triton_v3.5.x/.github/workflows/ascend-build-and-test.yml)

---

## 💫 Huawei Ascend（华为昇腾）[ascend](https://github.com/flagos-ai/FlagTree/blob/triton_v3.2.x/third_party/ascend) (Triton 3.2)

- Based on Triton 3.2, aarch64
- Available for 910B, 910C

### 1. Build and run environment

#### 1.1 Use the preinstalled image (910C)

This preinstalled image is created by executing the later step 1.x based on [Dockerfile-ubuntu22.04-python3.11-ascend](/dockerfiles/Dockerfile-ubuntu22.04-python3.11-ascend) and installing FlagTree.
If you use this preinstalled image, you do not need to perform the later step 1.x for 910C, and for 910B you only need to perform step 1.2.
If your network connection is available, you also do not need to perform the later step 1.x, because dependencies will be fetched automatically during the build.

```shell
# Plan A: docker pull (26.2GB)
IMAGE=harbor.baai.ac.cn/flagtree/flagtree-ascend-910c-py311-torch2.6.0-cann8.5.0-ubuntu22.04-aarch64:202603
docker pull ${IMAGE}
# Plan B: docker load (8.8GB)
IMAGE=flagtree-ascend-910c-py311-torch2.6.0-cann8.5.0-ubuntu22.04-aarch64:202603
wget https://baai-cp-web.ks3-cn-beijing.ksyuncs.com/trans/flagtree-ascend-910c-py311-torch2.6.0-cann8.5.0-ubuntu22.04-aarch64.202603.tar.gz
docker load -i flagtree-ascend-910c-py311-torch2.6.0-cann8.5.0-ubuntu22.04-aarch64.202603.tar.gz
```

```shell
CONTAINER=flagtree-dev-xxx
docker run -dit -u 0 --user=root \
    --network=host --pid=host --ipc=host --privileged \
    -v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
    -v /usr/local/Ascend/add-ons/:/usr/local/Ascend/add-ons/ \
    -v /usr/local/sbin/:/usr/local/sbin/ \
    -v /etc/ascend_install.info:/etc/ascend_install.info \
    --device=/dev/davinci0 --device=/dev/davinci1 \
    --device=/dev/davinci2 --device=/dev/davinci3 \
    --device=/dev/davinci4 --device=/dev/davinci5 \
    --device=/dev/davinci6 --device=/dev/davinci7 \
    --device=/dev/davinci_manager --device=/dev/devmm_svm --device=/dev/hisi_hdc \
    -v /etc/localtime:/etc/localtime:ro \
    -v /data:/data -v /home:/home -v /tmp:/tmp \
    -w /root --name ${CONTAINER} ${IMAGE} bash
docker exec -it ${CONTAINER} /bin/bash
```

#### 1.2 Install CANN

- A3 CANN is already installed in the 910C image. For 910B, register an account at https://www.hiascend.com/developer/download/community/result?module=cann and download the corresponding `cann-ops` for your platform.

```shell
# cann-toolkit (A2|A3)
chmod +x Ascend-cann-toolkit_8.5.0_linux-aarch64.run
./Ascend-cann-toolkit_8.5.0_linux-aarch64.run --install
# cann-ops for 910B (A2)
chmod +x Ascend-cann-910b-ops_8.5.0_linux-aarch64.run
./Ascend-cann-910b-ops_8.5.0_linux-aarch64.run --install
# cann-ops for 910C (A3)
chmod +x Ascend-cann-A3-ops_8.5.0_linux-aarch64.run
./Ascend-cann-A3-ops_8.5.0_linux-aarch64.run --install
```

#### 1.3 Manually download the FlagTree dependencies

```shell
mkdir -p ~/.flagtree/ascend; cd ~/.flagtree/ascend
wget https://baai-cp-web.ks3-cn-beijing.ksyuncs.com/trans/llvm-a66376b0-ubuntu-aarch64-python311-compat_v0.3.0.tar.gz
tar zxvf llvm-a66376b0-ubuntu-aarch64-python311-compat_v0.3.0.tar.gz
```

```shell
cd ${YOUR_CODE_DIR}/FlagTree/third_party
git clone https://github.com/flagos-ai/flir.git
cd flir
git checkout -b triton_v3.3.x origin/triton_v3.3.x  # For flagtree triton_v3.2.x triton_v3.3.x
```

```shell
cd ${YOUR_CODE_DIR}/FlagTree/third_party/ascend
git clone https://gitcode.com/Ascend/AscendNPU-IR.git
git checkout 5a3921f8
```

#### 1.4 Manually download the Triton dependencies

The Triton dependencies are already downloaded and installed in the preinstalled image.
If you do not need to build FlagTree or Triton from source, you do not need to download the Triton dependencies.

```shell
cd ${YOUR_CODE_DIR}/FlagTree
# For Triton 3.2 (aarch64)
wget https://baai-cp-web.ks3-cn-beijing.ksyuncs.com/trans/build-deps-triton_3.2.x-linux-aarch64.tar.gz
sh python/scripts/unpack_triton_build_deps.sh ./build-deps-triton_3.2.x-linux-aarch64.tar.gz
```

After executing the above script, the original ~/.triton directory will be renamed, and a new ~/.triton directory will be created to store the pre-downloaded packages.
Note that the script will prompt for manual confirmation during execution.

### 2. Installation Commands

#### 2.1 Source-free Installation

```shell
# Note: First install PyTorch, then execute the following commands
python3 -m pip uninstall -y triton  # Repeat the cmd until fully uninstalled
RES="--index-url=https://resource.flagos.net/repository/flagos-pypi-hosted/simple"
python3.11 -m pip install flagtree===0.6.0rc1+ascend3.2 $RES
```

`flagtree` is already installed in the preinstalled image. You can check it with:

```shell
python3 -m pip show flagtree
```

#### 2.2 Build from Source

```shell
cd ${YOUR_CODE_DIR}/FlagTree/python
git checkout -b triton_v3.2.x origin/triton_v3.2.x
export FLAGTREE_BACKEND=ascend
MAX_JOBS=32 python3 -m pip install . --no-build-isolation -v
```

### 3. Testing and validation

Before testing, you need to execute `source /usr/local/Ascend/ascend-toolkit/set_env.sh`

Refer to [Tests of ascend backend](https://github.com/flagos-ai/FlagTree/blob/triton_v3.2.x/.github/workflows/ascend-build-and-test.yml)
