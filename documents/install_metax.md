[[中文版](./install_metax_cn.md)|English]

## 💫 MetaX（沐曦股份）[metax](/third_party/metax/)

- Based on Triton 3.0, x64
- Available for C550

### 1. Build and run environment

#### 1.1 Use the preinstalled image (C550)

If you use this preinstalled image, you do not need to perform the later step 1.x.
If your network connection is available, you also do not need to perform the later step 1.x, because dependencies will be fetched automatically during the build.

```shell
IMAGE=flagtree-metax-py312-torch2.8.0-vllm0.15.0-metax3.5.3.x-ubuntu22.04:202604-0.5.1
# Plan A: docker pull (28.1GB)
docker pull harbor.baai.ac.cn/flagtree/${IMAGE}
# Plan B: docker load (8.1GB)
wget https://baai-cp-web.ks3-cn-beijing.ksyuncs.com/trans/flagtree-metax-py312-torch2.8.0-vllm0.15.0-metax3.5.3.x-ubuntu22.04.202604-0.5.1.tar.gz
docker load -i flagtree-metax-py312-torch2.8.0-vllm0.15.0-metax3.5.3.x-ubuntu22.04.202604-0.5.1.tar.gz
```

```shell
CONTAINER=flagtree-dev-xxx
docker run -dit \
    --net=host --uts=host --ipc=host --privileged=true \
    --group-add video \
    --shm-size 100gb --ulimit memlock=-1 \
    --security-opt seccomp=unconfined --security-opt apparmor=unconfined \
    --device=/dev/dri --device=/dev/mxcd \
    -v /etc/localtime:/etc/localtime:ro \
    -v /data:/data -v /home:/home -v /tmp:/tmp \
    -w /root --name ${CONTAINER} ${IMAGE} bash
docker exec -it ${CONTAINER} /bin/bash
```

#### 1.2 Manually download the FlagTree dependencies

```shell
mkdir -p ~/.flagtree/metax; cd ~/.flagtree/metax
wget https://baai-cp-web.ks3-cn-beijing.ksyuncs.com/trans/metaxTritonPlugin-cpython3.12-glibc2.35-glibcxx3.4.30-cxxabi1.3.13-linux-x86_64_v0.5.0.tar.gz
tar zxvf metaxTritonPlugin-cpython3.12-glibc2.35-glibcxx3.4.30-cxxabi1.3.13-linux-x86_64_v0.5.0.tar.gz
wget https://baai-cp-web.ks3-cn-beijing.ksyuncs.com/trans/ext_maca_mathlib_bc_v0.5.0.tar.gz
tar zxvf ext_maca_mathlib_bc_v0.5.0.tar.gz
# NOTE: Contact the Metax vendor to obtain maca-llvm-metax20250708.521-x86_64.tar.xz
tar xvf maca-llvm-metax20250708.521-x86_64.tar.xz
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
python3.12 -m pip install flagtree===0.5.1+metax3.0 $RES
```

`flagtree` is already installed in the preinstalled image. You can check it with:

```shell
python3 -m pip show flagtree
```

#### 2.2 Build from Source

```shell
apt update; apt install zlib1g zlib1g-dev libxml2 libxml2-dev
cd ${YOUR_CODE_DIR}/FlagTree/python
python3 -m pip install -r requirements.txt
export FLAGTREE_BACKEND=metax
MAX_JOBS=32 python3 -m pip install . --no-build-isolation -v
```

### 3. Testing and validation

Refer to [Tests of metax backend](/.github/workflows/metax-build-and-test.yml)
