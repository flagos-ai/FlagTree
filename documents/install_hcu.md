[[中文版](./install_hcu_cn.md)|English]

## 💫 HYGON（海光信息）[hcu](/third_party/hcu/)

- Based on Triton 3.0, x64
- Available for K100/BW1000

### 1. Build and run environment

#### 1.1 Use the preinstalled image (BW1000)

If you use this preinstalled image, you do not need to perform the later step 1.x.
If your network connection is available, you also do not need to perform the later step 1.x, because dependencies will be fetched automatically during the build.

```shell
IMAGE=flagtree-hcu-py310-torch2.9.0-ubuntu22.04:202603
# Plan A: docker pull (22.7GB)
docker pull harbor.baai.ac.cn/flagtree/${IMAGE}
# Plan B: docker load (5.7GB)
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

#### 1.2 Manually download the FlagTree dependencies

```shell
mkdir -p ~/.flagtree/hcu; cd ~/.flagtree/hcu
wget https://baai-cp-web.ks3-cn-beijing.ksyuncs.com/trans/hcu-llvm20-df0864e-glibc2.35-glibcxx3.4.30-ubuntu-x86_64_v0.3.0.tar.gz
tar zxvf hcu-llvm20-df0864e-glibc2.35-glibcxx3.4.30-ubuntu-x86_64_v0.3.0.tar.gz
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
python3 -m pip install flagtree===0.5.0+hcu3.0 $RES
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
export FLAGTREE_BACKEND=hcu
MAX_JOBS=32 python3 -m pip install . --no-build-isolation -v
```

### 3. Testing and validation

Refer to [Tests of hcu backend](/.github/workflows/hcu-build-and-test.yml)
