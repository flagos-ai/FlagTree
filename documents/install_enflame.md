[[中文版](./install_enflame_cn.md)|English]

## 💫 Enflame（燧原）[enflame](https://github.com/flagos-ai/FlagTree/tree/triton_v3.5.x/third_party/enflame/) (Triton 3.5)

- Based on Triton 3.5, x64
- Available for GCU300/GCU400

### 1. Build and run environment

#### 1.1 Use the image (Triton 3.5)

If your network connection is available, you do not need to perform the later step 1.x, because dependencies will be fetched automatically during the build.

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

#### 1.2 Manually download the FlagTree dependencies

```shell
mkdir -p ~/.flagtree/enflame; cd ~/.flagtree/enflame
wget https://baai-cp-web.ks3-cn-beijing.ksyuncs.com/trans/enflame-llvm22-189e06b-gcc9-x64_v0.4.0.tar.gz
tar zxvf enflame-llvm22-189e06b-gcc9-x64_v0.4.0.tar.gz
```

#### 1.3 Manually download the Triton dependencies

The Triton dependencies are already downloaded and installed in the preinstalled image.
If you do not need to build FlagTree or Triton from source, you do not need to download the Triton dependencies.

```shell
cd ${YOUR_CODE_DIR}/FlagTree
# For Triton 3.5 (x64)
wget https://baai-cp-web.ks3-cn-beijing.ksyuncs.com/trans/build-deps-triton_3.5.x-linux-x64.tar.gz
sh python/scripts/unpack_triton_build_deps.sh ./build-deps-triton_3.5.x-linux-x64.tar.gz
```

After executing the above script, the original ~/.triton directory will be renamed, and a new ~/.triton directory will be created to store the pre-downloaded packages.
Note that the script will prompt for manual confirmation during execution.

### 2. Installation Commands

#### 2.1 Source-free Installation

```shell
# Note: First install PyTorch, then execute the following commands
python3 -m pip uninstall -y triton --break-system-packages  # Repeat the cmd until fully uninstalled
RES="--index-url=https://resource.flagos.net/repository/flagos-pypi-hosted/simple"
python3 -m pip install flagtree===0.5.0+enflame3.5 --break-system-packages $RES
```

After installing `flagtree`, you can check it with:

```shell
python3 -m pip show flagtree
```

#### 2.2 Build from Source

```shell
apt update; apt install zlib1g zlib1g-dev libxml2 libxml2-dev
cd ${YOUR_CODE_DIR}/FlagTree
git checkout -b triton_v3.5.x origin/triton_v3.5.x
python3 -m pip install -r python/requirements.txt --break-system-packages
export FLAGTREE_BACKEND=enflame
MAX_JOBS=8 python3 -m pip install . --no-build-isolation -v --break-system-packages
```

### 3. Testing and validation

Refer to [Tests of enflame-gcu400 backend (Triton 3.5)](https://github.com/flagos-ai/FlagTree/blob/triton_v3.5.x/.github/workflows/enflame-gcu400-build-and-test.yml), [Tests of enflame-gcu300 backend (Triton 3.5)](https://github.com/flagos-ai/FlagTree/blob/triton_v3.5.x/.github/workflows/enflame-gcu300-build-and-test.yml)

---

## 💫 Enflame（燧原）[enflame](https://github.com/flagos-ai/FlagTree/tree/triton_v3.3.x/third_party/enflame/) (older Triton 3.3 version)

- Based on Triton 3.3, x64
- Available for GCU300

### 1. Build and run environment

#### 1.1 Use the image (Triton 3.3)

If your network connection is available, you do not need to perform the later step 1.x, because dependencies will be fetched automatically during the build.

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

#### 1.2 Manually download the FlagTree dependencies

```shell
mkdir -p ~/.flagtree/enflame; cd ~/.flagtree/enflame
wget https://baai-cp-web.ks3-cn-beijing.ksyuncs.com/trans/enflame-llvm21-d752c5b-gcc9-x64_v0.3.0.tar.gz
tar zxvf enflame-llvm21-d752c5b-gcc9-x64_v0.3.0.tar.gz
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
python3 -m pip install flagtree===0.4.0+enflame3.3 $RES
```

After installing `flagtree`, you can check it with:

```shell
python3 -m pip show flagtree
```

#### 2.2 Build from Source

```shell
apt update; apt install zlib1g zlib1g-dev libxml2 libxml2-dev
cd ${YOUR_CODE_DIR}/FlagTree/python
git checkout -b triton_v3.3.x origin/triton_v3.3.x
python3 -m pip install -r requirements.txt
export FLAGTREE_BACKEND=enflame
MAX_JOBS=8 python3 -m pip install . --no-build-isolation -v
```

### 3. Testing and validation

Refer to [Tests of enflame-gcu300 backend (Triton 3.3)](https://github.com/flagos-ai/FlagTree/blob/triton_v3.3.x/.github/workflows/enflame-gcu300-3.3-build-and-test.yml)
