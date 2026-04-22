[[中文版](./install_iluvatar_cn.md)|English]

## 💫 ILUVATAR（天数智芯）[iluvatar](https://github.com/flagos-ai/FlagTree/tree/main/third_party/iluvatar/)

- Based on Triton 3.1, x64
- Available for MR-V100/BI-V150

### 1. Build and run environment

#### 1.1 Use the preinstalled image (BI-V150)

If your network connection is available, you do not need to perform the later step 1.x, because dependencies will be fetched automatically during the build.

```shell
modinfo iluvatar | grep "description"  # f65d8ac7
# Plan A: docker pull (17.9GB)
IMAGE=harbor.baai.ac.cn/flagtree/flagtree-iluvatar-py312-torch2.7.1-4.4.0release_f65d8ac7-ubuntu24.04:202604-base
docker pull ${IMAGE}
# Plan B: docker load (5.1GB)
IMAGE=flagtree-iluvatar-py312-torch2.7.1-4.4.0release_f65d8ac7-ubuntu24.04:202604-base
wget https://baai-cp-web.ks3-cn-beijing.ksyuncs.com/trans/flagtree-iluvatar-py312-torch2.7.1-4.4.0release_f65d8ac7-ubuntu24.04.202604-base.tar.gz
docker load -i flagtree-iluvatar-py312-torch2.7.1-4.4.0release_f65d8ac7-ubuntu24.04.202604-base.tar.gz
```

```shell
CONTAINER=flagtree-dev-xxx
docker run -dit \
    --net=host --pid=host --privileged \
    --cap-add=ALL \
    --security-opt seccomp=unconfined \
    -v /lib/modules:/lib/modules -v /dev:/dev  \
    -v /etc/localtime:/etc/localtime:ro \
    -v /data1:/data1 -v /home:/home -v /tmp:/tmp \
    -w /root --name ${CONTAINER} ${IMAGE}
docker exec -it ${CONTAINER} /bin/bash
```

#### 1.2 Manually download the FlagTree dependencies

```shell
mkdir -p ~/.flagtree/iluvatar; cd ~/.flagtree/iluvatar
wget https://baai-cp-web.ks3-cn-beijing.ksyuncs.com/trans/iluvatar-llvm18-x86_64_v0.5.0.tar.gz
tar zxvf iluvatar-llvm18-x86_64_v0.5.0.tar.gz
```

```shell
ABI=$(echo | g++ -dM -E -x c++ - | awk '/__GXX_ABI_VERSION/{print $3}')
# For python3.12 in the image
case "${ABI}" in
  1018) PLUGIN_TGZ=iluvatarTritonPlugin-cpython3.12-glibc2.39-glibcxx3.4.33-cxxabi1.3.15-ubuntu-x86_64_v0.5.0.tar.gz ;;
  *) echo "Unsupported __GXX_ABI_VERSION=${ABI}"; exit 1 ;;
esac
# For python3.10, not suitable for the image
case "${ABI}" in
  1013) PLUGIN_TGZ=iluvatarTritonPlugin-cpython3.10-glibc2.17-glibcxx3.4.19-cxxabi1.3.12-linux-x86_64_v0.5.0.tar.gz ;;
  1016) PLUGIN_TGZ=iluvatarTritonPlugin-cpython3.10-glibc2.35-glibcxx3.4.30-cxxabi1.3.13-ubuntu-x86_64_v0.5.0.tar.gz ;;
  1018) PLUGIN_TGZ=iluvatarTritonPlugin-cpython3.10-glibc2.39-glibcxx3.4.33-cxxabi1.3.15-ubuntu-x86_64_v0.5.0.tar.gz ;;
  *) echo "Unsupported __GXX_ABI_VERSION=${ABI}"; exit 1 ;;
esac
#
wget "https://baai-cp-web.ks3-cn-beijing.ksyuncs.com/trans/${PLUGIN_TGZ}"
tar zxvf "${PLUGIN_TGZ}"
```

#### 1.3 Manually download the Triton dependencies

The Triton dependencies are already downloaded and installed in the image.
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
python3.12 -m pip install flagtree===0.5.1+iluvatar3.1 $RES
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
export FLAGTREE_BACKEND=iluvatar
MAX_JOBS=32 python3 -m pip install . --no-build-isolation -v
```

### 3. Testing and validation

Refer to [Tests of iluvatar backend](/.github/workflows/iluvatar-build-and-test.yml)
