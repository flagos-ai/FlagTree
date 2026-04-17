[[中文版](./install_sunrise_cn.md)|English]

## 💫 Sunrise（曦望芯科）[sunrise](https://github.com/flagos-ai/FlagTree/tree/triton_v3.4.x/third_party/sunrise/)

- Based on Triton 3.4, x64
- Available for S2

### 1. Build and run environment

#### 1.1 Use the preinstalled image (S2)

If you use this preinstalled image, you do not need to perform the later step 1.x.
If your network connection is available, you also do not need to perform the later step 1.x, because dependencies will be fetched automatically during the build.

```shell
TODO
```

#### 1.2 Manually download the FlagTree dependencies

```shell
mkdir -p ~/.flagtree/sunrise; cd ~/.flagtree/sunrise
wget https://baai-cp-web.ks3-cn-beijing.ksyuncs.com/trans/sunrise-llvm21-glibc2.39-glibcxx3.4.33-x86_64_v0.4.0.tar.gz
tar zxvf sunrise-llvm21-glibc2.39-glibcxx3.4.33-x86_64_v0.4.0.tar.gz
wget https://baai-cp-web.ks3-cn-beijing.ksyuncs.com/trans/sunriseTritonPlugin-cpython3.10-glibc2.39-glibcxx3.4.33-x86_64_v0.4.0.tar.gz
tar zxvf sunriseTritonPlugin-cpython3.10-glibc2.39-glibcxx3.4.33-x86_64_v0.4.0.tar.gz
```

#### 1.3 Manually download the Triton dependencies

The Triton dependencies are already downloaded and installed in the preinstalled image.
If you do not need to build FlagTree or Triton from source, you do not need to download the Triton dependencies.

```shell
cd ${YOUR_CODE_DIR}/FlagTree
# For Triton 3.4 (x64)
wget https://baai-cp-web.ks3-cn-beijing.ksyuncs.com/trans/build-deps-triton_3.4.x-linux-x64.tar.gz
sh python/scripts/unpack_triton_build_deps.sh ./build-deps-triton_3.4.x-linux-x64.tar.gz
```

After executing the above script, the original ~/.triton directory will be renamed, and a new ~/.triton directory will be created to store the pre-downloaded packages.
Note that the script will prompt for manual confirmation during execution.

### 2. Installation Commands

#### 2.1 Source-free Installation

```shell
# Note: First install PyTorch, then execute the following commands
python3 -m pip uninstall -y triton  # Repeat the cmd until fully uninstalled
RES="--index-url=https://resource.flagos.net/repository/flagos-pypi-hosted/simple"
python3 -m pip install flagtree===0.4.0+sunrise3.4 $RES
```

`flagtree` is already installed in the preinstalled image. You can check it with:

```shell
python3 -m pip show flagtree
```

#### 2.2 Build from Source

```shell
apt update; apt install zlib1g zlib1g-dev libxml2 libxml2-dev
cd ${YOUR_CODE_DIR}/FlagTree
git checkout -b triton_v3.4.x origin/triton_v3.4.x
python3 -m pip install -r python/requirements.txt
export TRITON_BUILD_WITH_CLANG_LLD=1
export TRITON_OFFLINE_BUILD=1
export TRITON_BUILD_PROTON=OFF
export FLAGTREE_BACKEND=sunrise
MAX_JOBS=32 python3 -m pip install . --no-build-isolation -v
```

### 3. Testing and validation

Refer to [Tests of sunrise backend](https://github.com/flagos-ai/FlagTree/blob/triton_v3.4.x/.github/workflows/sunrise-build-and-test.yml)
