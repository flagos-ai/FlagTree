<div align="right"><a href="/documents/build_cn.md">中文版</a></div>

## Install from source

### Tips for building

Automatic dependency library downloads may be limited by network conditions. You can manually download to the cache directory ~/.flagtree (modifiable via the FLAGTREE_CACHE_DIR environment variable). No need to manually set LLVM environment variables such as LLVM_BUILD_DIR. <br>
Complete build commands for each backend: <br>

#### ILUVATAR（天数智芯）[iluvatar](https://github.com/flagos-ai/FlagTree/tree/main/third_party/iluvatar/)

- Based on Triton 3.1, x64

##### 1. Build and run environment

- Recommended: Use Ubuntu 20.04

##### 2. Manually download the FlagTree dependencies

```shell
mkdir -p ~/.flagtree/iluvatar; cd ~/.flagtree/iluvatar
wget https://baai-cp-web.ks3-cn-beijing.ksyuncs.com/trans/iluvatar-llvm18-x86_64_v0.4.0.tar.gz
tar zxvf iluvatar-llvm18-x86_64_v0.4.0.tar.gz
wget https://baai-cp-web.ks3-cn-beijing.ksyuncs.com/trans/iluvatarTritonPlugin-cpython3.10-glibc2.35-glibcxx3.4.30-cxxabi1.3.13-ubuntu-x86_64_v0.4.0.tar.gz
tar zxvf iluvatarTritonPlugin-cpython3.10-glibc2.35-glibcxx3.4.30-cxxabi1.3.13-ubuntu-x86_64_v0.4.0.tar.gz
```

##### 3. Command to build from source

```shell
cd ${YOUR_CODE_DIR}/FlagTree/python
export FLAGTREE_BACKEND=iluvatar
python3 -m pip install . --no-build-isolation -v
```

#### KLX [xpu](https://github.com/flagos-ai/FlagTree/tree/main/third_party/xpu/)

- Based on Triton 3.0, x64

##### 1. Build and run environment

- Recommended: Use the Docker image (22GB) [ubuntu_2004_x86_64_v30.tar](https://su.bcebos.com/klx-sdk-release-public/xpytorch/docker/ubuntu2004_v030/ubuntu_2004_x86_64_v30.tar)
- Contact kunlunxin-support@baidu.com for support

##### 2. Manually download the FlagTree dependencies

```shell
mkdir -p ~/.flagtree/xpu; cd ~/.flagtree/xpu
wget https://baai-cp-web.ks3-cn-beijing.ksyuncs.com/trans/XTDK-llvm19-ubuntu2004_x86_64_v0.3.0.tar.gz
tar zxvf XTDK-llvm19-ubuntu2004_x86_64_v0.3.0.tar.gz
wget https://baai-cp-web.ks3-cn-beijing.ksyuncs.com/trans/xre-Linux-x86_64_v0.3.0.tar.gz
tar zxvf xre-Linux-x86_64_v0.3.0.tar.gz
wget https://baai-cp-web.ks3-cn-beijing.ksyuncs.com/trans/xpu-liblaunch_shared_so-ubuntu-x64_v0.3.1.tar.gz
tar zxvf xpu-liblaunch_shared_so-ubuntu-x64_v0.3.1.tar.gz
```

##### 3. Command to build from source

```shell
cd ${YOUR_CODE_DIR}/FlagTree/python
export FLAGTREE_BACKEND=xpu
python3 -m pip install . --no-build-isolation -v
```

#### Moore Threads（摩尔线程）[mthreads](https://github.com/flagos-ai/FlagTree/tree/main/third_party/mthreads/)

- Based on Triton 3.1, x64/aarch64

##### 1. Build and run environment

- Recommended: Use [Dockerfile-ubuntu22.04-python3.10-mthreads](/dockerfiles/Dockerfile-ubuntu22.04-python3.10-mthreads)

##### 2. Manually download the FlagTree dependencies

```shell
mkdir -p ~/.flagtree/mthreads; cd ~/.flagtree/mthreads
# x64
wget https://baai-cp-web.ks3-cn-beijing.ksyuncs.com/trans/mthreads-llvm19-glibc2.35-glibcxx3.4.30-x64_v0.4.0.tar.gz
tar zxvf mthreads-llvm19-glibc2.35-glibcxx3.4.30-x64_v0.4.0.tar.gz
wget https://baai-cp-web.ks3-cn-beijing.ksyuncs.com/trans/mthreadsTritonPlugin-cpython3.10-glibc2.35-glibcxx3.4.30-cxxabi1.3.13-ubuntu-x64_v0.4.1.tar.gz
tar zxvf mthreadsTritonPlugin-cpython3.10-glibc2.35-glibcxx3.4.30-cxxabi1.3.13-ubuntu-x64_v0.4.1.tar.gz
# aarch64
wget https://baai-cp-web.ks3-cn-beijing.ksyuncs.com/trans/mthreads-llvm19-glibc2.35-glibcxx3.4.30-aarch64_v0.4.0.tar.gz
tar zxvf mthreads-llvm19-glibc2.35-glibcxx3.4.30-aarch64_v0.4.0.tar.gz
wget https://baai-cp-web.ks3-cn-beijing.ksyuncs.com/trans/mthreadsTritonPlugin-cpython3.10-glibc2.35-glibcxx3.4.30-cxxabi1.3.13-ubuntu-aarch64_v0.4.0.tar.gz
tar zxvf mthreadsTritonPlugin-cpython3.10-glibc2.35-glibcxx3.4.30-cxxabi1.3.13-ubuntu-aarch64_v0.4.0.tar.gz
```

##### 3. Command to build from source

```shell
cd ${YOUR_CODE_DIR}/FlagTree/python
export FLAGTREE_BACKEND=mthreads
python3 -m pip install . --no-build-isolation -v
```

#### ARM China（安谋科技）[aipu](https://github.com/flagos-ai/FlagTree/tree/triton_v3.3.x/third_party/aipu/)

- Based on Triton 3.3, x64/arm64

##### 1. Build and run environment

- Recommended: Use Ubuntu 22.04

##### 2. Manually download the FlagTree dependencies

- llvm x64 in the simulated environment, llvm arm64 on the ARM development board

```shell
mkdir -p ~/.flagtree/aipu; cd ~/.flagtree/aipu
wget https://baai-cp-web.ks3-cn-beijing.ksyuncs.com/trans/llvm-a66376b0-ubuntu-x64-clang16-lld16_v0.4.0.tar.gz
tar zxvf llvm-a66376b0-ubuntu-x64-clang16-lld16_v0.4.0.tar.gz
```

##### 3. Command to build from source

```shell
cd ${YOUR_CODE_DIR}/FlagTree/python
git checkout -b triton_v3.3.x origin/triton_v3.3.x
export FLAGTREE_BACKEND=aipu
python3 -m pip install . --no-build-isolation -v
```

#### Tsingmicro（清微智能）[tsingmicro](https://github.com/flagos-ai/FlagTree/tree/triton_v3.3.x/third_party/tsingmicro/)

- Based on Triton 3.3, x64

##### 1. Build and run environment

- Recommended: Use Ubuntu 20.04


##### 2. Manually download the FlagTree dependencies

```shell
mkdir -p ~/.flagtree/tsingmicro; cd ~/.flagtree/tsingmicro
wget https://baai-cp-web.ks3-cn-beijing.ksyuncs.com/trans/tsingmicro-llvm21-glibc2.30-glibcxx3.4.28-python3.11-x64_v0.2.0.tar.gz
tar zxvf tsingmicro-llvm21-glibc2.30-glibcxx3.4.28-python3.11-x64_v0.2.0.tar.gz
wget https://baai-cp-web.ks3-cn-beijing.ksyuncs.com/trans/tx8_depends_release_20250814_195126_v0.2.0.tar.gz
tar zxvf tx8_depends_release_20250814_195126_v0.2.0.tar.gz
```

##### 3. Command to build from source

```shell
cd ${YOUR_CODE_DIR}/FlagTree/python
git checkout -b triton_v3.3.x origin/triton_v3.3.x
export TX8_DEPS_ROOT=~/.flagtree/tsingmicro/tx8_deps
export FLAGTREE_BACKEND=tsingmicro
python3 -m pip install . --no-build-isolation -v
```

#### Huawei Ascend（华为昇腾）[ascend](https://github.com/flagos-ai/FlagTree/blob/triton_v3.2.x/third_party/ascend)

- Based on Triton 3.2, aarch64

##### 1. Build and run environment

- Recommended: Use [Dockerfile-ubuntu22.04-python3.11-ascend](/dockerfiles/Dockerfile-ubuntu22.04-python3.11-ascend)
- After registering an account at https://www.hiascend.com/developer/download/community/result?module=cann, download the cann-toolkit and cann-kernels for the corresponding platform.

```shell
# cann-toolkit
chmod +x Ascend-cann-toolkit_8.3.RC1.alpha001_linux-aarch64.run
./Ascend-cann-toolkit_8.3.RC1.alpha001_linux-aarch64.run --install
# cann-kernels for 910B (A2)
chmod +x Ascend-cann-kernels-910b_8.3.RC1.alpha001_linux-aarch64.run
./Ascend-cann-kernels-910b_8.3.RC1.alpha001_linux-aarch64.run --install
# cann-kernels for 910C (A3)
chmod +x Atlas-A3-cann-kernels_8.3.RC1.alpha001_linux-aarch64.run
./Atlas-A3-cann-kernels_8.3.RC1.alpha001_linux-aarch64.run --install
```

##### 2. Manually download the FlagTree dependencies

```shell
mkdir -p ~/.flagtree/ascend; cd ~/.flagtree/ascend
wget https://baai-cp-web.ks3-cn-beijing.ksyuncs.com/trans/llvm-a66376b0-ubuntu-aarch64-python311-compat_v0.3.0.tar.gz
tar zxvf llvm-a66376b0-ubuntu-aarch64-python311-compat_v0.3.0.tar.gz
```

##### 3. Command to build from source

```shell
cd ${YOUR_CODE_DIR}/FlagTree/python
git checkout -b triton_v3.2.x origin/triton_v3.2.x
export FLAGTREE_BACKEND=ascend
python3 -m pip install . --no-build-isolation -v
```

#### HYGON（海光信息）[hcu](https://github.com/flagos-ai/FlagTree/tree/main/third_party/hcu/)

- Based on Triton 3.0, x64

##### 1. Build and run environment

- Recommended: Use [Dockerfile-ubuntu22.04-python3.10-hcu](/dockerfiles/Dockerfile-ubuntu22.04-python3.10-hcu)

##### 2. Manually download the FlagTree dependencies

```shell
mkdir -p ~/.flagtree/hcu; cd ~/.flagtree/hcu
wget https://baai-cp-web.ks3-cn-beijing.ksyuncs.com/trans/hcu-llvm20-df0864e-glibc2.35-glibcxx3.4.30-ubuntu-x86_64_v0.3.0.tar.gz
tar zxvf hcu-llvm20-df0864e-glibc2.35-glibcxx3.4.30-ubuntu-x86_64_v0.3.0.tar.gz
```

##### 3. Command to build from source

```shell
cd ${YOUR_CODE_DIR}/FlagTree/python
export FLAGTREE_BACKEND=hcu
python3 -m pip install . --no-build-isolation -v
```

#### Enflame（燧原）[enflame](https://github.com/flagos-ai/FlagTree/tree/triton_v3.3.x/third_party/enflame/)

- Based on Triton 3.3, x64

##### 1. Build and run environment

- Recommended: Use the Docker image (2.4GB) https://baai-cp-web.ks3-cn-beijing.ksyuncs.com/trans/enflame-flagtree-0.3.1.tar.gz

##### 2. Manually download the FlagTree dependencies

```shell
mkdir -p ~/.flagtree/enflame; cd ~/.flagtree/enflame
wget https://baai-cp-web.ks3-cn-beijing.ksyuncs.com/trans/enflame-llvm21-d752c5b-gcc9-x64_v0.3.0.tar.gz
tar zxvf enflame-llvm21-d752c5b-gcc9-x64_v0.3.0.tar.gz
```

##### 3. Command to build from source

```shell
cd ${YOUR_CODE_DIR}/FlagTree/python
export FLAGTREE_BACKEND=enflame
python3 -m pip install . --no-build-isolation -v
```

#### sunrise（曦望芯科）[sunrise](https://github.com/flagos-ai/FlagTree/tree/triton_v3.4.x/third_party/sunrise/)

- Based on Triton 3.4, x64

##### 1. Build and run environment

- Recommended: Use Ubuntu 22.04

##### 2. Manually download the FlagTree dependencies

```shell
mkdir -p ~/.flagtree/sunrise; cd ~/.flagtree/sunrise
wget https://baai-cp-web.ks3-cn-beijing.ksyuncs.com/trans/sunrise-llvm21-glibc2.39-glibcxx3.4.33-x86_64_v0.4.0.tar.gz
tar zxvf sunrise-llvm21-glibc2.39-glibcxx3.4.33-x86_64_v0.4.0.tar.gz
wget https://baai-cp-web.ks3-cn-beijing.ksyuncs.com/trans/sunriseTritonPlugin-cpython3.10-glibc2.39-glibcxx3.4.33-x86_64_v0.4.0.tar.gz
tar zxvf sunriseTritonPlugin-cpython3.10-glibc2.39-glibcxx3.4.33-x86_64_v0.4.0.tar.gz
```

##### 3. Command to build from source

```shell
cd ${YOUR_CODE_DIR}/FlagTree
export TRITON_BUILD_WITH_CLANG_LLD=1
export TRITON_OFFLINE_BUILD=1
export TRITON_BUILD_PROTON=OFF
export FLAGTREE_BACKEND=sunrise
python3 -m pip install . --no-build-isolation -v
```

#### NVIDIA & AMD [nvidia](/third_party/nvidia/) & [amd](/third_party/amd/)

- Based on Triton 3.1/3.2/3.3/3.4/3.5, x64/arm64

##### 1. Build and run environment

- Recommended: Use Ubuntu 22.04

##### 2. Manually download the LLVM

```shell
cd ${YOUR_LLVM_DOWNLOAD_DIR}
# For Triton 3.1
wget https://oaitriton.blob.core.windows.net/public/llvm-builds/llvm-10dc3a8e-ubuntu-x64.tar.gz
tar zxvf llvm-10dc3a8e-ubuntu-x64.tar.gz
export LLVM_SYSPATH=${YOUR_LLVM_DOWNLOAD_DIR}/llvm-10dc3a8e-ubuntu-x64
# For Triton 3.2
wget https://oaitriton.blob.core.windows.net/public/llvm-builds/llvm-86b69c31-ubuntu-x64.tar.gz
tar zxvf llvm-86b69c31-ubuntu-x64.tar.gz
export LLVM_SYSPATH=${YOUR_LLVM_DOWNLOAD_DIR}/llvm-86b69c31-ubuntu-x64
# For Triton 3.3
wget https://oaitriton.blob.core.windows.net/public/llvm-builds/llvm-a66376b0-ubuntu-x64.tar.gz
tar zxvf llvm-a66376b0-ubuntu-x64.tar.gz
export LLVM_SYSPATH=${YOUR_LLVM_DOWNLOAD_DIR}/llvm-a66376b0-ubuntu-x64
# For Triton 3.4
wget https://oaitriton.blob.core.windows.net/public/llvm-builds/llvm-8957e64a-ubuntu-x64.tar.gz
tar zxvf llvm-8957e64a-ubuntu-x64.tar.gz
export LLVM_SYSPATH=${YOUR_LLVM_DOWNLOAD_DIR}/llvm-8957e64a-ubuntu-x64
# For Triton 3.5
wget https://oaitriton.blob.core.windows.net/public/llvm-builds/llvm-7d5de303-ubuntu-x64.tar.gz
tar zxvf llvm-7d5de303-ubuntu-x64.tar.gz
export LLVM_SYSPATH=${YOUR_LLVM_DOWNLOAD_DIR}/llvm-7d5de303-ubuntu-x64
#
export LLVM_INCLUDE_DIRS=$LLVM_SYSPATH/include
export LLVM_LIBRARY_DIR=$LLVM_SYSPATH/lib
```

##### 3. Command to build from source

```shell
cd ${YOUR_CODE_DIR}/FlagTree
cd python  # For Triton 3.1, 3.2, 3.3, you need to enter the python directory to build
git checkout main                                   # For Triton 3.1
git checkout -b triton_v3.2.x origin/triton_v3.2.x  # For Triton 3.2
git checkout -b triton_v3.3.x origin/triton_v3.3.x  # For Triton 3.3
git checkout -b triton_v3.4.x origin/triton_v3.4.x  # For Triton 3.4
git checkout -b triton_v3.5.x origin/triton_v3.5.x  # For Triton 3.5
unset FLAGTREE_BACKEND
python3 -m pip install . --no-build-isolation -v
# If you need to build other backends afterward, you should clear LLVM-related environment variables
unset LLVM_SYSPATH LLVM_INCLUDE_DIRS LLVM_LIBRARY_DIR
```

### Offline build support

The above introduced how dependencies can be manually downloaded for various FlagTree backends during build time to avoid network environment limitations. Since Triton builds originally come with some dependency packages, we provide pre-downloaded packages that can be manually installed in your environment to prevent getting stuck at the automatic download stage during the build process.

```shell
cd ${YOUR_CODE_DIR}/FlagTree/python
sh README_offline_build.sh x86_64  # View readme
# For Triton 3.1 (x64)
wget https://baai-cp-web.ks3-cn-beijing.ksyuncs.com/trans/offline-build-pack-triton-3.1.x-linux-x64.zip
sh scripts/offline_build_unpack.sh \
    ./offline-build-pack-triton-3.1.x-linux-x64.zip ~/.triton
# For Triton 3.2 (x64)
wget https://baai-cp-web.ks3-cn-beijing.ksyuncs.com/trans/offline-build-pack-triton-3.2.x-linux-x64.zip
sh scripts/offline_build_unpack.sh \
    ./offline-build-pack-triton-3.2.x-linux-x64.zip ~/.triton
# For Triton 3.2 (aarch64)
wget https://baai-cp-web.ks3-cn-beijing.ksyuncs.com/trans/offline-build-pack-triton-3.2.x-linux-aarch64.zip
sh scripts/offline_build_unpack.sh \
    ./offline-build-pack-triton-3.2.x-linux-aarch64.zip ~/.triton
# For Triton 3.3 (x64)
wget https://baai-cp-web.ks3-cn-beijing.ksyuncs.com/trans/offline-build-pack-triton-3.3.x-linux-x64.zip
sh scripts/offline_build_unpack.sh \
    ./offline-build-pack-triton-3.3.x-linux-x64.zip ~/.triton
```

After executing the above script, the original ~/.triton directory will be renamed, and a new ~/.triton directory will be created to store the pre-downloaded packages.

### Q&A

#### Q: After installation, running the program reports: version GLIBC or GLIBCXX not found

A: Check which GLIBC / GLIBCXX versions are supported by libc.so.6 and libstdc++.so.6.0.30 in your environment:

```shell
strings /lib/x86_64-linux-gnu/libc.so.6 |grep GLIBC
strings /usr/lib/x86_64-linux-gnu/libstdc++.so.6.0.30 | grep GLIBCXX
```

If the required GLIBC / GLIBCXX version is supported, you can also try:

```shell
export LD_PRELOAD="/lib/x86_64-linux-gnu/libc.so.6"  # If GLIBC cannot be found
export LD_PRELOAD="/usr/lib/x86_64-linux-gnu/libstdc++.so.6.0.30"  # If GLIBCXX cannot be found
export LD_PRELOAD="/lib/x86_64-linux-gnu/libc.so.6 \
    /usr/lib/x86_64-linux-gnu/libstdc++.so.6.0.30"  # If neither GLIBC nor GLIBCXX can be found
```
