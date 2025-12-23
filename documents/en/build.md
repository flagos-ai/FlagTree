[中文](../zh-cn/build.md)

## Build

Each backend is based on different versions of Triton, and therefore resides in different protected branches:

- ([main](https://github.com/flagos-ai/flagtree/tree/main) for Triton 3.1
- [triton\_v3.2.x](https://github.com/flagos-ai/flagtree/tree/triton_v3.2.x) for Triton 3.2.x
- [triton\_v3.3.x](https://github.com/flagos-ai/flagtree/tree/triton_v3.3.x) for Triton 3.3.x
- [triton\_v3.4.x](https://github.com/flagos-ai/flagtree/tree/triton_v3.4.x) for Triton 3.4.x
- [triton\_v3.5.x](https://github.com/flagos-ai/flagtree/tree/triton_v3.5.x) for Triton 3.5.x

All these protected branches are maintained with equal foot.

Complete build commands for each backend:

### Illuvatar

Backend:

- [iluvatar](https://github.com/flagos-ai/flagtree/tree/main/third_party/iluvatar/)

Environment:

- Triton 3.1
- x86\_64

```shell
# Recommended: Use Ubuntu 20.04
mkdir -p ~/.flagtree/iluvatar; cd ~/.flagtree/iluvatar
wget https://baai-cp-web.ks3-cn-beijing.ksyuncs.com/trans/iluvatar-llvm18-x86_64_v0.4.0.tar.gz
tar zxvf iluvatar-llvm18-x86_64_v0.4.0.tar.gz
wget https://baai-cp-web.ks3-cn-beijing.ksyuncs.com/trans/iluvatarTritonPlugin-cpython3.10-glibc2.35-glibcxx3.4.30-cxxabi1.3.13-ubuntu-x86_64_v0.4.0.tar.gz
tar zxvf iluvatarTritonPlugin-cpython3.10-glibc2.35-glibcxx3.4.30-cxxabi1.3.13-ubuntu-x86_64_v0.4.0.tar.gz
cd ${YOUR_CODE_DIR}/flagtree/python
export FLAGTREE_BACKEND=iluvatar
python3 -m pip install . --no-build-isolation -v
```

### Kunlunxin

Backend:

- [xpu](https://github.com/flagos-ai/flagtree/tree/main/third_party/xpu/)

Environment:

- Triton 3.0
- x86\_64

> [!Tip]
> The recommended way to set up is to use the [Docker image](https://su.bcebos.com/klx-sdk-release-public/xpytorch/docker/ubuntu2004_v030/ubuntu_2004_x86_64_v30.tar)
> which is about 22GB.
>
> You can contact kunlunxin-support@baidu.com for support.

```shell
mkdir -p ~/.flagtree/xpu
cd ~/.flagtree/xpu

wget https://baai-cp-web.ks3-cn-beijing.ksyuncs.com/trans/XTDK-llvm19-ubuntu2004_x86_64_v0.3.0.tar.gz
tar zxvf XTDK-llvm19-ubuntu2004_x86_64_v0.3.0.tar.gz

wget https://baai-cp-web.ks3-cn-beijing.ksyuncs.com/trans/xre-Linux-x86_64_v0.3.0.tar.gz
tar zxvf xre-Linux-x86_64_v0.3.0.tar.gz

wget https://baai-cp-web.ks3-cn-beijing.ksyuncs.com/trans/xpu-liblaunch_shared_so-ubuntu-x64_v0.3.0.tar.gz
tar zxvf xpu-liblaunch_shared_so-ubuntu-x64_v0.3.0.tar.gz

cd ${YOUR_CODE_DIR}/flagtree/python
export FLAGTREE_BACKEND=xpu
python3 -m pip install . --no-build-isolation -v
```

### Moore Threads

Backend:

- [mthreads](https://github.com/flagos-ai/flagtree/tree/main/third_party/mthreads/)

Environment:

- Triton 3.1
- x86\_64/aarch64

> [!Tip]
> Recommended: Use the Dockerfile flagtree/dockerfiles/Dockerfile-ubuntu22.04-python3.10-mthreads

Prepare the directory:

```shell
mkdir -p ~/.flagtree/mthreads
cd ~/.flagtree/mthreads
```

For x86\_64 architecture:

```shell
wget https://baai-cp-web.ks3-cn-beijing.ksyuncs.com/trans/mthreads-llvm19-glibc2.35-glibcxx3.4.30-x64_v0.4.0.tar.gz
tar zxvf mthreads-llvm19-glibc2.35-glibcxx3.4.30-x64_v0.4.0.tar.gz
wget https://baai-cp-web.ks3-cn-beijing.ksyuncs.com/trans/mthreadsTritonPlugin-cpython3.10-glibc2.35-glibcxx3.4.30-cxxabi1.3.13-ubuntu-x64_v0.4.1.tar.gz
tar zxvf mthreadsTritonPlugin-cpython3.10-glibc2.35-glibcxx3.4.30-cxxabi1.3.13-ubuntu-x64_v0.4.1.tar.gz
```

For aarch64 architecture:

```shell
wget https://baai-cp-web.ks3-cn-beijing.ksyuncs.com/trans/mthreads-llvm19-glibc2.35-glibcxx3.4.30-aarch64_v0.4.0.tar.gz
tar zxvf mthreads-llvm19-glibc2.35-glibcxx3.4.30-aarch64_v0.4.0.tar.gz
wget https://baai-cp-web.ks3-cn-beijing.ksyuncs.com/trans/mthreadsTritonPlugin-cpython3.10-glibc2.35-glibcxx3.4.30-cxxabi1.3.13-ubuntu-aarch64_v0.4.0.tar.gz
tar zxvf mthreadsTritonPlugin-cpython3.10-glibc2.35-glibcxx3.4.30-cxxabi1.3.13-ubuntu-aarch64_v0.4.0.tar.gz
```

Trigger the installation action:

```shell
cd ${YOUR_CODE_DIR}/flagtree/python
export FLAGTREE_BACKEND=mthreads
python3 -m pip install . --no-build-isolation -v
```

### ARM APIU

Backend:

- [aipu](https://github.com/flagos-ai/flagtree/tree/triton_v3.3.x/third_party/aipu/)

Environment:

- Triton 3.3
- x86\_64/arm64

```shell
# Recommended: Use Ubuntu 22.04
mkdir -p ~/.flagtree/aipu; cd ~/.flagtree/aipu
# x64 in the simulated environment, arm64 on the ARM development board
wget https://baai-cp-web.ks3-cn-beijing.ksyuncs.com/trans/llvm-a66376b0-ubuntu-x64-clang16-lld16_v0.4.0.tar.gz
tar zxvf llvm-a66376b0-ubuntu-x64-clang16-lld16_v0.4.0.tar.gz
cd ${YOUR_CODE_DIR}/flagtree/python
git checkout -b triton_v3.3.x origin/triton_v3.3.x
export FLAGTREE_BACKEND=aipu
python3 -m pip install . --no-build-isolation -v
```

### TsingMicro

Backend:

- [tsingmicro](https://github.com/flagos-ai/flagtree/tree/triton_v3.3.x/third_party/tsingmicro/)

Environment:

- Triton 3.3
- x86\_64
- Ubuntu 24.04 (recommended)

```shell
mkdir -p ~/.flagtree/tsingmicro
cd ~/.flagtree/tsingmicro
wget https://baai-cp-web.ks3-cn-beijing.ksyuncs.com/trans/tsingmicro-llvm21-glibc2.30-glibcxx3.4.28-python3.11-x64_v0.2.0.tar.gz
tar zxvf tsingmicro-llvm21-glibc2.30-glibcxx3.4.28-python3.11-x64_v0.2.0.tar.gz

wget https://baai-cp-web.ks3-cn-beijing.ksyuncs.com/trans/tx8_depends_release_20250814_195126_v0.2.0.tar.gz
tar zxvf tx8_depends_release_20250814_195126_v0.2.0.tar.gz

export TX8_DEPS_ROOT=~/.flagtree/tsingmicro/tx8_deps
cd ${YOUR_CODE_DIR}/flagtree/python
git checkout -b triton_v3.3.x origin/triton_v3.3.x
export FLAGTREE_BACKEND=tsingmicro
python3 -m pip install . --no-build-isolation -v
```

### Ascend

Backend:

- [ascend](https://github.com/flagos-ai/flagtree/blob/triton_v3.2.x/third_party/ascend/)

Environment:

- Triton 3.2
- aarch64

> [!Tip]
> The recommended way to build and install is to use the Dockerfile
> `flagtree/dockerfiles/Dockerfile-ubuntu22.04-python3.11-ascend`.
> After registering an account at https://www.hiascend.com/developer/download/community/result?module=cann,
> you can download the `cann-toolkit` and `cann-kernels` for the corresponding platform.

Install the toolkits:

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

Build the software:

```shell
mkdir -p ~/.flagtree/ascend
cd ~/.flagtree/ascend
wget https://baai-cp-web.ks3-cn-beijing.ksyuncs.com/trans/llvm-a66376b0-ubuntu-aarch64-python311-compat_v0.3.0.tar.gz
tar zxvf llvm-a66376b0-ubuntu-aarch64-python311-compat_v0.3.0.tar.gz

cd ${YOUR_CODE_DIR}/flagtree/python
git checkout -b triton_v3.2.x origin/triton_v3.2.x
export FLAGTREE_BACKEND=ascend
python3 -m pip install . --no-build-isolation -v
```

### Hygon HCU

Backend:

- [hcu](https://github.com/flagos-ai/flagtree/tree/main/third_party/hcu/)

Environment:

- Triton 3.0
- x86\_64

> [!Tip]
> The recommended way to install is to use the Dockerfile
> `flagtree/dockerfiles/Dockerfile-ubuntu22.04-python3.10-hcu`.

```shell
mkdir -p ~/.flagtree/hcu; cd ~/.flagtree/hcu
wget https://baai-cp-web.ks3-cn-beijing.ksyuncs.com/trans/hcu-llvm20-df0864e-glibc2.35-glibcxx3.4.30-ubuntu-x86_64_v0.3.0.tar.gz
tar zxvf hcu-llvm20-df0864e-glibc2.35-glibcxx3.4.30-ubuntu-x86_64_v0.3.0.tar.gz

cd ${YOUR_CODE_DIR}/flagtree/python
export FLAGTREE_BACKEND=hcu
python3 -m pip install . --no-build-isolation -v
```

### Enflame

Backend:

- [enflame](https://github.com/flagos-ai/flagtree/tree/triton_v3.3.x/third_party/enflame/)

Environment:

- Triton 3.3
- x86\_64

> [!Tip]
> The recommended way for installation is to use the Docker image (2.4GB) at
> https://baai-cp-web.ks3-cn-beijing.ksyuncs.com/trans/enflame-flagtree-0.3.1.tar.gz


```shell
mkdir -p ~/.flagtree/enflame; cd ~/.flagtree/enflame
wget https://baai-cp-web.ks3-cn-beijing.ksyuncs.com/trans/enflame-llvm21-d752c5b-gcc9-x64_v0.3.0.tar.gz
tar zxvf enflame-llvm21-d752c5b-gcc9-x64_v0.3.0.tar.gz

cd ${YOUR_CODE_DIR}/flagtree/python
export FLAGTREE_BACKEND=enflame
python3 -m pip install . --no-build-isolation -v
```

### NVIDIA

Backend:

- [nvidia](https://github.com/flagos-ai/flagtree/tree/triton_v3.3.x/third_party/nvidia/)

To build with default backends NVIDIA, AMD, triton_shared CPU:

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

export LLVM_INCLUDE_DIRS=$LLVM_SYSPATH/include
export LLVM_LIBRARY_DIR=$LLVM_SYSPATH/lib
cd ${YOUR_CODE_DIR}/flagtree
cd python  # For Triton 3.1, 3.2, 3.3, you need to enter the python directory to build

git checkout main                                       # For Triton 3.1
git checkout -b triton_v3.2.x  origin/triton_v3.2.x     # For Triton 3.2
git checkout -b triton_v3.3.x  origin/triton_v3.3.x     # For Triton 3.3
git checkout -b triton_v3.4.x  origin/triton_v3.4.x     # For Triton 3.4
git checkout -b triton_v3.5.x  origin/triton_v3.5.x     # For Triton 3.5
unset FLAGTREE_BACKEND
python3 -m pip install . --no-build-isolation -v
```

If you need to build other backends afterward, you should clear LLVM-related environment
variables:

```shell
unset LLVM_SYSPATH LLVM_INCLUDE_DIRS LLVM_LIBRARY_DIR
```

## Offline Build

### Pre-downloading Dependency Packages

The above introduced how dependencies can be manually downloaded for various FlagTree backends
during build time to avoid network environment limitations.
Since Triton builds originally come with some dependency packages, we provide pre-downloaded packages
that can be manually installed in your environment to prevent getting stuck
at the automatic download stage during the build process.

```shell
cd ${YOUR_CODE_DIR}/flagtree/python
sh README_offline_build.sh x86_64  # View readme

# For Triton 3.1 (x64)
wget https://baai-cp-web.ks3-cn-beijing.ksyuncs.com/trans/offline-build-pack-triton-3.1.x-linux-x64.zip
sh scripts/offline_build_unpack.sh ./offline-build-pack-triton-3.1.x-linux-x64.zip ~/.triton

# For Triton 3.2 (x64)
wget https://baai-cp-web.ks3-cn-beijing.ksyuncs.com/trans/offline-build-pack-triton-3.2.x-linux-x64.zip
sh scripts/offline_build_unpack.sh ./offline-build-pack-triton-3.2.x-linux-x64.zip ~/.triton

# For Triton 3.2 (aarch64)
wget https://baai-cp-web.ks3-cn-beijing.ksyuncs.com/trans/offline-build-pack-triton-3.2.x-linux-aarch64.zip
sh scripts/offline_build_unpack.sh ./offline-build-pack-triton-3.2.x-linux-aarch64.zip ~/.triton

# For Triton 3.3 (x64)
wget https://baai-cp-web.ks3-cn-beijing.ksyuncs.com/trans/offline-build-pack-triton-3.3.x-linux-x64.zip
sh scripts/offline_build_unpack.sh ./offline-build-pack-triton-3.3.x-linux-x64.zip ~/.triton
```

After executing the above script, the original `~/.triton` directory will be renamed,
and a new `~/.triton` directory will be created to store the pre-downloaded packages.
