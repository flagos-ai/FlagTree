<div align="right"><a href="/README_cn.md">中文版</a></div>

## <img width="30" height="30" alt="FlagTree-GitHub" src="https://github.com/user-attachments/assets/d8d24c81-6f46-4adc-94e2-b89b03afcb43" /> FlagTree

FlagTree is an open source, unified compiler for multiple AI chips project dedicated to developing a diverse ecosystem of AI chip compilers and related tooling platforms, thereby fostering and strengthening the upstream and downstream Triton ecosystem. Currently in its initial phase, the project aims to maintain compatibility with existing adaptation solutions while unifying the codebase to rapidly implement single-repository multi-backend support. For upstream model users, it provides unified compilation capabilities across multiple backends; for downstream chip manufacturers, it offers examples of Triton ecosystem integration. <br>
Each backend is based on different versions of triton, and therefore resides in different protected branches ([main](https://github.com/flagos-ai/flagtree/tree/main) for triton 3.1, [triton_v3.2.x](https://github.com/flagos-ai/flagtree/tree/triton_v3.2.x), [triton_v3.3.x](https://github.com/flagos-ai/flagtree/tree/triton_v3.3.x), [triton_v3.4.x](https://github.com/flagos-ai/flagtree/tree/triton_v3.4.x), [triton_v3.5.x](https://github.com/flagos-ai/flagtree/tree/triton_v3.5.x)). All these protected branches have equal status. <br>

## Latest News
* 2025/12/24 Support pull and install [whl](/README.md#non-source-installation).
* 2025/12/08 Added [enflame](https://github.com/FlagTree/flagtree/tree/triton_v3.3.x/third_party/enflame/) backend integration (based on Triton 3.3), and added CI/CD.
* 2025/11/26 Add FlagTree_Backend_Specialization Unified Design Document [FlagTree_Backend_Specialization](reports/decoupling/).
* 2025/10/28 Provides offline build support (pre-downloaded dependency packages), improving the build experience when network environment is limited. See usage instructions below.
* 2025/09/30 Support flagtree_hints for shared memory on GPGPU.
* 2025/09/29 SDK storage migrated to ksyuncs, improving download stability.
* 2025/09/25 Support flagtree_hints for ascend backend compilation capability.
* 2025/09/16 Added [hcu](https://github.com/FlagTree/flagtree/tree/main/third_party/hcu/) backend integration (based on Triton 3.0), and added CI/CD.
* 2025/09/09 Forked and modified [llvm-project](https://github.com/FlagTree/llvm-project) to support [FLIR](https://github.com/flagos-ai/flir).
* 2025/09/01 Added adaptation for Paddle framework, and added CI/CD.
* 2025/08/16 Added adaptation for Beijing Super Cloud Computing Center.
* 2025/08/04 Added T*** backend integration (based on Triton 3.1).
* 2025/08/01 [FLIR](https://github.com/flagos-ai/flir) supports flagtree_hints for shared memory loading.
* 2025/07/30 Updated [cambricon](https://github.com/FlagTree/flagtree/tree/triton_v3.2.x/third_party/cambricon/) backend (based on Triton 3.2).
* 2025/07/25 Inspur team added adaptation for OpenAnolis OS.
* 2025/07/09 [FLIR](https://github.com/flagos-ai/flir) supports flagtree_hints for Async DMA.
* 2025/07/08 Added UnifiedHardware manager for multi-backend compilation.
* 2025/07/02 [FlagGems](https://github.com/flagos-ai/FlagGems) LibTuner adapted to triton_v3.3.x version.
* 2025/07/02 Added S*** backend integration (based on Triton 3.3).
* 2025/06/20 [FLIR](https://github.com/flagos-ai/flir) began supporting MLIR extension functionality.
* 2025/06/06 Added [tsingmicro](https://github.com/FlagTree/flagtree/tree/triton_v3.3.x/third_party/tsingmicro/) backend integration (based on Triton 3.3), and added CI/CD.
* 2025/06/04 Added [ascend](https://github.com/FlagTree/flagtree/blob/triton_v3.2.x/third_party/ascend) backend integration (based on Triton 3.2), and added CI/CD.
* 2025/06/03 Added [metax](https://github.com/FlagTree/flagtree/tree/main/third_party/metax/) backend integration (based on Triton 3.1), and added CI/CD.
* 2025/05/22 FlagGems LibEntry adapted to triton_v3.3.x version.
* 2025/05/21 [FLIR](https://github.com/flagos-ai/flir) began supporting conversion functionality to middle layer.
* 2025/04/09 Added arm [aipu](https://github.com/FlagTree/flagtree/tree/triton_v3.3.x/third_party/aipu/) backend integration (based on Triton 3.3), provided a torch standard extension [example](https://github.com/flagos-ai/flagtree/blob/triton_v3.3.x/third_party/aipu/backend/aipu_torch_dev.cpp), and added CI/CD.
* 2025/03/26 Integrated security compliance scanning.
* 2025/03/19 Added klx [xpu](https://github.com/FlagTree/flagtree/tree/main/third_party/xpu/) backend integration (based on Triton 3.0), and added CI/CD.
* 2025/03/19 Added [mthreads](https://github.com/FlagTree/flagtree/tree/main/third_party/mthreads/) backend integration (based on Triton 3.1), and added CI/CD.
* 2025/03/12 Added [iluvatar](https://github.com/FlagTree/flagtree/tree/main/third_party/iluvatar/) backend integration (based on Triton 3.1), and added CI/CD.

## Install from source
Installation dependencies (ensure you use the correct python3.x version):
```shell
apt install zlib1g zlib1g-dev libxml2 libxml2-dev  # ubuntu
cd python; python3 -m pip install -r requirements.txt
```

Building and Installation (Recommended for environments with good network connectivity):
```shell
cd python
export FLAGTREE_BACKEND=backendxxx
python3 -m pip install . --no-build-isolation -v
cd; python3 -c 'import triton; print(triton.__path__)'
```

### Tips for building

Automatic dependency library downloads may be limited by network conditions. You can manually download to the cache directory ~/.flagtree (modifiable via the FLAGTREE_CACHE_DIR environment variable). No need to manually set LLVM environment variables such as LLVM_BUILD_DIR.
Complete build commands for each backend:

[iluvatar](https://github.com/FlagTree/flagtree/tree/main/third_party/iluvatar/) <br>
Based on Triton 3.1, x64
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
klx [xpu](https://github.com/FlagTree/flagtree/tree/main/third_party/xpu/) <br>
Based on Triton 3.0, x64
```shell
# Recommended: Use the Docker image (22GB) https://su.bcebos.com/klx-sdk-release-public/xpytorch/docker/ubuntu2004_v030/ubuntu_2004_x86_64_v30.tar
# Contact kunlunxin-support@baidu.com for support
mkdir -p ~/.flagtree/xpu; cd ~/.flagtree/xpu
wget https://baai-cp-web.ks3-cn-beijing.ksyuncs.com/trans/XTDK-llvm19-ubuntu2004_x86_64_v0.3.0.tar.gz
tar zxvf XTDK-llvm19-ubuntu2004_x86_64_v0.3.0.tar.gz
wget https://baai-cp-web.ks3-cn-beijing.ksyuncs.com/trans/xre-Linux-x86_64_v0.3.0.tar.gz
tar zxvf xre-Linux-x86_64_v0.3.0.tar.gz
cd ${YOUR_CODE_DIR}/flagtree/python
export FLAGTREE_BACKEND=xpu
python3 -m pip install . --no-build-isolation -v
```
[mthreads](https://github.com/FlagTree/flagtree/tree/main/third_party/mthreads/) <br>
Based on Triton 3.1, x64/aarch64
```shell
# Recommended: Use the Dockerfile flagtree/dockerfiles/Dockerfile-ubuntu22.04-python3.10-mthreads
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
#
cd ${YOUR_CODE_DIR}/flagtree/python
export FLAGTREE_BACKEND=mthreads
python3 -m pip install . --no-build-isolation -v
```
arm [aipu](https://github.com/FlagTree/flagtree/tree/triton_v3.3.x/third_party/aipu/) <br>
Based on Triton 3.3, x64/arm64
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
[tsingmicro](https://github.com/FlagTree/flagtree/tree/triton_v3.3.x/third_party/tsingmicro/) <br>
Based on Triton 3.3, x64
```shell
# Recommended: Use Ubuntu 20.04
mkdir -p ~/.flagtree/tsingmicro; cd ~/.flagtree/tsingmicro
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
[ascend](https://github.com/FlagTree/flagtree/blob/triton_v3.2.x/third_party/ascend/) <br>
Based on Triton 3.2, aarch64
```shell
# Recommended: Use the Dockerfile flagtree/dockerfiles/Dockerfile-ubuntu22.04-python3.11-ascend
# After registering an account at https://www.hiascend.com/developer/download/community/result?module=cann,
# download the cann-toolkit and cann-kernels for the corresponding platform.
# cann-toolkit
chmod +x Ascend-cann-toolkit_8.3.RC1.alpha001_linux-aarch64.run
./Ascend-cann-toolkit_8.3.RC1.alpha001_linux-aarch64.run --install
# cann-kernels for 910B (A2)
chmod +x Ascend-cann-kernels-910b_8.3.RC1.alpha001_linux-aarch64.run
./Ascend-cann-kernels-910b_8.3.RC1.alpha001_linux-aarch64.run --install
# cann-kernels for 910C (A3)
chmod +x Atlas-A3-cann-kernels_8.3.RC1.alpha001_linux-aarch64.run
./Atlas-A3-cann-kernels_8.3.RC1.alpha001_linux-aarch64.run --install
# build
mkdir -p ~/.flagtree/ascend; cd ~/.flagtree/ascend
wget https://baai-cp-web.ks3-cn-beijing.ksyuncs.com/trans/llvm-a66376b0-ubuntu-aarch64-python311-compat_v0.3.0.tar.gz
tar zxvf llvm-a66376b0-ubuntu-aarch64-python311-compat_v0.3.0.tar.gz
cd ${YOUR_CODE_DIR}/flagtree/python
git checkout -b triton_v3.2.x origin/triton_v3.2.x
export FLAGTREE_BACKEND=ascend
python3 -m pip install . --no-build-isolation -v
```
[hcu](https://github.com/FlagTree/flagtree/tree/main/third_party/hcu/) <br>
Based on Triton 3.0, x64
```shell
# Recommended: Use the Dockerfile flagtree/dockerfiles/Dockerfile-ubuntu22.04-python3.10-hcu
mkdir -p ~/.flagtree/hcu; cd ~/.flagtree/hcu
wget https://baai-cp-web.ks3-cn-beijing.ksyuncs.com/trans/hcu-llvm20-df0864e-glibc2.35-glibcxx3.4.30-ubuntu-x86_64_v0.3.0.tar.gz
tar zxvf hcu-llvm20-df0864e-glibc2.35-glibcxx3.4.30-ubuntu-x86_64_v0.3.0.tar.gz
cd ${YOUR_CODE_DIR}/flagtree/python
export FLAGTREE_BACKEND=hcu
python3 -m pip install . --no-build-isolation -v
```
[enflame](https://github.com/FlagTree/flagtree/tree/triton_v3.3.x/third_party/enflame/) <br>
Based on Triton 3.3, x64
```shell
# Recommended: Use the Docker image (2.4GB) https://baai-cp-web.ks3-cn-beijing.ksyuncs.com/trans/enflame-flagtree-0.3.1.tar.gz
mkdir -p ~/.flagtree/enflame; cd ~/.flagtree/enflame
wget https://baai-cp-web.ks3-cn-beijing.ksyuncs.com/trans/enflame-llvm21-d752c5b-gcc9-x64_v0.3.0.tar.gz
tar zxvf enflame-llvm21-d752c5b-gcc9-x64_v0.3.0.tar.gz
cd ${YOUR_CODE_DIR}/flagtree/python
export FLAGTREE_BACKEND=enflame
python3 -m pip install . --no-build-isolation -v
```

[nvidia](/third_party/nvidia/) <br>
To build with default backends nvidia, amd, triton_shared cpu:
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
cd ${YOUR_CODE_DIR}/flagtree
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

### Offline Build Support: Pre-downloading Dependency Packages
The above introduced how dependencies can be manually downloaded for various FlagTree backends during build time to avoid network environment limitations. Since Triton builds originally come with some dependency packages, we provide pre-downloaded packages that can be manually installed in your environment to prevent getting stuck at the automatic download stage during the build process.
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
# After executing the above script, the original ~/.triton directory will be renamed, and a new ~/.triton directory will be created to store the pre-downloaded packages.
```

## Non-Source Installation
If you do not wish to build from source, you can directly pull and install whl (supports some backends).

```shell
# Note: First install PyTorch, then execute the following commands
python3 -m pip uninstall -y triton
RES="--index-url=https://resource.flagos.net/repository/flagos-pypi-hosted/simple --trusted-host=https://resource.flagos.net"
```
|Backend |Install cmd|Triton version|Python version|
|--------|-----------|--------------|--------------|
|nvidia  |python3 -m pip install flagtree==0.3.0rc1 $RES            |3.1|3.10, 3.11, 3.12|
|nvidia  |python3 -m pip install flagtree==0.3.0rc1+3.2 $RES        |3.2|3.10, 3.11, 3.12|
|nvidia  |python3 -m pip install flagtree==0.3.0rc1+3.3 $RES        |3.3|3.10, 3.11, 3.12|
|iluvatar|python3 -m pip install flagtree==0.3.0rc2+iluvatar3.1 $RES|3.1|3.10|
|mthreads|python3 -m pip install flagtree==0.3.0rc3+mthreads3.1 $RES|3.1|3.10|
|ascend  |python3 -m pip install flagtree==0.3.0rc1+ascend3.2 $RES  |3.2|3.11|
|hcu     |python3 -m pip install flagtree==0.3.0rc2+hcu3.0 $RES     |3.0|3.10|
|enflame |python3 -m pip install flagtree==0.3.0rc1+enflame3.3 $RES |3.3|3.10|

The flagtree version all have corresponding git tags.

## Running tests

After installation, you can run tests:
```shell
# nvidia
cd python/test
python3 -m pytest -s
# other backends
cd third_party/backendxxx/python/test
python3 -m pytest -s
python3 test_xxx.py
```

## Contributing

Contributions to FlagTree development are welcome. Please refer to [CONTRIBUTING.md](/CONTRIBUTING_cn.md) for details.

## License

FlagTree is licensed under the [MIT license](/LICENSE).
