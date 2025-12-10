<div align="right"><a href="/README.md">English</a></div>

## <img width="30" height="30" alt="FlagTree-GitHub" src="https://github.com/user-attachments/assets/d8d24c81-6f46-4adc-94e2-b89b03afcb43" /> FlagTree

FlagTree 是面向多种 AI 芯片的开源、统一编译器。FlagTree 致力于打造多元 AI 芯片编译器及相关工具平台，发展和壮大 Triton 上下游生态。项目当前处于初期，目标是兼容现有适配方案，统一代码仓库，快速实现单仓库多后端支持。对于上游模型用户，提供多后端的统一编译能力；对于下游芯片厂商，提供 Triton 生态接入范例。

## 新特性
* 2025/12/08 新增接入 enflame 后端，加入 CI/CD。
* 2025/11/26 添加 FlagTree 后端特化统一设计文档 [FlagTree_Backend_Specialization](reports/decoupling/)。
* 2025/10/28 提供离线构建支持（预下载依赖包），改善网络环境受限时的构建体验，使用方法见后文。
* 2025/09/30 在 GPGPU 上支持编译指导 shared memory。
* 2025/09/29 SDK 存储迁移至金山云，大幅提升下载稳定性。
* 2025/09/25 支持编译指导 ascend 的后端编译能力。
* 2025/09/16 新增接入 hcu 后端，加入 CI/CD。
* 2025/09/09 Fork 并修改 llvm-project，承接 FLIR 的支撑。
* 2025/09/01 新增适配 Paddle 框架，加入 CI/CD。
* 2025/08/16 新增适配北京超级云计算中心 AI 智算云。
* 2025/08/04 新增接入 T*** 后端。
* 2025/08/01 FLIR 支持编译指导 shared memory loading。
* 2025/07/30 更新 cambricon 后端。
* 2025/07/25 浪潮团队新增适配 OpenAnolis 龙蜥操作系统。
* 2025/07/09 FLIR 支持编译指导 Async DMA。
* 2025/07/08 新增多后端编译统一管理模块。
* 2025/07/02 FlagGems LibTuner 适配 triton_v3.3.x 版本。
* 2025/07/02 新增接入 S*** 后端。
* 2025/06/20 FLIR 开始承接 MLIR 扩展功能。
* 2025/06/06 新增接入 tsingmicro 后端，加入 CI/CD。
* 2025/06/04 新增接入 ascend 后端，加入 CI/CD。
* 2025/06/03 新增接入 metax 后端，加入 CI/CD。
* 2025/05/22 FlagGems LibEntry 适配 triton_v3.3.x 版本。
* 2025/05/21 FLIR 开始承接到中间层的转换功能。
* 2025/04/09 新增接入 arm aipu 后端，提供 torch 标准扩展范例，加入 CI/CD。
* 2025/03/26 接入安全合规扫描。
* 2025/03/19 新增接入 klx xpu 后端，加入 CI/CD。
* 2025/03/19 新增接入 mthreads 后端，加入 CI/CD。
* 2025/03/12 新增接入 iluvatar 后端，加入 CI/CD。

## 从源代码安装
安装依赖（注意使用正确的 python3.x 执行）：
```shell
apt install zlib1g zlib1g-dev libxml2 libxml2-dev  # ubuntu
cd python; python3 -m pip install -r requirements.txt
```

构建安装（网络畅通环境下推荐使用）：
```shell
cd python
export FLAGTREE_BACKEND=backendxxx
python3 -m pip install . --no-build-isolation -v
```

## 构建技巧

自动下载依赖库的速度可能受限于网络环境，编译前可自行下载至缓存目录 ~/.flagtree（可通过环境变量 FLAGTREE_CACHE_DIR 修改），无需自行设置 LLVM_BUILD_DIR 等环境变量。 <br>
各后端完整构建命令如下： <br>

[iluvatar](https://github.com/FlagTree/flagtree/tree/main/third_party/iluvatar/)
对应的 Triton 版本为 3.1
```shell
# 推荐使用镜像 Ubuntu 20.04
mkdir -p ~/.flagtree/iluvatar; cd ~/.flagtree/iluvatar
wget https://baai-cp-web.ks3-cn-beijing.ksyuncs.com/trans/iluvatar-llvm18-x86_64_v0.3.0.tar.gz
tar zxvf iluvatar-llvm18-x86_64_v0.3.0.tar.gz
wget https://baai-cp-web.ks3-cn-beijing.ksyuncs.com/trans/iluvatarTritonPlugin-cpython3.10-glibc2.30-glibcxx3.4.28-cxxabi1.3.12-ubuntu-x86_64_v0.3.0.tar.gz
tar zxvf iluvatarTritonPlugin-cpython3.10-glibc2.30-glibcxx3.4.28-cxxabi1.3.12-ubuntu-x86_64_v0.3.0.tar.gz
cd ${YOUR_CODE_DIR}/flagtree/python
export FLAGTREE_BACKEND=iluvatar
python3 -m pip install . --no-build-isolation -v
```
[xpu (klx)](https://github.com/FlagTree/flagtree/tree/main/third_party/xpu/)
对应的 Triton 版本为 3.0
```shell
# 推荐使用镜像（22GB）https://su.bcebos.com/klx-sdk-release-public/xpytorch/docker/ubuntu2004_v030/ubuntu_2004_x86_64_v30.tar
# 联系 kunlunxin-support@baidu.com 可获取进一步支持
mkdir -p ~/.flagtree/xpu; cd ~/.flagtree/xpu
wget https://baai-cp-web.ks3-cn-beijing.ksyuncs.com/trans/XTDK-llvm19-ubuntu2004_x86_64_v0.3.0.tar.gz
tar zxvf XTDK-llvm19-ubuntu2004_x86_64_v0.3.0.tar.gz
wget https://baai-cp-web.ks3-cn-beijing.ksyuncs.com/trans/xre-Linux-x86_64_v0.3.0.tar.gz
tar zxvf xre-Linux-x86_64_v0.3.0.tar.gz
cd ${YOUR_CODE_DIR}/flagtree/python
export FLAGTREE_BACKEND=xpu
python3 -m pip install . --no-build-isolation -v
```
[mthreads](https://github.com/FlagTree/flagtree/tree/main/third_party/mthreads/)
对应的 Triton 版本为 3.1
```shell
# 推荐使用镜像 flagtree/dockerfiles/Dockerfile-ubuntu22.04-python3.10-mthreads
mkdir -p ~/.flagtree/mthreads; cd ~/.flagtree/mthreads
wget https://baai-cp-web.ks3-cn-beijing.ksyuncs.com/trans/mthreads-llvm19-glibc2.35-glibcxx3.4.30-x64_v0.4.0.tar.gz
tar zxvf mthreads-llvm19-glibc2.35-glibcxx3.4.30-x64_v0.4.0.tar.gz
wget https://baai-cp-web.ks3-cn-beijing.ksyuncs.com/trans/mthreadsTritonPlugin-cpython3.10-glibc2.35-glibcxx3.4.30-cxxabi1.3.13-ubuntu-x86_64_v0.4.0.tar.gz
tar zxvf mthreadsTritonPlugin-cpython3.10-glibc2.35-glibcxx3.4.30-cxxabi1.3.13-ubuntu-x86_64_v0.4.0.tar.gz
cd ${YOUR_CODE_DIR}/flagtree/python
export FLAGTREE_BACKEND=mthreads
python3 -m pip install . --no-build-isolation -v
```
[aipu (arm npu)](https://github.com/FlagTree/flagtree/tree/triton_v3.3.x/third_party/aipu/)
对应的 Triton 版本为 3.3
```shell
# 推荐使用镜像 Ubuntu 22.04
mkdir -p ~/.flagtree/aipu; cd ~/.flagtree/aipu
# 模拟环境中使用 x64 版本，在 ARM 开发板上使用 arm64 版本
wget https://baai-cp-web.ks3-cn-beijing.ksyuncs.com/trans/llvm-a66376b0-ubuntu-x64-clang16-lld16_v0.4.0.tar.gz
tar zxvf llvm-a66376b0-ubuntu-x64-clang16-lld16_v0.4.0.tar.gz
cd ${YOUR_CODE_DIR}/flagtree/
git checkout -b triton_v3.3.x origin/triton_v3.3.x
export FLAGTREE_BACKEND=aipu
python3 -m pip install . --no-build-isolation -v
```
[tsingmicro](https://github.com/FlagTree/flagtree/tree/triton_v3.3.x/third_party/tsingmicro/)
对应的 Triton 版本为 3.3
```shell
# 推荐使用镜像 Ubuntu 20.04
mkdir -p ~/.flagtree/tsingmicro; cd ~/.flagtree/tsingmicro
wget https://baai-cp-web.ks3-cn-beijing.ksyuncs.com/trans/tsingmicro-llvm21-glibc2.30-glibcxx3.4.28-python3.11-x64_v0.2.0.tar.gz
tar zxvf tsingmicro-llvm21-glibc2.30-glibcxx3.4.28-python3.11-x64_v0.2.0.tar.gz
wget https://baai-cp-web.ks3-cn-beijing.ksyuncs.com/trans/tx8_depends_release_20250814_195126_v0.2.0.tar.gz
tar zxvf tx8_depends_release_20250814_195126_v0.2.0.tar.gz
export TX8_DEPS_ROOT=~/.flagtree/tsingmicro/tx8_deps
cd ${YOUR_CODE_DIR}/flagtree/
git checkout -b triton_v3.3.x origin/triton_v3.3.x
export FLAGTREE_BACKEND=tsingmicro
python3 -m pip install . --no-build-isolation -v
```
[ascend](https://github.com/FlagTree/flagtree/blob/triton_v3.2.x/python/setup_tools/setup_helper.py)
对应的 Triton 版本为 3.3，基于 aarch64 平台
```shell
# 推荐使用镜像 flagtree/dockerfiles/Dockerfile-ubuntu22.04-python3.11-ascend
# 在 https://www.hiascend.com/developer/download/community/result?module=cann
# 注册账号后下载对应平台的 cann-toolkit、cann-kernels
# cann-toolkit
chmod +x Ascend-cann-toolkit_8.3.RC1.alpha001_linux-aarch64.run
./Ascend-cann-toolkit_8.3.RC1.alpha001_linux-aarch64.run --install
# cann-kernels for 910B (A2)
chmod +x Ascend-cann-kernels-910b_8.3.RC1.alpha001_linux-aarch64.run
./Ascend-cann-kernels-910b_8.3.RC1.alpha001_linux-aarch64.run --install
# cann-kernels for 910C (A3)
chmod +x Atlas-A3-cann-kernels_8.3.RC1.alpha001_linux-aarch64.run
./Atlas-A3-cann-kernels_8.3.RC1.alpha001_linux-aarch64.run --install
# 构建安装
mkdir -p ~/.flagtree/ascend; cd ~/.flagtree/ascend
wget https://oaitriton.blob.core.windows.net/public/llvm-builds/llvm-b5cc222d-ubuntu-arm64.tar.gz
tar zxvf llvm-b5cc222d-ubuntu-arm64.tar.gz
cd ${YOUR_CODE_DIR}/flagtree/python
git checkout -b triton_v3.2.x origin/triton_v3.2.x
export FLAGTREE_BACKEND=ascend
python3 -m pip install . --no-build-isolation -v
```
[hcu](https://github.com/FlagTree/flagtree/tree/main/third_party/hcu/)
对应的 Triton 版本为 3.0
```shell
# 推荐使用镜像 flagtree/dockerfiles/Dockerfile-ubuntu22.04-python3.10-hcu
mkdir -p ~/.flagtree/hcu; cd ~/.flagtree/hcu
wget https://baai-cp-web.ks3-cn-beijing.ksyuncs.com/trans/hcu-llvm20-df0864e-glibc2.35-glibcxx3.4.30-ubuntu-x86_64_v0.3.0.tar.gz
tar zxvf hcu-llvm20-df0864e-glibc2.35-glibcxx3.4.30-ubuntu-x86_64_v0.3.0.tar.gz
cd ${YOUR_CODE_DIR}/flagtree/python
export FLAGTREE_BACKEND=hcu
python3 -m pip install . --no-build-isolation -v
```
[enflame](https://github.com/FlagTree/flagtree/tree/triton_v3.3.x/third_party/enflame/)
对应的 Triton 版本为 3.3
```shell
# 推荐使用镜像: https://baai-cp-web.ks3-cn-beijing.ksyuncs.com/trans/enflame-flagtree-0.3.1.tar.gz
mkdir -p ~/.flagtree/enflame; cd ~/.flagtree/enflame
wget https://baai-cp-web.ks3-cn-beijing.ksyuncs.com/trans/enflame-llvm21-d752c5b-gcc9-x64_v0.3.0.tar.gz
tar zxvf enflame-llvm21-d752c5b-gcc9-x64_v0.3.0.tar.gz
cd ${YOUR_CODE_DIR}/flagtree/python
export FLAGTREE_BACKEND=enflame
python3 -m pip install . --no-build-isolation -v
```

[nvidia](/third_party/nvidia/)
使用默认的构建命令，可以构建安装 nvidia、amd、triton_shared cpu 后端：
```shell
cd ${YOUR_LLVM_DOWNLOAD_DIR}
# 对应 Triton 3.1
wget https://oaitriton.blob.core.windows.net/public/llvm-builds/llvm-10dc3a8e-ubuntu-x64.tar.gz
tar zxvf llvm-10dc3a8e-ubuntu-x64.tar.gz
export LLVM_SYSPATH=${YOUR_LLVM_DOWNLOAD_DIR}/llvm-10dc3a8e-ubuntu-x64
# 对应 Triton 3.2
wget https://oaitriton.blob.core.windows.net/public/llvm-builds/llvm-86b69c31-ubuntu-x64.tar.gz
tar zxvf llvm-86b69c31-ubuntu-x64.tar.gz
export LLVM_SYSPATH=${YOUR_LLVM_DOWNLOAD_DIR}/llvm-86b69c31-ubuntu-x64
# 对应 Triton 3.3
wget https://oaitriton.blob.core.windows.net/public/llvm-builds/llvm-a66376b0-ubuntu-x64.tar.gz
tar zxvf llvm-a66376b0-ubuntu-x64.tar.gz
export LLVM_SYSPATH=${YOUR_LLVM_DOWNLOAD_DIR}/llvm-a66376b0-ubuntu-x64
# 对应 Triton 3.4
wget https://oaitriton.blob.core.windows.net/public/llvm-builds/llvm-8957e64a-ubuntu-x64.tar.gz
tar zxvf llvm-8957e64a-ubuntu-x64.tar.gz
export LLVM_SYSPATH=${YOUR_LLVM_DOWNLOAD_DIR}/llvm-8957e64a-ubuntu-x64
# 对应 Triton 3.5
wget https://oaitriton.blob.core.windows.net/public/llvm-builds/llvm-7d5de303-ubuntu-x64.tar.gz
tar zxvf llvm-7d5de303-ubuntu-x64.tar.gz
export LLVM_SYSPATH=${YOUR_LLVM_DOWNLOAD_DIR}/llvm-7d5de303-ubuntu-x64
#
export LLVM_INCLUDE_DIRS=$LLVM_SYSPATH/include
export LLVM_LIBRARY_DIR=$LLVM_SYSPATH/lib
cd ${YOUR_CODE_DIR}/flagtree
cd python  # 对应 Triton 3.1、3.2、3.3 时，需要进入 python 目录执行构建命令
git checkout main              # 对应 Triton 3.1
git checkout -b triton_v3.2.x  # 对应 Triton 3.2
git checkout -b triton_v3.3.x  # 对应 Triton 3.3
git checkout -b triton_v3.4.x  # 对应 Triton 3.4
git checkout -b triton_v3.5.x  # 对应 Triton 3.5
unset FLAGTREE_BACKEND
python3 -m pip install . --no-build-isolation -v
# 如果接下来需要构建安装其他后端，应清空 LLVM 相关环境变量
unset LLVM_SYSPATH LLVM_INCLUDE_DIRS LLVM_LIBRARY_DIR
```

## 离线构建支持：预下载依赖包
上文介绍了构建时 FlagTree 各后端可手动下载依赖包以避免受限于网络环境。但 Triton 构建时原本就带有一些依赖包，因此我们提供预下载包，可以手动安装至环境中，避免在构建时卡在自动下载阶段。
```shell
cd ${YOUR_CODE_DIR}/flagtree/python
sh README_offline_build.sh x86_64  # 查看说明
# 对应 Triton 3.1 (x64)
wget https://baai-cp-web.ks3-cn-beijing.ksyuncs.com/trans/offline-build-pack-triton-3.1.x-linux-x64.zip
sh scripts/offline_build_unpack.sh ./offline-build-pack-triton-3.1.x-linux-x64.zip ~/.triton
# 对应 Triton 3.2 (x64)
wget https://baai-cp-web.ks3-cn-beijing.ksyuncs.com/trans/offline-build-pack-triton-3.2.x-linux-x64.zip
sh scripts/offline_build_unpack.sh ./offline-build-pack-triton-3.2.x-linux-x64.zip ~/.triton
# 对应 Triton 3.2 (aarch64)
wget https://baai-cp-web.ks3-cn-beijing.ksyuncs.com/trans/offline-build-pack-triton-3.2.x-linux-aarch64.zip
sh scripts/offline_build_unpack.sh ./offline-build-pack-triton-3.2.x-linux-aarch64.zip ~/.triton
# 对应 Triton 3.3 (x64)
wget https://baai-cp-web.ks3-cn-beijing.ksyuncs.com/trans/offline-build-pack-triton-3.3.x-linux-x64.zip
sh scripts/offline_build_unpack.sh ./offline-build-pack-triton-3.3.x-linux-x64.zip ~/.triton
# 上述脚本执行后，会将原 ~/.triton 目录重命名，创建新的 ~/.triton 目录存放预下载包
```

## 运行测试

安装完成后可以在设备支持的环境下运行测试：
```shell
# nvidia
cd python/test
python3 -m pytest -s
# other backends
cd third_party/backendxxx/python/test
python3 -m pytest -s
python3 test_xxx.py
```

## 关于贡献

欢迎参与 FlagTree 的开发并贡献代码，详情请参考[CONTRIBUTING.md](/CONTRIBUTING_cn.md)。

## 许可证

FlagTree 使用 [MIT license](/LICENSE)。
