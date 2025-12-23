[English](../en/build.md)

## 构建指南

各后端基于不同版本的 Triton 适配，因此位于不同的主干分支：

- [main](https://github.com/flagos-ai/flagtree/tree/main) 对应 Triton 3.1
- [triton\_v3.2.x](https://github.com/flagos-ai/flagtree/tree/triton_v3.2.x)
- [triton\_v3.3.x](https://github.com/flagos-ai/flagtree/tree/triton_v3.3.x)
- [triton\_v3.4.x](https://github.com/flagos-ai/flagtree/tree/triton_v3.4.x)
- [triton\_v3.5.x](https://github.com/flagos-ai/flagtree/tree/triton_v3.5.x)

各主干分支均为保护分支且地位相等。

各后端完整构建命令如下：

### 天数智芯（Illuvatar）

后端：

- [iluvatar](https://github.com/flagos-ai/flagtree/tree/main/third_party/iluvatar/)

环境：

- Triton 3.1
- x86\_64
- Ubuntu 20.04 （建议）

```shell
mkdir -p ~/.flagtree/iluvatar; cd ~/.flagtree/iluvatar
wget https://baai-cp-web.ks3-cn-beijing.ksyuncs.com/trans/iluvatar-llvm18-x86_64_v0.4.0.tar.gz
tar zxvf iluvatar-llvm18-x86_64_v0.4.0.tar.gz

wget https://baai-cp-web.ks3-cn-beijing.ksyuncs.com/trans/iluvatarTritonPlugin-cpython3.10-glibc2.35-glibcxx3.4.30-cxxabi1.3.13-ubuntu-x86_64_v0.4.0.tar.gz
tar zxvf iluvatarTritonPlugin-cpython3.10-glibc2.35-glibcxx3.4.30-cxxabi1.3.13-ubuntu-x86_64_v0.4.0.tar.gz

cd ${YOUR_CODE_DIR}/flagtree/python
export FLAGTREE_BACKEND=iluvatar
python3 -m pip install . --no-build-isolation -v
```

### 昆仑芯（Kunlunxin）

后端：

- [xpu](https://github.com/flagos-ai/flagtree/tree/main/third_party/xpu/)

环境：

- Triton 3.0
- x86\_64

> [!Tip]
> 推荐使用镜像（22GB）https://su.bcebos.com/klx-sdk-release-public/xpytorch/docker/ubuntu2004_v030/ubuntu_2004_x86_64_v30.tar
> 联系 kunlunxin-support@baidu.com 可获取进一步支持

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

### 摩尔线程（Moore Threads）

后端：

- [mthreads](https://github.com/flagos-ai/flagtree/tree/main/third_party/mthreads/)

环境：

- Triton 3.1
- x86\_64/aarch64

> [!Tip]
> 推荐使用镜像 `dockerfiles/Dockerfile-ubuntu22.04-python3.10-mthreads`

准备目录：

```shell
mkdir -p ~/.flagtree/mthreads
cd ~/.flagtree/mthreads
```

对于 x86\_64 架构：

```
wget https://baai-cp-web.ks3-cn-beijing.ksyuncs.com/trans/mthreads-llvm19-glibc2.35-glibcxx3.4.30-x64_v0.4.0.tar.gz
tar zxvf mthreads-llvm19-glibc2.35-glibcxx3.4.30-x64_v0.4.0.tar.gz

wget https://baai-cp-web.ks3-cn-beijing.ksyuncs.com/trans/mthreadsTritonPlugin-cpython3.10-glibc2.35-glibcxx3.4.30-cxxabi1.3.13-ubuntu-x64_v0.4.1.tar.gz
tar zxvf mthreadsTritonPlugin-cpython3.10-glibc2.35-glibcxx3.4.30-cxxabi1.3.13-ubuntu-x64_v0.4.1.tar.gz
```

对于 aarch64 架构：

```shell
wget https://baai-cp-web.ks3-cn-beijing.ksyuncs.com/trans/mthreads-llvm19-glibc2.35-glibcxx3.4.30-aarch64_v0.4.0.tar.gz
tar zxvf mthreads-llvm19-glibc2.35-glibcxx3.4.30-aarch64_v0.4.0.tar.gz

wget https://baai-cp-web.ks3-cn-beijing.ksyuncs.com/trans/mthreadsTritonPlugin-cpython3.10-glibc2.35-glibcxx3.4.30-cxxabi1.3.13-ubuntu-aarch64_v0.4.0.tar.gz
tar zxvf mthreadsTritonPlugin-cpython3.10-glibc2.35-glibcxx3.4.30-cxxabi1.3.13-ubuntu-aarch64_v0.4.0.tar.gz
```

启动安装操作：

```shell
cd ${YOUR_CODE_DIR}/flagtree/python
export FLAGTREE_BACKEND=mthreads
python3 -m pip install . --no-build-isolation -v
```

### ARM AIPU

后端：

- [aipu](https://github.com/flagos-ai/flagtree/tree/triton_v3.3.x/third_party/aipu/)

环境：

- Triton 3.3
- x86\_64/arm64
- Ubuntu 22.04 （建议）

> [!Tip]
> 模拟环境中使用 x64 版本，在 ARM 开发板上使用 arm64 版本


```shell
mkdir -p ~/.flagtree/aipu
cd ~/.flagtree/aipu

wget https://baai-cp-web.ks3-cn-beijing.ksyuncs.com/trans/llvm-a66376b0-ubuntu-x64-clang16-lld16_v0.4.0.tar.gz
tar zxvf llvm-a66376b0-ubuntu-x64-clang16-lld16_v0.4.0.tar.gz

cd ${YOUR_CODE_DIR}/flagtree/python
git checkout -b triton_v3.3.x origin/triton_v3.3.x

export FLAGTREE_BACKEND=aipu
python3 -m pip install . --no-build-isolation -v
```

### 清微智能（TsingMicro）

后端：

- [tsingmicro](https://github.com/flagos-ai/flagtree/tree/triton_v3.3.x/third_party/tsingmicro/)

环境：

- Triton 3.3
- x86\_64
- Ubuntu 20.04 （建议）

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

### 华为昇腾（Ascend）

后端：

- [ascend](https://github.com/flagos-ai/flagtree/blob/triton_v3.2.x/third_party/ascend/)

环境：

- Triton 3.2
- aarch64

> [!Tip]
> 推荐使用镜像 dockerfiles/Dockerfile-ubuntu22.04-python3.11-ascend。
> 下载地址位于 https://www.hiascend.com/developer/download/community/result?module=cann。
> 注册账号后下载对应平台的 `cann-toolkit`、`cann-kernels`。

安装工具链：

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

构建安装：

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

### 海光（Hygon）

后端：

- [hcu](https://github.com/flagos-ai/flagtree/tree/main/third_party/hcu/)

环境：

- Triton 3.0
- x86\_64

> [!Tip]
> 推荐使用镜像 dockerfiles/Dockerfile-ubuntu22.04-python3.10-hcu

```shell
mkdir -p ~/.flagtree/hcu
cd ~/.flagtree/hcu

wget https://baai-cp-web.ks3-cn-beijing.ksyuncs.com/trans/hcu-llvm20-df0864e-glibc2.35-glibcxx3.4.30-ubuntu-x86_64_v0.3.0.tar.gz
tar zxvf hcu-llvm20-df0864e-glibc2.35-glibcxx3.4.30-ubuntu-x86_64_v0.3.0.tar.gz

cd ${YOUR_CODE_DIR}/flagtree/python
export FLAGTREE_BACKEND=hcu
python3 -m pip install . --no-build-isolation -v
```

### 燧原（Enflame）

后端：

- [enflame](https://github.com/FlagTree/flagtree/tree/triton_v3.3.x/third_party/enflame/)

环境：

- Triton 3.3
- x86\_64

> [!Tip]
> 推荐使用镜像: https://baai-cp-web.ks3-cn-beijing.ksyuncs.com/trans/enflame-flagtree-0.3.1.tar.gz

```shell
mkdir -p ~/.flagtree/enflame
cd ~/.flagtree/enflame

wget https://baai-cp-web.ks3-cn-beijing.ksyuncs.com/trans/enflame-llvm21-d752c5b-gcc9-x64_v0.3.0.tar.gz
tar zxvf enflame-llvm21-d752c5b-gcc9-x64_v0.3.0.tar.gz

cd ${YOUR_CODE_DIR}/flagtree/python
export FLAGTREE_BACKEND=enflame
python3 -m pip install . --no-build-isolation -v
```

### 英伟达（NVIDIA）

后端：

- [nvidia](https://github.com/flagos-ai/flagtree/tree/triton_v3.3.x/third_party/nvidia/)

使用默认的构建命令，可以构建安装 NVIDIA、AMD、triton\_shared CPU 后端：

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

export LLVM_INCLUDE_DIRS=$LLVM_SYSPATH/include
export LLVM_LIBRARY_DIR=$LLVM_SYSPATH/lib

cd ${YOUR_CODE_DIR}/flagtree
cd python  # 对应 Triton 3.1、3.2、3.3 时，需要进入 python 目录执行构建命令
git checkout main                                     # 对应 Triton 3.1
git checkout -b triton_v3.2.x  origin/triton_v3.2.x   # 对应 Triton 3.2
git checkout -b triton_v3.3.x  origin/triton_v3.3.x   # 对应 Triton 3.3
git checkout -b triton_v3.4.x  origin/triton_v3.4.x   # 对应 Triton 3.4
git checkout -b triton_v3.5.x  origin/triton_v3.5.x   # 对应 Triton 3.5
unset FLAGTREE_BACKEND
python3 -m pip install . --no-build-isolation -v
```

如果接下来需要构建安装其他后端，应清空 LLVM 相关环境变量

```shell
unset LLVM_SYSPATH LLVM_INCLUDE_DIRS LLVM_LIBRARY_DIR
```

## 离线构建支持：预下载依赖包

上文介绍了构建时 FlagTree 各后端可手动下载依赖包以避免受限于网络环境。
但 Triton 构建时原本就带有一些依赖包，因此我们提供预下载包，可以手动安装至环境中，
避免在构建时卡在自动下载阶段。

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
```

> [!Tip]
> 上述脚本执行后，会将原 `~/.triton` 目录重命名，创建新的 `~/.triton` 目录存放预下载包。
