[中文版|[English](./install_sunrise.md)]

## 💫 Sunrise（曦望芯科）[sunrise](https://github.com/flagos-ai/FlagTree/tree/triton_v3.4.x/third_party/sunrise/)

- 对应的 Triton 版本为 3.4，基于 x64 平台
- 可用于 S2

### 1. 构建及运行环境

#### 1.1 使用预装镜像（S2）

使用该预装镜像，则不必执行后续步骤 1.x。
如果网络环境畅通，也不必执行后续步骤 1.x，依赖库会在构建时自动拉取。

```shell
TODO
```

#### 1.2 手动下载 FlagTree 依赖库

```shell
mkdir -p ~/.flagtree/sunrise; cd ~/.flagtree/sunrise
wget https://baai-cp-web.ks3-cn-beijing.ksyuncs.com/trans/sunrise-llvm21-glibc2.39-glibcxx3.4.33-x86_64_v0.4.0.tar.gz
tar zxvf sunrise-llvm21-glibc2.39-glibcxx3.4.33-x86_64_v0.4.0.tar.gz
wget https://baai-cp-web.ks3-cn-beijing.ksyuncs.com/trans/sunriseTritonPlugin-cpython3.10-glibc2.39-glibcxx3.4.33-x86_64_v0.4.0.tar.gz
tar zxvf sunriseTritonPlugin-cpython3.10-glibc2.39-glibcxx3.4.33-x86_64_v0.4.0.tar.gz
```

#### 1.3 手动下载 Triton 依赖库

预装镜像中已下载安装 Triton 依赖库。
如果无需从源码构建 FlagTree 或 Triton，那么无需下载 Triton 依赖库。

```shell
cd ${YOUR_CODE_DIR}/FlagTree
# For Triton 3.4 (x64)
wget https://baai-cp-web.ks3-cn-beijing.ksyuncs.com/trans/build-deps-triton_3.4.x-linux-x64.tar.gz
sh python/scripts/unpack_triton_build_deps.sh ./build-deps-triton_3.4.x-linux-x64.tar.gz
```

执行完上述脚本后，原有的 ~/.triton 目录将被重命名，新的 ~/.triton 目录会被创建并存放预下载包。
注意执行脚本过程中会提示手动确认。

### 2. 安装命令

#### 2.1 免源码安装

```shell
# Note: First install PyTorch, then execute the following commands
python3 -m pip uninstall -y triton  # Repeat the cmd until fully uninstalled
RES="--index-url=https://resource.flagos.net/repository/flagos-pypi-hosted/simple"
python3 -m pip install flagtree===0.4.0+sunrise3.4 $RES
```

预装镜像中已安装 `flagtree`，可通过下列命令查看：

```shell
python3 -m pip show flagtree
```

#### 2.2 从源码构建

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

### 3. 测试验证

参考 [Tests of sunrise backend](https://github.com/flagos-ai/FlagTree/blob/triton_v3.4.x/.github/workflows/sunrise-build-and-test.yml)
