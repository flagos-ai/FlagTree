[中文版|[English](./install_mthreads.md)]

## 💫 Moore Threads（摩尔线程）[mthreads](https://github.com/flagos-ai/FlagTree/tree/main/third_party/mthreads/)

- 对应的 Triton 版本为 3.1，基于 x64/aarch64 平台
- 可用于 S4000/S5000

### 1. 构建及运行环境

#### 1.1 使用预装镜像（MTT-S5000）

使用该预装镜像，则不必执行后续步骤 1.x。
如果网络环境畅通，也不必执行后续步骤 1.x，依赖库会在构建时自动拉取。

```shell
IMAGE=flagtree-mthreads-py310-torch2.7.1-musa4.3.5-ubuntu22.04:202603
# Plan A: docker pull (55.3GB)
docker pull harbor.baai.ac.cn/flagtree/${IMAGE}
# Plan B: docker load (18GB)
wget https://baai-cp-web.ks3-cn-beijing.ksyuncs.com/trans/flagtree-mthreads-py310-torch2.7.1-musa4.3.5-ubuntu22.04.202603.tar.gz
docker load -i flagtree-mthreads-py310-torch2.7.1-musa4.3.5-ubuntu22.04.202603.tar.gz
```

```shell
CONTAINER=flagtree-dev-xxx
docker run -dit \
    --network=host --pid=host --privileged \
    --cap-add=SYS_PTRACE \
    --shm-size 16gb \
    --security-opt seccomp=unconfined \
    -e MTHREADS_VISIBLE_DEVICES=all -e MTHREADS_DRIVER_CAPABILITIES=all \
    -v /usr/lib/x86_64-linux-gnu:/usr/lib/x86_64-linux-gnu \
    -v /lib/x86_64-linux-gnu:/lib/x86_64-linux-gnu \
    -v /etc/alternatives:/etc/alternatives \
    -v /etc/localtime:/etc/localtime:ro \
    -v /data:/data -v /home:/home -v /tmp:/tmp \
    -w /root --name ${CONTAINER} ${IMAGE} bash
docker exec -it ${CONTAINER} /bin/bash
```

#### 1.2 手动下载 FlagTree 依赖库

```shell
mkdir -p ~/.flagtree/mthreads; cd ~/.flagtree/mthreads
# x64
wget https://baai-cp-web.ks3-cn-beijing.ksyuncs.com/trans/mthreads-llvm19-glibc2.35-glibcxx3.4.30-x64_v0.4.0.tar.gz
tar zxvf mthreads-llvm19-glibc2.35-glibcxx3.4.30-x64_v0.4.0.tar.gz \
    -C ./mthreads-llvm19-glibc2.35-glibcxx3.4.30 --strip-components=1
wget https://baai-cp-web.ks3-cn-beijing.ksyuncs.com/trans/mthreadsTritonPlugin-cpython3.10-glibc2.35-glibcxx3.4.30-cxxabi1.3.13-ubuntu-x64_v0.4.1.tar.gz
tar zxvf mthreadsTritonPlugin-cpython3.10-glibc2.35-glibcxx3.4.30-cxxabi1.3.13-ubuntu-x64_v0.4.1.tar.gz
# aarch64
wget https://baai-cp-web.ks3-cn-beijing.ksyuncs.com/trans/mthreads-llvm19-glibc2.35-glibcxx3.4.30-aarch64_v0.4.0.tar.gz
tar zxvf mthreads-llvm19-glibc2.35-glibcxx3.4.30-aarch64_v0.4.0.tar.gz \
    -C ./mthreads-llvm19-glibc2.35-glibcxx3.4.30 --strip-components=1
wget https://baai-cp-web.ks3-cn-beijing.ksyuncs.com/trans/mthreadsTritonPlugin-cpython3.10-glibc2.35-glibcxx3.4.30-cxxabi1.3.13-ubuntu-aarch64_v0.4.0.tar.gz
tar zxvf mthreadsTritonPlugin-cpython3.10-glibc2.35-glibcxx3.4.30-cxxabi1.3.13-ubuntu-aarch64_v0.4.0.tar.gz
```

#### 1.3 手动下载 Triton 依赖库

预装镜像中已下载安装 Triton 依赖库。
如果无需从源码构建 FlagTree 或 Triton，那么无需下载 Triton 依赖库。

```shell
cd ${YOUR_CODE_DIR}/FlagTree
# For Triton 3.1 (x64)
wget https://baai-cp-web.ks3-cn-beijing.ksyuncs.com/trans/build-deps-triton_3.1.x-linux-x64.tar.gz
sh python/scripts/unpack_triton_build_deps.sh ./build-deps-triton_3.1.x-linux-x64.tar.gz
```

执行完上述脚本后，原有的 ~/.triton 目录将被重命名，新的 ~/.triton 目录会被创建并存放预下载包。
注意执行脚本过程中会提示手动确认。

### 2. 安装命令

#### 2.1 免源码安装

```shell
# Note: First install PyTorch, then execute the following commands
python3 -m pip uninstall -y triton  # Repeat the cmd until fully uninstalled
RES="--index-url=https://resource.flagos.net/repository/flagos-pypi-hosted/simple"
python3 -m pip install flagtree===0.5.1+mthreads3.1 $RES
```

预装镜像中已安装 `flagtree`，可通过下列命令查看：

```shell
python3 -m pip show flagtree
```

#### 2.2 从源码构建

```shell
apt update; apt install zlib1g zlib1g-dev libxml2 libxml2-dev
cd ${YOUR_CODE_DIR}/FlagTree/python
python3 -m pip install -r requirements.txt
export FLAGTREE_BACKEND=mthreads
MAX_JOBS=32 python3 -m pip install . --no-build-isolation -v
```

### 3. 测试验证

参考 [Tests of mthreads backend](/.github/workflows/mthreads-build-and-test.yml)

对于使用 `tl.dot` 的 triton kernel，设置环境变量 `export MUSA_ENABLE_SQMMA=1` 可提升性能。
