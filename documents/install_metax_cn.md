[中文版|[English](./install_metax.md)]

## 💫 MetaX（沐曦股份）[metax](/third_party/metax/)

- 对应的 Triton 版本为 3.0，基于 x64 平台
- 可用于 C550

### 1. 构建及运行环境

#### 1.1 使用预装镜像（C550）

使用该预装镜像，则不必执行后续步骤 1.x。
如果网络环境畅通，也不必执行后续步骤 1.x，依赖库会在构建时自动拉取。

```shell
IMAGE=flagtree-metax-py312-torch2.8.0-vllm0.15.0-metax3.5.3.x-ubuntu22.04:202604-0.5.1
# Plan A: docker pull (28.1GB)
docker pull harbor.baai.ac.cn/flagtree/${IMAGE}
# Plan B: docker load (8.1GB)
wget https://baai-cp-web.ks3-cn-beijing.ksyuncs.com/trans/flagtree-metax-py312-torch2.8.0-vllm0.15.0-metax3.5.3.x-ubuntu22.04.202604-0.5.1.tar.gz
docker load -i flagtree-metax-py312-torch2.8.0-vllm0.15.0-metax3.5.3.x-ubuntu22.04.202604-0.5.1.tar.gz
```

```shell
CONTAINER=flagtree-dev-xxx
docker run -dit \
    --net=host --uts=host --ipc=host --privileged=true \
    --group-add video \
    --shm-size 100gb --ulimit memlock=-1 \
    --security-opt seccomp=unconfined --security-opt apparmor=unconfined \
    --device=/dev/dri --device=/dev/mxcd \
    -v /etc/localtime:/etc/localtime:ro \
    -v /data:/data -v /home:/home -v /tmp:/tmp \
    -w /root --name ${CONTAINER} ${IMAGE} bash
docker exec -it ${CONTAINER} /bin/bash
```

#### 1.2 手动下载 FlagTree 依赖库

```shell
mkdir -p ~/.flagtree/metax; cd ~/.flagtree/metax
wget https://baai-cp-web.ks3-cn-beijing.ksyuncs.com/trans/metaxTritonPlugin-cpython3.12-glibc2.35-glibcxx3.4.30-cxxabi1.3.13-linux-x86_64_v0.5.0.tar.gz
tar zxvf metaxTritonPlugin-cpython3.12-glibc2.35-glibcxx3.4.30-cxxabi1.3.13-linux-x86_64_v0.5.0.tar.gz
wget https://baai-cp-web.ks3-cn-beijing.ksyuncs.com/trans/ext_maca_mathlib_bc_v0.5.0.tar.gz
tar zxvf ext_maca_mathlib_bc_v0.5.0.tar.gz
# NOTE: Contact the Metax vendor to obtain maca-llvm-metax20250708.521-x86_64.tar.xz
tar xvf maca-llvm-metax20250708.521-x86_64.tar.xz
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
python3.12 -m pip install flagtree===0.5.1+metax3.0 $RES
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
export FLAGTREE_BACKEND=metax
MAX_JOBS=32 python3 -m pip install . --no-build-isolation -v
```

### 3. 测试验证

参考 [Tests of metax backend](/.github/workflows/metax-build-and-test.yml)
