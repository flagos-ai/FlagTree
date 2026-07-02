[中文版|[English](./install_tsingmicro.md)]

## 💫 Tsingmicro（清微智能）[tsingmicro](https://github.com/flagos-ai/FlagTree/tree/triton_v3.3.x/third_party/tsingmicro/) (Triton 3.3)

- 对应的 Triton 版本为 3.3，基于 x64 平台
- 可用于 TX81

### 1. 构建及运行环境

#### 1.1 使用镜像（TX81）

如果网络环境畅通，不必执行后续步骤 1.x，依赖库会在构建时自动拉取。

```shell
# Plan A: docker pull (13.2GB)
IMAGE=harbor.baai.ac.cn/flagtree/flagtree-tsingmicro3.3-py310-torch2.7.0-ubuntu22.04:202606-clean
docker pull ${IMAGE}
# Plan B: docker load (5.5GB)
IMAGE=flagtree-tsingmicro3.3-py310-torch2.7.0-ubuntu22.04:202606-clean
wget https://baai-cp-web.ks3-cn-beijing.ksyuncs.com/trans/flagtree-tsingmicro3.3-py310-torch2.7.0-ubuntu22.04.202606-clean.tar.gz
docker load -i flagtree-tsingmicro3.3-py310-torch2.7.0-ubuntu22.04.202606-clean.tar.gz
```

```shell
CONTAINER=flagtree-dev-xxx
docker run -dit \
    --network=host --ipc=host --privileged \
    --security-opt seccomp=unconfined \
    -v /dev:/dev -v /lib/modules:/lib/modules -v /sys:/sys \
    -v /etc/localtime:/etc/localtime:ro \
    -v /data:/data -v /home:/home -v /tmp:/tmp \
    -w /root --name ${CONTAINER} ${IMAGE} bash
docker exec -it ${CONTAINER} /bin/bash
```

#### 1.2 手动下载 FlagTree 依赖库

注意对于 tsingmicro 后端，运行时也需要 FlagTree 依赖库。

```shell
mkdir -p ~/.flagtree/tsingmicro; cd ~/.flagtree/tsingmicro
wget https://baai-cp-web.ks3-cn-beijing.ksyuncs.com/trans/tsingmicro-llvm21-glibc2.30-glibcxx3.4.28-python3.10-x64_v0.6.0.tar.gz
tar zxvf tsingmicro-llvm21-glibc2.30-glibcxx3.4.28-python3.10-x64_v0.6.0.tar.gz
wget https://baai-cp-web.ks3-cn-beijing.ksyuncs.com/trans/tx8_depends_dev_20260507_104051_v0.6.0.tar.gz
tar zxvf tx8_depends_dev_20260507_104051_v0.6.0.tar.gz
```

#### 1.3 手动下载 Triton 依赖库

镜像中已下载安装 Triton 依赖库。
如果无需从源码构建 FlagTree 或 Triton，那么无需下载 Triton 依赖库。

```shell
cd ${YOUR_CODE_DIR}/FlagTree
# For Triton 3.3 (x64)
wget https://baai-cp-web.ks3-cn-beijing.ksyuncs.com/trans/build-deps-triton_3.3.x-linux-x64.tar.gz
sh python/scripts/unpack_triton_build_deps.sh ./build-deps-triton_3.3.x-linux-x64.tar.gz
```

执行完上述脚本后，原有的 ~/.triton 目录将被重命名，新的 ~/.triton 目录会被创建并存放预下载包。
注意执行脚本过程中会提示手动确认。

### 2. 安装命令

#### 2.1 免源码安装

```shell
# Note: First install PyTorch, then execute the following commands
python3 -m pip uninstall -y triton  # Repeat the cmd until fully uninstalled
RES="--index-url=https://resource.flagos.net/repository/flagos-pypi-hosted/simple"
python3.10 -m pip install flagtree===0.6.0+tsingmicro3.3 $RES
```

安装 `flagtree` 后，可通过下列命令查看：

```shell
python3 -m pip show flagtree
```

#### 2.2 从源码构建

构建和测试前需设置如下环境变量：

```bash
export TX8_DEPS_ROOT=~/.flagtree/tsingmicro/tx8_deps
export LLVM_SYSPATH=~/.flagtree/tsingmicro/tsingmicro-llvm21-glibc2.30-glibcxx3.4.28-python3.10-x64
export LLVM_BINARY_DIR=${LLVM_SYSPATH}/bin
export PYTHONPATH=${LLVM_SYSPATH}/python_packages/mlir_core:$PYTHONPATH
export LD_LIBRARY_PATH=$TX8_DEPS_ROOT/lib:$LD_LIBRARY_PATH
export PATH=${LLVM_BINARY_DIR}:${PATH}
```

```shell
cd ${YOUR_CODE_DIR}/FlagTree/python
git checkout -b triton_v3.3.x origin/triton_v3.3.x
export FLAGTREE_BACKEND=tsingmicro
MAX_JOBS=32 python3 -m pip install . --no-build-isolation -v
```

### 3. 测试验证

测试前需设置环境变量，见上文。

参考 [Tests of tsingmicro3.3 backend](https://github.com/flagos-ai/FlagTree/blob/triton_v3.3.x/.github/workflows/tsingmicro3.3-build-and-test.yml)
