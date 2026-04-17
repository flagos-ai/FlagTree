[中文版|[English](./install_ascend.md)]

## 💫 Huawei Ascend（华为昇腾）[ascend](https://github.com/flagos-ai/FlagTree/blob/triton_v3.2.x/third_party/ascend)

- 对应的 Triton 版本为 3.2，基于 aarch64 平台
- 可用于 910B/910C

### 1. 构建及运行环境

#### 1.1 使用预装镜像（910C）

该预装镜像基于 [Dockerfile-ubuntu22.04-python3.11-ascend](/dockerfiles/Dockerfile-ubuntu22.04-python3.11-ascend) 执行后续步骤 1.x 并安装 FlagTree 制作而成。
使用该预装镜像，则对于 910C 不必执行后续步骤 1.x，对于 910B 也仅需执行步骤 1.2。
如果网络环境畅通，也不必执行后续步骤 1.x，依赖库会在构建时自动拉取。

```shell
IMAGE=flagtree-ascend-910c-py311-torch2.6.0-cann8.5.0-ubuntu22.04-aarch64:202603
# Plan A: docker pull (26.2GB)
docker pull harbor.baai.ac.cn/flagtree/${IMAGE}
# Plan B: docker load (8.8GB)
wget https://baai-cp-web.ks3-cn-beijing.ksyuncs.com/trans/flagtree-ascend-910c-py311-torch2.6.0-cann8.5.0-ubuntu22.04-aarch64.202603.tar.gz
docker load -i flagtree-ascend-910c-py311-torch2.6.0-cann8.5.0-ubuntu22.04-aarch64.202603.tar.gz
```

```shell
CONTAINER=flagtree-dev-xxx
docker run -dit -u 0 --user=root \
    --network=host --pid=host --ipc=host --privileged \
    -v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
    -v /usr/local/Ascend/add-ons/:/usr/local/Ascend/add-ons/ \
    -v /usr/local/sbin/:/usr/local/sbin/ \
    -v /etc/ascend_install.info:/etc/ascend_install.info \
    --device=/dev/davinci0 --device=/dev/davinci1 \
    --device=/dev/davinci2 --device=/dev/davinci3 \
    --device=/dev/davinci4 --device=/dev/davinci5 \
    --device=/dev/davinci6 --device=/dev/davinci7 \
    --device=/dev/davinci_manager --device=/dev/devmm_svm --device=/dev/hisi_hdc \
    -v /etc/localtime:/etc/localtime:ro \
    -v /data:/data -v /home:/home -v /tmp:/tmp \
    -w /root --name ${CONTAINER} ${IMAGE} bash
docker exec -it ${CONTAINER} /bin/bash
```

#### 1.2 安装 cann

- 在 https://www.hiascend.com/developer/download/community/result?module=cann 注册账号后下载对应平台的 `cann-toolkit`、`cann-ops`

```shell
# cann-toolkit
chmod +x Ascend-cann-toolkit_8.5.0_linux-aarch64.run
./Ascend-cann-toolkit_8.5.0_linux-aarch64.run --install
# cann-ops for 910B (A2)
chmod +x Ascend-cann-910b-ops_8.5.0_linux-aarch64.run
./Ascend-cann-910b-ops_8.5.0_linux-aarch64.run --install
# cann-ops for 910C (A3)
chmod +x Ascend-cann-A3-ops_8.5.0_linux-aarch64.run
./Ascend-cann-A3-ops_8.5.0_linux-aarch64.run --install
```

#### 1.3 手动下载 FlagTree 依赖库

```shell
mkdir -p ~/.flagtree/ascend; cd ~/.flagtree/ascend
wget https://baai-cp-web.ks3-cn-beijing.ksyuncs.com/trans/llvm-a66376b0-ubuntu-aarch64-python311-compat_v0.3.0.tar.gz
tar zxvf llvm-a66376b0-ubuntu-aarch64-python311-compat_v0.3.0.tar.gz
```

#### 1.4 手动下载 Triton 依赖库

预装镜像中已下载安装 Triton 依赖库。
如果无需从源码构建 FlagTree 或 Triton，那么无需下载 Triton 依赖库。

```shell
cd ${YOUR_CODE_DIR}/FlagTree
# For Triton 3.2 (aarch64)
wget https://baai-cp-web.ks3-cn-beijing.ksyuncs.com/trans/build-deps-triton_3.2.x-linux-aarch64.tar.gz
sh python/scripts/unpack_triton_build_deps.sh ./build-deps-triton_3.2.x-linux-aarch64.tar.gz
```

执行完上述脚本后，原有的 ~/.triton 目录将被重命名，新的 ~/.triton 目录会被创建并存放预下载包。
注意执行脚本过程中会提示手动确认。

### 2. 安装命令

#### 2.1 免源码安装

```shell
# Note: First install PyTorch, then execute the following commands
python3 -m pip uninstall -y triton  # Repeat the cmd until fully uninstalled
RES="--index-url=https://resource.flagos.net/repository/flagos-pypi-hosted/simple"
python3 -m pip install flagtree===0.5.0+ascend3.2 $RES
```

预装镜像中已安装 `flagtree`，可通过下列命令查看：

```shell
python3 -m pip show flagtree
```

#### 2.2 从源码构建

```shell
apt update; apt install zlib1g zlib1g-dev libxml2 libxml2-dev
cd ${YOUR_CODE_DIR}/FlagTree/python
git checkout -b triton_v3.2.x origin/triton_v3.2.x
python3 -m pip install -r requirements.txt
export FLAGTREE_BACKEND=ascend
MAX_JOBS=32 python3 -m pip install . --no-build-isolation -v
```

### 3. 测试验证

参考 [Tests of ascend backend](https://github.com/flagos-ai/FlagTree/blob/triton_v3.2.x/.github/workflows/ascend-build-and-test.yml)
