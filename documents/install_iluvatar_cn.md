[中文版|[English](./install_iluvatar.md)]

## 💫 ILUVATAR（天数智芯）[iluvatar](https://github.com/flagos-ai/FlagTree/tree/main/third_party/iluvatar/)

- 对应的 Triton 版本为 3.1，基于 x64 平台
- 可用于 MR-V100/BI-V150

### 1. 构建及运行环境

#### 1.1 使用预装镜像（BI-V150）

使用该预装镜像，则不必执行后续步骤 1.x。
如果网络环境畅通，也不必执行后续步骤 1.x，依赖库会在构建时自动拉取。

```shell
modinfo iluvatar | grep "description"  # f65d8ac7
IMAGE=flagtree-iluvatar-py312-torch2.7.1-4.4.0release_f65d8ac7-ubuntu24.04:202603  # TODO
# Plan A: docker pull (GB)
docker pull harbor.baai.ac.cn/flagtree/${IMAGE}
# Plan B: docker load (GB)
wget https://baai-cp-web.ks3-cn-beijing.ksyuncs.com/trans/flagtree-iluvatar-py312-torch2.7.1-4.4.0release_f65d8ac7-ubuntu24.04.202603.tar.gz
docker load -i flagtree-iluvatar-py312-torch2.7.1-4.4.0release_f65d8ac7-ubuntu24.04.202603.tar.gz
```

```shell
CONTAINER=flagtree-dev-xxx
docker run -dit \
    --net=host --pid=host --privileged \
    --cap-add=ALL \
    --security-opt seccomp=unconfined \
    -v /lib/modules:/lib/modules -v /dev:/dev  \
    -v /etc/localtime:/etc/localtime:ro \
    -v /data1:/data1 -v /home:/home -v /tmp:/tmp \
    -w /root --name ${CONTAINER} ${IMAGE}
docker exec -it ${CONTAINER} /bin/bash
```

#### 1.2 手动下载 FlagTree 依赖库

```shell
mkdir -p ~/.flagtree/iluvatar; cd ~/.flagtree/iluvatar
wget https://baai-cp-web.ks3-cn-beijing.ksyuncs.com/trans/iluvatar-llvm18-x86_64_v0.5.0.tar.gz
tar zxvf iluvatar-llvm18-x86_64_v0.5.0.tar.gz
ABI=$(echo | g++ -dM -E -x c++ - | awk '/__GXX_ABI_VERSION/{print $3}')
case "$ABI" in
  1013) PLUGIN_TGZ=iluvatarTritonPlugin-cpython3.10-glibc2.17-glibcxx3.4.19-cxxabi1.3.13-linux-x86_64_v0.5.0.tar.gz ;;
  1016) PLUGIN_TGZ=iluvatarTritonPlugin-cpython3.10-glibc2.35-glibcxx3.4.30-cxxabi1.3.16-ubuntu-x86_64_v0.5.0.tar.gz ;;
  1018) PLUGIN_TGZ=iluvatarTritonPlugin-cpython3.10-glibc2.39-glibcxx3.4.33-cxxabi1.3.18-ubuntu-x86_64_v0.5.0.tar.gz ;;
  *) echo "不支持的 __GXX_ABI_VERSION=$ABI，请更新 plugin 包映射"; exit 1 ;;
esac
wget "https://baai-cp-web.ks3-cn-beijing.ksyuncs.com/trans/${PLUGIN_TGZ}"  # TODO: py312
tar zxvf "${PLUGIN_TGZ}"
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
python3 -m pip install flagtree===0.5.1+iluvatar3.1 $RES
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
export FLAGTREE_BACKEND=iluvatar
MAX_JOBS=32 python3 -m pip install . --no-build-isolation -v
```

### 3. 测试验证

参考 [Tests of iluvatar backend](/.github/workflows/iluvatar-build-and-test.yml)
