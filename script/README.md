# FlagTree Ascend 后端编译步骤


## 步骤 1: 构建环境搭建

### 1.1 使用镜像 (910B)
``` bash
# Plan A: docker pull (13.3GB)
IMAGE=harbor.baai.ac.cn/flagtree/flagtree-ascend3.5-910b-py311-cann9.0.0-ubuntu22.04-aarch64:202606-torch2.9.0-base
docker pull ${IMAGE}
# Plan B: docker load (4.8GB)
IMAGE=flagtree-ascend3.5-910b-py311-cann9.0.0-ubuntu22.04-aarch64:202606-torch2.9.0-base
wget https://baai-cp-web.ks3-cn-beijing.ksyuncs.com/trans/flagtree-ascend3.5-910b-py311-cann9.0.0-ubuntu22.04-aarch64.202606-torch2.9.0-base.tar.gz
docker load -i flagtree-ascend3.5-910b-py311-cann9.0.0-ubuntu22.04-aarch64.202606-torch2.9.0-base.tar.gz
```
```bash
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

### 1.2 升级 CANN 至 9.1.0-beta.3
安装 CANN 社区版9.1.0-beta.3-昇腾社区 (同时需要 CANN Toolkit 和 ops 算子包)
参考:
https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/latest/softwareinst/instg/instg_0090.html?OS=openEuler&InstallType=netyum

安装后，确保在 /usr/local/Ascend 目录下 cann 指向 9.1.0-beta.3 版本。运行:
```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
```
## 步骤 2：拉取代码和依赖
### 2.1 拉取源码
```bash
git clone https://github.com/flagos-ai/flagtree

cd flagtree
```

### 2.2 手动下载 triton 依赖
```bash
# For Triton 3.5 (aarch64)
wget https://baai-cp-web.ks3-cn-beijing.ksyuncs.com/trans/build-deps-triton_3.5.x-linux-aarch64.tar.gz
sh python/scripts/unpack_triton_build_deps.sh ./build-deps-triton_3.5.x-linux-aarch64.tar.gz
```

### 2.3 手动下载 flagtree 依赖
``` bash
mkdir -p ~/.flagtree/ascend; cd ~/.flagtree/ascend
wget https://baai-cp-web.ks3-cn-beijing.ksyuncs.com/trans/llvm-7d5de303-ubuntu-aarch64-python311-compat_v0.6.0.tar.gz
tar zxvf llvm-7d5de303-ubuntu-aarch64-python311-compat_v0.6.0.tar.gz
```

## 步骤 3： 构建 & 测试
### 3.1 构建
``` bash
bash script/flagtree_build.sh
```
### 3.2 测试
```bash
python test/CommonIR/native_matmul.py

USE_CUSTOM_COMPILE_OPT=1 python test/CommonIR/matmul_add_residual_cv.py
```
