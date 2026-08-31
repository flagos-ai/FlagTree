# Flagtree 第三方后端 - 燧原加速器支持

## 概述

Flagtree 第三方后端包含针对燧原加速器后端，提供核心组件后端绑定和测试套件，用于在燧原硬件平台上开发和部署应用程序。

## 前提条件

- 支持 Docker 的 Linux 主机系统
- 燧原第三代、第四代加速卡
- 最小 16GB 内存（推荐 32GB）
- 100GB 可用磁盘空间

## 环境准备

### 1. 版本定义

```bash
# 软件包（GCU300 与 GCU400 使用不同的软件栈，请按目标平台选择）
# GCU400 平台使用：
SDK400=TopsRider_Triton_gcu-3.6.0+1.0.20260826.cc.1.10.25_deb_amd64.run
# GCU300 平台使用：
SDK300=TopsRider_Triton_gcu-3.6.0+1.0.20260826.cc.1.9.29_deb_amd64.run
# 根据实际平台设置 SDK 变量（二选一）：
SDK=${SDK400}   # GCU400 平台
# SDK=${SDK300} # GCU300 平台
# 工具链
LLVM=enflame-llvm23-fc83c68-gcc9-x64_v0.4.0.tar.gz
# 镜像
IMAGE_PREFIX=flagtree-enflame3.6-py312-torch2.10.0-ubuntu24.04
IMAGE_VERSION=202607-1.10.6-base
IMAGE=${IMAGE_PREFIX}:${IMAGE_VERSION}
CONTAINER=${IMAGE_PREFIX}.${IMAGE_VERSION}
```

### 2. 拉取源代码

```bash
# 拉取代码并切换到main分支
cd ~
git clone https://github.com/flagos-ai/FlagTree.git
cd FlagTree
git checkout main
```

### 3. 拉取软件包

```bash
cd ~
# 按目标平台拉取对应的软件包（与上方 SDK 设置保持一致）
# GCU400 平台：wget https://baai-cp-web.ks3-cn-beijing.ksyuncs.com/trans/${SDK400}
# GCU300 平台：wget https://baai-cp-web.ks3-cn-beijing.ksyuncs.com/trans/${SDK300}
wget https://baai-cp-web.ks3-cn-beijing.ksyuncs.com/trans/${SDK}
```

### 4. 安装驱动

```bash
cd ~
# 使用目标平台对应的软件包安装驱动（GCU400 用 SDK400，GCU300 用 SDK300）
bash ${SDK} --driver -y
# 检查驱动是否正常安装
efsmi
```
用 efsmi 检查驱动是否正常安装，正常输出示意：

```
-------------------------------------------------------------------------------
--------------------- Enflame System Management Interface ---------------------
--------- Enflame Tech, All Rights Reserved. 2024-2026 Copyright (C) ----------
-------------------------------------------------------------------------------

+2026-03-06, 10:12:03 CST-----------------------------------------------------+
| EFSMI: 1.7.2.14          Driver Ver: 1.7.2.14                               |
+-----------------------------+-------------------+---------------------------+
| DEV    NAME                 | Boot FW VER       | BUS-ID      ECC           |
| TEMP   Lpm   Pwr(Usage/Cap) | Mem      GCU Virt | DUsed       SN            |
|=============================================================================|
| 0      Enflame L300         | 40.2.8.3          | 00:2d:00.0  Enable        |
| 35℃    LP1      68W / 300W  | 147456MiB Disable | 0%          A098Q50610048 |
+-----------------------------+-------------------+---------------------------+
```

> 说明：efsmi 输出的设备型号（NAME 列）因平台而异 —— GCU300 平台显示 `Enflame S60`，GCU400 平台显示 `Enflame L300/L600`。以上示例为 GCU400 平台的 L300 设备。

### 5. 准备 Docker 镜像


```bash
# 方案A: 直接拉取容器镜像
docker pull harbor.baai.ac.cn/flagtree/${IMAGE}
```

```bash
# 方案B: 或手动下载后加载
wget https://baai-cp-web.ks3-cn-beijing.ksyuncs.com/trans/${CONTAINER}.tar.gz
docker load -i ${CONTAINER}.tar.gz
```

### 6. 启动Docker容器

```bash
# 如果需要重建容器，请先删除
# docker rm -f ${CONTAINER}

# 假设 flagtree 源码位于 ~/flagtree
docker run -itd --privileged --name ${CONTAINER} -v ~/FlagTree:/root/FlagTree ${IMAGE} bash
```

### 7. 进入Docker容器

```bash
# 执行docker
docker exec -it ${CONTAINER} bash
```

> 注意，后续所有命令都在容器内进行。

## 编译构建

### 1. 准备工具链

```bash
mkdir -p ~/.flagtree/enflame
cd ~/.flagtree/enflame
wget baai-cp-web.ks3-cn-beijing.ksyuncs.com/trans/${LLVM}
tar -xzf ${LLVM}
```

### 2. 安装软件包
```bash
cd ~
# 使用目标平台对应的软件包安装容器内软件栈（GCU400 用 SDK400，GCU300 用 SDK300）
bash ${SDK} --container -y
```

### 3. 配置构建环境

```bash
export FLAGTREE_BACKEND=enflame
git config --global --add safe.directory ~/FlagTree
```

### 4. 安装 Python 依赖

```bash
cd ~/FlagTree/python
pip3 install -r requirements.txt --break-system-packages
```

### 5. 构建和安装包

```bash
cd ~/FlagTree

# 初始构建
pip3 install . --no-build-isolation -v --break-system-packages

# 代码修改后重新构建
pip3 install . --no-build-isolation --force-reinstall -v --break-system-packages
```

## 测试验证

```bash
# 运行tutorial测试
cd ~/FlagTree
python python/tutorials/01-vector-add.py
```
