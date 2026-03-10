# Flagtree 第三方后端 - 燧原加速器支持

## 概述

Flagtree 第三方后端包含针对燧原加速器后端，提供核心组件后端绑定和测试套件，用于在燧原硬件平台上开发和部署应用程序。

## 前提条件

- 支持 Docker 的 Linux 主机系统
- 燧原第三代、第四代加速卡
- 最小 16GB 内存（推荐 32GB）
- 100GB 可用磁盘空间

## 环境准备

### 1. 拉取源代码

```bash
# 拉取代码并切换到triton_v3.5.x分支
cd ~
git clone https://github.com/flagos-ai/FlagTree.git
cd FlagTree
git checkout -b triton_v3.5.x origin/triton_v3.5.x
```

### 2. 准备 Docker 镜像

```bash
# 加载预构建的容器镜像
curl -sL https://baai-cp-web.ks3-cn-beijing.ksyuncs.com/trans/enflame-flagtree-0.4.0.tar.gz | docker load

# 或手动下载后加载
wget https://baai-cp-web.ks3-cn-beijing.ksyuncs.com/trans/enflame-flagtree-0.4.0.tar.gz
docker load -i enflame-flagtree-0.4.0.tar.gz
```

### 3. 启动Docker容器

```bash
# 如果需要重建容器，请先删除
# docker rm -f enflame-flagtree

# 假设 flagtree 源码位于 ~/flagtree
docker run -itd --privileged --name enflame-flagtree -v ~/FlagTree:/root/FlagTree enflame/flagtree:0.4.0 bash
```

### 4. 安装驱动

```bash
# 提取并安装燧原驱动程序
docker cp enflame-flagtree:/enflame enflame

sudo bash enflame/driver/enflame-x86_64-gcc-*.run
# 如果上面的命令提示你使用其它参数，请按照提示操作，比如
# sudo bash enflame/driver/enflame-x86_64-gcc-*.run --virt-host

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

### 5. 重启Docker容器后进入

```bash
# 重启docker
docker restart enflame-flagtree
# 执行docker
docker exec -it enflame-flagtree bash
```

> 注意，后续所有命令都在容器内进行。

## 编译构建

### 1. 准备工具链

```
mkdir -p ~/.flagtree/enflame
cd ~/.flagtree/enflame
wget baai-cp-web.ks3-cn-beijing.ksyuncs.com/trans/enflame-llvm22-189e06b-gcc9-x64_v0.4.0.tar.gz
tar -xzf enflame-llvm22-189e06b-gcc9-x64_v0.4.0.tar.gz
```

### 2. 配置构建环境

```bash
export FLAGTREE_BACKEND=enflame
git config --global --add safe.directory ~/FlagTree
```

### 3. 安装 Python 依赖

```bash
cd ~/FlagTree/python
pip3 install -r requirements.txt --break-system-packages
```

### 4. 构建和安装包

```bash
cd ~/FlagTree

# 初始构建
pip3 install . --no-build-isolation -v --break-system-packages

# 代码修改后重新构建
pip3 install . --no-build-isolation --force-reinstall -v --break-system-packages
```

## 测试验证

```bash
# 运行单元测试
cd ~/FlagTree
pytest third_party/enflame/python/test/unit
```
