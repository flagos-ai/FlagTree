# Flagtree 框架 - 燧原加速器支持

## 概述

Flagtree 是针对燧原加速器优化的高性能计算框架。本代码仓库提供核心组件后端绑定和测试套件，用于在燧原硬件平台上开发和部署应用程序。

## 前提条件

- 支持 Docker 的 Linux 主机系统
- 燧原第三代加速卡（S60）
- 最小 16GB 内存（推荐 32GB）
- 100GB 可用磁盘空间

## 环境准备

### 1. 拉取源代码

```bash
# 拉取代码并切换到triton_v3.3.x分支
cd ~
git clone https://github.com/flagos-ai/flagtree.git
cd flagtree
git checkout triton_v3.3.x
```

### 2. 准备 Docker 镜像

```bash
# 加载预构建的容器镜像
curl -sL https://baai-cp-web.ks3-cn-beijing.ksyuncs.com/trans/enflame-flagtree-0.3.1.tar.gz | docker load

# 或手动下载后加载
wget https://baai-cp-web.ks3-cn-beijing.ksyuncs.com/trans/enflame-flagtree-0.3.1.tar.gz
docker load -i enflame-flagtree-0.3.1.tar.gz
```

### 3. 启动Docker容器

```bash
# 如果需要重建容器，请先删除
# docker rm -f enflame-flagtree

# 假设 flagtree 源码位于 ~/flagtree
docker run -itd \
  --privileged \
  --name enflame-flagtree \
  -v ~/flagtree:/root/flagtree \
  enflame/flagtree:0.3.1 bash
```

### 4. 安装驱动

```bash
# 提取并安装燧原驱动程序
docker cp enflame-flagtree:/enflame enflame

sudo bash enflame/driver/enflame-x86_64-gcc-1.6.3.12-20251115104629.run
# 如果上面的命令提示你使用其它参数，请按照提示操作，比如
# sudo bash enflame/driver/enflame-x86_64-gcc-1.6.3.12-20251115104629.run --virt-host

efsmi
```

用 efsmi 检查驱动是否正常安装，正常输出示意：

```
-------------------------------------------------------------------------------
--------------------- Enflame System Management Interface ---------------------
--------- Enflame Tech, All Rights Reserved. 2024-2025 Copyright (C) ----------
-------------------------------------------------------------------------------

+2025-11-28, 10:50:14 CST-----------------------------------------------------+
| EFSMI: 1.6.3.12          Driver Ver: 1.6.3.12                               |
+-----------------------------+-------------------+---------------------------+
| DEV    NAME                 | FW VER            | BUS-ID      ECC           |
| TEMP   Lpm   Pwr(Usage/Cap) | Mem      GCU Virt | DUsed       SN            |
|=============================================================================|
| 0      Enflame S60G         | 31.5.3            | 00:2e:00.0  Disable       |
| 34℃    LP0      N/A         | 23552MiB  SRIOV   | 0%          A018K30520031 |
+-----------------------------+-------------------+---------------------------+
| 1      Enflame S60G         | 31.5.3            | 00:2f:00.0  Disable       |
| 34℃    LP0      N/A         | 23552MiB  SRIOV   | 0%          A018K30520031 |
+-----------------------------+-------------------+---------------------------+
```

### 5. 进入Docker容器

```bash
# 执行docker
docker exec -it enflame-flagtree bash
```

> 注意，后续所有命令都在容器内进行。

## 编译构建

### 1. 准备工具链

```
mkdir -p ~/.flagtree/enflame
cd ~/.flagtree/enflame
wget baai-cp-web.ks3-cn-beijing.ksyuncs.com/trans/enflame-llvm21-d752c5b-gcc9-x64_v0.3.0.tar.gz
tar -xzf enflame-llvm21-d752c5b-gcc9-x64_v0.3.0.tar.gz
```

### 2. 配置构建环境

```bash
export FLAGTREE_BACKEND=enflame
git config --global --add safe.directory ~/flagtree
```

### 3. 安装 Python 依赖

```bash
cd ~/flagtree/python
pip3 install -r requirements.txt
```

### 4. 构建和安装包

```bash
cd ~/flagtree/python

# 初始构建
pip3 install . --no-build-isolation -v

# 代码修改后重新构建
pip3 install . --no-build-isolation --force-reinstall -v
```

## 测试验证

```bash
# 运行单元测试
cd ~/flagtree
pytest third_party/enflame/python/test/unit
```
