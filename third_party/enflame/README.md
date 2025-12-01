# Flagtree Framework - Enflame Accelerator Support

## Overview

Flagtree is a high-performance computing framework optimized for Enflame accelerators. This repository provides core component backend bindings and test suites for developing and deploying applications on Enflame hardware platforms.

## Prerequisites

- Linux host system with Docker support
- Enflame 3rd Generation Accelerator Card (S60)
- Minimum 16GB RAM (32GB recommended)
- 100GB available disk space

## Environment Preparation

### 1. Pull Source Code

```bash
# Pull code and switch to triton_v3.3.x branch
cd ~
git clone https://github.com/flagos-ai/flagtree.git
cd flagtree
git checkout triton_v3.3.x
```

### 2. Prepare Docker Image

```bash
# Load pre-built container image
curl -sL https://baai-cp-web.ks3-cn-beijing.ksyuncs.com/trans/enflame-flagtree-0.3.1.tar.gz | docker load

# Or manually download and load
wget https://baai-cp-web.ks3-cn-beijing.ksyuncs.com/trans/enflame-flagtree-0.3.1.tar.gz
docker load -i enflame-flagtree-0.3.1.tar.gz
```

### 3. Start Docker Container

```bash
# To re-run container, remove the existing one
# docker rm -f enflame-flagtree

# Assuming flagtree source code is located at ~/flagtree
docker run -itd \
  --privileged \
  --name enflame-flagtree \
  -v ~/flagtree:/root/flagtree \
  enflame/flagtree:0.3.1 bash
```

### 4. Install Driver

```bash
# Extract and install Enflame driver
docker cp enflame-flagtree:/enflame enflame

sudo bash enflame/driver/enflame-x86_64-gcc-1.6.3.12-20251115104629.run
# Use other arguments if prompt, e.g.
# sudo bash enflame/driver/enflame-x86_64-gcc-1.6.3.12-20251115104629.run --virt-host

efsmi
```

Check driver status with efsmi. Example output:

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

### 5. Enter Docker Container

```bash
# Execute docker
docker exec -it enflame-flagtree bash
```

> Note: All subsequent commands should be executed within the container.

## Build and Install

### 1. Prepare Toolchain

```
mkdir -p ~/.flagtree/enflame
cd ~/.flagtree/enflame
wget baai-cp-web.ks3-cn-beijing.ksyuncs.com/trans/enflame-llvm21-d752c5b-gcc9-x64_v0.3.0.tar.gz
tar -xzf enflame-llvm21-d752c5b-gcc9-x64_v0.3.0.tar.gz
```

### 2. Configure Build Environment

```bash
export FLAGTREE_BACKEND=enflame
git config --global --add safe.directory ~/flagtree
```

### 3. Install Python Dependencies

```bash
cd ~/flagtree/python
pip3 install -r requirements.txt
```

### 4. Build and Install Package

```bash
cd ~/flagtree/python

# Initial build
pip3 install . --no-build-isolation -v

# Rebuild after code modification
pip3 install . --no-build-isolation --force-reinstall -v
```

## Test Validation

```bash
# Run unit tests
cd ~/flagtree
pytest third_party/enflame/python/test/unit
```
