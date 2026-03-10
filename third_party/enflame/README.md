# Flagtree Third Party Backend - Enflame Accelerator Support

## Overview

Flagtree Third Party Backend for Enflame accelerators, including core component backend bindings and test suites for developing and deploying applications on Enflame hardware platforms.

## Prerequisites

- Linux host system with Docker support
- Enflame 3rd/4th Generation Accelerator Card
- Minimum 16GB RAM (32GB recommended)
- 100GB available disk space

## Environment Preparation

### 1. Pull Source Code

```bash
# Pull code and switch to triton_v3.5.x branch
cd ~
git clone https://github.com/flagos-ai/FlagTree.git
cd FlagTree
git checkout -b triton_v3.5.x origin/triton_v3.5.x
```

### 2. Prepare Docker Image

```bash
# Load pre-built container image
curl -sL https://baai-cp-web.ks3-cn-beijing.ksyuncs.com/trans/enflame-flagtree-0.4.0.tar.gz | docker load

# Or manually download and load
wget https://baai-cp-web.ks3-cn-beijing.ksyuncs.com/trans/enflame-flagtree-0.4.0.tar.gz
docker load -i enflame-flagtree-0.4.0.tar.gz
```

### 3. Start Docker Container

```bash
# To re-run container, remove the existing one
# docker rm -f enflame-flagtree

# Assuming flagtree source code is located at ~/flagtree
docker run -itd --privileged --name enflame-flagtree -v ~/FlagTree:/root/FlagTree enflame/flagtree:0.4.0 bash
```

### 4. Install Driver

```bash
# Extract and install Enflame driver
docker cp enflame-flagtree:/enflame enflame

sudo bash enflame/driver/enflame-x86_64-gcc-*.run
# Use other arguments if prompt, e.g.
# sudo bash enflame/driver/enflame-x86_64-gcc-*.run --virt-host

efsmi
```

Check driver status with efsmi. Example output:

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

### 5. Restart Docker Container and Enter

```bash
# Restart docker
docker restart enflame-flagtree
# Execute docker
docker exec -it enflame-flagtree bash
```

> Note: All subsequent commands should be executed within the container.

## Build and Install

### 1. Prepare Toolchain

```
mkdir -p ~/.flagtree/enflame
cd ~/.flagtree/enflame
wget baai-cp-web.ks3-cn-beijing.ksyuncs.com/trans/enflame-llvm22-189e06b-gcc9-x64_v0.4.0.tar.gz
tar -xzf enflame-llvm22-189e06b-gcc9-x64_v0.4.0.tar.gz
```

### 2. Configure Build Environment

```bash
export FLAGTREE_BACKEND=enflame
git config --global --add safe.directory ~/FlagTree
```

### 3. Install Python Dependencies

```bash
cd ~/FlagTree/python
pip3 install -r requirements.txt --break-system-packages
```

### 4. Build and Install Package

```bash
cd ~/FlagTree

# Initial build
pip3 install . --no-build-isolation -v --break-system-packages

# Rebuild after code modification
pip3 install . --no-build-isolation --force-reinstall -v --break-system-packages
```

## Test Validation

```bash
# Run unit tests
cd ~/FlagTree
pytest third_party/enflame/python/test/unit
```
