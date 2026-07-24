# Flagtree Third-Party Backend - Enflame Accelerator Support
## Overview
The Flagtree third-party backend includes the backend implementation for Enflame accelerators. It provides core component backend bindings and test suites for developing and deploying applications on Enflame hardware platforms.

## Prerequisites
- Linux host system with Docker support
- Enflame 3rd-generation and 4th-generation accelerator cards
- Minimum 16GB memory (32GB recommended)
- 100GB available disk space

## Environment Preparation
### 1. Version Definition
```bash
# Software package
SDK=TopsRider_Triton_gcu-3.6.0-1.0.20260722.cc.1.10.6_deb_amd64.run
# Toolchain
LLVM=enflame-llvm23-fc83c68-gcc9-x64_v0.4.0.tar.gz
# Container image
IMAGE_PREFIX=flagtree-enflame3.6-py312-torch2.10.0-ubuntu24.04
IMAGE_VERSION=202607-1.10.6-base
IMAGE=${IMAGE_PREFIX}:${IMAGE_VERSION}
CONTAINER=${IMAGE_PREFIX}.${IMAGE_VERSION}
```

### 2. Clone Source Code
```bash
# Clone repository and switch to main branch
cd ~
git clone https://github.com/flagos-ai/FlagTree.git
cd FlagTree
git checkout main
```

### 3. Download Software Package
```bash
cd ~
wget https://baai-cp-web.ks3-cn-beijing.ksyuncs.com/trans/${SDK}
```

### 4. Install Driver
```bash
cd ~
bash ${SDK} --driver -y
# Verify driver installation
efsmi
```
Sample valid `efsmi` output for reference:
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

### 5. Prepare Docker Image
```bash
# Option A: Pull container image directly
docker pull harbor.baai.ac.cn/flagtree/${IMAGE}
```

```bash
# Option B: Download archive manually then load
wget https://baai-cp-web.ks3-cn-beijing.ksyuncs.com/trans/${CONTAINER}.tar.gz
docker load -i ${CONTAINER}.tar.gz
```

### 6. Launch Docker Container
```bash
# Delete old container if recreation is required
# docker rm -f ${CONTAINER}

# Assume FlagTree source locates at ~/FlagTree
docker run -itd --privileged --name ${CONTAINER} -v ~/FlagTree:/root/FlagTree ${IMAGE} bash
```

### 7. Enter Docker Container
```bash
# Attach to running container
docker exec -it ${CONTAINER} bash
```

> NOTICE: All subsequent commands shall be executed inside the container.

## Build & Compilation
### 1. Prepare Toolchain
```bash
mkdir -p ~/.flagtree/enflame
cd ~/.flagtree/enflame
wget baai-cp-web.ks3-cn-beijing.ksyuncs.com/trans/${LLVM}
tar -xzf ${LLVM}
```

### 2. Install Software Package
```bash
cd ~
bash ${SDK} --container -y
```

### 3. Configure Build Environment
```bash
export FLAGTREE_BACKEND=enflame
git config --global --add safe.directory ~/FlagTree
```

### 4. Install Python Dependencies
```bash
cd ~/FlagTree/python
pip3 install -r requirements.txt --break-system-packages
```

### 5. Build and Install Package
```bash
cd ~/FlagTree

# Initial build
pip3 install . --no-build-isolation -v --break-system-packages

# Rebuild after source code modification
pip3 install . --no-build-isolation --force-reinstall -v --break-system-packages
```

## Validation & Testing
```bash
# Run tutorial test case
cd ~/FlagTree
python python/tutorials/01-vector-add.py
```
