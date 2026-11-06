[[English](./install_cpu.md)|中文版]

## 💫 Kunpeng arm64 (openEuler) install

- 对应OpenEuler Triton 版本 3.2.0，基于 OpenEuler LLVM 20.1.8分支版本，aarch64 平台
- 目标平台：openEuler(24.03 LTS SP4) + Kunpeng arm64
- 暂未提供 docker 镜像，请按下述源码方式安装。

### 1. 编译流程概览

下面给出 AArch64 CPU 后端的编译流程概览（方括号内为后端编译阶段名 `ttir`、`ttsharedir`、`llir`、`obj`）。

```text
Triton Python Kernel（用户代码）
│
▼
TTIR —— Triton IR（tt.* 方言）  [阶段: ttir]
│   tt.load / tt.store / tt.dot / tt.reduce ...
│
│   triton-shared 转换（triton-to-linalg-experimental；enable-armpl?）
│     · 受支持的 2-D 浮点 tt.dot → tts.armpl_matmul/gemv → func.call @cblas_*
│     · 其余 dot / 算子           → linalg / tensor / memref / scf / arith
▼
MLIR 标准方言  [阶段: ttsharedir]
│   tensor 级 linalg（+ 被 ArmPL 接管的 dot 的 func.call @cblas_*）
│   _optimize_ttsharedir：选择 transform 调度并拼到 IR 末尾
│
│   拼入的调度在进入 llir 阶段时由 transform interpreter 执行
▼
编译器生成 lowering（在 llir 阶段开始时执行）
│   SME（aarch64 + sme + 含 matmul）：linalg_opts → batch matmul 切分
│        → matmul 切分 + 向量化 → bufferize（+ vector.contract → outerproduct）
│        → lower_to_arm_sme → arm-sme-outer-product-fusion
│   SVE（aarch64 + sve）：linalg_opts → contraction 切分/pad/向量化
│        → bufferize → legalize → loops
│   既无 SME 也无 SVE：通用 lowering（affine/linalg → loops）
▼
LLVM 方言 → mlir-translate → LLVM IR
│   opt：cleanup + loop-vectorize；-mtriple/-mattr 仅 SME/SVE 路径传入
│
│   llc -O3 -filetype=obj ...
▼
AArch64 机器码（.o）  [阶段: obj]
│   driver：clang++ -shared <kernel.o> <launcher.cxx> → kernel.so
▼
kernel.so（NEON / SVE / SME，由所选路径的 -mattr 决定）
```

### 2. 构建及运行环境

#### 2.1 系统依赖
openEuler使用
```shell
sudo dnf install -y \
    gcc gcc-c++ make cmake ninja-build git ccache pkgconf-pkg-config \
    libomp-devel jemalloc zlib zlib-devel libxml2 libxml2-devel \
    nlohmann-json-devel ca-certificates curl wget numactl \
    python3-devel python3-pip python3-virtualenv
```

#### 2.2 创建虚拟环境并安装 PyTorch
pip源基于网络情况可以换成阿里云/清华源
```shell
python3 -m venv ~/venv-triton-cpu
source ~/venv-triton-cpu/bin/activate
pip install "setuptools>=40.8.0" -i https://mirrors.aliyun.com/pypi/simple/
pip install wheel -i https://mirrors.aliyun.com/pypi/simple/
pip install "cmake>=3.18,<4.0" -i https://mirrors.aliyun.com/pypi/simple/
pip install "ninja>=1.11.1" -i https://mirrors.aliyun.com/pypi/simple/
pip install "pybind11>=2.13.1" -i https://mirrors.aliyun.com/pypi/simple/
pip install lit -i https://mirrors.aliyun.com/pypi/simple/
pip install "nanobind<3.0" -i https://mirrors.aliyun.com/pypi/simple/
pip install numpy -i https://mirrors.aliyun.com/pypi/simple/
pip install torch==2.10.0 -i https://mirrors.aliyun.com/pypi/simple/
```

### 3. 构建命令

#### 3.1 LLVM 源码构建
共两步：**克隆LLVM 20.1.8 → 构建安装LLVM**。

**Step 1 - 克隆LLVM 20.1.8**
```shell
cd ${YOUR_CODE_DIR}
git clone https://atomgit.com/openeuler/llvm-project.git -b dev_20.1.8
```

**Step 2 - LLVM源码构建**
```shell
cd ${YOUR_CODE_DIR}/llvm-project
mkdir build && cd build
# Cmake配置
cmake -G Ninja  \
-DCMAKE_BUILD_TYPE=Release \
-DLLVM_ENABLE_ASSERTIONS=ON ../llvm \
-DLLVM_ENABLE_PROJECTS="mlir;llvm;clang;clang-tools-extra;lld;compiler-rt;openmp" \
-DLLVM_TARGETS_TO_BUILD="NVPTX;AMDGPU;AArch64" \
-DCMAKE_INSTALL_PREFIX=../install \
-DMLIR_ENABLE_BINDINGS_PYTHON=ON \
-DPython3_EXECUTABLE=$(which python3) \
-DMLIR_INCLUDE_INTEGRATION_TESTS=ON \
-DLLVM_ENABLE_RTTI=ON \
-DBUILD_SHARED_LIBS=OFF \
-DLLVM_DEFAULT_TARGET_TRIPLE="aarch64-openEuler-linux"
# 编译安装
ninja
ninja install
```

#### 3.2 Triton-CPU源码构建安装

共两步：**克隆Triton-CPU → Triton-CPU 源码构建安装**。

**Step 1 — 克隆Triton-CPU**
> 注：源码中存在文件是由lfs管理 && 子模块需要用git submodule update更新
```shell
cd ${YOUR_CODE_DIR}
git lfs clone https://atomgit.com/openeuler/triton-cpu.git
cd triton-cpu
git lfs install
git lfs pull
git submodule init
git submodule update
rm -rf FlagGems #构建Triton-CPU源码，不需要FlagGems源码
```

**Step 2 — Triton-CPU 源码构建安装**
首次构建需要下载nvidia工具链(416MB)和nlohmann/json源码库(2MB)，以及googletest的源码(14MB)
```shell
cd ${YOUR_CODE_DIR}/triton-cpu
source ~/venv-triton-cpu/bin/activate
export LLVM_INSTALL_DIR=$YOUR_CODE_DIR/llvm-project/install
export LLVM_INCLUDE_DIRS=$LLVM_INSTALL_DIR/include
export LLVM_LIBRARY_DIR=$LLVM_INSTALL_DIR/lib
export LLVM_SYSPATH=$LLVM_INSTALL_DIR
export PYTHONPATH=$LLVM_INSTALL_DIR/python_packages/mlir_core
export PATH=$LLVM_INSTALL_DIR/bin:$PATH
export TRITON_BUILD_WITH_CLANG_LLD=true
export TRITON_PLUGIN_DIRS=$(pwd)/triton-shared
export TRITON_OFFLINE_BUILD=false
export JSON_SYSPATH=/usr/include/nlohmann/json.hpp

pip install --no-build-isolation -e python
```

### 4. 测试验证

运行Triton-CPU的TestCore验证Triton-CPU功能
**Step 1 — 安装测试依赖包**
```shell
source ~/venv-triton-cpu/bin/activate
pip install pytest -i https://mirrors.aliyun.com/pypi/simple/
pip install pytest-xdist -i https://mirrors.aliyun.com/pypi/simple/
pip install pytest-tqdm  -i https://mirrors.aliyun.com/pypi/simple/
```

**Step 2 — 环境变量配置**
```shell
cd ${YOUR_CODE_DIR}/triton-cpu
export LLVM_INSTALL_DIR=$YOUR_CODE_DIR/llvm-project/install
export LLVM_INCLUDE_DIRS=$LLVM_INSTALL_DIR/include
export LLVM_LIBRARY_DIR=$LLVM_INSTALL_DIR/lib
export LLVM_SYSPATH=$LLVM_INSTALL_DIR
export PYTHONPATH=$LLVM_INSTALL_DIR/python_packages/mlir_core
export PATH=$LLVM_INSTALL_DIR/bin:$PATH
export TRITON_BUILD_WITH_CLANG_LLD=true
export TRITON_PLUGIN_DIRS=$(pwd)/triton-shared
export TRITON_DISABLE_LINE_INFO=1
export TRITON_USE_SHARED_BACKEND=1
export LLVM_BINARY_DIR=$YOUR_CODE_DIR/llvm-project/install/bin
export TRITON_SHARED_OPT_PATH=$YOUR_CODE_DIR/triton-cpu/python/build/cmake.linux-aarch64-cpython-3.11/third_party/triton_shared/tools/triton-shared-opt/triton-shared-opt
```

**Step 3 - 运行测试**
```shell
cd ${YOUR_CODE_DIR}/triton-cpu
source ~/venv-triton-cpu/bin/activate
rm -rf ~/.triton/cache && python3 -m pytest -v -n 32 --device=cpu --continue-on-collection-errors python/test/unit/language/test_core.py -m cpu --tqdm
```
预期结果: 执行的用例只有skip跳过或Success成功的，不存在失败用例
