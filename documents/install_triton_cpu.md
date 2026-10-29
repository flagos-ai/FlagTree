[[中文版](./install_triton_cpu_cn.md)|English]

## 💫 ARM64 Triton CPU

- Corresponds to OpenEuler Triton version 3.2.0, based on the OpenEuler LLVM 20.1.8 branch, on the aarch64 platform.
- Target platform: AArch64 Linux (openEuler 24.03 LTS SP4).
- No Docker image is provided yet; please install from source as described below.

### 1. Compilation pipeline overview

The diagram below summarizes how the AArch64 CPU backend compiles a Triton kernel; the bracketed names are the backend compilation stages (`ttir`, `ttsharedir`, `llir`, `obj`).

```text
Triton Python kernel (user code)
│
▼
TTIR — Triton IR (tt.* dialect)  [stage: ttir]
│   tt.load / tt.store / tt.dot / tt.reduce / ...
│
│   triton-shared conversion (triton-to-linalg-experimental; enable-armpl?)
│     · supported 2-D float tt.dot → tts.armpl_matmul/gemv → func.call @cblas_*
│     · other dots / ops           → linalg / tensor / memref / scf / arith
▼
MLIR standard dialects  [stage: ttsharedir]
│   tensor-level linalg (+ func.call @cblas_* for the dots ArmPL took over)
│   _optimize_ttsharedir: select the transform schedule and append it to the IR
│
│   the appended schedule is interpreted when entering the llir stage
▼
compiler-generated lowering (executed at the start of llir)
│   SME (aarch64 + sme + matmul): linalg_opts → tile batch matmul
│        → tile + vectorize matmul → bufferize (+ vector.contract → outerproduct)
│        → lower_to_arm_sme → arm-sme-outer-product-fusion
│   SVE (aarch64 + sve): linalg_opts → contraction tiling/pad/vectorize
│        → bufferize → legalize → loops
│   neither SME nor SVE: generic lowering (affine/linalg → loops)
▼
LLVM dialect → mlir-translate → LLVM IR
│   opt: cleanup + loop-vectorize; -mtriple/-mattr only for the SME/SVE paths
│
│   llc -O3 -filetype=obj ...
▼
AArch64 machine code (.o)  [stage: obj]
│   driver: clang++ -shared <kernel.o> <launcher.cxx> → kernel.so
▼
kernel.so (NEON / SVE / SME, by the -mattr of the selected path)
```

### 2. Build and runtime environment

#### 2.1 System dependencies
On openEuler, run:
```shell
sudo dnf install -y \
    gcc gcc-c++ make cmake ninja-build git ccache pkgconf-pkg-config \
    libomp-devel jemalloc zlib zlib-devel libxml2 libxml2-devel \
    nlohmann-json-devel ca-certificates curl wget numactl \
    python3-devel python3-pip python3-virtualenv
```

#### 2.2 Create a virtual environment and install PyTorch
Depending on your network, you can switch the pip index to the Aliyun/Tsinghua mirror.
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

### 3. Build commands

#### 3.1 Build LLVM from source
Two steps in total: **clone LLVM 20.1.8 → build and install LLVM**.

**Step 1 - Clone LLVM 20.1.8**
```shell
cd ${YOUR_CODE_DIR}
git clone https://atomgit.com/openeuler/llvm-project.git -b dev_20.1.8
```

**Step 2 - Build LLVM from source**
```shell
cd ${YOUR_CODE_DIR}/llvm-project
mkdir build && cd build
# CMake configuration
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
# Build and install
ninja
ninja install
```

#### 3.2 Build and install Triton-CPU from source

Two steps in total: **clone Triton-CPU → build and install Triton-CPU from source**.

**Step 1 — Clone Triton-CPU**
> Note: Some files in the source are managed by LFS, and submodules need to be updated with `git submodule update`.
```shell
cd ${YOUR_CODE_DIR}
git lfs clone https://atomgit.com/openeuler/triton-cpu.git
cd triton-cpu
git lfs install
git lfs pull
git submodule init
git submodule update
rm -rf FlagGems
```

**Step 2 — Build and install Triton-CPU from source**
The first build downloads the NVIDIA toolchain (416 MB), the nlohmann/json source library (2 MB), and the googletest source (14 MB).
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

### 4. Test and verification

Run Triton-CPU's TestCore to verify Triton-CPU functionality.
**Step 1 — Install test dependencies**
```shell
source ~/venv-triton-cpu/bin/activate
pip install pytest -i https://mirrors.aliyun.com/pypi/simple/
pip install pytest-xdist -i https://mirrors.aliyun.com/pypi/simple/
pip install pytest-tqdm  -i https://mirrors.aliyun.com/pypi/simple/
```

**Step 2 — Configure environment variables**
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

**Step 3 - Run the tests**
```shell
cd ${YOUR_CODE_DIR}/triton-cpu
source ~/venv-triton-cpu/bin/activate
rm -rf ~/.triton/cache && python3 -m pytest -v -n 32 --device=cpu --continue-on-collection-errors python/test/unit/language/test_core.py -m cpu --tqdm
```
Expected result: the executed test cases are either skipped or successful; no test case fails.
