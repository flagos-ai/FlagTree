# How to Install FlagTree Triton Compiler on Huawei Ascend NPU: A Step-by-Step Guide

> This tutorial is based on real hands-on experience, documenting the full process of building FlagTree from source on an openEuler + Ascend910 environment — including 4 failed build attempts and how we debugged them. Hopefully this saves you some headaches.

## 1. What Is FlagTree?

FlagTree is a **multi-backend unified Triton compiler** developed by the FlagOS team, forked from triton-lang/triton. It supports multiple AI chip backends including NVIDIA, Huawei Ascend, Hygon DCU, Moore Threads, and more.

In short: **if you want to write Triton kernels on Ascend NPU, FlagTree is the compiler you need.**

Project repository: `github.com/flagos-ai/flagtree`

## 2. Environment

Here is the environment we are working with:

| Item | Version |
|------|---------|
| OS | openEuler 2203sp4, Linux 5.10.0 aarch64 |
| Python | 3.9.9 |
| Device | Ascend910_9382 (16 NPUs) |
| CANN | 8.5.0 |
| PyTorch | 2.8.0+cpu |
| torch_npu | 2.8.0.post2 |
| GCC | 10.3.1 |

> **Note**: This is an **aarch64** machine, not x86_64. This matters because many pre-built toolchains are x86-only — you need the aarch64 versions.

## 3. Installation Steps

### Step 1: Clone the FlagTree Repository

```bash
git clone https://github.com/flagos-ai/flagtree.git ~/FlagTree
cd ~/FlagTree
git submodule update --init --recursive
```

After cloning, verify that the third-party dependencies are in place:

```bash
ls third_party/ascend/AscendNPU-IR/    # Ascend NPU IR submodule
ls third_party/flir/                     # FLIR (FlagTree Linalg IR) submodule
```

If these directories are empty, the submodules were not pulled properly. Re-run `git submodule update --init`.

### Step 2: Check Build Dependencies

FlagTree requires the following build tools:

```bash
cmake --version    # >= 3.18 (we used 4.2.3)
ninja --version    # >= 1.11 (we used 1.13.0)
pip install pybind11  # >= 2.13.1
```

> If cmake or ninja are missing, `pip install cmake ninja` will do. The setup.py also auto-installs them into a temporary build environment.

### Step 3: Obtain Pre-built LLVM

This is the most critical step. FlagTree requires an LLVM toolchain with MLIR support to compile Triton.

For the Ascend backend, FlagTree provides a pre-built LLVM:

```bash
# If you have internet access, setup.py will download it automatically.
# If not, download and extract manually to ~/.flagtree/ascend/
mkdir -p ~/.flagtree/ascend
cd ~/.flagtree/ascend
wget https://baai-cp-web.ks3-cn-beijing.ksyuncs.com/trans/llvm-a66376b0-ubuntu-aarch64-python311-compat_v0.3.0.tar.gz
tar xzf llvm-a66376b0-ubuntu-aarch64-python311-compat_v0.3.0.tar.gz
```

After extraction, the directory structure looks like:

```
~/.flagtree/ascend/llvm-a66376b0-ubuntu-aarch64-python311-compat/
├── bin/          # clang, clang++, mlir-opt, etc.
├── include/      # LLVM/MLIR headers
└── lib/          # LLVM/MLIR static libraries + libstdc++.so.6.0.30
```

> **Key point**: This pre-built LLVM ships with its own `libstdc++.so.6.0.30`. You will need this later.

### Step 4: Extract Build Dependencies

FlagTree ships a pre-packaged dependency tarball in its repository:

```bash
cd ~
tar xzf ~/FlagTree/build-deps-triton_3.2.x-linux-aarch64.tar.gz
```

This extracts googletest, the JSON library, and other build dependencies into `~/.triton/`.

### Step 5: Set Environment Variables (Critical!)

This step is **the most error-prone part** of the entire process. It took us 4 attempts to get right. Here are the key lessons learned.

```bash
# 1. Specify the backend as Ascend
export FLAGTREE_BACKEND=ascend

# 2. Point to the pre-built LLVM
export LLVM_SYSPATH=~/.flagtree/ascend/llvm-a66376b0-ubuntu-aarch64-python311-compat

# 3. Add LLVM's bin to PATH (CMake needs to find clang/clang++)
export PATH=$LLVM_SYSPATH/bin:$PATH

# 4. [CRITICAL] Add LLVM's lib to the linker search path
# The pre-built LLVM's static libraries require GLIBCXX_3.4.30,
# but the system GCC 10 only provides GLIBCXX_3.4.28.
# The LLVM bundle includes libstdc++.so.6.0.30 — the linker must find it.
export LIBRARY_PATH=$LLVM_SYSPATH/lib:${LIBRARY_PATH:-}
export LD_LIBRARY_PATH=$LLVM_SYSPATH/lib:${LD_LIBRARY_PATH:-}

# 5. Offline build (optional — prevents downloads during build)
export TRITON_OFFLINE_BUILD=1

# 6. Disable Proton (profiling tool, not needed for Ascend)
export TRITON_BUILD_PROTON=OFF

# 7. [CRITICAL] Append extra CMake arguments to fix two build issues:
#    - Disable -Werror (LLVM headers trigger dangling-assignment-gsl warnings)
#    - Tell the linker to search LLVM's lib directory
export TRITON_APPEND_CMAKE_ARGS="-DLLVM_ENABLE_WERROR=OFF \
  -DCMAKE_CXX_FLAGS=-Wno-error=dangling-assignment-gsl \
  -DCMAKE_EXE_LINKER_FLAGS=-L$LLVM_SYSPATH/lib \
  -DCMAKE_SHARED_LINKER_FLAGS=-L$LLVM_SYSPATH/lib"

# 8. Limit parallel jobs (aarch64 machines have many cores but may lack memory)
export MAX_JOBS=16
```

### Step 6: Build and Install

Everything is ready. Start the build:

```bash
cd ~/FlagTree/python

# Clean any previously failed build artifacts
rm -rf build/

# Install in editable mode (convenient for development and debugging)
pip install -e . -v 2>&1 | tee ~/flagtree_build.log
```

The build takes approximately **10-20 minutes** (depending on `MAX_JOBS` and machine performance).

If everything goes well, you should see:

```
Successfully installed flagtree-0.5.0+gitXXXXXXX
```

### Step 7: Verify the Installation

```python
import triton
print(triton.__version__)   # 3.2.0
print(triton.__file__)      # Should point to ~/FlagTree/python/triton/__init__.py
```

Check that the Ascend backend is available:

```python
from triton.backends.ascend import driver as ascend_driver
print("Ascend backend loaded!")
```

Run a simple kernel test:

```python
import triton
import triton.language as tl
import torch
import torch_npu

@triton.jit
def add_kernel(x_ptr, y_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    output = x + y
    tl.store(output_ptr + offsets, output, mask=mask)

n = 1024
x = torch.randn(n, dtype=torch.float32).to('npu')
y = torch.randn(n, dtype=torch.float32).to('npu')
out = torch.empty_like(x)

add_kernel[(1,)](x, y, out, n, BLOCK_SIZE=1024)

# Verify results on CPU (avoids CANN OPP issues)
diff = torch.max(torch.abs(out.cpu() - (x.cpu() + y.cpu()))).item()
print(f"Max diff: {diff}")  # Should be 0.0
```

## 4. Troubleshooting: Lessons from 4 Failed Builds

If you follow the steps above exactly, you should succeed on the first try. But if you are curious why those "weird" environment variables are necessary, here is the record of our 4 failed attempts:

### Pitfall 1: clang Not Found

**Symptom**: CMake error — `CMAKE_C_COMPILER: clang is not a full path and was not found in the PATH`

**Root cause**: `LLVM_SYSPATH` was set, but the LLVM `bin` directory was not added to `PATH`. The CMakeLists.txt hardcodes `set(CMAKE_C_COMPILER clang)`, which only searches by name in `PATH`.

**Fix**: `export PATH=$LLVM_SYSPATH/bin:$PATH`

### Pitfall 2: -Werror Causes Compilation Failure

**Symptom**:
```
mlir/IR/OperationSupport.h:1000:27: error: object backing the pointer
will be destroyed [-Werror,-Wdangling-assignment-gsl]
```

**Root cause**: FlagTree builds with `-Werror` by default (all warnings treated as errors). However, the pre-built LLVM headers trigger a `dangling-assignment-gsl` warning introduced in clang-21. This is not a code bug — the compiler simply became stricter.

**Fix**: Append `-Wno-error=dangling-assignment-gsl` via `TRITON_APPEND_CMAKE_ARGS`.

### Pitfall 3: Linker Error — undefined reference to std::__throw_bad_array_new_length

**Symptom**:
```
undefined reference to `std::__throw_bad_array_new_length()'
```

**Root cause**: `std::__throw_bad_array_new_length` was introduced in GCC 12 / libstdc++ 12. Our system has GCC 10.3 (GLIBCXX_3.4.28), but the pre-built LLVM static libraries were compiled with GCC 12+ and require GLIBCXX_3.4.30.

**Fix**: The pre-built LLVM ships its own `libstdc++.so.6.0.30`. Set `LIBRARY_PATH` and `LD_LIBRARY_PATH` to point to the LLVM `lib` directory, and add `-L$LLVM_SYSPATH/lib` to the CMake linker flags.

### Pitfall 4: Using GCC Instead of Clang? Dead End.

**Symptom**: Setting `FLAGTREE_USE_SYSTEM_CC=1` to compile with GCC 10 results in a flood of template syntax errors.

**Root cause**: FlagTree's C++ code (especially the FLIR and AscendNPU-IR components) extensively uses clang-specific template syntax that GCC 10 cannot parse.

**Lesson**: **Do not use GCC to compile FlagTree — you must use clang.** The `FLAGTREE_USE_SYSTEM_CC` flag does not work in the Ascend aarch64 environment.

## 5. Environment Variable Quick Reference

Every time you open a new terminal, set the following variables to use FlagTree:

```bash
# Base environment (CANN + PyTorch + venv)
source /your/venv/setup_env.sh

# FlagTree runtime
export LLVM_SYSPATH=~/.flagtree/ascend/llvm-a66376b0-ubuntu-aarch64-python311-compat
export LD_LIBRARY_PATH=$LLVM_SYSPATH/lib:${LD_LIBRARY_PATH:-}
```

> We recommend creating a `setup_flagtree.sh` script to set everything up in one line.

## 6. Summary

| Step | Description | Time |
|------|-------------|------|
| Clone repo + submodules | Pull source code | ~5min |
| Install build deps | cmake, ninja, pybind11 | ~2min |
| Download pre-built LLVM | ~500MB | ~5min |
| Extract build deps | googletest, json | ~1min |
| Set environment variables | The most critical step | ~5min |
| Build and install | pip install -e . | ~15min |
| Verify | import triton + kernel test | ~2min |

**Key takeaways**:
1. You **must use clang** (from the LLVM bundle), not system GCC
2. You **must add LLVM's lib to the linker path** (libstdc++ version mismatch)
3. You **must append** `-Wno-error=dangling-assignment-gsl` (new clang-21 warning)
4. `TRITON_APPEND_CMAKE_ARGS` is your lifeline for passing extra CMake arguments

We hope this tutorial helps anyone working with Triton on Ascend NPU. Feel free to leave questions in the comments!

---

*This tutorial is based on FlagTree v0.5.0 (commit 4d9e18e), verified on Ascend910 + CANN 8.5.0 + openEuler 2203sp4 aarch64.*
