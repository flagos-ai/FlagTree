# Flagtree Third Party Backend - RPU Accelerator Support

## Overview

Flagtree Third Party Backend for the **Rhino RPU** accelerator, providing the Triton compiler backend and the test suite for developing and deploying Triton kernels on RPU hardware.

The Rhino RPU is an AI accelerator from **Huixi Intelligence** (辉羲智能, [rhino.auto](https://www.rhino.auto/)), a company building high-performance, low-power compute platforms and chips for embodied AI and autonomous driving — such as its publicly announced 光至 R1 SoC.

> 中文版: [README_cn.md](./README_cn.md)

## Prerequisites

- aarch64 Linux board with the RPU device node (`/dev/rpu`) and kernel module
- RPU LLVM toolchain (custom LLVM providing `bin/clang`)
- RPU launch runtime library `rhino-launch-kernel` (ships `librhino_launch.so`), plus the `launch_kernel_runner` CLI built against it (see "On-board launch_kernel check")
- Minimum 24 GB RAM and 60 GB free disk (clean build is ~18-25 minutes on a 24-core board)
- Python 3.10+, `cmake >= 3.20`, `ninja`, a recent `pip`

## Environment Preparation

### 1. Pull Source Code

```bash
cd ~
git clone https://github.com/flagos-ai/FlagTree.git
cd FlagTree
git checkout -b triton_v3.6.x origin/triton_v3.6.x
```

### 2. Install RPU Driver and Runtime

Install the RPU kernel driver (provides `/dev/rpu`) and the `rhino-launch-kernel` runtime **library** (`librhino_launch.so`, used by the on-board launch_kernel test) per vendor instructions. Verify the device node:

```bash
ls /dev/rpu
```

### 3. Prepare RPU LLVM Toolchain

The RPU backend uses a custom LLVM as its `.rpubin` emitter. Point
`RPU_LLVM_ROOT` at the toolchain install prefix (the directory containing
`bin/clang`):

```bash
# Toolchain layout:
#   $RPU_LLVM_ROOT/bin/clang
#   $RPU_LLVM_ROOT/lib/...
export RPU_LLVM_ROOT=/opt/rpu/llvm
```

Contact the vendor to obtain the toolchain.

## Build and Install

### 1. Configure Build Environment

```bash
export FLAGTREE_BACKEND=rpu
export MAX_JOBS=8                # tune to available RAM
```

### 2. Install Python Dependencies

```bash
cd ~/FlagTree/python
pip3 install -r requirements.txt
```

### 3. Build and Install Package

```bash
cd ~/FlagTree

# Initial build
pip3 install . --no-build-isolation -v

# Rebuild after source changes
pip3 install . --no-build-isolation --force-reinstall -v
```

The Triton MLIR LLVM is downloaded automatically from the public oaitriton blob the first time setup runs; no manual action needed.

## Test Validation

### Unit tests

The compile tests drive the real toolchain, so set `RPU_LLVM_ROOT` first (the
directory containing `bin/clang`). If it is unset or its path is missing, the
tests stop with a clear message naming the variable.

```bash
cd ~/FlagTree
export RPU_LLVM_ROOT=/opt/rpu/llvm
pytest -s third_party/rpu/python/test/unit
```

The suite covers the compile pipeline and per-op contracts.

### On-board launch_kernel check

This needs the `launch_kernel_runner` CLI, a thin front-end over the
`rhino-launch-kernel` runtime library. Build it together with the backend by
enabling `RPU_BUILD_LAUNCH_RUNNER` (off by default; it needs the
`rhino-launch-kernel` install) on the package build, then run the smoke test on
a board with `/dev/rpu`:

```bash
TRITON_APPEND_CMAKE_ARGS="-DRPU_BUILD_LAUNCH_RUNNER=ON -DCMAKE_PREFIX_PATH=/path/to/rhino-launch-kernel/install" \
    pip3 install . --no-build-isolation --force-reinstall -v
export RPU_LK_RUNNER=$PWD/third_party/rpu/tools/launch_runner/launch_kernel_runner
python3 third_party/rpu/python/test/board/lk_board_smoke.py --require-board
```

It compiles a small kernel, dispatches it on the device, and compares the
result to a numpy golden. Without `--require-board` it skips cleanly when the
toolchain / runtime / device is unavailable. (`tools/launch_runner` can also be
built standalone — see [its README](./tools/launch_runner).)
