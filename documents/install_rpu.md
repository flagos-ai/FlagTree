[[中文版](./install_rpu_cn.md)|English]

## 💫 Huixi Intelligence（辉羲智能）[rpu](https://github.com/flagos-ai/FlagTree/tree/triton_v3.6.x/third_party/rpu/)

- Based on Triton 3.6, aarch64

The **Rhino RPU** is the AI accelerator inside Huixi Intelligence's 光至 R1 SoC
([rhino.auto](https://www.rhino.auto/)). Unlike the CPU/GPU backends, the RPU
backend has **no preinstalled image and no CPU simulator** — it compiles and
runs only on a physical R1 SoC board, and its driver, runtime, and toolchain are
obtained from the vendor. Build and test directly on the board.

> The RPU driver, runtime, and LLVM toolchain are not publicly distributed.
> Contact **Huixi Intelligence** ([rhino.auto](https://www.rhino.auto/)) to
> obtain them.

### 1. Build and run environment

#### 1.1 Hardware and OS

- An R1 SoC board running aarch64 Linux that exposes the RPU device node
  `/dev/rpu` (kernel module loaded).
- Minimum 24 GB RAM and 60 GB free disk.
- Python 3.10+, `cmake >= 3.20`, `ninja`, and a recent `pip`.

There is no Docker image for the RPU backend; all steps below run natively on
the R1 SoC board. Verify the device node:

```shell
ls /dev/rpu
```

#### 1.2 RPU driver and runtime (vendor-provided)

Install, per the vendor's instructions:

- the RPU kernel driver that provides `/dev/rpu`;
- the `rhino-launch-kernel` runtime library (`librhino_launch.so`), used by the
  on-board launch_kernel test.

#### 1.3 RPU LLVM toolchain (vendor-provided)

The RPU backend uses a custom LLVM as its `.rpubin` emitter. Point
`RPU_LLVM_ROOT` at the toolchain install prefix (the directory containing
`bin/clang`):

```shell
# Toolchain layout:
#   $RPU_LLVM_ROOT/bin/clang
#   $RPU_LLVM_ROOT/lib/...
export RPU_LLVM_ROOT=/opt/rpu/llvm
```

### 2. Installation Commands

The RPU backend is built from source on the board. There is no source-free
(pip wheel) installation.

#### 2.1 Pull the source code

```shell
cd ~
git clone https://github.com/flagos-ai/FlagTree.git
cd FlagTree
git checkout -b triton_v3.6.x origin/triton_v3.6.x
```

#### 2.2 Build from source

```shell
export FLAGTREE_BACKEND=rpu
export MAX_JOBS=8                        # tune to available RAM

cd ~/FlagTree/python
pip3 install -r requirements.txt         # build-time dependencies

cd ~/FlagTree
# Initial build
pip3 install . --no-build-isolation -v
# Rebuild after source changes
pip3 install . --no-build-isolation --force-reinstall -v
```

The Triton MLIR LLVM is downloaded automatically from the public oaitriton blob
the first time setup runs; no manual action is needed.

### 3. Testing and validation

#### 3.1 Unit tests

The compile tests drive the real toolchain, so set `RPU_LLVM_ROOT` first (the
directory containing `bin/clang`). If it is unset or its path is missing, the
tests stop with a clear message naming the variable.

```shell
cd ~/FlagTree
export RPU_LLVM_ROOT=/opt/rpu/llvm
pytest -s third_party/rpu/python/test/unit
```

#### 3.2 On-board launch_kernel check

This needs the `launch_kernel_runner` CLI, a thin front-end over the
`rhino-launch-kernel` runtime library. Build it together with the backend by
enabling `RPU_BUILD_LAUNCH_RUNNER` (off by default; it needs the
`rhino-launch-kernel` install), then run the smoke test on a board with
`/dev/rpu`:

```shell
TRITON_APPEND_CMAKE_ARGS="-DRPU_BUILD_LAUNCH_RUNNER=ON -DCMAKE_PREFIX_PATH=/path/to/rhino-launch-kernel/install" \
    pip3 install . --no-build-isolation --force-reinstall -v
export RPU_LK_RUNNER=$PWD/third_party/rpu/tools/launch_runner/launch_kernel_runner
python3 third_party/rpu/python/test/board/lk_board_smoke.py --require-board
```

It compiles a small kernel, dispatches it on the device, and compares the result
to a numpy golden.

Refer to [Tests of rpu backend](https://github.com/flagos-ai/FlagTree/blob/triton_v3.6.x/.github/workflows/rpu3.6-build-and-test.yml)
