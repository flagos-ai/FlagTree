[[中文版](./install_arm64_cn.md)|English]

## 💫 ARM64 CPU [cpu](/third_party/cpu/) & [tle_arm64](/third_party/tle_arm64/)

- Triton version 3.3, based on LLVM **a66376b0**, aarch64 platform
- Target: AArch64 Linux with NEON / SVE2 + i8mm (e.g. Armv9-A Cortex-A720)
- ⚠️ The cpu backend's C++ extension layer (TritonCPU dialect + NEON/SVE2 C runtime) lives in the
  separate [triton-cpu](https://github.com/flagos-ai/triton-cpu) repo and must be built first, then
  symlinked into FlagTree. TLE ops are injected by the `third_party/tle_arm64` plugin as `create_cpu_*`
  builder methods.
- No docker image is provided yet; install from source as below.

### 1. Environment for build and run

#### 1.1 System dependencies

```shell
sudo apt-get update && sudo apt-get install -y \
    build-essential cmake ninja-build git ccache pkg-config \
    libomp-dev libjemalloc2 zlib1g zlib1g-dev libxml2 libxml2-dev nlohmann-json3-dev \
    ca-certificates curl wget numactl python3-dev python3-pip python3-venv
```

#### 1.2 Create a virtualenv and install PyTorch

```shell
python3 -m venv ~/venv-flagtree
source ~/venv-flagtree/bin/activate
pip install --upgrade pip setuptools wheel
# Install PyTorch first (aarch64 CPU build)
pip install torch==2.10.0+cpu --index-url https://download.pytorch.org/whl/cpu
```

#### 1.3 Manually download the LLVM

If `oaitriton.blob.core.windows.net` is reachable, the LLVM toolchain is fetched automatically on the
first build (step 2.2 Step 1) according to `cmake/llvm-hash.txt` (a66376b0) and cached under
`~/.triton/llvm/` — **no manual step needed**.

For restricted networks, download manually (note the **arm64** package, for Triton 3.3):

```shell
mkdir -p ~/.triton/llvm && cd ~/.triton/llvm
wget https://oaitriton.blob.core.windows.net/public/llvm-builds/llvm-a66376b0-ubuntu-arm64.tar.gz
tar zxvf llvm-a66376b0-ubuntu-arm64.tar.gz
export LLVM_SYSPATH=~/.triton/llvm/llvm-a66376b0-ubuntu-arm64
export LLVM_INCLUDE_DIRS=$LLVM_SYSPATH/include
export LLVM_LIBRARY_DIR=$LLVM_SYSPATH/lib
```

### 2. Installation Commands

#### 2.1 Source-free Installation

⚠️ There is no prebuilt wheel for the ARM64 cpu backend yet; build from source per 2.2 below.

#### 2.2 Build from Source

The cpu backend depends on the separate triton-cpu repo (C++ extension base). Three steps:
**build triton-cpu → symlink into FlagTree → build FlagTree**.

> Note: the commands below use the **post-merge** repos/branches (flagos-ai). To self-test before
> merge, substitute your unmerged PR repo/branch for `flagos-ai/triton-cpu` (main) and
> `flagos-ai/FlagTree` (`triton_v3.3.x`).

**Step 1 — Build triton-cpu (C++ extension base)**

```shell
cd ${YOUR_CODE_DIR}
git clone https://github.com/flagos-ai/triton-cpu.git
cd triton-cpu          # main branch (carries the a66376b0 ARM64 backend)
cd python
TRITON_BUILD_BACKENDS=cpu TRITON_OFFLINE_BUILD=1 TRITON_BUILD_PROTON=OFF \
    MAX_JOBS=$(nproc) pip install -e . --no-build-isolation --no-deps -v
cd ../..
```

The first build auto-downloads and caches LLVM a66376b0 (~4.4 GB) under `~/.triton/llvm/`; ~30–60 min
on ARM64 hardware.

**Step 2 — Symlink triton-cpu into FlagTree**

Clone FlagTree and check out the **`triton_v3.3.x`** branch that carries the cpu backend (the symlinks
must land in the correct branch's tree). The cpu backend's C++ sources, TritonCPU dialect headers,
NEON/SVE2 runtime and TLE Python builtins all live in triton-cpu and must be symlinked into the
FlagTree tree:

```shell
cd ${YOUR_CODE_DIR}
git clone https://github.com/flagos-ai/FlagTree.git
cd FlagTree
git checkout -b triton_v3.3.x origin/triton_v3.3.x   # flagos-ai/FlagTree's 3.3.x branch (cpu backend)
TRITON_CPU=$(realpath ../triton-cpu)

# TritonCPU MLIR dialect headers + impl
ln -sf $TRITON_CPU/include/triton/Dialect/TritonCPU  include/triton/Dialect/TritonCPU
ln -sf $TRITON_CPU/lib/Dialect/TritonCPU             lib/Dialect/TritonCPU

# cpu backend C++ sources + runtime + sleef
ln -sf $TRITON_CPU/third_party/cpu/CMakeLists.txt    third_party/cpu/CMakeLists.txt
ln -sf $TRITON_CPU/third_party/cpu/include           third_party/cpu/include
ln -sf $TRITON_CPU/third_party/cpu/lib               third_party/cpu/lib
ln -sf $TRITON_CPU/third_party/cpu/runtime           third_party/cpu/runtime
ln -sf $TRITON_CPU/third_party/cpu/triton_cpu.cc     third_party/cpu/triton_cpu.cc
ln -sf $TRITON_CPU/third_party/sleef                 third_party/sleef

# TLE Python builtins (tle_ops.py calls create_cpu_*)
ln -sf $TRITON_CPU/third_party/cpu/language/cpu/neon.py     third_party/cpu/language/cpu/neon.py
ln -sf $TRITON_CPU/third_party/cpu/language/cpu/runtime.py  third_party/cpu/language/cpu/runtime.py
ln -sf $TRITON_CPU/third_party/cpu/language/cpu/tle_ops.py  third_party/cpu/language/cpu/tle_ops.py

# Make triton.language.extra.cpu importable
ln -sf $(realpath third_party/cpu/language/cpu)  python/triton/language/extra/cpu
```

**Step 3 — Build FlagTree (cpu backend)**

```shell
cd ${YOUR_CODE_DIR}/FlagTree/python
FLAGTREE_BACKEND=cpu \
LLVM_SYSPATH=$(ls -d ~/.triton/llvm/llvm-a66376b0-ubuntu-arm64) \
TRITON_OFFLINE_BUILD=1 TRITON_BUILD_PROTON=OFF MAX_JOBS=$(nproc) \
    pip install -e . --no-build-isolation -v
```

> If you need to build other backends afterward, clear the LLVM-related environment variables first:
> `unset LLVM_SYSPATH LLVM_INCLUDE_DIRS LLVM_LIBRARY_DIR FLAGTREE_BACKEND`

### 3. Testing and validation

Confirm the cpu backend is registered and the `create_cpu_*` TLE builder methods injected by the
tle_arm64 plugin are visible:

```python
import triton
from triton.backends import backends
print(f"triton {triton.__version__}, cpu backend: {'cpu' in backends}")

import triton._C.libtriton as lt
b = lt.ir.builder(lt.ir.context())
cpu_ops = sorted(m for m in dir(b) if m.startswith("create_cpu_"))
print(f"TLE ARM64 ops ({len(cpu_ops)}): {cpu_ops}")
```

Expected output:

```
triton 3.3.0, cpu backend: True
TLE ARM64 ops (10): ['create_cpu_flash_attn_decode', 'create_cpu_fused_decode_step',
 'create_cpu_fused_mlp', 'create_cpu_fused_transformer_layer', 'create_cpu_neon_sdot',
 'create_cpu_rms_norm', 'create_cpu_sdot_gemv', 'create_cpu_sdot_gemv_fused_bf16',
 'create_cpu_sdot_pack_weights', 'create_cpu_swiglu']
```

Run a TLE op end to end (@triton.jit → `create_cpu_rms_norm` → TritonCPU dialect → NEON/SVE2 C runtime):

```python
import torch, triton, triton.language as tl
from triton.language.extra.cpu import tle_ops as tle_cpu

@triton.jit
def rms_kernel(x_ptr, w_ptr, out_ptr, D: tl.constexpr, eps: tl.constexpr):
    tle_cpu.rms_norm(x_ptr, w_ptr, out_ptr, D, eps)

D = 128
x = torch.randn(D, dtype=torch.bfloat16)
w = torch.randn(D, dtype=torch.bfloat16)
out = torch.empty(D, dtype=torch.bfloat16)
rms_kernel[(1,)](x, w, out, D, 1e-6)

ref = (x.float() / torch.sqrt((x.float()**2).mean() + 1e-6)) * w.float()
print("max err:", (out.float() - ref).abs().max().item(), "-> OK")
```

## Q&A

#### Q: Performance is only half on a big.LITTLE SoC?

A: On heterogeneous SoCs (e.g. Cortex-A720 big + A520 little), always pin to the **big cores only**
with `taskset` — little cores entering the OMP thread pool stall on the barrier and cost ~2x overall.
Also set the governor to performance:

```shell
for c in $(seq 0 $(($(nproc)-1))); do
    echo performance | sudo tee /sys/devices/system/cpu/cpu$c/cpufreq/scaling_governor >/dev/null
done
# Pin big cores only (adjust core ids to your SoC topology)
taskset -c 0,1,6,7,8,9,10,11 python your_inference.py ...
```

#### Q: Runtime reports version GLIBC / GLIBCXX not found?

A: Check the versions supported by your environment and LD_PRELOAD if needed (aarch64 paths):

```shell
strings /lib/aarch64-linux-gnu/libc.so.6 | grep GLIBC
strings /usr/lib/aarch64-linux-gnu/libstdc++.so.6 | grep GLIBCXX
export LD_PRELOAD=/lib/aarch64-linux-gnu/libc.so.6           # if GLIBC not found
export LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libstdc++.so.6  # if GLIBCXX not found
```

#### Q: `import triton.language.extra.cpu.tle_ops` fails with "No module named ..."?

A: The Step 2 symlinks are incomplete — in a clean environment the most easily missed are
`third_party/cpu/language/cpu/{neon,runtime,tle_ops}.py` and `python/triton/language/extra/cpu`.
Verify all of these point into triton-cpu:
`include/triton/Dialect/TritonCPU`, `lib/Dialect/TritonCPU`,
`third_party/cpu/{include,lib,runtime,triton_cpu.cc}`,
`third_party/cpu/language/cpu/{neon,runtime,tle_ops}.py`, `python/triton/language/extra/cpu`,
`third_party/sleef`.
