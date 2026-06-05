[[中文版](./install_cpu_cn.md)|English]

## 💫 ARM64 CPU [cpu](/third_party/cpu/) & [tle_arm64](/third_party/tle_arm64/)

- Triton version 3.3, based on LLVM **a66376b0**, aarch64 platform
- Target: AArch64 Linux with NEON / SVE2 + i8mm (e.g. Armv9-A Cortex-A720)
- ⚠️ The cpu backend's C++ extension layer (TritonCPU dialect + NEON/SVE2 C runtime) lives in the
  separate [flagtree-cpu](https://github.com/flagos-ai/flagtree-cpu) repo. The helper script
  `python/scripts/link_flagtree_cpu.sh` (run from the FlagTree root) clones it into
  `third_party/triton-cpu/` and creates the symlinks the build expects. TLE ops are injected by
  the `third_party/tle_arm64` plugin as `create_cpu_*` builder methods.
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
# Build dependencies — with --no-build-isolation (step 2.2 Step 3), pyproject
# build requirements are NOT installed automatically and must be present:
pip install pybind11
# Install PyTorch first (aarch64 CPU build)
pip install torch==2.10.0+cpu --index-url https://download.pytorch.org/whl/cpu
```

#### 1.3 LLVM toolchain

If `oaitriton.blob.core.windows.net` is reachable, the LLVM toolchain is fetched automatically on the
first build (step 2.2 Step 3) according to `cmake/llvm-hash.txt` (a66376b0) and cached under
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

Three steps: **clone FlagTree → wire up flagtree-cpu via the helper script → build FlagTree**.

> Note: the commands below use the **post-merge** repos/branches (flagos-ai). To self-test before
> merge, substitute your unmerged PR repo/branch for `flagos-ai/flagtree-cpu` (main) and
> `flagos-ai/FlagTree` (`triton_v3.3.x`). The helper script accepts `--flagtree-cpu-url` and
> `--flagtree-cpu-ref` to override its defaults.

**Step 1 — Clone FlagTree and check out the cpu-backend branch**

```shell
cd ${YOUR_CODE_DIR}
git clone https://github.com/flagos-ai/FlagTree.git
cd FlagTree
git checkout -b triton_v3.3.x origin/triton_v3.3.x   # FlagTree's 3.3.x branch (carries the cpu backend)
```

**Step 2 — Wire up flagtree-cpu (clone + symlinks)**

The C++ extension layer (TritonCPU MLIR dialect + NEON/SVE2 C runtime + Python TLE builtins) lives in
`flagos-ai/flagtree-cpu`. One helper script clones it into `third_party/triton-cpu/` and creates the
12 symlinks the cpu backend build expects (TritonCPU dialect headers, `third_party/cpu/*`, sleef,
the Python TLE builtins under `third_party/cpu/language/cpu/`, and
`python/triton/language/extra/cpu` so `triton.language.extra.cpu` imports work):

```shell
bash python/scripts/link_flagtree_cpu.sh
```

If you already have a local flagtree-cpu clone, reuse it instead of re-cloning:

```shell
bash python/scripts/link_flagtree_cpu.sh --flagtree-cpu-path /path/to/flagtree-cpu
```

All resulting symlinks are relative — the worktree can be moved without breaking links. Re-running
the script is safe (existing/correct symlinks are skipped; the script exits non-zero with a clear
message if any expected symlink path is occupied by a regular file).

**Step 3 — Build FlagTree (cpu backend)**

```shell
FLAGTREE_BACKEND=cpu TRITON_BUILD_PROTON=OFF MAX_JOBS=$(nproc) \
TRITON_APPEND_CMAKE_ARGS="-DCMAKE_INSTALL_PREFIX=/tmp/flagtree_install" \
    pip install -e python/ --no-build-isolation -v
```

`TRITON_APPEND_CMAKE_ARGS` redirects the `cmake --install` step: without it, the sleef
subproject tries to copy `libsleef.so` into `/usr/local/lib` and the build fails with
*Permission denied* for non-root users. The kernel-visible copy under `python/triton/_C/`
is installed either way.

If you did §1.3's manual LLVM download, the `LLVM_SYSPATH` you exported is picked up automatically;
you can additionally pass `TRITON_OFFLINE_BUILD=1` if you want to assert no network access is used.

> If you build another backend in the same shell afterward, clear the LLVM-related environment
> variables first: `unset LLVM_SYSPATH LLVM_INCLUDE_DIRS LLVM_LIBRARY_DIR FLAGTREE_BACKEND`

### 3. Testing and validation

⚠️ Before running anything that JIT-compiles kernels (the validation scripts below, and model
inference in general), export `FLAGTREE_BACKEND=cpu` — at runtime it enables the ARM `-march`
flags, OpenMP linkage and GCC-assembler compatibility handling for the emitted `kernel.s`:

```shell
export FLAGTREE_BACKEND=cpu
```

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

### Question: Performance is only half on a big.LITTLE SoC?

Answer: On heterogeneous SoCs (e.g. Cortex-A720 big + A520 little), always pin to the **big cores only**
with `taskset` — little cores entering the OMP thread pool stall on the barrier and cost ~2x overall.
Also set the governor to performance:

```shell
for c in $(seq 0 $(($(nproc)-1))); do
    echo performance | sudo tee /sys/devices/system/cpu/cpu$c/cpufreq/scaling_governor >/dev/null
done
# Pin big cores only (adjust core ids to your SoC topology)
taskset -c 0,1,6,7,8,9,10,11 python your_inference.py ...
```

### Question: Runtime reports version GLIBC / GLIBCXX not found?

Answer: Check the versions supported by your environment and LD_PRELOAD if needed (aarch64 paths):

```shell
strings /lib/aarch64-linux-gnu/libc.so.6 | grep GLIBC
strings /usr/lib/aarch64-linux-gnu/libstdc++.so.6 | grep GLIBCXX
export LD_PRELOAD=/lib/aarch64-linux-gnu/libc.so.6           # if GLIBC not found
export LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libstdc++.so.6  # if GLIBCXX not found
```

### Question: kernel compilation fails with `kernel.s: Error: junk at end of line, first unrecognized character is `"``?

Answer: `FLAGTREE_BACKEND=cpu` is not exported in the current shell. It is not only a build-time
switch — runtime JIT compilation of `kernel.s` also depends on it (it enables stripping of LLVM
`.file`/`.loc` debug directives that the GNU assembler rejects, plus the ARM `-march` flags).
Export it and re-run.

### Question: `import triton.language.extra.cpu.tle_ops` fails with "No module named ..."?

Answer: The Step 2 symlinks are incomplete. Re-run `bash python/scripts/link_flagtree_cpu.sh` — it is
idempotent and exits non-zero if any expected symlink fails to resolve.
