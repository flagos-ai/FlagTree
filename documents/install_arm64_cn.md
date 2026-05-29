[[English](./install_arm64.md)|中文版]

## 💫 ARM64 CPU [cpu](/third_party/cpu/) & [tle_arm64](/third_party/tle_arm64/)

- 对应 Triton 版本 3.3，基于 LLVM **a66376b0**，aarch64 平台
- 目标平台：AArch64 Linux，支持 NEON / SVE2 + i8mm（如 Armv9-A Cortex-A720）
- ⚠️ cpu 后端的 C++ 扩展层（TritonCPU dialect + NEON/SVE2 C runtime）位于独立的
  [triton-cpu](https://github.com/flagos-ai/triton-cpu) 仓库，需先构建再软链接进 FlagTree。
  TLE 算子由 `third_party/tle_arm64` 插件以 `create_cpu_*` builder 方法注入。
- 暂未提供 docker 镜像，请按下述源码方式安装。

### 1. 构建及运行环境

#### 1.1 系统依赖

```shell
sudo apt-get update && sudo apt-get install -y \
    build-essential cmake ninja-build git ccache pkg-config \
    libomp-dev libjemalloc2 zlib1g zlib1g-dev libxml2 libxml2-dev nlohmann-json3-dev \
    ca-certificates curl wget numactl python3-dev python3-pip python3-venv
```

#### 1.2 创建虚拟环境并安装 PyTorch

```shell
python3 -m venv ~/venv-flagtree
source ~/venv-flagtree/bin/activate
pip install --upgrade pip setuptools wheel
# 先装 PyTorch（aarch64 CPU 版）
pip install torch==2.10.0+cpu --index-url https://download.pytorch.org/whl/cpu
```

#### 1.3 手动下载 LLVM 依赖包

如果网络可访问 `oaitriton.blob.core.windows.net`，LLVM 工具链会在首次构建（步骤 2.2 Step 1）
时按 `cmake/llvm-hash.txt`（a66376b0）自动拉取并缓存到 `~/.triton/llvm/`，**无需手动操作**。

网络受限时手动下载（注意是 **arm64** 包，对应 Triton 3.3）：

```shell
mkdir -p ~/.triton/llvm && cd ~/.triton/llvm
wget https://oaitriton.blob.core.windows.net/public/llvm-builds/llvm-a66376b0-ubuntu-arm64.tar.gz
tar zxvf llvm-a66376b0-ubuntu-arm64.tar.gz
export LLVM_SYSPATH=~/.triton/llvm/llvm-a66376b0-ubuntu-arm64
export LLVM_INCLUDE_DIRS=$LLVM_SYSPATH/include
export LLVM_LIBRARY_DIR=$LLVM_SYSPATH/lib
```

### 2. 安装命令

#### 2.1 免源码安装

⚠️ ARM64 cpu 后端暂无预编译 wheel，请按下方 2.2 从源码构建。

#### 2.2 从源码构建

cpu 后端依赖独立的 triton-cpu 仓库（C++ 扩展基座），共三步：**构建 triton-cpu → 软链接进
FlagTree → 构建 FlagTree**。

> 注：下文用**合入后**的仓库/分支（flagos-ai）。合入前自测时，把 `flagos-ai/triton-cpu`(main) 与
> `flagos-ai/FlagTree`(`triton_v3.3.x`) 替换为各自未合入的 PR 仓库/分支即可。

**Step 1 — 构建 triton-cpu（C++ 扩展基座）**

```shell
cd ${YOUR_CODE_DIR}
git clone https://github.com/flagos-ai/triton-cpu.git
cd triton-cpu          # main 分支（含 a66376b0 ARM64 后端）
cd python
TRITON_BUILD_BACKENDS=cpu TRITON_OFFLINE_BUILD=1 TRITON_BUILD_PROTON=OFF \
    MAX_JOBS=$(nproc) pip install -e . --no-build-isolation --no-deps -v
cd ../..
```

首次构建会自动下载并缓存 LLVM a66376b0（约 4.4 GB）到 `~/.triton/llvm/`；ARM64 硬件上约 30–60 分钟。

**Step 2 — 软链接 triton-cpu 到 FlagTree**

克隆 FlagTree 并切到带 cpu 后端的 **`triton_v3.3.x`** 分支（软链接必须建在正确分支的树里）。
cpu 后端的 C++ 源码、TritonCPU dialect 头文件、NEON/SVE2 运行时与 TLE Python builtins 都驻留在
triton-cpu，需软链接进 FlagTree 树内：

```shell
cd ${YOUR_CODE_DIR}
git clone https://github.com/flagos-ai/FlagTree.git
cd FlagTree
git checkout -b triton_v3.3.x origin/triton_v3.3.x   # flagos-ai/FlagTree 的 3.3.x 分支（含 cpu 后端）
TRITON_CPU=$(realpath ../triton-cpu)

# TritonCPU MLIR dialect 头文件 + 实现
ln -sf $TRITON_CPU/include/triton/Dialect/TritonCPU  include/triton/Dialect/TritonCPU
ln -sf $TRITON_CPU/lib/Dialect/TritonCPU             lib/Dialect/TritonCPU

# cpu 后端 C++ 源 + 运行时 + sleef
ln -sf $TRITON_CPU/third_party/cpu/CMakeLists.txt    third_party/cpu/CMakeLists.txt
ln -sf $TRITON_CPU/third_party/cpu/include           third_party/cpu/include
ln -sf $TRITON_CPU/third_party/cpu/lib               third_party/cpu/lib
ln -sf $TRITON_CPU/third_party/cpu/runtime           third_party/cpu/runtime
ln -sf $TRITON_CPU/third_party/cpu/triton_cpu.cc     third_party/cpu/triton_cpu.cc
ln -sf $TRITON_CPU/third_party/sleef                 third_party/sleef

# TLE Python builtins（tle_ops.py 调用 create_cpu_*）
ln -sf $TRITON_CPU/third_party/cpu/language/cpu/neon.py     third_party/cpu/language/cpu/neon.py
ln -sf $TRITON_CPU/third_party/cpu/language/cpu/runtime.py  third_party/cpu/language/cpu/runtime.py
ln -sf $TRITON_CPU/third_party/cpu/language/cpu/tle_ops.py  third_party/cpu/language/cpu/tle_ops.py

# 让 triton.language.extra.cpu 可导入
ln -sf $(realpath third_party/cpu/language/cpu)  python/triton/language/extra/cpu
```

**Step 3 — 构建 FlagTree（cpu 后端）**

```shell
cd ${YOUR_CODE_DIR}/FlagTree/python
FLAGTREE_BACKEND=cpu \
LLVM_SYSPATH=$(ls -d ~/.triton/llvm/llvm-a66376b0-ubuntu-arm64) \
TRITON_OFFLINE_BUILD=1 TRITON_BUILD_PROTON=OFF MAX_JOBS=$(nproc) \
    pip install -e . --no-build-isolation -v
```

> 若之后要构建其他后端，请先清理 LLVM 相关环境变量：
> `unset LLVM_SYSPATH LLVM_INCLUDE_DIRS LLVM_LIBRARY_DIR FLAGTREE_BACKEND`

### 3. 测试验证

确认 cpu 后端已注册、且 tle_arm64 插件注入的 `create_cpu_*` TLE builder 方法可见：

```python
import triton
from triton.backends import backends
print(f"triton {triton.__version__}, cpu backend: {'cpu' in backends}")

import triton._C.libtriton as lt
b = lt.ir.builder(lt.ir.context())
cpu_ops = sorted(m for m in dir(b) if m.startswith("create_cpu_"))
print(f"TLE ARM64 ops ({len(cpu_ops)}): {cpu_ops}")
```

预期输出：

```
triton 3.3.0, cpu backend: True
TLE ARM64 ops (10): ['create_cpu_flash_attn_decode', 'create_cpu_fused_decode_step',
 'create_cpu_fused_mlp', 'create_cpu_fused_transformer_layer', 'create_cpu_neon_sdot',
 'create_cpu_rms_norm', 'create_cpu_sdot_gemv', 'create_cpu_sdot_gemv_fused_bf16',
 'create_cpu_sdot_pack_weights', 'create_cpu_swiglu']
```

端到端跑一个 TLE 算子（@triton.jit → `create_cpu_rms_norm` → TritonCPU dialect → NEON/SVE2 C runtime）：

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

### Q: big.LITTLE SoC 上性能只有一半？

A: 在大小核异构 SoC（如 Cortex-A720 大核 + A520 小核）上，务必用 `taskset` **只绑大核**——小核进入
OMP 线程池会卡在 barrier，整体掉约 2 倍。并把调频设为 performance：

```shell
for c in $(seq 0 $(($(nproc)-1))); do
    echo performance | sudo tee /sys/devices/system/cpu/cpu$c/cpufreq/scaling_governor >/dev/null
done
# 仅绑大核（核号按你的 SoC 拓扑调整）
taskset -c 0,1,6,7,8,9,10,11 python your_inference.py ...
```

### Q: 运行时报 version GLIBC / GLIBCXX not found？

A: 查询环境支持的版本，必要时 LD_PRELOAD（路径用 aarch64）：

```shell
strings /lib/aarch64-linux-gnu/libc.so.6 | grep GLIBC
strings /usr/lib/aarch64-linux-gnu/libstdc++.so.6 | grep GLIBCXX
export LD_PRELOAD=/lib/aarch64-linux-gnu/libc.so.6           # 找不到 GLIBC 时
export LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libstdc++.so.6  # 找不到 GLIBCXX 时
```

### Q: `import triton.language.extra.cpu.tle_ops` 报 No module named ...？

A: Step 2 的软链接没建全——清空环境下最容易漏 `third_party/cpu/language/cpu/{neon,runtime,tle_ops}.py`
和 `python/triton/language/extra/cpu`。确认下列均指向 triton-cpu：`include/triton/Dialect/TritonCPU`、
`lib/Dialect/TritonCPU`、`third_party/cpu/{include,lib,runtime,triton_cpu.cc}`、
`third_party/cpu/language/cpu/{neon,runtime,tle_ops}.py`、`python/triton/language/extra/cpu`、
`third_party/sleef`。
