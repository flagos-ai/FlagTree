[[English](./install_cpu.md)|中文版]

## 💫 ARM64 CPU [cpu](/third_party/cpu/) & [tle_arm64](/third_party/tle_arm64/)

- 对应 Triton 版本 3.3，基于 LLVM **a66376b0**，aarch64 平台
- 目标平台：AArch64 Linux，支持 NEON / SVE2 + i8mm（如 Armv9-A Cortex-A720）
- ⚠️ cpu 后端的 C++ 扩展层（TritonCPU dialect + NEON/SVE2 C runtime）位于独立的
  [flagtree-cpu](https://github.com/flagos-ai/flagtree-cpu) 仓库。辅助脚本
  `python/scripts/link_flagtree_cpu.sh`（在 FlagTree 根目录下运行）会把它克隆到
  `third_party/triton-cpu/` 并建好构建所需的软链接。TLE 算子由 `third_party/tle_arm64` 插件以
  `create_cpu_*` builder 方法注入。
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
# 构建依赖 —— 使用 --no-build-isolation（步骤 2.2 Step 3）时，pyproject 的
# build 依赖不会自动安装，必须预装：
pip install pybind11
# 先装 PyTorch（aarch64 CPU 版）
pip install torch==2.10.0+cpu --index-url https://download.pytorch.org/whl/cpu
```

#### 1.3 LLVM 工具链

如果网络可访问 `oaitriton.blob.core.windows.net`，LLVM 工具链会在首次构建（步骤 2.2 Step 3）
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

共三步：**克隆 FlagTree → 用辅助脚本接好 flagtree-cpu → 构建 FlagTree**。

> 注：下文用**合入后**的仓库/分支（flagos-ai）。合入前自测时，把 `flagos-ai/flagtree-cpu`(main) 与
> `flagos-ai/FlagTree`(`triton_v3.3.x`) 替换为各自未合入的 PR 仓库/分支即可（脚本支持
> `--flagtree-cpu-url` 与 `--flagtree-cpu-ref` 来覆盖默认值）。

**Step 1 — 克隆 FlagTree 并切到 cpu 后端分支**

```shell
cd ${YOUR_CODE_DIR}
git clone https://github.com/flagos-ai/FlagTree.git
cd FlagTree
git checkout -b triton_v3.3.x origin/triton_v3.3.x   # FlagTree 的 3.3.x 分支（含 cpu 后端）
```

**Step 2 — 接好 flagtree-cpu（克隆 + 软链接）**

C++ 扩展层（TritonCPU MLIR dialect + NEON/SVE2 C runtime + Python TLE builtins）位于
`flagos-ai/flagtree-cpu`。一行脚本完成两件事：把它克隆到 `third_party/triton-cpu/`，并建好 cpu 后端
构建所需的 12 条软链接（TritonCPU dialect 头文件、`third_party/cpu/*`、sleef、
`third_party/cpu/language/cpu/` 下的 Python TLE builtins，以及让
`triton.language.extra.cpu` 可导入的 `python/triton/language/extra/cpu`）：

```shell
bash python/scripts/link_flagtree_cpu.sh
```

如果本地已经有 flagtree-cpu clone，可以复用，免去重 clone：

```shell
bash python/scripts/link_flagtree_cpu.sh --flagtree-cpu-path /path/to/flagtree-cpu
```

所有生成的软链接均为**相对路径**——整个 worktree 可以整体移动而不会破坏链接。脚本可安全重复执行
（已正确的链接会跳过；若有非软链接的常规文件占位，脚本会以清晰错误信息退出而不静默覆盖）。

**Step 3 — 构建 FlagTree（cpu 后端）**

```shell
FLAGTREE_BACKEND=cpu TRITON_BUILD_PROTON=OFF MAX_JOBS=$(nproc) \
TRITON_APPEND_CMAKE_ARGS="-DCMAKE_INSTALL_PREFIX=/tmp/flagtree_install" \
    pip install -e python/ --no-build-isolation -v
```

`TRITON_APPEND_CMAKE_ARGS` 用于重定向 `cmake --install` 步骤：不加的话，sleef 子工程会把
`libsleef.so` 往 `/usr/local/lib` 复制，非 root 用户构建会报 *Permission denied*。kernel
实际链接的 `python/triton/_C/` 下的副本不受影响，两种方式都会安装。

如果做过 §1.3 的手动下载，你 `export` 的 `LLVM_SYSPATH` 会被自动 pick up；
若想严格断网构建，可额外传 `TRITON_OFFLINE_BUILD=1`。

> 若在同一 shell 之后要构建其他后端，请先清理 LLVM 相关环境变量：
> `unset LLVM_SYSPATH LLVM_INCLUDE_DIRS LLVM_LIBRARY_DIR FLAGTREE_BACKEND`

### 3. 测试验证

⚠️ 运行任何会 JIT 编译 kernel 的程序（包括下面的验证脚本和模型推理）前，需 export
`FLAGTREE_BACKEND=cpu` —— 运行时编译 `kernel.s` 所需的 ARM `-march` 标志、OpenMP 链接
和 GCC 汇编器兼容处理都由它启用：

```shell
export FLAGTREE_BACKEND=cpu
```

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

### 问题：big.LITTLE SoC 上性能只有一半？

回答：在大小核异构 SoC（如 Cortex-A720 大核 + A520 小核）上，务必用 `taskset` **只绑大核**——小核进入
OMP 线程池会卡在 barrier，整体掉约 2 倍。并把调频设为 performance：

```shell
for c in $(seq 0 $(($(nproc)-1))); do
    echo performance | sudo tee /sys/devices/system/cpu/cpu$c/cpufreq/scaling_governor >/dev/null
done
# 仅绑大核（核号按你的 SoC 拓扑调整）
taskset -c 0,1,6,7,8,9,10,11 python your_inference.py ...
```

### 问题：运行时报 version GLIBC / GLIBCXX not found？

回答：查询环境支持的版本，必要时 LD_PRELOAD（路径用 aarch64）：

```shell
strings /lib/aarch64-linux-gnu/libc.so.6 | grep GLIBC
strings /usr/lib/aarch64-linux-gnu/libstdc++.so.6 | grep GLIBCXX
export LD_PRELOAD=/lib/aarch64-linux-gnu/libc.so.6           # 找不到 GLIBC 时
export LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libstdc++.so.6  # 找不到 GLIBCXX 时
```

### 问题：kernel 编译报 `kernel.s: Error: junk at end of line, first unrecognized character is `"``？

回答：当前 shell 没有 export `FLAGTREE_BACKEND=cpu`。它不只是构建开关——运行时 JIT 编译
`kernel.s` 同样依赖它（启用对 GNU 汇编器不接受的 LLVM `.file`/`.loc` 调试指令的清理，以及
ARM `-march` 标志）。export 后重跑即可。

### 问题：`import triton.language.extra.cpu.tle_ops` 报 No module named ...？

回答：Step 2 的软链接没建全。重跑 `bash python/scripts/link_flagtree_cpu.sh` 即可——脚本幂等，且会在
任何应有的软链接 resolve 失败时以非零退出。
