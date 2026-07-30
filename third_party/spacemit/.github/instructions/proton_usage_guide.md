# Proton 性能分析工具使用指南 (K1/K3 CPU)

本文档详细介绍如何在 SpacemiT K1/K3 CPU 平台上使用 Proton 性能分析工具来分析 Triton 内核的性能。

## 目录

1. [概述](#1-概述)
2. [环境配置](#2-环境配置)
3. [基本使用方法](#3-基本使用方法)
4. [输出格式](#4-输出格式)
5. [代码示例](#5-代码示例)
6. [最佳实践](#6-最佳实践)
7. [故障排除](#7-故障排除)
8. [Kernel 外部 Proton 分析](#8-kernel-外部-proton-分析-protonscope)
9. [参考资源](#9-参考资源)
10. [更新日志](#10-更新日志)

---

## 1. 概述

### 1.1 什么是 Proton

Proton 是 Triton 生态系统中的性能分析工具，专门用于分析内核执行的性能特征。在 spine-triton 中，Proton 被适配到 CPU 平台，可以帮助开发者：

- 分析内核中各个操作的执行时间
- 识别性能瓶颈
- 优化内存访问模式（TODO）
- 调试内核执行流程 (TODO)

### 1.2 支持的平台

- **SpacemiT K1**: RISC-V 64位 CPU
- **SpacemiT K3**: RISC-V 64位 CPU
- **其他 RISC-V CPU**: 理论上支持，但未经充分测试

---

## 2. 环境配置

### 2.1 必要的导入

```python
import os
import torch
import triton
import triton.language as tl
import triton.language.extra.smt as smt
from triton.backends.spine_triton.driver import CPUDriver

# 导入 Proton 相关模块
import triton.profiler.language as pl
from triton.profiler.flags import flags
from triton.backends.spine_triton.proton import profiler

# 启用 CPU 驱动
triton.runtime.driver.set_active(CPUDriver())

# 启用 Proton 语义分析
pl.enable_semantic("triton")
flags.instrumentation_on = True
```

### 2.2 环境变量

| 环境变量 | 说明 | 示例值 |
|---------|------|--------|
| `PROTON_OUTPUT` | 指定输出文件路径和格式 | `profile.json` (Chrome Trace)<br>`profile.hatchet` (Hatchet 格式) |
| `PROTON_VERBOSE` | 启用详细输出模式 | `1` |
| `TRITON_DISABLE_PROTON` | 禁用 Proton 分析 | `1` |

### 2.3 输出格式说明

- **控制台输出**: 默认模式，直接在终端显示分析结果
- **Chrome Trace (JSON)**: 可在 `chrome://tracing` 或 `perfetto.dev` 中查看
- **Hatchet 格式**: 用于更深入的性能分析

---

## 3. 基本使用方法

### 3.1 使用上下文管理器 (推荐)

```python
# 方法1: 使用 with 语句
with profiler.profile():
    result = your_triton_kernel(input_data)
```

### 3.2 手动控制分析

```python
# 方法2: 手动开始和结束
profiler.start()
result = your_triton_kernel(input_data)
profiler.stop()
```

### 3.3 内核内部分析点

在 Triton 内核中使用 `pl.enter_scope()` 和 `pl.exit_scope()` 来标记分析区域：

```python
@triton.jit
def my_kernel(...):
    # 标记数据加载阶段
    pl.enter_scope("load_data")
    # ... 数据加载代码 ...
    pl.exit_scope("load_data")

    # 标记计算阶段
    pl.enter_scope("computation")
    # ... 计算代码 ...
    pl.exit_scope("computation")

    # 标记数据存储阶段
    pl.enter_scope("store_data")
    # ... 数据存储代码 ...
    pl.exit_scope("store_data")
```

---

## 4. 输出格式

### 4.1 控制台输出

```bash
# 默认输出到控制台
python3 your_script.py
```

输出示例：
```
[DEBUG] record called: is_start=True, scope_name=load_a, instrumentation_on=True
[DEBUG] record called: is_start=False, scope_name=load_a, instrumentation_on=True
[DEBUG] record called: is_start=True, scope_name=computation, instrumentation_on=True
[DEBUG] record called: is_start=False, scope_name=computation, instrumentation_on=True
```

### 4.2 Chrome Trace 格式

```bash
# 输出 Chrome Trace 格式
PROTON_OUTPUT=trace.json python3 your_script.py
```

然后在浏览器中打开 `chrome://tracing` 并加载 `trace.json` 文件。

### 4.3 Hatchet 格式

```bash
# 输出 Hatchet 格式
PROTON_OUTPUT=profile.hatchet python3 your_script.py
```

### 4.4 详细模式

```bash
# 启用详细输出
PROTON_VERBOSE=1 python3 your_script.py
```

---

## 5. 代码示例

### 5.1 完整的矩阵乘法示例

```python
"""
矩阵乘法 Proton 分析示例
"""
import os
import torch
import triton
import triton.language as tl
import triton.language.extra.smt as smt
from triton.backends.spine_triton.driver import CPUDriver

import triton.profiler.language as pl
from triton.profiler.flags import flags
from triton.backends.spine_triton.proton import profiler

# 配置环境
pl.enable_semantic("triton")
flags.instrumentation_on = True
triton.runtime.driver.set_active(CPUDriver())

@triton.jit
def mm_kernel_proton(a_ptr, b_ptr, c_ptr, M, N, K,
                    stride_am, stride_ak, stride_bk, stride_bn,
                    stride_cm, stride_cn,
                    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr,
                    BLOCK_SIZE_K: tl.constexpr,
                    MICRO_M: tl.constexpr, MICRO_K: tl.constexpr,
                    MICRO_N: tl.constexpr):

    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # 分析数据加载 A
    pl.enter_scope("load_a")
    a_block_ptr = tl.make_block_ptr(
        base=a_ptr,
        shape=[M, K],
        strides=[stride_am, stride_ak],
        offsets=[pid_m * BLOCK_SIZE_M, 0],
        block_shape=[BLOCK_SIZE_M, BLOCK_SIZE_K],
        order=[1, 0],
    )
    a_descriptor_load = smt.descriptor_load(a_block_ptr, (0, 0))
    a = smt.view(a_descriptor_load, (0, 0),
                 (BLOCK_SIZE_M, BLOCK_SIZE_K), (MICRO_M, MICRO_K))
    pl.exit_scope("load_a")

    # 分析数据加载 B
    pl.enter_scope("load_b")
    b_block_ptr = tl.make_block_ptr(
        base=b_ptr,
        shape=[K, N],
        strides=[stride_bk, stride_bn],
        offsets=[0, pid_n * BLOCK_SIZE_N],
        block_shape=[BLOCK_SIZE_K, BLOCK_SIZE_N],
        order=[1, 0],
    )
    b_descriptor_load = smt.descriptor_load(b_block_ptr, (0, 0))
    b = smt.view(b_descriptor_load, (0, 0),
                 (BLOCK_SIZE_K, BLOCK_SIZE_N), (MICRO_K, MICRO_N))
    pl.exit_scope("load_b")

    # 分析矩阵乘法计算
    pl.enter_scope("dot")
    accumulator = smt.dot(a, b)
    pl.exit_scope("dot")

    # 分析数据存储
    pl.enter_scope("store")
    accumulator = smt.view(accumulator, (0, 0),
                          (BLOCK_SIZE_M, BLOCK_SIZE_N), (1, 1))
    c = accumulator.to(c_ptr.dtype.element_ty)

    c_block_ptr = tl.make_block_ptr(
        base=c_ptr,
        shape=[M, N],
        strides=[stride_cm, stride_cn],
        offsets=[pid_m * BLOCK_SIZE_M, pid_n * BLOCK_SIZE_N],
        block_shape=[BLOCK_SIZE_M, BLOCK_SIZE_N],
        order=[1, 0],
    )
    tl.store(c_block_ptr, c, boundary_check=(0, 1))
    pl.exit_scope("store")

def triton_mm_with_profiling(a, b):
    """带 Proton 分析的矩阵乘法"""
    # 确保输入是连续的
    if a.stride(0) > 1 and a.stride(1) > 1:
        a = a.contiguous()
    if b.stride(0) > 1 and b.stride(1) > 1:
        b = b.contiguous()

    assert a.shape[1] == b.shape[0], "矩阵维度不匹配"
    M, K = a.shape
    _, N = b.shape
    c = torch.empty((M, N), device=a.device, dtype=a.dtype)

    def grid(META):
        return (
            triton.cdiv(M, META["BLOCK_SIZE_M"]),
            triton.cdiv(N, META["BLOCK_SIZE_N"]),
        )

    BLOCK_SIZE_K = triton.next_power_of_2(K)

    mm_kernel_proton[grid](
        a, b, c, M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        BLOCK_SIZE_M=256,
        BLOCK_SIZE_N=256,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
        MICRO_M=16,
        MICRO_N=32,
        MICRO_K=8,
    )
    return c

def main():
    # 检查输出模式
    output_file = os.environ.get("PROTON_OUTPUT", None)
    if output_file:
        print(f"分析结果将保存到: {output_file}")
        if output_file.endswith('.json'):
            print("格式: Chrome Trace (可在 chrome://tracing 中查看)")
        elif output_file.endswith('.hatchet'):
            print("格式: Hatchet")
    else:
        print("分析结果将输出到控制台")
        print("提示: 设置 PROTON_OUTPUT=<文件名>.json 可输出 Chrome Trace 格式")

    # 创建测试数据
    M, N, K = 1024, 1024, 1024
    print(f"\n测试矩阵乘法: ({M}, {K}) x ({K}, {N})")

    a = torch.randn((M, K), dtype=torch.float16)
    b = torch.randn((K, N), dtype=torch.float16)

    # 使用 Proton 分析
    print("开始 Proton 性能分析...")
    with profiler.profile():
        c_triton = triton_mm_with_profiling(a, b)

    # 验证正确性
    c_torch = torch.matmul(a, b)
    if torch.allclose(c_triton, c_torch, atol=1e-2, rtol=1e-2):
        print("✓ 计算结果正确!")
    else:
        max_diff = torch.max(torch.abs(c_triton - c_torch))
        print(f"✗ 计算结果有误! 最大差异: {max_diff}")

if __name__ == "__main__":
    main()
```

### 5.2 运行示例

```bash
# 1. 控制台输出
python3 mm_proton_example.py

# 2. Chrome Trace 输出
PROTON_OUTPUT=mm_trace.json python3 mm_proton_example.py

# 3. Hatchet 输出
PROTON_OUTPUT=mm_profile.hatchet python3 mm_proton_example.py

# 4. 详细模式
PROTON_VERBOSE=1 python3 mm_proton_example.py

# 5. 组合使用
PROTON_OUTPUT=trace.json PROTON_VERBOSE=1 python3 mm_proton_example.py
```

---

## 6. 最佳实践

### 6.1 分析点设置

1. **合理划分分析区域**: 将内核分解为逻辑上独立的部分
   - 数据加载 (`load_*`)
   - 计算操作 (`compute_*`, `dot`, `add` 等)
   - 数据存储 (`store_*`)

2. **避免过度细分**: 太多的分析点会影响性能和可读性

3. **使用描述性名称**: 分析点名称应该清楚地描述操作内容

### 6.2 性能优化指导

1. **识别瓶颈**: 通过 Proton 输出找到耗时最长的操作
2. **内存访问优化**: 关注数据加载和存储的时间
3. **计算密集度**: 比较计算时间与内存访问时间的比例

### 6.3 调试技巧

1. **逐步添加分析点**: 从粗粒度开始，逐步细化
2. **对比不同配置**: 使用不同的 block size 和 micro tile 参数
3. **结合其他工具**: 配合系统性能监控工具使用

---

## 7. 故障排除

### 7.1 常见问题

#### 问题 1: 导入错误
```
ImportError: cannot import name 'profiler' from 'triton.backends.spine_triton.proton'
```

**解决方案**:
- 确保使用正确版本的 spine-triton
- 检查 `PYTHONPATH` 是否包含构建目录
- 重新构建 spine-triton

#### 问题 2: 没有分析输出
```
# 运行后没有看到任何 Proton 输出
```

**解决方案**:
- 检查 `flags.instrumentation_on = True` 是否设置
- 确认 `pl.enable_semantic("triton")` 已调用
- 验证内核中是否有 `pl.enter_scope()` 和 `pl.exit_scope()` 调用

#### 问题 3: Chrome Trace 文件无法打开
```
# trace.json 文件生成但无法在 chrome://tracing 中打开
```

**解决方案**:
- 检查文件是否完整生成（程序是否正常结束）
- 尝试使用 `perfetto.dev` 替代 `chrome://tracing`
- 检查文件权限和路径

### 7.2 环境变量调试

```bash
# 启用详细调试信息
export PROTON_VERBOSE=1
export TRITON_ALWAYS_COMPILE=1

# 禁用 Proton (用于对比测试)
export TRITON_DISABLE_PROTON=1

# 设置 IR dump 路径 (用于调试编译问题)
export SPINE_TRITON_DUMP_PATH=./debug_ir
```

### 7.3 性能问题

如果 Proton 分析导致性能显著下降:

1. **减少分析点数量**: 只保留关键的分析区域
2. **使用采样模式**: 不是每次调用都进行分析
3. **临时禁用**: 设置 `TRITON_DISABLE_PROTON=1`

---

## 8. Kernel 外部 Proton 分析 (proton.scope)

### 8.1 概述

除了在 kernel 内部使用 `pl.enter_scope/exit_scope` 进行细粒度分析外，还可以在 kernel 外部使用 `proton.scope()` 进行 kernel 级别的性能分析。

### 8.2 GPU vs CPU 的差异

| 特性 | GPU (CUPTI) | CPU (当前实现) |
|------|-------------|----------------|
| **自动捕获 kernel** | ✅ 是 (多 kernel 时自动区分) | ❌ 否 |
| **工作原理** | CUPTI hook CUDA driver | 只测量 host 端 perf_event |
| **kernel 内部细节** | ❌ 不支持 (需要 `pl.enter_scope`) | ❌ 不支持 (需要 `pl.enter_scope`) |
| **多 kernel 区分** | ✅ 自动区分每个 kernel | ❌ 只有 scope 整体时间 |
| **代码修改** | 不需要修改 kernel 代码 | 需要手动添加 scope |

### 8.3 GPU 自动插桩原理

GPU 使用 **CUPTI Activity API** 实现自动插桩：

1. **Hook CUDA Driver**: CUPTI 会 hook `cuLaunchKernel` 等 CUDA driver 函数
2. **自动记录**: 每个 kernel 的启动时间、结束时间、grid/block 配置等都会被自动记录
3. **零代码修改**: 不需要修改 kernel 代码，只需在 host 端启用 profiling

> ⚠️ **注意**: CUPTI 也是 **kernel 级别**的测量，不能测量 kernel 内部的细节。
> 它的优势是当有多个 kernel 调用时，可以自动区分并记录每个 kernel 的时间。
> 要测量 kernel 内部操作，仍需使用 `pl.enter_scope/exit_scope`。

```python
# GPU 上使用 proton.scope() 会自动捕获内部的 kernel 调用
with proton.scope("my_computation"):
    kernel_A[grid](...)  # CUPTI 自动记录 kernel_A 的时间
    kernel_B[grid](...)  # CUPTI 自动记录 kernel_B 的时间
    # 输出会分别显示 kernel_A 和 kernel_B 的执行时间
```

### 8.4 CPU 当前实现

CPU 上 `proton.scope()` 只能测量 **host 端** 的 CPU cycles 和 instructions：

```python
import triton.profiler as proton

# 启动 Proton session (backend="cpu" 使用 CpuProfiler)
session_id = proton.start("my_profile", backend="cpu")

# 使用 scope 包裹 kernel 调用
for i in range(100):
    with proton.scope("mm_kernel_loop"):
        triton_mm_kernel[grid](a, b, c, ...)

# 结束并输出结果
proton.finalize()
```

输出示例：
```json
[
    {
        "children": [
            {
                "frame": {"name": "mm_kernel_loop", "type": "function"},
                "metrics": {"cycle": 17473435.0, "instruction": 5641682.0}
            }
        ],
        "frame": {"name": "ROOT", "type": "function"},
        "metrics": {"cycle": 0, "instruction": 0}
    }
]
```

### 8.5 完整示例：结合两种方法

```python
import os
import torch
import triton
import triton.language as tl
import triton.profiler as proton
from triton.backends.spine_triton.driver import CPUDriver

import triton.profiler.language as pl
from triton.profiler.flags import flags
from triton.backends.spine_triton.proton import profiler

# 配置环境
pl.enable_semantic("triton")
flags.instrumentation_on = True
triton.runtime.driver.set_active(CPUDriver())

@triton.jit
def my_kernel(...):
    # 方法1: kernel 内部细粒度分析
    pl.enter_scope("load")
    # ... load code ...
    pl.exit_scope("load")

    pl.enter_scope("compute")
    # ... compute code ...
    pl.exit_scope("compute")

def main():
    # Warmup (JIT compile)
    my_kernel[grid](...)

    # 方法1: Kernel 内部插桩 (细粒度)
    print("=== Method 1: Kernel 内部插桩 ===")
    with profiler.profile():
        my_kernel[grid](...)

    # 方法2: Kernel 级别 Proton (粗粒度)
    print("=== Method 2: Kernel 级别 Proton ===")
    session_id = proton.start("my_profile", backend="cpu")

    for i in range(10):
        with proton.scope("kernel_loop"):
            my_kernel[grid](...)

    proton.finalize()
```

### 8.6 CPU 自动 Kernel 捕获

> **✅ 已实现功能** (v1.2)
>
> 类似 GPU 的 CUPTI，CPU 现在支持自动 kernel 捕获，无需修改用户代码。

#### 8.6.1 启用方式

设置环境变量 `PROTON_KERNEL_CAPTURE=1` 即可启用自动 kernel 捕获。

> **⚠️ 重要**: 这是一个 **编译时** 选项。环境变量在 kernel 首次编译时读取，决定是否生成 proton hook 代码。
> - 如果不设置此环境变量，生成的代码中 **完全没有** proton 相关调用，零性能开销
> - 启用后需要清除 triton 缓存以重新编译 kernel

```bash
# 首次启用时，清除缓存以强制重新编译
rm -rf ~/.triton/cache

# 方式1: 控制台输出 (程序退出时自动打印)
PROTON_KERNEL_CAPTURE=1 python3 your_script.py

# 方式2: Chrome Trace 输出 (程序退出时自动保存)
PROTON_KERNEL_CAPTURE=1 PROTON_OUTPUT=trace.json python3 your_script.py

# 方式3: Hatchet 格式输出
PROTON_KERNEL_CAPTURE=1 PROTON_OUTPUT=profile.hatchet python3 your_script.py

# 禁用时 (默认)，无任何性能开销
python3 your_script.py
```

> **注意**:
> - 不需要在代码中使用 `proton.start()` 或 `proton.finalize()`！
> - 自动 kernel 捕获是完全自动的，程序退出时会自动输出结果
> - 只需在首次启用或切换设置时清除缓存一次，避免 `TRITON_ALWAYS_COMPILE=1` 影响性能测试

#### 8.6.2 使用示例

**完全自动，无需修改代码：**

```python
# your_script.py - 普通的 Triton 代码，不需要任何 proton 相关代码
import triton
import triton.language as tl
from triton.backends.spine_triton.driver import CPUDriver

triton.runtime.driver.set_active(CPUDriver())

@triton.jit
def my_kernel(x_ptr, y_ptr, N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N
    x = tl.load(x_ptr + offsets, mask=mask)
    tl.store(y_ptr + offsets, x * 2, mask=mask)

# 直接调用，自动记录
import torch
x = torch.randn(1024)
y = torch.empty(1024)
my_kernel[(4,)](x, y, 1024, BLOCK_SIZE=256)
```

运行：
```bash
# 首次运行前清除缓存
rm -rf ~/.triton/cache

# 控制台输出
PROTON_KERNEL_CAPTURE=1 python3 your_script.py

# Chrome Trace 输出
PROTON_KERNEL_CAPTURE=1 PROTON_OUTPUT=trace.json python3 your_script.py
```

程序退出时自动输出：
```
=== CPU Proton Profiling Results (Aggregated) ===
Scope                     Count      Total(cyc)       Min(cyc)       Max(cyc)       Avg(cyc)       Avg(us)
----------------------------------------------------------------------------------------------------------
my_kernel[4,1,1]              1         240000         240000         240000         240000         10.00
```

#### 8.6.3 实现原理

在 `spine-triton/backend/driver.py` 的 `_launch` 函数中自动添加了 hook：

```cpp
static void _launch(int gridX, int gridY, int gridZ, int64_t stream,
                    kernel_ptr_t kernel_ptr, ...) {
    // 自动记录 kernel 开始
    proton_enter_kernel(KERNEL_NAME, gridX, gridY, gridZ);

    // ... kernel 调度代码 ...

    // 自动记录 kernel 结束
    proton_exit_kernel(KERNEL_NAME, gridX, gridY, gridZ);
}
```

`CpuProtonProfiler` 在程序退出时自动调用 `dump()` 输出结果：
- 如果设置了 `PROTON_OUTPUT=xxx.json`，输出 Chrome Trace 格式
- 如果设置了 `PROTON_OUTPUT=xxx.hatchet`，输出 Hatchet 格式
- 否则输出到控制台

> **注意**: 这段代码只在 `PROTON_KERNEL_CAPTURE=1` 时编译进去，否则完全不存在。

#### 8.6.4 功能特性

| 特性 | 说明 |
|------|------|
| **自动捕获** | 无需修改 kernel 代码，自动记录所有 kernel 执行 |
| **Kernel 名称** | 自动获取 kernel 函数名 |
| **Grid 配置** | 记录 gridX, gridY, gridZ 配置 |
| **零开销** | 未启用时生成的代码中完全没有 proton 调用 (编译时决定) |
| **自动输出** | 程序退出时自动输出，无需手动调用 |
| **输出格式** | 控制台 (默认)、Chrome Trace (.json)、Hatchet (.hatchet) |

#### 8.6.5 输出示例

**控制台输出** (默认):

```
=== CPU Proton Profiling Results (Aggregated) ===
Scope                     Count      Total(cyc)       Min(cyc)       Max(cyc)       Avg(cyc)       Avg(us)
----------------------------------------------------------------------------------------------------------
mm_kernel[4,4,1]            100       240000000        2300000        2500000        2400000        100.00
softmax_kernel[8,1,1]        50        60000000        1100000        1300000        1200000         50.00
```

**Chrome Trace 输出** (`PROTON_OUTPUT=trace.json`):

生成 `trace.json` 文件，在 `chrome://tracing` 或 `https://ui.perfetto.dev` 中可以看到：
- 每个 kernel 的时间线
- 详细的时间戳和持续时间
- 多线程执行视图

#### 8.6.6 与 Kernel 内部插桩的对比

| 方法 | 粒度 | 代码修改 | 输出方式 | 使用场景 |
|------|------|----------|----------|----------|
| **自动 Kernel 捕获** | Kernel 级别 | 不需要 | 自动输出 (控制台/JSON/Hatchet) | 快速了解各 kernel 耗时分布 |
| **Kernel 内部插桩** (`pl.enter_scope`) | 操作级别 | 需要 | 需要 profiler.profile() | 深入分析 kernel 内部瓶颈 |

#### 8.6.7 常见问题

**Q: 为什么没有生成 trace.json 文件？**

A: 请检查：
1. 是否设置了 `PROTON_KERNEL_CAPTURE=1`
2. 是否设置了 `PROTON_OUTPUT=trace.json`
3. 是否清除了缓存以重新编译 kernel（`rm -rf ~/.triton/cache`）
4. 程序是否正常退出（异常退出可能不会触发输出）

正确的命令：
```bash
# 首次运行或切换设置时
rm -rf ~/.triton/cache
PROTON_KERNEL_CAPTURE=1 PROTON_OUTPUT=trace.json python3 your_script.py
```

#### 8.6.8 相关文件

- `spine-triton/backend/driver.py`: `_launch` 函数、`_generate_launcher` 函数
- `spine-triton/backend/include/ExecutionEngine/CpuProtonRuntime.cpp`: `proton_enter_kernel`、`proton_exit_kernel` API
- 测试脚本: `quick_test.py` (演示两种模式的使用)

---

## 9. 参考资源

### 9.1 相关文档
- [Spine-Triton 算子开发指南](./operator_development_guide.md)
- [Triton 官方文档](https://triton-lang.org/)

### 9.2 示例代码
- `test_mm_proton.py`: 矩阵乘法 Proton 分析示例
- `spine-triton/python/perf/`: 性能测试示例

### 9.3 工具链接
- [Chrome Tracing](chrome://tracing): Chrome 浏览器内置的性能分析工具
- [Perfetto](https://perfetto.dev/): 现代化的性能分析平台
- [Hatchet](https://github.com/hatchet/hatchet): 性能数据分析框架

---

## 10. 更新日志

- **v1.0** (2026-01-30): 初始版本，支持基本的 Proton 分析功能
- **v1.1** (2026-01-30): 添加 Kernel 外部 Proton 分析 (proton.scope) 文档，GPU vs CPU 差异说明
- **v1.2** (2026-02-03): 实现 CPU 自动 Kernel 捕获功能，通过 `PROTON_KERNEL_CAPTURE=1` 启用
- 后续版本将根据用户反馈和功能更新进行补充

---

**注意**: 本文档基于 spine-triton 在 SpacemiT K1/K3 平台上的实现。其他平台可能需要相应的适配。
