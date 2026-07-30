# Spine-Triton 算子开发指南

本文档详细介绍如何使用 `triton.language` (tl) 和 `smt` 模块在 spine-triton 上编写高性能算子。

## 目录

1. [环境配置](#1-环境配置)
2. [基础概念](#2-基础概念)
3. [triton.language (tl) 基础](#3-tritonlanguage-tl-基础)
4. [SMT 模块详解](#4-smt-模块详解)
5. [算子编写示例](#5-算子编写示例)
6. [最佳实践](#6-最佳实践)
7. [常见问题](#7-常见问题)

---

## 1. 环境配置

### 1.1 基本导入

```python
import torch
import triton
import triton.language as tl
import triton.language.extra.smt as smt
from triton.backends.spine_triton.driver import CPUDriver

# 设置 CPU 驱动（必须在导入后立即设置）
triton.runtime.driver.set_active(CPUDriver())
```

### 1.2 环境变量

| 环境变量 | 说明 | 示例 |
|---------|------|------|
| `PYTHONPATH` | 添加 spine-triton 构建路径 | `export PYTHONPATH=/path/to/spine-triton/build-riscv64:$PYTHONPATH` |
| `SPINE_TRITON_DUMP_PATH` | IR dump 输出路径 | `export SPINE_TRITON_DUMP_PATH=./ir_dumps` |
| `TRITON_ALWAYS_COMPILE` | 强制重新编译 | `export TRITON_ALWAYS_COMPILE=1` |

---

## 2. 基础概念

### 2.1 Spine-Triton vs 标准 Triton

| 特性 | 标准 Triton | Spine-Triton |
|------|------------|--------------|
| 目标硬件 | NVIDIA GPU | RISC-V CPU (SpacemiT) |
| 后端 | CUDA/PTX | spine-mlir → LLVM IR |
| 驱动 | GPU Driver | CPUDriver |
| 特殊扩展 | 无 | SMT 模块 (张量核心操作) |

### 2.2 编译流程

```
Triton Python DSL
       ↓
   Triton IR (TTIR)
       ↓ (triton-to-linalg)
   Linalg IR
       ↓ (spine-mlir)
   LLVM IR
       ↓ (llc)
   目标代码 (RISC-V)
```

### 2.3 算子类型选择

| 算子类型 | 推荐方式 | 说明 |
|---------|---------|------|
| 矩阵乘法 (mm, bmm, addmm) | `smt` 模块 | 利用张量核心加速 |
| 逐元素操作 (relu, gelu, silu) | `tl` 标准操作 | 简单高效 |
| 规约操作 (softmax, layernorm) | `tl` + 循环 | 需要累加器 |
| 矩阵向量乘 (mv) | `tl` + 循环 | 分块累加 |

---

## 3. triton.language (tl) 基础

### 3.1 程序标识

```python
@triton.jit
def kernel(...):
    # 获取程序 ID（类似 CUDA 的 blockIdx）
    pid = tl.program_id(0)      # 第 0 维
    pid_m = tl.program_id(0)    # M 维度
    pid_n = tl.program_id(1)    # N 维度
```

### 3.2 Block Pointer（推荐方式）

Block Pointer 是 spine-triton 推荐的内存访问方式：

```python
# 创建块指针
block_ptr = tl.make_block_ptr(
    base=ptr,                           # 基地址
    shape=[M, N],                       # 张量形状
    strides=[stride_m, stride_n],       # 步长
    offsets=[pid_m * BLOCK_M, 0],       # 偏移
    block_shape=[BLOCK_M, BLOCK_N],     # 块形状
    order=[1, 0],                       # 内存布局顺序
)

# 加载数据
data = tl.load(block_ptr, boundary_check=(0, 1))

# 存储数据
tl.store(block_ptr, data, boundary_check=(0, 1))
```

**参数说明：**
- `order=[1, 0]`：表示列优先（最后一维连续）
- `boundary_check=(0, 1)`：检查第 0 和第 1 维的边界

### 3.3 常用操作

```python
# 数学运算
y = tl.exp(x)
y = tl.log(x)
y = tl.sqrt(x)
y = tl.abs(x)

# 规约操作
sum_val = tl.sum(x, axis=1)
max_val = tl.max(x, axis=0)

# 条件操作
y = tl.where(condition, x, 0)

# 类型转换
y = x.to(tl.float32)
y = x.to(tl.float16)

# 广播
y = x[None, :]  # 添加维度
y = x[:, None]
```

### 3.4 constexpr 参数

编译时常量必须使用 `tl.constexpr`：

```python
@triton.jit
def kernel(
    ...,
    BLOCK_SIZE_M: tl.constexpr,  # 编译时常量
    BLOCK_SIZE_N: tl.constexpr,
):
    ...
```

---

## 4. SMT 模块详解

SMT (SpacemiT Tensor) 模块是 spine-triton 针对 SpacemiT 硬件的核心扩展，提供张量核心加速操作。

### 4.1 导入方式

```python
import triton.language.extra.smt as smt
```

### 4.2 核心 API

#### 4.2.1 `smt.descriptor_load` - 描述符加载

基于描述符的块加载操作，用于高效加载数据。

```python
# 语法
data = smt.descriptor_load(block_ptr, offsets)

# 参数
# - block_ptr: tl.make_block_ptr 创建的块指针
# - offsets: 偏移量元组，如 (0, 0) 或 (s * SUB_BLK_M, 0)

# 示例
a_block_ptr = tl.make_block_ptr(
    base=a_ptr,
    shape=[M, K],
    strides=[stride_am, stride_ak],
    offsets=[pid_m * BLOCK_SIZE_M, 0],
    block_shape=[BLOCK_SIZE_M, BLOCK_SIZE_K],
    order=[1, 0],
)
a_data = smt.descriptor_load(a_block_ptr, (0, 0))
```

#### 4.2.2 `smt.view` - 创建视图

创建张量的本地视图，支持指定 micro tile 大小。这是 SMT 模块最重要的操作之一。

```python
# 语法
view = smt.view(base, offsets, shape, micro_size)

# 参数
# - base: 基础张量（通常来自 descriptor_load）
# - offsets: 偏移量元组
# - shape: 视图形状
# - micro_size: micro tile 大小（用于张量核心）

# 示例：创建 4D packed 视图
a = smt.view(
    a_data,                              # 输入数据
    (0, 0),                              # 偏移
    (BLOCK_SIZE_M, BLOCK_SIZE_K),        # 形状
    (MICRO_M, MICRO_K)                   # micro tile 大小
)
# 结果形状: [BLOCK_SIZE_M/MICRO_M, BLOCK_SIZE_K/MICRO_K, MICRO_M, MICRO_K]

# 展平回 2D（micro_size 设为 (1, 1)）
result_2d = smt.view(result_4d, (0, 0), (BLOCK_SIZE_M, BLOCK_SIZE_N), (1, 1))
```

**Micro Tile 说明：**
- `(MICRO_M, MICRO_K)`: 将 2D 张量打包成 4D 格式，用于张量核心计算
- `(1, 1)`: 将 4D 结果展平回 2D 格式

#### 4.2.3 `smt.dot` - 4D 矩阵乘法

执行 4D 矩阵乘法 (mmt4d)，针对张量核心优化。

```python
# 语法
result = smt.dot(a_packed, b_packed)

# 参数
# - a_packed: packed A 矩阵，形状 [MB, KB, mb, kb]
# - b_packed: packed B 矩阵，形状 [KB, NB, kb, nb]
# 返回: 形状 [MB, NB, mb, nb]

# 示例
# A: [M, K] -> view -> [M/MICRO_M, K/MICRO_K, MICRO_M, MICRO_K]
# B: [K, N] -> view -> [K/MICRO_K, N/MICRO_N, MICRO_K, MICRO_N]
accumulator = smt.dot(a, b)
# 结果: [M/MICRO_M, N/MICRO_N, MICRO_M, MICRO_N]
```

#### 4.2.4 `smt.alloc` - 共享内存分配

在共享内存中分配张量。

```python
# 语法
buffer = smt.alloc(shape, type=tl.float32, storage="l2")

# 示例
shared_buffer = smt.alloc(shape=(BLOCK_SIZE_N, BLOCK_SIZE_K), type=tl.float16)
```

#### 4.2.5 `smt.parallel` - 并行迭代

并行执行语义的迭代器，用于多张量核心参与循环。

```python
# 语法
for s in smt.parallel(start, end, step=1):
    ...

# 示例：并行处理多个子块
sub_num = (BLOCK_SIZE_M + SUB_BLK_M - 1) // SUB_BLK_M
for s in smt.parallel(0, sub_num):
    a = smt.view(a_data, (s * SUB_BLK_M, 0), (SUB_BLK_M, BLOCK_SIZE_K), (MICRO_M, MICRO_K))
    accumulator = smt.dot(a, b)
```

### 4.3 SMT 数据流示意图

```
原始 2D 张量 A [M, K]
        ↓
   descriptor_load
        ↓
   smt.view (micro_size=(MICRO_M, MICRO_K))
        ↓
4D Packed 张量 [M/MICRO_M, K/MICRO_K, MICRO_M, MICRO_K]
        ↓
   smt.dot (与 B 的 4D packed 张量)
        ↓
4D 结果 [M/MICRO_M, N/MICRO_N, MICRO_M, MICRO_N]
        ↓
   smt.view (micro_size=(1, 1))
        ↓
2D 结果 [M, N]
```

---

## 5. 算子编写示例

### 5.1 矩阵乘法 (MM) - 使用 SMT

```python
@triton.jit
def mm_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    MICRO_M: tl.constexpr,
    MICRO_K: tl.constexpr,
    MICRO_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # 创建 A 的块指针
    a_block_ptr = tl.make_block_ptr(
        base=a_ptr,
        shape=[M, K],
        strides=[stride_am, stride_ak],
        offsets=[pid_m * BLOCK_SIZE_M, 0],
        block_shape=[BLOCK_SIZE_M, BLOCK_SIZE_K],
        order=[1, 0],
    )

    # 创建 B 的块指针
    b_block_ptr = tl.make_block_ptr(
        base=b_ptr,
        shape=[K, N],
        strides=[stride_bk, stride_bn],
        offsets=[0, pid_n * BLOCK_SIZE_N],
        block_shape=[BLOCK_SIZE_K, BLOCK_SIZE_N],
        order=[1, 0],
    )

    # 使用 SMT 加载并创建 4D 视图
    a_data = smt.descriptor_load(a_block_ptr, (0, 0))
    a = smt.view(a_data, (0, 0), (BLOCK_SIZE_M, BLOCK_SIZE_K), (MICRO_M, MICRO_K))

    b_data = smt.descriptor_load(b_block_ptr, (0, 0))
    b = smt.view(b_data, (0, 0), (BLOCK_SIZE_K, BLOCK_SIZE_N), (MICRO_K, MICRO_N))

    # 执行 4D 矩阵乘法
    accumulator = smt.dot(a, b)

    # 展平回 2D
    accumulator = smt.view(accumulator, (0, 0), (BLOCK_SIZE_M, BLOCK_SIZE_N), (1, 1))
    c = accumulator.to(c_ptr.dtype.element_ty)

    # 存储结果
    c_block_ptr = tl.make_block_ptr(
        base=c_ptr,
        shape=[M, N],
        strides=[stride_cm, stride_cn],
        offsets=[pid_m * BLOCK_SIZE_M, pid_n * BLOCK_SIZE_N],
        block_shape=[BLOCK_SIZE_M, BLOCK_SIZE_N],
        order=[1, 0],
    )
    tl.store(c_block_ptr, c, boundary_check=(0, 1))


def triton_mm(a, b):
    M, K = a.shape
    _, N = b.shape
    c = torch.empty((M, N), device=a.device, dtype=a.dtype)

    grid = lambda META: (
        triton.cdiv(M, META["BLOCK_SIZE_M"]),
        triton.cdiv(N, META["BLOCK_SIZE_N"]),
    )

    mm_kernel[grid](
        a, b, c,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        BLOCK_SIZE_M=256,
        BLOCK_SIZE_N=256,
        BLOCK_SIZE_K=triton.next_power_of_2(K),
        MICRO_M=16,
        MICRO_K=8,
        MICRO_N=32,
    )
    return c
```

### 5.2 逐元素操作 (GELU) - 使用 tl

```python
from triton.language.extra.cpu import libdevice as tl_extra_shim

@triton.jit
def gelu_none_kernel(
    in_ptr, out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """GeLU: 0.5 * x * (1 + erf(x / sqrt(2)))"""
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE

    # 使用 block pointer
    in_block_ptr = tl.make_block_ptr(
        base=in_ptr,
        shape=(n_elements,),
        strides=(1,),
        offsets=(block_start,),
        block_shape=(BLOCK_SIZE,),
        order=(0,),
    )
    out_block_ptr = tl.make_block_ptr(
        base=out_ptr,
        shape=(n_elements,),
        strides=(1,),
        offsets=(block_start,),
        block_shape=(BLOCK_SIZE,),
        order=(0,),
    )

    x = tl.load(in_block_ptr, boundary_check=(0,))

    # GeLU 计算
    scale: tl.constexpr = 0.7071067811  # 1 / sqrt(2)
    y = 0.5 * x * (1.0 + tl_extra_shim.erf(x * scale))

    tl.store(out_block_ptr, y, boundary_check=(0,))


def triton_gelu(x):
    output = torch.empty_like(x)
    n_elements = x.numel()
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)

    gelu_none_kernel[grid](x, output, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    return output
```

### 5.3 矩阵向量乘法 (MV) - 使用循环

```python
@triton.jit
def mv_kernel(
    mat_ptr, vec_ptr, out_ptr,
    M, N,
    stride_mm, stride_mn,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    """mv: out = mat @ vec，使用循环处理 N 维度"""
    pid_m = tl.program_id(0)

    # 初始化累加器
    acc = tl.zeros((BLOCK_SIZE_M,), dtype=tl.float32)

    # 循环处理 N 维度
    for n_offset in range(0, N, BLOCK_SIZE_N):
        # 加载矩阵块
        mat_block_ptr = tl.make_block_ptr(
            base=mat_ptr,
            shape=[M, N],
            strides=[stride_mm, stride_mn],
            offsets=[pid_m * BLOCK_SIZE_M, n_offset],
            block_shape=[BLOCK_SIZE_M, BLOCK_SIZE_N],
            order=[1, 0],
        )
        mat = tl.load(mat_block_ptr, boundary_check=(0, 1)).to(tl.float32)

        # 加载向量块
        vec_block_ptr = tl.make_block_ptr(
            base=vec_ptr,
            shape=[N],
            strides=[1],
            offsets=[n_offset],
            block_shape=[BLOCK_SIZE_N],
            order=[0],
        )
        vec = tl.load(vec_block_ptr, boundary_check=(0,)).to(tl.float32)

        # 累加
        acc += tl.sum(mat * vec[None, :], axis=1)

    # 存储结果
    out_block_ptr = tl.make_block_ptr(
        base=out_ptr,
        shape=[M],
        strides=[1],
        offsets=[pid_m * BLOCK_SIZE_M],
        block_shape=[BLOCK_SIZE_M],
        order=[0],
    )
    tl.store(out_block_ptr, acc.to(out_ptr.dtype.element_ty), boundary_check=(0,))
```

---

## 6. 最佳实践

### 6.1 Block Size 选择

| 操作类型 | 推荐 Block Size | 说明 |
|---------|----------------|------|
| 矩阵乘法 | 128-256 | 需要足够大以利用张量核心 |
| 逐元素 | 1024 | 简单操作可以用更大的块 |
| 规约 | 128-256 | 需要平衡并行度和内存 |

### 6.2 Micro Tile 选择

对于 SMT 的 `smt.view`，推荐的 micro tile 大小：

| 数据类型 | MICRO_M | MICRO_K | MICRO_N |
|---------|---------|---------|---------|
| float32 | 8 | 8 | 16 |
| float16 | 16 | 8 | 32 |

### 6.3 内存访问优化

```python
# ✅ 推荐：使用 block pointer
block_ptr = tl.make_block_ptr(...)
data = tl.load(block_ptr, boundary_check=(0, 1))

# ❌ 避免：使用 mask 的传统方式（在 spine-triton 上效率较低）
# offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
# mask = offsets < n_elements
# data = tl.load(ptr + offsets, mask=mask)
```

### 6.4 类型转换

```python
# 计算时使用 float32 以保证精度
x_fp32 = x.to(tl.float32)
result = some_computation(x_fp32)
# 存储时转回原始类型
result = result.to(x.dtype)
```

---

## 7. 常见问题

### 7.1 编译错误

**问题：** `AssertionError: incompatible dimensions`

**原因：** 矩阵维度不匹配

**解决：** 检查输入张量的形状，确保 `A.shape[1] == B.shape[0]`

---

**问题：** `BLOCK_SIZE_K must be power of 2`

**原因：** K 维度的块大小必须是 2 的幂

**解决：** 使用 `triton.next_power_of_2(K)` 计算

```python
BLOCK_SIZE_K = triton.next_power_of_2(K)
```

---

### 7.2 精度问题

**问题：** 结果与 PyTorch 参考实现差异较大

**原因：**
1. float16 精度有限
2. 累加顺序不同

**解决：**
1. 使用 float32 进行中间计算
2. 调整 atol 容差

```python
# 验证时使用合适的容差
atol = 1e-2 if dtype == torch.float16 else 1e-4
torch.testing.assert_close(output, ref, atol=atol, rtol=0)
```

---

### 7.3 性能问题

**问题：** Triton kernel 比 PyTorch 慢

**可能原因：**
1. Block size 太小
2. 没有使用 SMT 模块
3. 过多的边界检查

**解决：**
1. 增大 block size
2. 对矩阵乘法使用 `smt.dot`
3. 确保输入大小是 block size 的倍数

---

### 7.4 SMT 相关问题

**问题：** `smt.view` 报错

**原因：** shape 和 micro_size 不兼容

**解决：** 确保 shape 的每个维度都能被 micro_size 整除

```python
# ✅ 正确
smt.view(data, (0, 0), (256, 128), (16, 8))  # 256/16=16, 128/8=16

# ❌ 错误
smt.view(data, (0, 0), (256, 100), (16, 8))  # 100/8 不是整数
```

---

**问题：** `smt.dot` 结果形状不对

**原因：** 输入的 4D 张量形状不匹配

**解决：** 确保 A 的 KB 维度等于 B 的 KB 维度

```python
# A: [MB, KB, mb, kb]
# B: [KB, NB, kb, nb]
# 要求: A 的 KB == B 的 KB, A 的 kb == B 的 kb
```

---

### 7.5 调试技巧

**导出 IR 进行调试：**

```bash
export SPINE_TRITON_DUMP_PATH=./ir_dumps
python your_script.py
# 查看生成的 IR 文件
ls ./ir_dumps/
```

**强制重新编译：**

```bash
export TRITON_ALWAYS_COMPILE=1
python your_script.py
```

---

## 附录：API 速查表

### tl 常用 API

| API | 说明 |
|-----|------|
| `tl.program_id(axis)` | 获取程序 ID |
| `tl.make_block_ptr(...)` | 创建块指针 |
| `tl.load(ptr, boundary_check)` | 加载数据 |
| `tl.store(ptr, val, boundary_check)` | 存储数据 |
| `tl.zeros(shape, dtype)` | 创建零张量 |
| `tl.sum(x, axis)` | 求和规约 |
| `tl.max(x, axis)` | 最大值规约 |
| `tl.where(cond, x, y)` | 条件选择 |
| `tl.exp(x)` | 指数函数 |
| `tl.log(x)` | 对数函数 |

### smt 常用 API

| API | 说明 |
|-----|------|
| `smt.descriptor_load(ptr, offsets)` | 描述符加载 |
| `smt.view(base, offsets, shape, micro_size)` | 创建视图 |
| `smt.dot(a, b)` | 4D 矩阵乘法 |
| `smt.alloc(shape, type, storage)` | 分配共享内存 |
| `smt.parallel(start, end)` | 并行迭代 |

---

*文档版本: 1.0*
*最后更新: 2026-01-27*
