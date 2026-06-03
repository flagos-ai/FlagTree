# tle.dsa.copy

## 1. OP 概述

`tle.dsa.copy` 用于在 TLE DSA 中生成数据拷贝操作。它可以在 GM 指针 tensor 与 DSA buffer 之间拷贝，也可以从 DSA buffer 拷贝到 GM 指针 tensor。拷贝范围由 `shape` 参数指定。

```python
tle.dsa.copy(
    src,
    dst,
    shape,
    inter_no_alias=False,
    _builder=None,
) -> None
```

该接口是 TLE DSA builtin，只能在 `@triton.jit` 函数中使用。

## 2. OP 规格

### 2.1 参数说明

| 参数名 | 类型 | 必需 | 说明 |
|--------|------|------|------|
| `src` | `tl.tensor` / `buffer` | 是 | 拷贝源，可以是指针 tensor，也可以是 TLE DSA buffer |
| `dst` | `tl.tensor` / `buffer` | 是 | 拷贝目的，可以是指针 tensor，也可以是 TLE DSA buffer |
| `shape` | `List[int]` / `List[tl.constexpr]` / `List[tl.tensor]` | 是 | 拷贝范围，每个维度一个 extent；长度不能为 0 |
| `inter_no_alias` | `bool` / `tl.constexpr` | 否 | 标记不同迭代之间的 copy 不存在 alias 关系，默认 `False` |
| `_builder` | - | 否 | 编译器内部参数，不支持外部调用 |

返回值：无。

### 2.2 参数约束

需满足以下约束：

1. `shape` 长度不能为 0。
2. `src` 和 `dst` 需要具有可用于 DSA IR 的 `handle`，常见组合是：
   - GM 指针 tensor -> DSA buffer
   - DSA buffer -> GM 指针 tensor
   - `tle.dsa.to_buffer(...)` 得到的 buffer -> GM 指针 tensor

## 3. 使用方法

### 3.1 GM -> UB -> GM

以下示例来自 `python/test/tle/test_vec_add.py` 的用法模式：

```python
import triton
import triton.language as tl
import triton.experimental.tle as tle


@triton.jit
def add_kernel(x_ptr, y_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)

    a_ub = tle.dsa.alloc([BLOCK_SIZE], dtype=tl.float32, mem_addr_space=tle.dsa.ascend.UB)
    b_ub = tle.dsa.alloc([BLOCK_SIZE], dtype=tl.float32, mem_addr_space=tle.dsa.ascend.UB)
    c_ub = tle.dsa.alloc([BLOCK_SIZE], dtype=tl.float32, mem_addr_space=tle.dsa.ascend.UB)

    t0 = n_elements - block_start
    tail_size = tl.minimum(t0, BLOCK_SIZE)

    tle.dsa.copy(x_ptr + offsets, a_ub, [tail_size])
    tle.dsa.copy(y_ptr + offsets, b_ub, [tail_size])

    tle.dsa.add(a_ub, b_ub, c_ub)
    tle.dsa.copy(c_ub, output_ptr + offsets, [tail_size])
```

### 3.2 二维 copy

以下示例来自二维 vector add 的用法模式：

```python
@triton.jit
def add_2d_kernel(x_ptr, y_ptr, output_ptr, n_elements, n_cols, BLOCK_SIZE: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    block_start_m = pid_m * BLOCK_SIZE
    block_start_n = pid_n * BLOCK_SIZE
    offs_m = block_start_m + tl.arange(0, BLOCK_SIZE)
    offs_n = block_start_n + tl.arange(0, BLOCK_SIZE)

    x_ptrs = x_ptr + offs_m[:, None] * n_cols + offs_n[None, :]
    y_ptrs = y_ptr + offs_m[:, None] * n_cols + offs_n[None, :]
    out_ptrs = output_ptr + offs_m[:, None] * n_cols + offs_n[None, :]

    a_ub = tle.dsa.alloc([BLOCK_SIZE, BLOCK_SIZE], dtype=tl.float32, mem_addr_space=tle.dsa.ascend.UB)
    b_ub = tle.dsa.alloc([BLOCK_SIZE, BLOCK_SIZE], dtype=tl.float32, mem_addr_space=tle.dsa.ascend.UB)
    c_ub = tle.dsa.alloc([BLOCK_SIZE, BLOCK_SIZE], dtype=tl.float32, mem_addr_space=tle.dsa.ascend.UB)

    tail_size_m = tl.minimum(n_elements - block_start_m, BLOCK_SIZE)
    tail_size_n = tl.minimum(n_cols - block_start_n, BLOCK_SIZE)

    tle.dsa.copy(x_ptrs, a_ub, [tail_size_m, tail_size_n])
    tle.dsa.copy(y_ptrs, b_ub, [tail_size_m, tail_size_n])
    tle.dsa.add(a_ub, b_ub, c_ub)
    tle.dsa.copy(c_ub, out_ptrs, [tail_size_m, tail_size_n])
```

### 3.3 与 `tle.dsa.to_buffer` 配合

当已有 `tl.tensor` 计算结果需要通过 DSA copy 写回时，可先转换为 buffer：

```python
@triton.jit
def copy_tensor_result(x_ptr, y_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)

    a_ub = tle.dsa.alloc([BLOCK_SIZE], dtype=tl.float32, mem_addr_space=tle.dsa.ascend.UB)
    b_ub = tle.dsa.alloc([BLOCK_SIZE], dtype=tl.float32, mem_addr_space=tle.dsa.ascend.UB)
    c_ub = tle.dsa.alloc([BLOCK_SIZE], dtype=tl.float32, mem_addr_space=tle.dsa.ascend.UB)

    tail_size = tl.minimum(n_elements - block_start, BLOCK_SIZE)
    tle.dsa.copy(x_ptr + offsets, a_ub, [tail_size])
    tle.dsa.copy(y_ptr + offsets, b_ub, [tail_size])

    tle.dsa.add(a_ub, b_ub, c_ub)
    c_val = tle.dsa.to_tensor(c_ub)
    b_val = tle.dsa.to_tensor(b_ub)
    result = c_val - b_val

    d_ub = tle.dsa.to_buffer(result, tle.dsa.ascend.UB)
    tle.dsa.copy(d_ub, output_ptr + offsets, [tail_size])
```

## 4. 语义说明

- `shape` 描述拷贝范围，不会从 `src` 或 `dst` 自动推导，因此必须显式传入。
- `tle.dsa.copy` 不返回新对象，结果写入 `dst`。
