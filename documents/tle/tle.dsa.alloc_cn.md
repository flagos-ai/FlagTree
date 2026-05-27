# tle.dsa.alloc

## 1. OP 概述

`tle.dsa.alloc` 用于在指定后端地址空间上分配一个 TLE DSA buffer。该 buffer 与 `tl.tensor` 是不同的对象类型，通常与 `tle.dsa.copy`、`tle.dsa.to_tensor` 等接口配合使用。

```python
tle.dsa.alloc(
    shape: List[tl.constexpr],
    dtype: tl.dtype,
    mem_addr_space: address_space,
    _builder=None,
) -> buffer
```

该接口是 TLE DSA builtin，只能在 `@triton.jit` 函数中使用。

## 2. OP 规格

### 2.1 参数说明

| 参数名 | 类型 | 必需 | 说明 |
|--------|------|------|------|
| `shape` | `List[tl.constexpr]` / `tuple` | 是 | buffer 的形状。必须是 `list` 或 `tuple` |
| `dtype` | `tl.dtype` | 是 | buffer 的元素类型，例如 `tl.float32` |
| `mem_addr_space` | `tle.dsa.ascend.*` 地址空间 | 是 | buffer 所在的后端地址空间，不能为 `None` |
| `_builder` | - | 否 | 编译器内部参数，不支持外部调用 |

返回值：

- `buffer`：TLE DSA buffer，携带元素类型、shape 和地址空间信息。

### 2.2 地址空间

`mem_addr_space` 用于指定 buffer 分配到哪个后端地址空间。

例如：

```python
a_ub = tle.dsa.alloc([BLOCK_SIZE], dtype=tl.float32, mem_addr_space=tle.dsa.ascend.UB)
```

`tle.dsa.ascend.UB` 对应 Ascend CANN extension 中的 UB 地址空间。

### 2.3 实现约束

需满足以下约束：

1. `mem_addr_space` 不能为 `None`。
2. `shape` 经 `tl._unwrap_shape(shape)` 解包后必须是 `list` 或 `tuple`。
3. `dtype` 会通过 `tl._constexpr_to_value(dtype)` 解包，并转换为 IR element type。
4. `mem_addr_space` 会通过 `tl._constexpr_to_value(mem_addr_space)` 解包，并转换为 IR address space attribute。
5. 返回值是 `tle.dsa` 的 `buffer` 类型，不能直接当作 `tl.tensor` 使用；如需转换为 tensor，应使用 `tle.dsa.to_tensor`。

### 2.4 返回类型

底层实现会创建 DSA buffer type 和 alloc op：

```text
memref type = dsa_get_buffer_type(shape, dtype, address_space)
handle      = create_dsa_alloc(memref type)
返回类型    = buffer(handle, buffer_type(element_ty=dtype, shape=shape, space=mem_addr_space))
```

## 3. 使用方法

### 3.1 一维 UB buffer + copy + add

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

### 3.2 二维 UB buffer

以下示例展示二维 shape 的分配方式：

```python
@triton.jit
def add_2d_kernel(x_ptr, y_ptr, output_ptr, n_cols, BLOCK_SIZE: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    offs_n = pid_n * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    ptrs = offs_m[:, None] * n_cols + offs_n[None, :]

    a_ub = tle.dsa.alloc([BLOCK_SIZE, BLOCK_SIZE], dtype=tl.float32, mem_addr_space=tle.dsa.ascend.UB)
    b_ub = tle.dsa.alloc([BLOCK_SIZE, BLOCK_SIZE], dtype=tl.float32, mem_addr_space=tle.dsa.ascend.UB)
    c_ub = tle.dsa.alloc([BLOCK_SIZE, BLOCK_SIZE], dtype=tl.float32, mem_addr_space=tle.dsa.ascend.UB)

    tle.dsa.copy(x_ptr + ptrs, a_ub, [BLOCK_SIZE, BLOCK_SIZE])
    tle.dsa.copy(y_ptr + ptrs, b_ub, [BLOCK_SIZE, BLOCK_SIZE])
    tle.dsa.add(a_ub, b_ub, c_ub)
    tle.dsa.copy(c_ub, output_ptr + ptrs, [BLOCK_SIZE, BLOCK_SIZE])
```

### 3.3 与 `tle.dsa.to_tensor` 配合

`tle.dsa.alloc` 返回的是 buffer。如果需要作为 tensor 使用，可显式转换：

```python
@triton.jit
def bind_buffer():
    buffer1 = tle.dsa.alloc(shape=[32, 32], dtype=tl.float32, mem_addr_space=tle.dsa.ascend.UB)
    tensor1 = tle.dsa.to_tensor(buffer1, writable=True)
```

## 4. 语义说明

- `tle.dsa.alloc` 只负责分配 DSA buffer，不会自动初始化 buffer 内容。
- 当前实现要求 `mem_addr_space` 必须显式传入。
