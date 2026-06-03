# tle.dsa.copy

## 1. OP Overview

`tle.dsa.copy` generates a data-copy operation in TLE DSA. It can copy between GM pointer tensors and DSA buffers, and it can also copy from DSA buffers back to GM pointer tensors. The copied region is specified by the `shape` parameter.

```python
tle.dsa.copy(
    src,
    dst,
    shape,
    inter_no_alias=False,
    _builder=None,
) -> None
```

This API is a TLE DSA builtin and can only be used inside `@triton.jit` functions.

## 2. OP Specification

### 2.1 Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `src` | `tl.tensor` / `buffer` | Yes | Copy source. It can be a pointer tensor or a TLE DSA buffer. |
| `dst` | `tl.tensor` / `buffer` | Yes | Copy destination. It can be a pointer tensor or a TLE DSA buffer. |
| `shape` | `List[int]` / `List[tl.constexpr]` / `List[tl.tensor]` | Yes | Copy extents, one extent per dimension. It cannot be empty. |
| `inter_no_alias` | `bool` / `tl.constexpr` | No | Marks copy operations from different iterations as non-aliasing. Default is `False`. |
| `_builder` | - | No | Compiler-internal parameter. Do not pass it from user code. |

Return value: none.

### 2.2 Constraints

The following constraints apply:

1. `shape` cannot be empty.
2. `src` and `dst` must provide handles usable by DSA IR. Common combinations include:
   - GM pointer tensor -> DSA buffer
   - DSA buffer -> GM pointer tensor
   - buffer returned by `tle.dsa.to_buffer(...)` -> GM pointer tensor

## 3. Usage

### 3.1 GM -> UB -> GM

The following example follows the usage pattern in `python/test/tle/test_vec_add.py`:

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

### 3.2 2D copy

The following example follows the usage pattern of a 2D vector add:

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

### 3.3 Use with `tle.dsa.to_buffer`

If an existing `tl.tensor` computation result needs to be written back through DSA copy, convert it to a buffer first:

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

## 4. Semantics

- `shape` describes the copied region and is not inferred automatically from `src` or `dst`; it must be passed explicitly.
- `tle.dsa.copy` does not return a new object. It writes the result into `dst`.
