# tle.dsa.alloc

## 1. OP Overview

`tle.dsa.alloc` allocates a TLE DSA buffer in a specified backend address space. The returned buffer is a different object type from `tl.tensor` and is typically used together with APIs such as `tle.dsa.copy` and `tle.dsa.to_tensor`.

```python
tle.dsa.alloc(
    shape: List[tl.constexpr],
    dtype: tl.dtype,
    mem_addr_space: address_space,
    _builder=None,
) -> buffer
```

This API is a TLE DSA builtin and can only be used inside `@triton.jit` functions.

## 2. OP Specification

### 2.1 Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `shape` | `List[tl.constexpr]` / `tuple` | Yes | Buffer shape. It must be a `list` or `tuple`. |
| `dtype` | `tl.dtype` | Yes | Buffer element type, for example `tl.float32`. |
| `mem_addr_space` | `tle.dsa.ascend.*` address space | Yes | Backend address space where the buffer is allocated. It cannot be `None`. |
| `_builder` | - | No | Compiler-internal parameter. Do not pass it from user code. |

Return value:

- `buffer`: a TLE DSA buffer carrying element type, shape, and address-space information.

### 2.2 Address Space

`mem_addr_space` specifies which backend address space the buffer is allocated in.

For example:

```python
a_ub = tle.dsa.alloc([BLOCK_SIZE], dtype=tl.float32, mem_addr_space=tle.dsa.ascend.UB)
```

`tle.dsa.ascend.UB` corresponds to the UB address space in the Ascend CANN extension.

### 2.3 Implementation Constraints

The following constraints apply:

1. `mem_addr_space` cannot be `None`.
2. `shape` is unwrapped by `tl._unwrap_shape(shape)` and must be a `list` or `tuple`.
3. `dtype` is unwrapped by `tl._constexpr_to_value(dtype)` and converted to an IR element type.
4. `mem_addr_space` is unwrapped by `tl._constexpr_to_value(mem_addr_space)` and converted to an IR address-space attribute.
5. The return value is a `tle.dsa` `buffer`; it cannot be used directly as a `tl.tensor`. Use `tle.dsa.to_tensor` if a tensor is needed.

### 2.4 Return Type

The implementation creates a DSA buffer type and an alloc op:

```text
memref type = dsa_get_buffer_type(shape, dtype, address_space)
handle      = create_dsa_alloc(memref type)
return type = buffer(handle, buffer_type(element_ty=dtype, shape=shape, space=mem_addr_space))
```

## 3. Usage

### 3.1 1D UB buffer + copy + add

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

### 3.2 2D UB buffer

The following example shows how to allocate 2D buffers:

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

### 3.3 Use with `tle.dsa.to_tensor`

`tle.dsa.alloc` returns a buffer. Convert it explicitly if it needs to be used as a tensor:

```python
@triton.jit
def bind_buffer():
    buffer1 = tle.dsa.alloc(shape=[32, 32], dtype=tl.float32, mem_addr_space=tle.dsa.ascend.UB)
    tensor1 = tle.dsa.to_tensor(buffer1, writable=True)
```

## 4. Semantics

- `tle.dsa.alloc` only allocates a DSA buffer; it does not initialize the buffer contents.
- The current implementation requires `mem_addr_space` to be passed explicitly.
