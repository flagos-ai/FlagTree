# tle.dsa.to_buffer

## 1. OP Overview

`tle.dsa.to_buffer` explicitly converts a `tl.tensor` into a TLE DSA `buffer`. This API is typically used after tensor computation to convert the result back into a buffer so it can be written back to GM through `tle.dsa.copy` or used by later DSA operations.

```python
tle.dsa.to_buffer(
    tensor: tl.tensor,
    space: address_space,
    bind_buffer: buffer = None,
    _builder=None,
) -> buffer
```

This API is a TLE DSA builtin and can only be used inside `@triton.jit` functions.

## 2. OP Specification

### 2.1 Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `tensor` | `tl.tensor` | Yes | Input tensor. It must be a non-scalar tensor. |
| `space` | `address_space` | Yes | Target buffer address space, for example `tle.dsa.ascend.UB`. |
| `bind_buffer` | `buffer` / `None` | No | Binding to an existing buffer is not currently supported; this must be `None`. |
| `_builder` | - | No | Compiler-internal parameter. Do not pass it from user code. |

Return value:

- `buffer`: a TLE DSA buffer converted from the input tensor.

### 2.2 Constraints

The following constraints apply:

1. `tensor.shape` must be a non-empty `list` or `tuple`.
2. Scalar tensors cannot be converted to buffers.
3. `space` must be passed explicitly as a backend address space, such as `tle.dsa.ascend.UB`.
4. `bind_buffer` must currently be `None`; passing a non-`None` value raises an error.

### 2.3 Return Type

The returned buffer type is determined by the input tensor and target address space:

```text
return element type = tensor.dtype
return shape        = tensor.shape
return space        = space
```

The implementation creates the buffer handle by calling `builder.dsa_to_buffer(tensor.handle, addr_space_attr)`.

## 3. Usage

### 3.1 Convert a tensor computation result to a buffer and write it back

The following example follows the usage pattern in `python/test/tle/test_vec_add_mix.py`:

```python
import triton
import triton.language as tl
import triton.experimental.tle as tle


@triton.jit
def add_mix_kernel(x_ptr, y_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
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

### 3.2 Use with `tle.dsa.to_tensor`

`tle.dsa.to_tensor` and `tle.dsa.to_buffer` are commonly used together:

```text
buffer -> tle.dsa.to_tensor -> tl.tensor computation -> tle.dsa.to_buffer -> buffer
```

For example:

```python
c_val = tle.dsa.to_tensor(c_ub)
b_val = tle.dsa.to_tensor(b_ub)
result = c_val - b_val
d_ub = tle.dsa.to_buffer(result, tle.dsa.ascend.UB)
```

## 4. Semantics

- `tle.dsa.to_buffer` does not perform a data copy; it converts a tensor value into a DSA buffer representation.
- The converted buffer can be used as input to `tle.dsa.copy`.
- Binding to an existing buffer through `bind_buffer` is not currently supported.
- Use `tle.dsa.to_tensor` to convert a buffer back to a tensor.
