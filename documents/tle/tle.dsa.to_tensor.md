# tle.dsa.to_tensor

## 1. OP Overview

`tle.dsa.to_tensor` explicitly converts a TLE DSA `buffer` into a `tl.tensor`. The converted tensor can participate in Triton expression computation and can also be used as input to later `tl.store`, arithmetic expressions, or other tensor APIs.

```python
tle.dsa.to_tensor(
    memref: buffer,
    writable: bool = True,
    target_shape=None,
    _builder=None,
) -> tl.tensor
```

This API is a TLE DSA builtin and can only be used inside `@triton.jit` functions.

## 2. OP Specification

### 2.1 Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `memref` | `buffer` | Yes | Input TLE DSA buffer. |
| `writable` | `bool` | No | Whether the returned tensor is marked as writable. Default is `True`. |
| `target_shape` | `List[int]` / `tuple` / `None` | No | Target tensor shape. If not provided, `memref.shape` is used. |
| `_builder` | - | No | Compiler-internal parameter. Do not pass it from user code. |

Return value:

- `tl.tensor`: a tensor converted from `memref`, with the same element type as `memref.dtype`.

### 2.2 Constraints

The following constraints apply:

1. `memref` must be a TLE DSA `buffer`.
2. By default, the returned tensor shape is `memref.shape`.

### 2.3 Return Type

The returned tensor type is determined by the buffer dtype and final shape:

```text
return element type = memref.dtype
return shape        = target_shape if target_shape is not None else memref.shape
```

When `target_shape` is set, the implementation first creates a target buffer type, converts the layout through `builder.create_convert_layout`, and then runs `builder.dsa_to_tensor(memref_value, writable)`.

## 3. Usage

### 3.1 Convert a buffer allocated by `alloc` to a tensor

The following example follows the usage pattern in `python/test/tle/test_bind_buffer.py`:

```python
import triton
import triton.language as tl
import triton.experimental.tle as tle


@triton.jit
def bind_buffer():
    buffer1 = tle.dsa.alloc(shape=[32, 32], dtype=tl.float32, mem_addr_space=tle.dsa.ascend.UB)
    tensor1 = tle.dsa.to_tensor(buffer1, writable=True)
```

### 3.2 Use the converted tensor in tensor computation

The following example follows the usage pattern in `python/test/tle/test_vec_add_mix.py`:

```python
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

### 3.3 About `target_shape`

The current implementation supports passing `target_shape`. A layout conversion is performed before converting the buffer to a tensor.

Notes:

- `target_shape` is unwrapped through `tl._unwrap_shape(target_shape)`.
- `target_shape` cannot be exactly the same as the original `memref.shape`.
- The final shape must be a `list` or `tuple`.
- Whether a specific shape transformation is legal depends on the capabilities of the underlying `create_convert_layout` implementation.

## 4. Semantics

- `tle.dsa.to_tensor` is an explicit buffer-to-tensor conversion API.
- It does not modify the original buffer object; it returns a new `tl.tensor` view/value.
- Whether the returned tensor is writable is controlled by the `writable` parameter.
- Use `tle.dsa.to_buffer` to convert a tensor back to a buffer.
