# tle.dsa.insert_slice

## 1. OP Overview

`tle.dsa.insert_slice` inserts a subtensor into a specified slice region of another tensor and returns the resulting new tensor.

```python
tle.dsa.insert_slice(
    ful,
    sub,
    offsets,
    sizes,
    strides,
    _builder=None,
) -> tensor
```

This API is a TLE DSA builtin and can only be used inside `@triton.jit` functions.

## 2. OP Specification

### 2.1 Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `ful` | `tl.tensor` | Destination tensor that receives the insertion. It must be a ranked tensor. |
| `sub` | `tl.tensor` | Subtensor to insert. Its rank must match `ful`. |
| `offsets` | `List[tensor]` / `tuple` | Start offset of the insertion region in each dimension. `tl.constexpr` values are converted to tensors; other values are treated as tensors. |
| `sizes` | `List[int]` / `tuple` | Size of the insertion region in each dimension. |
| `strides` | `List[int]` / `tuple` | Stride of the insertion region in each dimension. |
| `_builder` | - | Compiler-internal parameter. Do not pass it from user code. |

Return value:

- `tl.tensor`: the new tensor after insertion, with the same element type and shape as `ful`.

### 2.2 Constraints

The current implementation requires:

1. `ful` must be a ranked tensor, that is, `len(ful.shape) > 0`.
2. `ful` and `sub` must have the same rank.
3. The lengths of `offsets`, `sizes`, and `strides` must match the rank of `ful`.
4. Each element of `sizes` must be greater than or equal to `1`.
5. Each element of `strides` must be greater than or equal to `0`.
6. `tl.constexpr` values in `offsets` are converted to tensors. Non-`tl.constexpr` values should already be tensors usable by DSA IR. Plain Python integer literals usually enter this conversion path as `tl.constexpr` values in JIT AST.

### 2.3 Return Shape

The return tensor type is determined by `ful`:

```text
return element type = ful.type.scalar
return shape        = ful.shape
```

In other words, `insert_slice` does not return the shape of `sub`; it returns the full shape of `ful` after the subtensor is inserted.

## 3. Usage

The following example extracts slices from input tensors, computes on them, and inserts the result back into a full tensor:

```python
import triton
import triton.language as tl
import triton.experimental.tle as tle


@triton.jit
def insert_slice_kernel(x_ptr, y_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr,
                        SLICE_OFFSET: tl.constexpr, SLICE_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = tl.load(y_ptr + offsets, mask=mask, other=0.0)

    x_sub = tle.dsa.extract_slice(x, (SLICE_OFFSET,), (SLICE_SIZE,), (1,))
    y_sub = tle.dsa.extract_slice(y, (SLICE_OFFSET,), (SLICE_SIZE,), (1,))
    out_sub = x_sub + y_sub

    out = tl.full((BLOCK_SIZE,), 0.0, tl.float32)
    out = tle.dsa.insert_slice(out, out_sub, (SLICE_OFFSET,), (SLICE_SIZE,), (1,))

    tl.store(output_ptr + offsets, out, mask=mask)
```

## 4. Semantics

`insert_slice` maps to the underlying DSA IR `create_dsa_insert_slice`. It returns a new tensor value and does not mutate the Python variable in place. Use the returned value for subsequent computation.
