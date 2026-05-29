# tle.dsa.extract_slice

## 1. OP Overview

`tle.dsa.extract_slice` extracts a subtensor from an input tensor using offsets, sizes, and strides.

```python
tle.dsa.extract_slice(
    ful,
    offsets,
    sizes,
    strides,
    _builder=None,
    _generator=None,
) -> tensor
```

This API is a TLE DSA builtin and can only be used inside `@triton.jit` functions.

## 2. OP Specification

### 2.1 Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `ful` | `tl.tensor` | Source tensor to slice. It must be a ranked tensor. |
| `offsets` | `List[tensor]` / `tuple` | Start offset in each dimension. `tl.constexpr` values are converted to tensors; other values are treated as tensors. |
| `sizes` | `List[int]` / `tuple` | Slice size in each dimension. |
| `strides` | `List[int]` / `tuple` | Slice stride in each dimension. |
| `_builder` | - | Compiler-internal parameter. Do not pass it from user code. |
| `_generator` | - | Compiler-internal parameter. Do not pass it from user code. |

Return value:

- `tl.tensor`: the extracted subtensor.

### 2.2 Constraints

The current implementation requires:

1. `ful` must be a ranked tensor, that is, `len(ful.shape) > 0`.
2. The lengths of `offsets`, `sizes`, and `strides` must match the rank of `ful`.
3. Each element of `sizes` must be greater than or equal to `1`.
4. Each element of `strides` must be greater than or equal to `0`.
5. `tl.constexpr` values in `offsets` are converted to tensors. Non-`tl.constexpr` values should already be tensors usable by DSA IR. Plain Python integer literals usually enter this conversion path as `tl.constexpr` values in JIT AST.

### 2.3 Return Shape

The return tensor type is determined by the element type of `ful` and `sizes`:

```text
return element type = ful.type.scalar
return shape        = sizes
```

Therefore, `sizes` not only specifies the slice extent, but also determines the returned tensor shape.

## 3. Usage

The following example extracts the first 32 elements from a 1D block tensor and writes them to output:

```python
import triton
import triton.language as tl
import triton.experimental.tle as tle


@triton.jit
def extract_slice_kernel(x_ptr, y_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = tl.load(y_ptr + offsets, mask=mask, other=0.0)
    out = x + y

    out_sub = tle.dsa.extract_slice(out, (0,), (32,), (1,))
    out_offsets = block_start + tl.arange(0, 32)
    out_mask = out_offsets < n_elements
    tl.store(output_ptr + out_offsets, out_sub, mask=out_mask)
```

2D tensor example:

```python
@triton.jit
def extract_slice_2d_kernel(x_ptr, output_ptr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    rows = tl.arange(0, BLOCK_M)[:, None]
    cols = tl.arange(0, BLOCK_N)[None, :]
    x = tl.load(x_ptr + rows * BLOCK_N + cols)

    sub = tle.dsa.extract_slice(x, (0, 0), (1, BLOCK_N), (1, 1))
    sub = tl.reshape(sub, (BLOCK_N,))

    tl.store(output_ptr + tl.arange(0, BLOCK_N), sub)
```

## 4. Semantics

`extract_slice` maps to the underlying DSA IR `create_dsa_extract_slice`. It returns a new tensor value whose shape matches `sizes`.
