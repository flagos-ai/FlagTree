# tle.dsa.extract_element

## 1. OP Overview

`tle.dsa.extract_element` extracts a single element from a ranked tensor using the given indices.

```python
tle.dsa.extract_element(
    src,
    indice,
    _builder=None,
    _generator=None,
) -> tensor
```

This API is a TLE DSA builtin and can only be used inside `@triton.jit` functions.

## 2. OP Specification

### 2.1 Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `src` | `tl.tensor` | Source tensor to access. It must be a ranked tensor. |
| `indice` | `List[tensor]` / `tuple` | Indices specifying the element position, one index per dimension. `tl.constexpr` values are converted to tensors; other values are treated as tensors. |
| `_builder` | - | Compiler-internal parameter. Do not pass it from user code. |
| `_generator` | - | Compiler-internal parameter. Do not pass it from user code. |

Return value:

- `tl.tensor`: the extracted scalar tensor, with the same element type as `src` and shape `None`.

### 2.2 Constraints

The current implementation requires:

1. `src` must be a ranked tensor, that is, `len(src.shape) > 0`.
2. The length of `indice` must match the rank of `src`.
3. `tl.constexpr` values in `indice` are converted to tensors. Non-`tl.constexpr` values should already be tensors usable by DSA IR. Plain Python integer literals usually enter this conversion path as `tl.constexpr` values in JIT AST.

If the length of `indice` does not match the rank of `src`, the semantic layer raises:

```text
ValueError: Indice's rank must be equal to src tensor's rank
```

### 2.3 Return Type

`extract_element` maps to the underlying DSA IR `create_dsa_extract_scalar`, and the result is wrapped through `wrap_tensor(result, src.type.scalar, None)`.

```text
return element type = src.type.scalar
return shape        = None
```

Therefore, the return value is a scalar tensor and can participate in Triton expression computation.

## 3. Usage

The following example extracts indices from a 1D index tensor and uses them for a manual gather:

```python
import triton
import triton.language as tl
import triton.experimental.tle as tle


@triton.jit
def index_select_manual_kernel(in_ptr, indices_ptr, out_ptr, g_stride: tl.constexpr,
                               indice_length: tl.constexpr, g_block: tl.constexpr,
                               g_block_sub: tl.constexpr, other_block: tl.constexpr):
    g_begin = tl.program_id(0) * g_block

    for goffs in range(0, g_block, g_block_sub):
        g_idx = tl.arange(0, g_block_sub) + g_begin + goffs
        g_mask = g_idx < indice_length
        indices = tl.load(indices_ptr + g_idx, g_mask, other=0)

        for other_offset in range(0, g_stride, other_block):
            tmp_buf = tl.zeros((g_block_sub, other_block), in_ptr.dtype.element_ty)
            other_idx = tl.arange(0, other_block) + other_offset
            other_mask = other_idx < g_stride

            for i in range(0, g_block_sub):
                gather_offset = tle.dsa.extract_element(indices, (i,)) * g_stride
                val = tl.load(in_ptr + gather_offset + other_idx, other_mask)
                tmp_buf = tle.dsa.insert_slice(
                    tmp_buf,
                    val[None, :],
                    offsets=(i, 0),
                    sizes=(1, other_block),
                    strides=(1, 1),
                )

            tl.store(
                out_ptr + g_idx[:, None] * g_stride + other_idx[None, :],
                tmp_buf,
                g_mask[:, None] & other_mask[None, :],
            )
```

2D tensor example:

```python
@triton.jit
def extract_element_2d_kernel(x_ptr, out_ptr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    rows = tl.arange(0, BLOCK_M)[:, None]
    cols = tl.arange(0, BLOCK_N)[None, :]
    x = tl.load(x_ptr + rows * BLOCK_N + cols)

    value = tle.dsa.extract_element(x, (0, 0))
    tl.store(out_ptr, value)
```

## 4. Semantics

`extract_element` extracts exactly one element. The arity of `indice` must equal the rank of the source tensor. Use `tle.dsa.extract_slice` to extract a subtensor.
