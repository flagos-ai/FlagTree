# tle.dsa.subview

## 1. OP Overview

`tle.dsa.subview` creates a subview buffer from an existing TLE DSA `buffer`. The subview is described by `offsets`, `sizes`, and `strides`. It does not copy the underlying data, and the return value is still a TLE DSA `buffer`.

```python
tle.dsa.subview(
    src: buffer,
    offsets: List[tl.constexpr],
    sizes: List[tl.constexpr],
    strides: List[tl.constexpr],
    _builder=None,
) -> buffer
```

This API is a TLE DSA builtin and can only be used inside `@triton.jit` functions.

## 2. OP Specification

### 2.1 Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `src` | `buffer` | Yes | Input TLE DSA buffer. |
| `offsets` | `List[int]` / `List[tl.constexpr]` / `List[tl.tensor]` | Yes | Start offset in each dimension. |
| `sizes` | `List[int]` / `List[tl.constexpr]` | Yes | Size of the subview in each dimension. |
| `strides` | `List[int]` / `List[tl.constexpr]` | Yes | Stride of the subview in each dimension. |
| `_builder` | - | No | Compiler-internal parameter. Do not pass it from user code. |

Return value:

- `buffer`: a subview buffer created from `src`.

### 2.2 Constraints

The following constraints apply:

1. `src` must be a TLE DSA `buffer`.
2. Elements of `sizes` must be `int` or `tl.constexpr`.
3. Elements of `strides` must be `int` or `tl.constexpr`.
4. Elements of `offsets` can be `int`, `tl.constexpr`, or already constructed tensors.
5. When an `offsets` element is an `int` or `tl.constexpr`, its value must be non-negative.
6. `sizes` and `strides` are unwrapped through `tl._unwrap_shape` before being passed to the underlying DSA subview builder.

### 2.3 Return Type

The returned buffer inherits the element type and address space from `src`; its shape is determined by `sizes`:

```text
return element type = src.dtype
return shape        = sizes
return space        = src.space
```

The memory strides of the returned buffer are computed from the source buffer memory strides and the provided `strides`:

```text
result_memory_strides[i] = src_memory_strides[i] * strides[i]
```

If `src` does not have explicit strides, the implementation computes default row-major strides from `src.shape`.

## 3. Usage

### 3.1 Create a subview from a 2D UB buffer

```python
import triton
import triton.language as tl
import triton.experimental.tle as tle


@triton.jit
def subview_kernel(BLOCK_SIZE: tl.constexpr):
    src = tle.dsa.alloc(
        [BLOCK_SIZE, BLOCK_SIZE],
        dtype=tl.float32,
        mem_addr_space=tle.dsa.ascend.UB,
    )

    view = tle.dsa.subview(
        src,
        offsets=[1, 0],
        sizes=[BLOCK_SIZE - 2, BLOCK_SIZE],
        strides=[1, 1],
    )
```

### 3.2 Create a subview with constexpr parameters

```python
@triton.jit
def subview_with_params(
    BLOCK_SIZE: tl.constexpr,
    offset: tl.constexpr,
    size: tl.constexpr,
    stride: tl.constexpr,
):
    src = tle.dsa.alloc(
        [BLOCK_SIZE, BLOCK_SIZE],
        dtype=tl.float32,
        mem_addr_space=tle.dsa.ascend.UB,
    )

    view = tle.dsa.subview(
        src,
        offsets=[offset, 0],
        sizes=[size, BLOCK_SIZE],
        strides=[stride, 1],
    )
```

### 3.3 Use with `tle.dsa.copy`

`tle.dsa.subview` returns a buffer, so it can continue to participate in DSA buffer operations. For example, a GM region can be copied into a subview:

```python
@triton.jit
def copy_to_subview(x_ptr, BLOCK_SIZE: tl.constexpr):
    offsets = tl.arange(0, BLOCK_SIZE)

    src = tle.dsa.alloc(
        [BLOCK_SIZE, BLOCK_SIZE],
        dtype=tl.float32,
        mem_addr_space=tle.dsa.ascend.UB,
    )
    view = tle.dsa.subview(
        src,
        offsets=[1, 0],
        sizes=[1, BLOCK_SIZE],
        strides=[1, 1],
    )

    tle.dsa.copy(x_ptr + offsets, view, [1, BLOCK_SIZE])
```

## 4. Semantics

- `tle.dsa.subview` does not copy data; it creates a subview buffer pointing to a local region of the original buffer.
- The returned buffer has the same dtype and address space as the source buffer.
- The returned buffer shape is determined by `sizes`.
- Plain integers and `tl.constexpr` values in `offsets` are converted to tensors before being passed to the underlying subview operation.
- `sizes` and `strides` only accept plain integers or `tl.constexpr`; tensors are not accepted.
