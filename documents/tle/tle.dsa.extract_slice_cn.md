# tle.dsa.extract_slice

## 1. OP 概述

`tle.dsa.extract_slice` 用于从输入张量中按指定偏移、大小和步长提取一个子张量。

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

该接口是 TLE DSA builtin，只能在 `@triton.jit` 函数中使用。

## 2. OP 规格

### 2.1 参数说明

| 参数名 | 类型 | 说明 |
|--------|------|------|
| `ful` | `tl.tensor` | 要提取切片的源张量，必须是有 rank 的 tensor |
| `offsets` | `List[tensor]` / `tuple` | 切片在每个维度上的起始偏移；`tl.constexpr` 会被转换为 tensor，其它值按 tensor 处理 |
| `sizes` | `List[int]` / `tuple` | 切片在每个维度上的大小 |
| `strides` | `List[int]` / `tuple` | 切片在每个维度上的步长 |
| `_builder` | - | 编译器内部参数，不支持外部调用 |
| `_generator` | - | 编译器内部参数，不支持外部调用 |

返回值：

- `tl.tensor`：提取得到的子张量。

### 2.2 参数约束

根据当前实现，需满足以下约束：

1. `ful` 必须是有 rank 的 tensor，即 `len(ful.shape) > 0`。
2. `offsets`、`sizes`、`strides` 的长度必须与 `ful` 的 rank 相同。
3. `sizes` 中每个元素必须大于等于 `1`。
4. `strides` 中每个元素必须大于等于 `0`。
5. `offsets` 中的 `tl.constexpr` 会被转换为 tensor；非 `tl.constexpr` 值应本身就是可用于 DSA IR 的 tensor。普通 Python 整数字面量在 JIT AST 中通常会以 `tl.constexpr` 形式进入该转换路径。

### 2.3 返回 shape

`extract_slice` 的返回 tensor 类型由 `ful` 的元素类型和 `sizes` 决定：

```text
返回元素类型 = ful.type.scalar
返回 shape   = sizes
```

因此，`sizes` 不只是切片大小参数，也决定返回 tensor 的 shape。

## 3. 使用方法

以下示例从一维 block tensor 中提取前 32 个元素并写回输出：

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

二维 tensor 示例：

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

## 4. 语义说明

`extract_slice` 对应底层 DSA IR 的 `create_dsa_extract_slice`。接口返回一个新的 tensor 值，返回 tensor 的 shape 与 `sizes` 一致。
