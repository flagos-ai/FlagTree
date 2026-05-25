# tle.dsa.insert_slice

## 1. OP 概述

`tle.dsa.insert_slice` 用于把一个子张量插入到另一个张量的指定切片区域中，并返回插入后的新张量。

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

该接口是 TLE DSA builtin，只能在 `@triton.jit` 函数中使用。

## 2. OP 规格

### 2.1 参数说明

| 参数名 | 类型 | 说明 |
|--------|------|------|
| `ful` | `tl.tensor` | 接收插入的目标张量，必须是有 rank 的 tensor |
| `sub` | `tl.tensor` | 要插入的子张量，rank 必须与 `ful` 相同 |
| `offsets` | `List[tensor]` / `tuple` | 插入区域在每个维度上的起始偏移；`tl.constexpr` 会被转换为 tensor，其它值按 tensor 处理 |
| `sizes` | `List[int]` / `tuple` | 插入区域在每个维度上的大小 |
| `strides` | `List[int]` / `tuple` | 插入区域在每个维度上的步长 |
| `_builder` | - | 编译器内部参数，不支持外部调用 |

返回值：

- `tl.tensor`：插入后的新张量，元素类型和 shape 与 `ful` 相同。

### 2.2 参数约束

根据当前实现，需满足以下约束：

1. `ful` 必须是有 rank 的 tensor，即 `len(ful.shape) > 0`。
2. `ful` 和 `sub` 的 rank 必须相同。
3. `offsets`、`sizes`、`strides` 的长度必须与 `ful` 的 rank 相同。
4. `sizes` 中每个元素必须大于等于 `1`。
5. `strides` 中每个元素必须大于等于 `0`。
6. `offsets` 中的 `tl.constexpr` 会被转换为 tensor；非 `tl.constexpr` 值应本身就是可用于 DSA IR 的 tensor。普通 Python 整数字面量在 JIT AST 中通常会以 `tl.constexpr` 形式进入该转换路径。

### 2.3 返回 shape

`insert_slice` 的返回 tensor 类型由 `ful` 决定：

```text
返回元素类型 = ful.type.scalar
返回 shape   = ful.shape
```

也就是说，`insert_slice` 不会返回 `sub` 的 shape，而是返回已经插入子张量后的完整 `ful` 形状。

## 3. 使用方法

以下示例从输入 tensor 中取出一段子张量，计算后插入回原 tensor：

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

## 4. 语义说明

`insert_slice` 对应底层 DSA IR 的 `create_dsa_insert_slice`。接口返回一个新的 tensor 值，不会原地修改 Python 变量本身；如果后续要使用插入后的结果，需要接收返回值。
