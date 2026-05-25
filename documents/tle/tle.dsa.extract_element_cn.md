# tle.dsa.extract_element

## 1. OP 概述

`tle.dsa.extract_element` 用于从有 rank 的 tensor 中按给定索引提取单个元素。

```python
tle.dsa.extract_element(
    src,
    indice,
    _builder=None,
    _generator=None,
) -> tensor
```

该接口是 TLE DSA builtin，只能在 `@triton.jit` 函数中使用。

## 2. OP 规格

### 2.1 参数说明

| 参数名 | 类型 | 说明 |
|--------|------|------|
| `src` | `tl.tensor` | 要访问的源 tensor，必须是有 rank 的 tensor |
| `indice` | `List[tensor]` / `tuple` | 指定元素位置的索引，每个维度一个索引；`tl.constexpr` 会被转换为 tensor，其它值按 tensor 处理 |
| `_builder` | - | 编译器内部参数，不支持外部调用 |
| `_generator` | - | 编译器内部参数，不支持外部调用 |

返回值：

- `tl.tensor`：提取出的标量 tensor，元素类型与 `src` 的元素类型相同，shape 为 `None`。

### 2.2 参数约束

根据当前实现，需满足以下约束：

1. `src` 必须是有 rank 的 tensor，即 `len(src.shape) > 0`。
2. `indice` 的长度必须与 `src` 的 rank 相同。
3. `indice` 中的 `tl.constexpr` 会被转换为 tensor；非 `tl.constexpr` 值应本身就是可用于 DSA IR 的 tensor。普通 Python 整数字面量在 JIT AST 中通常会以 `tl.constexpr` 形式进入该转换路径。

如果 `indice` 的长度与 `src` 的 rank 不一致，semantic 层会抛出：

```text
ValueError: Indice's rank must be equal to src tensor's rank
```

### 2.3 返回类型

`extract_element` 对应底层 DSA IR 的 `create_dsa_extract_scalar`，返回值通过 `wrap_tensor(result, src.type.scalar, None)` 包装。

```text
返回元素类型 = src.type.scalar
返回 shape   = None
```

因此，返回值是标量 tensor，可继续参与 Triton 表达式计算。

## 3. 使用方法

以下示例从一维索引 tensor 中逐元素取出索引值，并用于手写 gather：

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

二维 tensor 示例：

```python
@triton.jit
def extract_element_2d_kernel(x_ptr, out_ptr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    rows = tl.arange(0, BLOCK_M)[:, None]
    cols = tl.arange(0, BLOCK_N)[None, :]
    x = tl.load(x_ptr + rows * BLOCK_N + cols)

    value = tle.dsa.extract_element(x, (0, 0))
    tl.store(out_ptr, value)
```

## 4. 语义说明

`extract_element` 只提取单个元素，`indice` 的 arity 必须等于源 tensor 的 rank。若需要提取一个子 tensor，应使用 `tle.dsa.extract_slice`。
