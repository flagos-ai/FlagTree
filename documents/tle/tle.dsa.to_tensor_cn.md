# tle.dsa.to_tensor

## 1. OP 概述

`tle.dsa.to_tensor` 用于将 TLE DSA `buffer` 显式转换为 `tl.tensor`。转换后的 tensor 可以继续参与 Triton 表达式计算，也可以作为后续 `tl.store`、算术表达式或其它 tensor 接口的输入。

```python
tle.dsa.to_tensor(
    memref: buffer,
    writable: bool = True,
    target_shape=None,
    _builder=None,
) -> tl.tensor
```

该接口是 TLE DSA builtin，只能在 `@triton.jit` 函数中使用。

## 2. OP 规格

### 2.1 参数说明

| 参数名 | 类型 | 必需 | 说明 |
|--------|------|------|------|
| `memref` | `buffer` | 是 | 输入的 TLE DSA buffer |
| `writable` | `bool` | 否 | 是否将返回 tensor 标记为 writable，默认 `True` |
| `target_shape` | `List[int]` / `tuple` / `None` | 否 | 目标 tensor shape；不传时使用 `memref.shape` |
| `_builder` | - | 否 | 编译器内部参数，不支持外部调用 |

返回值：

- `tl.tensor`：由 `memref` 转换得到的 tensor，元素类型与 `memref.dtype` 相同。

### 2.2 参数约束

需满足以下约束：

1. `memref` 必须是 TLE DSA `buffer`。
2. 默认情况下返回 tensor 的 shape 使用 `memref.shape`。

### 2.3 返回类型

返回 tensor 类型由 buffer 的 dtype 和最终 shape 决定：

```text
返回元素类型 = memref.dtype
返回 shape   = target_shape if target_shape is not None else memref.shape
```

当设置了 `target_shape` 时，实现会先创建目标 buffer type，并通过 `builder.create_convert_layout` 转换 layout，然后再执行 `builder.dsa_to_tensor(memref_value, writable)`。

## 3. 使用方法

### 3.1 将 alloc 得到的 buffer 转为 tensor

以下示例来自 `python/test/tle/test_bind_buffer.py` 的用法模式：

```python
import triton
import triton.language as tl
import triton.experimental.tle as tle


@triton.jit
def bind_buffer():
    buffer1 = tle.dsa.alloc(shape=[32, 32], dtype=tl.float32, mem_addr_space=tle.dsa.ascend.UB)
    tensor1 = tle.dsa.to_tensor(buffer1, writable=True)
```

### 3.2 转换后参与 tensor 计算

以下示例来自 `python/test/tle/test_vec_add_mix.py` 的用法模式：

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

### 3.3 关于 `target_shape`

当前实现支持传入 `target_shape`，会在转换为 tensor 前先执行一次 layout 转换。

使用时需要注意：

- `target_shape` 会通过 `tl._unwrap_shape(target_shape)` 解包。
- `target_shape` 不能与原始 `memref.shape` 完全相同。
- 最终 shape 必须是 `list` 或 `tuple`。
- 具体 shape 变换是否合法，应以底层 `create_convert_layout` 支持能力为准。

## 4. 语义说明

- `tle.dsa.to_tensor` 是 buffer 到 tensor 的显式转换接口。
- 该接口不会改变原始 buffer 对象本身，而是返回一个新的 `tl.tensor` 视图/值。
- 返回 tensor 是否 writable 由 `writable` 参数控制。
- 如需将 tensor 转回 buffer，应使用 `tle.dsa.to_buffer`。
