# tle.dsa.to_buffer

## 1. OP 概述

`tle.dsa.to_buffer` 用于将 `tl.tensor` 显式转换为 TLE DSA `buffer`。该接口通常用于 tensor 计算完成后，将结果重新转换为 buffer，以便继续通过 `tle.dsa.copy` 等 DSA 接口写回 GM 或参与后续 DSA 操作。

```python
tle.dsa.to_buffer(
    tensor: tl.tensor,
    space: address_space,
    bind_buffer: buffer = None,
    _builder=None,
) -> buffer
```

该接口是 TLE DSA builtin，只能在 `@triton.jit` 函数中使用。

## 2. OP 规格

### 2.1 参数说明

| 参数名 | 类型 | 必需 | 说明 |
|--------|------|------|------|
| `tensor` | `tl.tensor` | 是 | 输入 tensor，必须是非标量 tensor |
| `space` | `address_space` | 是 | 目标 buffer 的地址空间，例如 `tle.dsa.ascend.UB` |
| `bind_buffer` | `buffer` / `None` | 否 | 当前实现暂不支持绑定已有 buffer，必须为 `None` |
| `_builder` | - | 否 | 编译器内部参数，不支持外部调用 |

返回值：

- `buffer`：由输入 tensor 转换得到的 TLE DSA buffer。

### 2.2 参数约束

需满足以下约束：

1. `tensor.shape` 必须是非空 `list` 或 `tuple`。
2. scalar tensor 不能转换为 buffer。
3. `space` 必须显式传入后端地址空间，例如 `tle.dsa.ascend.UB`。
4. `bind_buffer` 当前必须为 `None`；传入非 `None` 值会报错。

### 2.3 返回类型

返回 buffer 类型由输入 tensor 和目标地址空间决定：

```text
返回元素类型 = tensor.dtype
返回 shape   = tensor.shape
返回 space   = space
```

底层实现会调用 `builder.dsa_to_buffer(tensor.handle, addr_space_attr)` 创建 buffer handle。

## 3. 使用方法

### 3.1 将 tensor 计算结果转为 buffer 后写回

以下示例来自 `python/test/tle/test_vec_add_mix.py` 的用法模式：

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

### 3.2 与 `tle.dsa.to_tensor` 配合

`tle.dsa.to_tensor` 和 `tle.dsa.to_buffer` 常配合使用：

```text
buffer -> tle.dsa.to_tensor -> tl.tensor 计算 -> tle.dsa.to_buffer -> buffer
```

例如：

```python
c_val = tle.dsa.to_tensor(c_ub)
b_val = tle.dsa.to_tensor(b_ub)
result = c_val - b_val
d_ub = tle.dsa.to_buffer(result, tle.dsa.ascend.UB)
```

## 4. 语义说明

- `tle.dsa.to_buffer` 不执行数据拷贝，而是将 tensor 值转换为 DSA buffer 表示。
- 转换后的 buffer 可继续作为 `tle.dsa.copy` 的输入。
- 当前实现不支持通过 `bind_buffer` 绑定已有 buffer。
- 如果需要从 buffer 转回 tensor，应使用 `tle.dsa.to_tensor`。
