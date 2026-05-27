# tle.dsa.subview

## 1. OP 概述

`tle.dsa.subview` 用于基于已有 TLE DSA `buffer` 创建一个子视图 buffer。子视图通过 `offsets`、`sizes` 和 `strides` 描述访问区域，不复制底层数据，返回的仍然是 TLE DSA `buffer`。

```python
tle.dsa.subview(
    src: buffer,
    offsets: List[tl.constexpr],
    sizes: List[tl.constexpr],
    strides: List[tl.constexpr],
    _builder=None,
) -> buffer
```

该接口是 TLE DSA builtin，只能在 `@triton.jit` 函数中使用。

## 2. OP 规格

### 2.1 参数说明

| 参数名 | 类型 | 必需 | 说明 |
|--------|------|------|------|
| `src` | `buffer` | 是 | 输入的 TLE DSA buffer |
| `offsets` | `List[int]` / `List[tl.constexpr]` / `List[tl.tensor]` | 是 | 每个维度的起始偏移 |
| `sizes` | `List[int]` / `List[tl.constexpr]` | 是 | 子视图每个维度的大小 |
| `strides` | `List[int]` / `List[tl.constexpr]` | 是 | 子视图每个维度的步长 |
| `_builder` | - | 否 | 编译器内部参数，不支持外部调用 |

返回值：

- `buffer`：由 `src` 创建得到的子视图 buffer。

### 2.2 参数约束

需满足以下约束：

1. `src` 必须是 TLE DSA `buffer`。
2. `sizes` 中的元素必须是 `int` 或 `tl.constexpr`。
3. `strides` 中的元素必须是 `int` 或 `tl.constexpr`。
4. `offsets` 中的元素可以是 `int`、`tl.constexpr` 或已构造好的 tensor。
5. 当 `offsets` 元素是 `int` 或 `tl.constexpr` 时，值必须非负。
6. `sizes` 和 `strides` 会通过 `tl._unwrap_shape` 解包后传入底层 DSA subview builder。

### 2.3 返回类型

返回 buffer 的元素类型和地址空间继承自 `src`，shape 由 `sizes` 决定：

```text
返回元素类型 = src.dtype
返回 shape   = sizes
返回 space   = src.space
```

返回 buffer 的 memory strides 根据源 buffer 的 memory strides 和传入的 `strides` 计算：

```text
result_memory_strides[i] = src_memory_strides[i] * strides[i]
```

如果 `src` 本身没有显式 strides，实现会按 row-major 规则根据 `src.shape` 计算默认 strides。

## 3. 使用方法

### 3.1 从二维 UB buffer 创建子视图

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

### 3.2 使用 constexpr 参数创建子视图

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

### 3.3 与 `tle.dsa.copy` 配合

`tle.dsa.subview` 返回的是 buffer，因此可继续作为 DSA buffer 参与后续操作。例如可以将 GM 中的一块数据拷贝到子视图中：

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

## 4. 语义说明

- `tle.dsa.subview` 不复制数据，而是创建指向原 buffer 局部区域的子视图 buffer。
- 返回 buffer 的 dtype 和 address space 与源 buffer 相同。
- 返回 buffer 的 shape 由 `sizes` 决定。
- `offsets` 中的普通整数和 `tl.constexpr` 会先转换为 tensor，再传入底层 subview 操作。
- `sizes` 和 `strides` 只接受普通整数或 `tl.constexpr`，不能传入 tensor。
