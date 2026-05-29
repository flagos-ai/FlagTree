# tle.dsa.parallel

## 1. 函数概述

`parallel` 是 TLE DSA 提供的 JIT 专用循环迭代器，继承自 `tle.dsa.range`，用于表达循环迭代之间没有依赖、可按并行语义处理。

```python
tle.dsa.parallel(arg1, arg2=None, step=None, loop_unroll_factor=None)
```

该接口只能在 `@triton.jit` 修饰的函数中用于 `for` 循环，不能在普通 Python 代码中直接迭代。

## 2. 规格

### 2.1 参数说明

| 参数 | 类型 | 默认值 | 含义说明 |
|------|------|--------|----------|
| `arg1` | `int` / `tl.constexpr` / 标量值 | 必需 | 单参数调用时表示结束值，起始值默认为 `0`；双参数调用时表示起始值 |
| `arg2` | `int` / `tl.constexpr` / 标量值 | `None` | 结束值，不包含在迭代范围内 |
| `step` | `int` / `tl.constexpr` / 标量值 | `1` | 每次迭代的步长增量 |
| `loop_unroll_factor` | `int` | `None` | 传递给编译器的循环展开因子；小于 `2` 表示不展开 |

### 2.2 调用形式

```python
for i in tle.dsa.parallel(end):
    ...

for i in tle.dsa.parallel(start, end):
    ...

for i in tle.dsa.parallel(start, end, step):
    ...

for i in tle.dsa.parallel(start, end, step, loop_unroll_factor=4):
    ...
```

### 2.3 与 `tle.dsa.range` 的关系

`parallel` 继承自 `tle.dsa.range`，但构造函数只暴露以下参数：

- `arg1`
- `arg2`
- `step`
- `loop_unroll_factor`

`parallel` 不支持 `tle.dsa.range` 中的以下参数：

- `disallow_acc_multi_buffer`
- `flatten`
- `warp_specialize`
- `disable_licm`

### 2.4 类型支持

`parallel` 本身是循环迭代器，不直接对数据类型做算子级约束。循环变量类型由传入的 `start/end/step` 以及 Triton 编译语义决定。

## 3. 使用方法

```python
import triton
import triton.language as tl
import triton.experimental.tle as tle


@triton.jit
def parallel_kernel(input_ptr, output_ptr, n_elements: tl.constexpr):
    offsets = tl.arange(0, n_elements)
    acc = tl.full((n_elements,), 0, tl.int32)

    for i in tle.dsa.parallel(0, 4, 1):
        val = tl.load(input_ptr + offsets + i * n_elements)
        acc += val

    tl.store(output_ptr + offsets, acc)
```

## 4. 限制说明

- 只能在 `@triton.jit` 函数中使用。
- 只能作为 `for` 循环迭代器使用，不能在普通 Python 运行时直接调用 `iter()` 或 `next()`。
- `parallel` 表达的是循环迭代间无依赖的并行语义；如果循环体存在跨迭代数据依赖，不应使用该接口。
