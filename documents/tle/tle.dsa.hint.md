# tle.dsa.hint

## 1. 函数概述

`hint` 是 TLE DSA 提供的 JIT 编译期作用域提示接口，用于通过 `with` 语句给作用域内的 TLE DSA builtin 传递编译提示。

```python
with tle.dsa.hint(inter_no_alias=True):
    ...
```

当前实现中，`hint` 主要用于向作用域内的 `tle.dsa.copy` 自动传递 `inter_no_alias` 参数。

`tle.dsa.hint` 与 `extension.compile_hint` 互为补充：`tle.dsa.hint` 适用于 `tle.dsa.copy(...)` 这类没有左值接收结果的语句级场景，`extension.compile_hint` 适用于已经有左值张量或指针变量、需要给该值附加编译元数据的场景。

## 2. 规格

### 2.1 参数说明

| 参数 | 类型 | 默认值 | 含义说明 |
|------|------|--------|----------|
| `inter_no_alias` | Python 常量 `bool` | 无 | 标记不同迭代之间的 copy 操作不存在 alias 关系 |

`hint` 接收关键字参数，参数值必须是 Python AST 常量。例如 `True`、`False` 这类字面量可以使用，运行时变量不支持。

### 2.2 作用范围

`hint` 通过 `with` 语句建立作用域。编译器在访问 `tle.dsa.copy` 调用时，会从当前嵌套的 `with` 作用域由内向外查找最近的 `inter_no_alias` 提示。

建议尽量避免嵌套使用 `tle.dsa.hint`。虽然当前实现支持从内到外查找最近的 hint，但嵌套作用域会降低代码可读性，也容易让同一段代码中实际生效的 `inter_no_alias` 值变得不直观。

```python
with tle.dsa.hint(inter_no_alias=True):
    tle.dsa.copy(src0, dst0, [size])   # inter_no_alias=True

    with tle.dsa.hint(inter_no_alias=False):
        tle.dsa.copy(src1, dst1, [size])  # inter_no_alias=False

    tle.dsa.copy(src2, dst2, [size])   # inter_no_alias=True
```

如果 `tle.dsa.copy` 显式传入了 `inter_no_alias`，则显式参数优先，`hint` 不会覆盖该参数。

```python
with tle.dsa.hint(inter_no_alias=True):
    tle.dsa.copy(src, dst, [size], inter_no_alias=False)  # 使用显式传入的 False
```

### 2.3 生效接口

当前实现只在调用名为 `copy` 的 builtin 时提取并应用 `hint` 作用域中的 `inter_no_alias`。

等价关系如下：

```python
with tle.dsa.hint(inter_no_alias=True):
    tle.dsa.copy(src, dst, [size])
```

等价于：

```python
tle.dsa.copy(src, dst, [size], inter_no_alias=True)
```

## 3. 与 `extension.compile_hint` 的关系

`compile_hint` 是 Ascend 后端扩展提供的编译提示接口，位于 `triton.language.extra.cann.extension` 命名空间下，用于给某个已有张量附加元数据信息，后端可据此指导优化和代码生成。

```python
import triton.language.extra.cann.extension as extension

extension.compile_hint(ptr, hint_name, hint_val=None)
```

### 3.1 `extension.compile_hint` 参数说明

| 参数 | 类型 | 默认值 | 含义说明 |
|------|------|--------|----------|
| `ptr` | `tensor` | 必需 | 需要附加提示的张量对象 |
| `hint_name` | `str` / `tl.constexpr` | 必需 | 提示名称，必须表示字符串 |
| `hint_val` | `None` / `bool` / `int` / `tl.constexpr` / `list` | `None` | 提示值 |

`extension.compile_hint` 不改变计算语义，只给指定张量附加编译期元数据。同一个张量可以多次标注不同 hint。

### 3.2 适用场景对比

| 接口 | 适用场景 | 典型形式 |
|------|----------|----------|
| `tle.dsa.hint` | 没有左值的语句级调用，需要给作用域内的 builtin 传递提示 | `with tle.dsa.hint(inter_no_alias=True): tle.dsa.copy(...)` |
| `extension.compile_hint` | 有左值的张量或指针变量，需要给该值附加提示 | `extension.compile_hint(tmp, "hint_a")` |

例如，`tle.dsa.copy(...)` 本身没有返回值，也没有左值可以承载 hint，因此使用 `with tle.dsa.hint(...)` 包裹调用；而 `tl.load(...)` 的结果通常会赋给变量，可以对该变量使用 `extension.compile_hint(...)`。

### 3.3 `extension.compile_hint` 使用示例

```python
import triton.language.extra.cann.extension as extension


@triton.jit
def triton_compile_hint(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr, XBLOCK_SUB: tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    for xoffset_sub in range(0, XBLOCK, XBLOCK_SUB):
        xindex = xoffset + xoffset_sub + tl.arange(0, XBLOCK_SUB)[:]
        xmask = xindex < xnumel
        tmp0 = tl.load(in_ptr0 + xindex, xmask)

        extension.compile_hint(tmp0, "hint_a")

        tmp2 = tmp0
        extension.compile_hint(tmp2, "hint_b", 42)
        extension.compile_hint(tmp2, "hint_c", True)
        extension.compile_hint(tmp2, "hint_d", [XBLOCK, XBLOCK_SUB])

        tl.store(out_ptr0 + xindex, tmp2, xmask)
```

### 3.4 `extension.compile_hint` 限制说明

- `hint_name` 必须为字符串类型。
- `hint_val` 支持 `None`、布尔值、整数、`tl.constexpr` 和列表。
- 列表形式的 `hint_val` 仅支持整数数组，不支持浮点数或混合类型列表。
- `extension.compile_hint` 需要作用在已有张量值上，不适合 `tle.dsa.copy(...)` 这类无返回值调用。

## 4. `tle.dsa.hint` 使用方法

```python
import triton
import triton.language as tl
import triton.experimental.tle as tle


@triton.jit
def hint_kernel(result, index_value, k_cache_ptr, row_ids, token_num: tl.constexpr, HEAD_DIM: tl.constexpr,
                BLOCK_SIZE_TOKEN: tl.constexpr):
    for i in range(BLOCK_SIZE_TOKEN):
        offset_i = i
        if offset_i < token_num:
            reload_result = tle.dsa.extract_slice(result, (i, 0), (1, HEAD_DIM), (1, 1))
            reload_result = tl.reshape(reload_result, (HEAD_DIM,))

            k_cache_offset = tle.dsa.extract_element(index_value, (i,)) * HEAD_DIM
            res_buf = tle.dsa.to_buffer(reload_result)

            with tle.dsa.hint(inter_no_alias=True):
                tle.dsa.copy(
                    res_buf,
                    k_cache_ptr + k_cache_offset + row_ids,
                    [HEAD_DIM],
                )
```

上例中，`with tle.dsa.hint(inter_no_alias=True)` 会使作用域内的 `tle.dsa.copy` 获得 `inter_no_alias=True` 参数，用于表达不同迭代之间的 copy 目标不存在 alias。

## 5. 限制说明

- `hint` 只能作为 `with` 上下文使用。
- `hint` 本身不会在 JIT 运行时执行；它只用于 AST 解析和编译期提示提取。
- 关键字参数值必须是常量，不支持运行时变量。
- 当前只有 `inter_no_alias` 会被应用。
- 当前只有 `tle.dsa.copy` 会消费 `inter_no_alias` hint。
- 嵌套作用域中，离 `tle.dsa.copy` 最近的 `inter_no_alias` hint 生效；建议尽量避免嵌套使用 `tle.dsa.hint`，优先使用清晰的单层作用域。
