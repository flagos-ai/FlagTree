# tle.dsa.hint

## 1. Function Overview

`hint` is a JIT compile-time scoped hint API provided by TLE DSA. It uses a `with` statement to pass compilation hints to TLE DSA builtins inside the scope.

```python
with tle.dsa.hint(inter_no_alias=True):
    ...
```

In the current implementation, `hint` is mainly used to automatically pass the `inter_no_alias` parameter to `tle.dsa.copy` calls inside the scope.

`tle.dsa.hint` and `extension.compile_hint` complement each other: `tle.dsa.hint` is suitable for statement-level calls without an lvalue, such as `tle.dsa.copy(...)`, while `extension.compile_hint` is suitable when an existing tensor or pointer value is available and compile-time metadata needs to be attached to that value.

## 2. Specification

### 2.1 Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `inter_no_alias` | Python constant `bool` | None | Marks copy operations from different iterations as non-aliasing. |

`hint` accepts keyword arguments whose values must be Python AST constants. Literal values such as `True` and `False` are supported; runtime variables are not supported.

### 2.2 Scope

`hint` creates a scope through a `with` statement. When the compiler visits a `tle.dsa.copy` call, it searches the current nested `with` scopes from inner to outer and uses the nearest `inter_no_alias` hint.

Avoid nesting `tle.dsa.hint` when possible. Although the current implementation supports searching from inner scopes to outer scopes, nested scopes reduce readability and can make the effective `inter_no_alias` value less obvious.

```python
with tle.dsa.hint(inter_no_alias=True):
    tle.dsa.copy(src0, dst0, [size])   # inter_no_alias=True

    with tle.dsa.hint(inter_no_alias=False):
        tle.dsa.copy(src1, dst1, [size])  # inter_no_alias=False

    tle.dsa.copy(src2, dst2, [size])   # inter_no_alias=True
```

If `tle.dsa.copy` explicitly passes `inter_no_alias`, the explicit argument takes precedence and the hint does not override it.

```python
with tle.dsa.hint(inter_no_alias=True):
    tle.dsa.copy(src, dst, [size], inter_no_alias=False)  # uses explicit False
```

### 2.3 Affected APIs

The current implementation only extracts and applies `inter_no_alias` hints for builtin calls named `copy`.

The following forms are equivalent:

```python
with tle.dsa.hint(inter_no_alias=True):
    tle.dsa.copy(src, dst, [size])
```

```python
tle.dsa.copy(src, dst, [size], inter_no_alias=True)
```

## 3. Relationship with `extension.compile_hint`

`compile_hint` is a compile hint API provided by the Ascend backend extension. It is located in the `triton.language.extra.cann.extension` namespace and attaches metadata to an existing tensor so the backend can use it for optimization and code generation.

```python
import triton.language.extra.cann.extension as extension

extension.compile_hint(ptr, hint_name, hint_val=None)
```

### 3.1 `extension.compile_hint` Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `ptr` | `tensor` | Required | Tensor object to annotate. |
| `hint_name` | `str` / `tl.constexpr` | Required | Hint name. It must represent a string. |
| `hint_val` | `None` / `bool` / `int` / `tl.constexpr` / `list` | `None` | Hint value. |

`extension.compile_hint` does not change computation semantics. It only attaches compile-time metadata to the specified tensor. The same tensor can be annotated with multiple hints.

### 3.2 Use Case Comparison

| API | Use Case | Typical Form |
|-----|----------|--------------|
| `tle.dsa.hint` | Statement-level calls without an lvalue need hints passed to builtins inside a scope. | `with tle.dsa.hint(inter_no_alias=True): tle.dsa.copy(...)` |
| `extension.compile_hint` | Existing tensor or pointer values need metadata attached to that value. | `extension.compile_hint(tmp, "hint_a")` |

For example, `tle.dsa.copy(...)` has no return value and no lvalue to carry a hint, so it is wrapped by `with tle.dsa.hint(...)`. In contrast, the result of `tl.load(...)` is usually assigned to a variable, so `extension.compile_hint(...)` can be applied to that variable.

### 3.3 `extension.compile_hint` Example

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

### 3.4 `extension.compile_hint` Limitations

- `hint_name` must be a string.
- `hint_val` supports `None`, booleans, integers, `tl.constexpr`, and lists.
- A list `hint_val` only supports integer arrays; floating-point or mixed-type lists are not supported.
- `extension.compile_hint` must be applied to an existing tensor value and is not suitable for no-return calls such as `tle.dsa.copy(...)`.

## 4. `tle.dsa.hint` Usage

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
            res_buf = tle.dsa.to_buffer(reload_result, tle.dsa.ascend.UB)

            with tle.dsa.hint(inter_no_alias=True):
                tle.dsa.copy(
                    res_buf,
                    k_cache_ptr + k_cache_offset + row_ids,
                    [HEAD_DIM],
                )
```

In this example, `with tle.dsa.hint(inter_no_alias=True)` passes `inter_no_alias=True` to the scoped `tle.dsa.copy`, expressing that copy destinations from different iterations do not alias.

## 5. Limitations

- `hint` can only be used as a `with` context.
- `hint` itself is not executed at JIT runtime; it is only used for AST parsing and compile-time hint extraction.
- Keyword argument values must be constants; runtime variables are not supported.
- Currently, only `inter_no_alias` is applied.
- Currently, only `tle.dsa.copy` consumes the `inter_no_alias` hint.
- In nested scopes, the nearest `inter_no_alias` hint to `tle.dsa.copy` takes effect; prefer a clear single-layer scope and avoid nesting `tle.dsa.hint` when possible.
