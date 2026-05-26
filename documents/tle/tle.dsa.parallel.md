# tle.dsa.parallel

## 1. Function Overview

`parallel` is a JIT-only loop iterator provided by TLE DSA. It inherits from `tle.dsa.range` and expresses that loop iterations are independent and can be handled with parallel semantics.

```python
tle.dsa.parallel(arg1, arg2=None, step=None, loop_unroll_factor=None)
```

This API can only be used as a `for` loop iterator inside functions decorated with `@triton.jit`. It cannot be iterated directly in ordinary Python code.

## 2. Specification

### 2.1 Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `arg1` | `int` / `tl.constexpr` / scalar value | Required | In single-argument form, it is the end value and the start value defaults to `0`; in two-argument form, it is the start value. |
| `arg2` | `int` / `tl.constexpr` / scalar value | `None` | End value, exclusive. |
| `step` | `int` / `tl.constexpr` / scalar value | `1` | Step increment for each iteration. |
| `loop_unroll_factor` | `int` | `None` | Loop unroll factor passed to the compiler. Values smaller than `2` mean no unrolling. |

### 2.2 Call Forms

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

### 2.3 Relationship with `tle.dsa.range`

`parallel` inherits from `tle.dsa.range`, but its constructor only exposes the following parameters:

- `arg1`
- `arg2`
- `step`
- `loop_unroll_factor`

`parallel` does not support the following `tle.dsa.range` parameters:

- `disallow_acc_multi_buffer`
- `flatten`
- `warp_specialize`
- `disable_licm`

### 2.4 Type Support

`parallel` is a loop iterator and does not impose operator-level constraints on data types. The loop variable type is determined by the provided `start/end/step` values and Triton compilation semantics.

## 3. Usage

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

## 4. Limitations

- It can only be used inside `@triton.jit` functions.
- It can only be used as a `for` loop iterator; it cannot be directly passed to `iter()` or `next()` at ordinary Python runtime.
- `parallel` expresses that loop iterations have no cross-iteration dependency. Do not use it if the loop body has cross-iteration data dependencies.
