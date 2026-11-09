# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

import torch
import torch_npu
import triton
import triton.language as tl


def _assert_close(actual, expected, *, atol=1e-3, rtol=1e-3):
    torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)


@triton.jit
def _cf_maybe_loop(cond, x):
    if cond == 0:
        return x
    acc = x
    for _ in range(10):
        acc += 1
    return acc


@triton.jit
def _cf_with_scf_for_kernel(out_ptr, arg0, arg1):
    pid = tl.program_id(0)
    val = _cf_maybe_loop(arg0, arg1)
    tl.store(out_ptr + pid, val * 10)


def test_control_flow_cf_to_scf_with_scf_for():
    for arg0, arg1 in [(0, 3), (1, 3), (7, 0)]:
        out = torch.zeros((1, ), dtype=torch.int32, device="npu")
        _cf_with_scf_for_kernel[(1, )](out, arg0, arg1)
        val = arg1 if arg0 == 0 else arg1 + 10
        assert int(out[0].item()) == val * 10


@triton.jit
def _cf_helper_func(a, b, c):
    if a == 0:
        return b * 9
    if a == 1:
        return c * 9
    return 0


@triton.jit
def _cf_with_call_kernel(out_ptr, arg0, arg1, arg2, arg3):
    pid = tl.program_id(0)
    if arg0 != 0:
        tl.store(out_ptr + pid, 0)
        return
    result = _cf_helper_func(arg1, arg2, arg3) * 2
    tl.store(out_ptr + pid, result)


def test_control_flow_cf_to_scf_with_call():
    cases = [
        (1, 0, 5, 7, 0),
        (0, 0, 5, 7, 90),
        (0, 1, 5, 7, 126),
        (0, 2, 5, 7, 0),
    ]
    for arg0, arg1, arg2, arg3, expected in cases:
        out = torch.zeros((1, ), dtype=torch.int32, device="npu")
        _cf_with_call_kernel[(1, )](out, arg0, arg1, arg2, arg3)
        assert int(out[0].item()) == expected


@triton.jit
def _if_else_same_ptr_kernel(in_a_ptr, in_b_ptr, out_ptr, flag, n_elements, block: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * block + tl.arange(0, block)
    mask = offs < n_elements
    a = tl.load(in_a_ptr + offs, mask=mask)
    b = tl.load(in_b_ptr + offs, mask=mask)
    if flag != 0:
        tl.store(out_ptr + offs, a, mask=mask)
    else:
        tl.store(out_ptr + offs, b, mask=mask)


def test_control_flow_driver_if_else_same_tensor_ptr():
    n_elements, block = 128, 32
    a = torch.randn((n_elements, ), dtype=torch.float32, device="npu")
    b = torch.randn((n_elements, ), dtype=torch.float32, device="npu")
    for flag in [1, 0]:
        out = torch.zeros((n_elements, ), dtype=torch.float32, device="npu")
        _if_else_same_ptr_kernel[(triton.cdiv(n_elements, block), )](a, b, out, flag, n_elements, block)
        _assert_close(out, a if flag != 0 else b)


@triton.jit
def _if_else_diff_advance_kernel(in_ptr, out_ptr, flag, n_elements, block: tl.constexpr):
    in_bp = tl.make_block_ptr(
        base=in_ptr,
        shape=(1, n_elements),
        strides=(n_elements, 1),
        offsets=(0, 0),
        block_shape=(1, block),
        order=(1, 0),
    )
    if flag != 0:
        in_bp = tl.advance(in_bp, (0, 0))
    else:
        in_bp = tl.advance(in_bp, (0, block))
    val = tl.load(in_bp)
    out_bp = tl.make_block_ptr(
        base=out_ptr,
        shape=(1, block),
        strides=(block, 1),
        offsets=(0, 0),
        block_shape=(1, block),
        order=(1, 0),
    )
    tl.store(out_bp, val)


def test_control_flow_driver_if_else_diff_block_ptr_advance():
    n_elements, block = 64, 32
    x = torch.randn((1, n_elements), dtype=torch.float32, device="npu")
    for flag in [1, 0]:
        out = torch.empty((1, block), dtype=torch.float32, device="npu")
        _if_else_diff_advance_kernel[(1, )](x, out, flag, n_elements, block)
        expected = x[:, 0:block] if flag != 0 else x[:, block:2 * block]
        _assert_close(out, expected)


@triton.jit
def _while_block_ptr_advance_kernel(in_ptr, out_ptr, n_iters, block: tl.constexpr):
    bp = tl.make_block_ptr(
        base=in_ptr,
        shape=(n_iters * block, ),
        strides=(1, ),
        offsets=(0, ),
        block_shape=(block, ),
        order=(0, ),
    )
    acc = tl.zeros([block], tl.float32)
    i = 0
    while i < n_iters:
        acc += tl.load(bp)
        bp = tl.advance(bp, (block, ))
        i = i + 1
    tl.store(out_ptr + tl.arange(0, block), acc)


def test_control_flow_driver_while_block_ptr_advance_const():
    n_iters, block = 4, 16
    x = torch.randn((n_iters * block, ), dtype=torch.float32, device="npu")
    out = torch.empty((block, ), dtype=torch.float32, device="npu")
    _while_block_ptr_advance_kernel[(1, )](x, out, n_iters, block)
    _assert_close(out, x.reshape(n_iters, block).sum(dim=0))


@triton.jit
def _for_dynamic_advance_kernel(in_ptr, out_ptr, n_elements, block: tl.constexpr, n_loops: tl.constexpr):
    bp = tl.make_block_ptr(
        base=in_ptr,
        shape=(n_elements, ),
        strides=(1, ),
        offsets=(0, ),
        block_shape=(block, ),
        order=(0, ),
    )
    acc = tl.zeros([block], tl.float32)
    for i in range(0, n_loops):
        bp = tl.advance(bp, (i, ))
        acc += tl.load(bp)
    tl.store(out_ptr + tl.arange(0, block), acc)


def test_control_flow_driver_for_block_ptr_dynamic_advance():
    block, n_loops = 16, 4
    n_elements = block + (n_loops * (n_loops - 1)) // 2
    x = torch.randn((n_elements, ), dtype=torch.float32, device="npu")
    out = torch.empty((block, ), dtype=torch.float32, device="npu")
    _for_dynamic_advance_kernel[(1, )](x, out, n_elements, block, n_loops)
    pos = 0
    expected = torch.zeros((block, ), dtype=torch.float32, device="npu")
    for i in range(n_loops):
        pos += i
        expected += x[pos:pos + block]
    _assert_close(out, expected)


@triton.jit
def _for_advance_baseline_kernel(in_ptr, out_ptr, m_size, n_size, block_n: tl.constexpr, n_blocks: tl.constexpr):
    row = tl.program_id(0)
    in_bp = tl.make_block_ptr(
        base=in_ptr,
        shape=(m_size, n_size),
        strides=(n_size, 1),
        offsets=(row, 0),
        block_shape=(1, block_n),
        order=(1, 0),
    )
    acc = tl.zeros([1, block_n], tl.float32)
    for _ in range(0, n_blocks):
        acc += tl.load(in_bp)
        in_bp = tl.advance(in_bp, (0, block_n))
    tl.store(out_ptr + row, tl.sum(acc))


def test_control_flow_regression_for_block_ptr_advance_baseline():
    m_size, n_size, block_n = 4, 64, 16
    x = torch.randn((m_size, n_size), dtype=torch.float32, device="npu")
    out = torch.empty((m_size, ), dtype=torch.float32, device="npu")
    _for_advance_baseline_kernel[(m_size, )](x, out, m_size, n_size, block_n, n_size // block_n)
    _assert_close(out, x.sum(dim=1))


@triton.jit
def _for_if_advance_kernel(in_ptr, out_ptr, m_size, n_size, block_n: tl.constexpr, n_blocks: tl.constexpr):
    row = tl.program_id(0)
    in_bp = tl.make_block_ptr(
        base=in_ptr,
        shape=(m_size, n_size),
        strides=(n_size, 1),
        offsets=(row, 0),
        block_shape=(1, block_n),
        order=(1, 0),
    )
    acc = tl.zeros([1, block_n], tl.float32)
    for i in range(0, n_blocks):
        if i > 0:
            in_bp = tl.advance(in_bp, (0, block_n))
        acc += tl.load(in_bp)
    tl.store(out_ptr + row, tl.sum(acc))


def test_control_flow_regression_for_if_advance():
    m_size, n_size, block_n = 2, 32, 8
    x = torch.randn((m_size, n_size), dtype=torch.float32, device="npu")
    out = torch.empty((m_size, ), dtype=torch.float32, device="npu")
    _for_if_advance_kernel[(m_size, )](x, out, m_size, n_size, block_n, n_size // block_n)
    _assert_close(out, x.sum(dim=1))


@triton.jit
def _for_dynamic_ub_kernel(in_ptr, out_ptr, m_size, n_size, n_blocks, block_n: tl.constexpr):
    row = tl.program_id(0)
    in_bp = tl.make_block_ptr(
        base=in_ptr,
        shape=(m_size, n_size),
        strides=(n_size, 1),
        offsets=(row, 0),
        block_shape=(1, block_n),
        order=(1, 0),
    )
    acc = tl.zeros([1, block_n], tl.float32)
    for _ in range(0, n_blocks):
        acc += tl.load(in_bp)
        in_bp = tl.advance(in_bp, (0, block_n))
    tl.store(out_ptr + row, tl.sum(acc))


def test_control_flow_regression_for_block_ptr_dynamic_upper_bound():
    m_size, n_size, block_n = 2, 64, 16
    n_blocks = n_size // block_n
    x = torch.randn((m_size, n_size), dtype=torch.float32, device="npu")
    out = torch.empty((m_size, ), dtype=torch.float32, device="npu")
    _for_dynamic_ub_kernel[(m_size, )](x, out, m_size, n_size, n_blocks, block_n)
    _assert_close(out, x.sum(dim=1))


@triton.jit
def _while_tensor_ptr_addptr_kernel(in_ptr, out_ptr, n_iters, block: tl.constexpr):
    offs = tl.arange(0, block)
    ptrs = in_ptr + offs
    acc = tl.zeros([block], tl.float32)
    i = 0
    while i < n_iters:
        acc += tl.load(ptrs)
        ptrs = ptrs + block
        i = i + 1
    tl.store(out_ptr + offs, acc)


def test_control_flow_stress_while_tensor_ptr_addptr_const():
    n_iters, block = 4, 16
    x = torch.randn((n_iters * block, ), dtype=torch.float32, device="npu")
    out = torch.empty((block, ), dtype=torch.float32, device="npu")
    _while_tensor_ptr_addptr_kernel[(1, )](x, out, n_iters, block)
    _assert_close(out, x.reshape(n_iters, block).sum(dim=0))


@triton.jit
def _for_affine_advance_kernel(
    in_ptr,
    out_ptr,
    n_elements,
    block: tl.constexpr,
    n_loops: tl.constexpr,
    coeff_a: tl.constexpr,
    coeff_b: tl.constexpr,
):
    bp = tl.make_block_ptr(
        base=in_ptr,
        shape=(n_elements, ),
        strides=(1, ),
        offsets=(0, ),
        block_shape=(block, ),
        order=(0, ),
    )
    acc = tl.zeros([block], tl.float32)
    for i in range(0, n_loops):
        bp = tl.advance(bp, (coeff_a * i + coeff_b, ))
        acc += tl.load(bp)
    tl.store(out_ptr + tl.arange(0, block), acc)


def test_control_flow_stress_for_block_ptr_affine_advance():
    block, n_loops, coeff_a, coeff_b = 16, 4, 2, 3
    max_pos = sum(coeff_a * i + coeff_b for i in range(n_loops))
    n_elements = block + max_pos
    x = torch.randn((n_elements, ), dtype=torch.float32, device="npu")
    out = torch.empty((block, ), dtype=torch.float32, device="npu")
    _for_affine_advance_kernel[(1, )](x, out, n_elements, block, n_loops, coeff_a, coeff_b)
    pos = 0
    expected = torch.zeros((block, ), dtype=torch.float32, device="npu")
    for i in range(n_loops):
        pos += coeff_a * i + coeff_b
        expected += x[pos:pos + block]
    _assert_close(out, expected)


@triton.jit
def _nested_for_while_kernel(in_ptr, out_ptr, m_size, n_size, block: tl.constexpr, n_inner: tl.constexpr):
    offs = tl.arange(0, block)
    acc = tl.zeros([block], tl.float32)
    for row in range(0, m_size):
        bp = tl.make_block_ptr(
            base=in_ptr + row * n_size,
            shape=(n_size, ),
            strides=(1, ),
            offsets=(0, ),
            block_shape=(block, ),
            order=(0, ),
        )
        row_acc = tl.zeros([block], tl.float32)
        j = 0
        while j < n_inner:
            row_acc += tl.load(bp)
            bp = tl.advance(bp, (block, ))
            j = j + 1
        acc += row_acc
    tl.store(out_ptr + offs, acc)


def test_control_flow_stress_nested_for_while_block_ptr():
    m_size, block, n_inner = 3, 16, 4
    n_size = block * n_inner
    x = torch.randn((m_size, n_size), dtype=torch.float32, device="npu")
    out = torch.empty((block, ), dtype=torch.float32, device="npu")
    _nested_for_while_kernel[(1, )](x, out, m_size, n_size, block, n_inner)
    _assert_close(out, x.reshape(m_size, n_inner, block).sum(dim=(0, 1)))


@triton.jit
def _while_no_factual_iv_kernel(in_ptr, step_ptr, out_ptr, n_iters, block: tl.constexpr):
    offs = tl.arange(0, block)
    acc = tl.zeros([block], tl.float32)
    ptrs = in_ptr + offs
    i = 0
    k = 0
    while k < n_iters:
        acc += tl.load(ptrs)
        ptrs = ptrs + block
        step = tl.load(step_ptr + k)
        i = i + step
        k = k + 1
    tl.store(out_ptr + offs, acc)


def test_control_flow_stress_while_without_factual_iv():
    n_iters, block = 4, 16
    x = torch.arange(0, n_iters * block, dtype=torch.float32, device="npu")
    steps = torch.tensor([1, 2, 1, 3], dtype=torch.int32, device="npu")
    out = torch.empty((block, ), dtype=torch.float32, device="npu")
    _while_no_factual_iv_kernel[(1, )](x, steps, out, n_iters, block)
    _assert_close(out, x.reshape(n_iters, block).sum(dim=0))


@triton.jit
def _while_if_else_block_ptr_advance_kernel(in_ptr, out_ptr, n_iters, block: tl.constexpr):
    bp = tl.make_block_ptr(
        base=in_ptr,
        shape=(n_iters * block, ),
        strides=(1, ),
        offsets=(0, ),
        block_shape=(block, ),
        order=(0, ),
    )
    acc = tl.zeros([block], tl.float32)
    i = 0
    while i < n_iters:
        if i == 1:
            bp = tl.advance(bp, (3, ))
        else:
            bp = tl.advance(bp, (1, ))
        acc += tl.load(bp)
        bp = tl.advance(bp, (2, ))
        i = i + 1
    tl.store(out_ptr + tl.arange(0, block), acc)


def test_control_flow_stress_while_if_else_block_ptr_advance():
    n_iters, block = 4, 16
    x = torch.randn((n_iters * block, ), dtype=torch.float32, device="npu")
    out = torch.empty((block, ), dtype=torch.float32, device="npu")
    _while_if_else_block_ptr_advance_kernel[(1, )](x, out, n_iters, block)

    pos = 0
    expected = torch.zeros((block, ), dtype=torch.float32, device="npu")
    for i in range(n_iters):
        pos += 3 if i == 1 else 1
        expected += x[pos:pos + block]
        pos += 2
    _assert_close(out, expected)


@triton.jit
def _for_if_else_block_ptr_load_after_post_advance_kernel(
    in_ptr,
    out_ptr,
    flag,
    n_elements,
    block: tl.constexpr,
    n_loops: tl.constexpr,
):
    bp = tl.make_block_ptr(
        base=in_ptr,
        shape=(n_elements, ),
        strides=(1, ),
        offsets=(0, ),
        block_shape=(block, ),
        order=(0, ),
    )
    acc = tl.zeros([block], tl.float32)
    for _ in range(0, n_loops):
        if flag != 0:
            bp = tl.advance(bp, (1, ))
        else:
            bp = tl.advance(bp, (3, ))
        bp = tl.advance(bp, (2, ))
        acc += tl.load(bp)
    tl.store(out_ptr + tl.arange(0, block), acc)


def test_control_flow_stress_for_if_else_block_ptr_load_after_post_advance():
    block, n_loops = 16, 4
    n_elements = block + n_loops * 5
    x = torch.randn((n_elements, ), dtype=torch.float32, device="npu")

    for flag in [1, 0]:
        out = torch.empty((block, ), dtype=torch.float32, device="npu")
        _for_if_else_block_ptr_load_after_post_advance_kernel[(1, )](x, out, flag, n_elements, block, n_loops)

        pos = 0
        expected = torch.zeros((block, ), dtype=torch.float32, device="npu")
        for _ in range(n_loops):
            pos += 1 if flag != 0 else 3
            pos += 2
            expected += x[pos:pos + block]
        _assert_close(out, expected)


@triton.jit
def _for_if_else_tensor_ptr_load_after_post_addptr_kernel(
    in_ptr,
    out_ptr,
    flag,
    block: tl.constexpr,
    n_loops: tl.constexpr,
):
    offs = tl.arange(0, block)
    step_then = tl.zeros([block], tl.int32) + 1
    step_else = tl.zeros([block], tl.int32) + 3
    step_after = tl.zeros([block], tl.int32) + 2
    ptrs = in_ptr + offs
    acc = tl.zeros([block], tl.float32)
    for _ in range(0, n_loops):
        if flag != 0:
            ptrs = ptrs + step_then
        else:
            ptrs = ptrs + step_else
        ptrs = ptrs + step_after
        acc += tl.load(ptrs)
    tl.store(out_ptr + offs, acc)


def test_control_flow_stress_for_if_else_tensor_ptr_load_after_post_addptr():
    block, n_loops = 16, 4
    n_elements = block + n_loops * 5
    x = torch.randn((n_elements, ), dtype=torch.float32, device="npu")

    for flag in [1, 0]:
        out = torch.empty((block, ), dtype=torch.float32, device="npu")
        _for_if_else_tensor_ptr_load_after_post_addptr_kernel[(1, )](x, out, flag, block, n_loops)

        pos = 0
        expected = torch.zeros((block, ), dtype=torch.float32, device="npu")
        for _ in range(n_loops):
            pos += 1 if flag != 0 else 3
            pos += 2
            expected += x[pos:pos + block]
        _assert_close(out, expected)


@triton.jit
def _for_nested_if_block_ptr_load_after_repeated_advance_kernel(
    in_ptr,
    out_ptr,
    flag0,
    flag1,
    n_elements,
    block: tl.constexpr,
    n_loops: tl.constexpr,
):
    bp = tl.make_block_ptr(
        base=in_ptr,
        shape=(n_elements, ),
        strides=(1, ),
        offsets=(0, ),
        block_shape=(block, ),
        order=(0, ),
    )
    acc = tl.zeros([block], tl.float32)
    for _ in range(0, n_loops):
        if flag0 != 0:
            bp = tl.advance(bp, (1, ))
            if flag1 != 0:
                bp = tl.advance(bp, (2, ))
            else:
                bp = tl.advance(bp, (4, ))
        else:
            bp = tl.advance(bp, (3, ))
            if flag1 != 0:
                bp = tl.advance(bp, (1, ))
            else:
                bp = tl.advance(bp, (2, ))
        bp = tl.advance(bp, (1, ))
        acc += tl.load(bp)
    tl.store(out_ptr + tl.arange(0, block), acc)


def test_control_flow_stress_for_nested_if_block_ptr_load_after_repeated_advance():
    block, n_loops = 16, 4
    n_elements = block + n_loops * 6
    x = torch.randn((n_elements, ), dtype=torch.float32, device="npu")

    for flag0, flag1 in [(1, 1), (1, 0), (0, 1), (0, 0)]:
        out = torch.empty((block, ), dtype=torch.float32, device="npu")
        _for_nested_if_block_ptr_load_after_repeated_advance_kernel[(1, )](x, out, flag0, flag1, n_elements, block,
                                                                           n_loops)

        pos = 0
        expected = torch.zeros((block, ), dtype=torch.float32, device="npu")
        for _ in range(n_loops):
            if flag0 != 0:
                pos += 1
                pos += 2 if flag1 != 0 else 4
            else:
                pos += 3
                pos += 1 if flag1 != 0 else 2
            pos += 1
            expected += x[pos:pos + block]
        _assert_close(out, expected)


@triton.jit
def _for_while_if_block_ptr_load_after_nested_advance_kernel(
    in_ptr,
    out_ptr,
    flag,
    n_elements,
    inner_iters,
    block: tl.constexpr,
    outer_loops: tl.constexpr,
):
    bp = tl.make_block_ptr(
        base=in_ptr,
        shape=(n_elements, ),
        strides=(1, ),
        offsets=(0, ),
        block_shape=(block, ),
        order=(0, ),
    )
    acc = tl.zeros([block], tl.float32)
    for _ in range(0, outer_loops):
        j = 0
        while j < inner_iters:
            if flag != 0:
                bp = tl.advance(bp, (1, ))
            else:
                bp = tl.advance(bp, (2, ))
            j = j + 1
        bp = tl.advance(bp, (3, ))
        acc += tl.load(bp)
    tl.store(out_ptr + tl.arange(0, block), acc)


def test_control_flow_stress_for_while_if_block_ptr_load_after_nested_advance():
    block, outer_loops, inner_iters = 16, 3, 2
    n_elements = block + outer_loops * (inner_iters * 2 + 3)
    x = torch.randn((n_elements, ), dtype=torch.float32, device="npu")

    for flag in [1, 0]:
        out = torch.empty((block, ), dtype=torch.float32, device="npu")
        _for_while_if_block_ptr_load_after_nested_advance_kernel[(1, )](x, out, flag, n_elements, inner_iters, block,
                                                                        outer_loops)

        pos = 0
        expected = torch.zeros((block, ), dtype=torch.float32, device="npu")
        for _ in range(outer_loops):
            for _ in range(inner_iters):
                pos += 1 if flag != 0 else 2
            pos += 3
            expected += x[pos:pos + block]
        _assert_close(out, expected)


@triton.jit
def _for_2d_multi_advance_kernel(
    in_ptr,
    out_ptr,
    m_size,
    n_size,
    block_m: tl.constexpr,
    block_n: tl.constexpr,
    n_loops: tl.constexpr,
):
    bp = tl.make_block_ptr(
        base=in_ptr,
        shape=(m_size, n_size),
        strides=(n_size, 1),
        offsets=(0, 0),
        block_shape=(block_m, block_n),
        order=(1, 0),
    )
    acc = tl.zeros([block_m, block_n], tl.float32)
    for i in range(0, n_loops):
        bp = tl.advance(bp, (1, i))
        acc += tl.load(bp)
    out_bp = tl.make_block_ptr(
        base=out_ptr,
        shape=(block_m, block_n),
        strides=(block_n, 1),
        offsets=(0, 0),
        block_shape=(block_m, block_n),
        order=(1, 0),
    )
    tl.store(out_bp, acc)


def test_control_flow_for_block_ptr_2d_dynamic_multi_advance():
    block_m, block_n, n_loops = 2, 8, 4
    m_size = block_m + n_loops
    n_size = block_n + (n_loops * (n_loops - 1)) // 2
    x = torch.randn((m_size, n_size), dtype=torch.float32, device="npu")
    out = torch.empty((block_m, block_n), dtype=torch.float32, device="npu")
    _for_2d_multi_advance_kernel[(1, )](x, out, m_size, n_size, block_m, block_n, n_loops)

    row = 0
    col = 0
    expected = torch.zeros((block_m, block_n), dtype=torch.float32, device="npu")
    for i in range(n_loops):
        row += 1
        col += i
        expected += x[row:row + block_m, col:col + block_n]
    _assert_close(out, expected)


@triton.jit
def _while_chained_block_ptr_advance_kernel(
    in_ptr,
    out_ptr,
    n_iters,
    block: tl.constexpr,
):
    bp = tl.make_block_ptr(
        base=in_ptr,
        shape=(n_iters * block + 1, ),
        strides=(1, ),
        offsets=(0, ),
        block_shape=(block, ),
        order=(0, ),
    )
    acc = tl.zeros([block], tl.float32)
    i = 0
    while i < n_iters:
        bp = tl.advance(bp, (1, ))
        acc += tl.load(bp)
        bp = tl.advance(bp, (block - 1, ))
        i = i + 1
    tl.store(out_ptr + tl.arange(0, block), acc)


def test_control_flow_while_block_ptr_chained_advance():
    n_iters, block = 4, 16
    x = torch.randn((n_iters * block + 1, ), dtype=torch.float32, device="npu")
    out = torch.empty((block, ), dtype=torch.float32, device="npu")
    _while_chained_block_ptr_advance_kernel[(1, )](x, out, n_iters, block)

    pos = 0
    expected = torch.zeros((block, ), dtype=torch.float32, device="npu")
    for _ in range(n_iters):
        pos += 1
        expected += x[pos:pos + block]
        pos += block - 1
    _assert_close(out, expected)


@triton.jit
def _while_tensor_ptr_dynamic_step_kernel(
    in_ptr,
    step_ptr,
    out_ptr,
    n_iters,
    block: tl.constexpr,
):
    offs = tl.arange(0, block)
    ptrs = in_ptr + offs
    acc = tl.zeros([block], tl.float32)
    i = 0
    while i < n_iters:
        step = tl.load(step_ptr + i)
        step_vec = step + tl.zeros([block], tl.int32)
        ptrs = ptrs + step_vec
        acc += tl.load(ptrs)
        i = i + 1
    tl.store(out_ptr + offs, acc)


def test_control_flow_while_tensor_ptr_dynamic_step():
    block = 16
    steps_host = [1, 3, 2, 4]
    n_iters = len(steps_host)
    n_elements = block + sum(steps_host)
    x = torch.randn((n_elements, ), dtype=torch.float32, device="npu")
    steps = torch.tensor(steps_host, dtype=torch.int32, device="npu")
    out = torch.empty((block, ), dtype=torch.float32, device="npu")
    _while_tensor_ptr_dynamic_step_kernel[(1, )](x, steps, out, n_iters, block)

    pos = 0
    expected = torch.zeros((block, ), dtype=torch.float32, device="npu")
    for step in steps_host:
        pos += step
        expected += x[pos:pos + block]
    _assert_close(out, expected)


@triton.jit
def _for_nested_if_tensor_ptr_load_after_repeated_addptr_kernel(
    in_ptr,
    out_ptr,
    flag0,
    flag1,
    block: tl.constexpr,
    n_loops: tl.constexpr,
):
    offs = tl.arange(0, block)
    step1 = tl.zeros([block], tl.int32) + 1
    step2 = tl.zeros([block], tl.int32) + 2
    step3 = tl.zeros([block], tl.int32) + 3
    step4 = tl.zeros([block], tl.int32) + 4
    ptrs = in_ptr + offs
    acc = tl.zeros([block], tl.float32)
    for _ in range(0, n_loops):
        if flag0 != 0:
            ptrs = ptrs + step1
            if flag1 != 0:
                ptrs = ptrs + step2
            else:
                ptrs = ptrs + step4
        else:
            ptrs = ptrs + step3
            if flag1 != 0:
                ptrs = ptrs + step1
            else:
                ptrs = ptrs + step2
        ptrs = ptrs + step1
        acc += tl.load(ptrs)
    tl.store(out_ptr + offs, acc)


def test_control_flow_stress_for_nested_if_tensor_ptr_load_after_repeated_addptr():
    block, n_loops = 16, 4
    n_elements = block + n_loops * 6
    x = torch.randn((n_elements, ), dtype=torch.float32, device="npu")

    for flag0, flag1 in [(1, 1), (1, 0), (0, 1), (0, 0)]:
        out = torch.empty((block, ), dtype=torch.float32, device="npu")
        _for_nested_if_tensor_ptr_load_after_repeated_addptr_kernel[(1, )](x, out, flag0, flag1, block, n_loops)

        pos = 0
        expected = torch.zeros((block, ), dtype=torch.float32, device="npu")
        for _ in range(n_loops):
            if flag0 != 0:
                pos += 1
                pos += 2 if flag1 != 0 else 4
            else:
                pos += 3
                pos += 1 if flag1 != 0 else 2
            pos += 1
            expected += x[pos:pos + block]
        _assert_close(out, expected)
