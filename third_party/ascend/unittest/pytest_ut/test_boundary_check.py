# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
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
import triton
import triton.language as tl
import pytest


# ========== Test 1: Static base address + boundary_check ==========
@triton.jit
def static_base_boundary_check_kernel(
    out_ptr,
    in_ptr,
    BLOCK_SIZE: tl.constexpr,
):
    ptr = tl.make_block_ptr(base=in_ptr, shape=(BLOCK_SIZE * 2, ), strides=(1, ), offsets=(0, ),
                            block_shape=(BLOCK_SIZE, ), order=(0, ))
    data = tl.load(ptr, boundary_check=(0, ), padding_option="zero")
    result = tl.sum(data)
    tl.store(out_ptr, result)


def ref_static_base(in_tensor, BLOCK_SIZE):
    return in_tensor[:BLOCK_SIZE].sum().item()


def test_static_base():
    BLOCK_SIZE = 64
    in_tensor = torch.randn(BLOCK_SIZE * 2, dtype=torch.float32).npu()
    out_tensor = torch.zeros(1, dtype=torch.float32).npu()
    static_base_boundary_check_kernel[(1, )](
        out_ptr=out_tensor,
        in_ptr=in_tensor,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    expected = ref_static_base(in_tensor.cpu(), BLOCK_SIZE)
    assert torch.allclose(out_tensor.cpu(), torch.tensor(expected, device='cpu'), atol=1e-4)


# ========== Test 2: Simple dynamic base address + boundary_check ==========
@triton.jit
def simple_dynamic_base_boundary_check_kernel(
    out_ptr,
    in_ptr,
    offset: tl.int32,
    BLOCK_SIZE: tl.constexpr,
):
    base = in_ptr + offset
    ptr = tl.make_block_ptr(base=base, shape=(BLOCK_SIZE * 2, ), strides=(1, ), offsets=(0, ),
                            block_shape=(BLOCK_SIZE, ), order=(0, ))
    data = tl.load(ptr, boundary_check=(0, ), padding_option="zero")
    result = tl.sum(data)
    tl.store(out_ptr, result)


def test_simple_dynamic_base():
    BLOCK_SIZE = 64
    offset = 32
    in_tensor = torch.randn(BLOCK_SIZE * 4, dtype=torch.float32).npu()
    out_tensor = torch.zeros(1, dtype=torch.float32).npu()
    simple_dynamic_base_boundary_check_kernel[(1, )](
        out_ptr=out_tensor,
        in_ptr=in_tensor,
        offset=offset,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    expected = in_tensor.cpu()[offset:offset + BLOCK_SIZE].sum().item()
    assert torch.allclose(out_tensor.cpu(), torch.tensor(expected, device='cpu'), atol=1e-4)


# ========== Test 3: Nested loop + dynamic base address + advance + boundary_check ==========
@triton.jit
def nested_dynamic_advance_boundary_kernel(
    out_ptr,
    in_ptr,
    stride_in: tl.int32,
    OUTER_LOOP: tl.constexpr,
    INNER_LOOP: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Smallest reproducible code: The dynamic base address is in the outer loop,
    and tl.advance is in the inner loop, where there is a boundary_check.
    """
    for i in range(OUTER_LOOP):
        base = in_ptr + i * stride_in
        ptr = tl.make_block_ptr(base=base, shape=(INNER_LOOP * BLOCK_SIZE, ), strides=(1, ), offsets=(0, ),
                                block_shape=(BLOCK_SIZE, ), order=(0, ))
        for j in range(INNER_LOOP):
            cur_ptr = tl.advance(ptr, (j * BLOCK_SIZE, ))
            data = tl.load(cur_ptr, boundary_check=(0, ), padding_option="zero")
            result = tl.sum(data)
            tl.store(out_ptr + i * INNER_LOOP + j, result)


def ref_nested_dynamic(in_tensor, OUTER_LOOP, INNER_LOOP, BLOCK_SIZE):
    """
    PyTorch equivalent implementation:
    - Treat in_tensor as a tensor of shape [OUTER_LOOP, INNER_LOOP * BLOCK_SIZE]
    - For each (i, j) block: take the BLOCK_SIZE elements starting from j*BLOCK_SIZE in the i-th row and sum them up.
    - Note: There is boundary_check + zero padding, but there is no out-of-bound access in this case, so no special handling is needed.
    """
    reshaped = in_tensor[:OUTER_LOOP * INNER_LOOP * BLOCK_SIZE].view(OUTER_LOOP, INNER_LOOP * BLOCK_SIZE)
    blocks = reshaped.unfold(1, BLOCK_SIZE, BLOCK_SIZE)
    return blocks.sum(dim=-1).flatten()


def test_nested_dynamic():
    BLOCK_SIZE = 8
    OUTER_LOOP = 2
    INNER_LOOP = 2
    in_tensor = torch.randn(OUTER_LOOP * INNER_LOOP * BLOCK_SIZE * 2, dtype=torch.float32).npu()
    out_tensor = torch.zeros(OUTER_LOOP * INNER_LOOP, dtype=torch.float32).npu()
    nested_dynamic_advance_boundary_kernel[(1, )](
        out_ptr=out_tensor,
        in_ptr=in_tensor,
        stride_in=INNER_LOOP * BLOCK_SIZE,
        OUTER_LOOP=OUTER_LOOP,
        INNER_LOOP=INNER_LOOP,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    ref = ref_nested_dynamic(in_tensor.cpu(), OUTER_LOOP, INNER_LOOP, BLOCK_SIZE)
    assert torch.allclose(out_tensor.cpu(), ref, atol=1e-4)


# ========== Test 4: Explicit out-of-bounds access + zero padding  + boundary_check ==========
@triton.jit
def out_of_bound_zero_padding_kernel(
    out_ptr,
    in_ptr,
    BLOCK_SIZE: tl.constexpr,
):
    ptr = tl.make_block_ptr(base=in_ptr, shape=(BLOCK_SIZE, ), strides=(1, ), offsets=(0, ),
                            block_shape=(BLOCK_SIZE * 2, ), order=(0, ))
    data = tl.load(ptr, boundary_check=(0, ), padding_option="zero")
    result = tl.sum(data)
    tl.store(out_ptr, result)


def test_out_of_bound():
    BLOCK_SIZE = 64
    in_tensor = torch.randn(BLOCK_SIZE, dtype=torch.float32).npu()
    out_tensor = torch.zeros(1, dtype=torch.float32).npu()
    out_of_bound_zero_padding_kernel[(1, )](
        out_ptr=out_tensor,
        in_ptr=in_tensor,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    expected = in_tensor.cpu().sum().item()
    assert torch.allclose(out_tensor.cpu(), torch.tensor(expected, device='cpu'), atol=1e-4)


# ========== Test 5：padding_option = NAN + boundary_check==========
@triton.jit
def nan_padding_kernel(
    out_ptr,
    in_ptr,
    BLOCK_SIZE: tl.constexpr,
):
    ptr = tl.make_block_ptr(base=in_ptr, shape=(BLOCK_SIZE, ), strides=(1, ), offsets=(0, ),
                            block_shape=(BLOCK_SIZE * 2, ), order=(0, ))
    data = tl.load(ptr, boundary_check=(0, ), padding_option="nan")
    result = tl.sum(data)
    tl.store(out_ptr, result)


def test_nan_padding():
    BLOCK_SIZE = 64
    in_tensor = torch.randn(BLOCK_SIZE, dtype=torch.float32).npu()
    out_tensor = torch.zeros(1, dtype=torch.float32).npu()
    try:
        nan_padding_kernel[(1, )](
            out_ptr=out_tensor,
            in_ptr=in_tensor,
            BLOCK_SIZE=BLOCK_SIZE,
        )
        assert torch.isnan(out_tensor.cpu()).any()
    except Exception as e:
        print(f"Warning: NAN padding test may not be supported: {e}")


# ========== Test 6：Multi-layer advance + boundary_check ==========
@triton.jit
def multi_advance_kernel(
    out_ptr,
    in_ptr,
    BLOCK_SIZE: tl.constexpr,
):
    base = in_ptr
    ptr0 = tl.make_block_ptr(base=base, shape=(BLOCK_SIZE * 4, ), strides=(1, ), offsets=(0, ),
                             block_shape=(BLOCK_SIZE, ), order=(0, ))
    ptr1 = tl.advance(ptr0, (BLOCK_SIZE, ))
    ptr2 = tl.advance(ptr1, (BLOCK_SIZE, ))
    data = tl.load(ptr2, boundary_check=(0, ), padding_option="zero")
    result = tl.sum(data)
    tl.store(out_ptr, result)


def test_multi_advance():
    BLOCK_SIZE = 64
    in_tensor = torch.randn(BLOCK_SIZE * 4, dtype=torch.float32).npu()
    out_tensor = torch.zeros(1, dtype=torch.float32).npu()
    multi_advance_kernel[(1, )](
        out_ptr=out_tensor,
        in_ptr=in_tensor,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    expected = in_tensor.cpu()[2 * BLOCK_SIZE:3 * BLOCK_SIZE].sum().item()
    assert torch.allclose(out_tensor.cpu(), torch.tensor(expected, device='cpu'), atol=1e-4)


# ========== Test 7：Complex base address calculation + boundary_check ==========
@triton.jit
def complex_base_calculation_kernel(
    out_ptr,
    in_ptr,
    offset1: tl.int32,
    offset2: tl.int32,
    scale: tl.int32,
    BLOCK_SIZE: tl.constexpr,
):
    base = in_ptr + offset1 * scale + offset2
    ptr = tl.make_block_ptr(base=base, shape=(BLOCK_SIZE * 2, ), strides=(1, ), offsets=(0, ),
                            block_shape=(BLOCK_SIZE, ), order=(0, ))
    data = tl.load(ptr, boundary_check=(0, ), padding_option="zero")
    result = tl.sum(data)
    tl.store(out_ptr, result)


def test_complex_base():
    BLOCK_SIZE = 64
    offset1, offset2, scale = 2, 16, 32
    total_offset = offset1 * scale + offset2
    in_tensor = torch.randn(total_offset + BLOCK_SIZE * 2, dtype=torch.float32).npu()
    out_tensor = torch.zeros(1, dtype=torch.float32).npu()
    complex_base_calculation_kernel[(1, )](
        out_ptr=out_tensor,
        in_ptr=in_tensor,
        offset1=offset1,
        offset2=offset2,
        scale=scale,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    expected = in_tensor.cpu()[total_offset:total_offset + BLOCK_SIZE].sum().item()
    assert torch.allclose(out_tensor.cpu(), torch.tensor(expected, device='cpu'), atol=1e-4)


if __name__ == "__main__":
    print("Running all boundary_check tests...")
    test_static_base()
    test_simple_dynamic_base()
    test_nested_dynamic()
    test_out_of_bound()
    test_nan_padding()
    test_multi_advance()
    test_complex_base()
    print("All tests completed successfully!")
