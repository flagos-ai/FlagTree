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

def combine_fn_test_torch_add(a, b):
    return a + b

def torch_func_scan(x: torch.Tensor, dim: int, reverse: bool):
    """
    PyTorch implements associative_scan, with semantics fully aligned with Triton.
    """
    dim = dim % x.ndim

    if reverse:
        x = x.flip(dim)

    N = x.size(dim)
    tensors = torch.unbind(x, dim=dim)

    outputs = []
    carry = tensors[0]
    outputs.append(carry)

    for i in range(1, N):
        carry = combine_fn_test_torch_add(tensors[i], carry)
        outputs.append(carry)

    output = torch.stack(outputs, dim=dim)

    if reverse:
        output = output.flip(dim)

    return output

@triton.jit
def add_combine_fn(a_value, a_index, b_value, b_index):
    new_val = a_value + b_value
    new_idx = a_index + b_index
    return (new_val, new_idx)

@triton.jit
def prefix_scan_last_dim_kernel(
    vals_ptr, idx_ptr,
    out_vals_ptr, out_idxs_ptr,
    axis_size, total_slices,
    BLOCK_SIZE: tl.constexpr,
    reverse: tl.constexpr,
):
    slice_id = tl.program_id(0)
    if slice_id >= total_slices:
        return

    offs = tl.arange(0, BLOCK_SIZE)
    mask = offs < axis_size
    base = slice_id * axis_size

    vals = tl.load(vals_ptr + base + offs, mask=mask, other=0.0)
    idxs = tl.load(idx_ptr + base + offs, mask=mask, other=0)

    pre_vals, pre_idxs = tl.associative_scan(
        (vals, idxs), axis=0, combine_fn=add_combine_fn, reverse=reverse
    )

    tl.store(out_vals_ptr + base + offs, pre_vals, mask=mask)
    tl.store(out_idxs_ptr + base + offs, pre_idxs, mask=mask)


def multi_input_prefix_sum(values: torch.Tensor, index: torch.Tensor, axis=0, reverse=False):
    assert values.shape == index.shape
    assert values.device == index.device
    rank = values.ndim
    if axis < 0:
        axis += rank
    assert 0 <= axis < rank

    # 1. change to make axis the last dim
    order = list(range(rank))
    if axis != rank - 1:
        order[axis], order[-1] = order[-1], order[axis]
    inv_order = [0] * rank
    for i, o in enumerate(order):
        inv_order[o] = i

    vals_p = values.permute(order).contiguous()
    idxs_p = index.permute(order).contiguous()

    shape_p = vals_p.shape
    axis_size = shape_p[-1]
    total_slices = 1
    for d in shape_p[:-1]:
        total_slices *= d

    out_vals_p = torch.empty_like(vals_p)
    out_idxs_p = torch.empty_like(idxs_p)

    BLOCK_SIZE = 1 << (axis_size - 1).bit_length()

    prefix_scan_last_dim_kernel[(total_slices,)](
        vals_p, idxs_p,
        out_vals_p, out_idxs_p,
        axis_size, total_slices,
        BLOCK_SIZE=BLOCK_SIZE,
        reverse=reverse
    )

    # 2. permute back
    out_vals = out_vals_p.permute(inv_order)
    out_idxs = out_idxs_p.permute(inv_order)
    return out_vals, out_idxs


@pytest.mark.parametrize("shape, axis", [
    ((10,), 0),
    ((4, 4), 0),
    ((4, 4), 1),
    ((2, 10, 5), 0),
    ((2, 10, 5), 1),
    ((2, 10, 5), 2),
])
@pytest.mark.parametrize("reverse", [False,True])
def test_multi_input_prefix_sum(shape, axis, reverse):
    torch.manual_seed(0)
    device = "npu"

    values = torch.randn(shape, device=device, dtype=torch.float32)
    index = torch.arange(values.numel(), device=device, dtype=torch.int32).reshape(shape)

    triton_vals, triton_idxs = multi_input_prefix_sum(values, index, axis=axis, reverse=reverse)

    torch_vals = torch_func_scan(values, axis, reverse)
    torch_idxs = torch_func_scan(index, axis, reverse)

    assert torch.allclose(triton_vals, torch_vals, rtol=1e-3, atol=1e-3), \
        f"数值不匹配！shape={shape}, axis={axis}\nTriton: {triton_vals}\nPyTorch: {torch_vals}"
    assert torch.equal(triton_idxs, torch_idxs), \
        f"索引不匹配！shape={shape}, axis={axis}\nTriton: {triton_idxs}\nPyTorch: {torch_idxs}"