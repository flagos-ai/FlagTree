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

import pytest
import torch
import torch_npu
import triton
import triton.language as tl

from test_common import generate_tensor, validate_cmp


def torch_cummax(x, dim, reverse):
    if reverse:
        x = torch.flip(x, [dim])
    values, _ = torch.cummax(x, dim=dim)
    if reverse:
        values = torch.flip(values, [dim])
    return values


# ---------------------------------------------------------------------------
# Triton combine functions
# ---------------------------------------------------------------------------
@triton.jit
def _cummax_combine(a, b):
    return tl.maximum(a, b, propagate_nan=tl.PropagateNan.ALL)


# ---------------------------------------------------------------------------
# 1D kernel
# ---------------------------------------------------------------------------
@triton.jit
def cummax_kernel_1d(
    out_ptr,
    in_ptr,
    reverse: tl.constexpr,
    XBLOCK: tl.constexpr,
):
    idx = tl.arange(0, XBLOCK)
    x = tl.load(in_ptr + idx)
    ret = tl.associative_scan(x, axis=0, combine_fn=_cummax_combine, reverse=reverse)
    tl.store(out_ptr + idx, ret)


# ---------------------------------------------------------------------------
# 2D kernel
# ---------------------------------------------------------------------------
@triton.jit
def cummax_kernel_2d(
    out_ptr,
    in_ptr,
    dim: tl.constexpr,
    reverse: tl.constexpr,
    numel_x: tl.constexpr,
    numel_r: tl.constexpr,
    XBLOCK: tl.constexpr,
    RBLOCK: tl.constexpr,
):
    idx_x = tl.arange(0, XBLOCK)
    idx_r = tl.arange(0, RBLOCK)
    idx = idx_x[:, None] * numel_r + idx_r[None, :]
    x = tl.load(in_ptr + idx)
    ret = tl.associative_scan(x, axis=dim, combine_fn=_cummax_combine, reverse=reverse)
    tl.store(out_ptr + idx, ret)


# ---------------------------------------------------------------------------
# 3D kernel
# ---------------------------------------------------------------------------
@triton.jit
def cummax_kernel_3d(
    out_ptr,
    in_ptr,
    dim: tl.constexpr,
    reverse: tl.constexpr,
    numel_x: tl.constexpr,
    numel_r: tl.constexpr,
    numel_z: tl.constexpr,
    XBLOCK: tl.constexpr,
    RBLOCK: tl.constexpr,
    ZBLOCK: tl.constexpr,
):
    idx_x = tl.arange(0, XBLOCK)
    idx_r = tl.arange(0, RBLOCK)
    idx_z = tl.arange(0, ZBLOCK)
    idx = (idx_x[:, None, None] * numel_r * numel_z + idx_r[None, :, None] * numel_z + idx_z[None, None, :])
    x = tl.load(in_ptr + idx)
    ret = tl.associative_scan(x, axis=dim, combine_fn=_cummax_combine, reverse=reverse)
    tl.store(out_ptr + idx, ret)


# ---------------------------------------------------------------------------
# Dispatch helper
# ---------------------------------------------------------------------------
def triton_cummax(x, dim, reverse):
    res = torch.empty_like(x)
    shape = x.shape
    ndim = len(shape)
    if ndim == 1:
        cummax_kernel_1d[1, 1, 1](res, x, reverse, shape[0])
    elif ndim == 2:
        cummax_kernel_2d[1, 1, 1](
            res,
            x,
            dim,
            reverse,
            shape[0],
            shape[1],
            shape[0],
            shape[1],
        )
    elif ndim == 3:
        cummax_kernel_3d[1, 1, 1](
            res,
            x,
            dim,
            reverse,
            shape[0],
            shape[1],
            shape[2],
            shape[0],
            shape[1],
            shape[2],
        )
    else:
        pytest.skip(f"Unsupported tensor dimension: {ndim}")
    return res


# ---------------------------------------------------------------------------
# Test parameters
# ---------------------------------------------------------------------------
float_dtypes = ["float32", "float16"]
int_dtypes = ["int32", "int16"]

shapes_1d = [(64, ), (128, ), (7, )]
shapes_2d = [(7, 23), (16, 32), (3, 5)]
shapes_3d = [(4, 8, 16), (3, 5, 7)]


# ---------------------------------------------------------------------------
# 1D tests
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("dtype", float_dtypes + int_dtypes)
@pytest.mark.parametrize("shape", shapes_1d)
@pytest.mark.parametrize("reverse", [False, True])
def test_cummax_1d(dtype, shape, reverse):
    x = generate_tensor(shape=shape, dtype=dtype).npu()
    triton_cal = triton_cummax(x, dim=0, reverse=reverse)
    torch_ref = torch_cummax(x, dim=0, reverse=reverse)
    validate_cmp(dtype, torch_ref, triton_cal)


# ---------------------------------------------------------------------------
# 2D tests
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("dtype", float_dtypes + int_dtypes)
@pytest.mark.parametrize("shape", shapes_2d)
@pytest.mark.parametrize("dim", [0, 1])
@pytest.mark.parametrize("reverse", [False, True])
def test_cummax_2d(dtype, shape, dim, reverse):
    x = generate_tensor(shape=shape, dtype=dtype).npu()
    triton_cal = triton_cummax(x, dim=dim, reverse=reverse)
    torch_ref = torch_cummax(x, dim=dim, reverse=reverse)
    validate_cmp(dtype, torch_ref, triton_cal)


# ---------------------------------------------------------------------------
# 3D tests
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("dtype", float_dtypes + int_dtypes)
@pytest.mark.parametrize("shape", shapes_3d)
@pytest.mark.parametrize("dim", [0, 1])
@pytest.mark.parametrize("reverse", [False, True])
def test_cummax_3d(dtype, shape, dim, reverse):
    x = generate_tensor(shape=shape, dtype=dtype).npu()
    triton_cal = triton_cummax(x, dim=dim, reverse=reverse)
    torch_ref = torch_cummax(x, dim=dim, reverse=reverse)
    validate_cmp(dtype, torch_ref, triton_cal)


# ---------------------------------------------------------------------------
# Edge-case: constant input (all elements equal)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("dtype", float_dtypes + int_dtypes)
@pytest.mark.parametrize("dim", [0, 1])
def test_cummax_constant(dtype, dim):
    torch_dtype = eval("torch." + dtype)
    x = torch.full((8, 16), 42, dtype=torch_dtype).npu()
    triton_cal = triton_cummax(x, dim=dim, reverse=False)
    torch_ref = torch_cummax(x, dim=dim, reverse=False)
    validate_cmp(dtype, torch_ref, triton_cal)


# ---------------------------------------------------------------------------
# Edge-case: monotonically increasing / decreasing
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("dtype", ["float32", "int32"])
@pytest.mark.parametrize("reverse", [False, True])
def test_cummax_monotonic_increasing(dtype, reverse):
    torch_dtype = eval("torch." + dtype)
    x = torch.arange(1, 33, dtype=torch_dtype).reshape(4, 8).npu()
    triton_cal = triton_cummax(x, dim=1, reverse=reverse)
    torch_ref = torch_cummax(x, dim=1, reverse=reverse)
    validate_cmp(dtype, torch_ref, triton_cal)


@pytest.mark.parametrize("dtype", ["float32", "int32"])
@pytest.mark.parametrize("reverse", [False, True])
def test_cummax_monotonic_decreasing(dtype, reverse):
    torch_dtype = eval("torch." + dtype)
    x = torch.arange(32, 0, -1, dtype=torch_dtype).reshape(4, 8).npu()
    triton_cal = triton_cummax(x, dim=1, reverse=reverse)
    torch_ref = torch_cummax(x, dim=1, reverse=reverse)
    validate_cmp(dtype, torch_ref, triton_cal)


# ---------------------------------------------------------------------------
# Edge-case: negative values
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("dtype", ["float32", "int32"])
def test_cummax_negative_values(dtype):
    torch_dtype = eval("torch." + dtype)
    x = torch.tensor([[-3, -1, -4, -1, -5, -9, -2, -6]], dtype=torch_dtype).npu()
    triton_cal = triton_cummax(x, dim=1, reverse=False)
    torch_ref = torch_cummax(x, dim=1, reverse=False)
    validate_cmp(dtype, torch_ref, triton_cal)


# ---------------------------------------------------------------------------
# Edge-case: single row / single column
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("dtype", ["float32", "int32"])
def test_cummax_single_row(dtype):
    x = generate_tensor(shape=(1, 32), dtype=dtype).npu()
    triton_cal = triton_cummax(x, dim=1, reverse=False)
    torch_ref = torch_cummax(x, dim=1, reverse=False)
    validate_cmp(dtype, torch_ref, triton_cal)


@pytest.mark.parametrize("dtype", ["float32", "int32"])
def test_cummax_single_col(dtype):
    x = generate_tensor(shape=(32, 1), dtype=dtype).npu()
    triton_cal = triton_cummax(x, dim=0, reverse=False)
    torch_ref = torch_cummax(x, dim=0, reverse=False)
    validate_cmp(dtype, torch_ref, triton_cal)
