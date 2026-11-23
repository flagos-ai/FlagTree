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
from triton.runtime.libentry import libentry
from triton.backends.ascend.utils import is_compile_on_910_95

from test_common import _all_dtypes_no_bool, _uint_dtypes, validate_cmp


def torch_func(x, dim, reverse):
    is_bf16 = x.dtype == torch.bfloat16
    if is_bf16:
        x = x.to(torch.float32)
    if reverse:
        x = torch.flip(x, [dim])
    res = torch.cumprod(x, dim=dim)
    if is_bf16:
        res = res.to(torch.bfloat16)
    if reverse:
        res = torch.flip(res, [dim])
    return res


@libentry()
@triton.jit
def triton_kernel_1d(
    out_ptr0,
    in_ptr0,
    dim: tl.constexpr,
    reverse: tl.constexpr,
    numel_x: tl.constexpr,
    XBLOCK: tl.constexpr,
):
    tl.static_assert(numel_x == XBLOCK, "numel_x must be equal to XBLOCK in this kernel")
    idx = tl.arange(0, XBLOCK)
    x = tl.load(in_ptr0 + idx)
    ret = tl.cumprod(x, axis=dim, reverse=reverse)
    tl.store(out_ptr0 + idx, ret)


def triton_func_1d(x, dim, reverse):
    res = torch.empty_like(x)
    triton_kernel_1d[1, 1, 1](res, x, dim, reverse, x.shape[0], x.shape[0])
    return res


@libentry()
@triton.jit
def triton_kernel_2d(
    out_ptr0,
    in_ptr0,
    dim: tl.constexpr,
    reverse: tl.constexpr,
    numel_x: tl.constexpr,
    numel_r: tl.constexpr,
    XBLOCK: tl.constexpr,
    RBLOCK: tl.constexpr,
):
    tl.static_assert(numel_x == XBLOCK, "numel_x must be equal to XBLOCK in this kernel")
    tl.static_assert(numel_r == RBLOCK, "numel_r must be equal to RBLOCK in this kernel")
    idx_x = tl.arange(0, XBLOCK)
    idx_r = tl.arange(0, RBLOCK)
    idx = idx_x[:, None] * numel_r + idx_r[None, :]
    x = tl.load(in_ptr0 + idx)
    ret = tl.cumprod(x, axis=dim, reverse=reverse)
    tl.store(out_ptr0 + idx, ret)


def triton_func_2d(x, dim, reverse):
    res = torch.empty_like(x)
    triton_kernel_2d[1, 1, 1](res, x, dim, reverse, x.shape[0], x.shape[1], x.shape[0], x.shape[1])
    return res


# ---------------------------------------------------------------------------
# 5D kernel
# dim is the axis passed to tl.cumprod (0‥4), relative to the 5D block.
# The kernel is launched with a single program (grid=[1,1,1]) and loads the
# entire tensor in one shot, so every constexpr size must be a power-of-two
# (Triton tl.arange requirement).
# ---------------------------------------------------------------------------
@libentry()
@triton.jit
def triton_kernel_5d(
    out_ptr0,
    in_ptr0,
    dim: tl.constexpr,
    reverse: tl.constexpr,
    S0: tl.constexpr,
    S1: tl.constexpr,
    S2: tl.constexpr,
    S3: tl.constexpr,
    S4: tl.constexpr,
):
    # Build a 5D index grid, keeping each dimension independent so that
    # tl.cumprod can operate along the correct axis without any flattening.
    i0 = tl.arange(0, S0)
    i1 = tl.arange(0, S1)
    i2 = tl.arange(0, S2)
    i3 = tl.arange(0, S3)
    i4 = tl.arange(0, S4)

    # Broadcast to a full [S0, S1, S2, S3, S4] index tensor.
    # Each reshape adds the missing axes so broadcasting produces the right shape.
    idx = (i0[:, None, None, None, None] * (S1 * S2 * S3 * S4) + i1[None, :, None, None, None] * (S2 * S3 * S4) +
           i2[None, None, :, None, None] * (S3 * S4) + i3[None, None, None, :, None] * S4 +
           i4[None, None, None, None, :])

    x = tl.load(in_ptr0 + idx)

    ret = tl.cumprod(x, axis=dim, reverse=reverse)

    tl.store(out_ptr0 + idx, ret)


def triton_func_5d(x, dim, reverse):
    assert x.ndim == 5, "triton_func_5d expects a 5-D tensor"
    res = torch.empty_like(x)
    S0, S1, S2, S3, S4 = x.shape
    triton_kernel_5d[1, 1, 1](
        res,
        x,
        dim,
        reverse,
        S0,
        S1,
        S2,
        S3,
        S4,
    )
    return res


def cumprod_generate_tensor(shape, dtype):
    if dtype in ('float32', 'float16', 'bfloat16'):
        return torch.rand(size=shape, dtype=eval('torch.' + dtype))
    elif dtype in ('int32', 'int64', 'int16'):
        return torch.randint(low=0, high=10, size=shape, dtype=eval('torch.' + dtype))
    elif dtype == 'int8':
        return torch.randint(low=0, high=10, size=shape, dtype=eval('torch.' + dtype))
    elif dtype in ('uint8', 'uint32', 'uint64', 'uint16'):
        return torch.randint(low=0, high=10, size=shape, dtype=eval('torch.' + dtype))
    elif dtype == 'bool':
        return torch.randint(low=0, high=2, size=shape, dtype=eval('torch.' + dtype))
    else:
        raise ValueError(f"Unsupported dtype: {dtype}")


not_support_dtype = {'int8', 'bool'}
support_dtypes = (_all_dtypes_no_bool + _uint_dtypes) if is_compile_on_910_95(
    triton.runtime.driver.active.get_current_target().arch) else \
    [dtype for dtype in _all_dtypes_no_bool if dtype not in not_support_dtype]


@pytest.mark.parametrize("dtype", support_dtypes)
@pytest.mark.parametrize("shape", [(1024, )])
@pytest.mark.parametrize("dim", [0])
@pytest.mark.parametrize("reverse", [False, True])
def test_cumprod_1d(dtype, shape, dim, reverse):
    x0 = cumprod_generate_tensor(shape=shape, dtype=dtype).npu()
    triton_cal = triton_func_1d(x0, dim, reverse)
    torch_ref = torch_func(x0, dim, reverse).to(eval("torch." + dtype))
    validate_cmp(dtype, torch_ref, triton_cal)


@pytest.mark.parametrize("dtype", support_dtypes)
@pytest.mark.parametrize("shape", [(7, 23)])
@pytest.mark.parametrize("dim", [0, 1])
@pytest.mark.parametrize("reverse", [False, True])
def test_cumprod_2d(dtype, shape, dim, reverse):
    x0 = cumprod_generate_tensor(shape=shape, dtype=dtype).npu()
    triton_cal = triton_func_2d(x0, dim, reverse)
    torch_ref = torch_func(x0, dim, reverse).to(eval("torch." + dtype))
    validate_cmp(dtype, torch_ref, triton_cal)


@pytest.mark.parametrize("dtype", support_dtypes)
@pytest.mark.parametrize("shape", [(2, 4, 4, 4, 4)])
@pytest.mark.parametrize("dim", [0, 1, 4])
@pytest.mark.parametrize("reverse", [False, True])
def test_cumprod_5d(dtype, shape, dim, reverse):
    x0 = cumprod_generate_tensor(shape=shape, dtype=dtype).npu()
    triton_cal = triton_func_5d(x0, dim, reverse)
    torch_ref = torch_func(x0, dim, reverse).to(eval("torch." + dtype))
    validate_cmp(dtype, torch_ref, triton_cal)
