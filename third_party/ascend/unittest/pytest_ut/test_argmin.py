import logging
import math
import pytest
import torch
import torch_npu
import numpy as np
import triton
import triton.language as tl

import test_common


def torch_argmin(x0, dim, keepdim):
    x0 = x0 if x0.device == "cpu" else x0.cpu()
    return torch.argmin(x0, dim=dim, keepdim=keepdim).npu()


@triton.jit
def triton_argmin_1d(in_ptr0, out_ptr1, xnumel, XBLOCK: tl.constexpr):
    xoffset = tl.program_id(0) + tl.arange(0, XBLOCK)
    tmp0 = tl.load(in_ptr0 + xoffset, None)
    tmp4 = tl.argmin(tmp0, 0)
    tl.store(out_ptr1, tmp4, None)


@pytest.mark.parametrize('shape', [(128, ), (256, ), (37, ), (741, )])
@pytest.mark.parametrize('dtype', ['int32', 'float32', 'uint8', 'int8'])
def test_argmin_1d(dtype, shape):
    x0 = test_common.generate_tensor(shape, dtype).npu()
    triton_res = torch.empty(1, dtype=torch.int32).npu()
    numel = shape[0]
    triton_argmin_1d[(1, )](x0, triton_res, numel, numel)
    torch_res = torch_argmin(x0, dim=0, keepdim=True)
    test_common.validate_cmp("int32", triton_res, torch_res)


@triton.jit
def triton_argmin_2d(in_ptr0, out_ptr0, dim: tl.constexpr, M: tl.constexpr, N: tl.constexpr, MNUMEL: tl.constexpr,
                     NNUMEL: tl.constexpr):
    mblk_idx = tl.arange(0, MNUMEL)
    nblk_idx = tl.arange(0, NNUMEL)
    mmask = mblk_idx < M
    nmask = nblk_idx < N
    mask = (mmask[:, None]) & (nmask[None, :])
    idx = mblk_idx[:, None] * N + nblk_idx[None, :]
    x = tl.load(in_ptr0 + idx, mask=mask, other=float('inf'))
    tmp4 = tl.argmin(x, dim)
    if dim == 0:
        tl.store(out_ptr0 + tl.arange(0, N), tmp4, None)
    else:
        tl.store(out_ptr0 + tl.arange(0, M), tmp4, None)


@pytest.mark.parametrize('shape', [(37, 125), (29, 4), (7, 31)])
@pytest.mark.parametrize('dtype', ['int32', 'float32', 'uint8', 'int8'])
@pytest.mark.parametrize('dim', [0, 1])
def test_argmin_2d(dtype, shape, dim):
    shapex, shapey = shape
    x0 = test_common.generate_tensor(shape, dtype).npu()
    triton_res = torch.empty([
        shape[1 - dim],
    ], dtype=torch.int32).npu()
    triton_argmin_2d[(1, 1)](x0, triton_res, dim, shapex, shapey, shapex, shapey)
    torch_res = torch_argmin(x0, dim=dim, keepdim=False)
    test_common.validate_cmp("int32", triton_res, torch_res)
