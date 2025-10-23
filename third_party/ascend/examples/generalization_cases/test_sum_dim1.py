import triton
import triton.language as tl
import torch
import pytest
import test_common
from test_common import TestUtils
import math

def torch_sum(x0):
    res = torch.sum(x0, 1)
    return res

@triton.jit
def triton_sum(in_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, XBLOCK_SUB : tl.constexpr, RBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    for xoffset_sub in range(0, XBLOCK, XBLOCK_SUB):
        xindex = xoffset + xoffset_sub + tl.arange(0, XBLOCK_SUB)
        x0 = xindex
        r1 = rindex
        xmask = xindex[:, None] < xnumel
        xmask_prime = xindex < xnumel
        tmp0 = tl.load(in_ptr0 + (r1 + (RBLOCK*x0[:, None])), rmask & xmask)
        tmp2 = tl.reshape(tmp0, [XBLOCK_SUB, RBLOCK])
        tmp4 = tl.sum(tmp2, 1)
        tl.store(out_ptr1 + (xindex), tmp4, xmask_prime)

@pytest.mark.parametrize('shape', TestUtils.test_shape2d)
@pytest.mark.parametrize('dtype', ['float32', 'float16', 'int32'])
def test_case(dtype, shape):
    x0 = test_common.generate_tensor(shape, dtype).npu()

    rblock = shape[1]
    xblock = shape[0]
    ncore = 1 #if numel <= 32 else 32
    xblock_sub = xblock if xblock <= 16 else 16
    RBLOCK_tl = 256 if rblock > 1 else 1

    y_ref = torch_sum(x0)
    y_cal = torch.zeros(shape[:-1], dtype=eval('torch.' + dtype)).npu()
    triton_sum[ncore, 1, 1](x0, y_cal, xblock, rblock, xblock, xblock_sub, rblock)
    test_common.validate_cmp(dtype, y_cal, y_ref)