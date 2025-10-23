import triton
import triton.language as tl
import torch
import pytest
import test_common
from test_common import TestUtils, get_dtype_size
import math

def torch_sum(x0):
    res = torch.sum(x0, 0)
    return res

@triton.jit
def triton_sum(in_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr, RBLOCK_SUB : tl.constexpr):
    xindex = tl.arange(0, XBLOCK)
    xmask = xindex[:, None] < xnumel
    for roffset_sub in range(0, RBLOCK, RBLOCK_SUB):
        rindex = roffset_sub + tl.arange(0, RBLOCK_SUB)
        x0 = xindex
        r1 = rindex
        rmask = rindex < rnumel
        tmp0 = tl.load(in_ptr0 + (r1 + (RBLOCK*x0[:, None])), xmask & rmask)
        tmp2 = tl.reshape(tmp0, [XBLOCK, RBLOCK_SUB])
        tmp4 = tl.sum(tmp2, 0)
        tl.store(out_ptr1 + (rindex), tmp4, rmask)

def should_skip_due_to_mem(dtype, shape):
    dtype_size = get_dtype_size(dtype)
    total_mem = dtype_size * math.prod(shape)
    threshold = TestUtils.ub_size / 1.5

    if total_mem >= threshold:
        pytest.skip(f"dtype:{dtype} shape:{shape} mem overflow")

@pytest.mark.parametrize('shape', TestUtils.test_shape2d)
@pytest.mark.parametrize('dtype', ['float32', 'int32'])
def test_case(dtype, shape):
    should_skip_due_to_mem(dtype, shape)
    x0 = test_common.generate_tensor(shape, dtype).npu()

    rblock = shape[1]
    xblock = shape[0]
    ncore = 1 #if numel <= 32 else 32
    rblock_sub = rblock #if xblock <= 16 else 16
    RBLOCK_tl = 256 if rblock > 1 else 1

    y_ref = torch_sum(x0)
    y_cal = torch.zeros(shape[1], dtype=eval('torch.' + dtype)).npu()
    triton_sum[ncore, 1, 1](x0, y_cal, xblock, rblock, xblock, rblock, rblock_sub)
    test_common.validate_cmp(dtype, y_cal, y_ref)