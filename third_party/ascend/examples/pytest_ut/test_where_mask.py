import triton
import triton.language as tl
import torch
import torch_npu
import pytest
import test_common

def torch_where_lt_case2(x0, x1):
    res = torch.where(x0 < x1, x0, x1)
    return res

@triton.jit
def triton_where_lt_case2(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK: tl.constexpr, XBLOCK_SUB: tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    for xoffset_sub in range(0, XBLOCK, XBLOCK_SUB):
        xindex = xoffset + xoffset_sub + tl.arange(0, XBLOCK_SUB)[:]
        xmask = xindex < xnumel
        x0 = xindex
        tmp0 = tl.load(in_ptr0 + (x0), xmask)
        tmp1 = tl.load(in_ptr1 + (x0), xmask)
        tmp2 = tmp0 < tmp1
        tmp3 = tl.where(tmp2, tmp0, tmp1)
        tl.store(out_ptr0 + (xindex), tmp3, xmask)

@pytest.mark.parametrize('param_list',
                         [
                            ['float32', (2, 1024, 8), 2, 8192, 1024],
                            ['float16', (2, 1024, 8), 2, 8192, 1024],
                            ['int8', (2, 1024, 8), 2, 8192, 1024],
                         ]
                         )
def test_where_lt_case2(param_list):
    dtype, shape, ncore, xblock, xblock_sub = param_list
    x0 = test_common.generate_tensor(shape, dtype).npu()
    x1 = test_common.generate_tensor(shape, dtype).npu()
    y_ref = torch_where_lt_case2(x0, x1)
    y_cal = test_common.generate_tensor(shape, dtype).npu()
    triton_where_lt_case2[ncore, 1, 1](x0, x1, y_cal, x0.numel(), xblock, xblock_sub)
    test_common.validate_cmp(dtype, y_cal, y_ref)
