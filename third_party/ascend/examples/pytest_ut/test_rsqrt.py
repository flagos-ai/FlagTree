import triton
import triton.language as tl
import torch
import numpy as np
import pytest
import test_common

def numpy_rsqrt(x0, x1):
    res = x0 + 1.0 / (np.sqrt(x1))
    return res

@triton.jit
def triton_rsqrt(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK: tl.constexpr, XBLOCK_SUB: tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    for xoffset_sub in range(0, XBLOCK, XBLOCK_SUB):
        xindex = xoffset + xoffset_sub + tl.arange(0, XBLOCK_SUB)[:]
        xmask = xindex < xnumel
        x0 = xindex
        tmp0 = tl.load(in_ptr0 + (x0), xmask)
        tmp1 = tl.load(in_ptr1 + (x0), xmask)
        tmp2 = tmp0 + tl.rsqrt(tmp1)
        tl.store(out_ptr0 + (xindex), tmp2, xmask)

@pytest.mark.parametrize('param_list',
                         [
                             ['float32', (2, 4096, 8), 2, 32768, 1024],
                         ])

def test_rsqrt(param_list):
    # 生成数据
    dtype, shape, ncore, xblock, xblock_sub = param_list
    x0 = np.abs(np.random.randn(shape[0], shape[1], shape[2])).astype(eval('np.' + dtype))
    x1 = np.abs(np.random.randn(shape[0], shape[1], shape[2])).astype(eval('np.' + dtype))
    x0_npu = torch.tensor(x0).npu()
    x1_npu = torch.tensor(x1).npu()
    # numpy结果
    numpy_res = numpy_rsqrt(x0, x1)
    # triton结果
    triton_res = test_common.generate_tensor(shape, dtype).npu()
    triton_rsqrt[ncore, 1, 1](x0_npu, x1_npu, triton_res, x0_npu.numel(), xblock, xblock_sub)
    # 比较结果
    test_common.validate_cmp(dtype, triton_res, torch.tensor(numpy_res).npu())
