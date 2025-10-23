import triton
import triton.language as tl
import torch
import pytest
import test_common

def torch_sigmoid(x0, x1):
    res = x0 + torch.sigmoid(x1)
    return res

@triton.jit
def triton_sigmoid(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK: tl.constexpr, XBLOCK_SUB: tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    for xoffset_sub in range(0, XBLOCK, XBLOCK_SUB):
        xindex = xoffset + xoffset_sub + tl.arange(0, XBLOCK_SUB)[:]
        xmask = xindex < xnumel
        x0 = xindex
        tmp0 = tl.load(in_ptr0 + (x0), xmask)
        tmp1 = tl.load(in_ptr1 + (x0), xmask)
        tmp2 = tmp0 + tl.sigmoid(tmp1)
        tl.store(out_ptr0 + (xindex), tmp2, xmask)


@pytest.mark.parametrize('param_list',
                         [
                             ['float32', (2, 4096, 8), 2, 32768, 1024],
                         ])
def test_sigmoid(param_list):
    # 生成数据
    dtype, shape, ncore, xblock, xblock_sub = param_list
    x0 = test_common.generate_tensor(shape, dtype).npu()
    x1 = test_common.generate_tensor(shape, dtype).npu()
    # torch结果
    y_ref = torch_sigmoid(x0, x1)
    # triton结果
    y_cal = test_common.generate_tensor(shape, dtype).npu()
    triton_sigmoid[ncore, 1, 1](x0, x1, y_cal, x0.numel(), xblock, xblock_sub)
    # 比较结果
    test_common.validate_cmp(dtype, y_cal, y_ref)