import pytest
import triton
import triton.language as tl
import torch
import torch_npu
import test_common


def torch_invert(x0):
    res = ~(x0)
    return res


@triton.jit
def triton_invert(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr, XBLOCK_SUB: tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    for xoffset_sub in range(0, XBLOCK, XBLOCK_SUB):
        xindex = xoffset + xoffset_sub + tl.arange(0, XBLOCK_SUB)[:]
        xmask = xindex < xnumel
        x0 = xindex
        tmp0 = tl.load(in_ptr0 + (x0), xmask)
        tmp2 = ~tmp0
        tl.store(out_ptr0 + (xindex), tmp2, xmask)


@pytest.mark.parametrize('param_list',
                         [
                             ['int8', (2, 4096, 8), 2, 32768, 1024],
                             ['int16', (2, 4096, 8), 2, 32768, 1024],
                             ['int32', (2, 4096, 8), 2, 32768, 1024],
                             ['int64', (2, 4096, 8), 2, 32768, 1024],
                             ['bool', (2, 4096, 8), 2, 32768, 1024],
                         ])
def test_invert(param_list):
    # 生成数据
    dtype, shape, ncore, xblock, xblock_sub = param_list
    x0 = test_common.generate_tensor(shape, dtype).npu()
    # torch结果
    torch_res = torch_invert(x0)
    # triton结果
    triton_res = torch.zeros(shape, dtype=eval('torch.' + dtype)).npu()
    triton_invert[ncore, 1, 1](x0, triton_res, x0.numel(), xblock, xblock_sub)
    # 比较结果
    test_common.validate_cmp(dtype, triton_res, torch_res)
