import pytest
import triton
import triton.language as tl
import torch
import torch_npu
import test_common


def torch_cdiv(x0, x1):
    return torch.div(x0, x1, rounding_mode='trunc') + (x0 % x1 > 0).to(torch.int)


@triton.jit
def triton_cdiv(in_ptr0, in_ptr1, out_ptr0, XBLOCK: tl.constexpr, XBLOCK_SUB: tl.constexpr):
    offset = tl.program_id(0) * XBLOCK
    base1 = tl.arange(0, XBLOCK_SUB)
    loops1: tl.constexpr = tl.cdiv(XBLOCK, XBLOCK_SUB)
    for loop1 in range(loops1):
        x_index = offset + (loop1 * XBLOCK_SUB) + base1
        tmp0 = tl.load(in_ptr0 + x_index, None)
        tmp1 = tl.load(in_ptr1 + x_index, None)
        tmp2 = tl.cdiv(tmp0, tmp1)
        tl.store(out_ptr0 + x_index, tmp2, None)


@pytest.mark.parametrize('param_list',
                         [
                             ['int32', (4096,), 1, 4096, 4096],
                         ])
def test_cdiv(param_list):
    # 生成数据
    dtype, shape, ncore, xblock, xblock_sub = param_list
    x0 = test_common.generate_tensor(shape, dtype).npu()
    x1 = test_common.generate_tensor(shape, dtype).npu() + 1
    # torch结果
    torch_res = torch_cdiv(x0, x1)
    # triton结果
    triton_res = torch.zeros(shape, dtype=eval('torch.' + dtype)).npu()
    triton_cdiv[ncore, 1, 1](x0, x1, triton_res, xblock, xblock_sub)
    # 比较结果
    test_common.validate_cmp(dtype, triton_res, torch_res)
