import pytest
import triton
import triton.language as tl
import torch
import torch_npu
import test_common


def torch_ge(x0, x1, dtype):
    res = torch.where(torch.ge(x0, x1), torch.ones_like(x0), torch.zeros_like(x0)).to(eval('torch.' + dtype))
    return res


@triton.jit
def triton_ge(in_ptr0, in_ptr1, out_ptr0, XBLOCK: tl.constexpr, XBLOCK_SUB: tl.constexpr):
    offset = tl.program_id(0) * XBLOCK
    base1 = tl.arange(0, XBLOCK_SUB)
    loops1: tl.constexpr = XBLOCK // XBLOCK_SUB
    for loop1 in range(loops1):
        x0 = offset + (loop1 * XBLOCK_SUB) + base1
        tmp0 = tl.load(in_ptr0 + (x0), None)
        tmp1 = tl.load(in_ptr1 + (x0), None)
        tmp2 = tmp0 >= tmp1
        tl.store(out_ptr0 + (x0), tmp2, None)


@pytest.mark.parametrize('param_list',
                         [
                             ['float16', (2, 4096, 8), 2, 32768, 1024],
                             ['float32', (2, 4096, 8), 2, 32768, 1024],
                             ['int8', (2, 4096, 8), 2, 32768, 1024],
                         ])
def test_ge(param_list):
    # 生成数据
    dtype, shape, ncore, xblock, xblock_sub = param_list
    x0 = test_common.generate_tensor(shape, dtype).npu()
    x1 = test_common.generate_tensor(shape, dtype).npu()
    # torch结果
    torch_res = torch_ge(x0, x1, dtype)
    # triton结果
    triton_res = torch.empty_like(x0)
    triton_ge[ncore, 1, 1](x0, x1, triton_res, xblock, xblock_sub)
    # 比较结果
    test_common.validate_cmp(dtype, triton_res, torch_res)
