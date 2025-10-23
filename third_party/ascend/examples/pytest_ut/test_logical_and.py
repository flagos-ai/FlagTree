import pytest
import triton
import triton.language as tl
import torch
import torch_npu
import test_common


def torch_logical_and(x0, x1):
    res = torch.logical_and(x0, x1)
    return res


@triton.jit
def triton_logical_and(in_ptr0, in_ptr1, out_ptr0, XBLOCK: tl.constexpr, XBLOCK_SUB: tl.constexpr):
    offset = tl.program_id(0) * XBLOCK
    base1 = tl.arange(0, XBLOCK_SUB)
    loops1: tl.constexpr = XBLOCK // XBLOCK_SUB
    for loop1 in range(loops1):
        x_index = offset + (loop1 * XBLOCK_SUB) + base1
        tmp0 = tl.load(in_ptr0 + x_index)
        tmp1 = tl.load(in_ptr1 + x_index)
        tmp2 = tmp0.logical_and(tmp1)
        tl.store(out_ptr0 + x_index, tmp2)


@pytest.mark.parametrize('param_list',
                         [
                             ['bool', (2, 4096, 8), 2, 32768, 1024],
                         ])
def test_and(param_list):
    # 生成数据
    dtype, shape, ncore, xblock, xblock_sub = param_list
    x0 = test_common.generate_tensor(shape, dtype).npu()
    x1 = test_common.generate_tensor(shape, dtype).npu()
    # torch结果
    torch_res = torch.logical_and(x0, x1)
    # triton结果
    triton_res = torch.zeros(shape, dtype=eval('torch.' + dtype)).npu()
    triton_logical_and[ncore, 1, 1](x0, x1, triton_res, xblock, xblock_sub)
    # 比较结果
    test_common.validate_cmp(dtype, triton_res, torch_res)
