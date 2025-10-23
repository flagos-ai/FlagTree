import triton
import triton.language as tl
import torch
import torch_npu
import pytest
import test_common


def torch_pointwise(x0, x1):
    res = x0 % x1
    return res


@triton.jit
def triton_mod(in_ptr0, in_ptr1, out_ptr0, XBLOCK: tl.constexpr, XBLOCK_SUB: tl.constexpr):
    offset = tl.program_id(0) * XBLOCK
    base1 = tl.arange(0, XBLOCK_SUB)
    loops1: tl.constexpr = (XBLOCK + XBLOCK_SUB - 1) // XBLOCK_SUB
    for loop1 in range(loops1):
        x0_prime = offset + (loop1 * XBLOCK_SUB) + base1
        x0 = offset + (loop1 * XBLOCK_SUB) + base1
        tmp0 = tl.load(in_ptr0 + (x0), None)
        tmp1 = tl.load(in_ptr1 + (x0), None)
        tmp2 = tmp0 % tmp1
        tl.store(out_ptr0 + (x0), tmp2, None)


@pytest.mark.parametrize('param_list',
                         [
                             ['float32', (2, 4096, 8), 2, 32768, 1024],
                             ['float16', (2, 4096, 8), 2, 32768, 1024],
                             ['int8', (2, 4096, 8), 2, 32768, 1024],
                         ]
                         )

def test_case(param_list):
    dtype, shape, ncore, xblock, xblock_sub = param_list
    if dtype == 'int8':
        x0 = torch.randint(low=1, high=127, size=shape, dtype=eval('torch.' + dtype)).npu()
        x1 = torch.randint(low=1, high=127, size=shape, dtype=eval('torch.' + dtype)).npu()
    else:
        x0 = test_common.generate_tensor(shape, dtype).npu()
        x1 = test_common.generate_tensor(shape, dtype).npu()
    y_ref = torch_pointwise(x0.cpu(), x1.cpu())
    y_ref = y_ref.npu()
    y_cal = torch.zeros(shape, dtype = eval('torch.' + dtype)).npu()
    triton_mod[ncore, 1, 1](x0, x1, y_cal, xblock, xblock_sub)
    #test_common.validate_cmp(dtype, y_cal, y_ref.npu())
    if dtype == 'int8':
        torch.equal(y_cal, y_ref)
    else:
        res = torch.isclose(y_cal, y_ref, rtol=1e-3, atol=1e-3, equal_nan=True)
        if not res.all():
            max_diff = torch.max((y_ref - y_cal)).item()
            raise ValueError(f"Tensors are not close, diff is {max_diff}")
