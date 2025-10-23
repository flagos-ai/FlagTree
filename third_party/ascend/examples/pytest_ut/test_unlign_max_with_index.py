import torch
import torch_npu
import triton
import triton.language as tl
import pytest
import test_common


@triton.jit
def promote_to_tensor(x):
    # Addition promotes to tensor for us
    return x + tl.zeros((1,), tl.int1)


@triton.jit
def is_floating(x):
    return promote_to_tensor(x).dtype.is_floating()


@triton.jit
def maximum_with_index(a_value, a_index, b_value, b_index):
    mask = a_value > b_value
    equal = a_value == b_value
    if is_floating(a_value):
        a_isnan = a_value != a_value
        b_isnan = b_value != b_value
        mask |= a_isnan and not b_isnan
        # Consider NaNs as equal
        equal |= a_isnan and b_isnan

    # Prefer lowest index if values are equal
    mask |= equal & (a_index < b_index)
    return tl.where(mask, a_value, b_value), tl.where(mask, a_index, b_index)


@triton.jit
def max_with_index(value, index, dim):
    return tl.reduce((value, index), dim, maximum_with_index)


def torch_max_pr_index(x0):
    res = torch.max(x0, 0)
    return res


# x0_numel 128, r1_numel 65, 128, 128, 32
@triton.jit
def triton_kernel(in_ptr0, in_ptr1, out_ptr0, out_ptr1, x0_numel, r1_numel, XBLOCK: tl.constexpr,
                  XBLOCK_SUB: tl.constexpr, RBLOCK: tl.constexpr):
    offset = tl.program_id(0) * XBLOCK
    base1 = tl.arange(0, XBLOCK_SUB)
    loops1: tl.constexpr = (XBLOCK + XBLOCK_SUB - 1) // XBLOCK_SUB
    base2 = tl.arange(0, RBLOCK)
    loops2: tl.constexpr = (r1_numel + RBLOCK - 1) // RBLOCK
    for loop1 in range(loops1):
        x0 = offset + (loop1 * XBLOCK_SUB) + base1[:, None]
        xmask = x0 < x0_numel
        for loop2 in range(loops2):
            r1 = loop2 * RBLOCK + base2[None, :]
            rmask = r1 < r1_numel
            tmp0 = tl.load(in_ptr0 + (r1 + (65 * x0)), rmask & xmask, other=0)
            tmp2 = tl.load(in_ptr1 + (r1 + (65 * x0)), rmask & xmask, other=0)
            tmp11 = tl.reshape(tmp0, [XBLOCK_SUB, RBLOCK])
            tmp22 = tl.reshape(tmp2, [XBLOCK_SUB, RBLOCK])
            tmp4, tmp5 = max_with_index(tmp11, tmp22, 0)
            tl.store(out_ptr0 + r1, tmp4[None, :], None)
            tl.store(out_ptr1 + r1, tmp5[None, :], None)


@pytest.mark.parametrize('param_list',
                         [
                             ['float32', (128, 65), 1, 128, 65, 32],
                         ]
                         )
def test_max_with_index_dim0(param_list):
    dtype, shape, ncore, XB, YB, RBLOCK = param_list
    import math
    numel = math.prod(shape)
    xblock = numel // shape[-1] // ncore
    assert (ncore * xblock * shape[-1] == numel)

    b = test_common.generate_tensor(shape, dtype).npu()
    idx = torch.zeros((shape), device='npu', dtype=torch.int32)
    c, d = torch.max(b, dim=0)

    print(f"std_ret = {c[0:4]}")

    ret = torch.empty((YB), device='npu', dtype=torch.float32)
    ret1 = torch.zeros((YB), device='npu', dtype=torch.int32)
    triton_kernel[ncore, 1, 1](b, idx, ret, ret1, shape[0], shape[1], shape[0], shape[0], RBLOCK)
    print(f"triton_ret = {ret[0:4]}")
    d = d.to(torch.int32)
    ret1 = ret1.to(torch.int32)
    print(f"d = {d[0:4]}")
    print(f"triton_ret1 = {ret1[0:4]}")
    assert torch.allclose(c, ret, rtol=1e-03, atol=1e-03, equal_nan=True)
    assert torch.allclose(d, ret1, rtol=1e-03, atol=1e-03, equal_nan=True)
