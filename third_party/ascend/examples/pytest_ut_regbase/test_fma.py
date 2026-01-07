import triton
import triton.language as tl
import torch
import pytest
from test_common import (
    generate_tensor,
    validate_cmp,
    _float_dtypes_without_bf16 as _float_dtypes,
    _int_dtypes,
    _shape_1d,
)

# @pytest.mark.parametrize('dtype', ['float32'])
@pytest.mark.parametrize('xblock_sub', [32])
@pytest.mark.parametrize('dtype', _float_dtypes)
# @pytest.mark.parametrize('xblock_sub', _shape_1d)
def test_fma(dtype, xblock_sub):

    xblock = triton.next_power_of_2(xblock_sub)
    shape = (xblock,)

    def torch_func(x0, x1, x2):
        return x0 * x1 + x2

    def get_autotune_config():
        return [
            triton.Config({'XBLOCK': xblock, 'XBLOCK_SUB': xblock_sub}),
        ]

    @triton.autotune(
            configs=get_autotune_config(),
            key=['numel'],
        )
    @triton.jit
    def triton_kernel(in_ptr0, in_ptr1, in_ptr2, out_ptr0, numel, XBLOCK: tl.constexpr, XBLOCK_SUB: tl.constexpr):
        offset = tl.program_id(0) * XBLOCK
        base = tl.arange(0, XBLOCK_SUB)
        num_loop: tl.constexpr = (XBLOCK + XBLOCK_SUB - 1) // XBLOCK_SUB
        for loop in range(num_loop):
            idx = offset + loop * XBLOCK_SUB + base
            msk = idx < numel
            x0 = tl.load(in_ptr0 + idx, mask=msk)
            x1 = tl.load(in_ptr1 + idx, mask=msk)
            x2 = tl.load(in_ptr2 + idx, mask=msk)
            y0 = tl.fma(x0, x1, x2)
            tl.store(out_ptr0 + idx, y0, mask=msk)

    def triton_func(x0, x1, x2):
        numel = x0.numel()
        y0 = torch.empty_like(x0)
        grid = lambda meta: (triton.cdiv(numel, meta['XBLOCK']), )
        triton_kernel[grid](x0, x1, x2, y0, numel)
        return y0

    x0 = generate_tensor(shape, dtype).npu()
    x1 = generate_tensor(shape, dtype).npu()
    x2 = generate_tensor(shape, dtype).npu()

    torch_ref = torch_func(x0, x1, x2)
    triton_cal = triton_func(x0, x1, x2)
    validate_cmp(dtype, triton_cal, torch_ref)

# test_fma('float32', 32)