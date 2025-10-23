import pytest
import torch
import torch_npu
import triton
import triton.language as tl
from triton.runtime.libentry import libentry

from test_common import _all_dtypes_no_bool, validate_cmp


def torch_func(x, dim, reverse):
    is_bf16 = x.dtype == torch.bfloat16
    if is_bf16:
        x = x.to(torch.float32)
    if reverse:
        x = torch.flip(x, [dim])
    res = torch.cumprod(x, dim=dim)
    if is_bf16:
        res = res.to(torch.bfloat16)
    return res


@libentry()
@triton.jit
def triton_kernel(
        out_ptr0,
        in_ptr0,
        dim: tl.constexpr,
        reverse: tl.constexpr,
        numel_x: tl.constexpr,
        numel_r: tl.constexpr,
        XBLOCK: tl.constexpr,
        RBLOCK: tl.constexpr,
):
    tl.static_assert(
        numel_x == XBLOCK, "numel_x must be equal to XBLOCK in this kernel"
    )
    tl.static_assert(
        numel_r == RBLOCK, "numel_r must be equal to RBLOCK in this kernel"
    )
    idx_x = tl.arange(0, XBLOCK)
    idx_r = tl.arange(0, RBLOCK)
    idx = idx_x[:, None] * numel_r + idx_r[None, :]
    x = tl.load(in_ptr0 + idx)
    ret = tl.cumprod(x, axis=dim, reverse=reverse)
    tl.store(out_ptr0 + idx, ret)


def triton_func(x, dim, reverse):
    res = torch.empty_like(x)
    triton_kernel[1, 1, 1](
        res, x, dim, reverse, x.shape[0], x.shape[1], x.shape[0], x.shape[1]
    )
    return res


def cumprod_generate_tensor(shape, dtype):
    if dtype == 'float32' or dtype == 'float16' or dtype == 'bfloat16':
        return torch.rand(size=shape, dtype=eval('torch.' + dtype))
    elif dtype == 'int32' or dtype == 'int64' or dtype == 'int16':
        return torch.randint(low=0, high=3, size=shape, dtype=eval('torch.' + dtype))
    elif dtype == 'int8':
        return torch.randint(low=0, high=3, size=shape, dtype=eval('torch.' + dtype))
    else:
        raise ValueError(f"Unsupported dtype: {dtype}")


# dtype=int8, reverse=True not support;
not_support_dtype = {'int8', 'bool'}
support_dtypes = [dtype for dtype in _all_dtypes_no_bool if dtype not in not_support_dtype]


@pytest.mark.parametrize("dtype", support_dtypes)
@pytest.mark.parametrize("shape", [(7, 23)])
@pytest.mark.parametrize("dim", [0, 1])
@pytest.mark.parametrize("reverse", [False])
def test_cumprod(dtype, shape, dim, reverse):
    x0 = cumprod_generate_tensor(shape=shape, dtype=dtype).npu()
    triton_cal = triton_func(x0, dim, reverse)
    torch_ref = torch_func(x0, dim, reverse)
    validate_cmp(dtype, torch_ref, triton_cal)
