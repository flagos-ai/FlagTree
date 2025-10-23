import math
import pytest
import torch
import torch_npu
import triton
import triton.language as tl
from triton.runtime.libentry import libentry

from test_common import TestUtils, validate_cmp, get_dtype_size


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
def triton_kernel_1d(
        out_ptr0,
        in_ptr0,
        dim: tl.constexpr,
        reverse: tl.constexpr,
        numel_x: tl.constexpr,
        XBLOCK: tl.constexpr,
):
    tl.static_assert(
        numel_x == XBLOCK, "numel_x must be equal to XBLOCK in this kernel"
    )
    idx = tl.arange(0, XBLOCK)
    x = tl.load(in_ptr0 + idx)
    ret = tl.cumprod(x, axis=dim, reverse=reverse)
    tl.store(out_ptr0 + idx, ret)


@libentry()
@triton.jit
def triton_kernel_2d(
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


@libentry()
@triton.jit
def triton_kernel_3d(
        out_ptr0,
        in_ptr0,
        dim: tl.constexpr,
        reverse: tl.constexpr,
        numel_x: tl.constexpr,
        numel_r: tl.constexpr,
        numel_z: tl.constexpr,
        XBLOCK: tl.constexpr,
        RBLOCK: tl.constexpr,
        ZBLOCK: tl.constexpr,
):
    tl.static_assert(
        numel_x == XBLOCK, "numel_x must be equal to XBLOCK in this kernel"
    )
    tl.static_assert(
        numel_r == RBLOCK, "numel_r must be equal to RBLOCK in this kernel"
    )
    tl.static_assert(
        numel_z == ZBLOCK, "numel_z must be equal to ZBLOCK in this kernel"
    )
    idx_x = tl.arange(0, XBLOCK)
    idx_r = tl.arange(0, RBLOCK)
    idx_z = tl.arange(0, ZBLOCK)
    idx = idx_x[:, None, None] * numel_r * numel_z + idx_r[None, :, None] * numel_z + idx_z[None, None, :]
    x = tl.load(in_ptr0 + idx)
    ret = tl.cumprod(x, axis=dim, reverse=reverse)
    tl.store(out_ptr0 + idx, ret)


@libentry()
@triton.jit
def triton_kernel_4d(
        out_ptr0,
        in_ptr0,
        dim: tl.constexpr,
        reverse: tl.constexpr,
        XB: tl.constexpr,
        YB: tl.constexpr,
        ZB: tl.constexpr,
        MB: tl.constexpr,
):
    xidx = tl.arange(0, XB)
    yidx = tl.arange(0, YB)
    zidx = tl.arange(0, ZB)
    midx = tl.arange(0, MB)
    idx = (xidx[:, None, None, None] * YB * ZB * MB + yidx[None, :, None, None] * ZB * MB +
           zidx[None, None, :, None] * MB + midx[None, None, None, :])
    x = tl.load(in_ptr0 + idx)
    ret = tl.cumprod(x, axis=dim, reverse=reverse)
    tl.store(out_ptr0 + idx, ret)


@libentry()
@triton.jit
def triton_kernel_5d(
        out_ptr0,
        in_ptr0,
        dim: tl.constexpr,
        reverse: tl.constexpr,
        XB: tl.constexpr,
        YB: tl.constexpr,
        ZB: tl.constexpr,
        MB: tl.constexpr,
        NB: tl.constexpr,
):
    xidx = tl.arange(0, XB)
    yidx = tl.arange(0, YB)
    zidx = tl.arange(0, ZB)
    midx = tl.arange(0, MB)
    nidx = tl.arange(0, NB)
    idx = (xidx[:, None, None, None, None] * YB * ZB * MB * NB + yidx[None, :, None, None, None] * ZB * MB * NB +
           zidx[None, None, :, None, None] * MB * NB + midx[None, None, None, :, None] * NB +
           nidx[None, None, None, None, :])
    x = tl.load(in_ptr0 + idx)
    ret = tl.cumprod(x, axis=dim, reverse=reverse)
    tl.store(out_ptr0 + idx, ret)


def convert_cumprod_dtype(x: torch.Tensor) -> torch.Tensor:
    """
    根据 cumprod 类型转换规则，返回转换后的张量。
    """
    dtype_map = {
        torch.int8: torch.int64,
        torch.int16: torch.int64,
        torch.int32: torch.int64,
        torch.int64: torch.int64,
        torch.bfloat16: torch.bfloat16,
        torch.float16: torch.float16,
        torch.float32: torch.float32,
        torch.bool: torch.int64,
    }

    target_dtype = dtype_map.get(x.dtype, None)
    if target_dtype is None:
        raise ValueError(f"Unsupported input dtype for cumprod conversion: {x.dtype}")

    return x.to(target_dtype)


def triton_func(x, dim, reverse):
    x = convert_cumprod_dtype(x)

    res = torch.empty_like(x)
    shape = x.size()
    if len(shape) == 1:
        if dim >= 1:
            pytest.skip("dim >= 1 for 1D tensor, skipping.")
        triton_kernel_1d[1, 1, 1](
            res, x, dim, reverse, x.shape[0], x.shape[0]
        )
    elif len(shape) == 2:
        if dim >= 2:
            pytest.skip("dim >= 2 for 2D tensor, skipping.")
        triton_kernel_2d[1, 1, 1](
            res, x, dim, reverse, x.shape[0], x.shape[1], x.shape[0], x.shape[1]
        )
    elif len(shape) == 3:
        if dim >= 3:
            pytest.skip("dim >= 3 for 3D tensor, skipping.")
        triton_kernel_3d[1, 1, 1](
            res, x, dim, reverse, x.shape[0], x.shape[1], x.shape[2], x.shape[0], x.shape[1], x.shape[2]
        )
    elif len(shape) == 4:
        if dim >= 4:
            pytest.skip("dim >= 4 for 4D tensor, skipping.")
        triton_kernel_4d[1, 1, 1](
            res, x, dim, reverse, x.shape[0], x.shape[1], x.shape[2], x.shape[3]
        )
    elif len(shape) == 5:
        if dim >= 5:
            pytest.skip("dim >= 5 for 5D tensor, skipping.")
        triton_kernel_5d[1, 1, 1](
            res, x, dim, reverse, x.shape[0], x.shape[1], x.shape[2], x.shape[3], x.shape[4]
        )
    else:
        pytest.skip(f"Unsupported tensor dimension: {len(shape)}")

    return res


def cumprod_generate_tensor(shape, dtype):
    if dtype == 'float32' or dtype == 'float16' or dtype == 'bfloat16':
        return torch.rand(size=shape, dtype=eval('torch.' + dtype))
    elif dtype == 'int32' or dtype == 'int64' or dtype == 'int16':
        return torch.randint(low=1, high=5, size=shape, dtype=eval('torch.' + dtype))
    elif dtype == 'int8':
        return torch.randint(low=1, high=5, size=shape, dtype=eval('torch.' + dtype))
    elif dtype == 'bool':
        return torch.randint(low=0, high=2, size=shape).bool()
    else:
        raise ValueError(f"Unsupported dtype: {dtype}")


def should_skip_due_to_mem(dtype, shape):
    dtype_size = get_dtype_size(dtype)
    total_mem = dtype_size * math.prod(shape)

    if dtype in ('int8', 'bool'):
        threshold = TestUtils.ub_size / 13
    else:
        threshold = TestUtils.ub_size / 6

    if total_mem >= threshold:
        pytest.skip(f"dtype:{dtype} shape:{shape} mem overflow")


# reverse=True not support;
@pytest.mark.parametrize("dtype", TestUtils.full_dtype)
@pytest.mark.parametrize("shape", TestUtils.full_shape)
@pytest.mark.parametrize("dim", [0, 1, 2, 3, 4])
@pytest.mark.parametrize("reverse", [False])
def test_cumprod(dtype, shape, dim, reverse):
    should_skip_due_to_mem(dtype, shape)

    x = cumprod_generate_tensor(shape=shape, dtype=dtype)
    x_npu = x.npu()

    triton_res = triton_func(x_npu, dim, reverse)

    x_gold = x
    cpu_res = torch_func(x_gold, dim, reverse)

    validate_cmp(dtype, triton_res, cpu_res)
