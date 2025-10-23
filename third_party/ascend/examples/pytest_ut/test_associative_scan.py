import math
import pytest
import torch
import torch_npu
import triton
import triton.language as tl

import test_common


def torch_func(x, dim, reverse):
    if reverse:
        x = torch.flip(x, [dim])
    res = torch.cumsum(x, dim=dim)
    return res


def combine_fn_test_torch(a, b, combine_fn):
    return torch.maximum(a, b)


def torch_func_scan(x: torch.Tensor, dim: int, combine_fn='maximum', reverse=False):
    """
    PyTorch implements associative_scan, with semantics fully aligned with Triton.
    """
    dim = dim % x.ndim

    if reverse:
        x = x.flip(dim)

    N = x.size(dim)
    tensors = torch.unbind(x, dim=dim)

    outputs = []
    carry = tensors[0]
    outputs.append(carry)

    for i in range(1, N):
        carry = combine_fn_test_torch(tensors[i], carry, combine_fn)
        outputs.append(carry)

    output = torch.stack(outputs, dim=dim)

    if reverse:
        output = output.flip(dim)

    return output


@triton.jit
def combine_fn_test(a, b):
    return tl.maximum(a, b)


@triton.jit
def triton_kernel_1d_scan(
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
    ret = tl.associative_scan(x, axis=dim, reverse=reverse, combine_fn=combine_fn_test)
    tl.store(out_ptr0 + idx, ret)


@triton.jit
def triton_kernel_2d_scan(
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
    ret = tl.associative_scan(x, axis=dim, reverse=reverse, combine_fn=combine_fn_test)
    tl.store(out_ptr0 + idx, ret)


@triton.jit
def triton_kernel_3d_scan(
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
    ret = tl.associative_scan(x, axis=dim, reverse=reverse, combine_fn=combine_fn_test)
    tl.store(out_ptr0 + idx, ret)


def triton_func_scan(x, dim, reverse):
    res = torch.empty_like(x)
    print(f"res.dtype = {res.dtype}")
    shape = x.size()
    if len(shape) == 1:
        if dim >= 1:
            pytest.skip("dim >= 1 for 1D tensor, skipping.")
        triton_kernel_1d_scan[1, 1, 1](
            res, x, dim, reverse, x.shape[0], x.shape[0]
        )
    elif len(shape) == 2:
        if dim >= 2:
            pytest.skip("dim >= 2 for 2D tensor, skipping.")
        triton_kernel_2d_scan[1, 1, 1](
            res, x, dim, reverse, x.shape[0], x.shape[1], x.shape[0], x.shape[1]
        )
    elif len(shape) == 3:
        if dim >= 3:
            pytest.skip("dim >= 3 for 3D tensor, skipping.")
        triton_kernel_3d_scan[1, 1, 1](
            res, x, dim, reverse, x.shape[0], x.shape[1], x.shape[2], x.shape[0], x.shape[1], x.shape[2]
        )
    else:
        pytest.skip(f"This testcase unsupported tensor dimension: {len(shape)}")

    return res


@pytest.mark.parametrize("dtype", ['int32', 'float32'])
@pytest.mark.parametrize("shape", [(128,), (8, 4), (128, 4, 16)])
@pytest.mark.parametrize("dim", [0, 1, 2])
@pytest.mark.parametrize("combine_fn", ['maximum', ])
@pytest.mark.parametrize("reverse", [False])
def test_scan(dtype, shape, dim, combine_fn, reverse):
    torch.manual_seed(0)
    x = test_common.generate_tensor(shape=shape, dtype=dtype)
    x_gold = x
    cpu_res = torch_func_scan(x_gold, dim, combine_fn, reverse)
    print(f"cpu_res: {cpu_res}")

    x_npu = x.npu()
    triton_res = triton_func_scan(x_npu, dim, reverse)
    print(f"triton_res: {triton_res}")

    test_common.validate_cmp(dtype, triton_res, cpu_res)
    print(f"Validate PASS")