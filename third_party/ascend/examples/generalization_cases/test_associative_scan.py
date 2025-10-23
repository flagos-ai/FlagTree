import math
import pytest
import torch
import torch_npu
import triton
import triton.language as tl

import test_common
from test_common import TestUtils, get_dtype_size


def combine_fn_test_torch(a, b, combine_fn):
    if combine_fn == 'maximum_fn':
        return torch.maximum(a, b)  # 最大值
    elif combine_fn == 'minimum_fn':
        return torch.minimum(a, b)  # 最小值
    elif combine_fn == 'bitwise_xor_fn':
        return a ^ b  # 按位异或
    elif combine_fn == 'bitwise_or_fn':
        return a | b  # 按位异
    elif combine_fn == 'bitwise_and_fn':
        return a & b  # 按位与
    else:
        pytest.skip("The combine_fn is not within the following scope , skipping.")


def torch_func_scan(input: torch.Tensor, dim: int, combine_fn='maximum', reverse=False):
    """
    PyTorch 实现 associative_scan，语义与 Triton 完全对齐
    支持任意 combine_fn（如 a|b, a&b, min, max 等）
    """
    dim = dim % input.ndim

    if reverse:
        input = input.flip(dim)

    N = input.size(dim)

    tensors = torch.unbind(input, dim=dim)

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
def bitwise_and_fn(a, b):
    return a & b


@triton.jit
def bitwise_or_fn(a, b):
    return a | b


@triton.jit
def bitwise_xor_fn(a, b):
    return a ^ b


@triton.jit
def minimum_fn(a, b):
    return tl.minimum(a, b)


@triton.jit
def maximum_fn(a, b):
    return tl.maximum(a, b)


@triton.jit
def triton_kernel_1d_scan(
        out_ptr0,
        in_ptr0,
        dim: tl.constexpr,
        reverse: tl.constexpr,
        numel_x: tl.constexpr,
        XBLOCK: tl.constexpr,
        combine_fn_name: tl.constexpr,
):
    tl.static_assert(
        numel_x == XBLOCK, "numel_x must be equal to XBLOCK in this kernel"
    )
    idx = tl.arange(0, XBLOCK)
    x = tl.load(in_ptr0 + idx)
    if combine_fn_name == "maximum_fn":
        ret = tl.associative_scan(x, axis=dim, reverse=reverse, combine_fn=maximum_fn)
    elif combine_fn_name == "minimum_fn":
        ret = tl.associative_scan(x, axis=dim, reverse=reverse, combine_fn=minimum_fn)
    elif combine_fn_name == "bitwise_or_fn":
        ret = tl.associative_scan(x, axis=dim, reverse=reverse, combine_fn=bitwise_or_fn)
    elif combine_fn_name == "bitwise_xor_fn":
        ret = tl.associative_scan(x, axis=dim, reverse=reverse, combine_fn=bitwise_xor_fn)
    elif combine_fn_name == "bitwise_and_fn":
        ret = tl.associative_scan(x, axis=dim, reverse=reverse, combine_fn=bitwise_and_fn)

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
        combine_fn_name: tl.constexpr,
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

    if combine_fn_name == "maximum_fn":
        ret = tl.associative_scan(x, axis=dim, reverse=reverse, combine_fn=maximum_fn)
    elif combine_fn_name == "minimum_fn":
        ret = tl.associative_scan(x, axis=dim, reverse=reverse, combine_fn=minimum_fn)
    elif combine_fn_name == "bitwise_or_fn":
        ret = tl.associative_scan(x, axis=dim, reverse=reverse, combine_fn=bitwise_or_fn)
    elif combine_fn_name == "bitwise_xor_fn":
        ret = tl.associative_scan(x, axis=dim, reverse=reverse, combine_fn=bitwise_xor_fn)
    elif combine_fn_name == "bitwise_and_fn":
        ret = tl.associative_scan(x, axis=dim, reverse=reverse, combine_fn=bitwise_and_fn)
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
        combine_fn_name: tl.constexpr,
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
    if combine_fn_name == "maximum_fn":
        ret = tl.associative_scan(x, axis=dim, reverse=reverse, combine_fn=maximum_fn)
    elif combine_fn_name == "minimum_fn":
        ret = tl.associative_scan(x, axis=dim, reverse=reverse, combine_fn=minimum_fn)
    elif combine_fn_name == "bitwise_or_fn":
        ret = tl.associative_scan(x, axis=dim, reverse=reverse, combine_fn=bitwise_or_fn)
    elif combine_fn_name == "bitwise_xor_fn":
        ret = tl.associative_scan(x, axis=dim, reverse=reverse, combine_fn=bitwise_xor_fn)
    elif combine_fn_name == "bitwise_and_fn":
        ret = tl.associative_scan(x, axis=dim, reverse=reverse, combine_fn=bitwise_and_fn)
    tl.store(out_ptr0 + idx, ret)


@triton.jit
def triton_kernel_4d_scan(
        out_ptr0,
        in_ptr0,
        dim: tl.constexpr,
        reverse: tl.constexpr,
        XB: tl.constexpr,
        YB: tl.constexpr,
        ZB: tl.constexpr,
        MB: tl.constexpr,
        combine_fn_name: tl.constexpr,
):
    xidx = tl.arange(0, XB)
    yidx = tl.arange(0, YB)
    zidx = tl.arange(0, ZB)
    midx = tl.arange(0, MB)
    idx = (xidx[:, None, None, None] * YB * ZB * MB + yidx[None, :, None, None] * ZB * MB +
           zidx[None, None, :, None] * MB + midx[None, None, None, :])
    x = tl.load(in_ptr0 + idx)
    if combine_fn_name == "maximum_fn":
        ret = tl.associative_scan(x, axis=dim, reverse=reverse, combine_fn=maximum_fn)
    elif combine_fn_name == "minimum_fn":
        ret = tl.associative_scan(x, axis=dim, reverse=reverse, combine_fn=minimum_fn)
    elif combine_fn_name == "bitwise_or_fn":
        ret = tl.associative_scan(x, axis=dim, reverse=reverse, combine_fn=bitwise_or_fn)
    elif combine_fn_name == "bitwise_xor_fn":
        ret = tl.associative_scan(x, axis=dim, reverse=reverse, combine_fn=bitwise_xor_fn)
    elif combine_fn_name == "bitwise_and_fn":
        ret = tl.associative_scan(x, axis=dim, reverse=reverse, combine_fn=bitwise_and_fn)
    tl.store(out_ptr0 + idx, ret)


@triton.jit
def triton_kernel_5d_scan(
        out_ptr0,
        in_ptr0,
        dim: tl.constexpr,
        reverse: tl.constexpr,
        XB: tl.constexpr,
        YB: tl.constexpr,
        ZB: tl.constexpr,
        MB: tl.constexpr,
        NB: tl.constexpr,
        combine_fn_name: tl.constexpr,
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
    if combine_fn_name == "maximum_fn":
        ret = tl.associative_scan(x, axis=dim, reverse=reverse, combine_fn=maximum_fn)
    elif combine_fn_name == "minimum_fn":
        ret = tl.associative_scan(x, axis=dim, reverse=reverse, combine_fn=minimum_fn)
    elif combine_fn_name == "bitwise_or_fn":
        ret = tl.associative_scan(x, axis=dim, reverse=reverse, combine_fn=bitwise_or_fn)
    elif combine_fn_name == "bitwise_xor_fn":
        ret = tl.associative_scan(x, axis=dim, reverse=reverse, combine_fn=bitwise_xor_fn)
    elif combine_fn_name == "bitwise_and_fn":
        ret = tl.associative_scan(x, axis=dim, reverse=reverse, combine_fn=bitwise_and_fn)
    tl.store(out_ptr0 + idx, ret)


def triton_func_scan(x, dim, combine_fn, reverse):
    res = torch.empty_like(x)
    shape = x.size()

    if len(shape) == 1:
        if dim >= 1:
            pytest.skip("dim >= 1 for 1D tensor, skipping.")
        triton_kernel_1d_scan[1, 1, 1](
            res, x, dim, reverse, x.shape[0], x.shape[0], combine_fn
        )
    elif len(shape) == 2:
        if dim >= 2:
            pytest.skip("dim >= 2 for 2D tensor, skipping.")
        triton_kernel_2d_scan[1, 1, 1](
            res, x, dim, reverse, x.shape[0], x.shape[1], x.shape[0], x.shape[1], combine_fn
        )
    elif len(shape) == 3:
        if dim >= 3:
            pytest.skip("dim >= 3 for 3D tensor, skipping.")
        triton_kernel_3d_scan[1, 1, 1](
            res, x, dim, reverse, x.shape[0], x.shape[1], x.shape[2], x.shape[0], x.shape[1], x.shape[2], combine_fn
        )
    elif len(shape) == 4:
        if dim >= 4:
            pytest.skip("dim >= 4 for 4D tensor, skipping.")
        triton_kernel_4d_scan[1, 1, 1](
            res, x, dim, reverse, x.shape[0], x.shape[1], x.shape[2], x.shape[3], combine_fn
        )
    elif len(shape) == 5:
        if dim >= 5:
            pytest.skip("dim >= 5 for 5D tensor, skipping.")
        triton_kernel_5d_scan[1, 1, 1](
            res, x, dim, reverse, x.shape[0], x.shape[1], x.shape[2], x.shape[3], x.shape[4], combine_fn
        )
    else:
        pytest.skip(f"Unsupported tensor dimension: {len(shape)}")

    return res


def should_skip_due_to_mem(dtype, shape):
    dtype_size = get_dtype_size(dtype)
    total_mem = dtype_size * math.prod(shape)
    if dtype in ('int8', 'bool'):
        threshold = TestUtils.ub_size / 13
    else:
        threshold = TestUtils.ub_size / 6

    if total_mem >= threshold:
        pytest.skip(f"dtype:{dtype} shape:{shape} mem overflow")


@pytest.mark.parametrize("dtype", ['int8', 'int16', 'int32', 'int64', 'bool'])
@pytest.mark.parametrize("shape", TestUtils.test_shape1d)
@pytest.mark.parametrize("dim", [0])
@pytest.mark.parametrize("combine_fn",
                         ['maximum_fn', 'minimum_fn', 'bitwise_or_fn', 'bitwise_xor_fn', 'bitwise_and_fn'])
@pytest.mark.parametrize("reverse", [False])
def test_scan_1d(dtype, shape, dim, combine_fn, reverse):
    should_skip_due_to_mem(dtype, shape)
    torch.manual_seed(0)
    x = test_common.generate_tensor(shape=shape, dtype=dtype)
    x_gold = x
    cpu_res = torch_func_scan(x_gold, dim, combine_fn, reverse)

    x_npu = x.npu()
    triton_res = triton_func_scan(x_npu, dim, combine_fn, reverse)

    test_common.validate_cmp(dtype, triton_res, cpu_res)


@pytest.mark.parametrize("dtype", ['int8', 'int16', 'int32', 'int64', 'bool'])
@pytest.mark.parametrize("shape", TestUtils.test_shape2d)
@pytest.mark.parametrize("dim", [1])
@pytest.mark.parametrize("combine_fn",
                         ['maximum_fn', 'minimum_fn', 'bitwise_or_fn', 'bitwise_xor_fn', 'bitwise_and_fn'])
@pytest.mark.parametrize("reverse", [False])
def test_scan_2d(dtype, shape, dim, combine_fn, reverse):
    should_skip_due_to_mem(dtype, shape)
    torch.manual_seed(0)
    x = test_common.generate_tensor(shape=shape, dtype=dtype)
    x_gold = x
    cpu_res = torch_func_scan(x_gold, dim, combine_fn, reverse)

    x_npu = x.npu()
    triton_res = triton_func_scan(x_npu, dim, combine_fn, reverse)

    test_common.validate_cmp(dtype, triton_res, cpu_res)


@pytest.mark.parametrize("dtype", ['int8', 'int16', 'int32', 'int64', 'bool'])
@pytest.mark.parametrize("shape", TestUtils.test_shape3d)
@pytest.mark.parametrize("dim", [2])
@pytest.mark.parametrize("combine_fn",
                         ['maximum_fn', 'minimum_fn', 'bitwise_or_fn', 'bitwise_xor_fn', 'bitwise_and_fn'])
@pytest.mark.parametrize("reverse", [False])
def test_scan_3d(dtype, shape, dim, combine_fn, reverse):
    should_skip_due_to_mem(dtype, shape)
    torch.manual_seed(0)
    x = test_common.generate_tensor(shape=shape, dtype=dtype)
    x_gold = x
    cpu_res = torch_func_scan(x_gold, dim, combine_fn, reverse)

    x_npu = x.npu()
    triton_res = triton_func_scan(x_npu, dim, combine_fn, reverse)

    test_common.validate_cmp(dtype, triton_res, cpu_res)


@pytest.mark.parametrize("dtype", ['int8', 'int16', 'int32', 'int64', 'bool'])
@pytest.mark.parametrize("shape", TestUtils.test_shape4d)
@pytest.mark.parametrize("dim", [3])
@pytest.mark.parametrize("combine_fn",
                         ['maximum_fn', 'minimum_fn', 'bitwise_or_fn', 'bitwise_xor_fn', 'bitwise_and_fn'])
@pytest.mark.parametrize("reverse", [False])
def test_scan_4d(dtype, shape, dim, combine_fn, reverse):
    should_skip_due_to_mem(dtype, shape)
    torch.manual_seed(0)
    x = test_common.generate_tensor(shape=shape, dtype=dtype)
    x_gold = x
    cpu_res = torch_func_scan(x_gold, dim, combine_fn, reverse)

    x_npu = x.npu()
    triton_res = triton_func_scan(x_npu, dim, combine_fn, reverse)

    test_common.validate_cmp(dtype, triton_res, cpu_res)


@pytest.mark.parametrize("dtype", ['int8', 'int16', 'int32', 'int64', 'bool'])
@pytest.mark.parametrize("shape", TestUtils.test_shape5d)
@pytest.mark.parametrize("dim", [4])
@pytest.mark.parametrize("combine_fn",
                         ['maximum_fn', 'minimum_fn', 'bitwise_or_fn', 'bitwise_xor_fn', 'bitwise_and_fn'])
@pytest.mark.parametrize("reverse", [False])
def test_scan_5d(dtype, shape, dim, combine_fn, reverse):
    should_skip_due_to_mem(dtype, shape)
    torch.manual_seed(0)
    x = test_common.generate_tensor(shape=shape, dtype=dtype)
    x_gold = x
    cpu_res = torch_func_scan(x_gold, dim, combine_fn, reverse)

    x_npu = x.npu()
    triton_res = triton_func_scan(x_npu, dim, combine_fn, reverse)

    test_common.validate_cmp(dtype, triton_res, cpu_res)


@pytest.mark.parametrize("dtype", ['float16', 'float32'])
@pytest.mark.parametrize("shape", TestUtils.test_shape1d)
@pytest.mark.parametrize("dim", [0])
@pytest.mark.parametrize("combine_fn",
                         ['maximum_fn', 'minimum_fn'])
@pytest.mark.parametrize("reverse", [False])
def test_scan_float_1d(dtype, shape, dim, combine_fn, reverse):
    should_skip_due_to_mem(dtype, shape)
    torch.manual_seed(0)
    x = test_common.generate_tensor(shape=shape, dtype=dtype)
    x_gold = x
    cpu_res = torch_func_scan(x_gold, dim, combine_fn, reverse)

    x_npu = x.npu()
    triton_res = triton_func_scan(x_npu, dim, combine_fn, reverse)

    test_common.validate_cmp(dtype, triton_res, cpu_res)


@pytest.mark.parametrize("dtype", ['float16', 'float32'])
@pytest.mark.parametrize("shape", TestUtils.test_shape2d)
@pytest.mark.parametrize("dim", [1])
@pytest.mark.parametrize("combine_fn",
                         ['maximum_fn', 'minimum_fn'])
@pytest.mark.parametrize("reverse", [False])
def test_scan_float_2d(dtype, shape, dim, combine_fn, reverse):
    should_skip_due_to_mem(dtype, shape)
    torch.manual_seed(0)
    x = test_common.generate_tensor(shape=shape, dtype=dtype)
    x_gold = x
    cpu_res = torch_func_scan(x_gold, dim, combine_fn, reverse)

    x_npu = x.npu()
    triton_res = triton_func_scan(x_npu, dim, combine_fn, reverse)

    test_common.validate_cmp(dtype, triton_res, cpu_res)


@pytest.mark.parametrize("dtype", ['float16', 'float32'])
@pytest.mark.parametrize("shape", TestUtils.test_shape3d)
@pytest.mark.parametrize("dim", [2])
@pytest.mark.parametrize("combine_fn",
                         ['maximum_fn', 'minimum_fn'])
@pytest.mark.parametrize("reverse", [False])
def test_scan_float_1d(dtype, shape, dim, combine_fn, reverse):
    should_skip_due_to_mem(dtype, shape)
    torch.manual_seed(0)
    x = test_common.generate_tensor(shape=shape, dtype=dtype)
    x_gold = x
    cpu_res = torch_func_scan(x_gold, dim, combine_fn, reverse)

    x_npu = x.npu()
    triton_res = triton_func_scan(x_npu, dim, combine_fn, reverse)

    test_common.validate_cmp(dtype, triton_res, cpu_res)


@pytest.mark.parametrize("dtype", ['float16', 'float32'])
@pytest.mark.parametrize("shape", TestUtils.test_shape4d)
@pytest.mark.parametrize("dim", [3])
@pytest.mark.parametrize("combine_fn",
                         ['maximum_fn', 'minimum_fn'])
@pytest.mark.parametrize("reverse", [False])
def test_scan_float_1d(dtype, shape, dim, combine_fn, reverse):
    should_skip_due_to_mem(dtype, shape)
    torch.manual_seed(0)
    x = test_common.generate_tensor(shape=shape, dtype=dtype)
    x_gold = x
    cpu_res = torch_func_scan(x_gold, dim, combine_fn, reverse)

    x_npu = x.npu()
    triton_res = triton_func_scan(x_npu, dim, combine_fn, reverse)

    test_common.validate_cmp(dtype, triton_res, cpu_res)


@pytest.mark.parametrize("dtype", ['float16', 'float32'])
@pytest.mark.parametrize("shape", TestUtils.test_shape5d)
@pytest.mark.parametrize("dim", [4])
@pytest.mark.parametrize("combine_fn",
                         ['maximum_fn', 'minimum_fn'])
@pytest.mark.parametrize("reverse", [False])
def test_scan_float_1d(dtype, shape, dim, combine_fn, reverse):
    should_skip_due_to_mem(dtype, shape)
    torch.manual_seed(0)
    x = test_common.generate_tensor(shape=shape, dtype=dtype)
    x_gold = x
    cpu_res = torch_func_scan(x_gold, dim, combine_fn, reverse)

    x_npu = x.npu()
    triton_res = triton_func_scan(x_npu, dim, combine_fn, reverse)

    test_common.validate_cmp(dtype, triton_res, cpu_res)


@pytest.mark.parametrize("dtype", ['float16', 'float32'])
@pytest.mark.parametrize("shape", TestUtils.test_shape1d)
@pytest.mark.parametrize("dim", [0])
@pytest.mark.parametrize("combine_fn",
                         ['bitwise_or_fn', 'bitwise_xor_fn', 'bitwise_and_fn'])
@pytest.mark.parametrize("reverse", [False])
@test_common.raises_with_match(triton.compiler.errors.CompilationError, "unexpected type")
def test_scan_float_invalid(dtype, shape, dim, combine_fn, reverse):
    should_skip_due_to_mem(dtype, shape)
    torch.manual_seed(0)
    x = test_common.generate_tensor(shape=shape, dtype=dtype)

    x_npu = x.npu()
    triton_res = triton_func_scan(x_npu, dim, combine_fn, reverse)

@pytest.mark.parametrize("dtype", ['int32'])
@pytest.mark.parametrize("shape", TestUtils.test_shape1d)
@pytest.mark.parametrize("dim", [0])
@pytest.mark.parametrize("combine_fn",
                         ['maximum_fn', 'minimum_fn', 'bitwise_or_fn', 'bitwise_xor_fn', 'bitwise_and_fn'])
@pytest.mark.parametrize("reverse", [True])
@test_common.raises_with_match(triton.compiler.errors.MLIRCompilationError,
                               "reverse=True is not yet supported for scan op")
def test_scan_float_invalid_reverse(dtype, shape, dim, combine_fn, reverse):
    should_skip_due_to_mem(dtype, shape)
    torch.manual_seed(0)
    x = test_common.generate_tensor(shape=shape, dtype=dtype)

    x_npu = x.npu()
    triton_res = triton_func_scan(x_npu, dim, combine_fn, reverse)

