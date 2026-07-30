import pytest
import torch

import triton
from triton.backends.spine_triton.driver import CPUDriver

triton.runtime.driver.set_active(CPUDriver())
import flag_gems
import numpy as np
from .conftest import QUICK_MODE, TO_CPU
import random
import time
# Make sure every thread has same seed.
random.seed(time.time() // 100)

from .accuracy_utils import (
    FLOAT_DTYPES,
    SCALARS,
    gems_assert_close,
    to_reference,
    INT_DTYPES,
    POINTWISE_SHAPES,
    REDUCTION_SHAPES,
    gems_assert_equal,
    unsqueeze_tensor,
    CONTIGUOUS_SHAPE_STRIDES_2D,
    IRREGULAR_SHAPE_STRIDES,
    SHAPE_STRIDES,
    SPECIAL_SHAPES,
)

MN_SHAPES = [(1, 32), (160, 1024), (5333, 497)]
MNK_SHAPES = ([(1, 1, 32), (15, 160, 1024), (495, 5333, 71)])
SHAPE_DIAGONAL = list(zip(POINTWISE_SHAPES, [-2, -2, -1, 0, 1, 3]))


def replace_zeros(inp):
    return torch.where(inp == 0, 1, inp)


DIMS_LIST = [1] if QUICK_MODE else [0, 1, [0, 1], [1, 0]]
KEEPDIM_DIMS = ([(True, DIMS_LIST[0])] if QUICK_MODE else list(zip([True, False] * 2, DIMS_LIST)))
DIM_LIST = [1] if QUICK_MODE else [0, 1]
KEEPDIM_DIMS_SHAPE = ([(True, DIMS_LIST[0], REDUCTION_SHAPES[0])] if QUICK_MODE else list(
    zip([True, False] * 2, DIMS_LIST, REDUCTION_SHAPES + [(7, 4, 11, 1)])))
SMOOTH_IGNORE_SHAPE = ([(0.1, 1, REDUCTION_SHAPES[0])] if QUICK_MODE else list(
    zip([0, 0.1, 1], [1, 200, -100], REDUCTION_SHAPES)))
SMOOTH_SHAPE = ([(0.1, REDUCTION_SHAPES[0])] if QUICK_MODE else list(zip([1, 0.1, 0], REDUCTION_SHAPES)))
DIM_SHAPE_STRIDES = ([(1, *CONTIGUOUS_SHAPE_STRIDES_2D[1])] if QUICK_MODE else list(
    (random.randint(0,
                    len(shape) - 1), shape, stride) for shape, stride in SHAPE_STRIDES))
REGULAR_DIM_SHAPE_STRIDES = ([(1, *CONTIGUOUS_SHAPE_STRIDES_2D[1])] if QUICK_MODE else list(
    (random.randint(0,
                    len(shape) - 1), shape, stride) for shape, stride in CONTIGUOUS_SHAPE_STRIDES_2D))
IRREGULAR_DIM_SHAPE_STRIDES = [(3, *IRREGULAR_SHAPE_STRIDES)]

THRESHOLD_SHAPE = ([(0.3, REDUCTION_SHAPES[0])] if QUICK_MODE else list(zip([0.3, 0.5, 0.7], REDUCTION_SHAPES)))
CROSS_ENTROPY_LOSS_REDUCTION = ["mean"] if QUICK_MODE else ["mean", "none", "sum"]


@pytest.mark.addmm
@pytest.mark.linear
@pytest.mark.matmul
@pytest.mark.parametrize("M, N, K", MNK_SHAPES)
@pytest.mark.parametrize("scalar", SCALARS)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_addmm(M, N, K, scalar, dtype):
    mat1 = torch.randn((M, K), dtype=dtype, device=flag_gems.device, requires_grad=False)
    mat2 = torch.randn((K, N), dtype=dtype, device=flag_gems.device, requires_grad=False)
    bias1 = torch.randn((N, ), dtype=dtype, device=flag_gems.device, requires_grad=False)

    alpha = beta = scalar

    ref_out1 = torch.addmm(bias1, mat1, mat2, alpha=alpha, beta=beta)
    with flag_gems.use_gems():
        with torch.no_grad():
            res_out1 = torch.addmm(bias1, mat1, mat2, alpha=alpha, beta=beta)

    gems_assert_close(res_out1, ref_out1, dtype, reduce_dim=K)


@pytest.mark.bmm
@pytest.mark.parametrize("M, N, K", MNK_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_bmm(M, N, K, dtype):
    mat1 = torch.randn((4, M, K), dtype=dtype, device=flag_gems.device)
    mat2 = torch.randn((4, K, N), dtype=dtype, device=flag_gems.device)

    ref_out = torch.bmm(mat1, mat2)
    with flag_gems.use_gems():
        with torch.no_grad():
            res_out = torch.bmm(mat1, mat2)

    gems_assert_close(res_out, ref_out, dtype, reduce_dim=K)


CUMSUM_SHAPES = ([(2, 32)] if QUICK_MODE else REDUCTION_SHAPES + [(2637, ), (16, 1025, 255)])


@pytest.mark.cumsum
@pytest.mark.parametrize("shape", CUMSUM_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES + INT_DTYPES)
def test_accuracy_cumsum(shape, dtype):
    if flag_gems.vendor_name == "kunlunxin":
        torch.manual_seed(0)
        torch.cuda.manual_seed_all(0)

    dim = 1 if shape == REDUCTION_SHAPES[-1] else -1
    if dtype in INT_DTYPES:
        inp = torch.randint(-3, 3, shape, device=flag_gems.device).to(dtype)
        ref_inp = to_reference(inp)
    else:
        inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
        ref_inp = to_reference(inp, True)

    ref_out = torch.cumsum(ref_inp, dim=dim)
    if flag_gems.vendor_name == "kunlunxin":
        from flag_gems.runtime.backend._kunlunxin import ops as kl_ops

        res_out = kl_ops.cumsum(inp, dim=dim)
    else:
        with flag_gems.use_gems():
            res_out = torch.cumsum(inp, dim=dim)

    gems_assert_close(res_out, ref_out, dtype, reduce_dim=shape[dim])


@pytest.mark.dropout
@pytest.mark.native_dropout
@pytest.mark.parametrize("shape", SPECIAL_SHAPES)
@pytest.mark.parametrize("p", [0.3, 0.6, 0.9])
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_dropout(shape, p, dtype):
    if flag_gems.vendor_name == "kunlunxin":
        torch.manual_seed(0)
        torch.cuda.manual_seed_all(0)

    if TO_CPU or shape == (1, ):
        shape = (32768, )
    res_inp = torch.randn(
        shape,
        dtype=dtype,
        device=flag_gems.device,
    )
    ref_inp = to_reference(res_inp)

    # NOTE: ensure that scalars are float32(instead of float64)
    # in some cases, casting up then casting down have different result
    p = np.float32(p)
    one_minus_p = np.float32(1.0) - p

    ref_out = torch.nn.functional.dropout(ref_inp, p, True)
    with flag_gems.use_gems():
        res_out = torch.nn.functional.dropout(res_inp, p, True)

    res_out = to_reference(res_out)
    exp_equal = (p * p + one_minus_p * one_minus_p) * res_inp.numel()
    num_equal = torch.sum(torch.isclose(ref_out, res_out)).item()
    if TO_CPU:
        from flag_gems.testing import RESOLUTION

        zero_equal = torch.eq(res_out, torch.zeros_like(res_out))
        num_zero = torch.sum(zero_equal).item()
        assert abs(num_zero / res_inp.numel() - p) <= 0.05
        scale_equal = torch.isclose(res_out, ref_inp / one_minus_p, rtol=RESOLUTION[dtype])
        assert torch.all(torch.logical_or(zero_equal, scale_equal))
    else:
        assert (abs(num_equal - exp_equal) / exp_equal
                <= 0.05), f"num_equal: {num_equal}, exp_equal: {exp_equal}, num_total: {res_inp.numel()}"


@pytest.mark.gelu
@pytest.mark.parametrize("shape", POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
@pytest.mark.parametrize("approximate", ["none", "tanh"])
def test_accuracy_gelu(shape, dtype, approximate):
    res_inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = to_reference(res_inp, True)

    ref_out = torch.nn.functional.gelu(ref_inp, approximate=approximate)
    with flag_gems.use_gems():
        res_out = torch.nn.functional.gelu(res_inp, approximate=approximate)

    gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.layernorm
@pytest.mark.native_layer_norm
@pytest.mark.parametrize(
    "shape",
    ([(2, 40999)] if QUICK_MODE else [
        (200, 36),
        (4096, 100),
        (1, 40999),
        (100, 40499),
        (4096, 256),
    ]),
)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_layernorm(shape, dtype):
    if flag_gems.vendor_name == "kunlunxin":
        torch.manual_seed(0)
        torch.cuda.manual_seed_all(0)
    if flag_gems.vendor_name == "spacemit":
        torch.manual_seed(42)
    res_inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)

    res_weight = torch.randn(shape[1:], dtype=dtype, device=flag_gems.device)
    res_bias = torch.randn(shape[1:], dtype=dtype, device=flag_gems.device)
    eps = 1e-5

    ref_inp = to_reference(res_inp, True)
    ref_weight = to_reference(res_weight, True)
    ref_bias = to_reference(res_bias, True)

    ref_out = torch.layer_norm(
        ref_inp,
        shape[1:],
        weight=ref_weight,
        bias=ref_bias,
        eps=eps,
    )
    with flag_gems.use_gems():
        res_out = torch.layer_norm(
            res_inp,
            shape[1:],
            weight=res_weight,
            bias=res_bias,
            eps=eps,
        )

    gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.relu
@pytest.mark.parametrize("shape", POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_relu(shape, dtype):
    res_inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = to_reference(res_inp, True)

    ref_out = torch.relu(ref_inp)
    with flag_gems.use_gems():
        res_out = torch.nn.functional.relu(res_inp)

    gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.silu
@pytest.mark.parametrize("shape", POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_silu(shape, dtype):
    res_inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = to_reference(res_inp, True)

    ref_out = torch.nn.functional.silu(ref_inp)
    with flag_gems.use_gems():
        res_out = torch.nn.functional.silu(res_inp)

    gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.softmax
@pytest.mark.parametrize("shape", [(1, 256)] if QUICK_MODE else [(1, 256)])
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
@pytest.mark.parametrize("dim", DIM_LIST)
@pytest.mark.parametrize("neg_inf", [True])
def test_accuracy_softmax(shape, dtype, dim, neg_inf):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    if neg_inf:
        inp = torch.where(inp < 0.0, float("-inf"), inp)
    ref_inp = to_reference(inp, True)

    ref_out = torch.nn.functional.softmax(ref_inp, dim=dim)
    with flag_gems.use_gems():
        res_out = torch.nn.functional.softmax(inp, dim=dim)
    gems_assert_close(res_out, ref_out, dtype, equal_nan=True)


@pytest.mark.triu
@pytest.mark.parametrize("shape, diagonal", SHAPE_DIAGONAL)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_triu(shape, diagonal, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    inp = unsqueeze_tensor(inp, 2)
    ref_inp = to_reference(inp)

    ref_out = torch.triu(ref_inp, diagonal)
    with flag_gems.use_gems():
        res_out = torch.triu(inp, diagonal)

    gems_assert_equal(res_out, ref_out)


@pytest.mark.abs
@pytest.mark.parametrize("shape", POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_abs(shape, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = to_reference(inp)

    ref_out = torch.abs(ref_inp)
    with flag_gems.use_gems():
        res_out = torch.abs(inp)

    gems_assert_equal(res_out, ref_out)


@pytest.mark.add
@pytest.mark.parametrize("shape", POINTWISE_SHAPES)
@pytest.mark.parametrize("alpha", SCALARS)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_add(shape, alpha, dtype):
    inp1 = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    inp2 = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp1 = to_reference(inp1, True)
    ref_inp2 = to_reference(inp2, True)

    ref_out = torch.add(ref_inp1, ref_inp2, alpha=alpha)
    with flag_gems.use_gems():
        res_out = torch.add(inp1, inp2, alpha=alpha)

    gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.div
@pytest.mark.parametrize("shape", POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_div_tensor_tensor(shape, dtype):
    inp1 = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    inp2 = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp1 = to_reference(inp1, False)
    ref_inp2 = to_reference(inp2, False)

    ref_out = torch.div(ref_inp1, ref_inp2)
    with flag_gems.use_gems():
        res_out = torch.div(inp1, inp2)

    gems_assert_close(res_out, ref_out, dtype, equal_nan=True)


@pytest.mark.exp
@pytest.mark.parametrize("shape", POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_exp(shape, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = to_reference(inp, True)

    ref_out = torch.exp(ref_inp)
    with flag_gems.use_gems():
        res_out = torch.exp(inp)

    gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.mul
@pytest.mark.parametrize("shape", POINTWISE_SHAPES)
@pytest.mark.parametrize("scalar", SCALARS)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_mul_scalar_tensor(shape, scalar, dtype):
    inp1 = scalar
    inp2 = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp2 = to_reference(inp2, True)

    ref_out = torch.mul(inp1, ref_inp2)
    with flag_gems.use_gems():
        res_out = torch.mul(inp1, inp2)

    gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.pow
@pytest.mark.parametrize("shape", POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_pow(shape, dtype):
    inp1 = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    inp2 = torch.randn(shape, dtype=dtype, device=flag_gems.device)

    if flag_gems.vendor_name == "kunlunxin":
        inp1 = inp1.uniform_(-0.1, 0.1)
        inp2 = inp2.uniform_(-0.1, 0.1)

    ref_inp1 = to_reference(inp1, True)
    ref_inp2 = to_reference(inp2, True)

    ref_out = torch.pow(ref_inp1, ref_inp2)
    with flag_gems.use_gems():
        res_out = torch.pow(inp1, inp2)

    gems_assert_close(res_out, ref_out, dtype, equal_nan=True)


@pytest.mark.reciprocal
@pytest.mark.parametrize("shape", POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_reciprocal(shape, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = to_reference(inp, True)

    ref_out = torch.reciprocal(ref_inp)
    with flag_gems.use_gems():
        res_out = torch.reciprocal(inp)

    gems_assert_close(res_out, ref_out, dtype, equal_nan=True)


@pytest.mark.rsqrt
@pytest.mark.parametrize("shape", POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_rsqrt(shape, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = to_reference(inp, True)

    ref_out = torch.rsqrt(ref_inp)
    with flag_gems.use_gems():
        res_out = torch.rsqrt(inp)

    gems_assert_close(res_out, ref_out, dtype, equal_nan=True)


@pytest.mark.rsub
@pytest.mark.parametrize("shape", POINTWISE_SHAPES)
@pytest.mark.parametrize("alpha", SCALARS)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_rsub(shape, alpha, dtype):
    inp1 = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    inp2 = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp1 = to_reference(inp1, True)
    ref_inp2 = to_reference(inp2, True)

    ref_out = torch.rsub(ref_inp1, ref_inp2, alpha=alpha)
    with flag_gems.use_gems():
        if flag_gems.vendor_name == "kunlunxin":
            from flag_gems.runtime.backend._kunlunxin import ops as kl_ops

            res_out = kl_ops.rsub(inp1, inp2, alpha=alpha)
        else:
            res_out = torch.rsub(inp1, inp2, alpha=alpha)

    gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.sub
@pytest.mark.parametrize("shape", POINTWISE_SHAPES)
@pytest.mark.parametrize("alpha", SCALARS)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_sub(shape, alpha, dtype):
    inp1 = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    inp2 = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp1 = to_reference(inp1, True)
    ref_inp2 = to_reference(inp2, True)

    ref_out = torch.sub(ref_inp1, ref_inp2, alpha=alpha)
    with flag_gems.use_gems():
        res_out = torch.sub(inp1, inp2, alpha=alpha)

    gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.mean
@pytest.mark.parametrize("shape", REDUCTION_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_mean_without_dim(shape, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = to_reference(inp, True)

    ref_out = torch.mean(ref_inp)
    with flag_gems.use_gems():
        res_out = torch.mean(inp)

    gems_assert_close(res_out, ref_out, dtype)
