import pytest
import torch

import triton
from triton.backends.spine_triton.driver import CPUDriver

triton.runtime.driver.set_active(CPUDriver())
import flag_gems
from .conftest import QUICK_MODE
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
    ALL_INT_DTYPES,
    POINTWISE_SHAPES,
    REDUCTION_SHAPES,
    gems_assert_equal,
    CONTIGUOUS_SHAPE_STRIDES_2D,
    IRREGULAR_SHAPE_STRIDES,
    SHAPE_STRIDES,
    BOOL_TYPES,
    REDUCTION_SMALL_SHAPES,
    SkipVersion,
)

MN_SHAPES = [(1, 32), (160, 1024), (5333, 497)]
MNK_SHAPES = ([(1, 1, 32), (15, 160, 1024), (495, 5333, 71)])
SHAPE_DIAGONAL = list(zip(POINTWISE_SHAPES, [-2, -2, -1, 0, 1, 3]))


def replace_zeros(inp):
    return torch.where(inp == 0, 1, inp)


DIMS_LIST = [1] if QUICK_MODE else [0, 1, [0, 1], [1, 0]]
KEEPDIM_DIMS = ([(True, DIMS_LIST[0])] if QUICK_MODE else list(zip([True, False] * 2, DIMS_LIST)))
DIM_LIST = [1] if QUICK_MODE else [0, 1]
KEEPDIM_DIM = ([(True, DIM_LIST[0])] if QUICK_MODE else list(zip([True, False], DIM_LIST)))
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

KIND_KEEPDIM_DIMS_SHAPE = ([("normal", True, DIMS_LIST[0], REDUCTION_SHAPES[0])] if QUICK_MODE else list(
    zip(
        ["normal", "allTrue"] * 2,
        [True, False] * 2,
        DIMS_LIST,
        REDUCTION_SHAPES + [(7, 4, 11, 1)],
    )))


@pytest.mark.skipif(flag_gems.vendor_name == "spacemit", reason="TODO")
@pytest.mark.all
@pytest.mark.skipif(SkipVersion("torch", "<2.2"), reason="Skipping Pytorch version.")
@pytest.mark.parametrize("kind, keepdim, dim, shape", KIND_KEEPDIM_DIMS_SHAPE)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_all_dims(shape, dim, keepdim, dtype, kind):
    if kind == "allTrue":
        inp = torch.ones(shape, dtype=dtype, device=flag_gems.device)
    else:
        inp = torch.randint(0, 2, shape, dtype=dtype, device="cpu").to(flag_gems.device)
    ref_inp = to_reference(inp)

    ref_out = torch.all(ref_inp, dim=dim, keepdim=keepdim)
    with flag_gems.use_gems():
        res_out = torch.all(inp, dim=dim, keepdim=keepdim)

    gems_assert_equal(res_out, ref_out)


@pytest.mark.any
@pytest.mark.parametrize("shape", REDUCTION_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES + [torch.bool])
@pytest.mark.parametrize("kind", ["normal", "allFalse"])
def test_accuracy_any_without_dim(shape, dtype, kind):
    if kind == "allFalse":
        inp = torch.zeros(shape, dtype=dtype, device=flag_gems.device)
    else:
        inp = torch.randint(0, 2, shape, dtype=dtype, device="cpu").to(flag_gems.device)
    ref_inp = to_reference(inp)

    ref_out = torch.any(ref_inp)
    with flag_gems.use_gems():
        res_out = torch.any(inp)

    gems_assert_equal(res_out, ref_out)


@pytest.mark.bitwise_and
@pytest.mark.parametrize("shape", POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", INT_DTYPES + BOOL_TYPES)
def test_accuracy_bitwiseand(shape, dtype):
    if dtype in BOOL_TYPES:
        inp1 = torch.randint(0, 2, size=shape, dtype=dtype, device="cpu").to(flag_gems.device)
        inp2 = torch.randint(0, 2, size=shape, dtype=dtype, device="cpu").to(flag_gems.device)
    else:
        inp1 = torch.randint(low=-0x7FFF, high=0x7FFF, size=shape, dtype=dtype, device="cpu").to(flag_gems.device)
        inp2 = torch.randint(low=-0x7FFF, high=0x7FFF, size=shape, dtype=dtype, device="cpu").to(flag_gems.device)
    ref_inp1 = to_reference(inp1)
    ref_inp2 = to_reference(inp2)

    ref_out = torch.bitwise_and(ref_inp1, ref_inp2)
    with flag_gems.use_gems():
        res_out = torch.bitwise_and(inp1, inp2)

    gems_assert_equal(res_out, ref_out)


@pytest.mark.bitwise_not
@pytest.mark.parametrize("shape", POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", INT_DTYPES + BOOL_TYPES)
def test_accuracy_bitwisenot(shape, dtype):
    if dtype in BOOL_TYPES:
        inp = torch.randint(0, 2, size=shape, dtype=dtype, device="cpu").to(flag_gems.device)
    else:
        inp = torch.randint(low=-0x7FFF, high=0x7FFF, size=shape, dtype=dtype, device="cpu").to(flag_gems.device)
    ref_inp = to_reference(inp)

    ref_out = torch.bitwise_not(ref_inp)
    with flag_gems.use_gems():
        res_out = torch.bitwise_not(inp)

    gems_assert_equal(res_out, ref_out)


@pytest.mark.or_
@pytest.mark.bitwise_or
@pytest.mark.parametrize("shape", POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", INT_DTYPES + BOOL_TYPES)
def test_accuracy_bitwiseor(shape, dtype):
    if dtype in BOOL_TYPES:
        inp1 = torch.randint(0, 2, size=shape, dtype=dtype, device="cpu").to(flag_gems.device)
        inp2 = torch.randint(0, 2, size=shape, dtype=dtype, device="cpu").to(flag_gems.device)
    else:
        inp1 = torch.randint(low=-0x7FFF, high=0x7FFF, size=shape, dtype=dtype, device="cpu").to(flag_gems.device)
        inp2 = torch.randint(low=-0x7FFF, high=0x7FFF, size=shape, dtype=dtype, device="cpu").to(flag_gems.device)
    ref_inp1 = to_reference(inp1)
    ref_inp2 = to_reference(inp2)

    ref_out = torch.bitwise_or(ref_inp1, ref_inp2)
    with flag_gems.use_gems():
        res_out = torch.bitwise_or(inp1, inp2)

    gems_assert_equal(res_out, ref_out)


@pytest.mark.or_
@pytest.mark.bitwise_or
@pytest.mark.parametrize("shape", POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", INT_DTYPES + BOOL_TYPES)
def test_accuracy_bitwiseor_scalar(shape, dtype):
    if dtype in BOOL_TYPES:
        inp1 = torch.randint(0, 2, size=shape, dtype=dtype, device="cpu").to(flag_gems.device)
        inp2 = bool(random.randint(0, 2))
    else:
        inp1 = torch.randint(low=-0x7FFF, high=0x7FFF, size=shape, dtype=dtype, device="cpu").to(flag_gems.device)
        inp2 = 0x00FF
    ref_inp1 = to_reference(inp1)

    ref_out = torch.bitwise_or(ref_inp1, inp2)
    with flag_gems.use_gems():
        res_out = torch.bitwise_or(inp1, inp2)

    gems_assert_equal(res_out, ref_out)


@pytest.mark.or_
@pytest.mark.bitwise_or
@pytest.mark.parametrize("shape", POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", INT_DTYPES + BOOL_TYPES)
def test_accuracy_bitwiseor_scalar_tensor(shape, dtype):
    if dtype in BOOL_TYPES:
        inp1 = bool(random.randint(0, 2))
        inp2 = torch.randint(0, 2, size=shape, dtype=torch.bool, device="cpu").to(flag_gems.device)
    else:
        inp1 = 0x00FF
        inp2 = torch.randint(low=-0x7FFF, high=0x7FFF, size=shape, dtype=dtype, device="cpu").to(flag_gems.device)
    ref_inp2 = to_reference(inp2)

    ref_out = torch.bitwise_or(inp1, ref_inp2)
    with flag_gems.use_gems():
        res_out = torch.bitwise_or(inp1, inp2)

    gems_assert_equal(res_out, ref_out)


@pytest.mark.cos
@pytest.mark.parametrize("shape", POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_cos(shape, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = to_reference(inp, True)

    ref_out = torch.cos(ref_inp)
    with flag_gems.use_gems():
        res_out = torch.cos(inp)

    gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.clamp
@pytest.mark.parametrize("shape", POINTWISE_SHAPES)
@pytest.mark.parametrize("maxi", SCALARS)
@pytest.mark.parametrize("mini", SCALARS)
@pytest.mark.parametrize("isnone", [None, "max", "min"])
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_clamp(shape, maxi, mini, isnone, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    if isnone == "min":
        mini = None
    elif isnone == "max":
        maxi = None
    ref_inp = to_reference(inp)

    ref_out = torch.clamp(ref_inp, min=mini, max=maxi)
    with flag_gems.use_gems():
        res_out = torch.clamp(inp, min=mini, max=maxi)

    gems_assert_equal(res_out, ref_out)


@pytest.mark.eq
@pytest.mark.parametrize("shape", POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_eq(shape, dtype):
    inp1 = torch.randint(0, 10, shape, dtype=dtype, device=flag_gems.device)
    inp2 = torch.randint(0, 10, shape, dtype=dtype, device=flag_gems.device)
    ref_inp1 = to_reference(inp1)
    ref_inp2 = to_reference(inp2)

    ref_out = torch.eq(ref_inp1, ref_inp2)
    with flag_gems.use_gems():
        res_out = torch.eq(inp1, inp2)

    gems_assert_equal(res_out, ref_out)


@pytest.mark.ge
@pytest.mark.parametrize("shape", POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_ge(shape, dtype):
    inp1 = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    inp2 = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp1 = to_reference(inp1)
    ref_inp2 = to_reference(inp2)

    ref_out = torch.ge(ref_inp1, ref_inp2)
    with flag_gems.use_gems():
        res_out = torch.ge(inp1, inp2)

    gems_assert_equal(res_out, ref_out)


@pytest.mark.ge
@pytest.mark.parametrize("shape", POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_ge_scalar(shape, dtype):
    inp1 = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    inp2 = 0
    ref_inp1 = to_reference(inp1)

    ref_out = torch.ge(ref_inp1, inp2)
    with flag_gems.use_gems():
        res_out = torch.ge(inp1, inp2)

    gems_assert_equal(res_out, ref_out)


@pytest.mark.gt
@pytest.mark.parametrize("shape", POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_gt(shape, dtype):
    inp1 = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    inp2 = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp1 = to_reference(inp1)
    ref_inp2 = to_reference(inp2)

    ref_out = torch.gt(ref_inp1, ref_inp2)
    with flag_gems.use_gems():
        res_out = torch.gt(inp1, inp2)

    gems_assert_equal(res_out, ref_out)


@pytest.mark.gt
@pytest.mark.parametrize("shape", POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_gt_scalar(shape, dtype):
    inp1 = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp1 = to_reference(inp1)
    inp2 = 0

    ref_out = torch.gt(ref_inp1, inp2)
    with flag_gems.use_gems():
        res_out = torch.gt(inp1, inp2)

    gems_assert_equal(res_out, ref_out)


@pytest.mark.mv
@pytest.mark.parametrize("M, N", MN_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_mv(M, N, dtype):
    matrix = torch.randn((N, M), dtype=dtype, device=flag_gems.device)
    vector = torch.randn((M, ), dtype=dtype, device=flag_gems.device)

    ref_out = torch.mv(matrix, vector)
    with flag_gems.use_gems():
        res_out = torch.mv(matrix, vector)

    torch.testing.assert_close(ref_out, res_out, atol=1e-2, rtol=1e-2)


@pytest.mark.skipif(flag_gems.vendor_name == "spacemit", reason="TODO")
@pytest.mark.isinf
@pytest.mark.parametrize("shape", POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_isinf(shape, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    inp = torch.masked_fill(inp, inp > 1.0, -float("inf"))
    ref_inp = to_reference(inp)

    ref_out = torch.isinf(ref_inp)
    with flag_gems.use_gems():
        res_out = torch.isinf(inp)

    gems_assert_equal(res_out, ref_out)


@pytest.mark.isnan
@pytest.mark.parametrize("shape", POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_isnan(shape, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    inp = torch.masked_fill(inp, inp > 1.0, float("nan"))
    ref_inp = to_reference(inp)

    ref_out = torch.isnan(ref_inp)
    with flag_gems.use_gems():
        res_out = torch.isnan(inp)

    gems_assert_equal(res_out, ref_out)


@pytest.mark.le
@pytest.mark.parametrize("shape", POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_le(shape, dtype):
    inp1 = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    inp2 = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp1 = to_reference(inp1)
    ref_inp2 = to_reference(inp2)

    ref_out = torch.le(ref_inp1, ref_inp2)
    with flag_gems.use_gems():
        res_out = torch.le(inp1, inp2)

    gems_assert_equal(res_out, ref_out)


@pytest.mark.le
@pytest.mark.parametrize("shape", POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_le_scalar(shape, dtype):
    inp1 = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    inp2 = 0
    ref_inp1 = to_reference(inp1)

    ref_out = torch.le(ref_inp1, inp2)
    with flag_gems.use_gems():
        res_out = torch.le(inp1, inp2)

    gems_assert_equal(res_out, ref_out)


@pytest.mark.lt
@pytest.mark.parametrize("shape", POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_lt(shape, dtype):
    inp1 = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    inp2 = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp1 = to_reference(inp1)
    ref_inp2 = to_reference(inp2)

    ref_out = torch.lt(ref_inp1, ref_inp2)
    with flag_gems.use_gems():
        res_out = torch.lt(inp1, inp2)

    gems_assert_equal(res_out, ref_out)


@pytest.mark.lt
@pytest.mark.parametrize("shape", POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_lt_scalar(shape, dtype):
    inp1 = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    inp2 = 0
    ref_inp1 = to_reference(inp1)

    ref_out = torch.lt(ref_inp1, inp2)
    with flag_gems.use_gems():
        res_out = torch.lt(inp1, inp2)

    gems_assert_equal(res_out, ref_out)


@pytest.mark.ne
@pytest.mark.parametrize("shape", POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_ne(shape, dtype):
    inp1 = torch.randint(0, 10, shape, dtype=dtype, device=flag_gems.device)
    inp2 = torch.randint(0, 10, shape, dtype=dtype, device=flag_gems.device)
    ref_inp1 = to_reference(inp1)
    ref_inp2 = to_reference(inp2)

    ref_out = torch.ne(ref_inp1, ref_inp2)
    with flag_gems.use_gems():
        res_out = torch.ne(inp1, inp2)

    gems_assert_equal(res_out, ref_out)


@pytest.mark.ne
@pytest.mark.parametrize("shape", POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_ne_scalar(shape, dtype):
    inp1 = torch.randint(0, 10, shape, dtype=dtype, device=flag_gems.device)
    inp2 = 0
    ref_inp1 = to_reference(inp1)

    ref_out = torch.ne(ref_inp1, inp2)
    with flag_gems.use_gems():
        res_out = torch.ne(inp1, inp2)

    gems_assert_equal(res_out, ref_out)


@pytest.mark.neg
@pytest.mark.parametrize("shape", POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_neg(shape, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = to_reference(inp)

    ref_out = torch.neg(ref_inp)
    with flag_gems.use_gems():
        res_out = torch.neg(inp)

    gems_assert_equal(res_out, ref_out)


@pytest.mark.sin
@pytest.mark.parametrize("shape", POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_sin(shape, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = to_reference(inp, True)

    ref_out = torch.sin(ref_inp)
    with flag_gems.use_gems():
        res_out = torch.sin(inp)

    gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.tanh
@pytest.mark.parametrize("shape", POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_tanh(shape, dtype):
    res_inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = to_reference(res_inp, True)

    ref_out = torch.tanh(ref_inp)
    with flag_gems.use_gems():
        res_out = torch.tanh(res_inp)

    gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.amax
@pytest.mark.parametrize("keepdim, dim, shape", KEEPDIM_DIMS_SHAPE)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_amax(shape, dim, keepdim, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = to_reference(inp)

    ref_out = torch.amax(ref_inp, dim=dim, keepdim=keepdim)
    with flag_gems.use_gems():
        res_out = torch.amax(inp, dim=dim, keepdim=keepdim)

    gems_assert_equal(res_out, ref_out)


@pytest.mark.argmax
@pytest.mark.parametrize("shape", REDUCTION_SMALL_SHAPES)
@pytest.mark.parametrize("dim", DIM_LIST)
@pytest.mark.parametrize("keepdim", [True, False])
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_argmax(shape, dim, keepdim, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = to_reference(inp)

    ref_out = torch.argmax(ref_inp, dim=dim, keepdim=keepdim)
    with flag_gems.use_gems():
        res_out = torch.argmax(inp, dim=dim, keepdim=keepdim)

    gems_assert_equal(res_out, ref_out)


@pytest.mark.max
@pytest.mark.parametrize("shape", REDUCTION_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES + ALL_INT_DTYPES)
def test_accuracy_max_without_dim(shape, dtype):
    if dtype in FLOAT_DTYPES:
        inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    else:
        inp = torch.randint(-10000, 10000, shape, dtype=dtype, device=flag_gems.device)
    ref_inp = to_reference(inp)

    ref_out = torch.max(ref_inp)
    with flag_gems.use_gems():
        res_out = torch.max(inp)

    gems_assert_equal(res_out, ref_out)


@pytest.mark.max
@pytest.mark.parametrize("shape", REDUCTION_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_max_without_dim_all_neg_inf(shape, dtype):
    inp = torch.full(shape, fill_value=float("-inf"), dtype=dtype, device=flag_gems.device)
    ref_inp = to_reference(inp)

    ref_out = torch.max(ref_inp)
    with flag_gems.use_gems():
        res_out = torch.max(inp)

    gems_assert_equal(res_out, ref_out)


@pytest.mark.max
@pytest.mark.parametrize("shape", REDUCTION_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES + ALL_INT_DTYPES)
def test_accuracy_max_without_dim_uncontiguous(shape, dtype):
    if dtype in FLOAT_DTYPES:
        inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)[::2, ::2]
    else:
        inp = torch.randint(-10000, 10000, shape, dtype=dtype, device=flag_gems.device)[::2, ::2]
    ref_inp = to_reference(inp)

    ref_out = torch.max(ref_inp)
    with flag_gems.use_gems():
        res_out = torch.max(inp)

    gems_assert_equal(res_out, ref_out)


@pytest.mark.max
@pytest.mark.parametrize("shape", REDUCTION_SMALL_SHAPES)
@pytest.mark.parametrize("keepdim, dim", KEEPDIM_DIM)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES + ALL_INT_DTYPES)
def test_accuracy_max_dim(shape, dim, keepdim, dtype):
    if dtype in FLOAT_DTYPES:
        inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    else:
        inp = torch.randint(-10000, 10000, shape, dtype=dtype, device=flag_gems.device)
    ref_inp = to_reference(inp)

    ref_out_value, ref_out_index = torch.max(ref_inp, dim=dim, keepdim=keepdim)
    with flag_gems.use_gems():
        res_out_value, res_out_index = torch.max(inp, dim=dim, keepdim=keepdim)

    gems_assert_equal(res_out_index, ref_out_index)
    gems_assert_equal(res_out_value, ref_out_value)


@pytest.mark.max
@pytest.mark.parametrize("shape", [(4, 1048577, 4)])
@pytest.mark.parametrize("keepdim, dim", [(True, 1), (False, 1)])
@pytest.mark.parametrize("dtype", FLOAT_DTYPES + ALL_INT_DTYPES)
def test_accuracy_max_dim_big_shape(shape, dim, keepdim, dtype):
    if dtype in FLOAT_DTYPES:
        inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    else:
        inp = torch.randint(-10000, 10000, shape, dtype=dtype, device=flag_gems.device)
    ref_inp = to_reference(inp)

    ref_out_value, ref_out_index = torch.max(ref_inp, dim=dim, keepdim=keepdim)
    with flag_gems.use_gems():
        res_out_value, res_out_index = torch.max(inp, dim=dim, keepdim=keepdim)

    gems_assert_equal(res_out_index, ref_out_index)
    gems_assert_equal(res_out_value, ref_out_value)


@pytest.mark.min
@pytest.mark.parametrize("shape", REDUCTION_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES + ALL_INT_DTYPES)
def test_accuracy_min_without_dim(shape, dtype):
    if dtype in FLOAT_DTYPES:
        inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    else:
        inp = torch.randint(-10000, 10000, shape, dtype=dtype, device=flag_gems.device)
    ref_inp = to_reference(inp)

    ref_out = torch.min(ref_inp)
    with flag_gems.use_gems():
        res_out = torch.min(inp)

    gems_assert_equal(res_out, ref_out)


@pytest.mark.min
@pytest.mark.parametrize("shape", REDUCTION_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_min_without_dim_all_inf(shape, dtype):
    # ensure that padding value used in min is inf, not max value
    inp = torch.full(shape, fill_value=float("inf"), dtype=dtype, device=flag_gems.device)
    ref_inp = to_reference(inp)

    ref_out = torch.min(ref_inp)
    with flag_gems.use_gems():
        res_out = torch.min(inp)

    gems_assert_equal(res_out, ref_out)


@pytest.mark.min
@pytest.mark.parametrize("shape", REDUCTION_SMALL_SHAPES)
@pytest.mark.parametrize("keepdim, dim", KEEPDIM_DIM)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES + ALL_INT_DTYPES)
def test_accuracy_min_dim(shape, dim, keepdim, dtype):
    if dtype in FLOAT_DTYPES:
        inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    else:
        inp = torch.randint(-10000, 10000, shape, dtype=dtype, device=flag_gems.device)
    ref_inp = to_reference(inp)

    ref_out_value, ref_out_index = torch.min(ref_inp, dim=dim, keepdim=keepdim)
    with flag_gems.use_gems():
        res_out_value, res_out_index = torch.min(inp, dim=dim, keepdim=keepdim)

    gems_assert_equal(res_out_index, ref_out_index)
    gems_assert_equal(res_out_value, ref_out_value)


@pytest.mark.outer
@pytest.mark.parametrize("M, N", MN_SHAPES + ([(32, 131072)] if flag_gems.vendor_name == "cambricon" else []))
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_outer(M, N, dtype):
    inp1 = torch.randn(M, dtype=dtype, device=flag_gems.device, requires_grad=True)
    inp2 = torch.randn(N, dtype=dtype, device=flag_gems.device, requires_grad=True)
    ref_inp1 = to_reference(inp1, True)
    ref_inp2 = to_reference(inp2, True)

    ref_out = torch.outer(ref_inp1, ref_inp2)
    res_out = flag_gems.outer(inp1, inp2)
    gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.prod
@pytest.mark.parametrize("shape", REDUCTION_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_prod_without_dim(shape, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = to_reference(inp, True)

    ref_out = torch.prod(ref_inp)
    with flag_gems.use_gems():
        res_out = torch.prod(inp)

    gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.sum
@pytest.mark.parametrize("shape", REDUCTION_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_sum_without_dim(shape, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = to_reference(inp, True)

    ref_out = torch.sum(ref_inp)
    with flag_gems.use_gems():
        res_out = torch.sum(inp)

    gems_assert_close(res_out, ref_out, dtype, reduce_dim=inp.numel())


@pytest.mark.sum
@pytest.mark.parametrize("shape", REDUCTION_SHAPES)
@pytest.mark.parametrize("keepdim, dim", KEEPDIM_DIM + [(False, []), (True, [])])
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_sum_dim(shape, dim, keepdim, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = to_reference(inp, True)

    ref_out = torch.sum(ref_inp, dim=dim, keepdim=keepdim)
    with flag_gems.use_gems():
        res_out = torch.sum(inp, dim=dim, keepdim=keepdim)

    if isinstance(dim, int):
        dim = [dim]
    dim = [d % inp.ndim for d in dim]
    _dim = 1
    for d in dim:
        _dim *= shape[d]
    if dim == []:
        _dim = inp.numel()
    gems_assert_close(res_out, ref_out, dtype, reduce_dim=_dim)


@pytest.mark.var_mean
@pytest.mark.parametrize("shape", REDUCTION_SHAPES)
@pytest.mark.parametrize("dim", DIMS_LIST)
@pytest.mark.parametrize("correction", [1] if QUICK_MODE else [0, 1])
@pytest.mark.parametrize("keepdim", [True] if QUICK_MODE else [True, False])
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_varmean(shape, dim, correction, keepdim, dtype):
    if shape[0] == 1:  # TODO: res is inf, while ref is nan
        shape = (2, 2)
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = to_reference(inp, True)

    ref_var, ref_mean = torch.var_mean(ref_inp, dim, correction=correction, keepdim=keepdim)
    with flag_gems.use_gems():
        res_var, res_mean = torch.var_mean(inp, dim, correction=correction, keepdim=keepdim)

    gems_assert_close(res_mean, ref_mean, dtype)
    gems_assert_close(res_var, ref_var, dtype)


@pytest.mark.vector_norm
@pytest.mark.parametrize("shape", REDUCTION_SHAPES)
@pytest.mark.parametrize("ord", [2])
@pytest.mark.parametrize("keepdim, dim", KEEPDIM_DIMS)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_vectornorm(shape, ord, dim, keepdim, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = to_reference(inp, True)

    ref_out = torch.linalg.vector_norm(ref_inp, ord, dim, keepdim)
    with flag_gems.use_gems():
        res_out = torch.linalg.vector_norm(inp, ord, dim, keepdim)

    gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.CrossEntropyLoss
@pytest.mark.parametrize("label_smoothing, ignore_index, shape", SMOOTH_IGNORE_SHAPE)
@pytest.mark.parametrize("reduction", CROSS_ENTROPY_LOSS_REDUCTION)
@pytest.mark.parametrize("weight", [True, False])
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_cross_entropy_loss_indices(shape, dtype, weight, ignore_index, reduction, label_smoothing):
    dim = 1
    up_limit = shape[dim] - 1
    target_shape = list(shape)
    del target_shape[dim]

    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device, requires_grad=True)
    target = torch.randint(0, up_limit, target_shape, device=flag_gems.device)
    ref_inp = to_reference(inp, True)
    ref_target = to_reference(target)

    if weight:
        wgt = torch.randn(shape[dim], dtype=dtype, device=flag_gems.device)
        ref_wgt = to_reference(wgt, True)
    else:
        wgt = None
        ref_wgt = None
    ref_out = torch.nn.functional.cross_entropy(
        ref_inp,
        ref_target,
        weight=ref_wgt,
        ignore_index=ignore_index,
        reduction=reduction,
        label_smoothing=label_smoothing,
    )
    res_out = flag_gems.cross_entropy_loss(
        inp,
        target,
        weight=wgt,
        ignore_index=ignore_index,
        reduction=reduction,
        label_smoothing=label_smoothing,
    )
    gems_assert_close(res_out, ref_out, dtype, reduce_dim=shape[dim])


@pytest.mark.group_norm
@pytest.mark.native_group_norm
@pytest.mark.parametrize(
    "N, C, H, W, num_groups",
    [
        (16, 3, 16, 16, 1),
        (32, 32, 32, 32, 8),
        (1, 32, 32, 32, 8),
        (1, 32, 32, 32, 16),
        (1, 64, 32, 32, 16),
        (1, 64, 32, 32, 32),
        (1, 64, 32, 32, 64),
    ],
)
@pytest.mark.parametrize("wb_none", [False, True])
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_groupnorm(N, C, H, W, num_groups, dtype, wb_none):
    if flag_gems.vendor_name == "kunlunxin":
        torch.manual_seed(0)
        torch.cuda.manual_seed_all(0)

    res_inp = torch.randn(size=(N, C, H, W), dtype=dtype, device=flag_gems.device)
    if wb_none:
        res_weight = None
        res_bias = None
    else:
        res_weight = torch.randn(size=(C, ), dtype=dtype, device=flag_gems.device)
        res_bias = torch.randn(size=(C, ), dtype=dtype, device=flag_gems.device)
    eps = 1e-5

    ref_inp = to_reference(res_inp, True)
    ref_weight = to_reference(res_weight, True)
    ref_bias = to_reference(res_bias, True)

    ref_out = torch.nn.functional.group_norm(ref_inp, num_groups, weight=ref_weight, bias=ref_bias, eps=eps)

    with flag_gems.use_gems():
        res_out = torch.group_norm(res_inp, num_groups, weight=res_weight, bias=res_bias, eps=eps)

    gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.log_softmax
@pytest.mark.parametrize("shape", REDUCTION_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
@pytest.mark.parametrize("dim", [1])
def test_accuracy_log_softmax(shape, dtype, dim):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = to_reference(inp, True)

    ref_out = torch.nn.functional.log_softmax(ref_inp, dim=dim)
    with flag_gems.use_gems():
        res_out = torch.nn.functional.log_softmax(inp, dim=dim)
    gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.sigmoid
@pytest.mark.parametrize("shape", POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_sigmoid(shape, dtype):
    res_inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = to_reference(res_inp, True)

    ref_out = torch.sigmoid(ref_inp)
    with flag_gems.use_gems():
        res_out = torch.sigmoid(res_inp)

    gems_assert_close(res_out, ref_out, dtype)
