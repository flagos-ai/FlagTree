import pytest
import torch

import triton
from triton.backends.spine_triton.driver import CPUDriver

triton.runtime.driver.set_active(CPUDriver())
import flag_gems
from .conftest import QUICK_MODE, TO_CPU
import random
import time
import numpy as np
# Make sure every thread has same seed.
random.seed(time.time() // 100)

from .accuracy_utils import (
    FLOAT_DTYPES,
    SCALARS,
    init_seed,
    UT_SHAPES_1D,
    unsqueeze_tuple,
    gems_assert_close,
    to_reference,
    INT_DTYPES,
    STACK_DIM_LIST,
    SPECIAL_SHAPES,
    ALL_INT_DTYPES,
    POINTWISE_SHAPES,
    REDUCTION_SHAPES,
    gems_assert_equal,
    STACK_SHAPES,
    CONTIGUOUS_SHAPE_STRIDES_2D,
    IRREGULAR_SHAPE_STRIDES,
    SHAPE_STRIDES,
    BOOL_TYPES,
    DISTRIBUTION_SHAPES,
    UPSAMPLE_SHAPES,
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

device = flag_gems.device

SHAPE_DEPTHWISE = [
    ((32, 4, 8, 8), (32, 1, 2, 2), (2, 2)),
    ((18, 16, 4, 4), (16, 1, 2, 2), (2, 2)),
    ((9, 32, 4, 4), (128, 1, 2, 2), (2, 2)),
    ((32, 16, 8, 8), (32, 1, 4, 4), (4, 4)),
    ((18, 8, 4, 4), (16, 1, 2, 2), (2, 2)),
    ((9, 4, 4, 4), (128, 1, 2, 2), (2, 2)),
    ((32, 4, 8, 8), (32, 1, 3, 3), (3, 3)),
    ((18, 16, 13, 13), (16, 1, 5, 5), (5, 5)),
    ((9, 32, 8, 8), (128, 1, 3, 3), (3, 3)),
    ((32, 16, 9, 9), (32, 1, 5, 5), (5, 5)),
    ((18, 8, 7, 7), (16, 1, 3, 3), (3, 3)),
    ((9, 4, 6, 6), (128, 1, 3, 3), (3, 3)),
]


@pytest.mark.skipif(flag_gems.vendor_name == "spacemit", reason="TODO")
@pytest.mark.conv_depthwise2d
@pytest.mark.parametrize("shape_input, shape_weight,kernel ", SHAPE_DEPTHWISE)
@pytest.mark.parametrize("stride", [2])
@pytest.mark.parametrize("padding", [2])
@pytest.mark.parametrize("dtype", [torch.float32])
def test_accuracy_depthwise2d(shape_input, shape_weight, kernel, stride, padding, dtype):
    inp = torch.randn(shape_input, dtype=dtype, device=flag_gems.device, requires_grad=True)
    ref_inp = to_reference(inp, False)
    torch.backends.cudnn.allow_tf32 = False
    weight = torch.randn(shape_weight, dtype=dtype, device=flag_gems.device)
    ref_weight = to_reference(weight, False)
    ref_out = torch._C._nn._conv_depthwise2d(
        ref_inp,
        ref_weight,
        kernel,
        bias=None,
        stride=stride,
        padding=padding,
        dilation=1,
    )

    res_out = flag_gems._conv_depthwise2d(inp, weight, kernel, bias=None, stride=stride, padding=padding, dilation=1)
    gems_assert_close(res_out, ref_out, dtype)


SHAPE_CONV1D = [
    ((32, 2, 4), (17, 2, 2)),
    ((32, 15, 6), (17, 15, 2)),
    ((32, 16, 1024), (1024, 16, 8)),
    ((64, 64, 64), (128, 64, 7)),
    ((32, 12, 9), (17, 12, 3)),
    ((32, 6, 6), (64, 6, 2)),
]


@pytest.mark.conv1d
@pytest.mark.parametrize("shape, kernel", SHAPE_CONV1D)
@pytest.mark.parametrize("stride", [2])
@pytest.mark.parametrize("padding", [1])
@pytest.mark.parametrize("dtype", [torch.float32])
def test_accuracy_conv1d(shape, kernel, stride, padding, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device, requires_grad=True)
    ref_inp = to_reference(inp, True)
    weight = torch.randn(kernel, dtype=dtype, device=flag_gems.device)
    ref_weight = to_reference(weight, True)
    ref_out = torch.nn.functional.conv1d(ref_inp, ref_weight, bias=None, stride=stride, padding=padding, dilation=1)

    res_out = flag_gems.conv1d(inp, weight, bias=None, stride=stride, padding=padding, dilation=1)
    gems_assert_close(res_out, ref_out, dtype)


SHAPE_CONV2D = [
    ((1, 2, 5, 5), (1, 2, 3, 3), 1),
    ((2, 3, 9, 9), (1, 3, 3, 3), 1),
    ((2, 2, 3, 3), (1, 2, 2, 2), 1),
    ((32, 8, 8, 8), (32, 8, 2, 2), 1),
    ((18, 16, 4, 4), (16, 16, 2, 2), 1),
]


@pytest.mark.conv2d
@pytest.mark.parametrize("shape, kernel,groups", SHAPE_CONV2D)
@pytest.mark.parametrize("stride", [1, 2])
@pytest.mark.parametrize("padding", [0, 1])
@pytest.mark.parametrize("dtype", [torch.float32])
@pytest.mark.parametrize("dilation", [1, 2])
@pytest.mark.parametrize("bias", [True, False])
def test_accuracy_conv2d(shape, kernel, stride, padding, groups, dtype, dilation, bias):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device, requires_grad=True)
    ref_inp = to_reference(inp, True)
    torch.backends.cudnn.allow_tf32 = False
    weight = torch.randn(kernel, dtype=dtype, device=flag_gems.device, requires_grad=True)
    if bias is True:
        bias = torch.randn([weight.shape[0]], dtype=dtype, device=flag_gems.device, requires_grad=True)
        bias_ref = to_reference(bias, True)
    else:
        bias = None
        bias_ref = None

    ref_weight = to_reference(weight, True)
    ref_out = torch.nn.functional.conv2d(
        ref_inp,
        ref_weight,
        bias=bias_ref,
        groups=groups,
        stride=stride,
        padding=padding,
        dilation=dilation,
    ).to(dtype)

    res_out = flag_gems.conv2d(
        inp,
        weight,
        bias=bias,
        groups=groups,
        stride=stride,
        padding=padding,
        dilation=dilation,
    )

    gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.multinomial
@pytest.mark.parametrize("shape", [(1024, 10)])
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
@pytest.mark.parametrize("n_samples", [2048])
def test_accuracy_multinomial_with_replacement(shape, dtype, n_samples):
    # First use multinomial to generate a series of indices, then
    # use the index counts as the input probabilities (scaled)
    rand_indices = torch.multinomial(torch.rand(shape), n_samples, True).to(device)
    inp_counts = torch.nn.functional.one_hot(rand_indices).sum(1)
    with flag_gems.use_gems():
        out_indices = torch.multinomial(inp_counts.to(dtype=dtype), n_samples, True)
    out_counts = torch.nn.functional.one_hot(out_indices).sum(1)
    # Do a simple Chi-square test
    assert torch.equal(inp_counts.sum(-1), out_counts.sum(-1))


NONZERO_SHAPES = [(2, 32)] if QUICK_MODE else REDUCTION_SHAPES + [(2637, )]


@pytest.mark.nonzero
@pytest.mark.parametrize("shape", NONZERO_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES + INT_DTYPES + [torch.bool])
def test_accuracy_nonzero(shape, dtype):
    if dtype == torch.bool:
        inp = torch.randint(0, 2, shape, dtype=torch.int, device=flag_gems.device).to(dtype)
    elif dtype in INT_DTYPES:
        inp = torch.randint(-3, 3, shape, device=flag_gems.device).to(dtype)
    else:
        inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = to_reference(inp, False)

    ref_out = torch.nonzero(ref_inp)
    with flag_gems.use_gems():
        res_out = torch.nonzero(inp)

    gems_assert_equal(res_out, ref_out)


@pytest.mark.normal
@pytest.mark.parametrize("float", ["none", "mean", "std"])
@pytest.mark.parametrize("shape", DISTRIBUTION_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_normal(float, shape, dtype):
    loc = (3.0 if float == "mean" else torch.full(size=shape, fill_value=3.0, dtype=dtype, device=flag_gems.device))
    scale = (10.0 if float == "std" else torch.full(size=shape, fill_value=10.0, dtype=dtype, device=flag_gems.device))
    with flag_gems.use_gems():
        res_out = torch.normal(loc, scale)
    mean = torch.mean(res_out)
    std = torch.std(res_out)
    assert torch.abs(mean - 3.0) < 0.1
    assert torch.abs(std - 10.0) < 0.1


@pytest.mark.rand
@pytest.mark.parametrize("shape", DISTRIBUTION_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_rand(shape, dtype):
    with flag_gems.use_gems():
        res_out = torch.rand(shape, dtype=dtype, device=device)
    assert (res_out <= 1.0).all()
    assert (res_out >= 0.0).all()


@pytest.mark.rand_like
@pytest.mark.parametrize("shape", DISTRIBUTION_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_rand_like(shape, dtype):
    x = torch.randn(size=shape, dtype=dtype, device=device)
    with flag_gems.use_gems():
        res_out = torch.rand_like(x)
    assert (res_out <= 1.0).all()
    assert (res_out >= 0.0).all()


@pytest.mark.randn
@pytest.mark.parametrize("shape", DISTRIBUTION_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_randn(shape, dtype):
    if flag_gems.vendor_name == "cambricon":
        torch.manual_seed(42)
    with flag_gems.use_gems():
        res_out = torch.randn(shape, dtype=dtype, device=device)
    mean = torch.mean(res_out)
    std = torch.std(res_out)
    assert torch.abs(mean) < 0.01
    assert torch.abs(std - 1) < 0.01


@pytest.mark.topk
@pytest.mark.parametrize("batch_size", [4, 8])
@pytest.mark.parametrize("hiddensize", [128, 256])
@pytest.mark.parametrize("topk", [5])
@pytest.mark.parametrize("largest", [True, False])
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_topk(
    batch_size,
    hiddensize,
    topk,
    largest,
    dtype,
):
    x = torch.arange(hiddensize, dtype=dtype, device=flag_gems.device)
    x = x.repeat(batch_size).reshape(batch_size, hiddensize)

    # Each row use different shuffled index.
    for bsz in range(batch_size):
        col_indices = torch.randperm(x.size(1))
        x[bsz, :] = x[bsz, col_indices]
    ref_x = to_reference(x)

    if flag_gems.vendor_name == "kunlunxin" and dtype == torch.float16:
        ref_x = ref_x.cuda()

    ref_value, ref_index = torch.topk(ref_x, topk, largest=largest)

    if flag_gems.vendor_name == "kunlunxin" and dtype == torch.float16:
        if TO_CPU:
            ref_value = ref_value.cpu()
            ref_index = ref_index.cpu()

    with flag_gems.use_gems():
        res_value, res_index = torch.topk(x, topk, largest=largest)

    gems_assert_close(res_value, ref_value, dtype)
    gems_assert_equal(res_index, ref_index)


@pytest.mark.uniform_
@pytest.mark.parametrize("shape", DISTRIBUTION_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_uniform(shape, dtype):
    x = torch.randn(size=shape, dtype=dtype, device=flag_gems.device)
    with flag_gems.use_gems():
        x.uniform_(-3, 3)
    assert (x <= 3.0).all()
    assert (x >= -3.0).all()


@pytest.mark.embedding
@pytest.mark.parametrize("EmbeddingSize", [1024] if TO_CPU else [4096])
@pytest.mark.parametrize("Batch", [2] if TO_CPU else [2, 4])
@pytest.mark.parametrize("M", [4] if TO_CPU else [4, 8])
@pytest.mark.parametrize("N", [8] if TO_CPU else [128, 256, 4096])
@pytest.mark.parametrize("padding_idx", [None, -1, 1, 2])
@pytest.mark.parametrize("scale_grad_by_freq", [True, False])
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_embedding(EmbeddingSize, Batch, M, N, padding_idx, scale_grad_by_freq, dtype):
    if flag_gems.vendor_name == "kunlunxin":
        torch.manual_seed(0)
        torch.cuda.manual_seed_all(0)

    res_indices = torch.randint(0, EmbeddingSize, (Batch, M), device=flag_gems.device, requires_grad=False)
    res_embedding = torch.randn((EmbeddingSize, N), device=flag_gems.device, dtype=dtype, requires_grad=True)
    ref_embedding = to_reference(res_embedding)
    ref_indices = to_reference(res_indices)

    ref_out = torch.nn.functional.embedding(ref_indices, ref_embedding, padding_idx,
                                            scale_grad_by_freq=scale_grad_by_freq)
    with flag_gems.use_gems():
        res_out = torch.nn.functional.embedding(
            res_indices,
            res_embedding,
            padding_idx,
            scale_grad_by_freq=scale_grad_by_freq,
        )
    gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.NLLLoss
@pytest.mark.parametrize("reduction", ["mean", "none", "sum"])
@pytest.mark.parametrize("weight", [True, False])
@pytest.mark.parametrize("shape", REDUCTION_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
@pytest.mark.parametrize("ignore_index", [1, 200, -100])
def test_accuracy_nll_loss(shape, dtype, ignore_index, reduction, weight):
    dim = 1
    target_shape = list(shape)
    del target_shape[dim]

    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device, requires_grad=True)
    target = torch.randint(0, shape[dim], target_shape, device=flag_gems.device)
    if weight:
        weight = torch.randn(shape[dim], dtype=dtype, device=flag_gems.device)
    else:
        weight = None
    ref_inp = to_reference(inp, True)
    ref_target = to_reference(target)
    ref_weight = to_reference(weight, True)

    ref_out = torch.nn.functional.nll_loss(ref_inp, ref_target, ref_weight, reduction=reduction,
                                           ignore_index=ignore_index)
    with flag_gems.use_gems():
        res_out = torch.nn.functional.nll_loss(inp, target, weight, reduction=reduction, ignore_index=ignore_index)
    reduce_dim = 1 if reduction == "none" else target.numel()
    gems_assert_close(res_out, ref_out, dtype, reduce_dim=reduce_dim, equal_nan=True)


@pytest.mark.scaled_dot_product_attention
@pytest.mark.parametrize("batch", [8, 16])
@pytest.mark.parametrize("num_head", [1, 8])
@pytest.mark.parametrize("q_seq_len", [17, 64, 128])
@pytest.mark.parametrize("kv_seq_len", [7, 87, 128, 577, 2048])
@pytest.mark.parametrize("head_size", [64, 128])
@pytest.mark.parametrize("add_bias", [True, False])
@pytest.mark.parametrize("is_causal", [True, False])
@pytest.mark.parametrize("dtype", [torch.float32])
def test_scaled_dot_product_attention(batch, num_head, q_seq_len, kv_seq_len, head_size, add_bias, is_causal, dtype):
    np.random.seed(0)
    np_query = np.random.uniform(-0.05, 0.05, (batch, num_head, q_seq_len, head_size)).astype(np.float32)
    np_key = np.random.uniform(-0.05, 0.05, (batch, num_head, kv_seq_len, head_size)).astype(np.float32)
    np_value = np.random.uniform(-0.05, 0.05, (batch, num_head, kv_seq_len, head_size)).astype(np.float32)
    np_attn_bias = np.random.uniform(-0.05, 0.05, (batch, num_head, q_seq_len, kv_seq_len)).astype(np.float32)

    query = torch.tensor(np_query, device=device, dtype=dtype)
    key = torch.tensor(np_key, device=device, dtype=dtype)
    value = torch.tensor(np_value, device=device, dtype=dtype)
    if add_bias:
        attn_bias = torch.tensor(np_attn_bias, device=device, dtype=dtype)
    else:
        attn_bias = None

    ref_query = to_reference(query, False)
    ref_key = to_reference(key, False)
    ref_value = to_reference(value, False)
    ref_attn_bias = to_reference(attn_bias, False) if add_bias else None

    scale = float(1.0 / np.sqrt(head_size))

    if is_causal:
        torch_result = torch.nn.functional.scaled_dot_product_attention(
            ref_query,
            ref_key,
            ref_value,
            scale=scale,
            is_causal=is_causal,
        )
    else:
        torch_result = torch.nn.functional.scaled_dot_product_attention(
            ref_query,
            ref_key,
            ref_value,
            attn_mask=ref_attn_bias,
            scale=scale,
            is_causal=is_causal,
        )

    with flag_gems.use_gems():
        if is_causal:
            flaggem_result = torch.nn.functional.scaled_dot_product_attention(query, key, value, scale=scale,
                                                                              is_causal=is_causal)
        else:
            flaggem_result = torch.nn.functional.scaled_dot_product_attention(query, key, value, attn_mask=attn_bias,
                                                                              scale=scale, is_causal=is_causal)
    gems_assert_close(flaggem_result, torch_result, dtype, reduce_dim=head_size)


@pytest.mark.upsample_nearest2d
@pytest.mark.parametrize("scale", [(2, 2), (2.1, 3.7), (1.3, 5.1), (0.3, 0.5)])
@pytest.mark.parametrize("shape", UPSAMPLE_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_upsample_nearest2d(dtype, shape, scale):
    input = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_i = to_reference(input).to(torch.float32)
    output_size = [int(input.shape[i + 2] * scale[i]) for i in range(2)]
    ref_out = torch._C._nn.upsample_nearest2d(ref_i, output_size=output_size).to(dtype)
    with flag_gems.use_gems():
        res_out = torch._C._nn.upsample_nearest2d(input, output_size=output_size)
    gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.erf
@pytest.mark.parametrize("shape", POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_erf(shape, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = to_reference(inp)

    ref_out = torch.erf(ref_inp)
    with flag_gems.use_gems():
        res_out = torch.erf(inp)

    gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.resolve_conj
@pytest.mark.parametrize("shape", SPECIAL_SHAPES)
@pytest.mark.parametrize("dtype", [torch.cfloat])
def test_accuracy_resolve_conj(shape, dtype):
    x = torch.randn(size=shape, dtype=dtype, device="cpu")
    y = x.conj()
    assert y.is_conj()
    with flag_gems.use_gems():
        res_y = y.to(device=flag_gems.device)
        z = res_y.resolve_conj()
    assert not z.is_conj()


@pytest.mark.resolve_neg
@pytest.mark.parametrize("shape", SPECIAL_SHAPES)
@pytest.mark.parametrize("dtype", [torch.cfloat])
def test_accuracy_resolve_neg(shape, dtype):
    x = torch.randn(size=shape, dtype=dtype, device=flag_gems.device)
    y = x.conj()
    z = y.imag
    assert z.is_neg()
    with flag_gems.use_gems():
        out = z.resolve_neg()
    assert not out.is_neg()


@pytest.mark.arange
@pytest.mark.parametrize("start", [0, 1, 3])
@pytest.mark.parametrize("step", [1, 2, 5])
@pytest.mark.parametrize("end", [128, 256, 1024])
@pytest.mark.parametrize("dtype", FLOAT_DTYPES + ALL_INT_DTYPES + [None])
@pytest.mark.parametrize("device", [device, None])
@pytest.mark.parametrize("pin_memory", [False, None])
def test_arange(start, step, end, dtype, device, pin_memory):
    if TO_CPU:
        return
    ref_out = torch.arange(start, end, step, dtype=dtype, device=device, pin_memory=pin_memory)
    with flag_gems.use_gems():
        res_out = torch.arange(start, end, step, dtype=dtype, device=device, pin_memory=pin_memory)

    gems_assert_equal(res_out, ref_out)


CAT_SHAPES = [
    [(1, 32), (8, 32)],
    [(16, 128), (32, 128)],
    [(1024, 1024), (1024, 1024)],
    [(1, 1024, 256), (8, 1024, 256), (16, 1024, 256)],
    [(16, 320, 15), (32, 320, 15), (64, 320, 15)],
    [(16, 128, 64, 64), (16, 128, 64, 64), (24, 128, 64, 64), (32, 128, 64, 64)],
]


def gen_cat_shapes_dim(shapes):
    results = []
    for tensor_shapes in shapes:
        assert all([len(s) == len(tensor_shapes[0]) for s in tensor_shapes]), "All tensor rank must agree."
        assert all([s[-1] == tensor_shapes[0][-1]
                    for s in tensor_shapes]), "All tensor must have same shape except cat dim."
        rank = len(tensor_shapes[0])
        results.append([tensor_shapes, 0])
        for dim in range(1, rank):
            results.append([[(s[dim], *s[1:dim], s[0], *s[dim + 1:]) for s in tensor_shapes], dim])
            results.append([
                [(s[dim], *s[1:dim], s[0], *s[dim + 1:]) for s in tensor_shapes],
                dim - rank,
            ])
    return results


@pytest.mark.cat
@pytest.mark.parametrize("shape, dim", gen_cat_shapes_dim(CAT_SHAPES))
@pytest.mark.parametrize("dtype", FLOAT_DTYPES + INT_DTYPES)
def test_accuracy_cat(shape, dim, dtype):
    if dtype in FLOAT_DTYPES:
        inp = [torch.randn(s, dtype=dtype, device=flag_gems.device) for s in shape]
    else:
        inp = [torch.randint(low=0, high=0x7FFF, size=s, dtype=dtype, device="cpu").to(flag_gems.device) for s in shape]
    ref_inp = [to_reference(_) for _ in inp]
    ref_out = torch.cat(ref_inp, dim)

    with flag_gems.use_gems():
        res_out = torch.cat(inp, dim)
    gems_assert_equal(res_out, ref_out)


@pytest.mark.cat
@pytest.mark.parametrize(
    "shape, dim",
    [
        (((0, 3), (2, 3)), 0),
        (((0, 3), (0, 3)), 0),
        (((0, ), (0, )), 0),
    ],
)
@pytest.mark.parametrize("dtype", [torch.float32])
def test_accuracy_cat_empty_tensor(shape, dim, dtype):
    inp = [torch.randn(s, dtype=dtype, device=flag_gems.device) for s in shape]
    ref_inp = [to_reference(_) for _ in inp]
    ref_out = torch.cat(ref_inp, dim)

    with flag_gems.use_gems():
        res_out = torch.cat(inp, dim)
    gems_assert_equal(res_out, ref_out)


@pytest.mark.fill_
@pytest.mark.parametrize("value", [0, 1, 9])
@pytest.mark.parametrize("shape", SPECIAL_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_fill_(value, shape, dtype):
    # Test fill_.Scalar
    x = torch.ones(shape, device=flag_gems.device, dtype=dtype)
    ref_x = to_reference(x.clone(), False)

    ref_x.fill_(value)
    with flag_gems.use_gems():
        x.fill_(value)

    gems_assert_equal(x, ref_x)

    # Test fill_.Tensor
    x = torch.ones(shape, device=flag_gems.device, dtype=dtype)
    ref_x = to_reference(x.clone(), False)
    value_tensor = torch.tensor(value, device=flag_gems.device, dtype=dtype)
    if flag_gems.device == "musa":
        ref_x.fill_(value_tensor.cpu())
    else:
        ref_x.fill_(value_tensor)
    with flag_gems.use_gems():
        x.fill_(value_tensor)

    gems_assert_equal(x, ref_x)


@pytest.mark.full
@pytest.mark.parametrize("shape", POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
@pytest.mark.parametrize("fill_value", [3.1415926])
def test_accuracy_full(shape, dtype, fill_value):
    # without dtype
    ref_out = torch.full(shape, fill_value, device="cpu" if TO_CPU else device)
    with flag_gems.use_gems():
        res_out = torch.full(shape, fill_value, device=flag_gems.device)
    gems_assert_equal(res_out, ref_out)

    # with dtype
    ref_out = torch.full(shape, fill_value, dtype=dtype, device="cpu" if TO_CPU else device)
    with flag_gems.use_gems():
        res_out = torch.full(shape, fill_value, dtype=dtype, device=flag_gems.device)
    gems_assert_equal(res_out, ref_out)


@pytest.mark.full_like
@pytest.mark.parametrize("shape", POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
@pytest.mark.parametrize("xdtype", FLOAT_DTYPES)
@pytest.mark.parametrize("fill_value", [3.1415926])
def test_accuracy_full_like(shape, dtype, xdtype, fill_value):
    x = torch.empty(size=shape, dtype=xdtype, device="cpu" if TO_CPU else device)

    # without dtype
    with flag_gems.use_gems():
        res_out = torch.full_like(x, fill_value)
    gems_assert_equal(res_out, torch.full_like(x, fill_value))

    # with dtype
    with flag_gems.use_gems():
        res_out = torch.full_like(x, fill_value, dtype=dtype)
    gems_assert_equal(res_out, torch.full_like(x, fill_value, dtype=dtype))


@pytest.mark.gather
@pytest.mark.parametrize(
    "inp_shape",
    [(32, 8, 4)] if QUICK_MODE else [(512, 128, 32), (1024, 64, 16), (128, 32, 256)],
)
@pytest.mark.parametrize("dim", [0, 1, 2])
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_gather(inp_shape, dim, dtype):
    inp = torch.randn(inp_shape, dtype=dtype, device=flag_gems.device, requires_grad=True)
    size_dim = inp_shape[dim]

    import random

    index_shape = [
        random.randint(1, inp_shape[0]),
        random.randint(1, inp_shape[1]),
        random.randint(1, inp_shape[2]),
    ]
    index = torch.empty(tuple(index_shape), dtype=torch.long, device=flag_gems.device)

    m, n, o = index_shape

    index_size_dim = index_shape[dim]
    # make unique indices
    for i in range(1 if dim == 0 else m):
        for j in range(1 if dim == 1 else n):
            for k in range(1 if dim == 2 else o):
                ii = [i, j, k]
                ii[dim] = slice(0, index.size(dim) + 1)
                index[tuple(ii)] = torch.randperm(size_dim)[0:index_size_dim]

    ref_inp = to_reference(inp)
    ref_index = to_reference(index)
    ref_out = torch.gather(ref_inp, dim, ref_index)

    with flag_gems.use_gems():
        res_out = torch.gather(inp, dim, index)

    gems_assert_equal(res_out, ref_out)


INDEX_PUT_SHAPE_ACC_FALSE = (
    ((2**28, ), ((2**16, ), ), (2**16, )),
    ((32, 32), ((8, ), (8, )), (8, )),
    ((32, 32), ((8, ), (2, 8)), (8, )),
    ((32, 32), ((2, 8), ), (32, )),
    ((512, 512, 512), ((128, ), (128, ), (128, )), (128, )),
    ((512, 512, 512), ((2, 128), (128, ), (128, )), (128, )),
    ((512, 512, 512), ((2, 128), ), (512, )),
    (
        (64, 64, 64),
        (
            (2, 8),
            (2, 8),
        ),
        (2, 8, 64),
    ),
)


def gen_indices(input_shape, indices_shape, accumulate):
    indices = []
    for i, shape in enumerate(indices_shape):
        index = np.random.choice(np.arange(input_shape[i]), size=shape, replace=accumulate)
        indices.append(torch.tensor(index, device=flag_gems.device))
    return indices


@pytest.mark.skipif(flag_gems.vendor_name == "spacemit", reason="TODO")
@pytest.mark.index_put
@pytest.mark.parametrize("input_shape, indices_shape, values_shape", INDEX_PUT_SHAPE_ACC_FALSE)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_index_put_acc_false(input_shape, indices_shape, values_shape, dtype):
    accumulate = False
    inp = torch.randn(input_shape, dtype=dtype, device=flag_gems.device, requires_grad=False)
    indices = gen_indices(input_shape, indices_shape, accumulate)
    values = torch.randn(values_shape, dtype=dtype, device=flag_gems.device, requires_grad=False)

    ref_inp = to_reference(inp)
    ref_indices = [to_reference(index) for index in indices]
    ref_values = to_reference(values)
    ref_out = torch.index_put(ref_inp, ref_indices, ref_values, accumulate)
    out = flag_gems.index_put(inp, indices, values, accumulate)
    gems_assert_close(out, ref_out, dtype)


INDEX_PUT_SHAPE_ACC_TRUE = (
    ((2**28, ), ((2**16, ), ), (2**16, )),
    ((32, 32), ((8, ), (8, )), (8, )),
    ((512, 512, 512), ((128, ), (128, ), (128, )), (128, )),
    ((64, 64, 64), ((2, 8), (2, 8), (2, 8)), (2, 8)),
)


@pytest.mark.skipif(flag_gems.vendor_name == "spacemit", reason="TODO")
@pytest.mark.index_put
@pytest.mark.parametrize("input_shape, indices_shape, values_shape", INDEX_PUT_SHAPE_ACC_TRUE)
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_index_put_acc_true(input_shape, indices_shape, values_shape, dtype):
    init_seed(0)
    accumulate = True
    inp = torch.randn(input_shape, dtype=dtype, device=flag_gems.device, requires_grad=False)
    indices = gen_indices(input_shape, indices_shape, accumulate)
    values = torch.randn(values_shape, dtype=dtype, device=flag_gems.device, requires_grad=False)

    ref_inp = to_reference(inp, upcast=True)
    ref_indices = [to_reference(index) for index in indices]
    ref_values = to_reference(values, upcast=True)
    ref_out = torch.index_put(ref_inp, ref_indices, ref_values, accumulate)
    out = flag_gems.index_put(inp, indices, values, accumulate)
    gems_assert_close(out, ref_out, dtype)


@pytest.mark.skipif(flag_gems.vendor_name == "spacemit", reason="TODO")
@pytest.mark.index_put_
@pytest.mark.parametrize("input_shape, indices_shape, values_shape", INDEX_PUT_SHAPE_ACC_FALSE)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_index_put__acc_false(input_shape, indices_shape, values_shape, dtype):
    accumulate = False
    inp = torch.randn(input_shape, dtype=dtype, device=flag_gems.device, requires_grad=False)
    indices = gen_indices(input_shape, indices_shape, accumulate)
    values = torch.randn(values_shape, dtype=dtype, device=flag_gems.device, requires_grad=False)

    ref_inp = to_reference(inp)
    ref_indices = [to_reference(index) for index in indices]
    ref_values = to_reference(values)
    torch.index_put_(ref_inp, ref_indices, ref_values, accumulate)
    flag_gems.index_put_(inp, indices, values, accumulate)
    gems_assert_close(inp, ref_inp, dtype)


@pytest.mark.skipif(flag_gems.vendor_name == "spacemit", reason="TODO")
@pytest.mark.index_put_
@pytest.mark.parametrize("input_shape, indices_shape, values_shape", INDEX_PUT_SHAPE_ACC_TRUE)
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_index_put__acc_true(input_shape, indices_shape, values_shape, dtype):
    if flag_gems.vendor_name == "kunlunxin":
        torch.manual_seed(0)
        torch.cuda.manual_seed_all(0)

    accumulate = True
    inp = torch.randn(input_shape, dtype=dtype, device=flag_gems.device, requires_grad=False)
    indices = gen_indices(input_shape, indices_shape, accumulate)
    values = torch.randn(values_shape, dtype=dtype, device=flag_gems.device, requires_grad=False)

    ref_inp = to_reference(inp, upcast=True)
    ref_indices = [to_reference(index) for index in indices]
    ref_values = to_reference(values, upcast=True)
    torch.index_put_(ref_inp, ref_indices, ref_values, accumulate)
    flag_gems.index_put_(inp, indices, values, accumulate)
    gems_assert_close(inp, ref_inp, dtype)


@pytest.mark.index_select
@pytest.mark.parametrize("shape", REDUCTION_SHAPES)
@pytest.mark.parametrize("dim", DIM_LIST)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_index_select(shape, dim, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    index_size = inp.size(dim)
    from math import floor

    index = torch.randint(0, index_size, [floor(index_size * 0.8)], device=flag_gems.device)

    ref_inp = to_reference(inp)
    ref_index = to_reference(index)
    ref_out = torch.index_select(ref_inp, dim, ref_index)
    with flag_gems.use_gems():
        res_out = torch.index_select(inp, dim, index)

    gems_assert_equal(res_out, ref_out)


@pytest.mark.masked_fill
@pytest.mark.parametrize("shape", POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
@pytest.mark.parametrize("threshold", [0.3, 0.5, 0.7])
@pytest.mark.parametrize(
    "value",
    [
        torch.tensor(1024, device=flag_gems.device),
        torch.scalar_tensor(1024, device=flag_gems.device),
        1024,
    ],
)
def test_accuracy_masked_fill_(shape, dtype, threshold, value):
    inp = torch.zeros(shape, dtype=dtype, device=flag_gems.device)
    mask = torch.randn(shape, dtype=dtype, device=flag_gems.device) < threshold

    ref_inp = to_reference(inp)
    ref_mask = to_reference(mask)
    if torch.is_tensor(value):
        ref_inp.masked_fill_(ref_mask, to_reference(value))
    else:
        ref_inp.masked_fill_(ref_mask, value)
    with flag_gems.use_gems():
        inp.masked_fill_(mask, value)

    gems_assert_equal(inp, ref_inp)


@pytest.mark.ones
@pytest.mark.parametrize("shape", POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_ones(shape, dtype):
    # without dtype
    with flag_gems.use_gems():
        res_out = torch.ones(shape, device=flag_gems.device)
    gems_assert_equal(res_out, torch.ones(shape, device="cpu" if TO_CPU else device))

    # with dtype
    with flag_gems.use_gems():
        res_out = torch.ones(shape, dtype=dtype, device=flag_gems.device)
    gems_assert_equal(res_out, torch.ones(shape, dtype=dtype, device="cpu" if TO_CPU else device))


@pytest.mark.pad
@pytest.mark.parametrize("shape", [[1024, 1024], [64, 64, 64, 64]])
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
@pytest.mark.parametrize("pad_mode", ["constant", "reflect", "replicate", "circular"])
@pytest.mark.parametrize("contiguous", [True, False])
def test_pad(shape, dtype, pad_mode, contiguous):
    if flag_gems.vendor_name == "kunlunxin":
        torch.manual_seed(0)
        torch.cuda.manual_seed_all(0)

    x = torch.randn(size=shape, dtype=dtype, device=flag_gems.device)
    if not contiguous:
        if flag_gems.vendor_name == "kunlunxin":
            x = x.cpu()[::2, ::2].to(flag_gems.device)
        else:
            x = x[::2, ::2]

    ref_x = to_reference(x)
    if ref_x.dtype == torch.float16:
        ref_x = ref_x.to(torch.float32)

    rank = x.ndim
    pad_params = list(
        torch.randint(0, 10, (rank * 2, ), dtype=torch.int32, device="cpu") if pad_mode ==
        "constant" else torch.randint(0, 10, (rank, ), dtype=torch.int32, device="cpu"))
    pad_value = float(torch.randint(0, 1024, (1, ), dtype=torch.int32, device="cpu"))

    if pad_mode != "constant":
        pad_params = [(pad_val + 2 - 1) // 2 * 2 for pad_val in pad_params]
        pad_value = None

    ref_pad_params = [to_reference(pad_param) for pad_param in pad_params]

    ref_out = torch.nn.functional.pad(ref_x, ref_pad_params, pad_mode, pad_value)
    with flag_gems.use_gems():
        res_out = torch.nn.functional.pad(x, pad_params, pad_mode, pad_value)

    if ref_out.dtype != res_out.dtype:
        ref_out = ref_out.to(res_out.dtype)

    gems_assert_equal(res_out, ref_out)


REPEAT_SIZES = [(2, 3, 4, 5), (5, 0, 4)]


@pytest.mark.repeat
@pytest.mark.parametrize("shape", POINTWISE_SHAPES)
@pytest.mark.parametrize("sizes", REPEAT_SIZES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_repeat(shape, sizes, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = to_reference(inp)
    sizes = unsqueeze_tuple(sizes, inp.ndim)

    ref_out = ref_inp.repeat(*sizes)
    with flag_gems.use_gems():
        res_out = inp.repeat(*sizes)

    gems_assert_close(res_out, ref_out, dtype)


REPEAT_INTERLEAVE_SHAPES = [
    (1024, 1024),
    (20, 320, 15),
    (16, 128, 64, 60),
    (16, 7, 57, 32, 29),
]
REPEAT_INTERLEAVE_REPEATS = [2]
REPEAT_INTERLEAVE_DIM = [-1, 0, None]


@pytest.mark.repeat_interleave
@pytest.mark.parametrize("shape", REPEAT_INTERLEAVE_SHAPES + [(1, )])
@pytest.mark.parametrize("dim", REPEAT_INTERLEAVE_DIM)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_repeat_interleave_self_int(shape, dim, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    repeats = 2
    ref_inp = to_reference(inp)

    ref_out = torch.repeat_interleave(ref_inp, repeats, dim)
    with flag_gems.use_gems():
        res_out = torch.repeat_interleave(ref_inp, repeats, dim)
    gems_assert_equal(res_out, ref_out)


@pytest.mark.repeat_interleave
@pytest.mark.parametrize("shape", REPEAT_INTERLEAVE_SHAPES)
@pytest.mark.parametrize("dim", REPEAT_INTERLEAVE_DIM)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_repeat_interleave_self_int_non_contiguous(shape, dim, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)[::2]
    repeats = 2
    ref_inp = to_reference(inp)

    ref_out = torch.repeat_interleave(ref_inp, repeats, dim)
    with flag_gems.use_gems():
        res_out = torch.repeat_interleave(ref_inp, repeats, dim)
    gems_assert_equal(res_out, ref_out)


@pytest.mark.repeat_interleave
@pytest.mark.parametrize("shape", UT_SHAPES_1D)
@pytest.mark.parametrize("dtype", [torch.int32])
def test_accuracy_repeat_interleave_tensor(shape, dtype):
    repeats = torch.randint(0, 30, shape, dtype=dtype, device=flag_gems.device)
    ref_repeats = to_reference(repeats)
    ref_out = torch.repeat_interleave(ref_repeats)

    with flag_gems.use_gems():
        res_out = torch.repeat_interleave(repeats)
    gems_assert_equal(res_out, ref_out)


@pytest.mark.skipif(flag_gems.vendor_name == "spacemit", reason="TODO")
@pytest.mark.repeat_interleave
@pytest.mark.parametrize("shape", REPEAT_INTERLEAVE_SHAPES)
@pytest.mark.parametrize("dim", [-1, 0, 1])
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_repeat_interleave_self_tensor(shape, dim, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    repeats = torch.randint(0, 30, (shape[dim], ), device=flag_gems.device)
    ref_inp = to_reference(inp)
    ref_repeats = to_reference(repeats)

    ref_out = torch.repeat_interleave(ref_inp, ref_repeats, dim)
    with flag_gems.use_gems():
        res_out = torch.repeat_interleave(inp, repeats, dim)
    gems_assert_equal(res_out, ref_out)


@pytest.mark.scatter
@pytest.mark.parametrize("src_shape", [(32, 8, 4)] if QUICK_MODE else [(128, 16, 4), (256, 32, 8)])
@pytest.mark.parametrize("inp_shape", [(64, 16, 8)] if QUICK_MODE else [(512, 128, 32), (1024, 64, 16)])
@pytest.mark.parametrize("dim", [0, 1, 2])
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_scatter_src(src_shape, inp_shape, dim, dtype):
    inp = torch.randn(inp_shape, dtype=dtype, device=flag_gems.device)
    src = torch.randn(src_shape, dtype=dtype, device=flag_gems.device)
    size_dim = min(src_shape[dim], inp_shape[dim])

    import random

    index_shape = [
        random.randint(1, min(src_shape[0], inp_shape[0])),
        random.randint(1, min(src_shape[1], inp_shape[1])),
        random.randint(1, min(src_shape[2], inp_shape[2])),
    ]
    index = torch.empty(tuple(index_shape), dtype=torch.long, device=flag_gems.device)

    m, n, o = index_shape

    index_size_dim = index_shape[dim]
    # make unique indices
    for i in range(1 if dim == 0 else m):
        for j in range(1 if dim == 1 else n):
            for k in range(1 if dim == 2 else o):
                ii = [i, j, k]
                ii[dim] = slice(0, index.size(dim) + 1)
                index[tuple(ii)] = torch.randperm(size_dim)[0:index_size_dim]

    ref_inp = to_reference(inp)
    ref_index = to_reference(index)
    ref_src = to_reference(src)
    ref_out = torch.scatter(ref_inp, dim, ref_index, ref_src)
    with flag_gems.use_gems():
        res_out = torch.scatter(inp, dim, index, src)

    gems_assert_equal(res_out, ref_out)


@pytest.mark.skipif(flag_gems.vendor_name == "spacemit", reason="TODO")
@pytest.mark.index_add
@pytest.mark.parametrize("shape", REDUCTION_SHAPES)
@pytest.mark.parametrize("dim", DIM_LIST)
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_accuracy_index_add(shape, dim, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)

    src_shape = list(inp.shape)
    index_max = src_shape[dim]
    index_len = index_max
    index = torch.randperm(index_len, device=flag_gems.device)
    src_shape[dim] = index_len
    src = torch.randn(src_shape, dtype=dtype, device=flag_gems.device)
    alpha = 2

    ref_inp = to_reference(inp)
    ref_src = to_reference(src)
    ref_index = to_reference(index)
    ref_out = torch.index_add(ref_inp, dim, ref_index, ref_src, alpha=alpha)
    with flag_gems.use_gems():
        res_out = torch.index_add(inp, dim, index, src, alpha=alpha)

    gems_assert_close(res_out, ref_out, dtype=dtype, reduce_dim=dim)


@pytest.mark.select_scatter
@pytest.mark.parametrize("shape", REDUCTION_SHAPES)
@pytest.mark.parametrize("dim", DIM_LIST)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_select_scatter(shape, dim, dtype):
    import random

    index = random.randint(0, shape[dim] - 1)
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)

    src_shape = list(inp.shape)
    del src_shape[dim]
    src = torch.randn(src_shape, dtype=dtype, device=flag_gems.device)

    ref_inp = to_reference(inp)
    ref_src = to_reference(src)
    ref_out = torch.select_scatter(ref_inp, dim=dim, index=index, src=ref_src)
    with flag_gems.use_gems():
        res_out = torch.select_scatter(inp, dim=dim, index=index, src=src)

    gems_assert_equal(res_out, ref_out)


@pytest.mark.select_scatter
def test_accuracy_select_scatter_with_self_overlapping_input():
    dim = 0
    index = 1
    inp = torch.randn((1, 4), device=flag_gems.device).broadcast_to((3, 4))
    src = torch.randn((4, ), device=flag_gems.device)

    ref_inp = to_reference(inp)
    ref_src = to_reference(src)
    ref_out = torch.select_scatter(ref_inp, dim=dim, index=index, src=ref_src)
    with flag_gems.use_gems():
        res_out = torch.select_scatter(inp, dim=dim, index=index, src=src)

    gems_assert_equal(res_out, ref_out)


@pytest.mark.slice_scatter
@pytest.mark.parametrize(("dim", "shape", "stride"), REGULAR_DIM_SHAPE_STRIDES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
@pytest.mark.parametrize("start", [16, 64])
@pytest.mark.parametrize("end", [1024, 256])
@pytest.mark.parametrize("step", [1, 2])
def test_accuracy_slice_scatter(shape, stride, dim, dtype, start, end, step):
    inp = torch.empty_strided(shape, stride, dtype=dtype, device=flag_gems.device)
    inp.copy_(1)

    valid_shape = list(inp.shape)
    size = valid_shape[dim]

    start = start % size
    end = end % (size + 1)

    if end < start:
        end, start = start, end
    elif end == start:
        end = size

    valid_shape[dim] = (end - start + step - 1) // step

    src = torch.rand(valid_shape, dtype=dtype, device=flag_gems.device)

    ref_inp = to_reference(inp)
    ref_src = to_reference(src)
    ref_out = torch.slice_scatter(ref_inp, dim=dim, src=ref_src, start=start, end=end, step=step)

    res_out = flag_gems.ops.slice_scatter(inp, dim=dim, src=src, start=start, end=end, step=step)

    gems_assert_equal(res_out, ref_out)


@pytest.mark.slice_scatter
def test_accuracy_slice_scatter_with_self_overlapping_input():
    inp = torch.randn((3, 1), device=flag_gems.device).broadcast_to((3, 8))
    src = torch.rand((3, 4), device=flag_gems.device)

    start = 0
    end = 8
    step = 2
    dim = 1
    ref_inp = to_reference(inp)
    ref_src = to_reference(src)
    ref_out = torch.slice_scatter(ref_inp, dim=dim, src=ref_src, start=start, end=end, step=step)

    res_out = flag_gems.ops.slice_scatter(inp, dim=dim, src=src, start=start, end=end, step=step)

    gems_assert_equal(res_out, ref_out)


@pytest.mark.stack
@pytest.mark.parametrize("shape", STACK_SHAPES)
@pytest.mark.parametrize("dim", STACK_DIM_LIST)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES + INT_DTYPES)
def test_accuracy_stack(shape, dim, dtype):
    if dtype in FLOAT_DTYPES:
        inp = [torch.randn(s, dtype=dtype, device=flag_gems.device) for s in shape]
    else:
        inp = [torch.randint(low=0, high=0x7FFF, size=s, dtype=dtype, device="cpu").to(flag_gems.device) for s in shape]
    ref_inp = [to_reference(_) for _ in inp]
    ref_out = torch.stack(ref_inp, dim)

    with flag_gems.use_gems():
        res_out = torch.stack(inp, dim)
    gems_assert_equal(res_out, ref_out)


TILE_DIMS = [(0, ), (2, ), (2, 0), (0, 2), (2, 2), (2, 2, 2), (2, 2, 2, 2)]


@pytest.mark.tile
@pytest.mark.parametrize("shape", [(1024, 1024), (20, 320, 15), (16, 128, 64, 60), (16, 7, 57, 32, 29)])
@pytest.mark.parametrize("dims", TILE_DIMS)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_tile(shape, dims, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = to_reference(inp)

    ref_out = torch.tile(ref_inp, dims)
    with flag_gems.use_gems():
        res_out = torch.tile(inp, dims)

    gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.where
@pytest.mark.parametrize("shape", POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_where_self_out_cross_device(shape, dtype):
    inp1 = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    inp2 = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    cond = torch.randint(0, 2, shape, dtype=torch.bool, device=flag_gems.device)

    import itertools

    shapes = (shape, None)
    for a_shape, b_shape, c_shape in itertools.product(shapes, shapes, shapes):
        a = inp1 if a_shape else torch.tensor(0)
        b = inp2 if b_shape else torch.tensor(1)
        c = cond if c_shape else torch.tensor(True)

        ref_a = to_reference(a)
        ref_b = to_reference(b)
        ref_c = to_reference(c)

        ref_out = torch.where(ref_c, ref_a, ref_b)
        with flag_gems.use_gems():
            res_out = torch.where(c, a, b)

        gems_assert_equal(res_out, ref_out)


@pytest.mark.where
@pytest.mark.parametrize("shape", POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_where_self_out(shape, dtype):
    inp1 = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    inp2 = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    cond = torch.randint(0, 2, shape, dtype=torch.bool, device=flag_gems.device)
    out = torch.empty(shape, dtype=dtype, device=flag_gems.device)
    ref_out = to_reference(out)
    ref_inp1 = to_reference(inp1)
    ref_inp2 = to_reference(inp2)
    ref_cond = to_reference(cond)

    ref_out = torch.where(ref_cond, ref_inp1, ref_inp2, out=ref_out)
    with flag_gems.use_gems():
        res_out = torch.where(cond, inp1, inp2, out=out)

    gems_assert_equal(res_out, ref_out)


@pytest.mark.where
@pytest.mark.parametrize("shape", POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_where_self(shape, dtype):
    inp1 = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    inp2 = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp1 = to_reference(inp1)
    ref_inp2 = to_reference(inp2)

    ref_out = torch.where(ref_inp1 > 0, ref_inp1, ref_inp2)
    with flag_gems.use_gems():
        res_out = torch.where(inp1 > 0, inp1, inp2)

    gems_assert_equal(res_out, ref_out)


@pytest.mark.where
@pytest.mark.parametrize("shape", POINTWISE_SHAPES)
@pytest.mark.parametrize("scalar", SCALARS)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_where_scalar_self(shape, scalar, dtype):
    inp1 = scalar
    inp2 = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp2 = to_reference(inp2)

    ref_out = torch.where(ref_inp2 > 0, inp1, ref_inp2)
    with flag_gems.use_gems():
        res_out = torch.where(inp2 > 0, inp1, inp2)

    gems_assert_equal(res_out, ref_out)


@pytest.mark.where
@pytest.mark.parametrize("shape", POINTWISE_SHAPES)
@pytest.mark.parametrize("scalar", SCALARS)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_where_scalar_other(shape, scalar, dtype):
    inp1 = scalar
    inp2 = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp2 = to_reference(inp2)

    ref_out = torch.where(ref_inp2 > 0, ref_inp2, inp1)
    with flag_gems.use_gems():
        res_out = torch.where(inp2 > 0, inp2, inp1)

    gems_assert_equal(res_out, ref_out)


@pytest.mark.zeros
@pytest.mark.parametrize("shape", POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", BOOL_TYPES + ALL_INT_DTYPES + FLOAT_DTYPES)
def test_accuracy_zeros(shape, dtype):
    # without dtype
    with flag_gems.use_gems():
        res_out = torch.zeros(shape, device=flag_gems.device)
    gems_assert_equal(res_out, torch.zeros(shape, device="cpu" if TO_CPU else device))

    # with dtype
    with flag_gems.use_gems():
        res_out = torch.zeros(shape, dtype=dtype, device=flag_gems.device)
    gems_assert_equal(res_out, torch.zeros(shape, dtype=dtype, device="cpu" if TO_CPU else device))


@pytest.mark.maxpool
@pytest.mark.parametrize("shape", [(1, 3, 5, 5)])
@pytest.mark.parametrize("kernel_size", [3])
@pytest.mark.parametrize("stride", [2])
@pytest.mark.parametrize("padding", [0])
@pytest.mark.parametrize("dilation", [1])
@pytest.mark.parametrize("ceil_mode", [True])
@pytest.mark.parametrize("dtype", [torch.float32])
def test_accuracy_maxpool(shape, kernel_size, stride, padding, dilation, ceil_mode, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_pool = torch.nn.MaxPool2d(kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation,
                                  ceil_mode=ceil_mode, return_indices=False)
    ref_out = ref_pool(inp)

    import flag_gems.runtime.backend._spacemit.ops as sp
    res_out = sp.maxpool2d(
        inp,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
    )
    gems_assert_equal(res_out, ref_out)


@pytest.mark.unfold
def test_accuracy_unfold():
    input_tensor = torch.randn(2, 3, 8, 8, device="cpu")
    kernel_size = (3, 5)
    pt_unfold = torch.nn.Unfold(kernel_size, padding=1, stride=2)
    ref_out = pt_unfold(input_tensor)
    res_out = flag_gems.unfold(input_tensor, kernel_size, padding=1, stride=2)
    gems_assert_equal(res_out, ref_out)
