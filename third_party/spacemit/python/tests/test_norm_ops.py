import numpy as np
import pytest
import torch

import triton
from triton.backends.spine_triton.driver import CPUDriver

triton.runtime.driver.set_active(CPUDriver())
import flag_gems

from .accuracy_utils import (
    FLOAT_DTYPES,
    REDUCTION_SHAPES,
    gems_assert_close,
    to_reference,
)
from .conftest import QUICK_MODE

FLOAT_DTYPES = [torch.float32] if QUICK_MODE else FLOAT_DTYPES
DIMS_LIST = [1] if QUICK_MODE else [0, 1, [0, 1], [1, 0]]
KEEPDIM_DIMS = ([(True, DIMS_LIST[0])] if QUICK_MODE else list(zip([True, False] * 2, DIMS_LIST)))


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
    if flag_gems.vendor_name == "spacemit":
        torch.manual_seed(42)

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


@pytest.mark.skipif(flag_gems.vendor_name == "spacemit", reason="TODO")
@pytest.mark.layernorm
@pytest.mark.native_layer_norm
@pytest.mark.parametrize(
    "shape",
    ([(2, 40999)] if QUICK_MODE else [
        (200, 36),
        (4096, 100),
        # (1, 40999),
        (100, 40499),
        (4096, 256),
    ]),
)
@pytest.mark.parametrize("wb_none", [False, True])
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_layernorm(shape, dtype, wb_none):
    if flag_gems.vendor_name == "kunlunxin":
        torch.manual_seed(0)
        torch.cuda.manual_seed_all(0)
    if flag_gems.vendor_name == "spacemit":
        torch.manual_seed(42)
    res_inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    if wb_none:
        res_weight = None
        res_bias = None
    else:
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


@pytest.mark.rms_norm
@pytest.mark.parametrize("shape", REDUCTION_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_rmsnorm(shape, dtype):
    if flag_gems.vendor_name == "spacemit":
        torch.manual_seed(42)
    N = shape[1]
    layer_shape = [
        N,
    ]
    np.random.seed(0)
    np_inp = np.random.uniform(-0.1, 0.1, shape[:2]).astype(np.float32)
    np_grad = np.random.uniform(-0.01, 0.01, shape[:2]).astype(np.float32)
    np_weight = np.random.uniform(-0.1, 0.1, layer_shape).astype(np.float32)

    inp = torch.tensor(np_inp, dtype=dtype, device=flag_gems.device, requires_grad=True)
    weight = torch.tensor(np_weight, dtype=dtype, device=flag_gems.device, requires_grad=True)

    eps = 1e-5

    ref_inp = to_reference(inp)
    ref_weight = to_reference(weight)

    def _torch_rms_norm(x, weight, eps):
        upcast_x = x.to(torch.float32)
        variance = upcast_x.pow(2).mean(-1, keepdim=True)
        hidden_states = upcast_x * torch.rsqrt(variance + eps).to(torch.float32)
        hidden_states = hidden_states.to(x.dtype)
        return weight * hidden_states

    ref_out = _torch_rms_norm(ref_inp, weight=ref_weight, eps=eps)
    res_out = flag_gems.rms_norm(inp, list(layer_shape), weight=weight, eps=eps)

    res_grad = torch.tensor(np_grad, dtype=dtype, device=flag_gems.device, requires_grad=True)
    ref_grad = to_reference(res_grad)

    res_grad, res_weight_grad = torch.autograd.grad(res_out, (inp, weight), res_grad)
    ref_grad, ref_weight_grad = torch.autograd.grad(ref_out, (ref_inp, ref_weight), ref_grad)

    gems_assert_close(res_out, ref_out, dtype)
    if flag_gems.vendor_name == "kunlunxin" and shape == (200, 40999, 3):
        pytest.skip("wait for backward support")
    gems_assert_close(res_grad, ref_grad, dtype)
    gems_assert_close(res_weight_grad, ref_weight_grad, dtype, reduce_dim=N)


@pytest.mark.skip_layer_norm
@pytest.mark.parametrize("shape", REDUCTION_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_skip_layernorm(shape, dtype):
    if flag_gems.vendor_name == "spacemit":
        torch.manual_seed(42)
    N = shape[1]
    layer_shape = [
        N,
    ]
    inp = torch.randn(shape[:2], dtype=dtype, device=flag_gems.device)
    residual = torch.randn(shape[:2], dtype=dtype, device=flag_gems.device)
    weight = torch.randn(layer_shape, dtype=dtype, device=flag_gems.device)
    bias = torch.randn(layer_shape, dtype=dtype, device=flag_gems.device)
    eps = 1e-5

    ref_inp = to_reference(inp, True)
    ref_residual = to_reference(residual, True)
    ref_weight = to_reference(weight, True)
    ref_bias = to_reference(bias, True)

    ref_out = torch.layer_norm(
        ref_inp + ref_residual,
        list(layer_shape),
        weight=ref_weight,
        bias=ref_bias,
        eps=eps,
    )
    res_out = flag_gems.skip_layer_norm(inp, residual, list(layer_shape), weight=weight, bias=bias, eps=eps)

    gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.skip_rms_norm
@pytest.mark.parametrize("shape", REDUCTION_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_skip_rmsnorm(shape, dtype):
    if flag_gems.vendor_name == "spacemit":
        torch.manual_seed(42)
    N = shape[1]
    layer_shape = [
        N,
    ]
    inp = torch.randn(shape[:2], dtype=dtype, device=flag_gems.device)
    residual = torch.randn(shape[:2], dtype=dtype, device=flag_gems.device)
    weight = torch.randn(layer_shape, dtype=dtype, device=flag_gems.device)
    eps = 1e-5

    ref_inp = to_reference(inp, True)
    ref_residual = to_reference(residual, True)
    ref_weight = to_reference(weight, True)

    def _torch_rms_norm(x, residual, weight, eps):
        x = x + residual
        variance = x.pow(2).mean(-1, keepdim=True)
        hidden_states = x * torch.rsqrt(variance + eps)
        return weight * hidden_states

    ref_out = _torch_rms_norm(
        ref_inp,
        ref_residual,
        weight=ref_weight,
        eps=eps,
    )

    res_out = flag_gems.skip_rms_norm(inp, residual, list(layer_shape), weight=weight, eps=eps)

    gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.batch_norm
@pytest.mark.parametrize(
    "shape",
    [
        (16, 3),
        (32, 32, 32),
        (8, 32, 224, 224),
        (2050, 16, 32, 32),
        (8, 16, 3, 224, 224),
    ],
)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
@pytest.mark.parametrize("affine", [True, False])
def test_accuracy_batch_norm(shape, dtype, affine):
    if flag_gems.vendor_name == "cambricon":
        torch.manual_seed(23)
        torch.mlu.manual_seed_all(23)
    if flag_gems.vendor_name == "spacemit":
        torch.manual_seed(42)
    C = shape[1]
    inp = torch.randn(size=shape, dtype=dtype, device=flag_gems.device)
    weight = (torch.randn(size=(C, ), dtype=dtype, device=flag_gems.device) if affine else None)
    bias = (torch.randn(size=(C, ), dtype=dtype, device=flag_gems.device) if affine else None)

    running_mean = torch.zeros(size=(C, ), dtype=dtype, device=flag_gems.device)
    running_var = torch.ones(size=(C, ), dtype=dtype, device=flag_gems.device)

    eps = 1e-5

    ref_inp = to_reference(inp, True)
    ref_weight = to_reference(weight, True)
    ref_bias = to_reference(bias, True)
    ref_running_mean = to_reference(running_mean, True)
    ref_running_var = to_reference(running_var, True)

    ref_out = torch.nn.functional.batch_norm(
        ref_inp,
        ref_running_mean,
        ref_running_var,
        weight=ref_weight,
        bias=ref_bias,
        eps=eps,
    )

    with flag_gems.use_gems():
        res_out = torch.nn.functional.batch_norm(
            inp,
            running_mean,
            running_var,
            weight=weight,
            bias=bias,
            eps=eps,
        )

    gems_assert_close(res_out, ref_out, dtype)
    gems_assert_close(running_mean, ref_running_mean, dtype)
    gems_assert_close(running_var, ref_running_var, dtype)
