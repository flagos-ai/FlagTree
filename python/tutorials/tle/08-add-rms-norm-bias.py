"""
Copyright 2026- Xcoresigma Technology Co., Ltd

AddRmsNormBias
==============

The computation of AddRmsNormBias is as follows:

.. math::
    y = \\frac{x1_i + x2_i}{\\sqrt{\\frac{1}{n} * \\Sigma{(x1_i + x2_i)^2 + \\epsilon}}} * \\gamma + \\beta

where :math:`x1_i` and `x2_i` are the input tensor, :math:`y` is the output tensor,
:math:`n` is the number of elements in :math:`x`,
:math:`eps` is a small constant to avoid division by zero,

We compare the performance of the triton implementation with the ascendc implementation which in vllm-ascend(>=0.17), so requires vllm-ascend installation. Also, requires pytest and pytest-xdist to be installed.

Run the tests:
pytest -s -k "accuracy" -v 06-add-rms-norm-bias.py # only run accuracy test
pytest -s -k "benchmark" -v 06-add-rms-norm-bias.py # only run benchmark test

"""

import pytest
import math
from typing import Optional

import torch
import triton
import triton.language as tl
from triton.backends.ascend.testing import do_bench_npu
import triton.experimental.tle as tle

enable_vllm = True
try:
    # vllm and vllm_ascend requires >= 0.17.0
    import vllm  # noqa: F401
    if vllm.__version__ < "0.17.0":
        enable_vllm = False

except ImportError:
    print("vllm does not enabled.")
    enable_vllm = False


@triton.heuristics({
    "HAS_BIAS": lambda args: args["B"] is not None,
})
@triton.jit(do_not_specialize=["eps"])
def add_rms_norm_bias_kernel(
    X,  # pointer to the input
    R,  # pointer to the residual
    W,  # pointer to the weight
    B,  # pointer to the bias
    MAX_ROWS,  # maximum number of rows to process
    eps,  # epsilon to avoid division by zero
    N_ROWS,  # number of rows to process in one core
    NUM_COLUMNS: tl.constexpr,  # number of columns in X
    HAS_BIAS: tl.constexpr,
):
    pid = tl.program_id(0)
    X += pid * NUM_COLUMNS * N_ROWS
    R += pid * NUM_COLUMNS * N_ROWS

    # preload weights and bias if exists
    _var_base = tl.zeros([NUM_COLUMNS], dtype=tl.float32)
    col_off = tl.arange(0, NUM_COLUMNS)
    w = tl.load(W + col_off)
    if HAS_BIAS:
        b = tl.load(B + col_off)

    base_row = pid * N_ROWS
    rows = min(base_row + N_ROWS, MAX_ROWS) - base_row
    for row_off in tle.dsa.parallel(0, rows, 1):
        cols = row_off * NUM_COLUMNS + col_off
        x = tl.load(X + cols)
        r = tl.load(R + cols)
        x = (x + r).to(tl.float32)
        _var_base = (x * x) / NUM_COLUMNS
        var = tl.sum(_var_base)

        rrms = 1 / tl.sqrt(var + eps)

        y = (x * rrms * w).to(X.dtype.element_ty)

        if HAS_BIAS:
            y = y + b
        # write back to residual and input
        tl.store(R + cols, x)
        tl.store(X + cols, y)


def add_rms_norm_bias(x, residual, weight, bias: Optional[torch.Tensor] = None, eps=1e-5):
    """
    This function performs fused residual addition and RMS normalization **in-place**.
    Both `x` and `residual` tensors will be modified. Use with caution if these tensors
    are reused elsewhere or require gradients.

    Args:
        x: Input tensor
        residual: Residual tensor to add
        weight: RMSNorm weight tensor
        bias: Optional bias tensor, shape should match normalized_shape
        eps: Epsilon for numerical stability
    """
    normalized_shape = weight.shape
    dim = x.ndim - len(normalized_shape)
    M = min(math.prod(x.shape[:dim]), 65535)
    N = math.prod(normalized_shape)

    x = x.contiguous()
    residual = residual.contiguous()
    weight = weight.contiguous()

    def get_core_num():
        try:
            import torch_npu  # noqa: F401
            current_device = torch.npu.current_device()
            torch.npu.set_device(current_device)
            cores_dict = torch.npu.get_device_limit(current_device)
            return cores_dict["vector_core_num"]
        except (AttributeError, KeyError, TypeError):
            return None

    CORES = 24 if get_core_num() is None else get_core_num()

    def get_rows_and_cores(M):
        if M <= CORES:
            return 1, M

        num_rows = (M + CORES - 1) // CORES
        use_cores = (M + num_rows - 1) // num_rows
        return num_rows, use_cores

    NUM_ROWS, USE_CORES = get_rows_and_cores(M)

    # Validate bias if provided
    if bias is not None:
        assert bias.shape == tuple(normalized_shape), (
            f"bias shape {bias.shape} must match normalized_shape {normalized_shape}")
        assert bias.stride(-1) == 1, "bias must be contiguous in the last dimension"
        bias = bias.contiguous()
    else:
        # bias is None that won't be used when HAS_BIAS=False
        pass

    add_rms_norm_bias_kernel[
        USE_CORES,
    ](x, residual, weight, bias, M, eps=eps, N_ROWS=NUM_ROWS, NUM_COLUMNS=N, multibuffer=True,
      limit_auto_multi_buffer_only_for_local_buffer=False, limit_auto_multi_buffer_of_local_buffer="no-limit")
    return x, residual


def _torch_fused_add_rms_norm_with_bias(x, residual, weight, bias, eps):
    x = x + residual
    variance = x.pow(2).mean(-1, keepdim=True)
    hidden_states = x * torch.rsqrt(variance + eps)
    return weight * hidden_states + bias, x


if enable_vllm:

    def _ascendc_add_rms_norm_bias(x, residual, weight, bias, eps):
        try:
            # vllm and vllm_ascend requires >= 0.17.0
            import vllm  # noqa: F401
            if vllm.__version__ < "0.17.0":
                raise Exception("vllm version must be >= 0.17.0")

            import vllm_ascend  # noqa: F401
            from vllm_ascend.utils import enable_custom_op
            from vllm_ascend.ops.layernorm import AscendRMSNorm  # noqa: F401

            if not enable_custom_op():
                raise Exception("enable custom op failed")
        except ImportError:
            raise ImportError("vllm-ascend does not exist in the current environment.")

        output, _, x = torch.ops._C_ascend.npu_add_rms_norm_bias(x, residual, weight, bias, eps)
        return output, x


# ==============================================================================
# utils, reference from flaggems
# ==============================================================================
RESOLUTION = {
    torch.bool: 0,
    torch.uint8: 0,
    torch.int8: 0,
    torch.int16: 0,
    torch.int32: 0,
    torch.int64: 0,
    torch.float8_e4m3fn: 1e-3,
    torch.float8_e5m2: 1e-3,
    torch.float8_e4m3fnuz: 1e-3,
    torch.float8_e5m2fnuz: 1e-3,
    torch.float16: 1e-3,
    torch.float32: 1.3e-6,
    torch.bfloat16: 0.016,
    torch.float64: 1e-7,
    torch.complex32: 1e-3,
    torch.complex64: 1.3e-6,
}


def tk_assert_close(res, ref, dtype, equal_nan=False, reduce_dim=1, atol=1e-4) -> None:
    if dtype is None:
        dtype = torch.float32
    assert res.dtype == dtype
    ref = ref.to(dtype)
    rtol = RESOLUTION[dtype]
    torch.testing.assert_close(res, ref, atol=atol * reduce_dim, rtol=rtol, equal_nan=equal_nan)


#===============================================================================
# Accuracy Test
#===============================================================================

# FLOAT_DTYPES = [torch.float16, torch.float32, torch.bfloat16]
FLOAT_DTYPES = [torch.float16, torch.float32]

capture_shapes = [
    ((4, 4096), (4, 4096), 4096),
    ((8, 4096), (8, 4096), 4096),
    ((24, 4096), (24, 4096), 4096),
    ((8096, 4096), (8096, 4096), 4096),
    ((16, 4096), (16, 4096), 4096),
    ((484, 4096), (484, 4096), 4096),
    ((2544, 4096), (2544, 4096), 4096),
    ((20, 4096), (20, 4096), 4096),
    ((12, 4096), (12, 4096), 4096),
]


@pytest.mark.parametrize("input_shape, residual_shape, weight_shape", capture_shapes)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_add_rms_norm_with_bias(input_shape, residual_shape, weight_shape, dtype):
    device = "npu"
    inp = torch.randn(input_shape, dtype=dtype, device=device)
    residual = torch.randn(residual_shape, dtype=dtype, device=device)
    weight = torch.randn(weight_shape, dtype=dtype, device=device)
    bias = torch.randn(weight_shape, dtype=dtype, device=device)
    eps = 1e-5

    if enable_vllm:
        ref_ascendc_out, ref_ascendc_residual = _ascendc_add_rms_norm_bias(
            inp,
            residual,
            weight=weight,
            bias=bias,
            eps=eps,
        )

    ref_torch_out, ref_torch_residual = _torch_fused_add_rms_norm_with_bias(
        inp,
        residual,
        weight=weight,
        bias=bias,
        eps=eps,
    )

    res_triton_out, res_triton_residual = add_rms_norm_bias(inp, residual, weight=weight, bias=bias, eps=eps)

    # both ascendc kernel and triton kernel on npu should be close to torch kernel, but absolute tolerance should be larger on float16
    tk_assert_close(res_triton_out, ref_torch_out, dtype, reduce_dim=100)
    if enable_vllm:
        tk_assert_close(ref_ascendc_out, ref_torch_out, dtype, reduce_dim=100)
        tk_assert_close(res_triton_residual, ref_ascendc_residual, dtype)


# ==============================================================================
# Benchmark Test
# ==============================================================================
@pytest.mark.parametrize("input_shape, residual_shape, weight_shape", capture_shapes)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_benchmark_add_rms_norm_with_bias(input_shape, residual_shape, weight_shape, dtype):
    device = "npu"
    inp = torch.randn(input_shape, dtype=dtype, device=device)
    residual = torch.randn(residual_shape, dtype=dtype, device=device)
    weight = torch.randn(weight_shape, dtype=dtype, device=device)
    bias = torch.randn(weight_shape, dtype=dtype, device=device)
    eps = 1e-5

    if enable_vllm:

        def _torch_op():
            _, _ = _ascendc_add_rms_norm_bias(inp, residual, weight, bias, eps)

        torch_time = do_bench_npu(lambda: _torch_op(), clear_l2_cache=True, keep_res=False, collect_prof=False)
    else:

        def _torch_op():
            _, _ = _torch_fused_add_rms_norm_with_bias(inp, residual, weight, bias, eps)

        torch_time = do_bench_npu(lambda: _torch_op(), clear_l2_cache=True, keep_res=False, collect_prof=False)

    def _triton_op():
        _, _ = add_rms_norm_bias(inp, residual, weight, bias, eps)

    triton_time = do_bench_npu(lambda: _triton_op(), clear_l2_cache=True, keep_res=False, collect_prof=False)

    print("-" * 80)
    print("[do_bench_npu] | {:<20} | {:<20} |".format("torch_time", "triton_time"))
    print(f"[do_bench_npu] | {torch_time:<20} | {triton_time:<20} |")
    print("-" * 80)
