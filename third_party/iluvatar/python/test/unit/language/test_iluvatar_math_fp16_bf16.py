"""
Tests for tl.math ops with fp16 and bf16 input dtypes on Iluvatar GPU.

Support matrix
--------------
The Iluvatar SFU has native fp32 transcendental instructions only.  There are
no packed fp16/bf16 variants for exp, log, sin, cos, sqrt, or rsqrt.

  fp16 ops: the backend promotes fp16→fp32, applies the op, truncates to fp16.
            Supported: exp, exp2, log, log2, sin, cos, sqrt, rsqrt, floor, ceil

  bf16 ops: the promote-to-fp32 path for transcendentals is not wired up in
            the Iluvatar backend (C++ assertion, pending future fix).
            Supported: floor, ceil
            Not yet supported: exp, exp2, log, log2, sin, cos, sqrt, rsqrt

These tests verify:
  1. No Python-level _check_dtype error is raised for supported (dtype, op) pairs.
  2. The Triton result is numerically close to the fp32 reference cast to the
     target dtype.
"""

import math

import pytest
import torch

import triton
import triton.language as tl

SIZE = 1024  # power of 2; fits in one CTA with num_warps=4 (256 threads)

# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _run_unary_math_kernel(op_str: str, x: torch.Tensor) -> torch.Tensor:
    """Compile and run a kernel that applies a single tl.math op."""

    @triton.jit
    def kernel(X, Z, N: tl.constexpr):
        off = tl.arange(0, N)
        x = tl.load(X + off)
        z = REPLACE_OP
        tl.store(Z + off, z)

    # Patch the placeholder at Python level (same pattern as test_iluvatar_bf16.py)
    patched = triton.JITFunction(kernel.fn)
    patched._unsafe_update_src(patched.src.replace("REPLACE_OP", op_str))

    z = torch.empty_like(x)
    patched[(1, )](x, z, N=SIZE, num_warps=4)
    return z


def _fp32_reference(op_str: str, x: torch.Tensor, out_dtype: torch.dtype) -> torch.Tensor:
    """Compute reference: run the op in fp32, then cast to out_dtype."""
    x32 = x.float()
    op_map = {
        "tl.math.exp(x)": torch.exp,
        "tl.math.exp2(x)": torch.exp2,
        "tl.math.log(x)": torch.log,
        "tl.math.log2(x)": torch.log2,
        "tl.math.sin(x)": torch.sin,
        "tl.math.cos(x)": torch.cos,
        "tl.math.sqrt(x)": torch.sqrt,
        "tl.math.rsqrt(x)": lambda t: 1.0 / torch.sqrt(t),
        "tl.math.floor(x)": torch.floor,
        "tl.math.ceil(x)": torch.ceil,
    }
    return op_map[op_str](x32).to(out_dtype)


# ---------------------------------------------------------------------------
# Input helpers
# ---------------------------------------------------------------------------

_POSITIVE_OPS = {"tl.math.log(x)", "tl.math.log2(x)", "tl.math.sqrt(x)", "tl.math.rsqrt(x)"}
_TRIG_OPS = {"tl.math.sin(x)", "tl.math.cos(x)"}
_ROUND_OPS = {"tl.math.floor(x)", "tl.math.ceil(x)"}


def _make_input(op_str: str, dtype: torch.dtype) -> torch.Tensor:
    torch.manual_seed(42)
    if op_str in _POSITIVE_OPS:
        # log/sqrt: keep x in (0.5, 2.0) to avoid underflow in result
        x32 = torch.rand(SIZE, device="cuda") * 1.5 + 0.5
    elif op_str in _TRIG_OPS:
        x32 = torch.rand(SIZE, device="cuda") * 2 * math.pi - math.pi
    elif op_str in _ROUND_OPS:
        # small integers: rounding is exact in fp16/bf16
        x32 = torch.rand(SIZE, device="cuda") * 16 - 8
    else:
        # exp/exp2: avoid fp16 overflow (exp overflows at ~11.09 in fp16)
        x32 = torch.rand(SIZE, device="cuda") * 8 - 4
    return x32.to(dtype)


# tolerances: bf16 has 7-bit mantissa vs fp16's 10-bit → slightly looser
_RTOL = {torch.float16: 1e-2, torch.bfloat16: 2e-2}
_ATOL = {torch.float16: 1e-3, torch.bfloat16: 2e-3}

# ---------------------------------------------------------------------------
# Test 1: fp16 — all ten ops are supported
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("op_str", [
    "tl.math.exp(x)",
    "tl.math.exp2(x)",
    "tl.math.log(x)",
    "tl.math.log2(x)",
    "tl.math.sin(x)",
    "tl.math.cos(x)",
    "tl.math.sqrt(x)",
    "tl.math.rsqrt(x)",
    "tl.math.floor(x)",
    "tl.math.ceil(x)",
])
def test_math_op_fp16(op_str: str, device: str = "cuda"):
    """
    fp16 input: backend promotes fp16→fp32, applies op, truncates back to fp16.
    All ten tl.math ops are supported.
    """
    dtype = torch.float16
    x = _make_input(op_str, dtype)
    z_tri = _run_unary_math_kernel(op_str, x)
    z_ref = _fp32_reference(op_str, x, dtype)
    torch.testing.assert_close(z_tri, z_ref, rtol=_RTOL[dtype], atol=_ATOL[dtype])


# ---------------------------------------------------------------------------
# Test 2: bf16 — only floor and ceil are supported
#         (exp/log/sin/cos/sqrt/rsqrt bf16 pending backend fix)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("op_str", [
    "tl.math.floor(x)",
    "tl.math.ceil(x)",
])
def test_math_op_bf16(op_str: str, device: str = "cuda"):
    """
    bf16 input: only floor and ceil are supported.
    Transcendental ops (exp, log, sin, cos, sqrt, rsqrt) hit a C++ assertion in
    the Iluvatar backend's bf16 promote-to-fp32 path and are excluded here.
    """
    dtype = torch.bfloat16
    x = _make_input(op_str, dtype)
    z_tri = _run_unary_math_kernel(op_str, x)
    z_ref = _fp32_reference(op_str, x, dtype)
    torch.testing.assert_close(z_tri, z_ref, rtol=_RTOL[dtype], atol=_ATOL[dtype])
