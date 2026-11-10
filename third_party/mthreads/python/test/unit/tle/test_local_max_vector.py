"""Opt-in local maxnum tree; ordinary tl.max and sum semantics are isolated."""
import pytest
import triton
import triton.language as tl
import triton.experimental.tle.language as tle
from test_tle_utils import compile_musa, mthreads_backend, require_mthreads_libtriton

require_mthreads_libtriton()


@triton.jit
def _reduce(x, out, SUM: tl.constexpr, L: tl.constexpr):
    i = tl.arange(0, 32)[:, None] * 128 + tl.arange(0, 128)[None, :]
    values = tle.gpu.set_layout(tl.load(x + i), L)
    if SUM:
        result = tl.sum(values, 1)
    else:
        result = tl.max(values, 1)
    tl.store(out + tl.arange(0, 32), result)


@pytest.mark.parametrize("width", [1, 4])
def test_local_max_lowering(width):
    k = compile_musa(_reduce, {"x": "*fp32", "out": "*fp32", "SUM": "constexpr", "L": "constexpr"},
                     {"SUM": False, "L": tle.gpu.BlockEncoding([1, 16], [4, 8], [4, 1], [1, 0])},
                     options={"num_warps": 4, "local_max_vector_width": width})
    assert ("llvm.maxnum.v4f32" in k.asm["llir"]) == (width == 4)


def test_local_sum_unchanged():
    args = (_reduce, {"x": "*fp32", "out": "*fp32", "SUM": "constexpr", "L":
                      "constexpr"}, {"SUM": True, "L": tle.gpu.BlockEncoding([1, 16], [4, 8], [4, 1], [1, 0])})
    a = compile_musa(*args, options={"num_warps": 4, "local_max_vector_width": 1})
    b = compile_musa(*args, options={"num_warps": 4, "local_max_vector_width": 4})
    assert a.asm["llir"] == b.asm["llir"]


@triton.jit
def _sum_linear(x, out, L: tl.constexpr):
    i = tl.arange(0, 64)[:, None] * 128 + tl.arange(0, 128)[None, :]
    values = tle.gpu.set_layout(tl.load(x + i), L)
    four = tl.sum(values.reshape(64, 4, 4, 8), 1)
    result = tl.sum(tl.sum(four, 1), 1)
    tl.store(out + tl.arange(0, 64), result)


@pytest.mark.parametrize("width", [1, 4])
def test_independent_linear_sum_chains(width):
    k = compile_musa(_sum_linear, {"x": "*fp32", "out": "*fp32", "L": "constexpr"},
                     {"L": tle.gpu.mthreads.MusaSqmmaEncoding([3, 1], [8, 1], [32, 128, 128])},
                     options={"num_warps": 8, "fp32_vector_width": width, "enable_fp_fusion": False})
    assert ("fadd <4 x float>" in k.asm["llir"]) == (width == 4)
    assert " reassoc " not in k.asm["llir"]


@pytest.mark.parametrize("width", [0, 2, True, 4.0, "4"])
def test_invalid_max_width(width):
    _, backend = mthreads_backend()
    with pytest.raises(ValueError, match="local_max_vector_width must be 1 or 4"):
        backend.parse_options({"local_max_vector_width": width})
