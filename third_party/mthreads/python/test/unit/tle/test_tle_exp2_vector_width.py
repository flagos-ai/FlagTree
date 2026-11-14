"""Per-kernel FP32 EXP2 vectorization, independent of attention or FP8 inputs."""
import pytest
import triton
import triton.language as tl
from test_tle_utils import compile_musa, mthreads_backend, require_mthreads_libtriton

require_mthreads_libtriton()


@triton.jit
def _exp2(x, out, BLOCK: tl.constexpr):
    i = tl.arange(0, BLOCK)
    tl.store(out + i, tl.exp2(tl.load(x + i)))


@pytest.mark.parametrize("width", [1, 2, 4])
def test_exp2_option_reaches_lowering(width):
    compiled = compile_musa(_exp2, {"x": "*fp32", "out": "*fp32", "BLOCK": "constexpr"}, {"BLOCK": 1024},
                            options={"num_warps": 4, "exp2_vector_width": width})
    ir = compiled.asm["llir"]
    assert compiled.metadata.exp2_vector_width == width
    if width == 1:
        assert "llvm.musa.exp2.f.bst" not in ir
    else:
        assert f"llvm.musa.exp2.f.bst{width}" in ir
    if width == 2:
        assert "llvm.musa.exp2.f.bst4" not in ir


def test_short_fragment_falls_back_to_pair():
    compiled = compile_musa(_exp2, {"x": "*fp32", "out": "*fp32", "BLOCK": "constexpr"}, {"BLOCK": 256},
                            options={"num_warps": 4, "exp2_vector_width": 4})
    assert "llvm.musa.exp2.f.bst2" in compiled.asm["llir"]
    assert "llvm.musa.exp2.f.bst4" not in compiled.asm["llir"]


@pytest.mark.parametrize("width", [0, 3, -1, True, 4.0, "4"])
def test_invalid_vector_width(width):
    _, backend = mthreads_backend()
    with pytest.raises(ValueError, match="exp2_vector_width must be 1, 2, or 4"):
        backend.parse_options({"exp2_vector_width": width})


def test_default_and_cache_identity():
    _, backend = mthreads_backend()
    assert backend.parse_options({}).exp2_vector_width == 1
    assert backend.parse_options({"exp2_vector_width": 2}).hash() != backend.parse_options({"exp2_vector_width":
                                                                                            4}).hash()


@pytest.mark.parametrize("width", [2, 4])
def test_native_exp2_has_no_subnormal_correction(width):
    compiled = compile_musa(_exp2, {"x": "*fp32", "out": "*fp32", "BLOCK": "constexpr"}, {"BLOCK": 1024},
                            options={"num_warps": 4, "exp2_vector_width": width, "exp2_native": True})
    ir = compiled.asm["llir"]
    assert f"llvm.musa.exp2.f.bst{width}" in ir
    assert "icmp ugt" not in ir
    assert "fmul" not in ir
    assert compiled.metadata.exp2_native


def test_native_options_cache_identity():
    _, backend = mthreads_backend()
    base = backend.parse_options({})
    assert not base.exp2_native
    assert base.hash() != backend.parse_options({"exp2_native": True, "exp2_vector_width": 2}).hash()
    with pytest.raises(ValueError, match="exp2_native requires"):
        backend.parse_options({"exp2_native": True, "exp2_vector_width": 1})
    with pytest.raises(ValueError, match="exp2_native must be a bool"):
        backend.parse_options({"exp2_native": 1})
