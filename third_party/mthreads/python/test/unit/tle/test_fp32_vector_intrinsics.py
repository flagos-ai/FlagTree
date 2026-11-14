"""Standard componentwise FP32 FMA/max vectorization, no relaxed flags."""
import pytest
import triton
import triton.language as tl
from test_tle_utils import compile_musa, mthreads_backend, require_mthreads_libtriton

require_mthreads_libtriton()


@triton.jit
def _vector(x, y, z, out, OP: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.arange(0, BLOCK)
    a, b, c = tl.load(x + i), tl.load(y + i), tl.load(z + i)
    if OP == 0:
        result = tl.fma(a, b, c)
    else:
        result = tl.maximum(a, b)
    tl.store(out + i, result)


@pytest.mark.parametrize("op,symbol", [(0, "llvm.fma.v4f32"), (1, "llvm.maxnum.v4f32")])
def test_fp32_vector_intrinsic(op, symbol):
    k = compile_musa(
        _vector, {"x": "*fp32", "y": "*fp32", "z": "*fp32", "out": "*fp32", "OP": "constexpr", "BLOCK": "constexpr"},
        {"OP": op, "BLOCK": 1024}, options={"num_warps": 4, "fp32_vector_width": 4, "enable_fp_fusion": False})
    assert symbol in k.asm["llir"]
    assert " fast " not in k.asm["llir"] and " reassoc " not in k.asm["llir"]


def test_short_fragment_and_default():
    k = compile_musa(
        _vector, {"x": "*fp32", "y": "*fp32", "z": "*fp32", "out": "*fp32", "OP": "constexpr", "BLOCK": "constexpr"},
        {"OP": 0, "BLOCK": 128}, options={"num_warps": 4, "fp32_vector_width": 4})
    assert "llvm.fma.v4f32" not in k.asm["llir"]
    _, backend = mthreads_backend()
    assert backend.parse_options({}).fp32_vector_width == 1
    assert backend.parse_options({}).hash() != backend.parse_options({"fp32_vector_width": 4}).hash()


@pytest.mark.parametrize("width", [0, 2, True, 4.0, "4"])
def test_invalid_fp32_width(width):
    _, backend = mthreads_backend()
    with pytest.raises(ValueError, match="fp32_vector_width must be 1 or 4"):
        backend.parse_options({"fp32_vector_width": width})


@triton.jit
def _cast(x, out, BLOCK: tl.constexpr):
    i = tl.arange(0, BLOCK)
    tl.store(out + i, tl.load(x + i).to(tl.float32))


@pytest.mark.parametrize("width", [1, 4])
def test_signed_i32_conversion(width):
    k = compile_musa(_cast, {"x": "*i32", "out": "*fp32", "BLOCK": "constexpr"}, {"BLOCK": 1024},
                     options={"num_warps": 4, "fp32_vector_width": width})
    assert ("sitofp <4 x i32>" in k.asm["llir"]) == (width == 4)
