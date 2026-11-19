"""Target FP8 packing keeps ordinary casts, default behavior and rounding."""
import pytest
import triton
import triton.language as tl
from test_tle_utils import compile_musa, mthreads_backend, require_mthreads_libtriton

require_mthreads_libtriton()


@triton.jit
def _cast(x, out, BLOCK: tl.constexpr, E5: tl.constexpr, RTZ: tl.constexpr):
    i = tl.arange(0, BLOCK)
    value = tl.load(x + i)
    dtype: tl.constexpr = tl.float8e5 if E5 else tl.float8e4nv
    rounding: tl.constexpr = "rtz" if RTZ else "rtne"
    tl.store(out + i, value.to(dtype, fp_downcast_rounding=rounding).to(tl.int8, bitcast=True))


@pytest.mark.parametrize("enabled,e5,rtz,block", [(False, False, False, 1024), (True, False, False, 1024),
                                                  (True, True, False, 1024), (True, False, True, 1024),
                                                  (True, False, False, 128)])
def test_burst4_domain_and_fallback(enabled, e5, rtz, block):
    k = compile_musa(_cast, {"x": "*fp32", "out": "*i8", "BLOCK": "constexpr", "E5": "constexpr", "RTZ": "constexpr"},
                     {"BLOCK": block, "E5": e5, "RTZ": rtz},
                     options={"num_warps": 4, "enable_fp8_burst2": True, "fp8_downcast_burst4": enabled})
    assert ("CVT.RTE.DENORM.BST4" in k.asm["llir"]) == (enabled and not e5 and not rtz and block >= 512)


def test_burst4_is_explicit():
    _, backend = mthreads_backend()
    assert not backend.parse_options({}).fp8_downcast_burst4
    assert backend.parse_options({}).hash() != backend.parse_options({"fp8_downcast_burst4": True}).hash()
    with pytest.raises(ValueError, match="must be a bool"):
        backend.parse_options({"fp8_downcast_burst4": 1})
