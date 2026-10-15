import numpy as np
import pytest

import triton
import triton.language as tl
from triton._C.libtriton import ir as _ir
from triton.runtime.interpreter import _convert_float, _convert_float_legacy

# The interpreter converts floats in numpy, so these checks need no device. They pin the
# rewritten conversion (strict RTNE/RTZ, per-format fp8 special values) and its
# FLAGTREE_LOW_PRECISION_FLOAT=0 fallback to the upstream routine kept as _convert_float_legacy.

RTNE = _ir.ROUNDING_MODE.RTNE
RTZ = _ir.ROUNDING_MODE.RTZ


def _bits(vals, src_dtype, dst_dtype, rounding=RTNE, fn=_convert_float):
    out = fn(np.asarray(vals, dtype=np.float32), src_dtype, dst_dtype, rounding)
    return [int(b) for b in out.ravel()]


# Inputs where the two implementations differ, i.e. the upstream bugs the rewrite fixed:
# (input, src, dst, rounding, expected bits, expected legacy bits)
_DIVERGING_CASES = [
    # 1.0 + 2^-11 is halfway between fp16 0x3c00 and 0x3c01: RTNE keeps the even neighbour,
    # the legacy routine rounds half up
    pytest.param(1.0 + 2.0**-11, tl.float32, tl.float16, RTNE, 0x3C00, 0x3C01, id="rtne-tie-fp32-fp16"),
    # e4m3fn NaN is the all-ones code 0x7f; the legacy routine emits 0x7c, a finite value there
    pytest.param(np.nan, tl.float32, tl.float8e4nv, RTNE, 0x7F, 0x7C, id="nan-fp32-fp8e4nv"),
    # e4m3fn has no inf: overflow saturates to max finite 0x7e; the legacy routine lands on the NaN code
    pytest.param(1e6, tl.float32, tl.float8e4nv, RTNE, 0x7E, 0x7F, id="satfinite-fp32-fp8e4nv"),
]


@pytest.mark.parametrize("value, src, dst, rounding, new_bits, legacy_bits", _DIVERGING_CASES)
def test_convert_float_paths_diverge(value, src, dst, rounding, new_bits, legacy_bits):
    assert _bits([value], src, dst, rounding, fn=_convert_float) == [new_bits]
    assert _bits([value], src, dst, rounding, fn=_convert_float_legacy) == [legacy_bits]


@pytest.mark.parametrize("value, src, dst, rounding, new_bits, legacy_bits", _DIVERGING_CASES)
def test_convert_float_knob_selects_legacy(fresh_knobs, value, src, dst, rounding, new_bits, legacy_bits):
    # the knob defaults on; scope() restores both the override and the env var it propagates
    assert fresh_knobs.language.low_precision_float
    assert _bits([value], src, dst, rounding) == [new_bits]

    with fresh_knobs.language.scope():
        fresh_knobs.language.low_precision_float = False
        assert _bits([value], src, dst, rounding) == [legacy_bits]

    assert fresh_knobs.language.low_precision_float
    assert _bits([value], src, dst, rounding) == [new_bits]


def test_convert_float_knob_reads_env(fresh_knobs, monkeypatch):
    monkeypatch.setenv("FLAGTREE_LOW_PRECISION_FLOAT", "0")
    triton.knobs.refresh_knobs()
    assert not fresh_knobs.language.low_precision_float
    value, src, dst, rounding, _, legacy_bits = _DIVERGING_CASES[0].values
    assert _bits([value], src, dst, rounding) == [legacy_bits]


def test_convert_float_rtz_never_rounds_to_inf():
    # RTZ truncates toward zero: fp32 max clamps to fp16 max finite instead of overflowing to inf
    assert _bits([np.finfo(np.float32).max], tl.float32, tl.float16, RTZ) == [0x7BFF]
    assert _bits([-np.finfo(np.float32).max], tl.float32, tl.float16, RTZ) == [0xFBFF]


@pytest.mark.parametrize("dst", [tl.float8e4nv, tl.float8e5, tl.float8e4b8, tl.float8e5b16, tl.float8e4b15])
def test_convert_float_fp8_roundtrip_exact(dst):
    # every fp8 code decodes to fp32 and re-encodes to itself bit-exactly, except NaN codes,
    # which all collapse to the format's canonical NaN
    codes = np.arange(256, dtype=np.uint8)
    up = _convert_float(codes, dst, tl.float32, None).view(np.float32)
    down = _convert_float(up, tl.float32, dst, RTNE)
    is_nan = np.isnan(up)
    assert np.array_equal(down[~is_nan], codes[~is_nan])
    if is_nan.any():
        canonical = _convert_float(np.array([np.nan], dtype=np.float32), tl.float32, dst, RTNE)[0]
        assert np.all(down[is_nan] == canonical)
