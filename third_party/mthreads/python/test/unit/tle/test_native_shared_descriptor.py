"""Opt-in MP31 SDK shared descriptors. Compilation is not hardware validation."""

import pytest
import triton
import triton.language as tl
import triton.experimental.tle.language as tle
from test_tle_utils import compile_musa, mthreads_backend, require_mthreads_libtriton

require_mthreads_libtriton()


@triton.jit
def _dot(q, k, out, L: tl.constexpr, GRAN: tl.constexpr, PHASE: tl.constexpr, FP8: tl.constexpr):
    dtype: tl.constexpr = tl.float8e4nv if FP8 else tl.int8
    outtype: tl.constexpr = tl.float32 if FP8 else tl.int32
    shared: tl.constexpr = tle.gpu.swizzled_shared_layout(
        GRAN, PHASE, 256 // GRAN, [1, 0], [1, 1], [1, 1], [1, 1], [1, 0]
    )
    a = tle.gpu.alloc((32, 128), dtype, layout=shared)
    b = tle.gpu.alloc((128, 128), dtype, layout=shared)
    i, j = tl.arange(0, 32), tl.arange(0, 128)
    tle.gpu.copy(q + i[:, None] * 128 + j[None, :], a, (32, 128))
    tle.gpu.copy(k, b, (128, 128), (0, 0))
    zero = tle.gpu.set_layout(tl.zeros((32, 128), outtype), L)
    pending = tle.gpu.wgmma(a, b, acc=zero, trans_b=True, out_dtype=outtype)
    result = tle.gpu.set_layout(tle.gpu.wgmma_wait(0, pending), L)
    # Isolate descriptor lowering from implicit MMA-to-store propagation.
    # Both sides of the store have the same explicit representation.
    output_ptr = tle.gpu.set_layout(out + i[:, None] * 128 + j[None, :], L)
    tl.store(output_ptr, result)


def compile_descriptor(gran, phase=2, fp8=True, abi="native_mp31_shared"):
    dtype = "fp8e4nv" if fp8 else "i8"
    return compile_musa(
        _dot,
        {
            "q": "*" + dtype,
            "k": f"tensordesc<{dtype}[128,128]>",
            "out": "*fp32" if fp8 else "*i32",
            "L": "constexpr",
            "GRAN": "constexpr",
            "PHASE": "constexpr",
            "FP8": "constexpr",
        },
        {
            "L": tle.gpu.mthreads.MusaSqmmaEncoding([3, 1], [4, 1], [32, 128, 128]),
            "GRAN": gran,
            "PHASE": phase,
            "FP8": fp8,
        },
        options={"num_warps": 4, "sqmma_descriptor_abi": abi},
    )


@pytest.mark.parametrize("gran,code", [(16, 1), (32, 2), (64, 3)])
@pytest.mark.parametrize("fp8", [True])
def test_native_shared_codes_are_emitted_without_scalar_descriptor_load(gran, code, fp8):
    kernel = compile_descriptor(gran, fp8=fp8)
    text = kernel.asm["llir"]
    assert "llvm.musa.sqmma.desc." not in text
    assert str((5 << 18) | (code << 21)) in text
    assert kernel.module is None and kernel.function is None


def test_legacy_sg16_mode_does_not_silently_accept_sg64(capfd):
    # MLIR's Python boundary raises a generic pass-failure exception; the
    # target's semantic diagnostic is emitted on the native stderr fd.
    with pytest.raises(Exception):
        compile_descriptor(64, abi="native_mp31_sg16")
    assert "SG16, stride-3 swizzle" in capfd.readouterr().err


def test_noncanonical_tme_line_is_rejected_before_native_descriptor(capfd):
    with pytest.raises(Exception):
        compile_descriptor(64, phase=1)
    assert "unsupported TLE TME physical shared layout or block shape" in capfd.readouterr().err


def test_shared_mode_is_opt_in_with_a_distinct_cache_key():
    _, backend = mthreads_backend()
    default = backend.parse_options({})
    native = backend.parse_options({"sqmma_descriptor_abi": "native_mp31_shared"})
    assert default.sqmma_descriptor_abi == "pointer"
    assert default.hash() != native.hash()


@triton.jit
def _invalid_native_alias(out, START: tl.constexpr, LENGTH: tl.constexpr):
    shared: tl.constexpr = tle.gpu.swizzled_shared_layout(16, 2, 16, [1, 0], [1, 1], [1, 1], [1, 1], [1, 0])
    root = tle.gpu.alloc((32, 128), tl.float8e4nv, layout=shared)
    # The frontend must reject an invalid alias before descriptor lowering.
    alias = root.subslice(START, LENGTH, dim=1)
    tl.store(out, alias.shape[1])


@pytest.mark.parametrize("start,length", [(-1, 128), (1, 128), (128, 1), (0, 0), (0, 256)])
def test_native_descriptor_input_alias_rejects_out_of_bounds(start, length):
    with pytest.raises(Exception, match="out of range"):
        compile_musa(
            _invalid_native_alias,
            {"out": "*i32", "START": "constexpr", "LENGTH": "constexpr"},
            {"START": start, "LENGTH": length},
            options={"num_warps": 4, "sqmma_descriptor_abi": "native_mp31_shared"},
        )


def test_native_descriptor_rejects_unknown_abi():
    _, backend = mthreads_backend()
    with pytest.raises(ValueError, match="sqmma_descriptor_abi"):
        backend.parse_options({"sqmma_descriptor_abi": "unknown"})
