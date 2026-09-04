# flagtree tle
"""TsingMicro TX8 vendor-specific DSA primitives.

Exposed as ``triton.experimental.tle.language.dsa.tsingmicro``, mirroring the
upstream per-backend namespace convention (see ``dsa.ascend``). The hardware
random-generation family and vendor address-space selectors live here; the
multi-vendor DSA API surface stays at the ``dsa`` root.
"""

import triton.language.core as tl
from triton.language.core import PropagateNan
from triton.language import math as tlmath

from ..types import scope

# Address-space selector for the on-chip scratchpad memory (vendor naming).
SPM = scope("tsingmicro.spm", "spm", "spm")

# Fmt_INT64 in tsingmicro tx81 Data_Format enum (see instr_def / op_def.h).
_DSA_RANDGEN_FMT_INT64 = 11
_DSA_RANDGEN_NUM_STREAMS = 16  # values per step (128 bytes)


@tl.builtin
def randgen(seed0, seed1, n_out: tl.constexpr, _semantic=None):
    """
    Hardware random-number generator (xorshift128+ peri) on TX81.

    Args:
        seed0: block tensor ``[16]`` of ``int64`` / ``uint64`` seeds (stream a).
        seed1: block tensor ``[16]`` of ``int64`` / ``uint64`` seeds (stream b).
        n_out: number of ``int64`` random outputs; must be a multiple of 16
            (hardware emits 16 values / 128 bytes per step).

    Returns:
        ``(out, seed0_out, seed1_out)`` with ``out`` shaped ``[n_out]``.
        ``seed0_out`` / ``seed1_out`` mirror the inputs: the peri does not
        expose state readback, so identical seeds yield identical blocks.
        Vary the seeds across calls to obtain different data.

    Notes:
        Output values are raw xorshift128+ ``uint64`` bit patterns stored as
        ``int64``. Convert to Uniform(0,1) / Normal yourself (see ``rand`` /
        ``randn`` helpers), or feed downstream kernels.
    """
    n_out = int(tl._unwrap_if_constexpr(n_out))
    if n_out <= 0 or (n_out % _DSA_RANDGEN_NUM_STREAMS) != 0:
        raise ValueError(f"tle.dsa.tsingmicro.randgen n_out must be a positive multiple of "
                         f"{_DSA_RANDGEN_NUM_STREAMS}, got {n_out}")

    builder = _semantic.builder
    if not hasattr(builder, "create_dsa_randgen"):
        raise RuntimeError("builder missing create_dsa_randgen for DSA randgen")

    if not isinstance(seed0, tl.tensor):
        seed0 = tl.to_tensor(seed0, _semantic=_semantic)
    if not isinstance(seed1, tl.tensor):
        seed1 = tl.to_tensor(seed1, _semantic=_semantic)

    if not seed0.dtype.is_int() or not seed1.dtype.is_int():
        raise ValueError("tle.dsa.tsingmicro.randgen seeds must be integer tensors")
    if seed0.dtype != tl.int64:
        seed0 = seed0.to(tl.int64, _semantic=_semantic)
    if seed1.dtype != tl.int64:
        seed1 = seed1.to(tl.int64, _semantic=_semantic)

    seed0_ty = seed0.type
    seed1_ty = seed1.type
    if (not seed0_ty.is_block() or not seed1_ty.is_block()
            or tuple(int(tl._unwrap_if_constexpr(d)) for d in seed0_ty.shape) != (_DSA_RANDGEN_NUM_STREAMS, )
            or tuple(int(tl._unwrap_if_constexpr(d)) for d in seed1_ty.shape) != (_DSA_RANDGEN_NUM_STREAMS, )):
        raise ValueError("tle.dsa.tsingmicro.randgen seeds must be block tensors of shape [16]")

    out_ty = tl.block_type(tl.int64, [n_out])
    byte_count = n_out * 8

    rand_op = builder.create_dsa_randgen(
        out_ty.to_ir(builder),
        seed0_ty.to_ir(builder),
        seed1_ty.to_ir(builder),
        seed0.handle,
        seed1.handle,
        int(byte_count),
        int(_DSA_RANDGEN_FMT_INT64),
    )
    out = tl.tensor(rand_op.get_result(0), out_ty)
    seed0_out = tl.tensor(rand_op.get_result(1), seed0_ty)
    seed1_out = tl.tensor(rand_op.get_result(2), seed1_ty)
    return out, seed0_out, seed1_out


def _uint32_bits_to_uniform(bits32, semantic):
    """
    Map random int32 bits to Uniform(0, 1) via IEEE754 mantissa stuffing.

    u = bitcast((bits & 0x7FFFFF) | 0x3F800000, f32) - 1.0

    Avoids the sitofp / where / cmp / sub integer chain that lowers to
    per-element scf.for on TX81 (no int vector ALU). Uses only bitwise
    and/or + bitcast + float sub, which can stay on peri / float paths.

    ``semantic`` is the SemanticAnalyzer (bound methods, not a raw builder).
    """
    # 0x7FFFFF fits int32; only the bit pattern of 0x3F800000 matters for
    # the bitcast.
    mant_mask = semantic.to_tensor(0x7FFFFF)
    one_bits = semantic.to_tensor(0x3F800000)
    one_f = semantic.to_tensor(1.0)
    mant = semantic.and_(bits32, mant_mask)
    packed = semantic.or_(mant, one_bits)
    f12 = semantic.bitcast(packed, tl.float32)  # [1, 2)
    return semantic.sub(f12, one_f, True)  # [0, 1)


def _i64_as_i32_view(raw_i64, n_i32: int, builder):
    """
    Zero-copy view of an i64 buffer as i32 (little-endian: lo32, hi32, ...).

    ``raw_i64`` must have shape ``[n_i32 // 2]``. Emits vendor-neutral
    ``dsa.bitcast`` (backends alias the buffer; no elementwise ``trunci``).
    """
    n_i64 = int(tl._unwrap_if_constexpr(raw_i64.shape[0]))
    if n_i32 != n_i64 * 2:
        raise ValueError(f"i64->i32 view expects n_i32 == 2 * n_i64, got {n_i32} vs 2*{n_i64}")
    if not hasattr(builder, "create_dsa_bitcast"):
        raise RuntimeError("builder missing create_dsa_bitcast; rebuild libtriton with TLE DSA")
    dst_ty = tl.block_type(tl.int32, [n_i32])
    handle = builder.create_dsa_bitcast(dst_ty.to_ir(builder), raw_i64.handle)
    return tl.tensor(handle, dst_ty)


@tl.builtin
def rand(seed0, seed1, n_out: tl.constexpr, _semantic=None):
    """
    Uniform(0, 1) floats via hardware ``randgen`` + float scaling.

    ``n_out`` must be a multiple of 32 (``randgen`` emits i64; each i64
    contributes two i32 samples via a zero-copy view).

    Returns ``(u, seed0_out, seed1_out)`` with ``u`` shaped ``[n_out]`` float32.
    """
    n_out = int(tl._unwrap_if_constexpr(n_out))
    if n_out <= 0 or (n_out % 32) != 0:
        raise ValueError(f"tle.dsa.tsingmicro.rand n_out must be a positive multiple of 32, got {n_out}")

    builder = _semantic.builder
    # Half as many i64 draws: lo/hi 32-bit halves become two Uniform samples.
    raw64, seed0_out, seed1_out = randgen(seed0, seed1, n_out // 2, _semantic=_semantic)
    bits32 = _i64_as_i32_view(raw64, n_out, builder)
    u = _uint32_bits_to_uniform(bits32, _semantic)
    return u, seed0_out, seed1_out


@tl.builtin
def randn(seed0, seed1, n_out: tl.constexpr, _semantic=None):
    """
    Normal(0, 1) floats via hardware ``randgen`` + Box-Muller.

    ``n_out`` must be a multiple of 32 (two Uniform halves of size
    ``n_out // 2``, each backed by ``n_out // 4`` i64 draws viewed as i32).

    Uses a single ``randgen`` of length ``n_out // 2`` and pairs consecutive
    Uniform samples ``(u[2i], u[2i+1])`` for Box-Muller.  Calling ``rand``
    twice is unsafe while the peri does not advance ``seed0_out`` /
    ``seed1_out``.

    Returns ``(n, seed0_out, seed1_out)`` with ``n`` shaped ``[n_out]`` float32
    (concatenation of the two Box-Muller outputs).
    """
    n_out = int(tl._unwrap_if_constexpr(n_out))
    half = n_out // 2
    if n_out <= 0 or (n_out % 32) != 0:
        raise ValueError(f"tle.dsa.tsingmicro.randn n_out must be a positive multiple of 32, got {n_out}")

    # Box-Muller over consecutive Uniform pairs. Scalar f32 constants are
    # broadcast from u_half (``u*0+c``): direct ``semantic.to_tensor(float)``
    # scalars bias the result on TX81.
    uv, seed0_out, seed1_out = rand(seed0, seed1, n_out, _semantic=_semantic)

    pairs = _semantic.reshape(uv, [half, 2], False)
    u_half, v_half = _semantic.split(pairs)

    # Force f32-typed scalars via broadcast from u_half (avoids f64 pitfalls).
    zero = _semantic.mul(u_half, _semantic.to_tensor(0.0), True)
    eps = _semantic.add(zero, _semantic.to_tensor(1.0e-7), True)
    two_pi = _semantic.add(zero, _semantic.to_tensor(6.283185307179586), True)
    neg_two = _semantic.add(zero, _semantic.to_tensor(-2.0), True)

    u1 = _semantic.maximum(u_half, eps, PropagateNan.NONE)
    theta = _semantic.mul(two_pi, v_half, True)
    log_u1 = tlmath.log(u1, _semantic=_semantic)
    r = tlmath.sqrt(_semantic.mul(neg_two, log_u1, True), _semantic=_semantic)
    n0 = _semantic.mul(r, tlmath.cos(theta, _semantic=_semantic), True)
    n1 = _semantic.mul(r, tlmath.sin(theta, _semantic=_semantic), True)
    out = _semantic.reshape(_semantic.join(n0, n1), [n_out], False)
    return out, seed0_out, seed1_out
