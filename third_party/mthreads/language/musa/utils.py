"""MUSA language utilities.

Minimal helpers that are backend-agnostic but commonly used by kernels.
"""

from triton.language import core

_FP32_QNAN_BITS = 0x7FC00000


@core.builtin
def num_threads(_semantic=None):
    return core.constexpr(_semantic.builder.options.num_warps * 32)


@core.builtin
def num_warps(_semantic=None):
    return core.constexpr(_semantic.builder.options.num_warps)


@core.builtin
def _upcast_fp8_to_fp32(arg, exponent_bits, mantissa_bits, exponent_bias, nan_on_negzero, _semantic=None):
    raw = arg.to(core.uint8, bitcast=True, _semantic=_semantic)
    raw32 = raw.to(core.uint32, _semantic=_semantic)

    mantissa_mask = (1 << mantissa_bits) - 1
    exponent_mask = (1 << exponent_bits) - 1
    sign_shift = exponent_bits + mantissa_bits

    mantissa = raw32.__and__(mantissa_mask, _semantic=_semantic)
    exponent = raw32.__rshift__(mantissa_bits, _semantic=_semantic)
    exponent = exponent.__and__(exponent_mask, _semantic=_semantic)
    sign = raw32.__rshift__(sign_shift, _semantic=_semantic)

    value_bits = sign.__lshift__(31, _semantic=_semantic)
    value_bits = value_bits.__or__(exponent.__lshift__(23, _semantic=_semantic), _semantic=_semantic)
    value_bits = value_bits.__or__(mantissa.__lshift__(23 - mantissa_bits, _semantic=_semantic), _semantic=_semantic)
    value = value_bits.to(core.float32, bitcast=True, _semantic=_semantic)
    value = core.mul(value, 2.0**(127 - exponent_bias), _semantic=_semantic)

    if nan_on_negzero:
        nan_bits = core.full(raw.shape, _FP32_QNAN_BITS, core.uint32, _semantic=_semantic)
        qnan = nan_bits.to(core.float32, bitcast=True, _semantic=_semantic)
        is_negzero = raw.__eq__(0x80, _semantic=_semantic)
        value = core.where(is_negzero, qnan, value, _semantic=_semantic)

    return value


def _upcast_fp32_to_dst(upcast, dst_ty, _semantic):
    dst_scalar = dst_ty.scalar
    if dst_scalar.is_fp32():
        return upcast
    return upcast.to(dst_scalar, _semantic=_semantic)


@core.builtin
def convert_custom_float8(arg, dst_ty, fp_downcast_rounding=None, _semantic=None):
    src_ty = arg.type.scalar
    dst_scalar = dst_ty.scalar

    if dst_scalar.is_fp8e4b15() or dst_scalar.is_fp8e4b8() or dst_scalar.is_fp8e5b16():
        raise ValueError(f"conversion to {dst_scalar} is not supported in this architecture")

    if src_ty.is_fp8e4b15():
        if not (dst_scalar.is_fp16() or dst_scalar.is_fp32()):
            raise ValueError(f"conversion from {src_ty} to {dst_scalar} is not supported in this architecture")
        upcast = _upcast_fp8_to_fp32(arg, exponent_bits=4, mantissa_bits=3, exponent_bias=15, nan_on_negzero=False,
                                     _semantic=_semantic)
        return _upcast_fp32_to_dst(upcast, dst_ty, _semantic)

    if src_ty.is_fp8e4b8():
        if not (dst_scalar.is_fp16() or dst_scalar.is_bf16() or dst_scalar.is_fp32()):
            raise ValueError(f"conversion from {src_ty} to {dst_scalar} is not supported in this architecture")
        upcast = _upcast_fp8_to_fp32(arg, exponent_bits=4, mantissa_bits=3, exponent_bias=8, nan_on_negzero=True,
                                     _semantic=_semantic)
        return _upcast_fp32_to_dst(upcast, dst_ty, _semantic)

    if src_ty.is_fp8e5b16():
        if not (dst_scalar.is_fp16() or dst_scalar.is_fp32()):
            raise ValueError(f"conversion from {src_ty} to {dst_scalar} is not supported in this architecture")
        upcast = _upcast_fp8_to_fp32(arg, exponent_bits=5, mantissa_bits=2, exponent_bias=16, nan_on_negzero=True,
                                     _semantic=_semantic)
        return _upcast_fp32_to_dst(upcast, dst_ty, _semantic)

    raise ValueError(f"unsupported custom fp8 conversion from {src_ty} to {dst_scalar}")
