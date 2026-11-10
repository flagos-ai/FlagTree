"""HCU language utilities and packed-format operations."""
import warnings
from typing import TYPE_CHECKING, Optional

import triton
import triton.language as tl
from triton.language import core

if TYPE_CHECKING:
    from triton.language.semantic import TritonSemantic


@core.extern
def memrealtime(_semantic=None):
    """
    Returns a 64-bit real time-counter value
    """
    target_arch = _semantic.builder.options.arch
    if 'gfx11' in target_arch or 'gfx12' in target_arch:
        return core.inline_asm_elementwise(
            """
            s_sendmsg_rtn_b64 $0, sendmsg(MSG_RTN_GET_REALTIME)
            s_waitcnt lgkmcnt(0)
            """,
            "=r",
            [],
            dtype=core.int64,
            is_pure=False,
            pack=1,
            _semantic=_semantic,
        )
    else:
        return core.inline_asm_elementwise(
            """
            s_memrealtime $0
            s_waitcnt vmcnt(0)
            """,
            "=r",
            [],
            dtype=core.int64,
            is_pure=False,
            pack=1,
            _semantic=_semantic,
        )


def _convert_fp8(input, dst_ty, rounding, _semantic):
    """Keep floating conversions in HCU lowering; bridge integer casts in FP32.

    Every integer in the finite OCP FP8 range is exact in FP32, so this bridge
    does not introduce integer-to-FP8 double rounding. FP8-to-integer uses RTZ
    and the ordinary Triton representable-input requirement.
    """
    from triton._C.libtriton import ir

    if input.dtype.is_fp8() and dst_ty.scalar.is_fp8():
        # FP32 represents every FP8 value exactly; narrow once to the target
        # format with the same saturation and special-value rules as .to().
        return _semantic.cast(_semantic.cast(input, tl.float32), dst_ty)
    if input.dtype.is_int():
        input = _semantic.cast(input, tl.float32)
        rounding = ir.ROUNDING_MODE.RTNE
    if dst_ty.scalar.is_int():
        decoded_ty = input.type.with_element_ty(tl.float32) if input.type.is_block() else tl.float32
        decoded = _semantic.tensor(
            _semantic.builder.create_fp_to_fp(input.handle, decoded_ty.to_ir(_semantic.builder), None), decoded_ty)
        return _semantic.cast(decoded, dst_ty)
    return _semantic.tensor(_semantic.builder.create_fp_to_fp(input.handle, dst_ty.to_ir(_semantic.builder), rounding),
                            dst_ty)


# HIP extras load before triton.language reexports the core types.
@core.builtin
def fp_to_fp4(src: core.tensor, axis: int = -1, fp_downcast_rounding: Optional[str] = "rtne",
              _semantic: Optional["TritonSemantic"] = None) -> core.tensor:
    """Encode E2M1 into uint8, two values per byte, low nibble first.

    The packing axis must be even. RTNE/RTZ preserve signed zero and saturate
    overflow, Inf and NaN to signed magnitude 6. Scaling is explicit. FP64
    comparisons stay FP64. Encoding executes on the GPU without native FP4.
    """
    axis = tl.core._unwrap_if_constexpr(axis)
    rounding = tl.core._unwrap_if_constexpr(fp_downcast_rounding)
    if src.dtype not in (tl.float16, tl.bfloat16, tl.float32, tl.float64):
        raise ValueError(f"fp_to_fp4 requires FP16/BF16/FP32/FP64 input, got {src.dtype}")
    shape = src.type.get_block_shapes() if src.type.is_block() else []
    rank = len(shape)
    if not isinstance(axis, int) or not -rank <= axis < rank:
        raise ValueError(f"fp_to_fp4 axis {axis} must be in [-{rank}, {rank})")
    axis %= rank
    if shape[axis] % 2:
        raise ValueError(f"fp_to_fp4 packing dimension must be even, got {shape[axis]}")
    rounding = "rtne" if rounding is None else rounding
    if rounding not in ("rtne", "rtz"):
        raise ValueError(f"fp_to_fp4 rounding must be rtne or rtz, got {rounding}")
    warnings.warn("HCU fp_to_fp4 uses GPU software encoding (native=false)", UserWarning)
    if src.dtype in (tl.float16, tl.bfloat16):
        src = _semantic.cast(src, tl.float32)
    int_ty = tl.uint64 if src.dtype == tl.float64 else tl.uint32
    width = int_ty.primitive_bitwidth
    bits = _semantic.bitcast(src, int_ty)
    sign = _semantic.cast(_semantic.lshr(bits, _semantic.scalar_constant(width - 1, int_ty)), tl.uint8)
    sign = _semantic.shl(sign, _semantic.scalar_constant(3, tl.uint8))
    magnitude = _semantic.bitcast(_semantic.and_(bits, _semantic.scalar_constant((1 << (width - 1)) - 1, int_ty)),
                                  src.dtype)
    codes = _semantic.full(shape, 0, tl.uint8)
    thresholds = (.25, .75, 1.25, 1.75, 2.5, 3.5, 5.) if rounding == "rtne" else (.5, 1., 1.5, 2., 3., 4., 6.)
    for code, threshold in enumerate(thresholds, 1):
        boundary = _semantic.scalar_constant(threshold, src.dtype)
        above = _semantic.greater_equal(magnitude, boundary) if rounding == "rtz" else _semantic.greater_than(
            magnitude, boundary)
        if rounding == "rtne" and code % 2 == 0:
            above = _semantic.logical_or(above, _semantic.equal(magnitude, boundary))
        codes = _semantic.where(above, _semantic.scalar_constant(code, tl.uint8), codes)
    codes = _semantic.where(_semantic.not_equal(magnitude, magnitude), _semantic.scalar_constant(7, tl.uint8), codes)
    codes = _semantic.or_(codes, sign)
    paired = _semantic.reshape(codes, shape[:axis] + [shape[axis] // 2, 2] + shape[axis + 1:], can_reorder=False)
    order = [d for d in range(rank + 1) if d != axis + 1] + [axis + 1]
    low, high = _semantic.split(_semantic.permute(paired, order))
    return _semantic.or_(low, _semantic.shl(high, _semantic.scalar_constant(4, tl.uint8)))


@core.builtin
def _software_warning(message, _semantic=None):
    warnings.warn(tl.core._unwrap_if_constexpr(message), UserWarning)


@triton.constexpr_function
def _unpack_layout(shape, axis):
    rank = len(shape)
    order = tuple(range(axis + 1)) + (rank, ) + tuple(range(axis + 1, rank))
    result = tuple(shape[:axis]) + (shape[axis] * 2, ) + tuple(shape[axis + 1:])
    return order, result


@triton.jit
def _unpack_int4(packed: core.tensor, axis: core.constexpr) -> core.tensor:
    """Decode signed packed INT4 operands to INT8 for integer dot."""
    tl.static_assert(packed.dtype == tl.uint8, "INT4 operands require packed uint8")
    low = (packed << 4).to(tl.int8, bitcast=True) >> 4
    high = packed.to(tl.int8, bitcast=True) >> 4
    paired = tl.join(low, high)
    layout: core.constexpr = _unpack_layout(packed.shape, axis)
    return tl.permute(paired, layout[0]).reshape(layout[1])


@triton.jit
def fp4_to_fp_scaled(src: core.tensor, scale: "Optional[core.tensor]", elem_type: core.constexpr,
                     axis: core.constexpr) -> core.tensor:
    """Decode E2M1, then apply E8M0 scales in FP32 and narrow with RTNE.

    Scales are None, one byte per 32 logical values, or one byte per packed
    pair. E8M0 code 0 is 2**-127; 255 produces NaN. FP16/BF16 output preserves
    IEEE overflow/underflow. Decode and scale execute on the GPU in software.
    """
    tl.static_assert(src.dtype == tl.uint8, "FP4 must be packed uint8")
    tl.static_assert(elem_type == tl.float16 or elem_type == tl.bfloat16, "output must be FP16/BF16")
    tl.static_assert(-len(src.shape) <= axis and axis < len(src.shape), "invalid packing axis")
    packing_axis: core.constexpr = axis % len(src.shape)
    _software_warning("HCU fp4_to_fp_scaled uses GPU software decoding/scaling (native=false)")
    # Sign extension is useful for INT4, but E2M1 uses sign-magnitude codes.
    low = (src & 15).to(tl.uint16)
    high = (src >> 4).to(tl.uint16)
    paired = tl.join(low, high)
    layout: core.constexpr = _unpack_layout(src.shape, packing_axis)
    shape: core.constexpr = layout[1]
    nibbles = tl.permute(paired, layout[0]).reshape(shape)
    sign = (nibbles & 8) << 12
    exponent = (nibbles >> 1) & 3
    mantissa = nibbles & 1
    bits = sign | tl.where(exponent == 0, mantissa * 0x3800, ((exponent + 14) << 10) | (mantissa << 9))
    decoded = bits.to(tl.uint16).to(tl.float16, bitcast=True).to(tl.float32)
    if scale is not None:
        tl.static_assert(scale.dtype == tl.uint8, "E8M0 scale must be uint8")
        tl.static_assert(len(scale.shape) == len(src.shape), "scale rank must match packed input")
        for dim in tl.static_range(len(src.shape)):
            if dim != packing_axis:
                tl.static_assert(scale.shape[dim] == src.shape[dim], "scale shape mismatch")
        tl.static_assert(
            scale.shape[packing_axis] == src.shape[packing_axis]
            or scale.shape[packing_axis] * 16 == src.shape[packing_axis],
            "scale packing_axis must contain one byte per pair or 32 logical values")
        expanded_shape: core.constexpr = src.shape[:packing_axis] + [
            scale.shape[packing_axis], src.shape[packing_axis] * 2 // scale.shape[packing_axis]
        ] + src.shape[packing_axis + 1:]
        scale_shape: core.constexpr = src.shape[:packing_axis] + [scale.shape[packing_axis], 1
                                                                  ] + src.shape[packing_axis + 1:]
        codes = tl.broadcast_to(scale.reshape(scale_shape), expanded_shape).reshape(shape).to(tl.uint32)
        scale_bits = tl.where(codes == 0, 0x00400000, codes << 23)
        factors = scale_bits.to(tl.float32, bitcast=True)
        decoded = tl.where(codes == 255, float('nan'), decoded * factors)
    return decoded.to(elem_type, fp_downcast_rounding="rtne")


@triton.jit
def _dot_scaled_integer(lhs: core.tensor, lhs_scale: "Optional[core.tensor]", lhs_format: core.constexpr,
                        rhs: core.tensor, rhs_scale: "Optional[core.tensor]", rhs_format: core.constexpr,
                        acc: "Optional[core.tensor]", lhs_k_pack: core.constexpr, rhs_k_pack: core.constexpr,
                        out_dtype: core.constexpr) -> core.tensor:
    """Lower HCU integer formats from the standard tl.dot_scaled entry point.

    Scaled INT4 uses K-packed uint8, low nibble first. Each 32-element
    group's integer products and INT32 accumulation wrap
    modulo 2**32, then convert to FP32, multiply lhs scale then rhs scale, and
    add to the FP32 accumulator once. Scale shapes are [M,G] and [N,G]. None
    means one. Unscaled integer dot with INT32 output also supports M/N packing.
    """
    tl.static_assert((lhs_format == "int4" or lhs_format == "int8") and (rhs_format == "int4" or rhs_format == "int8"),
                     "integer operand formats must be int4/int8; INT32 is for accumulation/output")
    tl.static_assert(len(lhs.shape) == 2 and len(rhs.shape) == 2, "integer scaled dot requires 2D operands")
    if lhs_format == "int4":
        lhs = _unpack_int4(lhs, 1 if lhs_k_pack else 0)
    else:
        tl.static_assert(lhs.dtype == tl.int8 and lhs_k_pack,
                         "INT8 lhs requires signed INT8 storage and lhs_k_pack=True")
    if rhs_format == "int4":
        rhs = _unpack_int4(rhs, 0 if rhs_k_pack else 1)
    else:
        tl.static_assert(rhs.dtype == tl.int8 and rhs_k_pack,
                         "INT8 rhs requires signed INT8 storage and rhs_k_pack=True")
    if out_dtype == tl.int32:
        tl.static_assert(lhs_scale is None and rhs_scale is None,
                         "INT32 output requires both integer scales to be None")
        if lhs_format == "int4" or rhs_format == "int4":
            _software_warning("HCU INT4 dot uses GPU software unpacking and INT8 dot (native=false)")
        return tl.dot(lhs, rhs, acc=acc, out_dtype=tl.int32)
    else:
        tl.static_assert(out_dtype == tl.float32, "integer scaled dot requires FP32 output")
        tl.static_assert(lhs_k_pack and rhs_k_pack, "integer scaled dot requires K-packed operands")
        _software_warning("HCU integer scaled dot uses GPU groupwise integer dot and FP32 scaling (native=false)")
        group_size: core.constexpr = 32
        M: core.constexpr = lhs.shape[0]
        K: core.constexpr = lhs.shape[1]
        N: core.constexpr = rhs.shape[1]
        tl.static_assert(rhs.shape[0] == K and K % group_size == 0, "logical K mismatch or incomplete group")
        G: core.constexpr = K // group_size
        if lhs_scale is not None:
            tl.static_assert(lhs_scale.dtype == tl.float32 and lhs_scale.shape == [M, G],
                             "lhs scales must be FP32 [M,G]")
        if rhs_scale is not None:
            tl.static_assert(rhs_scale.dtype == tl.float32 and rhs_scale.shape == [N, G],
                             "rhs scales must be FP32 [N,G]")
        if acc is None:
            result = tl.full((M, N), 0, tl.float32)
        else:
            tl.static_assert(acc.dtype == tl.float32 and acc.shape == [M, N], "acc must be FP32 [M,N]")
            result = acc
        a = lhs.reshape((M, G, group_size))
        b = rhs.reshape((G, group_size, N))
        for group in tl.static_range(G):
            ag = tl.gather(a, tl.full((M, 1, group_size), group, tl.int32), 1).reshape((M, group_size))
            bg = tl.gather(b, tl.full((1, group_size, N), group, tl.int32), 0).reshape((group_size, N))
            partial = tl.dot(ag, bg, out_dtype=tl.int32).to(tl.float32)
            if lhs_scale is not None:
                partial = partial * tl.gather(lhs_scale, tl.full((M, 1), group, tl.int32), 1)
            if rhs_scale is not None:
                partial = partial * tl.gather(rhs_scale, tl.full((N, 1), group, tl.int32), 1).T
            result = result + partial
        return result


def _get_dot_scaled_lowering(lhs_format: str, rhs_format: str):
    integer_formats = ("int4", "int8", "int32")
    if lhs_format in integer_formats or rhs_format in integer_formats:
        return _dot_scaled_integer
    return None
