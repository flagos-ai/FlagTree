# Copyright (c) 2026 T-Head Semiconductor Co., Ltd. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining
# a copy of this software and associated documentation files
# (the "Software"), to deal in the Software without restriction,
# including without limitation the rights to use, copy, modify, merge,
# publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:
#
# The above copyright notice and this permission notice shall be
# included in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
# MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
# IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
# CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
# TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
# SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

from triton.language import core


@core.extern
def globaltimer(_semantic=None):
    return core.inline_asm_elementwise("ppu.mov.u64 $0, %globaltimer;", "=l", [], dtype=core.int64, is_pure=False,
                                       pack=1, _semantic=_semantic)


@core.extern
def smid(_semantic=None):
    return core.inline_asm_elementwise("ppu.mov.u32 $0, %smid;", "=r", [], dtype=core.int32, is_pure=True, pack=1,
                                       _semantic=_semantic)


@core.builtin
def num_threads(_semantic=None):
    return core.constexpr(_semantic.builder.options.num_warps * 32)


@core.builtin
def num_warps(_semantic=None):
    return core.constexpr(_semantic.builder.options.num_warps)


# ----- FP8E4M3B15 ------
# This data-type is a variant of the standard FP8E4M3 format.
# It was designed for fast software conversion to FP16 on
# GPUs that do not support it natively.
# This is the same format as FP8E4M3Nv, but:
#   - the exponent bias is 15 instead of 7
#   - 0xff and 0x7f are mapped to +-1.750 instead of +-nan
@core.builtin
def convert_fp8e4b15_to_float16(arg, _semantic=None):
    return core.inline_asm_elementwise(
        "{                                      \n"
        ".reg .b32 a<2>, b<2>;                  \n"
        "ppu.prmt.b32 a0, 0, $2, 0x5746;            \n"
        "ppu.and.b32 b0, a0, 0x7f007f00;            \n"
        "ppu.and.b32 b1, a0, 0x00ff00ff;            \n"
        "ppu.and.b32 a1, a0, 0x00800080;            \n"
        "ppu.shr.b32  b0, b0, 1;                    \n"
        "ppu.add.u32 b1, b1, a1;                    \n"
        "ppu.lop3.b32 $0, b0, 0x80008000, a0, 0xf8; \n"
        "ppu.shl.b32 $1, b1, 7;                     \n"
        "}                                      \n", "=r,=r,r", [arg], dtype=core.float16, is_pure=True, pack=4,
        _semantic=_semantic)


@core.builtin
def convert_float16_to_fp8e4b15(arg, has_minx2, _semantic=None):
    asm = """{
            .reg .pred p<4>;
            .reg .b32 a<2>, b<2>;
            .reg .b16 c<4>;
            .reg .b16 max_val_f16;
            .reg .b32 max_val_f16x2;
            ppu.mov.b16 max_val_f16,   0x3F00;
            ppu.mov.b32 max_val_f16x2, 0x3F003F00;
            ppu.and.b32 a0, $1, 0x7fff7fff;
            ppu.and.b32 a1, $2, 0x7fff7fff;"""
    if has_minx2:
        asm += """ppu.min.f16x2 a0, a0, max_val_f16x2;
                  ppu.min.f16x2 a1, a1, max_val_f16x2;"""
    else:
        asm += """ppu.setp.lt.f16x2  p0|p1, a0, max_val_f16x2;
                  ppu.setp.lt.f16x2  p2|p3, a1, max_val_f16x2;
                  ppu.mov.b32 {c0, c1}, a0;
                  ppu.mov.b32 {c2, c3}, a1;
                  ppu.selp.b16  c0, c0, max_val_f16, p0;
                  ppu.selp.b16  c1, c1, max_val_f16, p1;
                  ppu.selp.b16  c2, c2, max_val_f16, p2;
                  ppu.selp.b16  c3, c3, max_val_f16, p3;
                  ppu.mov.b32 a0, {c0, c1};
                  ppu.mov.b32 a1, {c2, c3};"""
    asm += """ppu.mad.lo.u32 a0, a0, 2, 0x00800080;
              ppu.mad.lo.u32 a1, a1, 2, 0x00800080;
              ppu.lop3.b32 b0, $1, 0x80008000, a0, 0xea;
              ppu.lop3.b32 b1, $2, 0x80008000, a1, 0xea;
              ppu.prmt.b32 $0, b0, b1, 0x7531;
              }"""
    return core.inline_asm_elementwise(asm, "=r,r,r", [arg], dtype=core.float8e4b15, is_pure=True, pack=4,
                                       _semantic=_semantic)


@core.builtin
def convert_custom_float8_internal(arg, dst_ty, fp_downcast_rounding, has_minx2, _semantic=None):
    if arg.type.scalar.is_fp8e4b15():
        upcast_val = convert_fp8e4b15_to_float16(arg, _semantic=_semantic)
        if dst_ty.scalar.is_fp32():
            upcast_val = upcast_val.to(core.float32, _semantic=_semantic)
        return upcast_val

    assert arg.type.scalar.is_fp16() or arg.type.scalar.is_fp32()
    downcast_val = arg
    if arg.type.scalar.is_fp32():
        downcast_val = downcast_val.to(core.float16, fp_downcast_rounding="rtz", _semantic=_semantic)
    downcast_val = convert_float16_to_fp8e4b15(downcast_val, has_minx2=has_minx2, _semantic=_semantic)
    return downcast_val


@core.builtin
def convert_custom_float8(arg, dst_ty, fp_downcast_rounding=None, _semantic=None):
    return convert_custom_float8_internal(arg, dst_ty, fp_downcast_rounding, has_minx2=True, _semantic=_semantic)


# ----- FP8E4M3FN (fp8e4nv) software cast, capability < 89 ------
# cap80 has no e4m3 cvt instructions; the numerical cast is implemented with
# ordinary integer/float ops. Rounding is strict RTNE (ties-to-even, with the
# mantissa carry propagating into the exponent) or RTZ, saturation is
# satfinite (|x| > 448 -> 0x7E), NaN maps to 0x7F, and 0x7F/0xFF decode to NaN.


def _rounding_is_rtz(fp_downcast_rounding):
    # accepts the ir.ROUNDING_MODE enum (from semantic.cast) or the string spelling
    if fp_downcast_rounding is None:
        return False
    if isinstance(fp_downcast_rounding, str):
        return fp_downcast_rounding.lower() == "rtz"
    from triton._C.libtriton import ir
    return fp_downcast_rounding == ir.ROUNDING_MODE.RTZ


@core.builtin
def _upcast_e4nv_to_f16(arg, _semantic=None):
    """Software OCP E4M3FN -> fp16; exact for every one of the 256 encodings."""
    sem = _semantic
    u = arg.to(core.uint8, bitcast=True, _semantic=sem).to(core.uint16, _semantic=sem)
    s = sem.shl(sem.and_(u, 0x0080), 8)
    em = sem.and_(u, 0x007F)
    e = sem.lshr(em, 3)
    m = sem.and_(em, 0x0007)
    # normal (e >= 1): f16 bits = ((e - 7 + 15) << 10) | (m << 7)
    normal = sem.or_(sem.shl(sem.add(e, 8, False), 10), sem.shl(m, 7))
    # subnormal (e == 0): value = m * 2^-9, exact via f32 then truncation to f16
    sub_f = sem.mul(m.to(core.float32, _semantic=sem), 2.0**-9, False)
    sub_bits = sub_f.to(core.float16, _semantic=sem).to(core.uint16, bitcast=True, _semantic=sem)
    body = sem.where(sem.equal(e, 0), sub_bits, normal)
    body = sem.where(sem.equal(em, 0x007F), 0x7E00, body)  # NaN codes 0x7F/0xFF
    return sem.or_(body, s).to(core.float16, bitcast=True, _semantic=sem)


@core.builtin
def _downcast_f32_to_e4nv(arg, rtz, _semantic=None):
    """Software fp32 -> OCP E4M3FN with single rounding (RTNE or RTZ) and
    satfinite saturation; bit-exact against the format's reference encoder."""
    sem = _semantic
    b = arg.to(core.int32, bitcast=True, _semantic=sem)
    sign = sem.shl(sem.and_(sem.lshr(b, 31), 1), 7)
    ab = sem.and_(b, 0x7FFFFFFF)
    is_nan = sem.greater_than(ab, 0x7F800000)
    e32 = sem.lshr(ab, 23)
    m32 = sem.and_(ab, 0x7FFFFF)
    # E: biased target exponent (bias 7); k: mantissa bits to drop (20 for the
    # normal range, up to 6 more in the subnormal range; >26 always rounds to 0)
    E = sem.sub(e32, 120, False)
    pn = core.PropagateNan.NONE
    k = sem.add(sem.minimum(sem.maximum(sem.sub(1, E, False), 0, pn), 6, pn), 20, False)
    sig = sem.or_(m32, 0x800000)
    keep = sem.lshr(sig, k)
    if rtz:
        pass
    else:  # strict RTNE: round up on >half, or ==half when the kept lsb is odd
        rem = sem.and_(sig, sem.sub(sem.shl(1, k), 1, False))
        half = sem.shl(1, sem.sub(k, 1, False))
        up = sem.or_(sem.greater_than(rem, half), sem.and_(sem.equal(rem, half), sem.equal(sem.and_(keep, 1), 1)))
        keep = sem.add(keep, up.to(core.int32, _semantic=sem), False)
    # (E<<3) + keep - 8 folds the mantissa carry into the exponent; the
    # subnormal branch is plain `keep` (keep == 8 becomes the minimum normal)
    expmant = sem.where(sem.greater_equal(E, 1), sem.sub(sem.add(sem.shl(E, 3), keep, False), 8, False), keep)
    expmant = sem.minimum(expmant, 0x7E, pn)  # satfinite (|x| > 448, inf)
    res = sem.or_(expmant, sign)
    res = sem.where(is_nan, 0x7F, res)
    return res.to(core.uint8, _semantic=sem).to(core.float8e4nv, bitcast=True, _semantic=sem)


@core.builtin
def convert_custom_float8_sub89(arg, dst_ty, fp_downcast_rounding=None, _semantic=None):
    """convert_custom_types implementation for capability < 89: the fp8e4b15
    path of convert_custom_float8 plus the software fp8e4nv cast."""
    src_sca = arg.type.scalar
    dst_sca = dst_ty.scalar
    if src_sca.is_fp8e4b15() or dst_sca.is_fp8e4b15():
        return convert_custom_float8_internal(arg, dst_ty, fp_downcast_rounding, has_minx2=True, _semantic=_semantic)
    if src_sca.is_fp8e4nv():
        up = _upcast_e4nv_to_f16(arg, _semantic=_semantic)
        if dst_sca.is_fp16():
            return up
        if dst_sca.is_fp32() or dst_sca.is_bf16() or dst_sca.is_fp64():
            return up.to(dst_sca, _semantic=_semantic)
        if dst_sca.is_fp8e5():
            rounding = "rtz" if _rounding_is_rtz(fp_downcast_rounding) else "rtne"
            return up.to(dst_sca, fp_downcast_rounding=rounding, _semantic=_semantic)
        raise ValueError(f"cast from fp8e4nv to {dst_sca} is not supported on this product")
    if dst_sca.is_fp8e4nv():
        if not src_sca.is_floating():
            raise ValueError(f"cast from {src_sca} to fp8e4nv is not supported on this product")
        # f16/bf16 widen exactly (single rounding); an fp64 source rounds twice
        x = arg if src_sca.is_fp32() else arg.to(core.float32, _semantic=_semantic)
        return _downcast_f32_to_e4nv(x, _rounding_is_rtz(fp_downcast_rounding), _semantic=_semantic)
    raise ValueError(f"unsupported custom fp8 conversion from {src_sca} to {dst_sca}")
