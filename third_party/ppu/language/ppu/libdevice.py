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
def clz(arg0, _semantic=None):
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("int32"), ): ("__ppu_clz", core.dtype("int32")),
            (core.dtype("int64"), ): ("__ppu_clzll", core.dtype("int32")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def popc(arg0, _semantic=None):
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("int32"), ): ("__ppu_popc", core.dtype("int32")),
            (core.dtype("int64"), ): ("__ppu_popcll", core.dtype("int32")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def byte_perm(arg0, arg1, arg2, _semantic=None):
    return core.extern_elementwise("", "", [arg0, arg1, arg2], {
        (core.dtype("int32"), core.dtype("int32"), core.dtype("int32")): ("__ppu_byte_perm", core.dtype("int32")),
    }, is_pure=True, _semantic=_semantic)


@core.extern
def mulhi(arg0, arg1, _semantic=None):
    return core.extern_elementwise(
        "", "", [arg0, arg1], {
            (core.dtype("int32"), core.dtype("int32")): ("__ppu_mulhi", core.dtype("int32")),
            (core.dtype("uint32"), core.dtype("uint32")): ("__ppu_umulhi", core.dtype("uint32")),
            (core.dtype("int64"), core.dtype("int64")): ("__ppu_mul64hi", core.dtype("int64")),
            (core.dtype("uint64"), core.dtype("uint64")): ("__ppu_umul64hi", core.dtype("uint64")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def mul24(arg0, arg1, _semantic=None):
    return core.extern_elementwise(
        "", "", [arg0, arg1], {
            (core.dtype("int32"), core.dtype("int32")): ("__ppu_mul24", core.dtype("int32")),
            (core.dtype("uint32"), core.dtype("uint32")): ("__ppu_umul24", core.dtype("uint32")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def brev(arg0, _semantic=None):
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("int32"), ): ("__ppu_brev", core.dtype("int32")),
            (core.dtype("int64"), ): ("__ppu_brevll", core.dtype("int64")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def sad(arg0, arg1, arg2, _semantic=None):
    return core.extern_elementwise(
        "", "", [arg0, arg1, arg2], {
            (core.dtype("int32"), core.dtype("int32"), core.dtype("uint32")): ("__ppu_sad", core.dtype("int32")),
            (core.dtype("uint32"), core.dtype("uint32"), core.dtype("uint32")): ("__ppu_usad", core.dtype("uint32")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def abs(arg0, _semantic=None):
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("int32"), ): ("__ppu_abs", core.dtype("int32")),
            (core.dtype("int64"), ): ("__ppu_llabs", core.dtype("int64")),
            (core.dtype("fp32"), ): ("__ppu_fabsf", core.dtype("fp32")),
            (core.dtype("fp64"), ): ("__ppu_fabs", core.dtype("fp64")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def floor(arg0, _semantic=None):
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp32"), ): ("__ppu_floorf", core.dtype("fp32")),
            (core.dtype("fp64"), ): ("__ppu_floor", core.dtype("fp64")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def rcp64h(arg0, _semantic=None):
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("fp64"), ): ("__ppu_rcp64h", core.dtype("fp64")),
    }, is_pure=True, _semantic=_semantic)


@core.extern
def rsqrt(arg0, _semantic=None):
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp32"), ): ("__ppu_rsqrtf", core.dtype("fp32")),
            (core.dtype("fp64"), ): ("__ppu_rsqrt", core.dtype("fp64")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def ceil(arg0, _semantic=None):
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp64"), ): ("__ppu_ceil", core.dtype("fp64")),
            (core.dtype("fp32"), ): ("__ppu_ceilf", core.dtype("fp32")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def trunc(arg0, _semantic=None):
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp64"), ): ("__ppu_trunc", core.dtype("fp64")),
            (core.dtype("fp32"), ): ("__ppu_truncf", core.dtype("fp32")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def exp2(arg0, _semantic=None):
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp32"), ): ("__ppu_exp2f", core.dtype("fp32")),
            (core.dtype("fp64"), ): ("__ppu_exp2", core.dtype("fp64")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def saturatef(arg0, _semantic=None):
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("fp32"), ): ("__ppu_saturatef", core.dtype("fp32")),
    }, is_pure=True, _semantic=_semantic)


@core.extern
def fma_rn(arg0, arg1, arg2, _semantic=None):
    return core.extern_elementwise(
        "", "", [arg0, arg1, arg2], {
            (core.dtype("fp32"), core.dtype("fp32"), core.dtype("fp32")): ("__ppu_fmaf_rn", core.dtype("fp32")),
            (core.dtype("fp64"), core.dtype("fp64"), core.dtype("fp64")): ("__ppu_fma_rn", core.dtype("fp64")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def fma_rz(arg0, arg1, arg2, _semantic=None):
    return core.extern_elementwise(
        "", "", [arg0, arg1, arg2], {
            (core.dtype("fp32"), core.dtype("fp32"), core.dtype("fp32")): ("__ppu_fmaf_rz", core.dtype("fp32")),
            (core.dtype("fp64"), core.dtype("fp64"), core.dtype("fp64")): ("__ppu_fma_rz", core.dtype("fp64")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def fma_rd(arg0, arg1, arg2, _semantic=None):
    return core.extern_elementwise(
        "", "", [arg0, arg1, arg2], {
            (core.dtype("fp32"), core.dtype("fp32"), core.dtype("fp32")): ("__ppu_fmaf_rd", core.dtype("fp32")),
            (core.dtype("fp64"), core.dtype("fp64"), core.dtype("fp64")): ("__ppu_fma_rd", core.dtype("fp64")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def fma_ru(arg0, arg1, arg2, _semantic=None):
    return core.extern_elementwise(
        "", "", [arg0, arg1, arg2], {
            (core.dtype("fp32"), core.dtype("fp32"), core.dtype("fp32")): ("__ppu_fmaf_ru", core.dtype("fp32")),
            (core.dtype("fp64"), core.dtype("fp64"), core.dtype("fp64")): ("__ppu_fma_ru", core.dtype("fp64")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def fast_dividef(arg0, arg1, _semantic=None):
    return core.extern_elementwise("", "", [arg0, arg1], {
        (core.dtype("fp32"), core.dtype("fp32")): ("__ppu_fast_fdividef", core.dtype("fp32")),
    }, is_pure=True, _semantic=_semantic)


@core.extern
def div_rn(arg0, arg1, _semantic=None):
    return core.extern_elementwise(
        "", "", [arg0, arg1], {
            (core.dtype("fp32"), core.dtype("fp32")): ("__ppu_fdiv_rn", core.dtype("fp32")),
            (core.dtype("fp64"), core.dtype("fp64")): ("__ppu_ddiv_rn", core.dtype("fp64")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def div_rz(arg0, arg1, _semantic=None):
    return core.extern_elementwise(
        "", "", [arg0, arg1], {
            (core.dtype("fp32"), core.dtype("fp32")): ("__ppu_fdiv_rz", core.dtype("fp32")),
            (core.dtype("fp64"), core.dtype("fp64")): ("__ppu_ddiv_rz", core.dtype("fp64")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def div_rd(arg0, arg1, _semantic=None):
    return core.extern_elementwise(
        "", "", [arg0, arg1], {
            (core.dtype("fp32"), core.dtype("fp32")): ("__ppu_fdiv_rd", core.dtype("fp32")),
            (core.dtype("fp64"), core.dtype("fp64")): ("__ppu_ddiv_rd", core.dtype("fp64")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def div_ru(arg0, arg1, _semantic=None):
    return core.extern_elementwise(
        "", "", [arg0, arg1], {
            (core.dtype("fp32"), core.dtype("fp32")): ("__ppu_fdiv_ru", core.dtype("fp32")),
            (core.dtype("fp64"), core.dtype("fp64")): ("__ppu_ddiv_ru", core.dtype("fp64")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def rcp_rn(arg0, _semantic=None):
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp32"), ): ("__ppu_frcp_rn", core.dtype("fp32")),
            (core.dtype("fp64"), ): ("__ppu_drcp_rn", core.dtype("fp64")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def rcp_rz(arg0, _semantic=None):
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp32"), ): ("__ppu_frcp_rz", core.dtype("fp32")),
            (core.dtype("fp64"), ): ("__ppu_drcp_rz", core.dtype("fp64")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def rcp_rd(arg0, _semantic=None):
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp32"), ): ("__ppu_frcp_rd", core.dtype("fp32")),
            (core.dtype("fp64"), ): ("__ppu_drcp_rd", core.dtype("fp64")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def rcp_ru(arg0, _semantic=None):
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp32"), ): ("__ppu_frcp_ru", core.dtype("fp32")),
            (core.dtype("fp64"), ): ("__ppu_drcp_ru", core.dtype("fp64")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def sqrt_rn(arg0, _semantic=None):
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp32"), ): ("__ppu_fsqrt_rn", core.dtype("fp32")),
            (core.dtype("fp64"), ): ("__ppu_dsqrt_rn", core.dtype("fp64")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def sqrt_rz(arg0, _semantic=None):
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp32"), ): ("__ppu_fsqrt_rz", core.dtype("fp32")),
            (core.dtype("fp64"), ): ("__ppu_dsqrt_rz", core.dtype("fp64")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def sqrt_rd(arg0, _semantic=None):
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp32"), ): ("__ppu_fsqrt_rd", core.dtype("fp32")),
            (core.dtype("fp64"), ): ("__ppu_dsqrt_rd", core.dtype("fp64")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def sqrt_ru(arg0, _semantic=None):
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp32"), ): ("__ppu_fsqrt_ru", core.dtype("fp32")),
            (core.dtype("fp64"), ): ("__ppu_dsqrt_ru", core.dtype("fp64")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def sqrt(arg0, _semantic=None):
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp32"), ): ("__ppu_sqrtf", core.dtype("fp32")),
            (core.dtype("fp64"), ): ("__ppu_sqrt", core.dtype("fp64")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def add_rn(arg0, arg1, _semantic=None):
    return core.extern_elementwise(
        "", "", [arg0, arg1], {
            (core.dtype("fp64"), core.dtype("fp64")): ("__ppu_dadd_rn", core.dtype("fp64")),
            (core.dtype("fp32"), core.dtype("fp32")): ("__ppu_fadd_rn", core.dtype("fp32")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def add_rz(arg0, arg1, _semantic=None):
    return core.extern_elementwise(
        "", "", [arg0, arg1], {
            (core.dtype("fp64"), core.dtype("fp64")): ("__ppu_dadd_rz", core.dtype("fp64")),
            (core.dtype("fp32"), core.dtype("fp32")): ("__ppu_fadd_rz", core.dtype("fp32")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def add_rd(arg0, arg1, _semantic=None):
    return core.extern_elementwise(
        "", "", [arg0, arg1], {
            (core.dtype("fp64"), core.dtype("fp64")): ("__ppu_dadd_rd", core.dtype("fp64")),
            (core.dtype("fp32"), core.dtype("fp32")): ("__ppu_fadd_rd", core.dtype("fp32")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def add_ru(arg0, arg1, _semantic=None):
    return core.extern_elementwise(
        "", "", [arg0, arg1], {
            (core.dtype("fp64"), core.dtype("fp64")): ("__ppu_dadd_ru", core.dtype("fp64")),
            (core.dtype("fp32"), core.dtype("fp32")): ("__ppu_fadd_ru", core.dtype("fp32")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def mul_rn(arg0, arg1, _semantic=None):
    return core.extern_elementwise(
        "", "", [arg0, arg1], {
            (core.dtype("fp64"), core.dtype("fp64")): ("__ppu_dmul_rn", core.dtype("fp64")),
            (core.dtype("fp32"), core.dtype("fp32")): ("__ppu_fmul_rn", core.dtype("fp32")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def mul_rz(arg0, arg1, _semantic=None):
    return core.extern_elementwise(
        "", "", [arg0, arg1], {
            (core.dtype("fp64"), core.dtype("fp64")): ("__ppu_dmul_rz", core.dtype("fp64")),
            (core.dtype("fp32"), core.dtype("fp32")): ("__ppu_fmul_rz", core.dtype("fp32")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def mul_rd(arg0, arg1, _semantic=None):
    return core.extern_elementwise(
        "", "", [arg0, arg1], {
            (core.dtype("fp64"), core.dtype("fp64")): ("__ppu_dmul_rd", core.dtype("fp64")),
            (core.dtype("fp32"), core.dtype("fp32")): ("__ppu_fmul_rd", core.dtype("fp32")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def mul_ru(arg0, arg1, _semantic=None):
    return core.extern_elementwise(
        "", "", [
            arg0,
            arg1,
        ], {
            (
                core.dtype("fp64"),
                core.dtype("fp64"),
            ): ("__ppu_dmul_ru", core.dtype("fp64")),
            (
                core.dtype("fp32"),
                core.dtype("fp32"),
            ): ("__ppu_fmul_ru", core.dtype("fp32")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def double2float_rn(arg0, _semantic=None):
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("fp64"), ): ("__ppu_double2float_rn", core.dtype("fp32")),
    }, is_pure=True, _semantic=_semantic)


@core.extern
def double2float_rz(arg0, _semantic=None):
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("fp64"), ): ("__ppu_double2float_rz", core.dtype("fp32")),
    }, is_pure=True, _semantic=_semantic)


@core.extern
def double2float_rd(arg0, _semantic=None):
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("fp64"), ): ("__ppu_double2float_rd", core.dtype("fp32")),
    }, is_pure=True, _semantic=_semantic)


@core.extern
def double2float_ru(arg0, _semantic=None):
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("fp64"), ): ("__ppu_double2float_ru", core.dtype("fp32")),
    }, is_pure=True, _semantic=_semantic)


@core.extern
def double2int_rn(arg0, _semantic=None):
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("fp64"), ): ("__ppu_double2int_rn", core.dtype("int32")),
    }, is_pure=True, _semantic=_semantic)


@core.extern
def double2int_rz(arg0, _semantic=None):
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("fp64"), ): ("__ppu_double2int_rz", core.dtype("int32")),
    }, is_pure=True, _semantic=_semantic)


@core.extern
def double2int_rd(arg0, _semantic=None):
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("fp64"), ): ("__ppu_double2int_rd", core.dtype("int32")),
    }, is_pure=True, _semantic=_semantic)


@core.extern
def double2int_ru(arg0, _semantic=None):
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("fp64"), ): ("__ppu_double2int_ru", core.dtype("int32")),
    }, is_pure=True, _semantic=_semantic)


@core.extern
def double2uint_rn(arg0, _semantic=None):
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("fp64"), ): ("__ppu_double2uint_rn", core.dtype("int32")),
    }, is_pure=True, _semantic=_semantic)


@core.extern
def double2uint_rz(arg0, _semantic=None):
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("fp64"), ): ("__ppu_double2uint_rz", core.dtype("int32")),
    }, is_pure=True, _semantic=_semantic)


@core.extern
def double2uint_rd(arg0, _semantic=None):
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("fp64"), ): ("__ppu_double2uint_rd", core.dtype("int32")),
    }, is_pure=True, _semantic=_semantic)


@core.extern
def double2uint_ru(arg0, _semantic=None):
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("fp64"), ): ("__ppu_double2uint_ru", core.dtype("int32")),
    }, is_pure=True, _semantic=_semantic)


@core.extern
def int2double_rn(arg0, _semantic=None):
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("int32"), ): ("__ppu_int2double_rn", core.dtype("fp64")),
    }, is_pure=True, _semantic=_semantic)


@core.extern
def uint2double_rn(arg0, _semantic=None):
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("uint32"), ): ("__ppu_uint2double_rn", core.dtype("fp64")),
    }, is_pure=True, _semantic=_semantic)


@core.extern
def float2int_rn(arg0, _semantic=None):
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("fp32"), ): ("__ppu_float2int_rn", core.dtype("int32")),
    }, is_pure=True, _semantic=_semantic)


@core.extern
def float2int_rz(arg0, _semantic=None):
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("fp32"), ): ("__ppu_float2int_rz", core.dtype("int32")),
    }, is_pure=True, _semantic=_semantic)


@core.extern
def float2int_rd(arg0, _semantic=None):
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("fp32"), ): ("__ppu_float2int_rd", core.dtype("int32")),
    }, is_pure=True, _semantic=_semantic)


@core.extern
def float2int_ru(arg0, _semantic=None):
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("fp32"), ): ("__ppu_float2int_ru", core.dtype("int32")),
    }, is_pure=True, _semantic=_semantic)


@core.extern
def float2uint_rn(arg0, _semantic=None):
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("fp32"), ): ("__ppu_float2uint_rn", core.dtype("int32")),
    }, is_pure=True, _semantic=_semantic)


@core.extern
def float2uint_rz(arg0, _semantic=None):
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("fp32"), ): ("__ppu_float2uint_rz", core.dtype("int32")),
    }, is_pure=True, _semantic=_semantic)


@core.extern
def float2uint_rd(arg0, _semantic=None):
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("fp32"), ): ("__ppu_float2uint_rd", core.dtype("int32")),
    }, is_pure=True, _semantic=_semantic)


@core.extern
def float2uint_ru(arg0, _semantic=None):
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("fp32"), ): ("__ppu_float2uint_ru", core.dtype("int32")),
    }, is_pure=True, _semantic=_semantic)


@core.extern
def int2float_rn(arg0, _semantic=None):
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("int32"), ): ("__ppu_int2float_rn", core.dtype("fp32")),
    }, is_pure=True, _semantic=_semantic)


@core.extern
def int2float_rz(arg0, _semantic=None):
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("int32"), ): ("__ppu_int2float_rz", core.dtype("fp32")),
    }, is_pure=True, _semantic=_semantic)


@core.extern
def int2float_rd(arg0, _semantic=None):
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("int32"), ): ("__ppu_int2float_rd", core.dtype("fp32")),
    }, is_pure=True, _semantic=_semantic)


@core.extern
def int2float_ru(arg0, _semantic=None):
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("int32"), ): ("__ppu_int2float_ru", core.dtype("fp32")),
    }, is_pure=True, _semantic=_semantic)


@core.extern
def uint2float_rn(arg0, _semantic=None):
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("uint32"), ): ("__ppu_uint2float_rn", core.dtype("fp32")),
    }, is_pure=True, _semantic=_semantic)


@core.extern
def uint2float_rz(arg0, _semantic=None):
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("uint32"), ): ("__ppu_uint2float_rz", core.dtype("fp32")),
    }, is_pure=True, _semantic=_semantic)


@core.extern
def uint2float_rd(arg0, _semantic=None):
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("uint32"), ): ("__ppu_uint2float_rd", core.dtype("fp32")),
    }, is_pure=True, _semantic=_semantic)


@core.extern
def uint2float_ru(arg0, _semantic=None):
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("uint32"), ): ("__ppu_uint2float_ru", core.dtype("fp32")),
    }, is_pure=True, _semantic=_semantic)


@core.extern
def hiloint2double(arg0, arg1, _semantic=None):
    return core.extern_elementwise("", "", [arg0, arg1], {
        (core.dtype("int32"), core.dtype("int32")): ("__ppu_hiloint2double", core.dtype("fp64")),
    }, is_pure=True, _semantic=_semantic)


@core.extern
def double2loint(arg0, _semantic=None):
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("fp64"), ): ("__ppu_double2loint", core.dtype("int32")),
    }, is_pure=True, _semantic=_semantic)


@core.extern
def double2hiint(arg0, _semantic=None):
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("fp64"), ): ("__ppu_double2hiint", core.dtype("int32")),
    }, is_pure=True, _semantic=_semantic)


@core.extern
def float2ll_rn(arg0, _semantic=None):
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("fp32"), ): ("__ppu_float2ll_rn", core.dtype("int64")),
    }, is_pure=True, _semantic=_semantic)


@core.extern
def float2ll_rz(arg0, _semantic=None):
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("fp32"), ): ("__ppu_float2ll_rz", core.dtype("int64")),
    }, is_pure=True, _semantic=_semantic)


@core.extern
def float2ll_rd(arg0, _semantic=None):
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("fp32"), ): ("__ppu_float2ll_rd", core.dtype("int64")),
    }, is_pure=True, _semantic=_semantic)


@core.extern
def float2ll_ru(arg0, _semantic=None):
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("fp32"), ): ("__ppu_float2ll_ru", core.dtype("int64")),
    }, is_pure=True, _semantic=_semantic)


@core.extern
def float2ull_rn(arg0, _semantic=None):
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("fp32"), ): ("__ppu_float2ull_rn", core.dtype("int64")),
    }, is_pure=True, _semantic=_semantic)


@core.extern
def float2ull_rz(arg0, _semantic=None):
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("fp32"), ): ("__ppu_float2ull_rz", core.dtype("int64")),
    }, is_pure=True, _semantic=_semantic)


@core.extern
def float2ull_rd(arg0, _semantic=None):
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("fp32"), ): ("__ppu_float2ull_rd", core.dtype("int64")),
    }, is_pure=True, _semantic=_semantic)


@core.extern
def float2ull_ru(arg0, _semantic=None):
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("fp32"), ): ("__ppu_float2ull_ru", core.dtype("int64")),
    }, is_pure=True, _semantic=_semantic)


@core.extern
def double2ll_rn(arg0, _semantic=None):
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("fp64"), ): ("__ppu_double2ll_rn", core.dtype("int64")),
    }, is_pure=True, _semantic=_semantic)


@core.extern
def double2ll_rz(arg0, _semantic=None):
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("fp64"), ): ("__ppu_double2ll_rz", core.dtype("int64")),
    }, is_pure=True, _semantic=_semantic)


@core.extern
def double2ll_rd(arg0, _semantic=None):
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("fp64"), ): ("__ppu_double2ll_rd", core.dtype("int64")),
    }, is_pure=True, _semantic=_semantic)


@core.extern
def double2ll_ru(arg0, _semantic=None):
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("fp64"), ): ("__ppu_double2ll_ru", core.dtype("int64")),
    }, is_pure=True, _semantic=_semantic)


@core.extern
def double2ull_rn(arg0, _semantic=None):
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("fp64"), ): ("__ppu_double2ull_rn", core.dtype("int64")),
    }, is_pure=True, _semantic=_semantic)


@core.extern
def double2ull_rz(arg0, _semantic=None):
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("fp64"), ): ("__ppu_double2ull_rz", core.dtype("int64")),
    }, is_pure=True, _semantic=_semantic)


@core.extern
def double2ull_rd(arg0, _semantic=None):
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("fp64"), ): ("__ppu_double2ull_rd", core.dtype("int64")),
    }, is_pure=True, _semantic=_semantic)


@core.extern
def double2ull_ru(arg0, _semantic=None):
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("fp64"), ): ("__ppu_double2ull_ru", core.dtype("int64")),
    }, is_pure=True, _semantic=_semantic)


@core.extern
def ll2float_rn(arg0, _semantic=None):
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("int64"), ): ("__ppu_ll2float_rn", core.dtype("fp32")),
    }, is_pure=True, _semantic=_semantic)


@core.extern
def ll2float_rz(arg0, _semantic=None):
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("int64"), ): ("__ppu_ll2float_rz", core.dtype("fp32")),
    }, is_pure=True, _semantic=_semantic)


@core.extern
def ll2float_rd(arg0, _semantic=None):
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("int64"), ): ("__ppu_ll2float_rd", core.dtype("fp32")),
    }, is_pure=True, _semantic=_semantic)


@core.extern
def ll2float_ru(arg0, _semantic=None):
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("int64"), ): ("__ppu_ll2float_ru", core.dtype("fp32")),
    }, is_pure=True, _semantic=_semantic)


@core.extern
def ull2float_rn(arg0, _semantic=None):
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("uint64"), ): ("__ppu_ull2float_rn", core.dtype("fp32")),
    }, is_pure=True, _semantic=_semantic)


@core.extern
def ull2float_rz(arg0, _semantic=None):
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("uint64"), ): ("__ppu_ull2float_rz", core.dtype("fp32")),
    }, is_pure=True, _semantic=_semantic)


@core.extern
def ull2float_rd(arg0, _semantic=None):
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("uint64"), ): ("__ppu_ull2float_rd", core.dtype("fp32")),
    }, is_pure=True, _semantic=_semantic)


@core.extern
def ull2float_ru(arg0, _semantic=None):
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("uint64"), ): ("__ppu_ull2float_ru", core.dtype("fp32")),
    }, is_pure=True, _semantic=_semantic)


@core.extern
def ll2double_rn(arg0, _semantic=None):
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("int64"), ): ("__ppu_ll2double_rn", core.dtype("fp64")),
    }, is_pure=True, _semantic=_semantic)


@core.extern
def ll2double_rz(arg0, _semantic=None):
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("int64"), ): ("__ppu_ll2double_rz", core.dtype("fp64")),
    }, is_pure=True, _semantic=_semantic)


@core.extern
def ll2double_rd(arg0, _semantic=None):
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("int64"), ): ("__ppu_ll2double_rd", core.dtype("fp64")),
    }, is_pure=True, _semantic=_semantic)


@core.extern
def ll2double_ru(arg0, _semantic=None):
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("int64"), ): ("__ppu_ll2double_ru", core.dtype("fp64")),
    }, is_pure=True, _semantic=_semantic)


@core.extern
def ull2double_rn(arg0, _semantic=None):
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("uint64"), ): ("__ppu_ull2double_rn", core.dtype("fp64")),
    }, is_pure=True, _semantic=_semantic)


@core.extern
def ull2double_rz(arg0, _semantic=None):
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("uint64"), ): ("__ppu_ull2double_rz", core.dtype("fp64")),
    }, is_pure=True, _semantic=_semantic)


@core.extern
def ull2double_rd(arg0, _semantic=None):
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("uint64"), ): ("__ppu_ull2double_rd", core.dtype("fp64")),
    }, is_pure=True, _semantic=_semantic)


@core.extern
def ull2double_ru(arg0, _semantic=None):
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("uint64"), ): ("__ppu_ull2double_ru", core.dtype("fp64")),
    }, is_pure=True, _semantic=_semantic)


@core.extern
def int_as_float(arg0, _semantic=None):
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("int32"), ): ("__ppu_int_as_float", core.dtype("fp32")),
    }, is_pure=True, _semantic=_semantic)


@core.extern
def float_as_int(arg0, _semantic=None):
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("fp32"), ): ("__ppu_float_as_int", core.dtype("int32")),
    }, is_pure=True, _semantic=_semantic)


@core.extern
def uint_as_float(arg0, _semantic=None):
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("uint32"), ): ("__ppu_uint_as_float", core.dtype("fp32")),
    }, is_pure=True, _semantic=_semantic)


@core.extern
def float_as_uint(arg0, _semantic=None):
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("fp32"), ): ("__ppu_float_as_uint", core.dtype("int32")),
    }, is_pure=True, _semantic=_semantic)


@core.extern
def longlong_as_double(arg0, _semantic=None):
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("int64"), ): ("__ppu_longlong_as_double", core.dtype("fp64")),
    }, is_pure=True, _semantic=_semantic)


@core.extern
def double_as_longlong(arg0, _semantic=None):
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("fp64"), ): ("__ppu_double_as_longlong", core.dtype("int64")),
    }, is_pure=True, _semantic=_semantic)


@core.extern
def fast_sinf(arg0, _semantic=None):
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("fp32"), ): ("__ppu_fast_sinf", core.dtype("fp32")),
    }, is_pure=True, _semantic=_semantic)


@core.extern
def fast_cosf(arg0, _semantic=None):
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("fp32"), ): ("__ppu_fast_cosf", core.dtype("fp32")),
    }, is_pure=True, _semantic=_semantic)


@core.extern
def fast_log2f(arg0, _semantic=None):
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("fp32"), ): ("__ppu_fast_log2f", core.dtype("fp32")),
    }, is_pure=True, _semantic=_semantic)


@core.extern
def fast_logf(arg0, _semantic=None):
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("fp32"), ): ("__ppu_fast_logf", core.dtype("fp32")),
    }, is_pure=True, _semantic=_semantic)


@core.extern
def fast_expf(arg0, _semantic=None):
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("fp32"), ): ("__ppu_fast_expf", core.dtype("fp32")),
    }, is_pure=True, _semantic=_semantic)


@core.extern
def fast_tanf(arg0, _semantic=None):
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("fp32"), ): ("__ppu_fast_tanf", core.dtype("fp32")),
    }, is_pure=True, _semantic=_semantic)


@core.extern
def fast_exp10f(arg0, _semantic=None):
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("fp32"), ): ("__ppu_fast_exp10f", core.dtype("fp32")),
    }, is_pure=True, _semantic=_semantic)


@core.extern
def fast_log10f(arg0, _semantic=None):
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("fp32"), ): ("__ppu_fast_log10f", core.dtype("fp32")),
    }, is_pure=True, _semantic=_semantic)


@core.extern
def fast_powf(arg0, arg1, _semantic=None):
    return core.extern_elementwise("", "", [arg0, arg1], {
        (core.dtype("fp32"), core.dtype("fp32")): ("__ppu_fast_powf", core.dtype("fp32")),
    }, is_pure=True, _semantic=_semantic)


@core.extern
def hadd(arg0, arg1, _semantic=None):
    return core.extern_elementwise(
        "", "", [arg0, arg1], {
            (core.dtype("int32"), core.dtype("int32")): ("__ppu_hadd", core.dtype("int32")),
            (core.dtype("uint32"), core.dtype("uint32")): ("__ppu_uhadd", core.dtype("uint32")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def rhadd(arg0, arg1, _semantic=None):
    return core.extern_elementwise(
        "", "", [arg0, arg1], {
            (core.dtype("int32"), core.dtype("int32")): ("__ppu_rhadd", core.dtype("int32")),
            (core.dtype("uint32"), core.dtype("uint32")): ("__ppu_urhadd", core.dtype("uint32")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def sub_rn(arg0, arg1, _semantic=None):
    return core.extern_elementwise(
        "", "", [arg0, arg1], {
            (core.dtype("fp32"), core.dtype("fp32")): ("__ppu_fsub_rn", core.dtype("fp32")),
            (core.dtype("fp64"), core.dtype("fp64")): ("__ppu_dsub_rn", core.dtype("fp64")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def sub_rz(arg0, arg1, _semantic=None):
    return core.extern_elementwise(
        "", "", [arg0, arg1], {
            (core.dtype("fp32"), core.dtype("fp32")): ("__ppu_fsub_rz", core.dtype("fp32")),
            (core.dtype("fp64"), core.dtype("fp64")): ("__ppu_dsub_rz", core.dtype("fp64")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def sub_rd(arg0, arg1, _semantic=None):
    return core.extern_elementwise(
        "", "", [arg0, arg1], {
            (core.dtype("fp32"), core.dtype("fp32")): ("__ppu_fsub_rd", core.dtype("fp32")),
            (core.dtype("fp64"), core.dtype("fp64")): ("__ppu_dsub_rd", core.dtype("fp64")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def sub_ru(arg0, arg1, _semantic=None):
    return core.extern_elementwise(
        "", "", [arg0, arg1], {
            (core.dtype("fp32"), core.dtype("fp32")): ("__ppu_fsub_ru", core.dtype("fp32")),
            (core.dtype("fp64"), core.dtype("fp64")): ("__ppu_dsub_ru", core.dtype("fp64")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def rsqrt_rn(arg0, _semantic=None):
    return core.extern_elementwise("", "", [
        arg0,
    ], {
        (core.dtype("fp32"), ): ("__ppu_frsqrt_rn", core.dtype("fp32")),
    }, is_pure=True, _semantic=_semantic)


@core.extern
def ffs(arg0, _semantic=None):
    return core.extern_elementwise(
        "", "", [
            arg0,
        ], {
            (core.dtype("int32"), ): ("__ppu_ffs", core.dtype("int32")),
            (core.dtype("int64"), ): ("__ppu_ffsll", core.dtype("int32")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def rint(arg0, _semantic=None):
    return core.extern_elementwise(
        "", "", [
            arg0,
        ], {
            (core.dtype("fp32"), ): ("__ppu_rintf", core.dtype("fp32")),
            (core.dtype("fp64"), ): ("__ppu_rint", core.dtype("fp64")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def llrint(arg0, _semantic=None):
    return core.extern_elementwise(
        "", "", [
            arg0,
        ], {
            (core.dtype("fp32"), ): ("__ppu_llrintf", core.dtype("int64")),
            (core.dtype("fp64"), ): ("__ppu_llrint", core.dtype("int64")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def nearbyint(arg0, _semantic=None):
    return core.extern_elementwise(
        "", "", [
            arg0,
        ], {
            (core.dtype("fp32"), ): ("__ppu_nearbyintf", core.dtype("fp32")),
            (core.dtype("fp64"), ): ("__ppu_nearbyint", core.dtype("fp64")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def isnan(arg0, _semantic=None):
    return core.extern_elementwise(
        "", "", [
            arg0,
        ], {
            (core.dtype("fp32"), ): ("__ppu_isnanf", core.dtype("int32")),
            (core.dtype("fp64"), ): ("__ppu_isnand", core.dtype("int32")),
        }, is_pure=True, _semantic=_semantic).to(core.int1, _semantic=_semantic)


@core.extern
def signbit(arg0, _semantic=None):
    return core.extern_elementwise(
        "", "", [
            arg0,
        ], {
            (core.dtype("fp32"), ): ("__ppu_signbitf", core.dtype("int32")),
            (core.dtype("fp64"), ): ("__ppu_signbitd", core.dtype("int32")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def copysign(arg0, arg1, _semantic=None):
    return core.extern_elementwise(
        "", "", [arg0, arg1], {
            (core.dtype("fp32"), core.dtype("fp32")): ("__ppu_copysignf", core.dtype("fp32")),
            (core.dtype("fp64"), core.dtype("fp64")): ("__ppu_copysign", core.dtype("fp64")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def finitef(arg0, _semantic=None):
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("fp32"), ): ("__ppu_finitef", core.dtype("int32")),
    }, is_pure=True, _semantic=_semantic).to(core.int1, _semantic=_semantic)


@core.extern
def isinf(arg0, _semantic=None):
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp32"), ): ("__ppu_isinff", core.dtype("int32")),
            (core.dtype("fp64"), ): ("__ppu_isinfd", core.dtype("int32")),
        }, is_pure=True, _semantic=_semantic).to(core.int1, _semantic=_semantic)


@core.extern
def nextafter(arg0, arg1, _semantic=None):
    return core.extern_elementwise(
        "", "", [arg0, arg1], {
            (core.dtype("fp32"), core.dtype("fp32")): ("__ppu_nextafterf", core.dtype("fp32")),
            (core.dtype("fp64"), core.dtype("fp64")): ("__ppu_nextafter", core.dtype("fp64")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def sin(arg0, _semantic=None):
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp32"), ): ("__ppu_sinf", core.dtype("fp32")),
            (core.dtype("fp64"), ): ("__ppu_sin", core.dtype("fp64")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def cos(arg0, _semantic=None):
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp32"), ): ("__ppu_cosf", core.dtype("fp32")),
            (core.dtype("fp64"), ): ("__ppu_cos", core.dtype("fp64")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def sinpi(arg0, _semantic=None):
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp32"), ): ("__ppu_sinpif", core.dtype("fp32")),
            (core.dtype("fp64"), ): ("__ppu_sinpi", core.dtype("fp64")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def cospi(arg0, _semantic=None):
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp32"), ): ("__ppu_cospif", core.dtype("fp32")),
            (core.dtype("fp64"), ): ("__ppu_cospi", core.dtype("fp64")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def tan(arg0, _semantic=None):
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp32"), ): ("__ppu_tanf", core.dtype("fp32")),
            (core.dtype("fp64"), ): ("__ppu_tan", core.dtype("fp64")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def log2(arg0, _semantic=None):
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp32"), ): ("__ppu_log2f", core.dtype("fp32")),
            (core.dtype("fp64"), ): ("__ppu_log2", core.dtype("fp64")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def exp(arg0, _semantic=None):
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp32"), ): ("__ppu_expf", core.dtype("fp32")),
            (core.dtype("fp64"), ): ("__ppu_exp", core.dtype("fp64")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def exp10(arg0, _semantic=None):
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp32"), ): ("__ppu_exp10f", core.dtype("fp32")),
            (core.dtype("fp64"), ): ("__ppu_exp10", core.dtype("fp64")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def cosh(arg0, _semantic=None):
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp32"), ): ("__ppu_coshf", core.dtype("fp32")),
            (core.dtype("fp64"), ): ("__ppu_cosh", core.dtype("fp64")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def sinh(arg0, _semantic=None):
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp32"), ): ("__ppu_sinhf", core.dtype("fp32")),
            (core.dtype("fp64"), ): ("__ppu_sinh", core.dtype("fp64")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def tanh(arg0, _semantic=None):
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp32"), ): ("__ppu_tanhf", core.dtype("fp32")),
            (core.dtype("fp64"), ): ("__ppu_tanh", core.dtype("fp64")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def atan2(arg0, arg1, _semantic=None):
    return core.extern_elementwise(
        "", "", [arg0, arg1], {
            (core.dtype("fp32"), core.dtype("fp32")): ("__ppu_atan2f", core.dtype("fp32")),
            (core.dtype("fp64"), core.dtype("fp64")): ("__ppu_atan2", core.dtype("fp64")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def atan(arg0, _semantic=None):
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp32"), ): ("__ppu_atanf", core.dtype("fp32")),
            (core.dtype("fp64"), ): ("__ppu_atan", core.dtype("fp64")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def asin(arg0, _semantic=None):
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp32"), ): ("__ppu_asinf", core.dtype("fp32")),
            (core.dtype("fp64"), ): ("__ppu_asin", core.dtype("fp64")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def acos(arg0, _semantic=None):
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp32"), ): ("__ppu_acosf", core.dtype("fp32")),
            (core.dtype("fp64"), ): ("__ppu_acos", core.dtype("fp64")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def log(arg0, _semantic=None):
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp32"), ): ("__ppu_logf", core.dtype("fp32")),
            (core.dtype("fp64"), ): ("__ppu_log", core.dtype("fp64")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def log10(arg0, _semantic=None):
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp32"), ): ("__ppu_log10f", core.dtype("fp32")),
            (core.dtype("fp64"), ): ("__ppu_log10", core.dtype("fp64")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def log1p(arg0, _semantic=None):
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp32"), ): ("__ppu_log1pf", core.dtype("fp32")),
            (core.dtype("fp64"), ): ("__ppu_log1p", core.dtype("fp64")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def acosh(arg0, _semantic=None):
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp32"), ): ("__ppu_acoshf", core.dtype("fp32")),
            (core.dtype("fp64"), ): ("__ppu_acosh", core.dtype("fp64")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def asinh(arg0, _semantic=None):
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp32"), ): ("__ppu_asinhf", core.dtype("fp32")),
            (core.dtype("fp64"), ): ("__ppu_asinh", core.dtype("fp64")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def atanh(arg0, _semantic=None):
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp32"), ): ("__ppu_atanhf", core.dtype("fp32")),
            (core.dtype("fp64"), ): ("__ppu_atanh", core.dtype("fp64")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def expm1(arg0, _semantic=None):
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp32"), ): ("__ppu_expm1f", core.dtype("fp32")),
            (core.dtype("fp64"), ): ("__ppu_expm1", core.dtype("fp64")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def hypot(arg0, arg1, _semantic=None):
    return core.extern_elementwise(
        "", "", [arg0, arg1], {
            (core.dtype("fp32"), core.dtype("fp32")): ("__ppu_hypotf", core.dtype("fp32")),
            (core.dtype("fp64"), core.dtype("fp64")): ("__ppu_hypot", core.dtype("fp64")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def rhypot(arg0, arg1, _semantic=None):
    return core.extern_elementwise(
        "", "", [arg0, arg1], {
            (core.dtype("fp32"), core.dtype("fp32")): ("__ppu_rhypotf", core.dtype("fp32")),
            (core.dtype("fp64"), core.dtype("fp64")): ("__ppu_rhypot", core.dtype("fp64")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def norm3d(arg0, arg1, arg2, _semantic=None):
    return core.extern_elementwise(
        "", "", [arg0, arg1, arg2], {
            (core.dtype("fp32"), core.dtype("fp32"), core.dtype("fp32")): ("__ppu_norm3df", core.dtype("fp32")),
            (core.dtype("fp64"), core.dtype("fp64"), core.dtype("fp64")): ("__ppu_norm3d", core.dtype("fp64")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def rnorm3d(arg0, arg1, arg2, _semantic=None):
    return core.extern_elementwise(
        "", "", [arg0, arg1, arg2], {
            (core.dtype("fp32"), core.dtype("fp32"), core.dtype("fp32")): ("__ppu_rnorm3df", core.dtype("fp32")),
            (core.dtype("fp64"), core.dtype("fp64"), core.dtype("fp64")): ("__ppu_rnorm3d", core.dtype("fp64")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def norm4d(arg0, arg1, arg2, arg3, _semantic=None):
    return core.extern_elementwise(
        "", "", [arg0, arg1, arg2, arg3], {
            (core.dtype("fp32"), core.dtype("fp32"), core.dtype("fp32"), core.dtype("fp32")):
            ("__ppu_norm4df", core.dtype("fp32")),
            (core.dtype("fp64"), core.dtype("fp64"), core.dtype("fp64"), core.dtype("fp64")):
            ("__ppu_norm4d", core.dtype("fp64")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def rnorm4d(arg0, arg1, arg2, arg3, _semantic=None):
    return core.extern_elementwise(
        "", "", [arg0, arg1, arg2, arg3], {
            (core.dtype("fp32"), core.dtype("fp32"), core.dtype("fp32"), core.dtype("fp32")):
            ("__ppu_rnorm4df", core.dtype("fp32")),
            (core.dtype("fp64"), core.dtype("fp64"), core.dtype("fp64"), core.dtype("fp64")):
            ("__ppu_rnorm4d", core.dtype("fp64")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def cbrt(arg0, _semantic=None):
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp32"), ): ("__ppu_cbrtf", core.dtype("fp32")),
            (core.dtype("fp64"), ): ("__ppu_cbrt", core.dtype("fp64")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def rcbrt(arg0, _semantic=None):
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp32"), ): ("__ppu_rcbrtf", core.dtype("fp32")),
            (core.dtype("fp64"), ): ("__ppu_rcbrt", core.dtype("fp64")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def j0(arg0, _semantic=None):
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp32"), ): ("__ppu_j0f", core.dtype("fp32")),
            (core.dtype("fp64"), ): ("__ppu_j0", core.dtype("fp64")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def j1(arg0, _semantic=None):
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp32"), ): ("__ppu_j1f", core.dtype("fp32")),
            (core.dtype("fp64"), ): ("__ppu_j1", core.dtype("fp64")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def y0(arg0, _semantic=None):
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp32"), ): ("__ppu_y0f", core.dtype("fp32")),
            (core.dtype("fp64"), ): ("__ppu_y0", core.dtype("fp64")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def y1(arg0, _semantic=None):
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp32"), ): ("__ppu_y1f", core.dtype("fp32")),
            (core.dtype("fp64"), ): ("__ppu_y1", core.dtype("fp64")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def yn(arg0, arg1, _semantic=None):
    return core.extern_elementwise(
        "", "", [arg0, arg1], {
            (core.dtype("int32"), core.dtype("fp32")): ("__ppu_ynf", core.dtype("fp32")),
            (core.dtype("int32"), core.dtype("fp64")): ("__ppu_yn", core.dtype("fp64")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def jn(arg0, arg1, _semantic=None):
    return core.extern_elementwise(
        "", "", [arg0, arg1], {
            (core.dtype("int32"), core.dtype("fp32")): ("__ppu_jnf", core.dtype("fp32")),
            (core.dtype("int32"), core.dtype("fp64")): ("__ppu_jn", core.dtype("fp64")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def cyl_bessel_i0(arg0, _semantic=None):
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp32"), ): ("__ppu_cyl_bessel_i0f", core.dtype("fp32")),
            (core.dtype("fp64"), ): ("__ppu_cyl_bessel_i0", core.dtype("fp64")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def cyl_bessel_i1(arg0, _semantic=None):
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp32"), ): ("__ppu_cyl_bessel_i1f", core.dtype("fp32")),
            (core.dtype("fp64"), ): ("__ppu_cyl_bessel_i1", core.dtype("fp64")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def erf(arg0, _semantic=None):
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp32"), ): ("__ppu_erff", core.dtype("fp32")),
            (core.dtype("fp64"), ): ("__ppu_erf", core.dtype("fp64")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def erfinv(arg0, _semantic=None):
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp32"), ): ("__ppu_erfinvf", core.dtype("fp32")),
            (core.dtype("fp64"), ): ("__ppu_erfinv", core.dtype("fp64")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def erfc(arg0, _semantic=None):
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp32"), ): ("__ppu_erfcf", core.dtype("fp32")),
            (core.dtype("fp64"), ): ("__ppu_erfc", core.dtype("fp64")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def erfcx(arg0, _semantic=None):
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp32"), ): ("__ppu_erfcxf", core.dtype("fp32")),
            (core.dtype("fp64"), ): ("__ppu_erfcx", core.dtype("fp64")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def erfcinv(arg0, _semantic=None):
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp32"), ): ("__ppu_erfcinvf", core.dtype("fp32")),
            (core.dtype("fp64"), ): ("__ppu_erfcinv", core.dtype("fp64")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def normcdfinv(arg0, _semantic=None):
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp32"), ): ("__ppu_normcdfinvf", core.dtype("fp32")),
            (core.dtype("fp64"), ): ("__ppu_normcdfinv", core.dtype("fp64")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def normcdf(arg0, _semantic=None):
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp32"), ): ("__ppu_normcdff", core.dtype("fp32")),
            (core.dtype("fp64"), ): ("__ppu_normcdf", core.dtype("fp64")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def lgamma(arg0, _semantic=None):
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp32"), ): ("__ppu_lgammaf", core.dtype("fp32")),
            (core.dtype("fp64"), ): ("__ppu_lgamma", core.dtype("fp64")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def ldexp(arg0, arg1, _semantic=None):
    return core.extern_elementwise(
        "", "", [arg0, arg1], {
            (core.dtype("fp32"), core.dtype("int32")): ("__ppu_ldexpf", core.dtype("fp32")),
            (core.dtype("fp64"), core.dtype("int32")): ("__ppu_ldexp", core.dtype("fp64")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def scalbn(arg0, arg1, _semantic=None):
    return core.extern_elementwise(
        "", "", [arg0, arg1], {
            (core.dtype("fp32"), core.dtype("int32")): ("__ppu_scalbnf", core.dtype("fp32")),
            (core.dtype("fp64"), core.dtype("int32")): ("__ppu_scalbn", core.dtype("fp64")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def fmod(arg0, arg1, _semantic=None):
    return core.extern_elementwise(
        "", "", [arg0, arg1], {
            (core.dtype("fp32"), core.dtype("fp32")): ("__ppu_fmodf", core.dtype("fp32")),
            (core.dtype("fp64"), core.dtype("fp64")): ("__ppu_fmod", core.dtype("fp64")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def remainder(arg0, arg1, _semantic=None):
    return core.extern_elementwise(
        "", "", [arg0, arg1], {
            (core.dtype("fp32"), core.dtype("fp32")): ("__ppu_remainderf", core.dtype("fp32")),
            (core.dtype("fp64"), core.dtype("fp64")): ("__ppu_remainder", core.dtype("fp64")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def fma(arg0, arg1, arg2, _semantic=None):
    return core.extern_elementwise(
        "", "", [arg0, arg1, arg2], {
            (core.dtype("fp32"), core.dtype("fp32"), core.dtype("fp32")): ("__ppu_fmaf", core.dtype("fp32")),
            (core.dtype("fp64"), core.dtype("fp64"), core.dtype("fp64")): ("__ppu_fma", core.dtype("fp64")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def pow(arg0, arg1, _semantic=None):
    return core.extern_elementwise(
        "", "", [arg0, arg1], {
            (core.dtype("fp32"), core.dtype("int32")): ("__ppu_powif", core.dtype("fp32")),
            (core.dtype("fp64"), core.dtype("int32")): ("__ppu_powi", core.dtype("fp64")),
            (core.dtype("fp32"), core.dtype("fp32")): ("__ppu_powf", core.dtype("fp32")),
            (core.dtype("fp64"), core.dtype("fp64")): ("__ppu_pow", core.dtype("fp64")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def tgamma(arg0, _semantic=None):
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp32"), ): ("__ppu_tgammaf", core.dtype("fp32")),
            (core.dtype("fp64"), ): ("__ppu_tgamma", core.dtype("fp64")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def round(arg0, _semantic=None):
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp32"), ): ("__ppu_roundf", core.dtype("fp32")),
            (core.dtype("fp64"), ): ("__ppu_round", core.dtype("fp64")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def llround(arg0, _semantic=None):
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp32"), ): ("__ppu_llroundf", core.dtype("int64")),
            (core.dtype("fp64"), ): ("__ppu_llround", core.dtype("int64")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def fdim(arg0, arg1, _semantic=None):
    return core.extern_elementwise(
        "", "", [arg0, arg1], {
            (core.dtype("fp32"), core.dtype("fp32")): ("__ppu_fdimf", core.dtype("fp32")),
            (core.dtype("fp64"), core.dtype("fp64")): ("__ppu_fdim", core.dtype("fp64")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def ilogb(arg0, _semantic=None):
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp32"), ): ("__ppu_ilogbf", core.dtype("int32")),
            (core.dtype("fp64"), ): ("__ppu_ilogb", core.dtype("int32")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def logb(arg0, _semantic=None):
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp32"), ): ("__ppu_logbf", core.dtype("fp32")),
            (core.dtype("fp64"), ): ("__ppu_logb", core.dtype("fp64")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def isfinited(arg0, _semantic=None):
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("fp64"), ): ("__ppu_isfinited", core.dtype("int32")),
    }, is_pure=True, _semantic=_semantic).to(core.int1, _semantic=_semantic)
