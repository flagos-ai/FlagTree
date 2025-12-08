import os
from triton.language import core
import triton.language.extra
from triton.language.extra import libdevice


def get_bool_env(env):
    s = os.getenv(env, "").lower()
    if (s == "1" or s == "true" or s == "on"):
        return True
    return False


# unary op
@core.extern
def abs(arg0, _builder=None):
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("int32"), ): ("__nv_abs", core.dtype("int32")),
            (core.dtype("int64"), ): ("__nv_llabs", core.dtype("int64")),
            (core.dtype("fp32"), ): ("__nv_fabsf", core.dtype("fp32")),
        }, is_pure=True, _builder=_builder)


@core.extern
def acos(arg0, _builder=None):
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("fp32"), ): ("__nv_acosf", core.dtype("fp32")),
    }, is_pure=True, _builder=_builder)


@core.extern
def acosh(arg0, _builder=None):
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("fp32"), ): ("__nv_acoshf", core.dtype("fp32")),
    }, is_pure=True, _builder=_builder)


@core.extern
def asin(arg0, _builder=None):
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("fp32"), ): ("__nv_asinf", core.dtype("fp32")),
    }, is_pure=True, _builder=_builder)


@core.extern
def asinh(arg0, _builder=None):
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("fp32"), ): ("__nv_asinhf", core.dtype("fp32")),
    }, is_pure=True, _builder=_builder)


@core.extern
def atan(arg0, _builder=None):
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("fp32"), ): ("__nv_atanf", core.dtype("fp32")),
    }, is_pure=True, _builder=_builder)


@core.extern
def atanh(arg0, _builder=None):
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("fp32"), ): ("__nv_atanhf", core.dtype("fp32")),
    }, is_pure=True, _builder=_builder)


@core.extern
def cbrt(arg0, _builder=None):
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("fp32"), ): ("__nv_cbrtf", core.dtype("fp32")),
    }, is_pure=True, _builder=_builder)


@core.extern
def ceil(arg0, _builder=None):
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("fp32"), ): ("__nv_ceilf", core.dtype("fp32")),
    }, is_pure=True, _builder=_builder)


@core.extern
def cos(arg0, _builder=None):
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("fp32"), ): ("__nv_cosf", core.dtype("fp32")),
    }, is_pure=True, _builder=_builder)


@core.extern
def cosh(arg0, _builder=None):
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("fp32"), ): ("__nv_coshf", core.dtype("fp32")),
    }, is_pure=True, _builder=_builder)


@core.extern
def cospi(arg0, _builder=None):
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("fp32"), ): ("__nv_cospif", core.dtype("fp32")),
    }, is_pure=True, _builder=_builder)


@core.extern
def erf(arg0, _builder=None):
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("fp32"), ): ("__nv_erff", core.dtype("fp32")),
    }, is_pure=True, _builder=_builder)


@core.extern
def exp10(arg0, _builder=None):
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("fp32"), ): ("__nv_exp10f", core.dtype("fp32")),
    }, is_pure=True, _builder=_builder)


@core.extern
def exp2(arg0, _builder=None):
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("fp32"), ): ("__nv_exp2f", core.dtype("fp32")),
    }, is_pure=True, _builder=_builder)


@core.extern
def exp(arg0, _builder=None):
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("fp32"), ): ("__nv_expf", core.dtype("fp32")),
    }, is_pure=True, _builder=_builder)


@core.extern
def expm1(arg0, _builder=None):
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("fp32"), ): ("__nv_expm1f", core.dtype("fp32")),
    }, is_pure=True, _builder=_builder)


@core.extern
def floor(arg0, _builder=None):
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("fp32"), ): ("__nv_floorf", core.dtype("fp32")),
    }, is_pure=True, _builder=_builder)


@core.extern
def rcp_rd(arg0, _builder=None):
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("fp32"), ): ("__nv_frcp_rd", core.dtype("fp32")),
    }, is_pure=True, _builder=_builder)


@core.extern
def rcp_rn(arg0, _builder=None):
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("fp32"), ): ("__nv_frcp_rn", core.dtype("fp32")),
    }, is_pure=True, _builder=_builder)


@core.extern
def rcp_ru(arg0, _builder=None):
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("fp32"), ): ("__nv_frcp_ru", core.dtype("fp32")),
    }, is_pure=True, _builder=_builder)


@core.extern
def rcp_rz(arg0, _builder=None):
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("fp32"), ): ("__nv_frcp_rz", core.dtype("fp32")),
    }, is_pure=True, _builder=_builder)


@core.extern
def rsqrt_rn(arg0, _builder=None):
    return core.extern_elementwise("", "", [
        arg0,
    ], {
        (core.dtype("fp32"), ): ("__nv_frsqrt_rn", core.dtype("fp32")),
    }, is_pure=True, _builder=_builder)


@core.extern
def sqrt_rd(arg0, _builder=None):
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("fp32"), ): ("__nv_fsqrt_rd", core.dtype("fp32")),
    }, is_pure=True, _builder=_builder)


@core.extern
def sqrt_rn(arg0, _builder=None):
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("fp32"), ): ("__nv_fsqrt_rn", core.dtype("fp32")),
    }, is_pure=True, _builder=_builder)


@core.extern
def sqrt_ru(arg0, _builder=None):
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("fp32"), ): ("__nv_fsqrt_ru", core.dtype("fp32")),
    }, is_pure=True, _builder=_builder)


@core.extern
def sqrt_rz(arg0, _builder=None):
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("fp32"), ): ("__nv_fsqrt_rz", core.dtype("fp32")),
    }, is_pure=True, _builder=_builder)


@core.extern
def j0(arg0, _builder=None):
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("fp32"), ): ("__nv_j0f", core.dtype("fp32")),
    }, is_pure=True, _builder=_builder)


@core.extern
def j1(arg0, _builder=None):
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("fp32"), ): ("__nv_j1f", core.dtype("fp32")),
    }, is_pure=True, _builder=_builder)


@core.extern
def lgamma(arg0, _builder=None):
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("fp32"), ): ("__nv_lgammaf", core.dtype("fp32")),
    }, is_pure=True, _builder=_builder)


@core.extern
def log10(arg0, _builder=None):
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("fp32"), ): ("__nv_log10f", core.dtype("fp32")),
    }, is_pure=True, _builder=_builder)


@core.extern
def log1p(arg0, _builder=None):
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("fp32"), ): ("__nv_log1pf", core.dtype("fp32")),
    }, is_pure=True, _builder=_builder)


@core.extern
def log2(arg0, _builder=None):
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("fp32"), ): ("__nv_log2f", core.dtype("fp32")),
    }, is_pure=True, _builder=_builder)


@core.extern
def log(arg0, _builder=None):
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("fp32"), ): ("__nv_logf", core.dtype("fp32")),
    }, is_pure=True, _builder=_builder)


@core.extern
def rint(arg0, _builder=None):
    return core.extern_elementwise("", "", [
        arg0,
    ], {
        (core.dtype("fp32"), ): ("__nv_rintf", core.dtype("fp32")),
    }, is_pure=True, _builder=_builder)


@core.extern
def round(arg0, _builder=None):
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("fp32"), ): ("__nv_roundf", core.dtype("fp32")),
    }, is_pure=True, _builder=_builder)


@core.extern
def rsqrt(arg0, _builder=None):
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("fp32"), ): ("__nv_rsqrtf", core.dtype("fp32")),
    }, is_pure=True, _builder=_builder)


@core.extern
def sin(arg0, _builder=None):
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("fp32"), ): ("__nv_sinf", core.dtype("fp32")),
    }, is_pure=True, _builder=_builder)


@core.extern
def sinh(arg0, _builder=None):
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("fp32"), ): ("__nv_sinhf", core.dtype("fp32")),
    }, is_pure=True, _builder=_builder)


@core.extern
def sqrt(arg0, _builder=None):
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("fp32"), ): ("__nv_sqrtf", core.dtype("fp32")),
    }, is_pure=True, _builder=_builder)


@core.extern
def tan(arg0, _builder=None):
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("fp32"), ): ("__nv_tanf", core.dtype("fp32")),
    }, is_pure=True, _builder=_builder)


@core.extern
def tanh(arg0, _builder=None):
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("fp32"), ): ("__nv_tanhf", core.dtype("fp32")),
    }, is_pure=True, _builder=_builder)


@core.extern
def trunc(arg0, _builder=None):
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("fp32"), ): ("__nv_truncf", core.dtype("fp32")),
    }, is_pure=True, _builder=_builder)


@core.extern
def y0(arg0, _builder=None):
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("fp32"), ): ("__nv_y0f", core.dtype("fp32")),
    }, is_pure=True, _builder=_builder)


@core.extern
def y1(arg0, _builder=None):
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("fp32"), ): ("__nv_y1f", core.dtype("fp32")),
    }, is_pure=True, _builder=_builder)


# unary int float
@core.extern
def ilogb(arg0, _builder=None):
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("fp32"), ): ("__nv_ilogbf", core.dtype("int32")),
    }, is_pure=True, _builder=_builder)


@core.extern
def signbit(arg0, _builder=None):
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("fp32"), ): ("__nv_signbitf", core.dtype("int32")),
    }, is_pure=True, _builder=_builder)


@core.extern
def float_as_int(arg0, _builder=None):
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("fp32"), ): ("__nv_float_as_int", core.dtype("int32")),
    }, is_pure=True, _builder=_builder)


@core.extern
def int_as_float(arg0, _builder=None):
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("int32"), ): ("__nv_int_as_float", core.dtype("fp32")),
    }, is_pure=True, _builder=_builder)


@core.extern
def float2int_rd(arg0, _builder=None):
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("fp32"), ): ("__nv_float2int_rd", core.dtype("int32")),
    }, is_pure=True, _builder=_builder)


@core.extern
def float2int_rn(arg0, _builder=None):
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("fp32"), ): ("__nv_float2int_rn", core.dtype("int32")),
    }, is_pure=True, _builder=_builder)


@core.extern
def float2int_ru(arg0, _builder=None):
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("fp32"), ): ("__nv_float2int_ru", core.dtype("int32")),
    }, is_pure=True, _builder=_builder)


@core.extern
def float2int_rz(arg0, _builder=None):
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("fp32"), ): ("__nv_float2int_rz", core.dtype("int32")),
    }, is_pure=True, _builder=_builder)


@core.extern
def float2uint_rd(arg0, _builder=None):
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("fp32"), ): ("__nv_float2uint_rd", core.dtype("uint32")),
    }, is_pure=True, _builder=_builder)


@core.extern
def float2uint_rn(arg0, _builder=None):
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("fp32"), ): ("__nv_float2uint_rn", core.dtype("uint32")),
    }, is_pure=True, _builder=_builder)


@core.extern
def float2uint_ru(arg0, _builder=None):
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("fp32"), ): ("__nv_float2uint_ru", core.dtype("uint32")),
    }, is_pure=True, _builder=_builder)


@core.extern
def float2uint_rz(arg0, _builder=None):
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("fp32"), ): ("__nv_float2uint_rz", core.dtype("uint32")),
    }, is_pure=True, _builder=_builder)


@core.extern
def int2float_rd(arg0, _builder=None):
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("int32"), ): ("__nv_int2float_rd", core.dtype("fp32")),
    }, is_pure=True, _builder=_builder)


@core.extern
def int2float_rn(arg0, _builder=None):
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("int32"), ): ("__nv_int2float_rn", core.dtype("fp32")),
    }, is_pure=True, _builder=_builder)


@core.extern
def int2float_ru(arg0, _builder=None):
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("int32"), ): ("__nv_int2float_ru", core.dtype("fp32")),
    }, is_pure=True, _builder=_builder)


@core.extern
def int2float_rz(arg0, _builder=None):
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("int32"), ): ("__nv_int2float_rz", core.dtype("fp32")),
    }, is_pure=True, _builder=_builder)


@core.extern
def uint2float_rd(arg0, _builder=None):
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("uint32"), ): ("__nv_uint2float_rd", core.dtype("fp32")),
    }, is_pure=True, _builder=_builder)


@core.extern
def uint2float_rn(arg0, _builder=None):
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("uint32"), ): ("__nv_uint2float_rn", core.dtype("fp32")),
    }, is_pure=True, _builder=_builder)


@core.extern
def uint2float_ru(arg0, _builder=None):
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("uint32"), ): ("__nv_uint2float_ru", core.dtype("fp32")),
    }, is_pure=True, _builder=_builder)


@core.extern
def uint2float_rz(arg0, _builder=None):
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("uint32"), ): ("__nv_uint2float_rz", core.dtype("fp32")),
    }, is_pure=True, _builder=_builder)


# unary int float


# unary half float
@core.extern
def float2half_rn(arg0, _builder=None):
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("fp32"), ): ("__nv_float2half_rn", core.dtype("uint16")),
    }, is_pure=True, _builder=_builder)


# unary half float


# unary long long float
@core.extern
def ffs(arg0, _builder=None):
    return core.extern_elementwise(
        "", "", [arg0], {
            # gcu don't support the return type int32
            # (core.dtype("int64"), ): ("__nv_ffsll", core.dtype("int32")),
            (
                core.dtype("int64"), ): ("__nv_ffsll", core.dtype("int64")),
        }, is_pure=True, _builder=_builder)


@core.extern
def llrint(arg0, _builder=None):
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("fp32"), ): ("__nv_llrintf", core.dtype("int64")),
    }, is_pure=True, _builder=_builder)


@core.extern
def llround(arg0, _builder=None):
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("fp32"), ): ("__nv_llroundf", core.dtype("int64")),
    }, is_pure=True, _builder=_builder)


@core.extern
def float2ll_rd(arg0, _builder=None):
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("fp32"), ): ("__nv_float2ll_rd", core.dtype("int64")),
    }, is_pure=True, _builder=_builder)


@core.extern
def float2ll_rn(arg0, _builder=None):
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("fp32"), ): ("__nv_float2ll_rn", core.dtype("int64")),
    }, is_pure=True, _builder=_builder)


@core.extern
def float2ll_ru(arg0, _builder=None):
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("fp32"), ): ("__nv_float2ll_ru", core.dtype("int64")),
    }, is_pure=True, _builder=_builder)


@core.extern
def float2ll_rz(arg0, _builder=None):
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("fp32"), ): ("__nv_float2ll_rz", core.dtype("int64")),
    }, is_pure=True, _builder=_builder)


@core.extern
def float2ull_rd(arg0, _builder=None):
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("fp32"), ): ("__nv_float2ull_rd", core.dtype("int64")),
    }, is_pure=True, _builder=_builder)


@core.extern
def float2ull_rn(arg0, _builder=None):
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("fp32"), ): ("__nv_float2ull_rn", core.dtype("int64")),
    }, is_pure=True, _builder=_builder)


@core.extern
def float2ull_ru(arg0, _builder=None):
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("fp32"), ): ("__nv_float2ull_ru", core.dtype("int64")),
    }, is_pure=True, _builder=_builder)


@core.extern
def float2ull_rz(arg0, _builder=None):
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("fp32"), ): ("__nv_float2ull_rz", core.dtype("int64")),
    }, is_pure=True, _builder=_builder)


@core.extern
def ll2float_rd(arg0, _builder=None):
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("int64"), ): ("__nv_ll2float_rd", core.dtype("fp32")),
    }, is_pure=True, _builder=_builder)


@core.extern
def ll2float_rn(arg0, _builder=None):
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("int64"), ): ("__nv_ll2float_rn", core.dtype("fp32")),
    }, is_pure=True, _builder=_builder)


@core.extern
def ll2float_ru(arg0, _builder=None):
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("int64"), ): ("__nv_ll2float_ru", core.dtype("fp32")),
    }, is_pure=True, _builder=_builder)


@core.extern
def ll2float_rz(arg0, _builder=None):
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("int64"), ): ("__nv_ll2float_rz", core.dtype("fp32")),
    }, is_pure=True, _builder=_builder)


@core.extern
def ull2float_rd(arg0, _builder=None):
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("uint64"), ): ("__nv_ull2float_rd", core.dtype("fp32")),
    }, is_pure=True, _builder=_builder)


@core.extern
def ull2float_rn(arg0, _builder=None):
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("uint64"), ): ("__nv_ull2float_rn", core.dtype("fp32")),
    }, is_pure=True, _builder=_builder)


@core.extern
def ull2float_ru(arg0, _builder=None):
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("uint64"), ): ("__nv_ull2float_ru", core.dtype("fp32")),
    }, is_pure=True, _builder=_builder)


@core.extern
def ull2float_rz(arg0, _builder=None):
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("uint64"), ): ("__nv_ull2float_rz", core.dtype("fp32")),
    }, is_pure=True, _builder=_builder)


# unary long long float
# unary op


# binary op
@core.extern
def max(arg0, arg1, _builder=None):
    return core.extern_elementwise(
        "", "", [arg0, arg1], {
            (core.dtype("int64"), core.dtype("int64")): ("__nv_llmax", core.dtype("int64")),
            (core.dtype("uint64"), core.dtype("uint64")): ("__nv_ullmax", core.dtype("uint64")),
            (core.dtype("int32"), core.dtype("int32")): ("__nv_max", core.dtype("int32")),
            (core.dtype("uint32"), core.dtype("uint32")): ("__nv_umax", core.dtype("uint32")),
            (core.dtype("fp32"), core.dtype("fp32")): ("__nv_fmaxf", core.dtype("fp32")),
        }, is_pure=True, _builder=_builder)


@core.extern
def min(arg0, arg1, _builder=None):
    return core.extern_elementwise(
        "", "", [arg0, arg1], {
            (core.dtype("int64"), core.dtype("int64")): ("__nv_llmin", core.dtype("int64")),
            (core.dtype("uint64"), core.dtype("uint64")): ("__nv_ullmin", core.dtype("uint64")),
            (core.dtype("int32"), core.dtype("int32")): ("__nv_min", core.dtype("int32")),
            (core.dtype("uint32"), core.dtype("uint32")): ("__nv_umin", core.dtype("uint32")),
            (core.dtype("fp32"), core.dtype("fp32")): ("__nv_fminf", core.dtype("fp32")),
        }, is_pure=True, _builder=_builder)


# binary i32
@core.extern
def mulhi(arg0, arg1, _builder=None):
    return core.extern_elementwise("", "", [arg0, arg1], {
        (core.dtype("uint32"), core.dtype("uint32")): ("__nv_umulhi", core.dtype("uint32")),
    }, is_pure=True, _builder=_builder)


# binary i32


# binary float
@core.extern
def atan2(arg0, arg1, _builder=None):
    return core.extern_elementwise("", "", [arg0, arg1], {
        (core.dtype("fp32"), core.dtype("fp32")): ("__nv_atan2f", core.dtype("fp32")),
    }, is_pure=True, _builder=_builder)


@core.extern
def copysign(arg0, arg1, _builder=None):
    return core.extern_elementwise("", "", [arg0, arg1], {
        (core.dtype("fp32"), core.dtype("fp32")): ("__nv_copysignf", core.dtype("fp32")),
    }, is_pure=True, _builder=_builder)


@core.extern
def add_rd(arg0, arg1, _builder=None):
    return core.extern_elementwise("", "", [arg0, arg1], {
        (core.dtype("fp32"), core.dtype("fp32")): ("__nv_fadd_rd", core.dtype("fp32")),
    }, is_pure=True, _builder=_builder)


@core.extern
def add_rn(arg0, arg1, _builder=None):
    return core.extern_elementwise("", "", [arg0, arg1], {
        (core.dtype("fp32"), core.dtype("fp32")): ("__nv_fadd_rn", core.dtype("fp32")),
    }, is_pure=True, _builder=_builder)


@core.extern
def add_ru(arg0, arg1, _builder=None):
    return core.extern_elementwise("", "", [arg0, arg1], {
        (core.dtype("fp32"), core.dtype("fp32")): ("__nv_fadd_ru", core.dtype("fp32")),
    }, is_pure=True, _builder=_builder)


@core.extern
def add_rz(arg0, arg1, _builder=None):
    return core.extern_elementwise("", "", [arg0, arg1], {
        (core.dtype("fp32"), core.dtype("fp32")): ("__nv_fadd_rz", core.dtype("fp32")),
    }, is_pure=True, _builder=_builder)


@core.extern
def fdim(arg0, arg1, _builder=None):
    return core.extern_elementwise("", "", [arg0, arg1], {
        (core.dtype("fp32"), core.dtype("fp32")): ("__nv_fdimf", core.dtype("fp32")),
    }, is_pure=True, _builder=_builder)


@core.extern
def div_rd(arg0, arg1, _builder=None):
    return core.extern_elementwise("", "", [arg0, arg1], {
        (core.dtype("fp32"), core.dtype("fp32")): ("__nv_fdiv_rd", core.dtype("fp32")),
    }, is_pure=True, _builder=_builder)


@core.extern
def div_rn(arg0, arg1, _builder=None):
    return core.extern_elementwise("", "", [arg0, arg1], {
        (core.dtype("fp32"), core.dtype("fp32")): ("__nv_fdiv_rn", core.dtype("fp32")),
    }, is_pure=True, _builder=_builder)


@core.extern
def div_ru(arg0, arg1, _builder=None):
    return core.extern_elementwise("", "", [arg0, arg1], {
        (core.dtype("fp32"), core.dtype("fp32")): ("__nv_fdiv_ru", core.dtype("fp32")),
    }, is_pure=True, _builder=_builder)


@core.extern
def div_rz(arg0, arg1, _builder=None):
    return core.extern_elementwise("", "", [arg0, arg1], {
        (core.dtype("fp32"), core.dtype("fp32")): ("__nv_fdiv_rz", core.dtype("fp32")),
    }, is_pure=True, _builder=_builder)


@core.extern
def fmod(arg0, arg1, _builder=None):
    return core.extern_elementwise("", "", [arg0, arg1], {
        (core.dtype("fp32"), core.dtype("fp32")): ("__nv_fmodf", core.dtype("fp32")),
    }, is_pure=True, _builder=_builder)


@core.extern
def mul_rd(arg0, arg1, _builder=None):
    return core.extern_elementwise("", "", [arg0, arg1], {
        (core.dtype("fp32"), core.dtype("fp32")): ("__nv_fmul_rd", core.dtype("fp32")),
    }, is_pure=True, _builder=_builder)


@core.extern
def mul_rn(arg0, arg1, _builder=None):
    return core.extern_elementwise("", "", [arg0, arg1], {
        (core.dtype("fp32"), core.dtype("fp32")): ("__nv_fmul_rn", core.dtype("fp32")),
    }, is_pure=True, _builder=_builder)


@core.extern
def mul_ru(arg0, arg1, _builder=None):
    return core.extern_elementwise("", "", [
        arg0,
        arg1,
    ], {
        (core.dtype("fp32"), core.dtype("fp32")): ("__nv_fmul_ru", core.dtype("fp32")),
    }, is_pure=True, _builder=_builder)


@core.extern
def mul_rz(arg0, arg1, _builder=None):
    return core.extern_elementwise("", "", [arg0, arg1], {
        (core.dtype("fp32"), core.dtype("fp32")): ("__nv_fmul_rz", core.dtype("fp32")),
    }, is_pure=True, _builder=_builder)


@core.extern
def sub_rd(arg0, arg1, _builder=None):
    return core.extern_elementwise("", "", [arg0, arg1], {
        (core.dtype("fp32"), core.dtype("fp32")): ("__nv_fsub_rd", core.dtype("fp32")),
    }, is_pure=True, _builder=_builder)


@core.extern
def sub_rn(arg0, arg1, _builder=None):
    return core.extern_elementwise("", "", [arg0, arg1], {
        (core.dtype("fp32"), core.dtype("fp32")): ("__nv_fsub_rn", core.dtype("fp32")),
    }, is_pure=True, _builder=_builder)


@core.extern
def sub_ru(arg0, arg1, _builder=None):
    return core.extern_elementwise("", "", [arg0, arg1], {
        (core.dtype("fp32"), core.dtype("fp32")): ("__nv_fsub_ru", core.dtype("fp32")),
    }, is_pure=True, _builder=_builder)


@core.extern
def sub_rz(arg0, arg1, _builder=None):
    return core.extern_elementwise("", "", [arg0, arg1], {
        (core.dtype("fp32"), core.dtype("fp32")): ("__nv_fsub_rz", core.dtype("fp32")),
    }, is_pure=True, _builder=_builder)


@core.extern
def hypot(arg0, arg1, _builder=None):
    return core.extern_elementwise("", "", [arg0, arg1], {
        (core.dtype("fp32"), core.dtype("fp32")): ("__nv_hypotf", core.dtype("fp32")),
    }, is_pure=True, _builder=_builder)


@core.extern
def pow(arg0, arg1, _builder=None):
    return core.extern_elementwise(
        "", "", [arg0, arg1], {
            (core.dtype("fp32"), core.dtype("int32")): ("__nv_powif", core.dtype("fp32")),
            (core.dtype("fp32"), core.dtype("fp32")): ("__nv_powf", core.dtype("fp32")),
        }, is_pure=True, _builder=_builder)


# binary float
# binary


# ternary
@core.extern
def fma(arg0, arg1, arg2, _builder=None):
    return core.extern_elementwise("", "", [arg0, arg1, arg2], {
        (core.dtype("fp32"), core.dtype("fp32"), core.dtype("fp32")): ("__nv_fmaf", core.dtype("fp32")),
    }, is_pure=True, _builder=_builder)


@core.extern
def fma_rd(arg0, arg1, arg2, _builder=None):
    return core.extern_elementwise("", "", [arg0, arg1, arg2], {
        (core.dtype("fp32"), core.dtype("fp32"), core.dtype("fp32")): ("__nv_fmaf_rd", core.dtype("fp32")),
    }, is_pure=True, _builder=_builder)


@core.extern
def fma_rn(arg0, arg1, arg2, _builder=None):
    return core.extern_elementwise("", "", [arg0, arg1, arg2], {
        (core.dtype("fp32"), core.dtype("fp32"), core.dtype("fp32")): ("__nv_fmaf_rn", core.dtype("fp32")),
    }, is_pure=True, _builder=_builder)


@core.extern
def fma_ru(arg0, arg1, arg2, _builder=None):
    return core.extern_elementwise("", "", [arg0, arg1, arg2], {
        (core.dtype("fp32"), core.dtype("fp32"), core.dtype("fp32")): ("__nv_fmaf_ru", core.dtype("fp32")),
    }, is_pure=True, _builder=_builder)


@core.extern
def fma_rz(arg0, arg1, arg2, _builder=None):
    return core.extern_elementwise("", "", [arg0, arg1, arg2], {
        (core.dtype("fp32"), core.dtype("fp32"), core.dtype("fp32")): ("__nv_fmaf_rz", core.dtype("fp32")),
    }, is_pure=True, _builder=_builder)


# ternary
# other
@core.extern
def isinf(arg0, _builder=None):
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("fp32"), ): ("__nv_isinff", core.dtype("int32")),
    }, is_pure=True, _builder=_builder).to(core.int1, _builder=_builder)


@core.extern
def isnan(arg0, _builder=None):
    return core.extern_elementwise("", "", [
        arg0,
    ], {
        (core.dtype("fp32"), ): ("__nv_isnanf", core.dtype("int32")),
    }, is_pure=True, _builder=_builder).to(core.int1, _builder=_builder)


@core.extern
def finitef(arg0, _builder=None):
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("fp32"), ): ("__nv_finitef", core.dtype("int32")),
    }, is_pure=True, _builder=_builder).to(core.int1, _builder=_builder)


# TODO: the implementation of logger is required to be added
@core.extern
def logger(arg0, _builder=None):
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("fp32"), ): ("__nv_dummy_logger", core.dtype("fp32")),
    }, is_pure=True, _builder=_builder)


@core.extern
def begin_clock(_builder=None):
    return core.extern_elementwise("", "", [], {
        (): ("__gcu_begin_clock", core.dtype("int64")),
    }, is_pure=False, _builder=_builder)


@core.extern
def end_clock(_builder=None):
    return core.extern_elementwise("", "", [], {
        (): ("__gcu_end_clock", core.dtype("int64")),
    }, is_pure=False, _builder=_builder)


# other

if (get_bool_env("TRITON_GCU_DEBUG")):
    libdevice.begin_clock = begin_clock
    libdevice.end_clock = end_clock

func_mapping = {
    # unary
    "abs": abs,
    "acos": acos,
    "acosh": acosh,
    "asin": asin,
    "asinh": asinh,
    "atan": atan,
    "atanh": atanh,
    "cbrt": cbrt,
    "ceil": ceil,
    "cos": cos,
    "cosh": cosh,
    "cospi": cospi,
    "erf": erf,
    "exp10": exp10,
    "exp2": exp2,
    "exp": exp,
    "expm1": expm1,
    "floor": floor,
    "rcp_rd": rcp_rd,
    "rcp_rn": rcp_rn,
    "rcp_ru": rcp_ru,
    "rcp_rz": rcp_rz,
    "rsqrt_rn": rsqrt_rn,
    "sqrt_rd": sqrt_rd,
    "sqrt_rn": sqrt_rn,
    "sqrt_ru": sqrt_ru,
    "sqrt_rz": sqrt_rz,
    "j0": j0,
    "j1": j1,
    "lgamma": lgamma,
    "log10": log10,
    "log1p": log1p,
    "log2": log2,
    "log": log,
    "rint": rint,
    "round": round,
    "rsqrt": rsqrt,
    "sin": sin,
    "sinh": sinh,
    "sqrt": sqrt,
    "tan": tan,
    "tanh": tanh,
    "trunc": trunc,
    "y0": y0,
    "y1": y1,
    # unary half float
    "float2half_rn": float2half_rn,
    # unary int float
    "ilogb": ilogb,
    "signbit": signbit,
    "float_as_int": float_as_int,
    "int_as_float": int_as_float,
    "float2int_rd": float2int_rd,
    "float2int_rn": float2int_rn,
    "float2int_ru": float2int_ru,
    "float2int_rz": float2int_rz,
    "float2uint_rd": float2uint_rd,
    "float2uint_rn": float2uint_rn,
    "float2uint_ru": float2uint_ru,
    "float2uint_rz": float2uint_rz,
    "int2float_rd": int2float_rd,
    "int2float_rn": int2float_rn,
    "int2float_ru": int2float_ru,
    "int2float_rz": int2float_rz,
    "uint2float_rd": uint2float_rd,
    "uint2float_rn": uint2float_rn,
    "uint2float_ru": uint2float_ru,
    "uint2float_rz": uint2float_rz,
    # unary long long float
    "ffs": ffs,
    "llrint": llrint,
    "llround": llround,
    "float2ll_rd": float2ll_rd,
    "float2ll_rn": float2ll_rn,
    "float2ll_ru": float2ll_ru,
    "float2ll_rz": float2ll_rz,
    "float2ull_rd": float2ull_rd,
    "float2ull_rn": float2ull_rn,
    "float2ull_ru": float2ull_ru,
    "float2ull_rz": float2ull_rz,
    "ll2float_rd": ll2float_rd,
    "ll2float_rn": ll2float_rn,
    "ll2float_ru": ll2float_ru,
    "ll2float_rz": ll2float_rz,
    "ull2float_rd": ull2float_rd,
    "ull2float_rn": ull2float_rn,
    "ull2float_ru": ull2float_ru,
    "ull2float_rz": ull2float_rz,
    # binary
    "max": max,
    "min": min,
    # binary i32
    "mulhi": mulhi,
    # binary float
    "atan2": atan2,
    "copysign": copysign,
    "add_rd": add_rd,
    "add_rn": add_rn,
    "add_ru": add_ru,
    "add_rz": add_rz,
    "fdim": fdim,
    "div_rd": div_rd,
    "div_rn": div_rn,
    "div_ru": div_ru,
    "div_rz": div_rz,
    "fmod": fmod,
    "mul_rd": mul_rd,
    "mul_rn": mul_rn,
    "mul_ru": mul_ru,
    "mul_rz": mul_rz,
    "sub_rd": sub_rd,
    "sub_rn": sub_rn,
    "sub_ru": sub_ru,
    "sub_rz": sub_rz,
    "hypot": hypot,
    "pow": pow,
    # ternary
    "fma": fma,
    "fma_rd": fma_rd,
    "fma_rn": fma_rn,
    "fma_ru": fma_ru,
    "fma_rz": fma_rz,
    # other
    "isinf": isinf,
    "isnan": isnan,
    "finitef": finitef,
    "logger": logger,
}

for name, func in func_mapping.items():
    if hasattr(libdevice, name):
        existing_func = getattr(libdevice, name)
        if existing_func is not None:
            setattr(libdevice, name, func)
