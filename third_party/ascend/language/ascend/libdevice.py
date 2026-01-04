from functools import wraps
from typing import List
import numbers
from triton.language import core
from triton.language.core import (
    _constexpr_to_value,
    constexpr,
    tensor,
    range
)
from triton.language.math import _add_math_1arg_docstr, _add_math_2arg_docstr, _add_math_3arg_docstr
from triton.language import semantic
from triton._C.libtriton import ir
from triton.language import math, semantic
from math import pi as math_pi

T = core.TypeVar('T')


def _check_dtype(dtypes: List[str]) -> T:
    """
    We're following libdevice's convention to check accepted data types for math functions.
    It is not a good practice to support all data types as accelerators/GPUs don't support
    many float16 and bfloat16 math operations.
    We should let the users know that they are using and invoke explicit cast to convert
    the data type to the supported one.
    """

    def wrapper(fn):

        @wraps(fn)
        def check(*args, **kwargs):
            # concatenate args and kwargs
            all_args = list(args) + list(kwargs.values())
            for arg in [a for a in all_args if isinstance(a, core.tensor)]:
                arg_type = arg.type.scalar.name
                if hasattr(arg, 'was_bool_to_int8') and arg.was_bool_to_int8:
                    # In Triton, int1 maps to the boolean type
                    arg_type = 'int1'
                if arg_type not in dtypes:
                    raise ValueError(f"Expected dtype {dtypes} but got {arg_type}")
            return fn(*args, **kwargs)

        return check

    return wrapper


@core.extern
@_check_dtype(dtypes=["int32", "uint32"])
@_add_math_2arg_docstr("most significant N bits of the 2N-bit product")
def umulhi(x, y, _builder=None):
    x = semantic.to_tensor(x, _builder)
    y = semantic.to_tensor(y, _builder)
    x, y = core.binary_op_type_legalization(x, y, _builder)
    return core.tensor(_builder.create_umulhi(x.handle, y.handle), x.type)

@core.extern
@_check_dtype(dtypes=["bf16", "fp16", "fp32", "fp8e4nv", "fp8e5"])
@_add_math_1arg_docstr("exponential")
@core._tensor_member_fn
def exp(x, _builder=None):
    x = semantic.to_tensor(x, _builder)
    return core.tensor(_builder.create_exp(x.handle), x.type)

@core.extern
@_check_dtype(dtypes=["bf16", "fp16", "fp32", "fp8e4nv", "fp8e5"])
@_add_math_1arg_docstr("exponential (base 2)")
@core._tensor_member_fn
def exp2(x, _builder=None):
    x = semantic.to_tensor(x, _builder)
    return core.tensor(_builder.create_exp2(x.handle), x.type)

@core.extern
@_check_dtype(dtypes=["bf16", "fp16", "fp32", "fp8e4nv", "fp8e5"])
@_add_math_1arg_docstr("natural logarithm")
@core._tensor_member_fn
def log(x, _builder=None):
    x = semantic.to_tensor(x, _builder)
    return core.tensor(_builder.create_log(x.handle), x.type)

@core.extern
@_check_dtype(dtypes=["bf16", "fp16", "fp32", "fp8e4nv", "fp8e5"])
@_add_math_1arg_docstr("logarithm (base 2)")
@core._tensor_member_fn
def log2(x, _builder=None):
    x = semantic.to_tensor(x, _builder)
    return core.tensor(_builder.create_log2(x.handle), x.type)

@core.extern
@_check_dtype(dtypes=["bf16", "fp16", "fp32", "fp8e4nv", "fp8e5"])
@_add_math_1arg_docstr("cosine")
@core._tensor_member_fn
def cos(x, _builder=None):
    x = semantic.to_tensor(x, _builder)
    return core.tensor(_builder.create_cos(x.handle), x.type)

@core.extern
@_check_dtype(dtypes=["bf16", "fp16", "fp32", "fp8e4nv", "fp8e5"])
@_add_math_1arg_docstr("sine")
@core._tensor_member_fn
def sin(x, _builder=None):
    x = semantic.to_tensor(x, _builder)
    return core.tensor(_builder.create_sin(x.handle), x.type)

@core.extern
@_check_dtype(dtypes=["bf16", "fp16", "fp32", "fp8e4nv", "fp8e5"])
@_add_math_1arg_docstr("fast square root")
@core._tensor_member_fn
def sqrt(x, _builder=None):
    x = semantic.to_tensor(x, _builder)
    return core.tensor(_builder.create_sqrt(x.handle), x.type)

@core.extern
@_check_dtype(dtypes=["bf16", "fp16", "fp32", "fp8e4nv", "fp8e5"])
@_add_math_1arg_docstr("precise square root (rounding to nearest wrt the IEEE standard)")
@core._tensor_member_fn
def sqrt_rn(x, _builder=None):
    x = semantic.to_tensor(x, _builder)
    return core.tensor(_builder.create_precise_sqrt(x.handle), x.type)

@core.extern
@_check_dtype(dtypes=["bf16", "fp16", "fp32", "fp8e4nv", "fp8e5"])
@_add_math_1arg_docstr("inverse square root")
@core._tensor_member_fn
def rsqrt(x, _builder=None):
    x = semantic.to_tensor(x, _builder)
    return core.tensor(_builder.create_rsqrt(x.handle), x.type)

@core.extern
@_check_dtype(dtypes=["bf16", "fp16", "fp32", "fp8e4nv", "fp8e5"])
@_add_math_2arg_docstr("precise division (rounding to nearest wrt the IEEE standard)")
def div_rn(x, y, _builder=None):
    x = semantic.to_tensor(x, _builder)
    y = semantic.to_tensor(y, _builder)
    x, y = core.binary_op_type_legalization(x, y, _builder)
    return core.tensor(_builder.create_precise_divf(x.handle, y.handle), x.type)

@core.extern
@_check_dtype(dtypes=["bf16", "fp16", "fp32", "fp8e4nv", "fp8e5"])
@_add_math_1arg_docstr("error function")
@core._tensor_member_fn
def erf(x, _builder=None):
    x = semantic.to_tensor(x, _builder)
    return core.tensor(_builder.create_erf(x.handle), x.type)

@core.extern
@_check_dtype(dtypes=["bf16", "fp16", "fp32"])
@_add_math_1arg_docstr("error function")
@core._tensor_member_fn
def tanh(x, _builder=None):
    x = semantic.to_tensor(x, _builder)
    return core.tensor(_builder.create_tanh(x.handle), x.type)

@core.extern
@_check_dtype(dtypes=["bf16", "fp16", "fp32", "fp8e4nv", "fp8e5"])
@_add_math_1arg_docstr("floor")
@core._tensor_member_fn
def floor(x, _builder=None):
    x = semantic.to_tensor(x, _builder)
    return core.tensor(_builder.create_floor(x.handle), x.type)


@core.extern
@_check_dtype(dtypes=["bf16", "fp16", "fp32"])
@_add_math_1arg_docstr("ceil")
@core._tensor_member_fn
def ceil(x, _builder=None):
    x = semantic.to_tensor(x, _builder)
    if x.type.scalar.is_int():
        return x
    elif x.type.scalar.is_floating():
        return core.tensor(_builder.create_ceil(x.handle), x.type)
    raise ValueError("ceil does not support boolean type")


@core.extern
@_check_dtype(dtypes=["bf16", "fp16", "fp32", "fp8e4nv", "fp8e5"])
@_add_math_3arg_docstr("fused multiply-add")
def fma(x, y, z, _builder=None):
    x = semantic.to_tensor(x, _builder)
    y = semantic.to_tensor(y, _builder)
    z = semantic.to_tensor(z, _builder)
    x, y = core.binary_op_type_legalization(x, y, _builder)
    z, x = core.binary_op_type_legalization(z, x, _builder)
    z, y = core.binary_op_type_legalization(z, y, _builder)
    return core.tensor(_builder.create_fma(x.handle, y.handle, z.handle), x.type)


@core.extern
def reciprocal(arg0, _builder=None):
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp32"),): ("__hmf_recipf", core.dtype("fp32")),
            (core.dtype("fp16"),): ("__hmf_recipDh", core.dtype("fp16")),
        }, is_pure=True, _builder=_builder)

@core.extern
def log1p(arg0, _builder=None):
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp32"),): ("__hmf_log1pf", core.dtype("fp32")),
            (core.dtype("fp16"),): ("__hmf_log1pDh", core.dtype("fp16")),
        }, is_pure=True, _builder=_builder)


@core.extern
def relu(arg0, _builder=None):
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp32"),): ("__hmf_reluf", core.dtype("fp32")),
            (core.dtype("fp16"),): ("__hmf_reluDh", core.dtype("fp16")),
        }, is_pure=True, _builder=_builder)


@core.extern
def isinf(arg0, _builder=None):
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp32"),): ("__hmf_isinf", core.dtype("int1")),
            (core.dtype("fp16"),): ("__hmf_isinf", core.dtype("int1")),
            (core.dtype("bf16"),): ("__hmf_isinf", core.dtype("int1")),
        }, is_pure=True, _builder=_builder)


@core.extern
def tan(arg0, _builder=None):
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp32"),): ("__hmf_tanf", core.dtype("fp32")),
            (core.dtype("fp16"),): ("__hmf_tanDh", core.dtype("fp16")),
        }, is_pure=True, _builder=_builder)


@core.extern
def atan(arg0, _builder=None):
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp32"),): ("__hmf_atanf", core.dtype("fp32")),
            (core.dtype("fp16"),): ("__hmf_atanDh", core.dtype("fp16")),
        }, is_pure=True, _builder=_builder)

@core.extern
def tanh(arg0, _builder=None):
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp32"), ): ("__hmf_tanhf", core.dtype("fp32")),
            (core.dtype("fp16"), ): ("__hmf_tanhDh", core.dtype("fp16")),
        }, is_pure=True, _builder=_builder)

@core.extern
def ilogb(arg0, _builder=None):
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp32"),): ("__hmf_ilogbf", core.dtype("fp32")),
            (core.dtype("fp16"),): ("__hmf_ilogbDh", core.dtype("fp16")),
        }, is_pure=True, _builder=_builder)


@core.extern
def ldexp(arg0, arg1, _builder=None):
    return core.extern_elementwise(
        "", "", [arg0, arg1], {
            (core.dtype("fp32"), core.dtype("fp32")): ("__hmf_ldexpf", core.dtype("fp32")),
            (core.dtype("fp16"), core.dtype("fp16")): ("__hmf_ldexpDh", core.dtype("fp16")),
        }, is_pure=True, _builder=_builder)

@core.extern
def pow(arg0, arg1, _builder=None):
    return core.extern_elementwise(
        "", "", [arg0, arg1], {
            (core.dtype("fp32"), core.dtype("fp32")): ("__hmf_powf", core.dtype("fp32")),
            (core.dtype("fp16"), core.dtype("fp16")): ("__hmf_powf", core.dtype("fp16")),
            (core.dtype("bf16"), core.dtype("bf16")): ("__hmf_powf", core.dtype("bf16")),
            (core.dtype("int64"), core.dtype("int64")): ("__hmf_powi", core.dtype("int64")),
            (core.dtype("int32"), core.dtype("int32")): ("__hmf_powi", core.dtype("int32")),
            (core.dtype("int16"), core.dtype("int16")): ("__hmf_powi", core.dtype("int16")),
            (core.dtype("int8"), core.dtype("int8")): ("__hmf_powi", core.dtype("int8")),
        }, is_pure=True, _builder=_builder)

@core.extern
def isnan(arg0, _builder=None):
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp32"),): ("__hmf_isnan", core.dtype("int1")),
            (core.dtype("fp16"),): ("__hmf_isnan", core.dtype("int1")),
            (core.dtype("bf16"),): ("__hmf_isnan", core.dtype("int1")),
        }, is_pure=True, _builder=_builder)

@core.extern
def flip(arg0, arg1=None, _builder=None):
    if _builder is None:
        from triton.language import core as tl
        _builder = tl.get_builder()
    from triton.language.core import flip as ascend_flip
    return ascend_flip(arg0, arg1, _builder=_builder)

@core.extern
def atan2(arg0, arg1, _builder=None):
    core.static_print("The func atan2 is supported in math.atan2 lowlevel. So use math.atan2 instead.")
    return math.atan2(arg1, arg0)

@core.extern
def div_rz(arg0, arg1, _builder=None):
    core.static_print("tl.div_rz is unsupported for now. Use libdevice.div_rz instead.")
    core.static_assert(False)

@core.extern
def fmod(arg0, arg1, _builder=None):
    core.static_print("tl.fmod is unsupported for now. Use libdevice.fmod instead.")
    core.static_assert(False)

@core.extern
def round(arg0, _builder=None):
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp32"), ): ("__hmf_roundf", core.dtype("fp32")),
        }, is_pure=True, _builder=_builder)

@core.extern
@_add_math_2arg_docstr("cdiv")
@core._tensor_member_fn
def cdiv(x, div, _builder=None):
    if isinstance(x, core.constexpr):
        x = x.value
    if isinstance(div, core.constexpr):
        div = div.value
    from math import ceil as py_ceil
    if isinstance(x, numbers.Number) and isinstance(div, numbers.Number):
        if isinstance(x, bool) or isinstance(div, bool):
            raise TypeError("cdiv does not support boolean type")
        if isinstance(x, int) and isinstance(div, int):
            res = x //div
            rem = x % div
            return res + (1 if rem != 0 else 0)
        else:
            return py_ceil(x / div)

    x = semantic.to_tensor(x, _builder)
    div = semantic.to_tensor(div, _builder)
    x_scalar_type = x.type.scalar
    div_scalar_type = div.type.scalar
    if x_scalar_type.is_bool() or div_scalar_type.is_bool():
        raise ValueError("cdiv does not support boolean type")
    elif x_scalar_type.is_int() and div_scalar_type.is_int():
        # integer cdiv: (x + div - 1) // div as before
        return semantic.floordiv(
            semantic.add(x, semantic.sub(div, 1, True, _builder), True, _builder),
            div,
            _builder
        )
    else:
        div_res = semantic.truediv(x, div, _builder)
        cdiv_res = core.tensor(_builder.create_ceil(div_res.handle), div_res.type)
        return semantic.cast(cdiv_res, x_scalar_type, _builder)

@core.builtin
@math._check_dtype(dtypes=["bf16", "fp16", "fp32"])
@math._add_math_1arg_docstr("acos")
def acos(arg0: core.tensor, _builder: ir.builder):
    pi = 3.1415926536
    pi_half = 1.5707963268
    sqrt2 = 1.4142135624
    eps = 1e-8

    # |x| < 0.5, acos(x) = pi/2 - [x + x*x²*(0.1666667 + x²*(0.075 + x²*(0.0446429 + 0.0303810*x²))]
    arg0 = semantic.to_tensor(arg0, _builder)
    abs_x = math.abs(arg0, _builder=_builder)
    dtype = arg0.dtype
    arg0_2 = semantic.mul(arg0, arg0, True, _builder)
    arg0_4 = semantic.mul(arg0_2, arg0_2, True, _builder)
    arg0_6 = semantic.mul(arg0_4, arg0_2, True, _builder)
    arg0_8 = semantic.mul(arg0_6, arg0_2, True, _builder)
    arg0_10 = semantic.mul(arg0_8, arg0_2, True, _builder)
    poly = semantic.add(1.0, semantic.mul(0.166667, arg0_2, True, _builder), True, _builder)
    poly = semantic.add(poly, semantic.mul(0.075, arg0_4, True, _builder), True, _builder)
    poly = semantic.add(poly, semantic.mul(0.044643, arg0_6, True, _builder), True, _builder)
    poly = semantic.add(poly, semantic.mul(0.030380, arg0_8, True, _builder), True, _builder)
    poly = semantic.add(poly, semantic.mul(0.022372, arg0_10, True, _builder), True, _builder)
    acos_center = semantic.sub(pi_half, semantic.mul(arg0, poly, True, _builder), True, _builder)

    # 0.5<|x|<0.9, acos(x) = 2*arctan(t), t=sqrt((1-abs_x)/(1+abs_x))
    numerator_mid = semantic.sub(1.0, abs_x, True, _builder)
    denom_mid = semantic.add(1.0, abs_x, True, _builder)
    div_mid = semantic.truediv(numerator_mid, denom_mid,  _builder)
    t_mid = math.sqrt(div_mid, _builder=_builder)
    t2_mid = semantic.mul(t_mid, t_mid, True, _builder)
    t4_mid = semantic.mul(t2_mid, t2_mid, True, _builder)
    t6_mid = semantic.mul(t4_mid, t2_mid, True, _builder)

    # 1 + t2*(-0.3333310 + t2*(0.1999341 + t2*(-0.1420890 + t2*0.1065976)))
    poly_mid1 = semantic.mul(0.1065976, t2_mid, True, _builder)
    poly_mid2 = semantic.add(-0.1420890, poly_mid1, True, _builder)
    poly_mid3 = semantic.mul(poly_mid2, t2_mid, True, _builder)
    poly_mid4 = semantic.add(0.1999341, poly_mid3, True, _builder)
    poly_mid5 = semantic.mul(poly_mid4, t2_mid, True, _builder)
    poly_mid6 = semantic.add(-0.3333310, poly_mid5, True, _builder)
    poly_mid = semantic.add(1.0, semantic.mul(poly_mid6, t2_mid, True, _builder), True, _builder)
    arctan_t = semantic.mul(t_mid, poly_mid, True, _builder)
    acos_mid = semantic.mul(2.0, arctan_t, True, _builder)
    is_neg_mid = semantic.less_than(arg0, 0.0, _builder)
    acos_mid_signed = semantic.where(is_neg_mid, semantic.sub(pi, acos_mid, True, _builder), acos_mid, _builder)

    is_center = semantic.less_than(abs_x, 0.5,  _builder)
    res_mid_boundary = semantic.where(is_center, acos_center, acos_mid_signed, _builder)
    return res_mid_boundary

@core.builtin
@math._check_dtype(dtypes=["bf16", "fp16", "fp32"])
@math._add_math_1arg_docstr("sinh")
def sinh(arg0: core.tensor, _builder: ir.builder):
    arg0 = semantic.to_tensor(arg0, _builder)
    exp0 = core.tensor(_builder.create_exp(arg0.handle), arg0.type)
    exp1 = semantic.truediv(1.0, exp0, _builder)
    tmp = semantic.sub(exp0, exp1, True, _builder)
    ret = semantic.truediv(tmp, 2.0, _builder)
    return ret

@core.builtin
@math._check_dtype(dtypes=["bf16", "fp16", "fp32"])
@math._add_math_1arg_docstr("cosh")
def cosh(arg0: core.tensor, _builder: ir.builder):
    arg0 = semantic.to_tensor(arg0, _builder)
    exp0 = core.tensor(_builder.create_exp(arg0.handle), arg0.type)
    exp1 = semantic.truediv(1.0, exp0, _builder)
    tmp = semantic.add(exp0, exp1, True, _builder)
    ret = semantic.truediv(tmp, 2.0, _builder)
    return ret

@core.builtin
@math._check_dtype(dtypes=["bf16", "fp16", "fp32"])
@math._add_math_1arg_docstr("acosh")
def acosh(arg0: core.tensor, _builder: ir.builder):
    arg0 = semantic.to_tensor(arg0, _builder)
    tmp = semantic.sub(semantic.mul(arg0, arg0, True, _builder), 1.0, True, _builder)
    sqrt_res = core.tensor(_builder.create_sqrt(tmp.handle), tmp.type)
    sum_res = semantic.add(arg0, sqrt_res, True, _builder)
    return core.tensor(_builder.create_log(sum_res.handle), sum_res.type)

@core.builtin
@math._check_dtype(dtypes=["bf16", "fp16", "fp32"])
@math._add_math_1arg_docstr("asinh")
def asinh(arg0: core.tensor, _builder: ir.builder):
    arg0 = semantic.to_tensor(arg0, _builder)
    tmp = semantic.add(semantic.mul(arg0, arg0, True, _builder), 1.0, True, _builder)
    sqrt_res = core.tensor(_builder.create_sqrt(tmp.handle), tmp.type)
    sum_res = semantic.add(arg0, sqrt_res, True, _builder)
    return core.tensor(_builder.create_log(sum_res.handle), sum_res.type)

@core.builtin
@math._check_dtype(dtypes=["bf16", "fp16", "fp32"])
@math._add_math_1arg_docstr("atanh")
def atanh(arg0: core.tensor, _builder: ir.builder):
    arg0 = semantic.to_tensor(arg0, _builder)
    a = semantic.add(1.0, arg0, True, _builder)
    b = semantic.sub(1.0, arg0, True, _builder)
    lna = core.tensor(_builder.create_log(a.handle), a.type)
    lnb = core.tensor(_builder.create_log(b.handle), b.type)
    tmp = semantic.sub(lna, lnb, True, _builder)
    return semantic.mul(tmp, 0.5, True, _builder)

@core.builtin
@math._check_dtype(dtypes=["bf16", "fp16", "fp32"])
@math._add_math_1arg_docstr("expm1")
def expm1(arg0: core.tensor, _builder: ir.builder):
    arg0 = semantic.to_tensor(arg0, _builder)
    tmp = core.tensor(_builder.create_exp(arg0.handle), arg0.type)
    return semantic.sub(tmp, 1, True, _builder)

@core.builtin
@math._check_dtype(dtypes=["bf16", "fp16", "fp32"])
@math._add_math_2arg_docstr("nextafter")
def nextafter(arg0: core.tensor, arg1: core.tensor, _builder: ir.builder):
    x = semantic.to_tensor(arg0, _builder)
    y = semantic.to_tensor(arg1, _builder)
    dtype_map = {
        "bf16": core.int16,
        "fp16": core.int16,
        "fp32": core.int32
    }
    min_pos_bit = {
        "bf16": 0x0001,
        "fp16": 0x0001,
        "fp32": 0x00000001
    }
    max_neg_bit = {
        "bf16": 0x8001,
        "fp16": 0x8001,
        "fp32": 0x80000001
    }
    int_type = dtype_map[x.type.scalar.name]
    x_eq_y = semantic.equal(x, y, _builder)
    x_gt_0 = semantic.greater_than(x, 0, _builder)
    y_gt_x = semantic.greater_than(y, x, _builder)
    next_neg = semantic.xor_(x_gt_0, y_gt_x, _builder)
    next_pos = semantic.not_(next_neg, _builder)

    p1 = semantic.full(x.shape, 1, int_type, _builder)
    n1 = semantic.full(x.shape, -1, int_type, _builder)
    dir_xy = semantic.where(next_pos, p1, n1, _builder)
    x_abs = math.abs(x, _builder=_builder)
    x_is_0 = semantic.equal(x_abs, 0, _builder)

    min_pos = semantic.full(x.shape, min_pos_bit[x.type.scalar.name], int_type, _builder)
    max_neg = semantic.full(x.shape, max_neg_bit[x.type.scalar.name], int_type, _builder)
    min_pos = semantic.bitcast(min_pos, x.dtype, _builder)
    max_neg = semantic.bitcast(max_neg, x.dtype, _builder)
    bits_x = semantic.bitcast(x, int_type, _builder)
    bits_next = semantic.add(bits_x, dir_xy, True, _builder)
    next_val = semantic.bitcast(bits_next, x.dtype, _builder)

    need_min_pos = semantic.logical_and(x_is_0, next_pos, _builder)
    need_max_neg = semantic.logical_and(x_is_0, next_neg, _builder)
    next_val = semantic.where(need_min_pos, min_pos, next_val, _builder)
    next_val = semantic.where(need_max_neg, max_neg, next_val, _builder)
    return semantic.where(x_eq_y, x, next_val, _builder)

@core.builtin
@math._check_dtype(dtypes=["bf16", "fp16", "fp32"])
@math._add_math_2arg_docstr("hypot(Euclidean Distance)")
def hypot(arg0: core.tensor, arg1: core.tensor, _builder: ir.builder):
    arg0 = semantic.to_tensor(arg0, _builder)
    arg1 = semantic.to_tensor(arg1, _builder)
    x2 = semantic.mul(arg0, arg0, True, _builder)
    y2 = semantic.mul(arg1, arg1, True, _builder)
    sum_res = semantic.add(x2, y2, True, _builder)
    return core.tensor(_builder.create_sqrt(sum_res.handle), sum_res.type)

# This function is derived from the Cephes Math Library release 2.8: June, 2000
# https://netlib.org/cephes/
# Copyright (c) 1984, 1987, 2000 by Stephen L. Moshier
# All rights reserved.
@core.builtin
@math._check_dtype(dtypes=["fp16", "fp32"])
@math._add_math_2arg_docstr("besseli0 (Modified Bessel function of the first kind, order 0).")
def cyl_bessel_i0(arg0: core.tensor, _builder: ir.builder):
    param1 = [
            -4.41534164647933937950e-18,
            +3.33079451882223809783e-17,
            -2.43127984654795469359e-16,
            +1.71539128555513303061e-15,
            -1.16853328779934516808e-14,
            +7.67618549860493561688e-14,
            -4.85644678311192946090e-13,
            +2.95505266312963983461e-12,
            -1.72682629144155570723e-11,
            +9.67580903537323691224e-11,
            -5.18979560163526290666e-10,
            +2.65982372468238665035e-09,
            -1.30002500998624804212e-08,
            +6.04699502254191894932e-08,
            -2.67079385394061173391e-07,
            +1.11738753912010371815e-06,
            -4.41673835845875056359e-06,
            +1.64484480707288970893e-05,
            -5.75419501008210370398e-05,
            +1.88502885095841655729e-04,
            -5.76375574538582365885e-04,
            +1.63947561694133579842e-03,
            -4.32430999505057594430e-03,
            +1.05464603945949983183e-02,
            -2.37374148058994688156e-02,
            +4.93052842396707084878e-02,
            -9.49010970480476444210e-02,
            +1.71620901522208775349e-01,
            -3.04682672343198398683e-01,
            +6.76795274409476084995e-01,
    ]
    param2 = [
            -7.23318048787475395456e-18,
            -4.83050448594418207126e-18,
            +4.46562142029675999901e-17,
            +3.46122286769746109310e-17,
            -2.82762398051658348494e-16,
            -3.42548561967721913462e-16,
            +1.77256013305652638360e-15,
            +3.81168066935262242075e-15,
            -9.55484669882830764870e-15,
            -4.15056934728722208663e-14,
            +1.54008621752140982691e-14,
            +3.85277838274214270114e-13,
            +7.18012445138366623367e-13,
            -1.79417853150680611778e-12,
            -1.32158118404477131188e-11,
            -3.14991652796324136454e-11,
            +1.18891471078464383424e-11,
            +4.94060238822496958910e-10,
            +3.39623202570838634515e-09,
            +2.26666899049817806459e-08,
            +2.04891858946906374183e-07,
            +2.89137052083475648297e-06,
            +6.88975834691682398426e-05,
            +3.36911647825569408990e-03,
            +8.04490411014108831608e-01,
    ]
    arg0 = semantic.to_tensor(arg0, _builder)
    abs_x = core.tensor(_builder.create_fabs(arg0.handle), arg0.type)
    x_a = semantic.sub(semantic.mul(abs_x, 0.5, True, _builder), 2.0, True, _builder)
    a_n_2 = 0
    a_n_1 = 0
    a_n = param1[0]
    for i in range(1, 30):
        a_n_2 = a_n_1
        a_n_1 = a_n
        a_n = semantic.sub(semantic.mul(x_a, a_n_1, True, _builder), a_n_2, True, _builder)
        a_n = semantic.add(a_n, param1[i], True, _builder)

    f_32 = semantic.full(abs_x.shape, 32.0, abs_x.type.scalar, _builder)
    x_b = semantic.sub(semantic.fdiv(f_32, abs_x, True, _builder), 2.0, True, _builder)
    b_n_2 = 0
    b_n_1 = 0
    b_n = param2[0]
    for i in range(1, 25):
        b_n_2 = b_n_1
        b_n_1 = b_n
        b_n = semantic.sub(semantic.mul(x_b, b_n_1, True, _builder), b_n_2, True, _builder)
        b_n = semantic.add(b_n, param2[i], True, _builder)

    half_exp = semantic.mul(core.tensor(_builder.create_exp(abs_x.handle), abs_x.type), 0.5, True, _builder)
    res_a = semantic.mul(half_exp, semantic.sub(a_n, a_n_2, True, _builder), True, _builder)
    res_b = semantic.fdiv(semantic.mul(half_exp, semantic.sub(b_n, b_n_2, True, _builder), True, _builder), \
        core.tensor(_builder.create_sqrt(abs_x.handle), abs_x.type), True, _builder)
    cond = semantic.less_equal(abs_x, 8.0, _builder)
    res = semantic.where(cond, res_a, res_b, _builder)
    return res


@core.extern
@math._check_dtype(dtypes=["fp16", "fp32"])
def signbit(arg0, _builder=None):
    arg0_scalar_ty = arg0.type.scalar
    if arg0_scalar_ty == core.float32:
        int_ty = core.int32
    else: # arg0 type: float16 / bfloat16
        int_ty = core.int16

    arg0 = semantic.to_tensor(arg0, _builder)
    int_tensor = semantic.bitcast(arg0, int_ty, _builder)
    if int_ty == core.int32:
        shift = 31
    elif int_ty == core.int16:
        shift = 15

    shift = semantic.full(arg0.shape, shift, int_ty, _builder)
    sign_bit_tensor = semantic.lshr(int_tensor, shift, _builder)
    sign_bit_tensor = semantic.and_(
        sign_bit_tensor, semantic.full(arg0.shape, 1, int_ty, _builder), _builder)
    return semantic.equal(sign_bit_tensor, 1, _builder)


# Note:
# For inputs x very close to ±1 (criterion: 1 - |x| < 1.1e-4), erfinv(x) → ±∞ and the
# inverse error function becomes extremely sensitive to tiny changes in x. The asymptotic
# behavior includes terms like sqrt(-ln(1-|x|)), so tiny relative changes in (1-|x|) map
# to large absolute changes in erfinv, leading to numerical instability and loss of precision,
# resulting in deviations from the reference results.
@core.extern
@math._check_dtype(dtypes=["fp32"])
def erfinv(arg0, _builder=None):
    arg0_scalar_ty = arg0.type.scalar
    arg0 = semantic.to_tensor(arg0, _builder)

    inv_sqrt_pi_times_2 = semantic.full(
        arg0.shape, 1.128379167, arg0_scalar_ty, _builder).handle  # 2 / sqrt(pi)
    coeff_low_numerator = [-0.140543331, 0.914624893, -1.645349621, 0.886226899]
    coeff_low_denominator = [0.012229801, -0.329097515, 1.442710462, -2.118377725, 1.0]
    coeff_high_numerator = [1.641345311, 3.429567803, -1.624906493, -1.970840454]
    coeff_high_denominator = [1.6370678, 3.5438892, 1.0]

    # low cal
    arg0_squared = _builder.create_fmul(arg0.handle, arg0.handle)
    numerator_low_range = semantic.full(
        arg0.shape, coeff_low_numerator[0], arg0_scalar_ty, _builder).handle
    for i in range(1, len(coeff_low_numerator)):
        numerator_low_range = _builder.create_fma(numerator_low_range, arg0_squared,
            semantic.full(arg0.shape, coeff_low_numerator[i], arg0_scalar_ty, _builder).handle)

    denominator_low_range = semantic.full(
        arg0.shape, coeff_low_denominator[0], arg0_scalar_ty, _builder).handle
    for i in range(1, len(coeff_low_denominator)):
        denominator_low_range = _builder.create_fma(
            denominator_low_range, arg0_squared, semantic.full(
                arg0.shape, coeff_low_denominator[i], arg0_scalar_ty, _builder).handle)

    low_res = _builder.create_fmul(arg0.handle, _builder.create_fdiv(numerator_low_range, denominator_low_range))

    # high cal
    arg0_erf_trans = _builder.create_sqrt(  # (log2-log(1-|arg0|))^1/2
        _builder.create_fmul(
            semantic.full(arg0.shape, -1, arg0_scalar_ty, _builder).handle,
            _builder.create_log(
                _builder.create_fdiv(
                    _builder.create_fsub(
                        semantic.full(arg0.shape, 1, arg0_scalar_ty, _builder).handle,
                        _builder.create_fabs(arg0.handle)
                    ),
                    semantic.full(arg0.shape, 2, arg0_scalar_ty, _builder).handle
                )
            )
        )
    )
    numerator_high_range = semantic.full(arg0.shape, coeff_high_numerator[0], arg0_scalar_ty, _builder).handle
    for i in range(1, len(coeff_high_numerator)):
        numerator_high_range = _builder.create_fma(
            numerator_high_range, arg0_erf_trans, semantic.full(
                arg0.shape, coeff_high_numerator[i], arg0_scalar_ty, _builder).handle)

    denominator_high_range = semantic.full(arg0.shape, coeff_high_denominator[0], arg0_scalar_ty, _builder).handle
    for i in range(1, len(coeff_high_denominator)):
        denominator_high_range = _builder.create_fma(
            denominator_high_range, arg0_erf_trans, semantic.full(
                arg0.shape, coeff_high_denominator[i], arg0_scalar_ty, _builder).handle)

    high_res = _builder.create_fdiv(numerator_high_range, denominator_high_range)
    high_res = semantic.mul(
        semantic.where(
            signbit(arg0, _builder=_builder),
            semantic.full(arg0.shape, -1, arg0_scalar_ty, _builder),
            semantic.full(arg0.shape, 1, arg0_scalar_ty, _builder),
            _builder),
        core.tensor(high_res, arg0.type), True, _builder
    ).handle

    for i in range(2):
        low_res = _builder.create_fsub(
            low_res, _builder.create_fdiv(
                _builder.create_fsub(
                    _builder.create_erf(low_res), arg0.handle
                ),
                _builder.create_fmul(
                    inv_sqrt_pi_times_2, _builder.create_exp(
                        _builder.create_fmul(
                            semantic.full(arg0.shape, -1, arg0_scalar_ty, _builder).handle,
                            _builder.create_fmul(low_res, low_res)
                        )
                    )
                )
            )
        )

        high_res = _builder.create_fsub(
            high_res, _builder.create_fdiv(
                _builder.create_fsub(
                    _builder.create_erf(high_res), arg0.handle
                ),
                _builder.create_fmul(
                    inv_sqrt_pi_times_2, _builder.create_exp(
                        _builder.create_fmul(
                            semantic.full(arg0.shape, -1, arg0_scalar_ty, _builder).handle,
                            _builder.create_fmul(high_res, high_res)
                        )
                    )
                )
            )
        )

    arg0_abs = core.tensor(_builder.create_fabs(arg0.handle), arg0.type)
    # Check if |arg0| > 1
    arg0_over = semantic.greater_than(
        arg0_abs, semantic.full(arg0.shape, 1, arg0_scalar_ty, _builder), _builder)
    nan_tensor = semantic.full(arg0.shape, float("nan"), arg0_scalar_ty, _builder)
    # Check if |arg0| = 1
    arg0_equal1 = semantic.equal(
        arg0_abs, semantic.full(arg0.shape, 1, arg0_scalar_ty, _builder),_builder
    )
    pos_inf_tensor = semantic.full(arg0.shape, float("inf"), arg0_scalar_ty, _builder)
    neg_inf_tensor = semantic.full(arg0.shape, float("-inf"), arg0_scalar_ty, _builder)
    inf_res = semantic.where(
        signbit(arg0, _builder=_builder), neg_inf_tensor, pos_inf_tensor, _builder
    )
    # Check if |arg0| >= 0.7
    arg0_high = semantic.greater_equal(
        arg0_abs, semantic.full(arg0.shape, 0.7, arg0_scalar_ty, _builder), _builder
    )

    return semantic.where(
        arg0_equal1, inf_res, semantic.where(
            arg0_over, nan_tensor, semantic.where(
                arg0_high, core.tensor(high_res, arg0.type), core.tensor(low_res, arg0.type), _builder
            ), _builder
        ), _builder
    )


# Note:
# The gamma function is implemented using the reflection formula for negative inputs:
# gamma(x) = pi / (sin(pi * x) * gamma(1 - x)). For inputs x close to a negative integer
# (e.g., -1, -2, ... ), criterion: x = -1 ± 0.66e-3, x = -2 ± 1.30e-3, x = -3 ± 2.30e-3, ...
# The denominator sin(pi * x) approaches zero, leading to numerical instability and loss
# of precision. Resulting in deviations from the reference results;
# Similar issues occur near other negative integers.
@core.extern
@math._check_dtype(dtypes=["fp32"])
def gamma(arg0, _builder=None):
    arg0_scalar_ty = arg0.type.scalar
    arg0 = semantic.to_tensor(arg0, _builder)
    pi_tensor = semantic.full(arg0.shape, math_pi, arg0_scalar_ty, _builder).handle
    sqrt_2pi_tensor = semantic.full(arg0.shape, 2.506628275, arg0_scalar_ty, _builder).handle  # sqrt(2*pi)
    lanczos_coeff = [
        676.5203681218851,
        -1259.1392167224028,
        771.32342877765313,
        -176.61502916214059,
        12.507343278686905,
        -0.13857109526572012,
        9.9843695780195716e-6,
        1.5056327351493116e-7
    ]
    condition = semantic.less_than(arg0, 0.5, _builder)  # 1 - x = x -> x = 0.5
    reflect_arg0 = semantic.where(
        condition, semantic.sub(1, arg0, True, _builder), arg0, _builder
    )

    x = semantic.full(arg0.shape, 0.99999999999980993, arg0_scalar_ty, _builder)
    for i in range(0, len(lanczos_coeff)):
        x = semantic.add(
            x, semantic.fdiv(
                semantic.full(arg0.shape, lanczos_coeff[i], arg0_scalar_ty, _builder),
                semantic.add(reflect_arg0, i, True, _builder), True, _builder
            ), True, _builder
        )
    t = semantic.add(reflect_arg0, 6.5, True, _builder)

    gamma_res = _builder.create_fmul(
        _builder.create_fmul(
            sqrt_2pi_tensor, pow(
                t, semantic.sub(reflect_arg0, 0.5, True, _builder), _builder=_builder
            ).handle
        ),
        _builder.create_fmul(
            x.handle, _builder.create_exp(
                _builder.create_fmul(
                    t.handle, semantic.full(arg0.shape, -1, arg0_scalar_ty, _builder).handle
                )
            )
        )
    )

    gamma_res_reflect = _builder.create_fdiv(
        _builder.create_fdiv(pi_tensor, gamma_res),
        _builder.create_sin(_builder.create_fmul(pi_tensor, arg0.handle))
    )

    is_neg_int = semantic.logical_and(
        semantic.equal(math.floor(arg0, _builder=_builder), arg0, _builder),
        semantic.less_than(arg0, 0, _builder), _builder
    )
    pos_inf_tensor = semantic.full(arg0.shape, float('inf'), arg0_scalar_ty, _builder)
    neg_inf_tensor = semantic.full(arg0.shape, float('-inf'), arg0_scalar_ty, _builder)
    gamma_res_reflect = semantic.where(
        is_neg_int, pos_inf_tensor, core.tensor(gamma_res_reflect, arg0.type), _builder)

    res = semantic.where(condition, gamma_res_reflect, core.tensor(gamma_res, arg0.type), _builder)
    is_pos_inf_input = semantic.equal(arg0, pos_inf_tensor, _builder)
    is_neg_inf_input = semantic.equal(arg0, neg_inf_tensor, _builder)

    return semantic.where(is_pos_inf_input, pos_inf_tensor, semantic.where(
            is_neg_inf_input, neg_inf_tensor, res, _builder), _builder)


# Note:
# The lgamma function computes the natural logarithm of the absolute value of the gamma function.
# Since it uses gamma(x) internally, it inherits the same numerical instability near negative integers:
# For inputs x close to a negative integer (e.g., -1, -2, ...), criterion: x = -1 ± 5.75e-5,
# x = -2 ± 1.39e-6, ..., the computation involves log(|pi / (sin(pi * x) * gamma(1 - x))|).
# As sin(pi * x) approaches zero near negative integers, this leads to numerical instability and loss
# of precision, resulting in deviations from the reference results.
# Similar issues occur near other negative integers.
@core.extern
@math._check_dtype(dtypes=["fp32"])
def lgamma(arg0, _builder=None):
    arg0_scalar_ty = arg0.type.scalar
    arg0 = semantic.to_tensor(arg0, _builder)

    inf_tensor = semantic.full(arg0.shape, float('inf'), arg0_scalar_ty, _builder)
    is_inf = semantic.equal(
        core.tensor(_builder.create_fabs(arg0.handle), arg0.type), inf_tensor, _builder
    )
    gamma_res = _builder.create_fabs(gamma(arg0, _builder=_builder).handle)
    lgamma_res = _builder.create_log(gamma_res)

    return semantic.where(is_inf, inf_tensor, core.tensor(lgamma_res, arg0.type), _builder)


@core.builtin
@math._check_dtype(dtypes=["fp32",])
@math._add_math_1arg_docstr("trunc")
def trunc(arg0: core.tensor, _builder: ir.builder):
    """
    Truncate the input to the nearest integer toward zero.

    For positive numbers, this is equivalent to floor(x).
    For negative numbers, this is equivalent to ceil(x).

        Special cases:
        - trunc(±0) returns ±0.
        - trunc(±inf) returns ±inf.
        - trunc(NaN) returns NaN.
    """
    arg0 = semantic.to_tensor(arg0, _builder)

    zero = semantic.full(arg0.shape, 0.0, arg0.type.scalar, _builder)
    condition = semantic.greater_equal(arg0, zero, _builder)

    floor_result = core.tensor(_builder.create_floor(arg0.handle), arg0.type)
    ceil_result = core.tensor(_builder.create_ceil(arg0.handle), arg0.type)

    return semantic.where(condition, floor_result, ceil_result, _builder)


@core.builtin
@math._check_dtype(dtypes=["fp32",])
@math._add_math_1arg_docstr("nearbyint")
def nearbyint(arg0: core.tensor, _builder: ir.builder):
    """
    Round argument x to an integer value in floating-point format.

    Uses the current rounding mode (round-to-nearest-even, aka banker's rounding).
    """
    arg0 = semantic.to_tensor(arg0, _builder)

    half = semantic.full(arg0.shape, 0.5, arg0.type.scalar, _builder)

    positive_adjust = semantic.add(arg0, half, True, _builder)
    negative_adjust = semantic.sub(arg0, half, True, _builder)

    positive_result = core.tensor(_builder.create_floor(positive_adjust.handle), arg0.type)
    negative_result = core.tensor(_builder.create_ceil(negative_adjust.handle), arg0.type)

    zero = semantic.full(arg0.shape, 0.0, arg0.type.scalar, _builder)
    is_positive = semantic.greater_equal(arg0, zero, _builder)
    basic_round = semantic.where(is_positive, positive_result, negative_result, _builder)

    # Banker's rounding special treatment: For values exactly in the middle, round to the nearest even number.
    fractional = semantic.sub(arg0, basic_round, True, _builder)
    abs_fractional = core.tensor(_builder.create_fabs(fractional.handle), fractional.type)

    is_half = semantic.equal(abs_fractional, half, _builder)

    two = semantic.full(arg0.shape, 2.0, arg0.type.scalar, _builder)

    half_value = math.fdiv(basic_round, two, _builder=_builder)
    half_floor = core.tensor(_builder.create_floor(half_value.handle), half_value.type)
    double_half = semantic.mul(half_floor, two, True, _builder)

    is_even = semantic.equal(basic_round, double_half, _builder)

    adjustment = semantic.where(is_positive,
                               semantic.full(arg0.shape, -1.0, arg0.type.scalar, _builder),
                               semantic.full(arg0.shape, 1.0, arg0.type.scalar, _builder),
                               _builder)

    banker_result = semantic.where(is_even, basic_round,
                                  semantic.add(basic_round, adjustment, True, _builder),
                                  _builder)

    # Final result: Use banker's rounding for cases exactly at 0.5, otherwise use basic rounding.
    return semantic.where(is_half, banker_result, basic_round, _builder)


@core.builtin
@math._check_dtype(dtypes=["fp32",])
@math._add_math_1arg_docstr("arcsine")
def asin(arg0: core.tensor, _builder: ir.builder):
    """
    Calculate the principal value of the arc sine of the input argument x.

    Returns result in radians, in the interval [-π/2, +π/2] for x inside [-1, +1].
    Returns NaN for x outside [-1, +1].
    """
    arg0 = semantic.to_tensor(arg0, _builder)

    # asin(x) = π/2 - acos(x)
    half_pi = semantic.full(arg0.shape, 1.5707963267948966, arg0.type.scalar, _builder)  # π/2
    acos_val = acos(arg0, _builder=_builder)
    return semantic.sub(half_pi, acos_val, True, _builder)


@core.builtin
@math._check_dtype(dtypes=["fp32",])
@math._add_math_1arg_docstr("base-10 logarithm")
def log10(arg0: core.tensor, _builder: ir.builder):
    """
    Calculate the base 10 logarithm of the input argument x.

    Returns NaN for x < 0, -inf for x = 0, and +0 for x = 1.
    log10(x) = log(x) / log(10)
    """
    arg0 = semantic.to_tensor(arg0, _builder)

    log_val = math.log(arg0, _builder=_builder)
    log10_const = semantic.full(arg0.shape, 2.302585092994046, arg0.type.scalar, _builder)

    return math.fdiv(log_val, log10_const, _builder=_builder)

@core.builtin
@math._check_dtype(dtypes=["fp32",])
@math._add_math_2arg_docstr("copysign")
def copysign(arg0: core.tensor, arg1: core.tensor, _builder: ir.builder):
    """
    Create a floating-point value with the magnitude of x and the sign of y.
    """
    x = semantic.to_tensor(arg0, _builder)
    y = semantic.to_tensor(arg1, _builder)

    magnitude = core.tensor(_builder.create_fabs(x.handle), x.type)

    zero = semantic.full(y.shape, 0.0, y.type.scalar, _builder)
    one = semantic.full(y.shape, 1.0, y.type.scalar, _builder)

    is_zero = semantic.equal(y, zero, _builder)
    reciprocal = math.fdiv(one, y, _builder=_builder)
    is_negative_reciprocal = semantic.less_than(reciprocal, zero, _builder)
    is_negative_zero = semantic.and_(is_zero, is_negative_reciprocal, _builder)

    is_negative_nonzero = semantic.less_than(y, zero, _builder)
    is_negative = semantic.or_(is_negative_zero, is_negative_nonzero, _builder)

    neg_magnitude = semantic.mul(magnitude, semantic.full(magnitude.shape, -1.0, magnitude.type.scalar, _builder), True, _builder)

    return semantic.where(is_negative, neg_magnitude, magnitude, _builder)

@core.builtin
def index_put(
    ptr: tensor,
    index: tensor,
    value: tensor,
    dim: int,
    index_boundary: int,
    end_offset: tuple,
    start_offset: tuple,
    dst_stride: tuple,
    _builder=None
):
    """
    Index put values from a tensor into a destination tensor.

    Index put operation for different tensor ranks:
    1. 2D index scatter (0 <= dim < 1):
        1.1 dim = 0
        out[index[i]][start_offset[1]:end_offset[1]] = value[i][0:end_offset[1]-start_offset[1]]
    2. 3D index scatter (0 <= dim < 2):
        2.1 dim = 0
            out[index[i]][start_offset[1]:end_offset[1]][start_offset[2]:end_offset[2]]
                = value[i][0:end_offset[1]-start_offset[1]][0:end_offset[2]-start_offset[2]]
        2.2 dim = 1
            out[start_offset[0]:end_offset[0]][index[j]][start_offset[2]:end_offset[2]]
                = value[0:end_offset[0]-start_offset[0]][j][0:end_offset[2]-start_offset[2]]


    :param ptr: pointer type, the destination tensor pointer (in GM)
    :param index: tensor, a index to scatter (in UB)
    :param value: tensor, a value to store (in UB)
    :param dim: int32, the dimension to scatter along
    :param index_boundary: int64, the upper boundary for index values
    :param end_offset: tuple of int, the offsets of each dimension for the end of the scatter region
    :param start_offset: tuple of int, the offsets of each dimension for the start of the scatter region
    :param dst_stride: tuple of int, the stride of each dimension of destination tensor

    Constraints
    ***********
    - `ptr` and `value` must have the same rank.
    - `ptr.dtype` only supports `float16`, `bfloat16`, `float32` currently.
    - `index` must be an integer tensor. If `index.rank` != 1, it will be reshaped to 1D.
    - `index.numel` must equal `value.shape[dim]`.
    - `value` support 2~5D tensors.
    - `dim` must be valid (0 <= dim < rank(value) - 1).

    Example
    *******
    .. code-block:: python

        import torch
        import triton
        import triton.language as tl
        from triton.language.extra.ascend.libdevice import index_put

        @triton.jit
        def simple_index_put_kernel(value_ptr, index_ptr, dst_ptr):
            # index tile shape: [2]
            index_local = tl.arange(0, 2)
            x1_local = tl.arange(0, 2)[None, :]  # shape=(1,2)

            index_tile = tl.load(index_ptr + index_local)
            value_tile = tl.load(value_ptr + index_local[:, None]*2 + x1_local)

            index_put(
                ptr=dst_ptr,
                index=index_tile,
                value=value_tile,
                dim=0,
                index_boundary=4,
                end_offset=(2, 2),
                start_offset=(0, 0),
                dst_stride=(2, 1)
            )

        dst = torch.zeros((4,2), device='npu', dtype=torch.float32)
        value = torch.tensor([[1.,2.], [3.,4.]], device='npu')
        index = torch.tensor([2, 0], device='npu')

        simple_index_put_kernel[(1,)](value, index, dst)
        print("IndexPut result:", dst) # ref:[[3.,4.], [0.,0.], [1.,2.], [0.,0.]]
    """
    dim = _constexpr_to_value(dim)
    index_boundary = _constexpr_to_value(index_boundary)
    return semantic.index_put(ptr, index, value, dim, index_boundary,
                              end_offset, start_offset, dst_stride, _builder)

@core.builtin
def gather_out_to_ub(
    src: tensor,
    index: tensor,
    index_boundary: int,
    dim: int,
    src_stride: tuple,
    end_offset: tuple,
    start_offset: tuple,
    other=None,
    _builder=None
):
    """
    Gather from a source tensor in Global Memory (GM) to Unified Buffer (UB)
    along a specified dimension with out-of-bound handling.

    Gather operation for different tensor ranks:
    1. 1D index gather:
        out[i] = src[start_offset[0] + index[i]]
    2. 2D index gather (0 <= dim < 2):
        2.1 dim = 0
            out[i][j] = src[start_offset[0] + index[i][j]][start_offset[1] + j]
        2.2 dim = 1
            out[i][j] = src[start_offset[0] + i][start_offset[1] + index[i][j]]
    3. 3D index gather (0 <= dim < 3):
        3.1 dim = 0
            out[i][j][k] = src[start_offset[0] + index[i][j][k]][start_offset[1] + j][start_offset[2] + k]
        3.2 dim = 1
            out[i][j][k] = src[start_offset[0] + i][start_offset[1] + index[i][j][k]][start_offset[2] + k]
        3.3 dim = 2
            out[i][j][k] = src[start_offset[0] + i][start_offset[1] + j][start_offset[2] + index[i][j][k]]

    :param src: pointer type, the source tensor pointer (in GM)
    :param index: tensor, a tensor to gather (in UB)
    :param index_boundary: int64, the upper boundary for index values
    :param dim: int32, the dimension to gather along
    :param src_stride: tuple of int64, the stride of each dimension of src tensor
    :param end_offset: tuple of int32, the end offsets of each dimension for index tensor
    :param start_offset: tuple of int32, the start offsets of each dimension for index tensor
    :param other(Optional): scalar value, the default value when index is out of boundary (in UB)
    :return: tensor, with the same shape as `index.shape` (in UB)

    Constraints
    ***********
    - `src` and `index` must have the same rank.
    - `src.dtype` only supports `float16`, `bfloat16`, `float32` currently.
    - `index` must be an integer tensor, with rank between 1 and 5.
    - `dim` must be valid (0 <= dim < rank(index)).
    - `other` must be a scalar value.
    - For every dimension `i` not equal to `dim`, `index.size[i]` <= `src.size[i]`.
    - The output shape is the same as `index.shape`. If `index` is None, \
        the output tensor will be an empty tensor with the same shape as `index`.

    Example
    *******
    .. code-block:: python

        import torch
        import triton
        import triton.language as tl
        from triton.language.extra.ascend.libdevice import gather_out_to_ub

        @triton.jit
        def simple_gather_kernel(src_ptr, index_ptr, out_ptr):
            # index tile shape: [2,2]
            y0_local = tl.arange(0, 2)[:, None]  # [0,1] rows
            x1_local = tl.arange(0, 2)[None, :]  # [0,1] cols
            mask = (y0_local < 2) & (x1_local < 2)

            # Load index tile to UB
            index = tl.load(index_ptr + y0_local*2 + x1_local, mask)

            # Call gather_out_to_ub: gather values from src along dim=0
            gathered = gather_out_to_ub(
                src=src_ptr,
                index=index,
                index_boundary=4,
                dim=0,
                src_stride=(2, 1),
                end_offset=(2, 2),
                start_offset=(0, 0)
            )

            tl.store(out_ptr + y0_local*2 + x1_local, gathered, mask)

        src = torch.tensor([[1.,2.], [3.,4.], [5.,6.], [7.,8.]], device='npu')
        index = torch.tensor([[0,1], [2,3]], device='npu')
        out = torch.empty((2,2), device='npu', dtype=torch.float32)

        simple_gather_kernel[(1,)](src, index, out)
        print("Gather result:", out)  # ref: [[1.,4.], [5.,8.]]
    """
    dim = _constexpr_to_value(dim)
    index_boundary = _constexpr_to_value(index_boundary)
    return semantic.gather_out_to_ub(
        src, index, index_boundary, dim,
        src_stride, end_offset, start_offset, other, _builder
    )

@core.builtin
def scatter_ub_to_out(
    ptr: tensor,
    value: tensor,
    index: tensor,
    index_boundary: int,
    dim: int,
    dst_stride: tuple,
    end_offset: tuple,
    start_offset: tuple,
    _builder=None
):
    """
    Scatter a tile from Unified Buffer (UB) into a destination tensor in Global Memory (GM)
    along a specified dimension, with index-boundary checking.

    Scatter operation for different tensor ranks:
    1. 1D index scatter:
        out[start_offset[0] + index[i]] = value[i]
    2. 2D index scatter (0 <= dim < 2):
        2.1 dim = 0
            out[start_offset[0] + index[i][j]][start_offset[1] + j] = value[i][j]
        2.2 dim = 1
            out[start_offset[0] + i][start_offset[1] + index[i][j]] = value[i][j]
    3. 3D index scatter (0 <= dim < 3):
        3.1 dim = 0
            out[start_offset[0] + index[i][j][k]][start_offset[1] + j][start_offset[2] + k] = value[i][j][k]
        3.2 dim = 1
            out[start_offset[0] + i][start_offset[1] + index[i][j][k]][start_offset[2] + k] = value[i][j][k]
        3.3 dim = 2
            out[start_offset[0] + i][start_offset[1] + j][start_offset[2] + index[i][j][k]] = value[i][j][k]

    :param ptr: pointer type, the destination tensor pointer (in GM)
    :param value: tensor, a tile value to store (in UB)
    :param index: tensor, a index to scatter (in UB)
    :param index_boundary: int64, the upper boundary for index values
    :param dim: int32, the dimension to scatter along
    :param dst_stride: tuple of int64, the stride of each dimension of destination tensor
    :param end_offset: tuple of int32, the end offsets of each dimension for index tensor
    :param start_offset: tuple of int32, the start offsets of each dimension for index tensor

    Constraints
    ***********
    - `ptr`, `index` and `value` must have the same rank.
    - `ptr.dtype` only supports `float16`, `bfloat16`, `float32` currently.
    - `index` must be an integer tensor, with rank between 1 and 5.
    - `dim` must be valid (0 <= dim < rank(index)).
    - For every dimension `i` not equal to `dim`, `index.size[i]` <= `ptr.size[i]`.
    - The output shape is the same as `index.shape`. If `index` is None, \
        the output tensor will be an empty tensor with the same shape as `index`.

    Example
    *******
    .. code-block:: python

        import torch
        import triton
        import triton.language as tl
        from triton.language.extra.ascend.libdevice import scatter_ub_to_out

        @triton.jit
        def simple_scatter_kernel(value_ptr, index_ptr, dst_ptr):
            # index tile shape: [2,2]
            y0_local = tl.arange(0, 2)[:, None]  # [0,1] rows
            x1_local = tl.arange(0, 2)[None, :]  # [0,1] cols
            mask = (y0_local < 2) & (x1_local < 2)

            value = tl.load(value_ptr + y0_local*2 + x1_local, mask)
            index = tl.load(index_ptr + y0_local*2 + x1_local, mask)

            scatter_ub_to_out(
                ptr=dst_ptr,
                value=value,
                index=index,
                index_boundary=4,
                dim=0,
                dst_stride=(2, 1),
                end_offset=(2, 2),
                start_offset=(0, 0)
            )

        dst = torch.zeros((4,2), device='npu', dtype=torch.float32)
        value = torch.tensor([[1.,2.], [3.,4.]], device='npu')
        index = torch.tensor([[1,2], [3,0]], device='npu')

        simple_scatter_kernel[(1,)](value, index, dst)
        print("Scatter result:", dst)  # ref:[[0.,4.], [1.,0.], [0.,2.], [3.,0.]]
    """
    dim = _constexpr_to_value(dim)
    index_boundary = _constexpr_to_value(index_boundary)
    return semantic.scatter_ub_to_out(
        ptr, value, index, index_boundary, dim,
        dst_stride, end_offset, start_offset, _builder
    )

@core.builtin
def index_select_simd(
    src,
    dim,
    index,
    src_shape,
    src_offset,
    read_shape,
    _builder=None
) -> tensor:
    """
    Parallel index_select operation from Global Memory to Unified Buffer (SIMD version).

    Selects data from multiple indices along a specified dimension and loads
    them as tiles from GM directly to UB with zero-copy semantics.

    :param src: Source tensor pointer (in GM)
    :type src: tensor (pointer type)
    :param dim: The dimension along which to select indices
    :type dim: int or constexpr
    :param index: 1D tensor of indices to select (in UB)
    :type index: tensor
    :param src_shape: Complete shape of the source tensor (can be int or tensor)
    :type src_shape: List[Union[int, tensor]]
    :param src_offset: Starting offset for reading (can be int or tensor)
    :type src_offset: List[Union[int, tensor]]
    :param read_shape: Size to read (tile shape, can be int or tensor)
    :type read_shape: List[Union[int, tensor]]

    **Constraints:**

    - ``read_shape[dim]`` must be ``-1``
    - ``src_offset[dim]`` can be ``-1`` (will be ignored)
    - Boundary handling: ``src_offset + read_shape > src_shape`` automatically
      truncates to ``src_shape`` boundary
    - Does not check if ``index`` contains out-of-bounds values

    **Example:**

    .. code-block:: python

        @triton.jit
        def kernel(src_ptr, output_ptr, indices_ptr, M, N, D, ...):
            # Load indices (e.g., [5, 10, 15, 20])
            indices = tl.load(indices_ptr + tl.arange(0, 4))

            # Example 1: Static shapes (constants)
            # Index select from dimension 1
            # src: [8, 100, 256], index_select at dim=1
            # Read: [4, ?, 128] starting from [4, ?, 128]
            result = libdevice.index_select_simd(
                src_ptr,
                dim=1,
                index=indices,
                src_shape=[8, 100, 256],
                src_offset=[4, -1, 128],
                read_shape=[4, -1, 128]
            )
            # result shape: [4, 4, 128]

            # Example 2: Dynamic shapes (variables)
            result2 = libdevice.index_select_simd(
                src_ptr,
                dim=1,
                index=indices,
                src_shape=[M, N, D],
                src_offset=[4, -1, 128],
                read_shape=[4, -1, 128]
            )

            tl.store(output_ptr + ..., result)

    :return: Result tensor in UB with shape where ``dim`` is replaced
        by the length of ``index``
    :rtype: tensor
    """
    dim = _constexpr_to_value(dim)

    # Process shape parameters: convert constexpr to values, keep tensors as-is
    def process_param(val):
        """Convert constexpr to value, keep tensor or int as-is"""
        if isinstance(val, tensor):
            return val
        else:
            return _constexpr_to_value(val)

    newsrc_shape = [
        semantic.to_tensor(o, _builder) if isinstance(o, constexpr) else o
        for o in src_shape
    ]
    newsrc_offset = [
        semantic.to_tensor(o, _builder) if isinstance(o, constexpr) else o
        for o in src_offset
    ]
    assert len(index.shape) == 1, "index must be a 1D tensor"

    return semantic.index_select_simd(
        src, dim, index, newsrc_shape, newsrc_offset, read_shape, _builder
    )
