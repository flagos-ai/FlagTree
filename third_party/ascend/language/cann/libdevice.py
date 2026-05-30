# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

from math import pi as math_pi
from triton.language import core, math, semantic
from triton._C.libtriton import ir
from triton.runtime.jit import jit
from triton.backends.ascend.utils import get_ascend_arch_from_env, triton_enable_libdevice_simt
from triton.tools.get_ascend_devices import is_compile_on_910_95
import triton


@core.extern
def reciprocal(arg0, _semantic=None):
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp32"), ): ("__hmf_recipf", core.dtype("fp32")),
            (core.dtype("fp16"), ): ("__hmf_recipDh", core.dtype("fp16")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def log1p(arg0, _semantic=None):
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp32"), ): ("__hmf_log1pf", core.dtype("fp32")),
            (core.dtype("fp16"), ): ("__hmf_log1pDh", core.dtype("fp16")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def relu(arg0, _semantic=None):
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp32"), ): ("__hmf_reluf", core.dtype("fp32")),
            (core.dtype("fp16"), ): ("__hmf_reluDh", core.dtype("fp16")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def isinf(arg0, _semantic=None):
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp32"), ): ("__hmf_isinf", core.dtype("int1")),
            (core.dtype("fp16"), ): ("__hmf_isinf", core.dtype("int1")),
            (core.dtype("bf16"), ): ("__hmf_isinf", core.dtype("int1")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def tan(arg0, _semantic=None):
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp32"), ): ("__hmf_tanf", core.dtype("fp32")),
            (core.dtype("fp16"), ): ("__hmf_tanDh", core.dtype("fp16")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def atan(arg0, _semantic=None):
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp32"), ): ("__hmf_atanf", core.dtype("fp32")),
            (core.dtype("fp16"), ): ("__hmf_atanDh", core.dtype("fp16")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def tanh(arg0, _semantic=None):
    arg0 = _semantic.to_tensor(arg0)
    original_dtype = arg0.dtype
    if original_dtype == core.dtype("bf16"):
        arg0 = _semantic.cast(arg0, core.float32)

    if triton_enable_libdevice_simt() and is_compile_on_910_95:
        dispatch = {
            (core.dtype("fp32"), ): ("__hmf_tanh_fp32", core.dtype("fp32")),
        }
    else:
        dispatch = {
            (core.dtype("fp32"), ): ("__hmf_tanhf", core.dtype("fp32")),
            (core.dtype("fp16"), ): ("__hmf_tanhDh", core.dtype("fp16")),
        }

    res = core.extern_elementwise("", "", [arg0], dispatch, is_pure=True, _semantic=_semantic)
    if original_dtype == core.dtype("bf16"):
        return _semantic.cast(res, core.dtype("bf16"))
    return res


@core.extern
def ilogb(arg0, _semantic=None):
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp32"), ): ("__hmf_ilogbf", core.dtype("fp32")),
            (core.dtype("fp16"), ): ("__hmf_ilogbDh", core.dtype("fp16")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def ldexp(arg0, arg1, _semantic=None):
    return core.extern_elementwise(
        "", "", [arg0, arg1], {
            (core.dtype("fp32"), core.dtype("int32")): ("__hmf_ldexpf", core.dtype("fp32")),
            (core.dtype("fp16"), core.dtype("int32")): ("__hmf_ldexpDh", core.dtype("fp16")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def pow(arg0, arg1, _semantic=None):
    if triton_enable_libdevice_simt() and is_compile_on_910_95:
        return core.extern_elementwise("", "", [arg0, arg1], {
            (core.dtype("fp32"), core.dtype("fp32")): ("__hmf_pow_fp32", core.dtype("fp32")),
        }, is_pure=True, _semantic=_semantic)
    else:
        return core.extern_elementwise(
            "", "", [arg0, arg1], {
                (core.dtype("fp32"), core.dtype("fp32")): ("__hmf_powf", core.dtype("fp32")),
                (core.dtype("fp16"), core.dtype("fp16")): ("__hmf_powDh", core.dtype("fp16")),
                (core.dtype("bf16"), core.dtype("bf16")): ("__hmf_powDb", core.dtype("bf16")),
            }, is_pure=True, _semantic=_semantic)


@core.extern
def isnan(arg0, _semantic=None):
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp32"), ): ("__hmf_isnan", core.dtype("int1")),
            (core.dtype("fp16"), ): ("__hmf_isnan", core.dtype("int1")),
            (core.dtype("bf16"), ): ("__hmf_isnan", core.dtype("int1")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def div_rz(arg0, arg1, _semantic=None):
    return core.extern_elementwise("", "", [arg0, arg1], {
        (core.dtype("fp32"), core.dtype("fp32")): ("__hmf_div_rz_fp32", core.dtype("fp32")),
    }, is_pure=True, _semantic=_semantic)


@core.builtin
def fast_dividef(arg0, arg1, _semantic=None):
    arg0 = _semantic.to_tensor(arg0)
    arg1 = _semantic.to_tensor(arg1)
    ret = _semantic.fdiv(arg0, arg1, False)
    return ret


@core.builtin
def fast_expf(arg0, _semantic=None):
    arg0 = _semantic.to_tensor(arg0)
    ret = core.tensor(_semantic.builder.create_exp(arg0.handle), arg0.type)
    return ret


@core.extern
def fmod(arg0, arg1, _semantic=None):
    return core.extern_elementwise("", "", [arg0, arg1], {
        (core.dtype("fp32"), core.dtype("fp32")): ("__hmf_fmod_fp32", core.dtype("fp32")),
    }, is_pure=True, _semantic=_semantic)


@core.extern
def float_as_int(arg0, _semantic=None):
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("fp32"), ): ("__hmf_float_as_int_fp32", core.dtype("int32")),
    }, is_pure=True, _semantic=_semantic)


@core.extern
def atan2(arg0, arg1, _semantic=None):
    if arg0.dtype == core.dtype("bf16") or arg1.dtype == core.dtype("bf16"):
        core.static_print("extern livdevice.atan2 for dtype bf16 is unspported for now.")
        core.static_assert(False)
    return core.extern_elementwise(
        "", "", [arg0, arg1], {
            (core.dtype("fp16"), core.dtype("fp16")): ("__hmf_atan2_fp16", core.dtype("fp16")),
            (core.dtype("fp32"), core.dtype("fp32")): ("__hmf_atan2_fp32", core.dtype("fp32")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def round(arg0, _semantic=None):
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("fp32"), ): ("__hmf_roundf", core.dtype("fp32")),
    }, is_pure=True, _semantic=_semantic)


@core.builtin
@math._check_dtype(dtypes=["bf16", "fp16", "fp32"])
@math._add_math_1arg_docstr("acos")
def acos(arg0: core.tensor, _semantic=None):
    if triton_enable_libdevice_simt() and is_compile_on_910_95:
        if arg0.dtype == core.dtype("bf16"):
            core.static_print("extern livdevice.acos for dtype bf16 is unspported for now.")
            core.static_assert(False)
        return core.extern_elementwise(
            "", "", [arg0], {
                (core.dtype("fp16"), ): ("__hmf_acos_fp16", core.dtype("fp16")),
                (core.dtype("fp32"), ): ("__hmf_acos_fp32", core.dtype("fp32")),
            }, is_pure=True, _semantic=_semantic)
    else:
        pi = 3.1415926536
        pi_half = 1.5707963268
        sqrt2 = 1.4142135624
        eps = 1e-8

        # |x| < 0.5, acos(x) = pi/2 - [x + x*x²*(0.1666667 + x²*(0.075 + x²*(0.0446429 + 0.0303810*x²))]
        arg0 = _semantic.to_tensor(arg0)
        abs_x = math.abs(arg0, _semantic=_semantic)
        dtype = arg0.dtype
        arg0_2 = _semantic.mul(arg0, arg0, True)
        arg0_4 = _semantic.mul(arg0_2, arg0_2, True)
        arg0_6 = _semantic.mul(arg0_4, arg0_2, True)
        arg0_8 = _semantic.mul(arg0_6, arg0_2, True)
        arg0_10 = _semantic.mul(arg0_8, arg0_2, True)
        poly = _semantic.add(1.0, _semantic.mul(0.166667, arg0_2, True), True)
        poly = _semantic.add(poly, _semantic.mul(0.075, arg0_4, True), True)
        poly = _semantic.add(poly, _semantic.mul(0.044643, arg0_6, True), True)
        poly = _semantic.add(poly, _semantic.mul(0.030380, arg0_8, True), True)
        poly = _semantic.add(poly, _semantic.mul(0.022372, arg0_10, True), True)
        acos_center = _semantic.sub(pi_half, _semantic.mul(arg0, poly, True), True)

        # 0.5<|x|<0.9, acos(x) = 2*arctan(t), t=sqrt((1-abs_x)/(1+abs_x))
        numerator_mid = _semantic.sub(1.0, abs_x, True)
        denom_mid = _semantic.add(1.0, abs_x, True)
        div_mid = _semantic.truediv(numerator_mid, denom_mid)
        t_mid = math.sqrt(div_mid, _semantic=_semantic)
        t2_mid = _semantic.mul(t_mid, t_mid, True)
        t4_mid = _semantic.mul(t2_mid, t2_mid, True)
        t6_mid = _semantic.mul(t4_mid, t2_mid, True)

        poly_mid1 = _semantic.mul(0.1065976, t2_mid, True)
        poly_mid2 = _semantic.add(-0.1420890, poly_mid1, True)
        poly_mid3 = _semantic.mul(poly_mid2, t2_mid, True)
        poly_mid4 = _semantic.add(0.1999341, poly_mid3, True)
        poly_mid5 = _semantic.mul(poly_mid4, t2_mid, True)
        poly_mid6 = _semantic.add(-0.3333310, poly_mid5, True)
        poly_mid = _semantic.add(1.0, _semantic.mul(poly_mid6, t2_mid, True), True)
        arctan_t = _semantic.mul(t_mid, poly_mid, True)
        acos_mid = _semantic.mul(2.0, arctan_t, True)
        is_neg_mid = _semantic.less_than(arg0, 0.0)
        acos_mid_signed = _semantic.where(is_neg_mid, _semantic.sub(pi, acos_mid, True), acos_mid)

        is_center = _semantic.less_than(abs_x, 0.6)
        res_mid_boundary = _semantic.where(is_center, acos_center, acos_mid_signed)
        return res_mid_boundary


@core.builtin
@math._check_dtype(dtypes=["bf16", "fp16", "fp32"])
@math._add_math_1arg_docstr("sinh")
def sinh(arg0: core.tensor, _semantic=None):
    if triton_enable_libdevice_simt() and is_compile_on_910_95:
        if arg0.dtype == core.dtype("bf16"):
            core.static_print("extern livdevice.sinh for dtype bf16 is unspported for now.")
            core.static_assert(False)
        return core.extern_elementwise(
            "", "", [arg0], {
                (core.dtype("fp16"), ): ("__hmf_sinh_fp16", core.dtype("fp16")),
                (core.dtype("fp32"), ): ("__hmf_sinh_fp32", core.dtype("fp32")),
            }, is_pure=True, _semantic=_semantic)
    else:
        arg0 = _semantic.to_tensor(arg0)
        exp0 = core.tensor(_semantic.builder.create_exp(arg0.handle), arg0.type)
        exp1 = _semantic.truediv(1.0, exp0)
        tmp = _semantic.sub(exp0, exp1, True)
        ret = _semantic.truediv(tmp, 2.0)
        return ret


@core.builtin
@math._check_dtype(dtypes=["bf16", "fp16", "fp32"])
@math._add_math_1arg_docstr("cosh")
def cosh(arg0: core.tensor, _semantic=None):
    if triton_enable_libdevice_simt() and is_compile_on_910_95:
        if arg0.dtype == core.dtype("bf16"):
            core.static_print("extern livdevice.cosh for dtype bf16 is unspported for now.")
            core.static_assert(False)
        return core.extern_elementwise(
            "", "", [arg0], {
                (core.dtype("fp16"), ): ("__hmf_cosh_fp16", core.dtype("fp16")),
                (core.dtype("fp32"), ): ("__hmf_cosh_fp32", core.dtype("fp32")),
            }, is_pure=True, _semantic=_semantic)
    else:
        arg0 = _semantic.to_tensor(arg0)
        exp0 = core.tensor(_semantic.builder.create_exp(arg0.handle), arg0.type)
        exp1 = _semantic.truediv(1.0, exp0)
        tmp = _semantic.add(exp0, exp1, True)
        ret = _semantic.truediv(tmp, 2.0)
        return ret


@core.builtin
@math._check_dtype(dtypes=["bf16", "fp16", "fp32"])
@math._add_math_1arg_docstr("acosh")
def acosh(arg0: core.tensor, _semantic=None):
    if triton_enable_libdevice_simt() and is_compile_on_910_95:
        if arg0.dtype == core.dtype("bf16"):
            core.static_print("extern livdevice.acosh for dtype bf16 is unspported for now.")
            core.static_assert(False)
        return core.extern_elementwise(
            "", "", [arg0], {
                (core.dtype("fp16"), ): ("__hmf_acosh_fp16", core.dtype("fp16")),
                (core.dtype("fp32"), ): ("__hmf_acosh_fp32", core.dtype("fp32")),
            }, is_pure=True, _semantic=_semantic)
    else:
        arg0 = _semantic.to_tensor(arg0)
        tmp = _semantic.sub(_semantic.mul(arg0, arg0, True), 1.0, True)
        sqrt_res = core.tensor(_semantic.builder.create_sqrt(tmp.handle), tmp.type)
        sum_res = _semantic.add(arg0, sqrt_res, True)
        return core.tensor(_semantic.builder.create_log(sum_res.handle), sum_res.type)


@core.builtin
@math._check_dtype(dtypes=["bf16", "fp16", "fp32"])
@math._add_math_1arg_docstr("asinh")
def asinh(arg0: core.tensor, _semantic=None):
    if triton_enable_libdevice_simt() and is_compile_on_910_95:
        if arg0.dtype == core.dtype("bf16"):
            core.static_print("extern livdevice.asinh for dtype bf16 is unspported for now.")
            core.static_assert(False)
        return core.extern_elementwise(
            "", "", [arg0], {
                (core.dtype("fp16"), ): ("__hmf_asinh_fp16", core.dtype("fp16")),
                (core.dtype("fp32"), ): ("__hmf_asinh_fp32", core.dtype("fp32")),
            }, is_pure=True, _semantic=_semantic)
    else:
        arg0 = _semantic.to_tensor(arg0)
        tmp = _semantic.add(_semantic.mul(arg0, arg0, True), 1.0, True)
        sqrt_res = core.tensor(_semantic.builder.create_sqrt(tmp.handle), tmp.type)
        sum_res = _semantic.add(arg0, sqrt_res, True)
        return core.tensor(_semantic.builder.create_log(sum_res.handle), sum_res.type)


@core.builtin
@math._check_dtype(dtypes=["bf16", "fp16", "fp32"])
@math._add_math_1arg_docstr("atanh")
def atanh(arg0: core.tensor, _semantic=None):
    if triton_enable_libdevice_simt() and is_compile_on_910_95:
        if arg0.dtype == core.dtype("bf16"):
            core.static_print("extern livdevice.atanh for dtype bf16 is unspported for now.")
            core.static_assert(False)
        return core.extern_elementwise(
            "", "", [arg0], {
                (core.dtype("fp16"), ): ("__hmf_atanh_fp16", core.dtype("fp16")),
                (core.dtype("fp32"), ): ("__hmf_atanh_fp32", core.dtype("fp32")),
            }, is_pure=True, _semantic=_semantic)
    else:
        arg0 = _semantic.to_tensor(arg0)
        a = _semantic.add(1.0, arg0, True)
        b = _semantic.sub(1.0, arg0, True)
        lna = core.tensor(_semantic.builder.create_log(a.handle), a.type)
        lnb = core.tensor(_semantic.builder.create_log(b.handle), b.type)
        tmp = _semantic.sub(lna, lnb, True)
        return _semantic.mul(tmp, 0.5, True)


@core.builtin
@math._check_dtype(dtypes=["bf16", "fp16", "fp32"])
@math._add_math_1arg_docstr("expm1")
def expm1(arg0: core.tensor, _semantic=None):
    if triton_enable_libdevice_simt() and is_compile_on_910_95:
        if arg0.dtype == core.dtype("bf16"):
            core.static_print("extern livdevice.expm1 for dtype bf16 is unspported for now.")
            core.static_assert(False)
        return core.extern_elementwise(
            "", "", [arg0], {
                (core.dtype("fp16"), ): ("__hmf_expm1_fp16", core.dtype("fp16")),
                (core.dtype("fp32"), ): ("__hmf_expm1_fp32", core.dtype("fp32")),
            }, is_pure=True, _semantic=_semantic)
    else:
        arg0 = _semantic.to_tensor(arg0)
        tmp = core.tensor(_semantic.builder.create_exp(arg0.handle), arg0.type)
        return _semantic.sub(tmp, 1, True)


@core.builtin
@math._check_dtype(dtypes=["fp16", "fp32"])
@math._add_math_2arg_docstr("nextafter")
def nextafter(arg0: core.tensor, arg1: core.tensor, _semantic=None):
    if triton_enable_libdevice_simt() and is_compile_on_910_95:
        return core.extern_elementwise(
            "", "", [arg0, arg1], {
                (core.dtype("fp16"), core.dtype("fp16")): ("__hmf_nextafter_fp16", core.dtype("fp16")),
                (core.dtype("fp32"), core.dtype("fp32")): ("__hmf_nextafter_fp32", core.dtype("fp32")),
            }, is_pure=True, _semantic=_semantic)
    else:
        x = _semantic.to_tensor(arg0)
        y = _semantic.to_tensor(arg1)
        dtype_map = {"bf16": core.int16, "fp16": core.int16, "fp32": core.int32}
        min_pos_bit = {"bf16": 0x0001, "fp16": 0x0001, "fp32": 0x00000001}
        max_neg_bit = {"bf16": 0x8001, "fp16": 0x8001, "fp32": 0x80000001}
        int_type = dtype_map[x.type.scalar.name]
        x_eq_y = _semantic.equal(x, y)
        x_gt_0 = _semantic.greater_than(x, 0)
        y_gt_x = _semantic.greater_than(y, x)
        next_neg = _semantic.xor_(x_gt_0, y_gt_x)
        next_pos = _semantic.not_(next_neg)

        p1 = _semantic.full(x.shape, 1, int_type)
        n1 = _semantic.full(x.shape, -1, int_type)
        dir_xy = _semantic.where(next_pos, p1, n1)
        x_abs = math.abs(x, _semantic=_semantic)
        x_is_0 = _semantic.equal(x_abs, 0)

        min_pos = _semantic.full(x.shape, min_pos_bit[x.type.scalar.name], int_type)
        max_neg = _semantic.full(x.shape, max_neg_bit[x.type.scalar.name], int_type)
        min_pos = _semantic.bitcast(min_pos, x.dtype)
        max_neg = _semantic.bitcast(max_neg, x.dtype)
        bits_x = _semantic.bitcast(x, int_type)
        bits_next = _semantic.add(bits_x, dir_xy, True)
        next_val = _semantic.bitcast(bits_next, x.dtype)

        need_min_pos = _semantic.logical_and(x_is_0, next_pos)
        need_max_neg = _semantic.logical_and(x_is_0, next_neg)
        next_val = _semantic.where(need_min_pos, min_pos, next_val)
        next_val = _semantic.where(need_max_neg, max_neg, next_val)
        return _semantic.where(x_eq_y, x, next_val)


@core.builtin
@math._check_dtype(dtypes=["bf16", "fp16", "fp32"])
@math._add_math_2arg_docstr("hypot(Euclidean Distance)")
def hypot(arg0: core.tensor, arg1: core.tensor, _semantic=None):
    if triton_enable_libdevice_simt() and is_compile_on_910_95:
        if arg0.dtype == core.dtype("bf16"):
            core.static_print("extern livdevice.hypot for dtype bf16 is unspported for now.")
            core.static_assert(False)
        return core.extern_elementwise(
            "", "", [arg0, arg1], {
                (core.dtype("fp16"), core.dtype("fp16")): ("__hmf_hypot_fp16", core.dtype("fp16")),
                (core.dtype("fp32"), core.dtype("fp32")): ("__hmf_hypot_fp32", core.dtype("fp32")),
            }, is_pure=True, _semantic=_semantic)
    else:
        arg0 = _semantic.to_tensor(arg0)
        arg1 = _semantic.to_tensor(arg1)
        x2 = _semantic.mul(arg0, arg0, True)
        y2 = _semantic.mul(arg1, arg1, True)
        sum_res = _semantic.add(x2, y2, True)
        return core.tensor(_semantic.builder.create_sqrt(sum_res.handle), sum_res.type)


# This function is derived from the Cephes Math Library release 2.8: June, 2000
# https://netlib.org/cephes/
# Copyright (c) 1984, 1987, 2000 by Stephen L. Moshier
# All rights reserved.
@core.builtin
@math._check_dtype(dtypes=["fp16", "fp32"])
@math._add_math_2arg_docstr("besseli0 (Modified Bessel function of the first kind, order 0).")
def cyl_bessel_i0(arg0: core.tensor, _semantic=None):
    if triton_enable_libdevice_simt() and is_compile_on_910_95:
        if arg0.dtype == core.dtype("fp16"):
            core.static_print("extern livdevice.cyl_bessel_i0 for dtype bf16 is unspported for now.")
            core.static_assert(False)
        return core.extern_elementwise("", "", [arg0], {
            (core.dtype("fp32"), ): ("__hmf_cyl_bessel_i0_fp32", core.dtype("fp32")),
        }, is_pure=True, _semantic=_semantic)
    else:
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
        arg0 = _semantic.to_tensor(arg0)
        abs_x = core.tensor(_semantic.builder.create_fabs(arg0.handle), arg0.type)
        x_a = _semantic.sub(_semantic.mul(abs_x, 0.5, True), 2.0, True)
        a_n_2 = 0
        a_n_1 = 0
        a_n = param1[0]
        for i in range(1, 30):
            a_n_2 = a_n_1
            a_n_1 = a_n
            a_n = _semantic.sub(_semantic.mul(x_a, a_n_1, True), a_n_2, True)
            a_n = _semantic.add(a_n, param1[i], True)

        f_32 = _semantic.full(abs_x.shape, 32.0, abs_x.type.scalar)
        x_b = _semantic.sub(_semantic.fdiv(f_32, abs_x, True), 2.0, True)
        b_n_2 = 0
        b_n_1 = 0
        b_n = param2[0]
        for i in range(1, 25):
            b_n_2 = b_n_1
            b_n_1 = b_n
            b_n = _semantic.sub(_semantic.mul(x_b, b_n_1, True), b_n_2, True)
            b_n = _semantic.add(b_n, param2[i], True)

        half_exp = _semantic.mul(core.tensor(_semantic.builder.create_exp(abs_x.handle), abs_x.type), 0.5, True)
        res_a = _semantic.mul(half_exp, _semantic.sub(a_n, a_n_2, True), True)
        res_b = _semantic.fdiv(_semantic.mul(half_exp, _semantic.sub(b_n, b_n_2, True), True), \
            core.tensor(_semantic.builder.create_sqrt(abs_x.handle), abs_x.type), True)
        cond = _semantic.less_equal(abs_x, 8.0)
        res = _semantic.where(cond, res_a, res_b)
        return res


@core.extern
@math._check_dtype(dtypes=["fp16", "fp32"])
def signbit(arg0, _semantic=None):
    if triton_enable_libdevice_simt() and is_compile_on_910_95:
        return core.extern_elementwise(
            "", "", [arg0], {
                (core.dtype("fp16"), ): ("__hmf_signbit_fp16", core.dtype("int32")),
                (core.dtype("fp32"), ): ("__hmf_signbit_fp32", core.dtype("int32")),
            }, is_pure=True, _semantic=_semantic)
    else:
        arg0_scalar_ty = arg0.type.scalar
        if arg0_scalar_ty == core.float32:
            int_ty = core.int32
        else:  # arg0 type: float16 / bfloat16
            int_ty = core.int16

        arg0 = _semantic.to_tensor(arg0)
        int_tensor = _semantic.bitcast(arg0, int_ty)
        if int_ty == core.int32:
            shift = 31
        elif int_ty == core.int16:
            shift = 15

        shift = _semantic.full(arg0.shape, shift, int_ty)
        sign_bit_tensor = _semantic.lshr(int_tensor, shift)
        sign_bit_tensor = _semantic.and_(sign_bit_tensor, _semantic.full(arg0.shape, 1, int_ty))
        return _semantic.equal(sign_bit_tensor, 1)


# Note:
# For inputs x very close to ±1 (criterion: 1 - |x| < 1.1e-4), erfinv(x) → ±∞ and the
# inverse error function becomes extremely sensitive to tiny changes in x. The asymptotic
# behavior includes terms like sqrt(-ln(1-|x|)), so tiny relative changes in (1-|x|) map
# to large absolute changes in erfinv, leading to numerical instability and loss of precision,
# resulting in deviations from the reference results.
@core.extern
@math._check_dtype(dtypes=["fp32"])
def erfinv(arg0, _semantic=None):
    if triton_enable_libdevice_simt() and is_compile_on_910_95:
        return core.extern_elementwise("", "", [arg0], {
            (core.dtype("fp32"), ): ("__hmf_erfinv_fp32", core.dtype("fp32")),
        }, is_pure=True, _semantic=_semantic)
    else:
        arg0_scalar_ty = arg0.type.scalar
        arg0 = _semantic.to_tensor(arg0)

        inv_sqrt_pi_times_2 = _semantic.full(arg0.shape, 1.128379167, arg0_scalar_ty).handle  # 2 / sqrt(pi)
        coeff_low_numerator = [-0.140543331, 0.914624893, -1.645349621, 0.886226899]
        coeff_low_denominator = [0.012229801, -0.329097515, 1.442710462, -2.118377725, 1.0]
        coeff_high_numerator = [1.641345311, 3.429567803, -1.624906493, -1.970840454]
        coeff_high_denominator = [1.6370678, 3.5438892, 1.0]

        # low cal
        arg0_squared = _semantic.builder.create_fmul(arg0.handle, arg0.handle)
        numerator_low_range = _semantic.full(arg0.shape, coeff_low_numerator[0], arg0_scalar_ty).handle
        for i in range(1, len(coeff_low_numerator)):
            numerator_low_range = _semantic.builder.create_fma(
                numerator_low_range, arg0_squared,
                _semantic.full(arg0.shape, coeff_low_numerator[i], arg0_scalar_ty).handle)

        denominator_low_range = _semantic.full(arg0.shape, coeff_low_denominator[0], arg0_scalar_ty).handle
        for i in range(1, len(coeff_low_denominator)):
            denominator_low_range = _semantic.builder.create_fma(
                denominator_low_range, arg0_squared,
                _semantic.full(arg0.shape, coeff_low_denominator[i], arg0_scalar_ty).handle)

        low_res = _semantic.builder.create_fmul(
            arg0.handle, _semantic.builder.create_fdiv(numerator_low_range, denominator_low_range))

        # high cal
        arg0_erf_trans = _semantic.builder.create_sqrt(  # (log2-log(1-|arg0|))^1/2
            _semantic.builder.create_fmul(
                _semantic.full(arg0.shape, -1, arg0_scalar_ty).handle,
                _semantic.builder.create_log(
                    _semantic.builder.create_fdiv(
                        _semantic.builder.create_fsub(
                            _semantic.full(arg0.shape, 1, arg0_scalar_ty).handle,
                            _semantic.builder.create_fabs(arg0.handle)),
                        _semantic.full(arg0.shape, 2, arg0_scalar_ty).handle))))
        numerator_high_range = _semantic.full(arg0.shape, coeff_high_numerator[0], arg0_scalar_ty).handle
        for i in range(1, len(coeff_high_numerator)):
            numerator_high_range = _semantic.builder.create_fma(
                numerator_high_range, arg0_erf_trans,
                _semantic.full(arg0.shape, coeff_high_numerator[i], arg0_scalar_ty).handle)

        denominator_high_range = _semantic.full(arg0.shape, coeff_high_denominator[0], arg0_scalar_ty).handle
        for i in range(1, len(coeff_high_denominator)):
            denominator_high_range = _semantic.builder.create_fma(
                denominator_high_range, arg0_erf_trans,
                _semantic.full(arg0.shape, coeff_high_denominator[i], arg0_scalar_ty).handle)

    high_res = _semantic.builder.create_fdiv(numerator_high_range, denominator_high_range)
    high_res = _semantic.mul(
        _semantic.where(signbit(arg0, _semantic=_semantic), _semantic.full(arg0.shape, -1, arg0_scalar_ty),
                        _semantic.full(arg0.shape, 1, arg0_scalar_ty)), core.tensor(high_res, arg0.type), True).handle

    for _ in range(2):
        low_res = _semantic.builder.create_fsub(
            low_res,
            _semantic.builder.create_fdiv(
                _semantic.builder.create_fsub(_semantic.builder.create_erf(low_res), arg0.handle),
                _semantic.builder.create_fmul(
                    inv_sqrt_pi_times_2,
                    _semantic.builder.create_exp(
                        _semantic.builder.create_fmul(
                            _semantic.full(arg0.shape, -1, arg0_scalar_ty).handle,
                            _semantic.builder.create_fmul(low_res, low_res))))))

        high_res = _semantic.builder.create_fsub(
            high_res,
            _semantic.builder.create_fdiv(
                _semantic.builder.create_fsub(_semantic.builder.create_erf(high_res), arg0.handle),
                _semantic.builder.create_fmul(
                    inv_sqrt_pi_times_2,
                    _semantic.builder.create_exp(
                        _semantic.builder.create_fmul(
                            _semantic.full(arg0.shape, -1, arg0_scalar_ty).handle,
                            _semantic.builder.create_fmul(high_res, high_res))))))

    arg0_abs = core.tensor(_semantic.builder.create_fabs(arg0.handle), arg0.type)
    # Check if |arg0| > 1
    arg0_over = _semantic.greater_than(arg0_abs, _semantic.full(arg0.shape, 1, arg0_scalar_ty))
    nan_tensor = _semantic.full(arg0.shape, float("nan"), arg0_scalar_ty)
    # Check if |arg0| = 1
    arg0_equal1 = _semantic.equal(arg0_abs, _semantic.full(arg0.shape, 1, arg0_scalar_ty))
    pos_inf_tensor = _semantic.full(arg0.shape, float("inf"), arg0_scalar_ty)
    neg_inf_tensor = _semantic.full(arg0.shape, float("-inf"), arg0_scalar_ty)
    inf_res = _semantic.where(signbit(arg0, _semantic=_semantic), neg_inf_tensor, pos_inf_tensor)
    # Check if |arg0| >= 0.7
    arg0_high = _semantic.greater_equal(arg0_abs, _semantic.full(arg0.shape, 0.7, arg0_scalar_ty))

    return _semantic.where(
        arg0_equal1, inf_res,
        _semantic.where(arg0_over, nan_tensor,
                        _semantic.where(arg0_high, core.tensor(high_res, arg0.type), core.tensor(low_res, arg0.type))))


# Note:
# The gamma function is implemented using the reflection formula for negative inputs:
# gamma(x) = pi / (sin(pi * x) * gamma(1 - x)). For inputs x close to a negative integer
# (e.g., -1, -2, ... ), criterion: x = -1 ± 0.66e-3, x = -2 ± 1.30e-3, x = -3 ± 2.30e-3, ...
# The denominator sin(pi * x) approaches zero, leading to numerical instability and loss
# of precision. Resulting in deviations from the reference results;
# Similar issues occur near other negative integers.
@core.extern
@math._check_dtype(dtypes=["fp32"])
def gamma(arg0, _semantic=None):
    arg0_scalar_ty = arg0.type.scalar
    arg0 = _semantic.to_tensor(arg0)
    pi_tensor = _semantic.full(arg0.shape, math_pi, arg0_scalar_ty).handle
    sqrt_2pi_tensor = _semantic.full(arg0.shape, 2.506628275, arg0_scalar_ty).handle  # sqrt(2*pi)
    lanczos_coeff = [
        676.5203681218851, -1259.1392167224028, 771.32342877765313, -176.61502916214059, 12.507343278686905,
        -0.13857109526572012, 9.9843695780195716e-6, 1.5056327351493116e-7
    ]
    condition = _semantic.less_than(arg0, 0.5)  # 1 - x = x -> x = 0.5
    reflect_arg0 = _semantic.where(condition, _semantic.sub(1, arg0, True), arg0)

    x = _semantic.full(arg0.shape, 0.99999999999980993, arg0_scalar_ty)
    for i in range(0, len(lanczos_coeff)):
        x = _semantic.add(
            x,
            _semantic.fdiv(_semantic.full(arg0.shape, lanczos_coeff[i], arg0_scalar_ty),
                           _semantic.add(reflect_arg0, i, True), True), True)
    t = _semantic.add(reflect_arg0, 6.5, True)

    gamma_res = _semantic.builder.create_fmul(
        _semantic.builder.create_fmul(sqrt_2pi_tensor,
                                      pow(t, _semantic.sub(reflect_arg0, 0.5, True), _semantic=_semantic).handle),
        _semantic.builder.create_fmul(
            x.handle,
            _semantic.builder.create_exp(
                _semantic.builder.create_fmul(t.handle,
                                              _semantic.full(arg0.shape, -1, arg0_scalar_ty).handle))))

    gamma_res_reflect = _semantic.builder.create_fdiv(
        _semantic.builder.create_fdiv(pi_tensor, gamma_res),
        _semantic.builder.create_sin(_semantic.builder.create_fmul(pi_tensor, arg0.handle)))

    is_neg_int = _semantic.logical_and(_semantic.equal(math.floor(arg0, _semantic=_semantic), arg0),
                                       _semantic.less_than(arg0, 0))
    pos_inf_tensor = _semantic.full(arg0.shape, float('inf'), arg0_scalar_ty)
    neg_inf_tensor = _semantic.full(arg0.shape, float('-inf'), arg0_scalar_ty)
    gamma_res_reflect = _semantic.where(is_neg_int, pos_inf_tensor, core.tensor(gamma_res_reflect, arg0.type))

    res = _semantic.where(condition, gamma_res_reflect, core.tensor(gamma_res, arg0.type))
    is_pos_inf_input = _semantic.equal(arg0, pos_inf_tensor)
    is_neg_inf_input = _semantic.equal(arg0, neg_inf_tensor)

    return _semantic.where(is_pos_inf_input, pos_inf_tensor, _semantic.where(is_neg_inf_input, neg_inf_tensor, res))


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
def lgamma(arg0, _semantic=None):
    if triton_enable_libdevice_simt() and is_compile_on_910_95:
        return core.extern_elementwise("", "", [arg0], {
            (core.dtype("fp32"), ): ("__hmf_lgamma_fp32", core.dtype("fp32")),
        }, is_pure=True, _semantic=_semantic)
    else:
        arg0_scalar_ty = arg0.type.scalar
        arg0 = _semantic.to_tensor(arg0)

        inf_tensor = _semantic.full(arg0.shape, float('inf'), arg0_scalar_ty)
        is_inf = _semantic.equal(core.tensor(_semantic.builder.create_fabs(arg0.handle), arg0.type), inf_tensor)
        gamma_res = _semantic.builder.create_fabs(gamma(arg0, _semantic=_semantic).handle)
        lgamma_res = _semantic.builder.create_log(gamma_res)

        return _semantic.where(is_inf, inf_tensor, core.tensor(lgamma_res, arg0.type))


@core.builtin
@math._check_dtype(dtypes=[
    "fp32",
])
@math._add_math_1arg_docstr("trunc")
def trunc(arg0: core.tensor, _semantic=None):
    if triton_enable_libdevice_simt() and is_compile_on_910_95:
        return core.extern_elementwise(
            "", "", [arg0], {
                (core.dtype("fp16"), ): ("__hmf_trunc_fp16", core.dtype("fp16")),
                (core.dtype("fp32"), ): ("__hmf_trunc_fp32", core.dtype("fp32")),
            }, is_pure=True, _semantic=_semantic)
    else:
        """
        Truncate the input to the nearest integer toward zero.

        For positive numbers, this is equivalent to floor(x).
        For negative numbers, this is equivalent to ceil(x).

            Special cases:
            - trunc(±0) returns ±0.
            - trunc(±inf) returns ±inf.
            - trunc(NaN) returns NaN.
        """
        arg0 = _semantic.to_tensor(arg0)

        zero = _semantic.full(arg0.shape, 0.0, arg0.type.scalar)
        condition = _semantic.greater_equal(arg0, zero)

        floor_result = core.tensor(_semantic.builder.create_floor(arg0.handle), arg0.type)
        ceil_result = core.tensor(_semantic.builder.create_ceil(arg0.handle), arg0.type)

        return _semantic.where(condition, floor_result, ceil_result)


@core.builtin
@math._check_dtype(dtypes=[
    "fp32",
])
@math._add_math_1arg_docstr("nearbyint")
def nearbyint(arg0: core.tensor, _semantic=None):
    if triton_enable_libdevice_simt() and is_compile_on_910_95:
        return core.extern_elementwise("", "", [arg0], {
            (core.dtype("fp32"), ): ("__hmf_nearbyint_fp32", core.dtype("fp32")),
        }, is_pure=True, _semantic=_semantic)
    else:
        """
        Round argument x to an integer value in floating-point format.

        Uses the current rounding mode (round-to-nearest-even, aka banker's rounding).
        """
        arg0 = _semantic.to_tensor(arg0)

        half = _semantic.full(arg0.shape, 0.5, arg0.type.scalar)

        positive_adjust = _semantic.add(arg0, half, True)
        negative_adjust = _semantic.sub(arg0, half, True)

        positive_result = core.tensor(_semantic.builder.create_floor(positive_adjust.handle), arg0.type)
        negative_result = core.tensor(_semantic.builder.create_ceil(negative_adjust.handle), arg0.type)

        zero = _semantic.full(arg0.shape, 0.0, arg0.type.scalar)
        is_positive = _semantic.greater_equal(arg0, zero)
        basic_round = _semantic.where(is_positive, positive_result, negative_result)

        # Banker's rounding special treatment: For values exactly in the middle, round to the nearest even number.
        fractional = _semantic.sub(arg0, basic_round, True)
        abs_fractional = core.tensor(_semantic.builder.create_fabs(fractional.handle), fractional.type)

        is_half = _semantic.equal(abs_fractional, half)

        two = _semantic.full(arg0.shape, 2.0, arg0.type.scalar)

        half_value = math.fdiv(basic_round, two, _semantic=_semantic)
        half_floor = core.tensor(_semantic.builder.create_floor(half_value.handle), half_value.type)
        double_half = _semantic.mul(half_floor, two, True)

        is_even = _semantic.equal(basic_round, double_half)

    adjustment = _semantic.where(is_positive, _semantic.full(arg0.shape, -1.0, arg0.type.scalar),
                                 _semantic.full(arg0.shape, 1.0, arg0.type.scalar))

    banker_result = _semantic.where(
        is_even,
        basic_round,
        _semantic.add(basic_round, adjustment, True),
    )
    # Final result: Use banker's rounding for cases exactly at 0.5, otherwise use basic rounding.
    return _semantic.where(is_half, banker_result, basic_round)


@core.builtin
@math._check_dtype(dtypes=[
    "fp32",
])
@math._add_math_1arg_docstr("arcsine")
def asin(arg0: core.tensor, _semantic=None):
    if triton_enable_libdevice_simt() and is_compile_on_910_95:
        return core.extern_elementwise(
            "", "", [arg0], {
                (core.dtype("fp16"), ): ("__hmf_asin_fp16", core.dtype("fp16")),
                (core.dtype("fp32"), ): ("__hmf_asin_fp32", core.dtype("fp32")),
            }, is_pure=True, _semantic=_semantic)
    else:
        """
        Calculate the principal value of the arc sine of the input argument x.

        Returns result in radians, in the interval [-π/2, +π/2] for x inside [-1, +1].
        Returns NaN for x outside [-1, +1].
        """
        arg0 = _semantic.to_tensor(arg0)

        # asin(x) = π/2 - acos(x)
        half_pi = _semantic.full(arg0.shape, 1.5707963267948966, arg0.type.scalar)  # π/2
        acos_val = acos(arg0, _semantic=_semantic)
        return _semantic.sub(half_pi, acos_val, True)


@core.builtin
@math._check_dtype(dtypes=[
    "fp32",
])
@math._add_math_1arg_docstr("base-10 logarithm")
def log10(arg0: core.tensor, _semantic=None):
    if triton_enable_libdevice_simt() and is_compile_on_910_95:
        return core.extern_elementwise("", "", [arg0], {
            (core.dtype("fp32"), ): ("__hmf_log10_fp32", core.dtype("fp32")),
        }, is_pure=True, _semantic=_semantic)
    else:
        """
        Calculate the base 10 logarithm of the input argument x.

        Returns NaN for x < 0, -inf for x = 0, and +0 for x = 1.
        log10(x) = log(x) / log(10)
        """
        arg0 = _semantic.to_tensor(arg0)

        log_val = math.log(arg0, _semantic=_semantic)
        log10_const = _semantic.full(arg0.shape, 2.302585092994046, arg0.type.scalar)

        return math.fdiv(log_val, log10_const, _semantic=_semantic)


@core.builtin
@math._check_dtype(dtypes=[
    "fp32",
])
@math._add_math_2arg_docstr("copysign")
def copysign(arg0: core.tensor, arg1: core.tensor, _semantic=None):
    if triton_enable_libdevice_simt() and is_compile_on_910_95:
        return core.extern_elementwise("", "", [arg0, arg1], {
            (core.dtype("fp32"), core.dtype("fp32")): ("__hmf_copysign_fp32", core.dtype("fp32")),
        }, is_pure=True, _semantic=_semantic)
    else:
        """
        Create a floating-point value with the magnitude of x and the sign of y.
        """
        x = _semantic.to_tensor(arg0)
        y = _semantic.to_tensor(arg1)

        magnitude = core.tensor(_semantic.builder.create_fabs(x.handle), x.type)

        zero = _semantic.full(y.shape, 0.0, y.type.scalar)
        one = _semantic.full(y.shape, 1.0, y.type.scalar)

        is_zero = _semantic.equal(y, zero)
        y_reciprocal = math.fdiv(one, y, _semantic=_semantic)
        is_negative_reciprocal = _semantic.less_than(y_reciprocal, zero)
        is_negative_zero = _semantic.and_(is_zero, is_negative_reciprocal)

        is_negative_nonzero = _semantic.less_than(y, zero)
        is_negative = _semantic.or_(is_negative_zero, is_negative_nonzero)

        neg_magnitude = _semantic.mul(magnitude, _semantic.full(magnitude.shape, -1.0, magnitude.type.scalar), True)

        return _semantic.where(is_negative, neg_magnitude, magnitude)


if get_ascend_arch_from_env() == "Ascend910_9589":
    # if we have hardware support
    @core.extern
    def rint(arg0, _semantic=None):
        return core.extern_elementwise(
            "", "", [arg0], {
                (core.dtype("fp32"), ): ("__hmf_rint", core.dtype("fp32")),
                (core.dtype("fp16"), ): ("__hmf_rint", core.dtype("fp16")),
                (core.dtype("bf16"), ): ("__hmf_rint", core.dtype("bf16")),
            }, is_pure=True, _semantic=_semantic)
else:

    @core.builtin
    @math._check_dtype(dtypes=["fp16", "fp32", "bf16"])
    @math._add_math_1arg_docstr("rint")
    def rint(arg0: core.tensor, _semantic=None):
        arg0 = _semantic.to_tensor(arg0)

        floor_x = math.floor(arg0, _semantic=_semantic)
        fractional = _semantic.sub(arg0, floor_x, True)

        half = _semantic.full(arg0.shape, 0.5, arg0.type.scalar)
        eps = _semantic.full(arg0.shape, 1e-8, arg0.type.scalar)
        is_half = _semantic.less_than(math.abs(_semantic.sub(fractional, half, True), _semantic=_semantic), eps)

        floor_int = floor_x.to(core.int32, _semantic=_semantic) if hasattr(floor_x, "to") else _semantic.cast(
            floor_x, core.int32)
        two_i32 = _semantic.full(arg0.shape, 2, core.int32)
        is_even = _semantic.equal(_semantic.mod(floor_int, two_i32), _semantic.full(arg0.shape, 0, core.int32))

        zero = _semantic.full(arg0.shape, 0.0, arg0.type.scalar)
        is_pos = _semantic.greater_equal(arg0, zero)

        round_pos = math.floor(_semantic.add(arg0, half, True), _semantic=_semantic)
        round_neg = math.ceil(_semantic.sub(arg0, half, True), _semantic=_semantic)
        normal_round = _semantic.where(is_pos, round_pos, round_neg)

        half_round = _semantic.where(is_even, floor_x, _semantic.add(floor_x, 1.0, True))

        return _semantic.where(is_half, half_round, normal_round)
