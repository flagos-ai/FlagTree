# SPDX-FileCopyrightText: Copyright (c) 2025 SpacemiT. All rights reserved.
# SPDX-License-Identifier: MIT

from triton.language import core


@core.extern
def abs(arg0, _semantic=None):
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("int32"), ): ("linalg.abs", core.dtype("int32")),
            (core.dtype("int64"), ): ("linalg.abs", core.dtype("int64")),
            (core.dtype("fp32"), ): ("linalg.abs", core.dtype("fp32")),
            (core.dtype("fp64"), ): ("linalg.abs", core.dtype("fp64")),
            (core.dtype("fp16"), ): ("linalg.abs", core.dtype("fp16")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def ceil(arg0, _semantic=None):
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp64"), ): ("linalg.ceil", core.dtype("fp64")),
            (core.dtype("fp32"), ): ("linalg.ceil", core.dtype("fp32")),
            (core.dtype("fp16"), ): ("linalg.ceil", core.dtype("fp16")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def exp(arg0, _semantic=None):
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp32"), ): ("linalg.exp", core.dtype("fp32")),
            (core.dtype("fp64"), ): ("linalg.exp", core.dtype("fp64")),
            (core.dtype("fp16"), ): ("linalg.exp", core.dtype("fp16")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def floor(arg0, _semantic=None):
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp32"), ): ("linalg.floor", core.dtype("fp32")),
            (core.dtype("fp64"), ): ("linalg.floor", core.dtype("fp64")),
            (core.dtype("fp16"), ): ("linalg.floor", core.dtype("fp16")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def log(arg0, _semantic=None):
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp32"), ): ("linalg.log", core.dtype("fp32")),
            (core.dtype("fp64"), ): ("linalg.log", core.dtype("fp64")),
            (core.dtype("fp16"), ): ("linalg.log", core.dtype("fp16")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def round(arg0, _semantic=None):
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp32"), ): ("linalg.round", core.dtype("fp32")),
            (core.dtype("fp64"), ): ("linalg.round", core.dtype("fp64")),
            (core.dtype("fp16"), ): ("linalg.round", core.dtype("fp16")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def rsqrt(arg0, _semantic=None):
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp32"), ): ("linalg.rsqrt", core.dtype("fp32")),
            (core.dtype("fp64"), ): ("linalg.rsqrt", core.dtype("fp64")),
            (core.dtype("fp16"), ): ("linalg.rsqrt", core.dtype("fp16")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def sqrt(arg0, _semantic=None):
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp32"), ): ("linalg.sqrt", core.dtype("fp32")),
            (core.dtype("fp64"), ): ("linalg.sqrt", core.dtype("fp64")),
            (core.dtype("fp16"), ): ("linalg.sqrt", core.dtype("fp16")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def tanh(arg0, _semantic=None):
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp32"), ): ("linalg.tanh", core.dtype("fp32")),
            (core.dtype("fp64"), ): ("linalg.tanh", core.dtype("fp64")),
            (core.dtype("fp16"), ): ("linalg.tanh", core.dtype("fp16")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def erf(arg0, _semantic=None):
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp32"), ): ("linalg.erf", core.dtype("fp32")),
            (core.dtype("fp64"), ): ("linalg.erf", core.dtype("fp64")),
            (core.dtype("fp16"), ): ("linalg.erf", core.dtype("fp16")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def pow(arg0, arg1, _semantic=None):
    if _semantic is not None:
        # binary_op_type_checking_impl only routes numbers.Number through
        # to_tensor; constexpr-wrapped constants must be unwrapped first or
        # its `.type.scalar` access raises on constexpr_type (e.g. pow(x, 2)
        # inside @triton.jit, where 2 arrives as constexpr[2]).
        if isinstance(arg0, core.constexpr):
            arg0 = arg0.value
        if isinstance(arg1, core.constexpr):
            arg1 = arg1.value
        arg0, arg1 = _semantic.binary_op_type_checking_impl(arg0, arg1)
    return core.extern_elementwise(
        "", "", [arg0, arg1], {
            (core.dtype("fp32"), core.dtype("fp32")): ("linalg.powf", core.dtype("fp32")),
            (core.dtype("fp64"), core.dtype("fp64")): ("linalg.powf", core.dtype("fp64")),
            (core.dtype("fp16"), core.dtype("fp16")): ("linalg.powf", core.dtype("fp16")),
            (core.dtype("fp32"), core.dtype("int32")): ("linalg.powf", core.dtype("fp32")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def gelu_tanh(arg0, _semantic=None):
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp32"), ): ("linalg.gelu_tanh", core.dtype("fp32")),
            (core.dtype("fp64"), ): ("linalg.gelu_tanh", core.dtype("fp64")),
            (core.dtype("fp16"), ): ("linalg.gelu_tanh", core.dtype("fp16")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def gelu_none(arg0, _semantic=None):
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp32"), ): ("linalg.gelu_none", core.dtype("fp32")),
            (core.dtype("fp64"), ): ("linalg.gelu_none", core.dtype("fp64")),
            (core.dtype("fp16"), ): ("linalg.gelu_none", core.dtype("fp16")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def silu(arg0, _semantic=None):
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp32"), ): ("linalg.silu", core.dtype("fp32")),
            (core.dtype("fp64"), ): ("linalg.silu", core.dtype("fp64")),
            (core.dtype("fp16"), ): ("linalg.silu", core.dtype("fp16")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def cos(arg0, _semantic=None):
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp32"), ): ("math.cos", core.dtype("fp32")),
            (core.dtype("fp64"), ): ("math.cos", core.dtype("fp64")),
            (core.dtype("fp16"), ): ("math.cos", core.dtype("fp16")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def sin(arg0, _semantic=None):
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp32"), ): ("math.sin", core.dtype("fp32")),
            (core.dtype("fp64"), ): ("math.sin", core.dtype("fp64")),
            (core.dtype("fp16"), ): ("math.sin", core.dtype("fp16")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def acos(arg0, _semantic=None):
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp32"), ): ("math.acos", core.dtype("fp32")),
            (core.dtype("fp64"), ): ("math.acos", core.dtype("fp64")),
            (core.dtype("fp16"), ): ("math.acos", core.dtype("fp16")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def acosh(arg0, _semantic=None):
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp32"), ): ("math.acosh", core.dtype("fp32")),
            (core.dtype("fp64"), ): ("math.acosh", core.dtype("fp64")),
            (core.dtype("fp16"), ): ("math.acosh", core.dtype("fp16")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def asin(arg0, _semantic=None):
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp32"), ): ("math.asin", core.dtype("fp32")),
            (core.dtype("fp64"), ): ("math.asin", core.dtype("fp64")),
            (core.dtype("fp16"), ): ("math.asin", core.dtype("fp16")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def asinh(arg0, _semantic=None):
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp32"), ): ("math.asinh", core.dtype("fp32")),
            (core.dtype("fp64"), ): ("math.asinh", core.dtype("fp64")),
            (core.dtype("fp16"), ): ("math.asinh", core.dtype("fp16")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def atan(arg0, _semantic=None):
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp32"), ): ("math.atan", core.dtype("fp32")),
            (core.dtype("fp64"), ): ("math.atan", core.dtype("fp64")),
            (core.dtype("fp16"), ): ("math.atan", core.dtype("fp16")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def atanh(arg0, _semantic=None):
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp32"), ): ("math.atanh", core.dtype("fp32")),
            (core.dtype("fp64"), ): ("math.atanh", core.dtype("fp64")),
            (core.dtype("fp16"), ): ("math.atanh", core.dtype("fp16")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def atan2(arg0, arg1, _semantic=None):
    return core.extern_elementwise(
        "", "", [arg0, arg1], {
            (core.dtype("fp32"), core.dtype("fp32")): ("math.atan2", core.dtype("fp32")),
            (core.dtype("fp64"), core.dtype("fp64")): ("math.atan2", core.dtype("fp64")),
            (core.dtype("fp16"), core.dtype("fp16")): ("math.atan2", core.dtype("fp16")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def cbrt(arg0, _semantic=None):
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp32"), ): ("math.cbrt", core.dtype("fp32")),
            (core.dtype("fp64"), ): ("math.cbrt", core.dtype("fp64")),
            (core.dtype("fp16"), ): ("math.cbrt", core.dtype("fp16")),
        }, is_pure=True, _semantic=_semantic)


# @core.extern
# def cos(arg0, _semantic=None):
#     return core.tensor(_semantic.create_cos(arg0.handle), arg0.type)


@core.extern
def cosh(arg0, _semantic=None):
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp32"), ): ("math.cosh", core.dtype("fp32")),
            (core.dtype("fp64"), ): ("math.cosh", core.dtype("fp64")),
            (core.dtype("fp16"), ): ("math.cosh", core.dtype("fp16")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def exp2(arg0, _semantic=None):
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp32"), ): ("math.exp2", core.dtype("fp32")),
            (core.dtype("fp64"), ): ("math.exp2", core.dtype("fp64")),
            (core.dtype("fp16"), ): ("math.exp2", core.dtype("fp16")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def expm1(arg0, _semantic=None):
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp32"), ): ("math.expm1", core.dtype("fp32")),
            (core.dtype("fp64"), ): ("math.expm1", core.dtype("fp64")),
            (core.dtype("fp16"), ): ("math.expm1", core.dtype("fp16")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def log2(arg0, _semantic=None):
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp32"), ): ("math.log2", core.dtype("fp32")),
            (core.dtype("fp64"), ): ("math.log2", core.dtype("fp64")),
            (core.dtype("fp16"), ): ("math.log2", core.dtype("fp16")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def log10(arg0, _semantic=None):
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp32"), ): ("math.log10", core.dtype("fp32")),
            (core.dtype("fp64"), ): ("math.log10", core.dtype("fp64")),
            (core.dtype("fp16"), ): ("math.log10", core.dtype("fp16")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def log1p(arg0, _semantic=None):
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp32"), ): ("math.log1p", core.dtype("fp32")),
            (core.dtype("fp64"), ): ("math.log1p", core.dtype("fp64")),
            (core.dtype("fp16"), ): ("math.log1p", core.dtype("fp16")),
        }, is_pure=True, _semantic=_semantic)


# @core.extern
# def sin(arg0, _semantic=None):
#     return core.tensor(_semantic.create_sin(arg0.handle), arg0.type)


@core.extern
def sinh(arg0, _semantic=None):
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp32"), ): ("math.sinh", core.dtype("fp32")),
            (core.dtype("fp64"), ): ("math.sinh", core.dtype("fp64")),
            (core.dtype("fp16"), ): ("math.sinh", core.dtype("fp16")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def tan(arg0, _semantic=None):
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp32"), ): ("math.tan", core.dtype("fp32")),
            (core.dtype("fp64"), ): ("math.tan", core.dtype("fp64")),
            (core.dtype("fp16"), ): ("math.tan", core.dtype("fp16")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def ffs(arg0, _semantic=None):
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("int32"), ): ("math.ffs", core.dtype("int32")),
            (core.dtype("int64"), ): ("math.ffs", core.dtype("int64")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def trunc(arg0, _semantic=None):
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp32"), ): ("math.trunc", core.dtype("fp32")),
            (core.dtype("fp64"), ): ("math.trunc", core.dtype("fp64")),
            (core.dtype("fp16"), ): ("math.trunc", core.dtype("fp16")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def div_rn(arg0, arg1, _semantic=None):
    return core.extern_elementwise(
        "", "", [arg0, arg1], {
            (core.dtype("fp32"), core.dtype("fp32")): ("linalg.div_rn", core.dtype("fp32")),
            (core.dtype("fp64"), core.dtype("fp64")): ("linalg.div_rn", core.dtype("fp64")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def div_rz(arg0, arg1, _semantic=None):
    return core.extern_elementwise(
        "", "", [arg0, arg1], {
            (core.dtype("fp32"), core.dtype("fp32")): ("linalg.div_rz", core.dtype("fp32")),
            (core.dtype("fp64"), core.dtype("fp64")): ("linalg.div_rz", core.dtype("fp64")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def div_rd(arg0, arg1, _semantic=None):
    return core.extern_elementwise(
        "", "", [arg0, arg1], {
            (core.dtype("fp32"), core.dtype("fp32")): ("linalg.div_rd", core.dtype("fp32")),
            (core.dtype("fp64"), core.dtype("fp64")): ("linalg.div_rd", core.dtype("fp64")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def fmod(arg0, arg1, _semantic=None):
    return core.extern_elementwise(
        "", "", [arg0, arg1], {
            (core.dtype("fp32"), core.dtype("fp32")): ("linalg.fmod", core.dtype("fp32")),
            (core.dtype("fp64"), core.dtype("fp64")): ("linalg.fmod", core.dtype("fp64")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def div_ru(arg0, arg1, _semantic=None):
    return core.extern_elementwise(
        "", "", [arg0, arg1], {
            (core.dtype("fp32"), core.dtype("fp32")): ("linalg.div_ru", core.dtype("fp32")),
            (core.dtype("fp64"), core.dtype("fp64")): ("linalg.div_ru", core.dtype("fp64")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def rint(arg0, _semantic=None):
    return core.extern_elementwise(
        "", "", [
            arg0,
        ], {
            (core.dtype("fp32"), ): ("linalg.rint", core.dtype("fp32")),
            (core.dtype("fp64"), ): ("linalg.rint", core.dtype("fp64")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def finitef(arg0, _semantic=None):
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("fp32"), ): ("math.isfinite", core.dtype("int32")),
    }, is_pure=True, _semantic=_semantic).to(core.int1, _semantic=_semantic)


@core.extern
def isinf(arg0, _semantic=None):
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp16"), ): ("math.isinf", core.dtype("int32")),
            (core.dtype("fp32"), ): ("math.isinf", core.dtype("int32")),
            (core.dtype("fp64"), ): ("math.isinf", core.dtype("int32")),
        }, is_pure=True, _semantic=_semantic).to(core.int1, _semantic=_semantic)


@core.extern
def isnan(arg0, _semantic=None):
    return core.extern_elementwise(
        "", "", [
            arg0,
        ], {
            (core.dtype("fp16"), ): ("math.isnan", core.dtype("int32")),
            (core.dtype("fp32"), ): ("math.isnan", core.dtype("int32")),
            (core.dtype("fp64"), ): ("math.isnan", core.dtype("int32")),
        }, is_pure=True, _semantic=_semantic).to(core.int1, _semantic=_semantic)


@core.extern
def isfinited(arg0, _semantic=None):
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("fp64"), ): ("math.isfinite", core.dtype("int32")),
    }, is_pure=True, _semantic=_semantic).to(core.int1, _semantic=_semantic)
