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
from triton.language import core, math, semantic, standard
from triton._C.libtriton import ir
from triton.runtime.jit import jit
from triton.backends.ascend.utils import is_compile_on_910_95, triton_enable_libdevice_simt

from .utils import _deprecated


def _is_libdevice_simt_enabled(_semantic) -> bool:
    return triton_enable_libdevice_simt(_semantic.builder.options.arch)


def _is_a5_target(_semantic) -> bool:
    return is_compile_on_910_95(_semantic.builder.options.arch)


class _FlipStaticRange:

    def __init__(self, arg1, arg2=None, step=None):
        self.step = core.constexpr(1) if step is None else step
        if arg2 is None:
            self.start = core.constexpr(0)
            self.end = arg1
        else:
            self.start = arg1
            self.end = arg2

    def __iter__(self):
        self._current = core._unwrap_if_constexpr(self.start)
        self._end = core._unwrap_if_constexpr(self.end)
        self._step = core._unwrap_if_constexpr(self.step)
        return self

    def __next__(self):
        if self._current >= self._end:
            raise StopIteration
        value = self._current
        self._current += self._step
        return value


@core.builtin
def flip(ptr, dim=-1, _semantic=None, _generator=None):
    """Flips a tensor along the specified dimension."""

    def flip_impl(ptr: core.tensor, dim: int, builder: ir.builder, generator=None):

        def _get_flip_dim(dim, shape):
            dim = core._unwrap_if_constexpr(dim)
            shape = core._unwrap_if_constexpr(shape)
            if dim is None:
                dim = len(shape) - 1
            if dim < 0:
                dim += len(shape)
            return core.constexpr(dim)

        def _log2(i: core.constexpr):
            log2 = 0
            n = core.constexpr(i).value
            while n > 1:
                n >>= 1
                log2 += 1
            return core.constexpr(log2)

        def flip_simd(ptr: core.tensor, dim: int, builder: ir.builder):
            shape = getattr(ptr, "shape", None)
            if shape is None or shape == ():
                shape = getattr(getattr(ptr, "type", None), "shape", None)

            rank = None
            if shape is not None:
                try:
                    rank = len(shape)
                except Exception:
                    rank = len(list(shape))

            if rank is not None:
                if rank < 1:
                    raise ValueError("ascend.flip requires tensor rank >= 1")
                norm_dim = dim if dim >= 0 else dim + rank
                if not (0 <= norm_dim < rank):
                    raise ValueError(f"ascend.flip got invalid dim={dim} for shape {tuple(shape)}")
                dim = norm_dim
            elif dim < 0:
                raise ValueError("ascend.flip with unknown rank requires non-negative dim")

            flipped_vals = builder.create_flip(ptr.handle, dim)
            return core.tensor(flipped_vals, type=ptr.type)

        if not builder.is_simt_mode():
            return flip_simd(ptr, dim, builder)
        if not (-len(ptr.shape) <= dim < len(ptr.shape)):
            raise ValueError(f"invalid dim={dim} for shape {tuple(ptr.shape)}")
        flip_dim = core._unwrap_if_constexpr(_get_flip_dim(dim, ptr.shape))
        if not standard._is_power_of_two(ptr.shape[flip_dim]):
            raise ValueError("flip in SIMT mode requires the flipped dimension to be a power of two")
        steps = core._unwrap_if_constexpr(_log2(ptr.shape[flip_dim]))
        if steps == 0:
            return ptr

        idtype = core.get_int_dtype(bitwidth=ptr.dtype.primitive_bitwidth, signed=True)
        reshaped = core.reshape(
            ptr.to(idtype, bitcast=True, _semantic=_semantic),
            ptr.shape.__getitem__(slice(None, flip_dim)) + [2] * steps +
            ptr.shape.__getitem__(slice(flip_dim + 1, None)),
            _semantic=_semantic,
            _generator=_generator,
        )
        for i in _FlipStaticRange(steps):
            reduced = core.reduce(
                reshaped,
                flip_dim + i,
                standard._xor_combine,
                keep_dims=True,
                _semantic=_semantic,
                _generator=generator,
            )
            reshaped = reshaped.__xor__(reduced, _semantic=_semantic)
        return core.reshape(reshaped, ptr.shape, _semantic=_semantic, _generator=_generator).to(
            ptr.dtype,
            bitcast=True,
            _semantic=_semantic,
        )

    try:
        dim = int(dim.value) if hasattr(dim, "value") else int(dim)
    except Exception as exc:
        raise TypeError(f"dim must be an integer (or tl.constexpr int), got {dim!r}") from exc

    dim = len(ptr.shape) - 1 if dim == -1 else dim
    return flip_impl(ptr, dim, _semantic.builder, _generator)


@core.extern
def reciprocal(arg0, _semantic=None):
    """
    Computes the reciprocal of x (i.e., 1 / x).

    :param arg0: ``x``. Supported dtype(s): ``float32``.
    :type arg0: scalar or tl.tensor
    :return: 1 / x.
    :rtype: ``float32``
    """
    if _is_libdevice_simt_enabled(_semantic):
        return core.extern_elementwise("", "", [arg0], {
            (core.dtype("fp32"), ): ("__hmf_reciprocal_fp32", core.dtype("fp32")),
        }, is_pure=True, _semantic=_semantic)
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp32"), ): ("__hmf_recipf", core.dtype("fp32")),
            (core.dtype("fp16"), ): ("__hmf_recipDh", core.dtype("fp16")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def log1p(arg0, _semantic=None):
    """
    Computes the value of log(1 + x).

    :param arg0: ``x``. Supported dtype(s): ``float32``.
    :type arg0: scalar or tl.tensor
    :return: The result of log(1 + x).
    :rtype: ``float32``
    """
    if _is_libdevice_simt_enabled(_semantic):
        return core.extern_elementwise("", "", [arg0], {
            (core.dtype("fp32"), ): ("__hmf_log1p_fp32", core.dtype("fp32")),
        }, is_pure=True, _semantic=_semantic)
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp32"), ): ("__hmf_log1pf", core.dtype("fp32")),
            (core.dtype("fp16"), ): ("__hmf_log1pDh", core.dtype("fp16")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def relu(arg0, _semantic=None):
    """
    Rectified linear unit function, returns x when x > 0, otherwise returns 0.

    :param arg0: ``x``. Supported dtype(s): ``float32``.
    :type arg0: scalar or tl.tensor
    :return: The result of the rectified linear unit.
    :rtype: ``float32``
    """
    if _is_libdevice_simt_enabled(_semantic):
        return core.extern_elementwise("", "", [arg0], {
            (core.dtype("fp32"), ): ("__hmf_relu_fp32", core.dtype("fp32")),
        }, is_pure=True, _semantic=_semantic)
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp32"), ): ("__hmf_reluf", core.dtype("fp32")),
            (core.dtype("fp16"), ): ("__hmf_reluDh", core.dtype("fp16")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def isinf(arg0, _semantic=None):
    """
    Determines whether the input is infinity.

    :param arg0: ``x``. Supported dtype(s): ``float32``.
    :type arg0: scalar or tl.tensor
    :return: True if the input is infinity; otherwise, False.
    :rtype: ``bool``
    """
    if _is_libdevice_simt_enabled(_semantic):
        return core.extern_elementwise("", "", [arg0], {
            (core.dtype("fp32"), ): ("__hmf_isinf_fp32", core.dtype("int1")),
        }, is_pure=True, _semantic=_semantic)
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp32"), ): ("__hmf_isinf", core.dtype("int1")),
            (core.dtype("fp16"), ): ("__hmf_isinf", core.dtype("int1")),
            (core.dtype("bf16"), ): ("__hmf_isinf", core.dtype("int1")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def tan(arg0, _semantic=None):
    """
    Computes the tangent of input parameter x (in radians).

    :param arg0: ``x``. Supported dtype(s): ``float32``.
    :type arg0: scalar or tl.tensor
    :return: The tangent of input x.
    :rtype: ``float32``
    """
    if _is_libdevice_simt_enabled(_semantic):
        return core.extern_elementwise("", "", [arg0], {
            (core.dtype("fp32"), ): ("__hmf_tan_fp32", core.dtype("fp32")),
        }, is_pure=True, _semantic=_semantic)
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp32"), ): ("__hmf_tanf", core.dtype("fp32")),
            (core.dtype("fp16"), ): ("__hmf_tanDh", core.dtype("fp16")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def atan(arg0, _semantic=None):
    """
    Computes the inverse tangent (arctan) of the input parameter.

    :param arg0: ``x``. Supported dtype(s): ``float32``.
    :type arg0: scalar or tl.tensor
    :return: The inverse tangent of the input parameter, in the range [-π/2, π/2] radians.
    :rtype: ``float32``
    """
    if _is_libdevice_simt_enabled(_semantic):
        return core.extern_elementwise("", "", [arg0], {
            (core.dtype("fp32"), ): ("__hmf_atan_fp32", core.dtype("fp32")),
        }, is_pure=True, _semantic=_semantic)
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp32"), ): ("__hmf_atanf", core.dtype("fp32")),
            (core.dtype("fp16"), ): ("__hmf_atanDh", core.dtype("fp16")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def tanh(arg0, _semantic=None):
    """
    Computes the hyperbolic tangent of input parameter x.

    :param arg0: ``x``. Supported dtype(s): ``float32``.
    :type arg0: scalar or tl.tensor
    :return: The hyperbolic tangent of input x.
    :rtype: ``float32``
    """
    arg0 = _semantic.to_tensor(arg0)
    original_dtype = arg0.dtype
    if original_dtype == core.dtype("bf16"):
        arg0 = _semantic.cast(arg0, core.float32)

    if _is_libdevice_simt_enabled(_semantic):
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
    """
    Extracts the unbiased exponent (base-2 integer logarithm) of a floating-point number.

    :param arg0: ``x``. Supported dtype(s): ``float32``.
    :type arg0: scalar or tl.tensor
    :return: The unbiased exponent of the input parameter.
    :rtype: ``float32``
    """
    if _is_libdevice_simt_enabled(_semantic):
        return core.extern_elementwise("", "", [arg0], {
            (core.dtype("fp32"), ): ("__hmf_ilogb_fp32", core.dtype("fp32")),
        }, is_pure=True, _semantic=_semantic)
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp32"), ): ("__hmf_ilogbf", core.dtype("fp32")),
            (core.dtype("fp16"), ): ("__hmf_ilogbDh", core.dtype("fp16")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def logb(arg0, _semantic=None):
    """
    Extracts the exponent value of a floating-point number.

    :param arg0: ``x``. Supported dtype(s): ``float32``.
    :type arg0: scalar or tl.tensor
    :return: The exponent value of the input parameter.
    :rtype: ``float32``
    """
    if not _is_a5_target(_semantic):
        core.static_print("libdevice.logb for simd is unsupported for now.")
        core.static_assert(False)
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("fp32"), ): ("__hmf_logb_fp32", core.dtype("fp32")),
    }, is_pure=True, _semantic=_semantic)


@core.extern
def ldexp(arg0, arg1, _semantic=None):
    """
    Computes the value of x × 2^exp.

    :param arg0: ``x``. Supported dtype(s): ``float32``.
    :type arg0: scalar or tl.tensor
    :param arg1: ``exp``. Supported dtype(s): ``int32``.
    :type arg1: scalar or tl.tensor
    :return: The result of x × 2^exp.
    :rtype: ``float32``
    """
    if _is_libdevice_simt_enabled(_semantic):
        return core.extern_elementwise("", "", [arg0, arg1], {
            (core.dtype("fp32"), core.dtype("int32")): ("__hmf_ldexp_fp32", core.dtype("fp32")),
        }, is_pure=True, _semantic=_semantic)
    return core.extern_elementwise(
        "", "", [arg0, arg1], {
            (core.dtype("fp32"), core.dtype("int32")): ("__hmf_ldexpf", core.dtype("fp32")),
            (core.dtype("fp16"), core.dtype("int32")): ("__hmf_ldexpDh", core.dtype("fp16")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def scalbn(arg0, arg1, _semantic=None):
    """
    Computes the value of x × 2^n.

    :param arg0: ``x``. Supported dtype(s): ``float32``.
    :type arg0: scalar or tl.tensor
    :param arg1: ``n``. Supported dtype(s): ``int32``.
    :type arg1: scalar or tl.tensor
    :return: The result of x × 2^n.
    :rtype: ``float32``
    """
    if not _is_a5_target(_semantic):
        core.static_print("libdevice.scalbn for simd is unsupported for now.")
        core.static_assert(False)
    return core.extern_elementwise("", "", [arg0, arg1], {
        (core.dtype("fp32"), core.dtype("int32")): ("__hmf_scalbn_fp32", core.dtype("fp32")),
    }, is_pure=True, _semantic=_semantic)


@core.extern
def pow(arg0, arg1, _semantic=None):
    """
    Power function, computes x raised to the power of y.

    :param arg0: ``x``. Supported dtype(s): ``float32``.
    :type arg0: scalar or tl.tensor
    :param arg1: ``y``. Supported dtype(s): ``float32``.
    :type arg1: scalar or tl.tensor
    :return: x raised to the power of y.
    :rtype: ``float32``
    """
    arg0 = _semantic.to_tensor(arg0)
    arg1 = _semantic.to_tensor(arg1)
    if arg1.dtype == core.dtype("int32"):
        arg1 = _semantic.cast(arg1, arg0.dtype)

    if arg0.dtype == core.dtype("fp32") and _is_a5_target(_semantic):
        return core.extern_elementwise(
            "", "", [arg0, arg1], {
                (core.dtype("fp32"), core.dtype("fp32")): ("__hmf_pow_fp32", core.dtype("fp32")),
                (core.dtype("fp32"), core.dtype("int32")): ("__hmf_powi_fp32", core.dtype("fp32")),
            }, is_pure=True, _semantic=_semantic)
    return core.extern_elementwise(
        "", "", [arg0, arg1], {
            (core.dtype("fp32"), core.dtype("fp32")): ("__hmf_powf", core.dtype("fp32")),
            (core.dtype("fp16"), core.dtype("fp16")): ("__hmf_powDh", core.dtype("fp16")),
            (core.dtype("bf16"), core.dtype("bf16")): ("__hmf_powDb", core.dtype("bf16")),
        }, is_pure=True, _semantic=_semantic)


@core._tensor_member_fn
@jit
@math._add_math_1arg_docstr("isfinited")
def isfinited(arg0):
    _is_int8_type: core.constexpr = arg0.dtype.is_int8()
    core.static_assert(
        not _is_int8_type,
        "Expected dtype fp16/fp32/bf16, but got int8 or int1",
    )
    _is_floating_type: core.constexpr = arg0.dtype.is_floating()
    core.static_assert(
        _is_floating_type == True,
        f"Expected dtype fp16/fp32/bf16, but got {core.constexpr(arg0.dtype)}",
    )
    nan_mask = isnan(arg0)
    inf_mask = isinf(arg0)
    return (~nan_mask & ~inf_mask).to(core.int1)


@core.extern
def finitef(arg0, _semantic=None):
    """
    Determines whether the input is a finite floating-point number.

    :param arg0: ``x``. Supported dtype(s): ``float32``.
    :type arg0: scalar or tl.tensor
    :return: True if the input is finite; otherwise, False.
    :rtype: ``bool``
    """
    if _is_libdevice_simt_enabled(_semantic):
        return core.extern_elementwise("", "", [arg0], {
            (core.dtype("fp32"), ): ("__hmf_finite_fp32", core.dtype("int1")),
        }, is_pure=True, _semantic=_semantic)
    arg0 = _semantic.to_tensor(arg0)
    if arg0.dtype.is_int8():
        raise ValueError("finitef only supports float32, but got int8 or int1")
    if arg0.dtype != core.float32:
        raise ValueError(f"finitef only supports float32, but got {core.constexpr(arg0.dtype)}")
    nan_mask = isnan(arg0, _semantic=_semantic)
    inf_mask = isinf(arg0, _semantic=_semantic)
    return _semantic.logical_and(_semantic.not_(nan_mask), _semantic.not_(inf_mask))


@core.extern
def isnan(arg0, _semantic=None):
    """
    Determines whether the input is NaN (not a number).

    :param arg0: ``x``. Supported dtype(s): ``float32``.
    :type arg0: scalar or tl.tensor
    :return: True if the input is NaN; otherwise, False.
    :rtype: ``bool``
    """
    if _is_libdevice_simt_enabled(_semantic):
        return core.extern_elementwise("", "", [arg0], {
            (core.dtype("fp32"), ): ("__hmf_isnan_fp32", core.dtype("int1")),
        }, is_pure=True, _semantic=_semantic)
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp32"), ): ("__hmf_isnan", core.dtype("int1")),
            (core.dtype("fp16"), ): ("__hmf_isnan", core.dtype("int1")),
            (core.dtype("bf16"), ): ("__hmf_isnan", core.dtype("int1")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def clz(arg0, _semantic=None):
    """
    Counts the number of leading zeros in a 32-bit integer.

    :param arg0: ``x``. Supported dtype(s): ``int32``.
    :type arg0: scalar or tl.tensor
    :return: The number of leading zeros in the input parameter. Range: [0, 32].
    :rtype: ``int32``
    """
    if not _is_a5_target(_semantic):
        core.static_print("libdevice.clz for simd is unsupported for now.")
        core.static_assert(False)
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("int32"), ): ("__hmf_clz_i32", core.dtype("int32")),
    }, is_pure=True, _semantic=_semantic)


@core.extern
def popc(arg0, _semantic=None):
    """
    Counts the number of bits set to 1 in x.

    :param arg0: ``x``. Supported dtype(s): ``int32``.
    :type arg0: scalar or tl.tensor
    :return: The number of bits set to 1 in x. Range: [0, 32].
    :rtype: ``int32``
    """
    if not _is_a5_target(_semantic):
        core.static_print("libdevice.popc for simd is unsupported for now.")
        core.static_assert(False)
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("int32"), ): ("__hmf_popc_i32", core.dtype("int32")),
    }, is_pure=True, _semantic=_semantic)


@core.extern
def byte_perm(arg0, arg1, arg2, _semantic=None):
    """
    Selects bytes from two 32-bit integers x and y according to selector s and combines them into a new integer.

    :param arg0: ``x``. Supported dtype(s): ``int32``.
    :type arg0: scalar or tl.tensor
    :param arg1: ``y``. Supported dtype(s): ``int32``.
    :type arg1: scalar or tl.tensor
    :param arg2: ``s``. Supported dtype(s): ``int32``.
    :type arg2: scalar or tl.tensor
    :return: The integer whose n-th byte is selected from x and y by selector s.
    :rtype: ``int32``
    """
    if not _is_a5_target(_semantic):
        core.static_print("libdevice.byte_perm for simd is unsupported for now.")
        core.static_assert(False)
    return core.extern_elementwise("", "", [arg0, arg1, arg2], {
        (core.dtype("int32"), core.dtype("int32"), core.dtype("int32")): ("__hmf_byte_perm_i32", core.dtype("int32")),
    }, is_pure=True, _semantic=_semantic)


@core.extern
def mulhi(arg0, arg1, _semantic=None):
    """
    Computes the high 32 bits of the multiplication result of x and y.

    :param arg0: ``x``. Supported dtype(s): ``int32`` or ``uint32``.
    :type arg0: scalar or tl.tensor
    :param arg1: ``y``. Supported dtype(s): ``int32`` or ``uint32``; must have the same type as x.
    :type arg1: scalar or tl.tensor
    :return: The high 32 bits of the multiplication result of x and y.
    :rtype: Same as the input type (``int32`` or ``uint32``)
    """
    if not _is_a5_target(_semantic):
        arg0 = _semantic.to_tensor(arg0)
        arg1 = _semantic.to_tensor(arg1)
        if arg0.dtype == core.uint32 and arg1.dtype == arg0.dtype:
            return core.tensor(_semantic.builder.create_umulhi(arg0.handle, arg1.handle), arg0.type)
        core.static_print("libdevice.mulhi for this dtype in simd is unsupported for now.")
        core.static_assert(False)
    return core.extern_elementwise(
        "", "", [arg0, arg1], {
            (core.dtype("int32"), core.dtype("int32")): ("__hmf_mulhi_i32", core.dtype("int32")),
            (core.dtype("uint32"), core.dtype("uint32")): ("__hmf_umulhi_u32", core.dtype("uint32")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def mul24(arg0, arg1, _semantic=None):
    """
    Computes the lower 24-bit multiplication result of x and y.

    :param arg0: ``x``. Supported dtype(s): ``int32`` or ``uint32``.
    :type arg0: scalar or tl.tensor
    :param arg1: ``y``. Supported dtype(s): ``int32`` or ``uint32``; must have the same type as x.
    :type arg1: scalar or tl.tensor
    :return: The lower 24-bit multiplication result of x and y.
    :rtype: Same as the input type (``int32`` or ``uint32``)
    """
    if not _is_a5_target(_semantic):
        core.static_print("libdevice.mul24 for simd is unsupported for now.")
        core.static_assert(False)
    return core.extern_elementwise(
        "", "", [arg0, arg1], {
            (core.dtype("int32"), core.dtype("int32")): ("__hmf_mul24_i32", core.dtype("int32")),
            (core.dtype("uint32"), core.dtype("uint32")): ("__hmf_umul24_u32", core.dtype("uint32")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def brev(arg0, _semantic=None):
    """
    Bit reversal function, reverses the bit order of a 32-bit integer.

    :param arg0: ``x``. Supported dtype(s): ``int32``.
    :type arg0: scalar or tl.tensor
    :return: The 32-bit integer with reversed bit order.
    :rtype: ``int32``
    """
    if not _is_a5_target(_semantic):
        core.static_print("libdevice.brev for simd is unsupported for now.")
        core.static_assert(False)
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("int32"), ): ("__hmf_brev_i32", core.dtype("int32")),
    }, is_pure=True, _semantic=_semantic)


@core.extern
def sad(arg0, arg1, arg2, _semantic=None):
    """
    Computes abs(x - y) + z for signed or unsigned 32-bit integers.

    :param arg0: ``x``.
    :type arg0: scalar or tl.tensor
    :param arg1: ``y``.
    :type arg1: scalar or tl.tensor
    :param arg2: ``z``.
    :type arg2: scalar or tl.tensor

    Supported dtype signatures:

    - (``int32``, ``int32``, ``uint32``) -> ``int32``
    - (``uint32``, ``uint32``, ``uint32``) -> ``uint32``

    :return: The result of abs(x - y) + z.
    :rtype: ``int32`` or ``uint32``, as specified by the supported signatures
    """
    if not _is_a5_target(_semantic):
        core.static_print("libdevice.sad for simd is unsupported for now.")
        core.static_assert(False)
    return core.extern_elementwise(
        "", "", [arg0, arg1, arg2], {
            (core.dtype("int32"), core.dtype("int32"), core.dtype("uint32")): ("__hmf_sad_i32", core.dtype("int32")),
            (core.dtype("uint32"), core.dtype("uint32"), core.dtype("uint32")):
            ("__hmf_usad_u32", core.dtype("uint32")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def ffs(arg0, _semantic=None):
    """
    Finds the first bit set to 1 and returns the index of the lowest bit set to 1.

    :param arg0: ``x``. Supported dtype(s): ``int32``.
    :type arg0: scalar or tl.tensor
    :return: The index of the lowest bit set to 1. Range: [0, 32].
    :rtype: ``int32``
    """
    arg0 = _semantic.to_tensor(arg0)
    dtype = arg0.dtype
    if _is_a5_target(_semantic):
        return core.extern_elementwise(
            "", "", [arg0], {
                (core.dtype("int32"), ): ("__hmf_ffs_i32", core.dtype("int32")),
                (core.dtype("int64"), ): ("__hmf_ffs_i64", core.dtype("int32")),
            }, is_pure=True, _semantic=_semantic)
    core.static_print(f"libdevice.ffs for {dtype} is unsupported for now.")
    core.static_assert(False)


@core.extern
def saturatef(arg0, _semantic=None):
    """
    Clamps x to the range [+0.0, 1.0].

    :param arg0: ``x``. Supported dtype(s): ``float32``.
    :type arg0: scalar or tl.tensor
    :return: The saturated value of x, in the range [+0.0, 1.0].
    :rtype: ``float32``
    """
    if not _is_a5_target(_semantic):
        core.static_print("libdevice.saturatef for simd is unsupported for now.")
        core.static_assert(False)
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("fp32"), ): ("__hmf_saturate_fp32", core.dtype("fp32")),
    }, is_pure=True, _semantic=_semantic)


@core.extern
def hadd(arg0, arg1, _semantic=None):
    """
    Computes the average of x and y.

    :param arg0: ``x``. Supported dtype(s): ``int32`` or ``uint32``.
    :type arg0: scalar or tl.tensor
    :param arg1: ``y``. Supported dtype(s): ``int32`` or ``uint32``; must have the same type as x.
    :type arg1: scalar or tl.tensor
    :return: The average of x and y.
    :rtype: Same as the input type (``int32`` or ``uint32``)
    """
    if not _is_a5_target(_semantic):
        core.static_print("libdevice.hadd for simd is unsupported for now.")
        core.static_assert(False)
    return core.extern_elementwise(
        "", "", [arg0, arg1], {
            (core.dtype("int32"), core.dtype("int32")): ("__hmf_hadd_i32", core.dtype("int32")),
            (core.dtype("uint32"), core.dtype("uint32")): ("__hmf_uhadd_u32", core.dtype("uint32")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def rhadd(arg0, arg1, _semantic=None):
    """
    Computes the rounded average of x and y.

    :param arg0: ``x``. Supported dtype(s): ``int32`` or ``uint32``.
    :type arg0: scalar or tl.tensor
    :param arg1: ``y``. Supported dtype(s): ``int32`` or ``uint32``; must have the same type as x.
    :type arg1: scalar or tl.tensor
    :return: The rounded average of x and y.
    :rtype: Same as the input type (``int32`` or ``uint32``)
    """
    if not _is_a5_target(_semantic):
        core.static_print("libdevice.rhadd for simd is unsupported for now.")
        core.static_assert(False)
    return core.extern_elementwise(
        "", "", [arg0, arg1], {
            (core.dtype("int32"), core.dtype("int32")): ("__hmf_rhadd_i32", core.dtype("int32")),
            (core.dtype("uint32"), core.dtype("uint32")): ("__hmf_urhadd_u32", core.dtype("uint32")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def fdim(arg0, arg1, _semantic=None):
    """
    Computes the positive difference between x and y. When x > y, returns x - y; otherwise returns 0.

    :param arg0: ``x``. Supported dtype(s): ``float32``.
    :type arg0: scalar or tl.tensor
    :param arg1: ``y``. Supported dtype(s): ``float32``.
    :type arg1: scalar or tl.tensor
    :return: The positive difference between x and y.
    :rtype: ``float32``
    """
    if not _is_a5_target(_semantic):
        core.static_print("libdevice.fdim for simd is unsupported for now.")
        core.static_assert(False)
    return core.extern_elementwise("", "", [arg0, arg1], {
        (core.dtype("fp32"), core.dtype("fp32")): ("__hmf_fdim_fp32", core.dtype("fp32")),
    }, is_pure=True, _semantic=_semantic)


@core.extern
def exp10(arg0, _semantic=None):
    """
    Base-10 exponential function, computes 10 raised to the power of x.

    :param arg0: ``x``. Supported dtype(s): ``float32``.
    :type arg0: scalar or tl.tensor
    :return: The result of 10 raised to the power of x.
    :rtype: ``float32``
    """
    if not _is_a5_target(_semantic):
        core.static_print("libdevice.exp10 for simd is unsupported for now.")
        core.static_assert(False)
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("fp32"), ): ("__hmf_exp10_fp32", core.dtype("fp32")),
    }, is_pure=True, _semantic=_semantic)


@core.extern
def add_rn(arg0, arg1, _semantic=None):
    """
    Floating-point addition with round-to-nearest-even rounding mode.

    :param arg0: ``x``. Supported dtype(s): ``float32``.
    :type arg0: scalar or tl.tensor
    :param arg1: ``y``. Supported dtype(s): ``float32``.
    :type arg1: scalar or tl.tensor
    :return: The addition result rounded to the nearest even number.
    :rtype: ``float32``
    """
    if not _is_a5_target(_semantic):
        core.static_print("libdevice.add_rn for simd is unsupported for now.")
        core.static_assert(False)
    return core.extern_elementwise("", "", [arg0, arg1], {
        (core.dtype("fp32"), core.dtype("fp32")): ("__hmf_add_rn_fp32", core.dtype("fp32")),
    }, is_pure=True, _semantic=_semantic)


@core.extern
def add_rz(arg0, arg1, _semantic=None):
    """
    Floating-point addition with round-toward-zero rounding mode.

    :param arg0: ``x``. Supported dtype(s): ``float32``.
    :type arg0: scalar or tl.tensor
    :param arg1: ``y``. Supported dtype(s): ``float32``.
    :type arg1: scalar or tl.tensor
    :return: The addition result rounded toward zero.
    :rtype: ``float32``
    """
    if not _is_a5_target(_semantic):
        core.static_print("libdevice.add_rz for simd is unsupported for now.")
        core.static_assert(False)
    return core.extern_elementwise("", "", [arg0, arg1], {
        (core.dtype("fp32"), core.dtype("fp32")): ("__hmf_add_rz_fp32", core.dtype("fp32")),
    }, is_pure=True, _semantic=_semantic)


@core.extern
def add_rd(arg0, arg1, _semantic=None):
    """
    Floating-point addition with round-down (toward negative infinity) rounding mode.

    :param arg0: ``x``. Supported dtype(s): ``float32``.
    :type arg0: scalar or tl.tensor
    :param arg1: ``y``. Supported dtype(s): ``float32``.
    :type arg1: scalar or tl.tensor
    :return: The addition result rounded down.
    :rtype: ``float32``
    """
    if not _is_a5_target(_semantic):
        core.static_print("libdevice.add_rd for simd is unsupported for now.")
        core.static_assert(False)
    return core.extern_elementwise("", "", [arg0, arg1], {
        (core.dtype("fp32"), core.dtype("fp32")): ("__hmf_add_rd_fp32", core.dtype("fp32")),
    }, is_pure=True, _semantic=_semantic)


@core.extern
def add_ru(arg0, arg1, _semantic=None):
    """
    Floating-point addition with round-up (toward positive infinity) rounding mode.

    :param arg0: ``x``. Supported dtype(s): ``float32``.
    :type arg0: scalar or tl.tensor
    :param arg1: ``y``. Supported dtype(s): ``float32``.
    :type arg1: scalar or tl.tensor
    :return: The addition result rounded up.
    :rtype: ``float32``
    """
    if not _is_a5_target(_semantic):
        core.static_print("libdevice.add_ru for simd is unsupported for now.")
        core.static_assert(False)
    return core.extern_elementwise("", "", [arg0, arg1], {
        (core.dtype("fp32"), core.dtype("fp32")): ("__hmf_add_ru_fp32", core.dtype("fp32")),
    }, is_pure=True, _semantic=_semantic)


@core.extern
def sub_rn(arg0, arg1, _semantic=None):
    """
    Floating-point subtraction with round-to-nearest-even rounding mode.

    :param arg0: ``x``. Supported dtype(s): ``float32``.
    :type arg0: scalar or tl.tensor
    :param arg1: ``y``. Supported dtype(s): ``float32``.
    :type arg1: scalar or tl.tensor
    :return: The subtraction result rounded to the nearest even number.
    :rtype: ``float32``
    """
    if not _is_a5_target(_semantic):
        core.static_print("libdevice.sub_rn for simd is unsupported for now.")
        core.static_assert(False)
    return core.extern_elementwise("", "", [arg0, arg1], {
        (core.dtype("fp32"), core.dtype("fp32")): ("__hmf_sub_rn_fp32", core.dtype("fp32")),
    }, is_pure=True, _semantic=_semantic)


@core.extern
def sub_rz(arg0, arg1, _semantic=None):
    """
    Floating-point subtraction with round-toward-zero rounding mode.

    :param arg0: ``x``. Supported dtype(s): ``float32``.
    :type arg0: scalar or tl.tensor
    :param arg1: ``y``. Supported dtype(s): ``float32``.
    :type arg1: scalar or tl.tensor
    :return: The subtraction result rounded toward zero.
    :rtype: ``float32``
    """
    if not _is_a5_target(_semantic):
        core.static_print("libdevice.sub_rz for simd is unsupported for now.")
        core.static_assert(False)
    return core.extern_elementwise("", "", [arg0, arg1], {
        (core.dtype("fp32"), core.dtype("fp32")): ("__hmf_sub_rz_fp32", core.dtype("fp32")),
    }, is_pure=True, _semantic=_semantic)


@core.extern
def sub_rd(arg0, arg1, _semantic=None):
    """
    Floating-point subtraction with round-down rounding mode.

    :param arg0: ``x``. Supported dtype(s): ``float32``.
    :type arg0: scalar or tl.tensor
    :param arg1: ``y``. Supported dtype(s): ``float32``.
    :type arg1: scalar or tl.tensor
    :return: The subtraction result rounded down.
    :rtype: ``float32``
    """
    if not _is_a5_target(_semantic):
        core.static_print("libdevice.sub_rd for simd is unsupported for now.")
        core.static_assert(False)
    return core.extern_elementwise("", "", [arg0, arg1], {
        (core.dtype("fp32"), core.dtype("fp32")): ("__hmf_sub_rd_fp32", core.dtype("fp32")),
    }, is_pure=True, _semantic=_semantic)


@core.extern
def sub_ru(arg0, arg1, _semantic=None):
    """
    Floating-point subtraction with round-up rounding mode.

    :param arg0: ``x``. Supported dtype(s): ``float32``.
    :type arg0: scalar or tl.tensor
    :param arg1: ``y``. Supported dtype(s): ``float32``.
    :type arg1: scalar or tl.tensor
    :return: The subtraction result rounded up.
    :rtype: ``float32``
    """
    if not _is_a5_target(_semantic):
        core.static_print("libdevice.sub_ru for simd is unsupported for now.")
        core.static_assert(False)
    return core.extern_elementwise("", "", [arg0, arg1], {
        (core.dtype("fp32"), core.dtype("fp32")): ("__hmf_sub_ru_fp32", core.dtype("fp32")),
    }, is_pure=True, _semantic=_semantic)


@core.extern
def mul_rn(arg0, arg1, _semantic=None):
    """
    Floating-point multiplication with round-to-nearest-even rounding mode.

    :param arg0: ``x``. Supported dtype(s): ``float32``.
    :type arg0: scalar or tl.tensor
    :param arg1: ``y``. Supported dtype(s): ``float32``.
    :type arg1: scalar or tl.tensor
    :return: The floating-point multiplication result.
    :rtype: ``float32``
    """
    if not _is_a5_target(_semantic):
        core.static_print("libdevice.mul_rn for simd is unsupported for now.")
        core.static_assert(False)
    return core.extern_elementwise("", "", [arg0, arg1], {
        (core.dtype("fp32"), core.dtype("fp32")): ("__hmf_mul_rn_fp32", core.dtype("fp32")),
    }, is_pure=True, _semantic=_semantic)


@core.extern
def mul_rz(arg0, arg1, _semantic=None):
    """
    Floating-point multiplication with round-toward-zero rounding mode.

    :param arg0: ``x``. Supported dtype(s): ``float32``.
    :type arg0: scalar or tl.tensor
    :param arg1: ``y``. Supported dtype(s): ``float32``.
    :type arg1: scalar or tl.tensor
    :return: The floating-point multiplication result.
    :rtype: ``float32``
    """
    if not _is_a5_target(_semantic):
        core.static_print("libdevice.mul_rz for simd is unsupported for now.")
        core.static_assert(False)
    return core.extern_elementwise("", "", [arg0, arg1], {
        (core.dtype("fp32"), core.dtype("fp32")): ("__hmf_mul_rz_fp32", core.dtype("fp32")),
    }, is_pure=True, _semantic=_semantic)


@core.extern
def mul_ru(arg0, arg1, _semantic=None):
    """
    Floating-point multiplication with round-up rounding mode.

    :param arg0: ``x``. Supported dtype(s): ``float32``.
    :type arg0: scalar or tl.tensor
    :param arg1: ``y``. Supported dtype(s): ``float32``.
    :type arg1: scalar or tl.tensor
    :return: The floating-point multiplication result.
    :rtype: ``float32``
    """
    if not _is_a5_target(_semantic):
        core.static_print("libdevice.mul_ru for simd is unsupported for now.")
        core.static_assert(False)
    return core.extern_elementwise("", "", [arg0, arg1], {
        (core.dtype("fp32"), core.dtype("fp32")): ("__hmf_mul_ru_fp32", core.dtype("fp32")),
    }, is_pure=True, _semantic=_semantic)


@core.extern
def mul_rd(arg0, arg1, _semantic=None):
    """
    Floating-point multiplication with round-down rounding mode.

    :param arg0: ``x``. Supported dtype(s): ``float32``.
    :type arg0: scalar or tl.tensor
    :param arg1: ``y``. Supported dtype(s): ``float32``.
    :type arg1: scalar or tl.tensor
    :return: The floating-point multiplication result.
    :rtype: ``float32``
    """
    if not _is_a5_target(_semantic):
        core.static_print("libdevice.mul_rd for simd is unsupported for now.")
        core.static_assert(False)
    return core.extern_elementwise("", "", [arg0, arg1], {
        (core.dtype("fp32"), core.dtype("fp32")): ("__hmf_mul_rd_fp32", core.dtype("fp32")),
    }, is_pure=True, _semantic=_semantic)


@core.extern
def div_rd(arg0, arg1, _semantic=None):
    """
    Floating-point division with round-down (toward negative infinity) rounding mode.

    :param arg0: ``x``. Supported dtype(s): ``float32``.
    :type arg0: scalar or tl.tensor
    :param arg1: ``y``. Supported dtype(s): ``float32``.
    :type arg1: scalar or tl.tensor
    :return: The division result.
    :rtype: ``float32``
    """
    if not _is_a5_target(_semantic):
        core.static_print("libdevice.div_rd for simd is unsupported for now.")
        core.static_assert(False)
    return core.extern_elementwise("", "", [arg0, arg1], {
        (core.dtype("fp32"), core.dtype("fp32")): ("__hmf_div_rd_fp32", core.dtype("fp32")),
    }, is_pure=True, _semantic=_semantic)


@core.extern
def div_ru(arg0, arg1, _semantic=None):
    """
    Floating-point division with round-up (toward positive infinity) rounding mode.

    :param arg0: ``x``. Supported dtype(s): ``float32``.
    :type arg0: scalar or tl.tensor
    :param arg1: ``y``. Supported dtype(s): ``float32``.
    :type arg1: scalar or tl.tensor
    :return: The division result.
    :rtype: ``float32``
    """
    if not _is_a5_target(_semantic):
        core.static_print("libdevice.div_ru for simd is unsupported for now.")
        core.static_assert(False)
    return core.extern_elementwise("", "", [arg0, arg1], {
        (core.dtype("fp32"), core.dtype("fp32")): ("__hmf_div_ru_fp32", core.dtype("fp32")),
    }, is_pure=True, _semantic=_semantic)


@core.extern
def div_rz(arg0, arg1, _semantic=None):
    """
    Floating-point division with round-toward-zero rounding mode.

    :param arg0: ``x``. Supported dtype(s): ``float32``.
    :type arg0: scalar or tl.tensor
    :param arg1: ``y``. Supported dtype(s): ``float32``.
    :type arg1: scalar or tl.tensor
    :return: The division result.
    :rtype: ``float32``
    """
    if not _is_a5_target(_semantic):
        arg0 = _semantic.to_tensor(arg0)
        arg1 = _semantic.to_tensor(arg1)
        ret = _semantic.fdiv(arg0, arg1, False)
        return ret
    return core.extern_elementwise("", "", [arg0, arg1], {
        (core.dtype("fp32"), core.dtype("fp32")): ("__hmf_div_rz_fp32", core.dtype("fp32")),
    }, is_pure=True, _semantic=_semantic)


@core.extern
def rcp_rn(arg0, _semantic=None):
    """
    Floating-point reciprocal with round-to-nearest-even rounding mode.

    :param arg0: ``x``. Supported dtype(s): ``float32``.
    :type arg0: scalar or tl.tensor
    :return: 1 / x.
    :rtype: ``float32``
    """
    if not _is_a5_target(_semantic):
        core.static_print("libdevice.rcp_rn for simd is unsupported for now.")
        core.static_assert(False)
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("fp32"), ): ("__hmf_rcp_rn_fp32", core.dtype("fp32")),
    }, is_pure=True, _semantic=_semantic)


@core.extern
def rcp_rz(arg0, _semantic=None):
    """
    Floating-point reciprocal with round-toward-zero rounding mode.

    :param arg0: ``x``. Supported dtype(s): ``float32``.
    :type arg0: scalar or tl.tensor
    :return: 1 / x.
    :rtype: ``float32``
    """
    if not _is_a5_target(_semantic):
        core.static_print("libdevice.rcp_rz for simd is unsupported for now.")
        core.static_assert(False)
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("fp32"), ): ("__hmf_rcp_rz_fp32", core.dtype("fp32")),
    }, is_pure=True, _semantic=_semantic)


@core.extern
def rcp_rd(arg0, _semantic=None):
    """
    Floating-point reciprocal with round-down rounding mode.

    :param arg0: ``x``. Supported dtype(s): ``float32``.
    :type arg0: scalar or tl.tensor
    :return: 1 / x.
    :rtype: ``float32``
    """
    if not _is_a5_target(_semantic):
        core.static_print("libdevice.rcp_rd for simd is unsupported for now.")
        core.static_assert(False)
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("fp32"), ): ("__hmf_rcp_rd_fp32", core.dtype("fp32")),
    }, is_pure=True, _semantic=_semantic)


@core.extern
def rcp_ru(arg0, _semantic=None):
    """
    Floating-point reciprocal with round-up rounding mode.

    :param arg0: ``x``. Supported dtype(s): ``float32``.
    :type arg0: scalar or tl.tensor
    :return: 1 / x.
    :rtype: ``float32``
    """
    if not _is_a5_target(_semantic):
        core.static_print("libdevice.rcp_ru for simd is unsupported for now.")
        core.static_assert(False)
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("fp32"), ): ("__hmf_rcp_ru_fp32", core.dtype("fp32")),
    }, is_pure=True, _semantic=_semantic)


@core.extern
@math._check_dtype(dtypes=["fp32"])
def sqrt_rn(arg0, _semantic=None):
    """
    Computes the square root of x with round-to-nearest-even rounding mode.

    :param arg0: ``x``. Supported dtype(s): ``float32``.
    :type arg0: scalar or tl.tensor
    :return: The square root of x.
    :rtype: ``float32``
    """
    if _is_libdevice_simt_enabled(_semantic):
        return core.extern_elementwise("", "", [arg0], {
            (core.dtype("fp32"), ): ("__hmf_sqrt_rn_fp32", core.dtype("fp32")),
        }, is_pure=True, _semantic=_semantic)
    arg0 = _semantic.to_tensor(arg0)
    return core.tensor(_semantic.builder.create_precise_sqrt(arg0.handle), arg0.type)


@core.extern
def sqrt_rz(arg0, _semantic=None):
    """
    Computes the square root of x with round-toward-zero rounding mode.

    :param arg0: ``x``. Supported dtype(s): ``float32``.
    :type arg0: scalar or tl.tensor
    :return: The square root of x.
    :rtype: ``float32``
    """
    if not _is_a5_target(_semantic):
        core.static_print("libdevice.sqrt_rz for simd is unsupported for now.")
        core.static_assert(False)
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("fp32"), ): ("__hmf_sqrt_rz_fp32", core.dtype("fp32")),
    }, is_pure=True, _semantic=_semantic)


@core.extern
def sqrt_rd(arg0, _semantic=None):
    """
    Computes the square root of x with round-down rounding mode.

    :param arg0: ``x``. Supported dtype(s): ``float32``.
    :type arg0: scalar or tl.tensor
    :return: The square root of x.
    :rtype: ``float32``
    """
    if not _is_a5_target(_semantic):
        core.static_print("libdevice.sqrt_rd for simd is unsupported for now.")
        core.static_assert(False)
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("fp32"), ): ("__hmf_sqrt_rd_fp32", core.dtype("fp32")),
    }, is_pure=True, _semantic=_semantic)


@core.extern
def sqrt_ru(arg0, _semantic=None):
    """
    Computes the square root of x with round-up rounding mode.

    :param arg0: ``x``. Supported dtype(s): ``float32``.
    :type arg0: scalar or tl.tensor
    :return: The square root of x.
    :rtype: ``float32``
    """
    if not _is_a5_target(_semantic):
        core.static_print("libdevice.sqrt_ru for simd is unsupported for now.")
        core.static_assert(False)
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("fp32"), ): ("__hmf_sqrt_ru_fp32", core.dtype("fp32")),
    }, is_pure=True, _semantic=_semantic)


@core.extern
def rsqrt_rn(arg0, _semantic=None):
    """
    Computes the reciprocal square root of x using round-to-nearest-even rounding mode.

    :param arg0: ``x``. Supported dtype(s): ``float32``.
    :type arg0: scalar or tl.tensor
    :return: The reciprocal square root of x.
    :rtype: ``float32``
    """
    if not _is_a5_target(_semantic):
        core.static_print("libdevice.rsqrt_rn for simd is unsupported for now.")
        core.static_assert(False)
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("fp32"), ): ("__hmf_rsqrt_rn_fp32", core.dtype("fp32")),
    }, is_pure=True, _semantic=_semantic)


@core.extern
def fma_rn(arg0, arg1, arg2, _semantic=None):
    """
    Fused multiply-add operation with round-to-nearest-even rounding mode.

    :param arg0: ``x``. Supported dtype(s): ``float32``.
    :type arg0: scalar or tl.tensor
    :param arg1: ``y``. Supported dtype(s): ``float32``.
    :type arg1: scalar or tl.tensor
    :param arg2: ``z``. Supported dtype(s): ``float32``.
    :type arg2: scalar or tl.tensor
    :return: The result of fused multiply-add.
    :rtype: ``float32``
    """
    if not _is_a5_target(_semantic):
        core.static_print("libdevice.fma_rn for simd is unsupported for now.")
        core.static_assert(False)
    return core.extern_elementwise("", "", [arg0, arg1, arg2], {
        (core.dtype("fp32"), core.dtype("fp32"), core.dtype("fp32")): ("__hmf_fma_rn_fp32", core.dtype("fp32")),
    }, is_pure=True, _semantic=_semantic)


@core.extern
def fma_rz(arg0, arg1, arg2, _semantic=None):
    """
    Fused multiply-add operation with round-toward-zero rounding mode.

    :param arg0: ``x``. Supported dtype(s): ``float32``.
    :type arg0: scalar or tl.tensor
    :param arg1: ``y``. Supported dtype(s): ``float32``.
    :type arg1: scalar or tl.tensor
    :param arg2: ``z``. Supported dtype(s): ``float32``.
    :type arg2: scalar or tl.tensor
    :return: The result of fused multiply-add.
    :rtype: ``float32``
    """
    if not _is_a5_target(_semantic):
        core.static_print("libdevice.fma_rz for simd is unsupported for now.")
        core.static_assert(False)
    return core.extern_elementwise("", "", [arg0, arg1, arg2], {
        (core.dtype("fp32"), core.dtype("fp32"), core.dtype("fp32")): ("__hmf_fma_rz_fp32", core.dtype("fp32")),
    }, is_pure=True, _semantic=_semantic)


@core.extern
def fma_rd(arg0, arg1, arg2, _semantic=None):
    """
    Fused multiply-add operation with round-down rounding mode.

    :param arg0: ``x``. Supported dtype(s): ``float32``.
    :type arg0: scalar or tl.tensor
    :param arg1: ``y``. Supported dtype(s): ``float32``.
    :type arg1: scalar or tl.tensor
    :param arg2: ``z``. Supported dtype(s): ``float32``.
    :type arg2: scalar or tl.tensor
    :return: The result of fused multiply-add.
    :rtype: ``float32``
    """
    if not _is_a5_target(_semantic):
        core.static_print("libdevice.fma_rd for simd is unsupported for now.")
        core.static_assert(False)
    return core.extern_elementwise("", "", [arg0, arg1, arg2], {
        (core.dtype("fp32"), core.dtype("fp32"), core.dtype("fp32")): ("__hmf_fma_rd_fp32", core.dtype("fp32")),
    }, is_pure=True, _semantic=_semantic)


@core.extern
def fma_ru(arg0, arg1, arg2, _semantic=None):
    """
    Fused multiply-add operation with round-up rounding mode.

    :param arg0: ``x``. Supported dtype(s): ``float32``.
    :type arg0: scalar or tl.tensor
    :param arg1: ``y``. Supported dtype(s): ``float32``.
    :type arg1: scalar or tl.tensor
    :param arg2: ``z``. Supported dtype(s): ``float32``.
    :type arg2: scalar or tl.tensor
    :return: The result of fused multiply-add.
    :rtype: ``float32``
    """
    if not _is_a5_target(_semantic):
        core.static_print("libdevice.fma_ru for simd is unsupported for now.")
        core.static_assert(False)
    return core.extern_elementwise("", "", [arg0, arg1, arg2], {
        (core.dtype("fp32"), core.dtype("fp32"), core.dtype("fp32")): ("__hmf_fma_ru_fp32", core.dtype("fp32")),
    }, is_pure=True, _semantic=_semantic)


@core.builtin
def fast_dividef(arg0, arg1, _semantic=None):
    """
    Fast approximate division.

    :param arg0: ``x``. Supported dtype(s): ``float32``.
    :type arg0: scalar or tl.tensor
    :param arg1: ``y``. Supported dtype(s): ``float32``.
    :type arg1: scalar or tl.tensor
    :return: The result of fast approximate division.
    :rtype: ``float32``
    """
    if _is_libdevice_simt_enabled(_semantic):
        return core.extern_elementwise("", "", [arg0, arg1], {
            (core.dtype("fp32"), core.dtype("fp32")): ("__hmf_fast_divide_fp32", core.dtype("fp32")),
        }, is_pure=True, _semantic=_semantic)
    arg0 = _semantic.to_tensor(arg0)
    arg1 = _semantic.to_tensor(arg1)
    ret = _semantic.fdiv(arg0, arg1, False)
    return ret


@core.builtin
def fast_expf(arg0, _semantic=None):
    """
    Fast approximate exponential function.

    :param arg0: ``x``. Supported dtype(s): ``float32``.
    :type arg0: scalar or tl.tensor
    :return: The result of the fast approximate exponential function.
    :rtype: ``float32``
    """
    if _is_libdevice_simt_enabled(_semantic):
        return core.extern_elementwise("", "", [arg0], {
            (core.dtype("fp32"), ): ("__hmf_fast_exp_fp32", core.dtype("fp32")),
        }, is_pure=True, _semantic=_semantic)
    arg0 = _semantic.to_tensor(arg0)
    ret = core.tensor(_semantic.builder.create_exp(arg0.handle), arg0.type)
    return ret


@core.builtin
def fast_exp10f(arg0, _semantic=None):
    """
    Fast approximate base-10 exponential function.

    :param arg0: ``x``. Supported dtype(s): ``float32``.
    :type arg0: scalar or tl.tensor
    :return: The result of the fast approximate base-10 exponential function.
    :rtype: ``float32``
    """
    if not _is_a5_target(_semantic):
        core.static_print("libdevice.fast_exp10f for simd is unsupported for now.")
        core.static_assert(False)
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("fp32"), ): ("__hmf_fast_exp10_fp32", core.dtype("fp32")),
    }, is_pure=True, _semantic=_semantic)


@core.builtin
def fast_sinf(arg0, _semantic=None):
    """
    Fast approximate sine function.

    :param arg0: ``x``. Supported dtype(s): ``float32``.
    :type arg0: scalar or tl.tensor
    :return: The result of the fast approximate sine function.
    :rtype: ``float32``
    """
    if not _is_a5_target(_semantic):
        core.static_print("libdevice.fast_sinf for simd is unsupported for now.")
        core.static_assert(False)
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("fp32"), ): ("__hmf_fast_sin_fp32", core.dtype("fp32")),
    }, is_pure=True, _semantic=_semantic)


@core.builtin
def fast_cosf(arg0, _semantic=None):
    """
    Fast approximate cosine function.

    :param arg0: ``x``. Supported dtype(s): ``float32``.
    :type arg0: scalar or tl.tensor
    :return: The result of the fast approximate cosine function.
    :rtype: ``float32``
    """
    if not _is_a5_target(_semantic):
        core.static_print("libdevice.fast_cosf for simd is unsupported for now.")
        core.static_assert(False)
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("fp32"), ): ("__hmf_fast_cos_fp32", core.dtype("fp32")),
    }, is_pure=True, _semantic=_semantic)


@core.builtin
def fast_tanf(arg0, _semantic=None):
    """
    Fast approximate tangent function.

    :param arg0: ``x``. Supported dtype(s): ``float32``.
    :type arg0: scalar or tl.tensor
    :return: The result of the fast approximate tangent function.
    :rtype: ``float32``
    """
    if not _is_a5_target(_semantic):
        core.static_print("libdevice.fast_tanf for simd is unsupported for now.")
        core.static_assert(False)
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("fp32"), ): ("__hmf_fast_tan_fp32", core.dtype("fp32")),
    }, is_pure=True, _semantic=_semantic)


@core.builtin
def fast_tanhf(arg0, _semantic=None):
    """
    Computes the hyperbolic tangent of x using a fast approximation.

    :param arg0: ``x``. Supported dtype(s): ``float32``.
    :type arg0: scalar or tl.tensor
    :return: The fast approximate hyperbolic tangent of x.
    :rtype: ``float32``
    """
    if not _is_a5_target(_semantic):
        core.static_print("libdevice.fast_tanhf for simd is unsupported for now.")
        core.static_assert(False)
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("fp32"), ): ("__hmf_fast_tanh_fp32", core.dtype("fp32")),
    }, is_pure=True, _semantic=_semantic)


@core.builtin
def fast_log2f(arg0, _semantic=None):
    """
    Fast approximate base-2 logarithm function.

    :param arg0: ``x``. Supported dtype(s): ``float32``.
    :type arg0: scalar or tl.tensor
    :return: The result of the fast approximate base-2 logarithm function.
    :rtype: ``float32``
    """
    if not _is_a5_target(_semantic):
        core.static_print("libdevice.fast_log2f for simd is unsupported for now.")
        core.static_assert(False)
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("fp32"), ): ("__hmf_fast_log2_fp32", core.dtype("fp32")),
    }, is_pure=True, _semantic=_semantic)


@core.builtin
def fast_logf(arg0, _semantic=None):
    """
    Fast approximate natural logarithm function.

    :param arg0: ``x``. Supported dtype(s): ``float32``.
    :type arg0: scalar or tl.tensor
    :return: The result of the fast approximate natural logarithm function.
    :rtype: ``float32``
    """
    if not _is_a5_target(_semantic):
        core.static_print("libdevice.fast_logf for simd is unsupported for now.")
        core.static_assert(False)
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("fp32"), ): ("__hmf_fast_log_fp32", core.dtype("fp32")),
    }, is_pure=True, _semantic=_semantic)


@core.builtin
def fast_log10f(arg0, _semantic=None):
    """
    Fast approximate base-10 logarithm function.

    :param arg0: ``x``. Supported dtype(s): ``float32``.
    :type arg0: scalar or tl.tensor
    :return: The result of the fast approximate base-10 logarithm function.
    :rtype: ``float32``
    """
    if not _is_a5_target(_semantic):
        core.static_print("libdevice.fast_log10f for simd is unsupported for now.")
        core.static_assert(False)
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("fp32"), ): ("__hmf_fast_log10_fp32", core.dtype("fp32")),
    }, is_pure=True, _semantic=_semantic)


@core.builtin
def fast_powf(arg0, arg1, _semantic=None):
    """
    Fast approximate power function.

    :param arg0: ``x``. Supported dtype(s): ``float32``.
    :type arg0: scalar or tl.tensor
    :param arg1: ``y``. Supported dtype(s): ``float32``.
    :type arg1: scalar or tl.tensor
    :return: The result of fast approximate power function.
    :rtype: ``float32``
    """
    if not _is_a5_target(_semantic):
        core.static_print("libdevice.fast_powf for simd is unsupported for now.")
        core.static_assert(False)
    return core.extern_elementwise("", "", [arg0, arg1], {
        (core.dtype("fp32"), core.dtype("fp32")): ("__hmf_fast_pow_fp32", core.dtype("fp32")),
    }, is_pure=True, _semantic=_semantic)


@core.extern
def fmod(arg0, arg1, _semantic=None):
    """
    Floating-point modulo, computes the remainder of x / y, with the same sign as x.

    :param arg0: ``x``. Supported dtype(s): ``float32``.
    :type arg0: scalar or tl.tensor
    :param arg1: ``y``. Supported dtype(s): ``float32``.
    :type arg1: scalar or tl.tensor
    :return: The floating-point modulo result.
    :rtype: ``float32``
    """
    if not _is_a5_target(_semantic):
        arg0 = _semantic.to_tensor(arg0)
        arg1 = _semantic.to_tensor(arg1)
        ret = _semantic.mod(arg0, arg1)
        return ret
    return core.extern_elementwise("", "", [arg0, arg1], {
        (core.dtype("fp32"), core.dtype("fp32")): ("__hmf_fmod_fp32", core.dtype("fp32")),
    }, is_pure=True, _semantic=_semantic)


@core.extern
def remainder(arg0, arg1, _semantic=None):
    """
    Computes the remainder of x divided by y, where r = x - ny, and n is the nearest integer to x / y.

    :param arg0: ``x``. Supported dtype(s): ``float32``.
    :type arg0: scalar or tl.tensor
    :param arg1: ``y``. Supported dtype(s): ``float32``.
    :type arg1: scalar or tl.tensor
    :return: The remainder of x divided by y.
    :rtype: ``float32``
    """
    if not _is_a5_target(_semantic):
        core.static_print("libdevice.remainder for simd is unsupported for now.")
        core.static_assert(False)
    return core.extern_elementwise("", "", [arg0, arg1], {
        (core.dtype("fp32"), core.dtype("fp32")): ("__hmf_remainder_fp32", core.dtype("fp32")),
    }, is_pure=True, _semantic=_semantic)


@core.extern
def float_as_int(arg0, _semantic=None):
    """
    Reinterprets the bit pattern of a floating-point number as a 32-bit integer. No numeric conversion is performed.

    :param arg0: ``x``. Supported dtype(s): ``float32``.
    :type arg0: scalar or tl.tensor
    :return: The bit pattern of the floating-point number reinterpreted as a 32-bit integer.
    :rtype: ``int32``
    """
    if not _is_a5_target(_semantic):
        core.static_print("libdevice.float_as_int for simd is unsupported for now.")
        core.static_assert(False)
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("fp32"), ): ("__hmf_float_as_int_fp32", core.dtype("int32")),
    }, is_pure=True, _semantic=_semantic)


@core.extern
def int_as_float(arg0, _semantic=None):
    """
    Reinterprets the bit pattern of a 32-bit integer as a floating-point number. No numeric conversion is performed.

    :param arg0: ``x``. Supported dtype(s): ``int32``.
    :type arg0: scalar or tl.tensor
    :return: The bit pattern of the 32-bit integer reinterpreted as a floating-point number.
    :rtype: ``float32``
    """
    if not _is_a5_target(_semantic):
        core.static_print("libdevice.int_as_float for simd is unsupported for now.")
        core.static_assert(False)
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("int32"), ): ("__hmf_int_as_float_fp32", core.dtype("fp32")),
    }, is_pure=True, _semantic=_semantic)


@core.extern
def float_as_uint(arg0, _semantic=None):
    """
    Reinterprets the bit pattern of a floating-point number as a 32-bit unsigned integer. No numeric conversion is
    performed.

    :param arg0: ``x``. Supported dtype(s): ``float32``.
    :type arg0: scalar or tl.tensor
    :return: The bit pattern of the floating-point number reinterpreted as a 32-bit unsigned integer.
    :rtype: ``uint32``
    """
    if not _is_a5_target(_semantic):
        core.static_print("libdevice.float_as_uint for simd is unsupported for now.")
        core.static_assert(False)
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("fp32"), ): ("__hmf_float_as_uint_fp32", core.dtype("uint32")),
    }, is_pure=True, _semantic=_semantic)


@core.extern
def uint_as_float(arg0, _semantic=None):
    """
    Reinterprets the bit pattern of a 32-bit unsigned integer as a floating-point number. No numeric conversion is
    performed.

    :param arg0: ``x``. Supported dtype(s): ``uint32``.
    :type arg0: scalar or tl.tensor
    :return: The bit pattern of the 32-bit unsigned integer reinterpreted as a floating-point number.
    :rtype: ``float32``
    """
    if not _is_a5_target(_semantic):
        core.static_print("libdevice.uint_as_float for simd is unsupported for now.")
        core.static_assert(False)
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("uint32"), ): ("__hmf_uint_as_float_fp32", core.dtype("fp32")),
    }, is_pure=True, _semantic=_semantic)


@core.extern
def float2int_rn(arg0, _semantic=None):
    """
    Converts a floating-point number to a 32-bit integer with round-to-nearest-even mode.

    :param arg0: ``x``. Supported dtype(s): ``float32``.
    :type arg0: scalar or tl.tensor
    :return: The converted 32-bit integer.
    :rtype: ``int32``
    """
    if not _is_a5_target(_semantic):
        core.static_print("libdevice.float2int_rn for simd is unsupported for now.")
        core.static_assert(False)
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("fp32"), ): ("__hmf_float2int_rn_fp32", core.dtype("int32")),
    }, is_pure=True, _semantic=_semantic)


@core.extern
def float2int_rz(arg0, _semantic=None):
    """
    Converts a floating-point number to a 32-bit integer with round-toward-zero mode.

    :param arg0: ``x``. Supported dtype(s): ``float32``.
    :type arg0: scalar or tl.tensor
    :return: The converted 32-bit integer.
    :rtype: ``int32``
    """
    if not _is_a5_target(_semantic):
        core.static_print("libdevice.float2int_rz for simd is unsupported for now.")
        core.static_assert(False)
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("fp32"), ): ("__hmf_float2int_rz_fp32", core.dtype("int32")),
    }, is_pure=True, _semantic=_semantic)


@core.extern
def float2int_rd(arg0, _semantic=None):
    """
    Converts a floating-point number to a 32-bit integer with round-down mode.

    :param arg0: ``x``. Supported dtype(s): ``float32``.
    :type arg0: scalar or tl.tensor
    :return: The converted 32-bit integer.
    :rtype: ``int32``
    """
    if not _is_a5_target(_semantic):
        core.static_print("libdevice.float2int_rd for simd is unsupported for now.")
        core.static_assert(False)
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("fp32"), ): ("__hmf_float2int_rd_fp32", core.dtype("int32")),
    }, is_pure=True, _semantic=_semantic)


@core.extern
def float2int_ru(arg0, _semantic=None):
    """
    Converts a floating-point number to a 32-bit integer with round-up mode.

    :param arg0: ``x``. Supported dtype(s): ``float32``.
    :type arg0: scalar or tl.tensor
    :return: The converted 32-bit integer.
    :rtype: ``int32``
    """
    if not _is_a5_target(_semantic):
        core.static_print("libdevice.float2int_ru for simd is unsupported for now.")
        core.static_assert(False)
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("fp32"), ): ("__hmf_float2int_ru_fp32", core.dtype("int32")),
    }, is_pure=True, _semantic=_semantic)


@core.extern
def int2float_rn(arg0, _semantic=None):
    """
    Converts a 32-bit integer to a floating-point number with round-to-nearest-even mode.

    :param arg0: ``x``. Supported dtype(s): ``int32``.
    :type arg0: scalar or tl.tensor
    :return: The converted floating-point number.
    :rtype: ``float32``
    """
    if not _is_a5_target(_semantic):
        core.static_print("libdevice.int2float_rn for simd is unsupported for now.")
        core.static_assert(False)
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("int32"), ): ("__hmf_int2float_rn_fp32", core.dtype("fp32")),
    }, is_pure=True, _semantic=_semantic)


@core.extern
def int2float_rz(arg0, _semantic=None):
    """
    Converts a 32-bit integer to a floating-point number with round-toward-zero mode.

    :param arg0: ``x``. Supported dtype(s): ``int32``.
    :type arg0: scalar or tl.tensor
    :return: The converted floating-point number.
    :rtype: ``float32``
    """
    if not _is_a5_target(_semantic):
        core.static_print("libdevice.int2float_rz for simd is unsupported for now.")
        core.static_assert(False)
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("int32"), ): ("__hmf_int2float_rz_fp32", core.dtype("fp32")),
    }, is_pure=True, _semantic=_semantic)


@core.extern
def int2float_rd(arg0, _semantic=None):
    """
    Converts a 32-bit integer to a floating-point number with round-down mode.

    :param arg0: ``x``. Supported dtype(s): ``int32``.
    :type arg0: scalar or tl.tensor
    :return: The converted floating-point number.
    :rtype: ``float32``
    """
    if not _is_a5_target(_semantic):
        core.static_print("libdevice.int2float_rd for simd is unsupported for now.")
        core.static_assert(False)
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("int32"), ): ("__hmf_int2float_rd_fp32", core.dtype("fp32")),
    }, is_pure=True, _semantic=_semantic)


@core.extern
def int2float_ru(arg0, _semantic=None):
    """
    Converts a 32-bit integer to a floating-point number with round-up mode.

    :param arg0: ``x``. Supported dtype(s): ``int32``.
    :type arg0: scalar or tl.tensor
    :return: The converted floating-point number.
    :rtype: ``float32``
    """
    if not _is_a5_target(_semantic):
        core.static_print("libdevice.int2float_ru for simd is unsupported for now.")
        core.static_assert(False)
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("int32"), ): ("__hmf_int2float_ru_fp32", core.dtype("fp32")),
    }, is_pure=True, _semantic=_semantic)


@core.extern
def float2uint_rn(arg0, _semantic=None):
    """
    Converts a floating-point number to a 32-bit unsigned integer with round-to-nearest-even mode.

    :param arg0: ``x``. Supported dtype(s): ``float32``.
    :type arg0: scalar or tl.tensor
    :return: The converted 32-bit unsigned integer.
    :rtype: ``uint32``
    """
    if not _is_a5_target(_semantic):
        core.static_print("libdevice.float2uint_rn for simd is unsupported for now.")
        core.static_assert(False)
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("fp32"), ): ("__hmf_float2uint_rn_fp32", core.dtype("uint32")),
    }, is_pure=True, _semantic=_semantic)


@core.extern
def float2uint_rz(arg0, _semantic=None):
    """
    Converts a floating-point number to a 32-bit unsigned integer with round-toward-zero mode.

    :param arg0: ``x``. Supported dtype(s): ``float32``.
    :type arg0: scalar or tl.tensor
    :return: The converted 32-bit unsigned integer.
    :rtype: ``uint32``
    """
    if not _is_a5_target(_semantic):
        core.static_print("libdevice.float2uint_rz for simd is unsupported for now.")
        core.static_assert(False)
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("fp32"), ): ("__hmf_float2uint_rz_fp32", core.dtype("uint32")),
    }, is_pure=True, _semantic=_semantic)


@core.extern
def float2uint_rd(arg0, _semantic=None):
    """
    Converts a floating-point number to a 32-bit unsigned integer with round-down mode.

    :param arg0: ``x``. Supported dtype(s): ``float32``.
    :type arg0: scalar or tl.tensor
    :return: The converted 32-bit unsigned integer.
    :rtype: ``uint32``
    """
    if not _is_a5_target(_semantic):
        core.static_print("libdevice.float2uint_rd for simd is unsupported for now.")
        core.static_assert(False)
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("fp32"), ): ("__hmf_float2uint_rd_fp32", core.dtype("uint32")),
    }, is_pure=True, _semantic=_semantic)


@core.extern
def float2uint_ru(arg0, _semantic=None):
    """
    Converts a floating-point number to a 32-bit unsigned integer with round-up mode.

    :param arg0: ``x``. Supported dtype(s): ``float32``.
    :type arg0: scalar or tl.tensor
    :return: The converted 32-bit unsigned integer.
    :rtype: ``uint32``
    """
    if not _is_a5_target(_semantic):
        core.static_print("libdevice.float2uint_ru for simd is unsupported for now.")
        core.static_assert(False)
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("fp32"), ): ("__hmf_float2uint_ru_fp32", core.dtype("uint32")),
    }, is_pure=True, _semantic=_semantic)


@core.extern
def uint2float_rn(arg0, _semantic=None):
    """
    Converts a 32-bit unsigned integer to a floating-point number with round-to-nearest-even mode.

    :param arg0: ``x``. Supported dtype(s): ``uint32``.
    :type arg0: scalar or tl.tensor
    :return: The converted floating-point number.
    :rtype: ``float32``
    """
    if not _is_a5_target(_semantic):
        core.static_print("libdevice.uint2float_rn for simd is unsupported for now.")
        core.static_assert(False)
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("uint32"), ): ("__hmf_uint2float_rn_fp32", core.dtype("fp32")),
    }, is_pure=True, _semantic=_semantic)


@core.extern
def uint2float_rz(arg0, _semantic=None):
    """
    Converts a 32-bit unsigned integer to a floating-point number with round-toward-zero mode.

    :param arg0: ``x``. Supported dtype(s): ``uint32``.
    :type arg0: scalar or tl.tensor
    :return: The converted floating-point number.
    :rtype: ``float32``
    """
    if not _is_a5_target(_semantic):
        core.static_print("libdevice.uint2float_rz for simd is unsupported for now.")
        core.static_assert(False)
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("uint32"), ): ("__hmf_uint2float_rz_fp32", core.dtype("fp32")),
    }, is_pure=True, _semantic=_semantic)


@core.extern
def uint2float_rd(arg0, _semantic=None):
    """
    Converts a 32-bit unsigned integer to a floating-point number with round-down mode.

    :param arg0: ``x``. Supported dtype(s): ``uint32``.
    :type arg0: scalar or tl.tensor
    :return: The converted floating-point number.
    :rtype: ``float32``
    """
    if not _is_a5_target(_semantic):
        core.static_print("libdevice.uint2float_rd for simd is unsupported for now.")
        core.static_assert(False)
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("uint32"), ): ("__hmf_uint2float_rd_fp32", core.dtype("fp32")),
    }, is_pure=True, _semantic=_semantic)


@core.extern
def uint2float_ru(arg0, _semantic=None):
    """
    Converts a 32-bit unsigned integer to a floating-point number with round-up mode.

    :param arg0: ``x``. Supported dtype(s): ``uint32``.
    :type arg0: scalar or tl.tensor
    :return: The converted floating-point number.
    :rtype: ``float32``
    """
    if not _is_a5_target(_semantic):
        core.static_print("libdevice.uint2float_ru for simd is unsupported for now.")
        core.static_assert(False)
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("uint32"), ): ("__hmf_uint2float_ru_fp32", core.dtype("fp32")),
    }, is_pure=True, _semantic=_semantic)


@core.extern
def float2ll_rn(arg0, _semantic=None):
    """
    Converts a floating-point number to a 64-bit integer with round-to-nearest-even mode.

    :param arg0: ``x``. Supported dtype(s): ``float32``.
    :type arg0: scalar or tl.tensor
    :return: The converted 64-bit integer.
    :rtype: ``int64``
    """
    if not _is_a5_target(_semantic):
        core.static_print("libdevice.float2ll_rn for simd is unsupported for now.")
        core.static_assert(False)
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("fp32"), ): ("__hmf_float2ll_rn_fp32", core.dtype("int64")),
    }, is_pure=True, _semantic=_semantic)


@core.extern
def float2ll_rz(arg0, _semantic=None):
    """
    Converts a floating-point number to a 64-bit integer with round-toward-zero mode.

    :param arg0: ``x``. Supported dtype(s): ``float32``.
    :type arg0: scalar or tl.tensor
    :return: The converted 64-bit integer.
    :rtype: ``int64``
    """
    if not _is_a5_target(_semantic):
        core.static_print("libdevice.float2ll_rz for simd is unsupported for now.")
        core.static_assert(False)
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("fp32"), ): ("__hmf_float2ll_rz_fp32", core.dtype("int64")),
    }, is_pure=True, _semantic=_semantic)


@core.extern
def float2ll_rd(arg0, _semantic=None):
    """
    Converts a floating-point number to a 64-bit integer with round-down mode.

    :param arg0: ``x``. Supported dtype(s): ``float32``.
    :type arg0: scalar or tl.tensor
    :return: The converted 64-bit integer.
    :rtype: ``int64``
    """
    if not _is_a5_target(_semantic):
        core.static_print("libdevice.float2ll_rd for simd is unsupported for now.")
        core.static_assert(False)
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("fp32"), ): ("__hmf_float2ll_rd_fp32", core.dtype("int64")),
    }, is_pure=True, _semantic=_semantic)


@core.extern
def float2ll_ru(arg0, _semantic=None):
    """
    Converts a floating-point number to a 64-bit integer with round-up mode.

    :param arg0: ``x``. Supported dtype(s): ``float32``.
    :type arg0: scalar or tl.tensor
    :return: The converted 64-bit integer.
    :rtype: ``int64``
    """
    if not _is_a5_target(_semantic):
        core.static_print("libdevice.float2ll_ru for simd is unsupported for now.")
        core.static_assert(False)
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("fp32"), ): ("__hmf_float2ll_ru_fp32", core.dtype("int64")),
    }, is_pure=True, _semantic=_semantic)


@core.extern
def ll2float_rn(arg0, _semantic=None):
    """
    Converts a 64-bit integer to a floating-point number with round-to-nearest-even mode.

    :param arg0: ``x``. Supported dtype(s): ``int64``.
    :type arg0: scalar or tl.tensor
    :return: The converted floating-point number.
    :rtype: ``float32``
    """
    if not _is_a5_target(_semantic):
        core.static_print("libdevice.ll2float_rn for simd is unsupported for now.")
        core.static_assert(False)
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("int64"), ): ("__hmf_ll2float_rn_fp32", core.dtype("fp32")),
    }, is_pure=True, _semantic=_semantic)


@core.extern
def ll2float_rz(arg0, _semantic=None):
    """
    Converts a 64-bit integer to a floating-point number with round-toward-zero mode.

    :param arg0: ``x``. Supported dtype(s): ``int64``.
    :type arg0: scalar or tl.tensor
    :return: The converted floating-point number.
    :rtype: ``float32``
    """
    if not _is_a5_target(_semantic):
        core.static_print("libdevice.ll2float_rz for simd is unsupported for now.")
        core.static_assert(False)
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("int64"), ): ("__hmf_ll2float_rz_fp32", core.dtype("fp32")),
    }, is_pure=True, _semantic=_semantic)


@core.extern
def ll2float_rd(arg0, _semantic=None):
    """
    Converts a 64-bit integer to a floating-point number with round-down mode.

    :param arg0: ``x``. Supported dtype(s): ``int64``.
    :type arg0: scalar or tl.tensor
    :return: The converted floating-point number.
    :rtype: ``float32``
    """
    if not _is_a5_target(_semantic):
        core.static_print("libdevice.ll2float_rd for simd is unsupported for now.")
        core.static_assert(False)
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("int64"), ): ("__hmf_ll2float_rd_fp32", core.dtype("fp32")),
    }, is_pure=True, _semantic=_semantic)


@core.extern
def ll2float_ru(arg0, _semantic=None):
    """
    Converts a 64-bit integer to a floating-point number with round-up mode.

    :param arg0: ``x``. Supported dtype(s): ``int64``.
    :type arg0: scalar or tl.tensor
    :return: The converted floating-point number.
    :rtype: ``float32``
    """
    if not _is_a5_target(_semantic):
        core.static_print("libdevice.ll2float_ru for simd is unsupported for now.")
        core.static_assert(False)
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("int64"), ): ("__hmf_ll2float_ru_fp32", core.dtype("fp32")),
    }, is_pure=True, _semantic=_semantic)


@core.extern
def float2ull_rn(arg0, _semantic=None):
    """
    Converts a floating-point number to a 64-bit unsigned integer with round-to-nearest-even mode.

    :param arg0: ``x``. Supported dtype(s): ``float32``.
    :type arg0: scalar or tl.tensor
    :return: The converted 64-bit unsigned integer.
    :rtype: ``uint64``
    """
    if not _is_a5_target(_semantic):
        core.static_print("libdevice.float2ull_rn for simd is unsupported for now.")
        core.static_assert(False)
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("fp32"), ): ("__hmf_float2ull_rn_fp32", core.dtype("uint64")),
    }, is_pure=True, _semantic=_semantic)


@core.extern
def float2ull_rz(arg0, _semantic=None):
    """
    Converts a floating-point number to a 64-bit unsigned integer with round-toward-zero mode.

    :param arg0: ``x``. Supported dtype(s): ``float32``.
    :type arg0: scalar or tl.tensor
    :return: The converted 64-bit unsigned integer.
    :rtype: ``uint64``
    """
    if not _is_a5_target(_semantic):
        core.static_print("libdevice.float2ull_rz for simd is unsupported for now.")
        core.static_assert(False)
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("fp32"), ): ("__hmf_float2ull_rz_fp32", core.dtype("uint64")),
    }, is_pure=True, _semantic=_semantic)


@core.extern
def float2ull_rd(arg0, _semantic=None):
    """
    Converts a floating-point number to a 64-bit unsigned integer with round-down mode.

    :param arg0: ``x``. Supported dtype(s): ``float32``.
    :type arg0: scalar or tl.tensor
    :return: The converted 64-bit unsigned integer.
    :rtype: ``uint64``
    """
    if not _is_a5_target(_semantic):
        core.static_print("libdevice.float2ull_rd for simd is unsupported for now.")
        core.static_assert(False)
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("fp32"), ): ("__hmf_float2ull_rd_fp32", core.dtype("uint64")),
    }, is_pure=True, _semantic=_semantic)


@core.extern
def float2ull_ru(arg0, _semantic=None):
    """
    Converts a floating-point number to a 64-bit unsigned integer with round-up mode.

    :param arg0: ``x``. Supported dtype(s): ``float32``.
    :type arg0: scalar or tl.tensor
    :return: The converted 64-bit unsigned integer.
    :rtype: ``uint64``
    """
    if not _is_a5_target(_semantic):
        core.static_print("libdevice.float2ull_ru for simd is unsupported for now.")
        core.static_assert(False)
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("fp32"), ): ("__hmf_float2ull_ru_fp32", core.dtype("uint64")),
    }, is_pure=True, _semantic=_semantic)


@core.extern
def ull2float_rn(arg0, _semantic=None):
    """
    Converts a 64-bit unsigned integer to a floating-point number with round-to-nearest-even mode.

    :param arg0: ``x``. Supported dtype(s): ``uint64``.
    :type arg0: scalar or tl.tensor
    :return: The converted floating-point number.
    :rtype: ``float32``
    """
    if not _is_a5_target(_semantic):
        core.static_print("libdevice.ull2float_rn for simd is unsupported for now.")
        core.static_assert(False)
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("uint64"), ): ("__hmf_ull2float_rn_fp32", core.dtype("fp32")),
    }, is_pure=True, _semantic=_semantic)


@core.extern
def ull2float_rz(arg0, _semantic=None):
    """
    Converts a 64-bit unsigned integer to a floating-point number with round-toward-zero mode.

    :param arg0: ``x``. Supported dtype(s): ``uint64``.
    :type arg0: scalar or tl.tensor
    :return: The converted floating-point number.
    :rtype: ``float32``
    """
    if not _is_a5_target(_semantic):
        core.static_print("libdevice.ull2float_rz for simd is unsupported for now.")
        core.static_assert(False)
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("uint64"), ): ("__hmf_ull2float_rz_fp32", core.dtype("fp32")),
    }, is_pure=True, _semantic=_semantic)


@core.extern
def ull2float_rd(arg0, _semantic=None):
    """
    Converts a 64-bit unsigned integer to a floating-point number with round-down mode.

    :param arg0: ``x``. Supported dtype(s): ``uint64``.
    :type arg0: scalar or tl.tensor
    :return: The converted floating-point number.
    :rtype: ``float32``
    """
    if not _is_a5_target(_semantic):
        core.static_print("libdevice.ull2float_rd for simd is unsupported for now.")
        core.static_assert(False)
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("uint64"), ): ("__hmf_ull2float_rd_fp32", core.dtype("fp32")),
    }, is_pure=True, _semantic=_semantic)


@core.extern
def ull2float_ru(arg0, _semantic=None):
    """
    Converts a 64-bit unsigned integer to a floating-point number with round-up mode.

    :param arg0: ``x``. Supported dtype(s): ``uint64``.
    :type arg0: scalar or tl.tensor
    :return: The converted floating-point number.
    :rtype: ``float32``
    """
    if not _is_a5_target(_semantic):
        core.static_print("libdevice.ull2float_ru for simd is unsupported for now.")
        core.static_assert(False)
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("uint64"), ): ("__hmf_ull2float_ru_fp32", core.dtype("fp32")),
    }, is_pure=True, _semantic=_semantic)


@core.extern
def atan2(arg0, arg1, _semantic=None):
    """
    Two-argument inverse tangent function, computes the arctangent of x / y.

    :param arg0: ``x``. Supported dtype(s): ``float32``.
    :type arg0: scalar or tl.tensor
    :param arg1: ``y``. Supported dtype(s): ``float32``.
    :type arg1: scalar or tl.tensor
    :return: The arctangent of x / y, in the range [-π, π] radians.
    :rtype: ``float32``
    """
    if _is_libdevice_simt_enabled(_semantic):
        if arg0.dtype == core.dtype("bf16") or arg1.dtype == core.dtype("bf16"):
            core.static_print("extern libdevice.atan2 for dtype bf16 is unsupported for now.")
            core.static_assert(False)
        return core.extern_elementwise(
            "", "", [arg0, arg1], {
                (core.dtype("fp16"), core.dtype("fp16")): ("__hmf_atan2_fp16", core.dtype("fp16")),
                (core.dtype("fp32"), core.dtype("fp32")): ("__hmf_atan2_fp32", core.dtype("fp32")),
            }, is_pure=True, _semantic=_semantic)

    arg0 = _semantic.to_tensor(arg0)
    arg1 = _semantic.to_tensor(arg1)
    pi = 3.1415926536

    _is_int8_type_x: core.constexpr = arg1.dtype.is_int8()
    core.static_assert(not _is_int8_type_x, "Expected dtype fp16/fp32/bf16, but got int8 or int1", _semantic=_semantic)

    _is_int8_type_y: core.constexpr = arg0.dtype.is_int8()
    core.static_assert(not _is_int8_type_y, "Expected dtype fp16/fp32/bf16, but got int8 or int1", _semantic=_semantic)

    _is_floating_type_x: core.constexpr = arg1.dtype.is_floating()
    core.static_assert(_is_floating_type_x == True,
                       f"Expected dtype fp16/fp32/bf16, but got {core.constexpr(arg1.dtype)}", _semantic=_semantic)

    _is_floating_type_y: core.constexpr = arg0.dtype.is_floating()
    core.static_assert(_is_floating_type_y == True,
                       f"Expected dtype fp16/fp32/bf16, but got {core.constexpr(arg0.dtype)}", _semantic=_semantic)

    half_pi: core.constexpr = 0.5 * pi
    arg0_fp32 = arg0.to(core.dtype("fp32"), _semantic=_semantic)
    arg1_fp32 = arg1.to(core.dtype("fp32"), _semantic=_semantic)
    atan_input = _semantic.truediv(arg0_fp32, arg1_fp32)

    # Ordering comparisons cannot tell -0.0 from +0.0, so the sign bit decides
    # the quadrant of a zero operand: atan2(-0.0, -1.0) is -pi, not +pi.
    y_negative = signbit(arg0_fp32, _semantic=_semantic)
    x_is_zero = _semantic.equal(arg1_fp32, 0)
    y_is_zero = _semantic.equal(arg0_fp32, 0)
    x_is_negative_zero = _semantic.logical_and(x_is_zero, signbit(arg1_fp32, _semantic=_semantic))

    # y / x is not finite at x == 0, so those quadrants are selected directly:
    # +-pi / 2 for a non-zero y, and +-0.0 (or +-pi when x == -0.0) for y == 0.
    zero_axis = _semantic.where(y_is_zero, _semantic.where(x_is_negative_zero, pi, 0.0), half_pi)
    zero_axis = _semantic.where(y_negative, _semantic.mul(zero_axis, -1.0, True), zero_axis)
    base = _semantic.where(x_is_zero, zero_axis, atan(atan_input, _semantic=_semantic))

    # x < 0 rotates the result by +-pi, with the sign taken from y.
    rotated = _semantic.where(y_negative, _semantic.sub(base, pi, True), _semantic.add(base, pi, True))
    ret = _semantic.where(_semantic.less_than(arg1_fp32, 0), rotated, base)
    return ret.to(arg1.dtype, _semantic=_semantic)


@core.builtin
@math._check_dtype(dtypes=["fp32"])
def trunc(arg0, _semantic=None):
    """
    Truncation operation, rounds toward zero to the nearest integer.

    :param arg0: ``x``. Supported dtype(s): ``float32``.
    :type arg0: scalar or tl.tensor
    :return: The truncation result.
    :rtype: ``float32``
    """
    if _is_libdevice_simt_enabled(_semantic):
        return core.extern_elementwise(
            "", "", [arg0], {
                (core.dtype("fp16"), ): ("__hmf_trunc_fp16", core.dtype("fp16")),
                (core.dtype("fp32"), ): ("__hmf_trunc_fp32", core.dtype("fp32")),
            }, is_pure=True, _semantic=_semantic)
    else:
        arg0 = _semantic.to_tensor(arg0)

        zero = _semantic.full(arg0.shape, 0.0, arg0.type.scalar)
        condition = _semantic.greater_equal(arg0, zero)

        floor_result = core.tensor(_semantic.builder.create_floor(arg0.handle), arg0.type)
        ceil_result = core.tensor(_semantic.builder.create_ceil(arg0.handle), arg0.type)

        return _semantic.where(condition, floor_result, ceil_result)


@core.extern
def round(arg0, _semantic=None):
    """
    Computes the nearest integer to x using round-to-nearest-even rounding mode.

    :param arg0: ``x``. Supported dtype(s): ``float32``.
    :type arg0: scalar or tl.tensor
    :return: The nearest integer to x.
    :rtype: ``float32``
    """
    if _is_libdevice_simt_enabled(_semantic):
        return core.extern_elementwise("", "", [arg0], {
            (core.dtype("fp32"), ): ("__hmf_round_fp32", core.dtype("fp32")),
        }, is_pure=True, _semantic=_semantic)
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("fp32"), ): ("__hmf_roundf", core.dtype("fp32")),
    }, is_pure=True, _semantic=_semantic)


@core.builtin
@math._check_dtype(dtypes=["bf16", "fp16", "fp32"])
def acos(arg0: core.tensor, _semantic=None):
    """
    Computes the inverse cosine (arccos) of the input parameter.

    :param arg0: ``x``. Supported dtype(s): ``float32``.
    :type arg0: scalar or tl.tensor
    :return: The inverse cosine of the input parameter, in the range [0, π] radians.
    :rtype: ``float32``
    """
    if arg0.dtype == core.dtype("fp32") and _is_a5_target(_semantic):
        return core.extern_elementwise("", "", [arg0], {
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
def sinh(arg0: core.tensor, _semantic=None):
    """
    Computes the hyperbolic sine of input parameter x.

    :param arg0: ``x``. Supported dtype(s): ``float32``.
    :type arg0: scalar or tl.tensor
    :return: The hyperbolic sine of input x.
    :rtype: ``float32``
    """
    if _is_libdevice_simt_enabled(_semantic):
        if arg0.dtype == core.dtype("bf16"):
            core.static_print("extern libdevice.sinh for dtype bf16 is unsupported for now.")
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
def cosh(arg0: core.tensor, _semantic=None):
    """
    Computes the hyperbolic cosine of the input parameter.

    :param arg0: ``x``. Supported dtype(s): ``float32``.
    :type arg0: scalar or tl.tensor
    :return: The hyperbolic cosine of the input parameter.
    :rtype: ``float32``
    """
    if _is_libdevice_simt_enabled(_semantic):
        if arg0.dtype == core.dtype("bf16"):
            core.static_print("extern libdevice.cosh for dtype bf16 is unsupported for now.")
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
def acosh(arg0: core.tensor, _semantic=None):
    """
    Computes the inverse hyperbolic cosine of the input parameter.

    :param arg0: ``x``. Supported dtype(s): ``float32``.
    :type arg0: scalar or tl.tensor
    :return: The inverse hyperbolic cosine of the input parameter, in the range [0, +∞].
    :rtype: ``float32``
    """
    if _is_libdevice_simt_enabled(_semantic):
        if arg0.dtype == core.dtype("bf16"):
            core.static_print("extern libdevice.acosh for dtype bf16 is unsupported for now.")
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
def asinh(arg0: core.tensor, _semantic=None):
    """
    Computes the inverse hyperbolic sine of the input parameter.

    :param arg0: ``x``. Supported dtype(s): ``float32``.
    :type arg0: scalar or tl.tensor
    :return: The inverse hyperbolic sine of the input parameter.
    :rtype: ``float32``
    """
    if _is_libdevice_simt_enabled(_semantic):
        if arg0.dtype == core.dtype("bf16"):
            core.static_print("extern libdevice.asinh for dtype bf16 is unsupported for now.")
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
def atanh(arg0: core.tensor, _semantic=None):
    """
    Inverse hyperbolic tangent function, computes the inverse hyperbolic tangent of the input parameter.

    :param arg0: ``x``. Supported dtype(s): ``float32``.
    :type arg0: scalar or tl.tensor
    :return: The inverse hyperbolic tangent of the input parameter, in the range (-∞, +∞).
    :rtype: ``float32``
    """
    if _is_libdevice_simt_enabled(_semantic):
        if arg0.dtype == core.dtype("bf16"):
            core.static_print("extern libdevice.atanh for dtype bf16 is unsupported for now.")
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
def expm1(arg0: core.tensor, _semantic=None):
    """
    Computes e raised to the power of x, minus 1.

    :param arg0: ``x``. Supported dtype(s): ``float32``.
    :type arg0: scalar or tl.tensor
    :return: The result of e raised to the power of x, minus 1.
    :rtype: ``float32``
    """
    if _is_libdevice_simt_enabled(_semantic):
        if arg0.dtype == core.dtype("bf16"):
            core.static_print("extern libdevice.expm1 for dtype bf16 is unsupported for now.")
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
def nextafter(arg0: core.tensor, arg1: core.tensor, _semantic=None):
    """
    Computes the next representable floating-point number from x toward y.

    :param arg0: ``x``. Supported dtype(s): ``float32``.
    :type arg0: scalar or tl.tensor
    :param arg1: ``y``. Supported dtype(s): ``float32``.
    :type arg1: scalar or tl.tensor
    :return: The next representable floating-point number.
    :rtype: ``float32``
    """
    if arg0.dtype == core.dtype("fp32") and _is_a5_target(_semantic):
        return core.extern_elementwise("", "", [arg0, arg1], {
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
def hypot(arg0: core.tensor, arg1: core.tensor, _semantic=None):
    """
    Computes the Euclidean distance between x and y.

    :param arg0: ``x``. Supported dtype(s): ``float32``.
    :type arg0: scalar or tl.tensor
    :param arg1: ``y``. Supported dtype(s): ``float32``.
    :type arg1: scalar or tl.tensor
    :return: The Euclidean distance between x and y.
    :rtype: ``float32``
    """
    if _is_libdevice_simt_enabled(_semantic):
        if arg0.dtype == core.dtype("bf16"):
            core.static_print("extern libdevice.hypot for dtype bf16 is unsupported for now.")
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


@core.extern
def cbrt(arg0, _semantic=None):
    """
    Computes the cube root of the input parameter.

    :param arg0: ``x``. Supported dtype(s): ``float32``.
    :type arg0: scalar or tl.tensor
    :return: The cube root of the input parameter.
    :rtype: ``float32``
    """
    if not _is_a5_target(_semantic):
        core.static_print("libdevice.cbrt for simd is unsupported for now.")
        core.static_assert(False)
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("fp32"), ): ("__hmf_cbrt_fp32", core.dtype("fp32")),
    }, is_pure=True, _semantic=_semantic)


@core.extern
def rcbrt(arg0, _semantic=None):
    """
    Computes the reciprocal cube root of x.

    :param arg0: ``x``. Supported dtype(s): ``float32``.
    :type arg0: scalar or tl.tensor
    :return: The reciprocal cube root of x.
    :rtype: ``float32``
    """
    if not _is_a5_target(_semantic):
        core.static_print("libdevice.rcbrt for simd is unsupported for now.")
        core.static_assert(False)
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("fp32"), ): ("__hmf_rcbrt_fp32", core.dtype("fp32")),
    }, is_pure=True, _semantic=_semantic)


@core.extern
def rhypot(arg0, arg1, _semantic=None):
    """
    Computes the reciprocal of the Euclidean distance between x and y.

    :param arg0: ``x``. Supported dtype(s): ``float32``.
    :type arg0: scalar or tl.tensor
    :param arg1: ``y``. Supported dtype(s): ``float32``.
    :type arg1: scalar or tl.tensor
    :return: The reciprocal of the Euclidean distance between x and y.
    :rtype: ``float32``
    """
    if not _is_a5_target(_semantic):
        core.static_print("libdevice.rhypot for simd is unsupported for now.")
        core.static_assert(False)
    return core.extern_elementwise("", "", [arg0, arg1], {
        (core.dtype("fp32"), core.dtype("fp32")): ("__hmf_rhypot_fp32", core.dtype("fp32")),
    }, is_pure=True, _semantic=_semantic)


@core.extern
def norm3d(arg0, arg1, arg2, _semantic=None):
    """
    Computes the Euclidean norm of a 3D vector, i.e., sqrt(x² + y² + z²).

    :param arg0: ``x``. Supported dtype(s): ``float32``.
    :type arg0: scalar or tl.tensor
    :param arg1: ``y``. Supported dtype(s): ``float32``.
    :type arg1: scalar or tl.tensor
    :param arg2: ``z``. Supported dtype(s): ``float32``.
    :type arg2: scalar or tl.tensor
    :return: The Euclidean norm of the 3D vector.
    :rtype: ``float32``
    """
    if not _is_a5_target(_semantic):
        core.static_print("libdevice.norm3d for simd is unsupported for now.")
        core.static_assert(False)
    return core.extern_elementwise("", "", [arg0, arg1, arg2], {
        (core.dtype("fp32"), core.dtype("fp32"), core.dtype("fp32")): ("__hmf_norm3d_fp32", core.dtype("fp32")),
    }, is_pure=True, _semantic=_semantic)


@core.extern
def rnorm3d(arg0, arg1, arg2, _semantic=None):
    """
    Computes the reciprocal of the Euclidean norm of a 3D vector.

    :param arg0: ``x``. Supported dtype(s): ``float32``.
    :type arg0: scalar or tl.tensor
    :param arg1: ``y``. Supported dtype(s): ``float32``.
    :type arg1: scalar or tl.tensor
    :param arg2: ``z``. Supported dtype(s): ``float32``.
    :type arg2: scalar or tl.tensor
    :return: The reciprocal of the Euclidean norm of the 3D vector.
    :rtype: ``float32``
    """
    if not _is_a5_target(_semantic):
        core.static_print("libdevice.rnorm3d for simd is unsupported for now.")
        core.static_assert(False)
    return core.extern_elementwise("", "", [arg0, arg1, arg2], {
        (core.dtype("fp32"), core.dtype("fp32"), core.dtype("fp32")): ("__hmf_rnorm3d_fp32", core.dtype("fp32")),
    }, is_pure=True, _semantic=_semantic)


@core.extern
def norm4d(arg0, arg1, arg2, arg3, _semantic=None):
    """
    Computes the Euclidean norm of a 4D vector, i.e., sqrt(x² + y² + z² + w²).

    :param arg0: ``x``. Supported dtype(s): ``float32``.
    :type arg0: scalar or tl.tensor
    :param arg1: ``y``. Supported dtype(s): ``float32``.
    :type arg1: scalar or tl.tensor
    :param arg2: ``z``. Supported dtype(s): ``float32``.
    :type arg2: scalar or tl.tensor
    :param arg3: ``w``. Supported dtype(s): ``float32``.
    :type arg3: scalar or tl.tensor
    :return: The Euclidean norm of the 4D vector.
    :rtype: ``float32``
    """
    if not _is_a5_target(_semantic):
        core.static_print("libdevice.norm4d for simd is unsupported for now.")
        core.static_assert(False)
    return core.extern_elementwise(
        "", "", [arg0, arg1, arg2, arg3], {
            (core.dtype("fp32"), core.dtype("fp32"), core.dtype("fp32"), core.dtype("fp32")):
            ("__hmf_norm4d_fp32", core.dtype("fp32")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def rnorm4d(arg0, arg1, arg2, arg3, _semantic=None):
    """
    Computes the reciprocal of the Euclidean norm of a 4D vector.

    :param arg0: ``x``. Supported dtype(s): ``float32``.
    :type arg0: scalar or tl.tensor
    :param arg1: ``y``. Supported dtype(s): ``float32``.
    :type arg1: scalar or tl.tensor
    :param arg2: ``z``. Supported dtype(s): ``float32``.
    :type arg2: scalar or tl.tensor
    :param arg3: ``w``. Supported dtype(s): ``float32``.
    :type arg3: scalar or tl.tensor
    :return: The reciprocal of the Euclidean norm of the 4D vector.
    :rtype: ``float32``
    """
    if not _is_a5_target(_semantic):
        core.static_print("libdevice.rnorm4d for simd is unsupported for now.")
        core.static_assert(False)
    return core.extern_elementwise(
        "", "", [arg0, arg1, arg2, arg3], {
            (core.dtype("fp32"), core.dtype("fp32"), core.dtype("fp32"), core.dtype("fp32")):
            ("__hmf_rnorm4d_fp32", core.dtype("fp32")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def j0(arg0, _semantic=None):
    """
    Computes the Bessel function of the first kind of order 0 of the input parameter.

    :param arg0: ``x``. Supported dtype(s): ``float32``.
    :type arg0: scalar or tl.tensor
    :return: The Bessel function of the first kind of order 0 of the input parameter.
    :rtype: ``float32``
    """
    if not _is_a5_target(_semantic):
        core.static_print("libdevice.j0 for simd is unsupported for now.")
        core.static_assert(False)
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("fp32"), ): ("__hmf_j0_fp32", core.dtype("fp32")),
    }, is_pure=True, _semantic=_semantic)


@core.extern
def j1(arg0, _semantic=None):
    """
    Computes the Bessel function of the first kind of order 1 of the input parameter.

    :param arg0: ``x``. Supported dtype(s): ``float32``.
    :type arg0: scalar or tl.tensor
    :return: The Bessel function of the first kind of order 1 of the input parameter.
    :rtype: ``float32``
    """
    if not _is_a5_target(_semantic):
        core.static_print("libdevice.j1 for simd is unsupported for now.")
        core.static_assert(False)
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("fp32"), ): ("__hmf_j1_fp32", core.dtype("fp32")),
    }, is_pure=True, _semantic=_semantic)


@core.extern
def jn(arg0, arg1, _semantic=None):
    """
    Computes the Bessel function of the first kind of order n of the input parameter.

    :param arg0: ``n``. Supported dtype(s): ``int32``.
    :type arg0: scalar or tl.tensor
    :param arg1: ``x``. Supported dtype(s): ``float32``.
    :type arg1: scalar or tl.tensor
    :return: The Bessel function of the first kind of order n of the input parameter.
    :rtype: ``float32``
    """
    if not _is_a5_target(_semantic):
        core.static_print("libdevice.jn for simd is unsupported for now.")
        core.static_assert(False)
    return core.extern_elementwise("", "", [arg0, arg1], {
        (core.dtype("int32"), core.dtype("fp32")): ("__hmf_jn_fp32", core.dtype("fp32")),
    }, is_pure=True, _semantic=_semantic)


@core.extern
def y0(arg0, _semantic=None):
    """
    Computes the Bessel function of the second kind of order 0 of the input parameter.

    :param arg0: ``x``. Supported dtype(s): ``float32``.
    :type arg0: scalar or tl.tensor
    :return: The Bessel function of the second kind of order 0 of the input parameter.
    :rtype: ``float32``
    """
    if not _is_a5_target(_semantic):
        core.static_print("libdevice.y0 for simd is unsupported for now.")
        core.static_assert(False)
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("fp32"), ): ("__hmf_y0_fp32", core.dtype("fp32")),
    }, is_pure=True, _semantic=_semantic)


@core.extern
def y1(arg0, _semantic=None):
    """
    Computes the Bessel function of the second kind of order 1 of the input parameter.

    :param arg0: ``x``. Supported dtype(s): ``float32``.
    :type arg0: scalar or tl.tensor
    :return: The Bessel function of the second kind of order 1 of the input parameter.
    :rtype: ``float32``
    """
    if not _is_a5_target(_semantic):
        core.static_print("libdevice.y1 for simd is unsupported for now.")
        core.static_assert(False)
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("fp32"), ): ("__hmf_y1_fp32", core.dtype("fp32")),
    }, is_pure=True, _semantic=_semantic)


@core.extern
def yn(arg0, arg1, _semantic=None):
    """
    Computes the Bessel function of the second kind of order n of the input parameter.

    :param arg0: ``n``. Supported dtype(s): ``int32``.
    :type arg0: scalar or tl.tensor
    :param arg1: ``x``. Supported dtype(s): ``float32``.
    :type arg1: scalar or tl.tensor
    :return: The Bessel function of the second kind of order n of the input parameter.
    :rtype: ``float32``
    """
    if not _is_a5_target(_semantic):
        core.static_print("libdevice.yn for simd is unsupported for now.")
        core.static_assert(False)
    return core.extern_elementwise("", "", [arg0, arg1], {
        (core.dtype("int32"), core.dtype("fp32")): ("__hmf_yn_fp32", core.dtype("fp32")),
    }, is_pure=True, _semantic=_semantic)


# This function is derived from the Cephes Math Library release 2.8: June, 2000
# https://netlib.org/cephes/
# Copyright (c) 1984, 1987, 2000 by Stephen L. Moshier
# All rights reserved.
@core.builtin
@math._check_dtype(dtypes=["fp16", "fp32"])
def cyl_bessel_i0(arg0: core.tensor, _semantic=None):
    """
    Computes the modified Bessel function of the first kind, order 0, of the input parameter.

    :param arg0: ``x``. Supported dtype(s): ``float32``.
    :type arg0: scalar or tl.tensor
    :return: The modified Bessel function of the first kind, order 0, of the input parameter.
    :rtype: ``float32``
    """
    if _is_libdevice_simt_enabled(_semantic):
        if arg0.dtype == core.dtype("fp16"):
            core.static_print("extern libdevice.cyl_bessel_i0 for dtype bf16 is unsupported for now.")
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
def cyl_bessel_i1(arg0, _semantic=None):
    """
    Computes the modified Bessel function of the first kind, order 1, of the input parameter.

    :param arg0: ``x``. Supported dtype(s): ``float32``.
    :type arg0: scalar or tl.tensor
    :return: The modified Bessel function of the first kind, order 1, of the input parameter.
    :rtype: ``float32``
    """
    if not _is_a5_target(_semantic):
        core.static_print("libdevice.cyl_bessel_i1 for simd is unsupported for now.")
        core.static_assert(False)
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("fp32"), ): ("__hmf_cyl_bessel_i1_fp32", core.dtype("fp32")),
    }, is_pure=True, _semantic=_semantic)


@core.extern
@math._check_dtype(dtypes=["fp16", "fp32"])
def signbit(arg0, _semantic=None):
    """
    Extracts the sign bit of x.

    :param arg0: ``x``. Supported dtype(s): ``float32``.
    :type arg0: scalar or tl.tensor
    :return: The sign bit of x.
    :rtype: ``int32``
    """
    if _is_libdevice_simt_enabled(_semantic):
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


@core.extern
@math._check_dtype(dtypes=["fp32", "fp64"])
def erf(arg0, _semantic=None):
    """
    Computes the error function of the input parameter.

    :param arg0: ``x``. Supported dtype(s): ``float32``.
    :type arg0: scalar or tl.tensor
    :return: The error function of the input parameter.
    :rtype: ``float32``
    """
    if _is_libdevice_simt_enabled(_semantic):
        return core.extern_elementwise("", "", [arg0], {
            (core.dtype("fp32"), ): ("__hmf_erf_fp32", core.dtype("fp32")),
        }, is_pure=True, _semantic=_semantic)
    arg0 = _semantic.to_tensor(arg0)
    return core.tensor(_semantic.builder.create_erf(arg0.handle), arg0.type)


@core.extern
def erfc(arg0, _semantic=None):
    """
    Computes the complementary error function of the input parameter, i.e., 1 - erf(x).

    :param arg0: ``x``. Supported dtype(s): ``float32``.
    :type arg0: scalar or tl.tensor
    :return: The complementary error function of the input parameter.
    :rtype: ``float32``
    """
    if not _is_a5_target(_semantic):
        core.static_print("libdevice.erfc for simd is unsupported for now.")
        core.static_assert(False)
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("fp32"), ): ("__hmf_erfc_fp32", core.dtype("fp32")),
    }, is_pure=True, _semantic=_semantic)


@core.extern
def erfcx(arg0, _semantic=None):
    """
    Computes the scaled complementary error function of the input parameter, i.e., exp(x²) × erfc(x).

    :param arg0: ``x``. Supported dtype(s): ``float32``.
    :type arg0: scalar or tl.tensor
    :return: The scaled complementary error function of the input parameter.
    :rtype: ``float32``
    """
    if not _is_a5_target(_semantic):
        core.static_print("libdevice.erfcx for simd is unsupported for now.")
        core.static_assert(False)
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("fp32"), ): ("__hmf_erfcx_fp32", core.dtype("fp32")),
    }, is_pure=True, _semantic=_semantic)


@core.extern
def erfcinv(arg0, _semantic=None):
    """
    Inverse complementary error function, finds the value y such that x = erfc(y).

    :param arg0: ``x``. Supported dtype(s): ``float32``.
    :type arg0: scalar or tl.tensor
    :return: The inverse complementary error function of the input parameter.
    :rtype: ``float32``
    """
    if not _is_a5_target(_semantic):
        core.static_print("libdevice.erfcxinv for simd is unsupported for now.")
        core.static_assert(False)
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("fp32"), ): ("__hmf_erfcinv_fp32", core.dtype("fp32")),
    }, is_pure=True, _semantic=_semantic)


# Note:
# For inputs x very close to ±1 (criterion: 1 - |x| < 1.1e-4), erfinv(x) → ±∞ and the
# inverse error function becomes extremely sensitive to tiny changes in x. The asymptotic
# behavior includes terms like sqrt(-ln(1-|x|)), so tiny relative changes in (1-|x|) map
# to large absolute changes in erfinv, leading to numerical instability and loss of precision,
# resulting in deviations from the reference results.
@core.extern
@math._check_dtype(dtypes=["fp32"])
def erfinv(arg0, _semantic=None):
    """
    Inverse error function, finds the value y such that x = erf(y).

    :param arg0: ``x``. Supported dtype(s): ``float32``.
    :type arg0: scalar or tl.tensor
    :return: The inverse error function of the input parameter.
    :rtype: ``float32``
    """
    if _is_libdevice_simt_enabled(_semantic):
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
            _semantic.where(
                signbit(arg0, _semantic=_semantic),
                _semantic.full(arg0.shape, -1, arg0_scalar_ty),
                _semantic.full(arg0.shape, 1, arg0_scalar_ty),
            ), core.tensor(high_res, arg0.type), True).handle

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
            _semantic.where(
                arg0_over, nan_tensor,
                _semantic.where(arg0_high, core.tensor(high_res, arg0.type), core.tensor(low_res, arg0.type))))


@core.extern
def normcdf(arg0, _semantic=None):
    """
    Computes the cumulative distribution function of the standard normal distribution.

    :param arg0: ``x``. Supported dtype(s): ``float32``.
    :type arg0: scalar or tl.tensor
    :return: The cumulative distribution function of the standard normal distribution.
    :rtype: ``float32``
    """
    if not _is_a5_target(_semantic):
        core.static_print("libdevice.normcdf for simd is unsupported for now.")
        core.static_assert(False)
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("fp32"), ): ("__hmf_normcdf_fp32", core.dtype("fp32")),
    }, is_pure=True, _semantic=_semantic)


@core.extern
def normcdfinv(arg0, _semantic=None):
    """
    Computes the inverse of the cumulative distribution function of the standard normal distribution.

    :param arg0: ``x``. Supported dtype(s): ``float32``.
    :type arg0: scalar or tl.tensor
    :return: The inverse of the cumulative distribution function of the standard normal distribution.
    :rtype: ``float32``
    """
    if not _is_a5_target(_semantic):
        core.static_print("libdevice.normcdfinv for simd is unsupported for now.")
        core.static_assert(False)
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("fp32"), ): ("__hmf_normcdfinv_fp32", core.dtype("fp32")),
    }, is_pure=True, _semantic=_semantic)


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
    """
    Computes the gamma function of the input parameter.

    :param arg0: ``x``. Supported dtype(s): ``float32``.
    :type arg0: scalar or tl.tensor
    :return: The gamma function of the input parameter.
    :rtype: ``float32``
    """
    if _is_libdevice_simt_enabled(_semantic):
        return core.extern_elementwise("", "", [arg0], {
            (core.dtype("fp32"), ): ("__hmf_tgamma_fp32", core.dtype("fp32")),
        }, is_pure=True, _semantic=_semantic)
    else:
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


@core.extern
def tgamma(arg0, _semantic=None):
    """
    Computes the gamma function of the input parameter.

    :param arg0: ``x``. Supported dtype(s): ``float32``.
    :type arg0: scalar or tl.tensor
    :return: The gamma function of the input parameter.
    :rtype: ``float32``
    """
    if not _is_a5_target(_semantic):
        core.static_print("libdevice.tgamma for simd is unsupported for now.")
        core.static_assert(False)
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("fp32"), ): ("__hmf_tgamma_fp32", core.dtype("fp32")),
    }, is_pure=True, _semantic=_semantic)


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
    """
    Computes the natural logarithm of the absolute value of the gamma function for input x.

    :param arg0: ``x``. Supported dtype(s): ``float32``.
    :type arg0: scalar or tl.tensor
    :return: The natural logarithm of the absolute value of the gamma function for input x.
    :rtype: ``float32``
    """
    if _is_libdevice_simt_enabled(_semantic):
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
def nearbyint(arg0: core.tensor, _semantic=None):
    """
    Converts x to the nearest integer.

    :param arg0: ``x``. Supported dtype(s): ``float32``.
    :type arg0: scalar or tl.tensor
    :return: The nearest integer.
    :rtype: ``float32``
    """
    if _is_libdevice_simt_enabled(_semantic):
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


@core.extern
def sinpi(arg0, _semantic=None):
    """
    Computes the value of sin(π × x).

    :param arg0: ``x``. Supported dtype(s): ``float32``.
    :type arg0: scalar or tl.tensor
    :return: The value of sin(π × x).
    :rtype: ``float32``
    """
    if not _is_a5_target(_semantic):
        core.static_print("libdevice.sinpi for simd is unsupported for now.")
        core.static_assert(False)
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("fp32"), ): ("__hmf_sinpi_fp32", core.dtype("fp32")),
    }, is_pure=True, _semantic=_semantic)


@core.extern
def cospi(arg0, _semantic=None):
    """
    Computes the value of cos(π × x).

    :param arg0: ``x``. Supported dtype(s): ``float32``.
    :type arg0: scalar or tl.tensor
    :return: The value of cos(π × x).
    :rtype: ``float32``
    """
    if not _is_a5_target(_semantic):
        core.static_print("libdevice.cospi for simd is unsupported for now.")
        core.static_assert(False)
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("fp32"), ): ("__hmf_cospi_fp32", core.dtype("fp32")),
    }, is_pure=True, _semantic=_semantic)


@core.builtin
@math._check_dtype(dtypes=[
    "fp32",
])
def asin(arg0: core.tensor, _semantic=None):
    """
    Computes the inverse sine (arcsin) of the input parameter.

    :param arg0: ``x``. Supported dtype(s): ``float32``.
    :type arg0: scalar or tl.tensor
    :return: The inverse sine of the input parameter, in the range [-π/2, π/2] radians.
    :rtype: ``float32``
    """
    if _is_libdevice_simt_enabled(_semantic):
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
def log10(arg0: core.tensor, _semantic=None):
    """
    Computes the base-10 logarithm of input x.

    :param arg0: ``x``. Supported dtype(s): ``float32``.
    :type arg0: scalar or tl.tensor
    :return: The base-10 logarithm of input x.
    :rtype: ``float32``
    """
    if _is_libdevice_simt_enabled(_semantic):
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
def copysign(arg0: core.tensor, arg1: core.tensor, _semantic=None):
    """
    Generates a floating-point number with magnitude equal to the magnitude of x and sign equal to the sign of y.

    :param arg0: ``x``. Supported dtype(s): ``float32``.
    :type arg0: scalar or tl.tensor
    :param arg1: ``y``. Supported dtype(s): ``float32``.
    :type arg1: scalar or tl.tensor
    :return: A floating-point number with magnitude equal to the magnitude of x and sign equal to the sign of y.
    :rtype: ``float32``
    """
    if _is_libdevice_simt_enabled(_semantic):
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


@core.builtin
@math._check_dtype(dtypes=["fp16", "fp32", "bf16"])
def rint(arg0: core.tensor, _semantic=None):
    """
    Computes the nearest integer to x using round-to-nearest-even rounding mode.

    :param arg0: ``x``. Supported dtype(s): ``float32``.
    :type arg0: scalar or tl.tensor
    :return: The nearest integer to x.
    :rtype: ``float32``
    """
    arg0 = _semantic.to_tensor(arg0)
    if _is_a5_target(_semantic):
        if arg0.dtype != core.dtype("fp32"):
            arg0 = _semantic.cast(arg0, core.dtype("fp32"))
        return core.extern_elementwise("", "", [
            arg0,
        ], {
            (core.dtype("fp32"), ): ("__hmf_rint_fp32", core.dtype("fp32")),
        }, is_pure=True, _semantic=_semantic)

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


@core.extern
def llrint(arg0, _semantic=None):
    """
    Rounds a floating-point number to the nearest 64-bit integer value.

    :param arg0: ``x``. Supported dtype(s): ``float32``.
    :type arg0: scalar or tl.tensor
    :return: The rounded 64-bit integer.
    :rtype: ``int64``
    """
    if not _is_a5_target(_semantic):
        core.static_print("libdevice.llrint for simd is unsupported for now.")
        core.static_assert(False)
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("fp32"), ): ("__hmf_llrint_fp32", core.dtype("int64")),
    }, is_pure=True, _semantic=_semantic)


@core.extern
def llround(arg0, _semantic=None):
    """
    Rounds a floating-point number to the nearest 64-bit integer value.

    :param arg0: ``x``. Supported dtype(s): ``float32``.
    :type arg0: scalar or tl.tensor
    :return: The rounded 64-bit integer.
    :rtype: ``int64``
    """
    if not _is_a5_target(_semantic):
        core.static_print("libdevice.llround for simd is unsupported for now.")
        core.static_assert(False)
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("fp32"), ): ("__hmf_llround_fp32", core.dtype("int64")),
    }, is_pure=True, _semantic=_semantic)


@core.extern
def abs(arg0, _semantic=None):
    """
    Computes the absolute value of the input parameter.

    :param arg0: ``x``. Supported dtype(s): ``int32``, ``float32``.
    :type arg0: scalar or tl.tensor
    :return: The absolute value of the input parameter.
    :rtype: ``int32``, ``float32``
    """
    if _is_libdevice_simt_enabled(_semantic):
        return core.extern_elementwise(
            "", "", [arg0], {
                (core.dtype("fp32"), ): ("__hmf_abs_fp32", core.dtype("fp32")),
                (core.dtype("int32"), ): ("__hmf_abs_i32", core.dtype("int32")),
            }, is_pure=True, _semantic=_semantic)

    arg0 = _semantic.to_tensor(arg0)
    dtype = arg0.dtype
    if dtype.is_fp8e4b15():
        mask = core.full(arg0.shape, 0x7F, core.int8, _semantic=_semantic)
        return core.tensor(_semantic.builder.create_and(arg0.handle, mask.handle), arg0.type)
    if dtype.is_floating():
        return core.tensor(_semantic.builder.create_fabs(arg0.handle), arg0.type)
    if dtype.is_int_signed():
        return core.tensor(_semantic.builder.create_iabs(arg0.handle), arg0.type)
    if dtype.is_int_unsigned():
        return arg0
    assert False, f"Unexpected dtype {dtype}"


@core.extern
@math._check_dtype(dtypes=["fp32", "fp64"])
def ceil(arg0, _semantic=None):
    """
    Ceiling operation, returns the smallest integer greater than or equal to x.

    :param arg0: ``x``. Supported dtype(s): ``float32``.
    :type arg0: scalar or tl.tensor
    :return: The ceiling result.
    :rtype: ``float32``
    """
    if _is_libdevice_simt_enabled(_semantic):
        return core.extern_elementwise("", "", [arg0], {
            (core.dtype("fp32"), ): ("__hmf_ceil_fp32", core.dtype("fp32")),
        }, is_pure=True, _semantic=_semantic)
    arg0 = _semantic.to_tensor(arg0)
    return core.tensor(_semantic.builder.create_ceil(arg0.handle), arg0.type)


@core.extern
@math._check_dtype(dtypes=["fp32", "fp64"])
def cos(arg0, _semantic=None):
    """
    Computes the cosine of the input parameter (in radians).

    :param arg0: ``x``. Supported dtype(s): ``float32``.
    :type arg0: scalar or tl.tensor
    :return: The cosine of the input parameter.
    :rtype: ``float32``
    """
    if _is_libdevice_simt_enabled(_semantic):
        return core.extern_elementwise("", "", [arg0], {
            (core.dtype("fp32"), ): ("__hmf_cos_fp32", core.dtype("fp32")),
        }, is_pure=True, _semantic=_semantic)
    arg0 = _semantic.to_tensor(arg0)
    return core.tensor(_semantic.builder.create_cos(arg0.handle), arg0.type)


@core.extern
@math._check_dtype(dtypes=["fp32"])
def div_rn(arg0, arg1, _semantic=None):
    """
    Floating-point division with round-to-nearest-even rounding mode.

    :param arg0: ``x``. Supported dtype(s): ``float32``.
    :type arg0: scalar or tl.tensor
    :param arg1: ``y``. Supported dtype(s): ``float32``.
    :type arg1: scalar or tl.tensor
    :return: The division result.
    :rtype: ``float32``
    """
    if _is_libdevice_simt_enabled(_semantic):
        return core.extern_elementwise("", "", [arg0, arg1], {
            (core.dtype("fp32"), core.dtype("fp32")): ("__hmf_div_rn_fp32", core.dtype("fp32")),
        }, is_pure=True, _semantic=_semantic)
    arg0 = _semantic.to_tensor(arg0)
    arg1 = _semantic.to_tensor(arg1)
    arg0, arg1 = core.binary_op_type_legalization(arg0, arg1, _semantic)
    return core.tensor(_semantic.builder.create_precise_divf(arg0.handle, arg1.handle), arg0.type)


@core.builtin
@math._add_math_2arg_docstr("division")
def fdiv(arg0, arg1, ieee_rounding=False, _semantic=None):
    ieee_rounding = core._unwrap_if_constexpr(ieee_rounding)
    arg0 = _semantic.to_tensor(arg0)
    arg1 = _semantic.to_tensor(arg1)
    return _semantic.fdiv(arg0, arg1, ieee_rounding)


@core.extern
@math._check_dtype(dtypes=["fp16", "fp32", "fp64"])
def exp(arg0, _semantic=None):
    """
    Exponential function, computes e raised to the power of x.

    :param arg0: ``x``. Supported dtype(s): ``float32``.
    :type arg0: scalar or tl.tensor
    :return: The result of e raised to the power of x.
    :rtype: ``float32``
    """
    if _is_libdevice_simt_enabled(_semantic):
        return core.extern_elementwise("", "", [arg0], {
            (core.dtype("fp32"), ): ("__hmf_exp_fp32", core.dtype("fp32")),
        }, is_pure=True, _semantic=_semantic)
    arg0 = _semantic.to_tensor(arg0)
    return core.tensor(_semantic.builder.create_exp(arg0.handle), arg0.type)


@core.extern
@math._check_dtype(dtypes=["fp32", "fp64"])
def exp2(arg0, _semantic=None):
    """
    Base-2 exponential function, computes 2 raised to the power of x.

    :param arg0: ``x``. Supported dtype(s): ``float32``.
    :type arg0: scalar or tl.tensor
    :return: The result of 2 raised to the power of x.
    :rtype: ``float32``
    """
    if _is_libdevice_simt_enabled(_semantic):
        return core.extern_elementwise("", "", [arg0], {
            (core.dtype("fp32"), ): ("__hmf_exp2_fp32", core.dtype("fp32")),
        }, is_pure=True, _semantic=_semantic)
    arg0 = _semantic.to_tensor(arg0)
    return core.tensor(_semantic.builder.create_exp2(arg0.handle), arg0.type)


@core.extern
def float2half_rn(arg0, _semantic=None):
    """
    Converts x from a 32-bit floating-point value to a 16-bit floating-point value using round-to-nearest-even.

    :param arg0: ``x``. Supported dtype(s): ``float32``.
    :type arg0: scalar or tl.tensor
    :return: The converted 16-bit floating-point value.
    :rtype: ``float16``
    """
    if not _is_a5_target(_semantic):
        core.static_print("libdevice.float2half_rn for simd is unsupported for now.")
        core.static_assert(False)
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("fp32"), ): ("__hmf_float2half_rn_fp32", core.dtype("fp16")),
    }, is_pure=True, _semantic=_semantic)


@core.extern
@math._check_dtype(dtypes=["fp32", "fp64"])
def floor(arg0, _semantic=None):
    """
    Floor operation, returns the largest integer less than or equal to x.

    :param arg0: ``x``. Supported dtype(s): ``float32``.
    :type arg0: scalar or tl.tensor
    :return: The floor result.
    :rtype: ``float32``
    """
    if _is_libdevice_simt_enabled(_semantic):
        return core.extern_elementwise("", "", [arg0], {
            (core.dtype("fp32"), ): ("__hmf_floor_fp32", core.dtype("fp32")),
        }, is_pure=True, _semantic=_semantic)
    arg0 = _semantic.to_tensor(arg0)
    return core.tensor(_semantic.builder.create_floor(arg0.handle), arg0.type)


@core.extern
def fma(arg0, arg1, arg2, _semantic=None):
    """
    Fused multiply-add, computes x × y + z.

    :param arg0: ``x``. Supported dtype(s): ``float32``.
    :type arg0: scalar or tl.tensor
    :param arg1: ``y``. Supported dtype(s): ``float32``.
    :type arg1: scalar or tl.tensor
    :param arg2: ``z``. Supported dtype(s): ``float32``.
    :type arg2: scalar or tl.tensor
    :return: The result of fused multiply-add.
    :rtype: ``float32``
    """
    if _is_libdevice_simt_enabled(_semantic):
        return core.extern_elementwise("", "", [arg0, arg1, arg2], {
            (core.dtype("fp32"), core.dtype("fp32"), core.dtype("fp32")): ("__hmf_fma_fp32", core.dtype("fp32")),
        }, is_pure=True, _semantic=_semantic)
    arg0 = _semantic.to_tensor(arg0)
    arg1 = _semantic.to_tensor(arg1)
    arg2 = _semantic.to_tensor(arg2)
    arg0, arg1 = core.binary_op_type_legalization(arg0, arg1, _semantic)
    arg2, arg0 = core.binary_op_type_legalization(arg2, arg0, _semantic)
    arg2, arg1 = core.binary_op_type_legalization(arg2, arg1, _semantic)
    return core.tensor(_semantic.builder.create_fma(arg0.handle, arg1.handle, arg2.handle), arg0.type)


@core.extern
def max(arg0, arg1, _semantic=None):
    """
    Computes the element-wise maximum of x and y.

    :param arg0: ``x``. Supported dtype(s): ``int32``, ``float32``.
    :type arg0: scalar or tl.tensor
    :param arg1: ``y``. Supported dtype(s): ``int32``, ``float32``; must have the same type as x.
    :type arg1: scalar or tl.tensor
    :return: The element-wise maximum of x and y.
    :rtype: Same as the input type (``int32`` or ``float32``)
    """
    if not _is_a5_target(_semantic):
        core.static_print("libdevice.max for simd is unsupported for now.")
        core.static_assert(False)
    return core.extern_elementwise(
        "", "", [arg0, arg1], {
            (core.dtype("int32"), core.dtype("int32")): ("__hmf_max_i32", core.dtype("int32")),
            (core.dtype("fp32"), core.dtype("fp32")): ("__hmf_fmax_fp32", core.dtype("fp32")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def min(arg0, arg1, _semantic=None):
    """
    Computes the element-wise minimum of x and y.

    :param arg0: ``x``. Supported dtype(s): ``int32``, ``float32``.
    :type arg0: scalar or tl.tensor
    :param arg1: ``y``. Supported dtype(s): ``int32``, ``float32``; must have the same type as x.
    :type arg1: scalar or tl.tensor
    :return: The element-wise minimum of x and y.
    :rtype: Same as the input type (``int32`` or ``float32``)
    """
    if not _is_a5_target(_semantic):
        core.static_print("libdevice.min for simd is unsupported for now.")
        core.static_assert(False)
    return core.extern_elementwise(
        "", "", [arg0, arg1], {
            (core.dtype("int32"), core.dtype("int32")): ("__hmf_min_i32", core.dtype("int32")),
            (core.dtype("fp32"), core.dtype("fp32")): ("__hmf_fmin_fp32", core.dtype("fp32")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def half2float(arg0, _semantic=None):
    """
    Converts x from a 16-bit floating-point value to a 32-bit floating-point value.

    :param arg0: ``x``. Supported dtype(s): ``float16``.
    :type arg0: scalar or tl.tensor
    :return: The converted 32-bit floating-point value.
    :rtype: ``float32``
    """
    if not _is_a5_target(_semantic):
        core.static_print("libdevice.half2float for simd is unsupported for now.")
        core.static_assert(False)
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("fp16"), ): ("__hmf_half2float_fp16", core.dtype("fp32")),
    }, is_pure=True, _semantic=_semantic)


@core.extern
@math._check_dtype(dtypes=["fp32", "fp64"])
def log(arg0, _semantic=None):
    """
    Computes the natural (base-e) logarithm of input x.

    :param arg0: ``x``. Supported dtype(s): ``float32``.
    :type arg0: scalar or tl.tensor
    :return: The natural logarithm of input x.
    :rtype: ``float32``
    """
    if _is_libdevice_simt_enabled(_semantic):
        return core.extern_elementwise("", "", [arg0], {
            (core.dtype("fp32"), ): ("__hmf_log_fp32", core.dtype("fp32")),
        }, is_pure=True, _semantic=_semantic)
    arg0 = _semantic.to_tensor(arg0)
    return core.tensor(_semantic.builder.create_log(arg0.handle), arg0.type)


@core.extern
@math._check_dtype(dtypes=["fp32", "fp64"])
def log2(arg0, _semantic=None):
    """
    Computes the base-2 logarithm of input x.

    :param arg0: ``x``. Supported dtype(s): ``float32``.
    :type arg0: scalar or tl.tensor
    :return: The base-2 logarithm of input x.
    :rtype: ``float32``
    """
    if _is_libdevice_simt_enabled(_semantic):
        return core.extern_elementwise("", "", [arg0], {
            (core.dtype("fp32"), ): ("__hmf_log2_fp32", core.dtype("fp32")),
        }, is_pure=True, _semantic=_semantic)
    arg0 = _semantic.to_tensor(arg0)
    return core.tensor(_semantic.builder.create_log2(arg0.handle), arg0.type)


@core.extern
def nan(arg0, _semantic=None):
    """
    Generates a NaN value from x.

    :param arg0: ``x``. Supported dtype(s): ``float32``.
    :type arg0: scalar or tl.tensor
    :return: The generated NaN value.
    :rtype: ``float32``
    """
    if not _is_a5_target(_semantic):
        core.static_print("libdevice.nan for simd is unsupported for now.")
        core.static_assert(False)
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("fp32"), ): ("__hmf_nan_fp32", core.dtype("fp32")),
    }, is_pure=True, _semantic=_semantic)


@core.extern
@math._check_dtype(dtypes=["fp32", "fp64"])
def rsqrt(arg0, _semantic=None):
    """
    Computes the reciprocal square root of x.

    :param arg0: ``x``. Supported dtype(s): ``float32``.
    :type arg0: scalar or tl.tensor
    :return: The reciprocal square root of x.
    :rtype: ``float32``
    """
    if _is_libdevice_simt_enabled(_semantic):
        return core.extern_elementwise("", "", [arg0], {
            (core.dtype("fp32"), ): ("__hmf_rsqrt_fp32", core.dtype("fp32")),
        }, is_pure=True, _semantic=_semantic)
    arg0 = _semantic.to_tensor(arg0)
    return core.tensor(_semantic.builder.create_rsqrt(arg0.handle), arg0.type)


@core.extern
def sin(arg0, _semantic=None):
    """
    Computes the sine of the input parameter x (in radians).

    :param arg0: ``x``. Supported dtype(s): ``float32``.
    :type arg0: scalar or tl.tensor
    :return: The sine of input x.
    :rtype: ``float32``
    """
    arg0 = _semantic.to_tensor(arg0)
    if arg0.dtype == core.dtype("fp32") and _is_a5_target(_semantic):
        return core.extern_elementwise("", "", [arg0], {
            (core.dtype("fp32"), ): ("__hmf_sin_fp32", core.dtype("fp32")),
        }, is_pure=True, _semantic=_semantic)
    return core.tensor(_semantic.builder.create_sin(arg0.handle), arg0.type)


@core.extern
@math._check_dtype(dtypes=["fp32", "fp64"])
def sqrt(arg0, _semantic=None):
    """
    Computes the square root of x.

    :param arg0: ``x``. Supported dtype(s): ``float32``.
    :type arg0: scalar or tl.tensor
    :return: The square root of x.
    :rtype: ``float32``
    """
    if _is_libdevice_simt_enabled(_semantic):
        return core.extern_elementwise("", "", [arg0], {
            (core.dtype("fp32"), ): ("__hmf_sqrt_fp32", core.dtype("fp32")),
        }, is_pure=True, _semantic=_semantic)
    arg0 = _semantic.to_tensor(arg0)
    return core.tensor(_semantic.builder.create_sqrt(arg0.handle), arg0.type)


@core.extern
@_deprecated(fn_name="triton.language.extra.cann.libdevice.uhadd",
             replacement="triton.language.extra.cann.libdevice.hadd")
def uhadd(arg0, arg1, _semantic=None):
    return hadd(arg0, arg1, _semantic=_semantic)


@core.extern
@_deprecated(fn_name="triton.language.extra.cann.libdevice.umul24",
             replacement="triton.language.extra.cann.libdevice.mul24")
def umul24(arg0, arg1, _semantic=None):
    return mul24(arg0, arg1, _semantic=_semantic)


@core.extern
@_deprecated(fn_name="triton.language.extra.cann.libdevice.umulhi",
             replacement="triton.language.extra.cann.libdevice.mulhi")
def umulhi(arg0, arg1, _semantic=None):
    return mulhi(arg0, arg1, _semantic=_semantic)


@core.extern
@_deprecated(fn_name="triton.language.extra.cann.libdevice.urhadd",
             replacement="triton.language.extra.cann.libdevice.rhadd")
def urhadd(arg0, arg1, _semantic=None):
    return rhadd(arg0, arg1, _semantic=_semantic)


@core.extern
@_deprecated(fn_name="triton.language.extra.cann.libdevice.usad",
             replacement="triton.language.extra.cann.libdevice.sad")
def usad(arg0, arg1, arg2, _semantic=None):
    return sad(arg0, arg1, arg2, _semantic=_semantic)
