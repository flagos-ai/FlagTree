from triton.language import core
from triton.language import math as _math

_FP32 = core.dtype("fp32")


@core.extern
def clz(arg0, _semantic=None):
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("int32"), ): ("Unsupported.clz", core.dtype("int32")),
            (core.dtype("int64"), ): ("Unsupported.clz", core.dtype("int32")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def popc(arg0, _semantic=None):
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("int32"), ): ("Unsupported.popc", core.dtype("int32")),
            (core.dtype("int64"), ): ("Unsupported.popc", core.dtype("int32")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def byte_perm(arg0, arg1, arg2, _semantic=None):
    return core.extern_elementwise("", "", [arg0, arg1, arg2], {
        (core.dtype("int32"), core.dtype("int32"), core.dtype("int32")): ("Unsupported.byte_perm", core.dtype("int32")),
    }, is_pure=True, _semantic=_semantic)


@core.extern
def mulhi(arg0, arg1, _semantic=None):
    return core.extern_elementwise(
        "", "", [arg0, arg1], {
            (core.dtype("int32"), core.dtype("int32")): ("Unsupported.mulhi", core.dtype("int32")),
            (core.dtype("uint32"), core.dtype("uint32")): ("_ZN3xpu6umulhiEjj", core.dtype("uint32")),
            (core.dtype("int64"), core.dtype("int64")): ("Unsupported.mulhi", core.dtype("int64")),
            (core.dtype("uint64"), core.dtype("uint64")): ("Unsupported.mulhi", core.dtype("uint64")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def mul24(arg0, arg1, _semantic=None):
    return core.extern_elementwise(
        "", "", [arg0, arg1], {
            (core.dtype("int32"), core.dtype("int32")): ("Unsupported.mul24", core.dtype("int32")),
            (core.dtype("uint32"), core.dtype("uint32")): ("Unsupported.mul24", core.dtype("uint32")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def brev(arg0, _semantic=None):
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("int32"), ): ("Unsupported.brev", core.dtype("int32")),
            (core.dtype("int64"), ): ("Unsupported.brev", core.dtype("int64")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def sad(arg0, arg1, arg2, _semantic=None):
    return core.extern_elementwise(
        "", "", [arg0, arg1, arg2], {
            (core.dtype("int32"), core.dtype("int32"), core.dtype("uint32")): ("Unsupported.sad", core.dtype("int32")),
            (core.dtype("uint32"), core.dtype("uint32"), core.dtype("uint32")):
            ("Unsupported.sad", core.dtype("uint32")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def abs(arg0, _semantic=None):
    # tt.absf / tt.absi cover every supported dtype directly; no device symbol
    # is needed. Reached from FlagGems via libdevice.abs (special_round).
    return _math.abs(arg0, _semantic=_semantic)


@core.extern
def floor(arg0, _semantic=None):
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp32"), ): ("_ZN3xpu9xpu_floorEf", core.dtype("fp32")),
            (core.dtype("fp64"), ): ("_ZN3xpu9xpu_floorEd", core.dtype("fp64")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def rcp64h(arg0, _semantic=None):
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("fp64"), ): ("Unsupported.rcp64h", core.dtype("fp64")),
    }, is_pure=True, _semantic=_semantic)


@core.extern
def rsqrt(arg0, _semantic=None):
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp16"), ): ("_ZN3xpu6hrsqrtEDF16_", core.dtype("fp16")),
            (core.dtype("fp32"), ): ("_ZN3xpu6rsqrtfEf", core.dtype("fp32")),
            (core.dtype("fp64"), ): ("_ZN3xpu6rsqrtfEd", core.dtype("fp64")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def ceil(arg0, _semantic=None):
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp64"), ): ("_ZN3xpu8xpu_ceilEd", core.dtype("fp64")),
            (core.dtype("fp32"), ): ("_ZN3xpu8xpu_ceilEf", core.dtype("fp32")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def trunc(arg0, _semantic=None):
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp64"), ): ("Unsupported.trunc", core.dtype("fp64")),
            (core.dtype("fp32"), ): ("_ZN3xpu6truncfEf", core.dtype("fp32")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def exp2(arg0, _semantic=None):
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp32"), ): ("_ZN3xpu5exp2fEf", core.dtype("fp32")),
            (core.dtype("fp64"), ): ("Unsupported.exp2", core.dtype("fp64")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def saturatef(arg0, _semantic=None):
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("fp32"), ): ("Unsupported.saturatef", core.dtype("fp32")),
    }, is_pure=True, _semantic=_semantic)


@core.extern
def fma_rn(arg0, arg1, arg2, _semantic=None):
    return core.extern_elementwise(
        "", "", [arg0, arg1, arg2], {
            (core.dtype("fp32"), core.dtype("fp32"), core.dtype("fp32")): ("Unsupported.fma_rn", core.dtype("fp32")),
            (core.dtype("fp64"), core.dtype("fp64"), core.dtype("fp64")): ("Unsupported.fma_rn", core.dtype("fp64")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def fma_rz(arg0, arg1, arg2, _semantic=None):
    return core.extern_elementwise(
        "", "", [arg0, arg1, arg2], {
            (core.dtype("fp32"), core.dtype("fp32"), core.dtype("fp32")): ("Unsupported.fma_rz", core.dtype("fp32")),
            (core.dtype("fp64"), core.dtype("fp64"), core.dtype("fp64")): ("Unsupported.fma_rz", core.dtype("fp64")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def fma_rd(arg0, arg1, arg2, _semantic=None):
    return core.extern_elementwise(
        "", "", [arg0, arg1, arg2], {
            (core.dtype("fp32"), core.dtype("fp32"), core.dtype("fp32")): ("Unsupported.fma_rd", core.dtype("fp32")),
            (core.dtype("fp64"), core.dtype("fp64"), core.dtype("fp64")): ("Unsupported.fma_rd", core.dtype("fp64")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def fma_ru(arg0, arg1, arg2, _semantic=None):
    return core.extern_elementwise(
        "", "", [arg0, arg1, arg2], {
            (core.dtype("fp32"), core.dtype("fp32"), core.dtype("fp32")): ("Unsupported.fma_ru", core.dtype("fp32")),
            (core.dtype("fp64"), core.dtype("fp64"), core.dtype("fp64")): ("Unsupported.fma_ru", core.dtype("fp64")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def fast_dividef(arg0, arg1, _semantic=None):
    return core.extern_elementwise("", "", [arg0, arg1], {
        (core.dtype("fp32"), core.dtype("fp32")): ("libdevice.fast_dividef", core.dtype("fp32")),
    }, is_pure=True, _semantic=_semantic)


@core.extern
def div_rn(arg0, arg1, _semantic=None):
    return core.extern_elementwise(
        "", "", [arg0, arg1], {
            (core.dtype("fp32"), core.dtype("fp32")): ("_ZN3xpu9__fdiv_rnEff", core.dtype("fp32")),
            (core.dtype("fp64"), core.dtype("fp64")): ("Unsupported.div_rn", core.dtype("fp64")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def div_rz(arg0, arg1, _semantic=None):
    return core.extern_elementwise(
        "", "", [arg0, arg1], {
            (core.dtype("fp32"), core.dtype("fp32")): ("_ZN3xpu9__fdiv_rzEff", core.dtype("fp32")),
            (core.dtype("fp64"), core.dtype("fp64")): ("Unsupported.div_rz", core.dtype("fp64")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def div_rd(arg0, arg1, _semantic=None):
    return core.extern_elementwise(
        "", "", [arg0, arg1], {
            (core.dtype("fp32"), core.dtype("fp32")): ("Unsupported.div_rd", core.dtype("fp32")),
            (core.dtype("fp64"), core.dtype("fp64")): ("Unsupported.div_rd", core.dtype("fp64")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def div_ru(arg0, arg1, _semantic=None):
    return core.extern_elementwise(
        "", "", [arg0, arg1], {
            (core.dtype("fp32"), core.dtype("fp32")): ("Unsupported.div_ru", core.dtype("fp32")),
            (core.dtype("fp64"), core.dtype("fp64")): ("Unsupported.div_ru", core.dtype("fp64")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def rcp_rn(arg0, _semantic=None):
    arg0 = _semantic.to_tensor(arg0)
    if arg0.dtype == core.dtype("fp32"):
        return arg0.__rtruediv__(1.0, _semantic=_semantic)
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("fp64"), ): ("Unsupported.rcp_rn", core.dtype("fp64")),
    }, is_pure=True, _semantic=_semantic)


@core.extern
def rcp_rz(arg0, _semantic=None):
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp32"), ): ("Unsupported.rcp_rz", core.dtype("fp32")),
            (core.dtype("fp64"), ): ("Unsupported.rcp_rz", core.dtype("fp64")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def rcp_rd(arg0, _semantic=None):
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp32"), ): ("Unsupported.rcp_rd", core.dtype("fp32")),
            (core.dtype("fp64"), ): ("Unsupported.rcp_rd", core.dtype("fp64")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def rcp_ru(arg0, _semantic=None):
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp32"), ): ("Unsupported.rcp_ru", core.dtype("fp32")),
            (core.dtype("fp64"), ): ("Unsupported.rcp_ru", core.dtype("fp64")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def sqrt_rn(arg0, _semantic=None):
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp32"), ): ("_ZN3xpu10__fsqrt_rnEf", core.dtype("fp32")),
            (core.dtype("fp64"), ): ("_ZN3xpu10__dsqrt_rnEd", core.dtype("fp64")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def sqrt_rz(arg0, _semantic=None):
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp32"), ): ("Unsupported.sqrt_rz", core.dtype("fp32")),
            (core.dtype("fp64"), ): ("Unsupported.sqrt_rz", core.dtype("fp64")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def sqrt_rd(arg0, _semantic=None):
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp32"), ): ("Unsupported.sqrt_rd", core.dtype("fp32")),
            (core.dtype("fp64"), ): ("Unsupported.sqrt_rd", core.dtype("fp64")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def sqrt_ru(arg0, _semantic=None):
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp32"), ): ("Unsupported.sqrt_ru", core.dtype("fp32")),
            (core.dtype("fp64"), ): ("Unsupported.sqrt_ru", core.dtype("fp64")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def sqrt(arg0, _semantic=None):
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp32"), ): ("_ZN3xpu8xpu_sqrtEf", core.dtype("fp32")),
            (core.dtype("fp64"), ): ("_ZN3xpu8xpu_sqrtEd", core.dtype("fp64")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def add_rn(arg0, arg1, _semantic=None):
    return core.extern_elementwise(
        "", "", [arg0, arg1], {
            (core.dtype("fp64"), core.dtype("fp64")): ("Unsupported.add_rn", core.dtype("fp64")),
            (core.dtype("fp32"), core.dtype("fp32")): ("Unsupported.add_rn", core.dtype("fp32")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def add_rz(arg0, arg1, _semantic=None):
    return core.extern_elementwise(
        "", "", [arg0, arg1], {
            (core.dtype("fp64"), core.dtype("fp64")): ("Unsupported.add_rz", core.dtype("fp64")),
            (core.dtype("fp32"), core.dtype("fp32")): ("Unsupported.add_rz", core.dtype("fp32")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def add_rd(arg0, arg1, _semantic=None):
    return core.extern_elementwise(
        "", "", [arg0, arg1], {
            (core.dtype("fp64"), core.dtype("fp64")): ("Unsupported.add_rd", core.dtype("fp64")),
            (core.dtype("fp32"), core.dtype("fp32")): ("Unsupported.add_rd", core.dtype("fp32")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def add_ru(arg0, arg1, _semantic=None):
    return core.extern_elementwise(
        "", "", [arg0, arg1], {
            (core.dtype("fp64"), core.dtype("fp64")): ("Unsupported.add_ru", core.dtype("fp64")),
            (core.dtype("fp32"), core.dtype("fp32")): ("Unsupported.add_ru", core.dtype("fp32")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def mul_rn(arg0, arg1, _semantic=None):
    return core.extern_elementwise(
        "", "", [arg0, arg1], {
            (core.dtype("fp64"), core.dtype("fp64")): ("Unsupported.mul_rn", core.dtype("fp64")),
            (core.dtype("fp32"), core.dtype("fp32")): ("Unsupported.mul_rn", core.dtype("fp32")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def mul_rz(arg0, arg1, _semantic=None):
    return core.extern_elementwise(
        "", "", [arg0, arg1], {
            (core.dtype("fp64"), core.dtype("fp64")): ("Unsupported.mul_rz", core.dtype("fp64")),
            (core.dtype("fp32"), core.dtype("fp32")): ("Unsupported.mul_rz", core.dtype("fp32")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def mul_rd(arg0, arg1, _semantic=None):
    return core.extern_elementwise(
        "", "", [arg0, arg1], {
            (core.dtype("fp64"), core.dtype("fp64")): ("Unsupported.mul_rd", core.dtype("fp64")),
            (core.dtype("fp32"), core.dtype("fp32")): ("Unsupported.mul_rd", core.dtype("fp32")),
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
            ): ("Unsupported.mul_ru", core.dtype("fp64")),
            (
                core.dtype("fp32"),
                core.dtype("fp32"),
            ): ("Unsupported.mul_ru", core.dtype("fp32")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def double2float_rn(arg0, _semantic=None):
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("fp64"), ): ("Unsupported.double2float_rn", core.dtype("fp32")),
    }, is_pure=True, _semantic=_semantic)


@core.extern
def double2float_rz(arg0, _semantic=None):
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("fp64"), ): ("Unsupported.double2float_rz", core.dtype("fp32")),
    }, is_pure=True, _semantic=_semantic)


@core.extern
def double2float_rd(arg0, _semantic=None):
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("fp64"), ): ("Unsupported.double2float_rd", core.dtype("fp32")),
    }, is_pure=True, _semantic=_semantic)


@core.extern
def double2float_ru(arg0, _semantic=None):
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("fp64"), ): ("Unsupported.double2float_ru", core.dtype("fp32")),
    }, is_pure=True, _semantic=_semantic)


@core.extern
def double2int_rn(arg0, _semantic=None):
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("fp64"), ): ("Unsupported.double2int_rn", core.dtype("int32")),
    }, is_pure=True, _semantic=_semantic)


@core.extern
def double2int_rz(arg0, _semantic=None):
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("fp64"), ): ("Unsupported.double2int_rz", core.dtype("int32")),
    }, is_pure=True, _semantic=_semantic)


@core.extern
def double2int_rd(arg0, _semantic=None):
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("fp64"), ): ("Unsupported.double2int_rd", core.dtype("int32")),
    }, is_pure=True, _semantic=_semantic)


@core.extern
def double2int_ru(arg0, _semantic=None):
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("fp64"), ): ("Unsupported.double2int_ru", core.dtype("int32")),
    }, is_pure=True, _semantic=_semantic)


@core.extern
def double2uint_rn(arg0, _semantic=None):
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("fp64"), ): ("Unsupported.double2uint_rn", core.dtype("int32")),
    }, is_pure=True, _semantic=_semantic)


@core.extern
def double2uint_rz(arg0, _semantic=None):
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("fp64"), ): ("Unsupported.double2uint_rz", core.dtype("int32")),
    }, is_pure=True, _semantic=_semantic)


@core.extern
def double2uint_rd(arg0, _semantic=None):
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("fp64"), ): ("Unsupported.double2uint_rd", core.dtype("int32")),
    }, is_pure=True, _semantic=_semantic)


@core.extern
def double2uint_ru(arg0, _semantic=None):
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("fp64"), ): ("Unsupported.double2uint_ru", core.dtype("int32")),
    }, is_pure=True, _semantic=_semantic)


@core.extern
def int2double_rn(arg0, _semantic=None):
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("int32"), ): ("Unsupported.int2double_rn", core.dtype("fp64")),
    }, is_pure=True, _semantic=_semantic)


@core.extern
def uint2double_rn(arg0, _semantic=None):
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("uint32"), ): ("Unsupported.uint2double_rn", core.dtype("fp64")),
    }, is_pure=True, _semantic=_semantic)


@core.extern
def float2int_rn(arg0, _semantic=None):
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("fp32"), ): ("Unsupported.float2int_rn", core.dtype("int32")),
    }, is_pure=True, _semantic=_semantic)


@core.extern
def float2int_rz(arg0, _semantic=None):
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("fp32"), ): ("Unsupported.float2int_rz", core.dtype("int32")),
    }, is_pure=True, _semantic=_semantic)


@core.extern
def float2int_rd(arg0, _semantic=None):
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("fp32"), ): ("Unsupported.float2int_rd", core.dtype("int32")),
    }, is_pure=True, _semantic=_semantic)


@core.extern
def float2int_ru(arg0, _semantic=None):
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("fp32"), ): ("Unsupported.float2int_ru", core.dtype("int32")),
    }, is_pure=True, _semantic=_semantic)


@core.extern
def float2uint_rn(arg0, _semantic=None):
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("fp32"), ): ("Unsupported.float2uint_rn", core.dtype("int32")),
    }, is_pure=True, _semantic=_semantic)


@core.extern
def float2uint_rz(arg0, _semantic=None):
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("fp32"), ): ("Unsupported.float2uint_rz", core.dtype("int32")),
    }, is_pure=True, _semantic=_semantic)


@core.extern
def float2uint_rd(arg0, _semantic=None):
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("fp32"), ): ("Unsupported.float2uint_rd", core.dtype("int32")),
    }, is_pure=True, _semantic=_semantic)


@core.extern
def float2uint_ru(arg0, _semantic=None):
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("fp32"), ): ("Unsupported.float2uint_ru", core.dtype("int32")),
    }, is_pure=True, _semantic=_semantic)


@core.extern
def int2float_rn(arg0, _semantic=None):
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("int32"), ): ("Unsupported.int2float_rn", core.dtype("fp32")),
    }, is_pure=True, _semantic=_semantic)


@core.extern
def int2float_rz(arg0, _semantic=None):
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("int32"), ): ("Unsupported.int2float_rz", core.dtype("fp32")),
    }, is_pure=True, _semantic=_semantic)


@core.extern
def int2float_rd(arg0, _semantic=None):
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("int32"), ): ("Unsupported.int2float_rd", core.dtype("fp32")),
    }, is_pure=True, _semantic=_semantic)


@core.extern
def int2float_ru(arg0, _semantic=None):
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("int32"), ): ("Unsupported.int2float_ru", core.dtype("fp32")),
    }, is_pure=True, _semantic=_semantic)


@core.extern
def uint2float_rn(arg0, _semantic=None):
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("uint32"), ): ("Unsupported.uint2float_rn", core.dtype("fp32")),
    }, is_pure=True, _semantic=_semantic)


@core.extern
def uint2float_rz(arg0, _semantic=None):
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("uint32"), ): ("Unsupported.uint2float_rz", core.dtype("fp32")),
    }, is_pure=True, _semantic=_semantic)


@core.extern
def uint2float_rd(arg0, _semantic=None):
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("uint32"), ): ("Unsupported.uint2float_rd", core.dtype("fp32")),
    }, is_pure=True, _semantic=_semantic)


@core.extern
def uint2float_ru(arg0, _semantic=None):
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("uint32"), ): ("Unsupported.uint2float_ru", core.dtype("fp32")),
    }, is_pure=True, _semantic=_semantic)


@core.extern
def hiloint2double(arg0, arg1, _semantic=None):
    return core.extern_elementwise("", "", [arg0, arg1], {
        (core.dtype("int32"), core.dtype("int32")): ("Unsupported.hiloint2double", core.dtype("fp64")),
    }, is_pure=True, _semantic=_semantic)


@core.extern
def double2loint(arg0, _semantic=None):
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("fp64"), ): ("Unsupported.double2loint", core.dtype("int32")),
    }, is_pure=True, _semantic=_semantic)


@core.extern
def double2hiint(arg0, _semantic=None):
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("fp64"), ): ("Unsupported.double2hiint", core.dtype("int32")),
    }, is_pure=True, _semantic=_semantic)


@core.extern
def float2ll_rn(arg0, _semantic=None):
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("fp32"), ): ("Unsupported.float2ll_rn", core.dtype("int64")),
    }, is_pure=True, _semantic=_semantic)


@core.extern
def float2ll_rz(arg0, _semantic=None):
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("fp32"), ): ("Unsupported.float2ll_rz", core.dtype("int64")),
    }, is_pure=True, _semantic=_semantic)


@core.extern
def float2ll_rd(arg0, _semantic=None):
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("fp32"), ): ("Unsupported.float2ll_rd", core.dtype("int64")),
    }, is_pure=True, _semantic=_semantic)


@core.extern
def float2ll_ru(arg0, _semantic=None):
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("fp32"), ): ("Unsupported.float2ll_ru", core.dtype("int64")),
    }, is_pure=True, _semantic=_semantic)


@core.extern
def float2ull_rn(arg0, _semantic=None):
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("fp32"), ): ("Unsupported.float2ull_rn", core.dtype("int64")),
    }, is_pure=True, _semantic=_semantic)


@core.extern
def float2ull_rz(arg0, _semantic=None):
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("fp32"), ): ("Unsupported.float2ull_rz", core.dtype("int64")),
    }, is_pure=True, _semantic=_semantic)


@core.extern
def float2ull_rd(arg0, _semantic=None):
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("fp32"), ): ("Unsupported.float2ull_rd", core.dtype("int64")),
    }, is_pure=True, _semantic=_semantic)


@core.extern
def float2ull_ru(arg0, _semantic=None):
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("fp32"), ): ("Unsupported.float2ull_ru", core.dtype("int64")),
    }, is_pure=True, _semantic=_semantic)


@core.extern
def double2ll_rn(arg0, _semantic=None):
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("fp64"), ): ("Unsupported.double2ll_rn", core.dtype("int64")),
    }, is_pure=True, _semantic=_semantic)


@core.extern
def double2ll_rz(arg0, _semantic=None):
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("fp64"), ): ("Unsupported.double2ll_rz", core.dtype("int64")),
    }, is_pure=True, _semantic=_semantic)


@core.extern
def double2ll_rd(arg0, _semantic=None):
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("fp64"), ): ("Unsupported.double2ll_rd", core.dtype("int64")),
    }, is_pure=True, _semantic=_semantic)


@core.extern
def double2ll_ru(arg0, _semantic=None):
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("fp64"), ): ("Unsupported.double2ll_ru", core.dtype("int64")),
    }, is_pure=True, _semantic=_semantic)


@core.extern
def double2ull_rn(arg0, _semantic=None):
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("fp64"), ): ("Unsupported.double2ull_rn", core.dtype("int64")),
    }, is_pure=True, _semantic=_semantic)


@core.extern
def double2ull_rz(arg0, _semantic=None):
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("fp64"), ): ("Unsupported.double2ull_rz", core.dtype("int64")),
    }, is_pure=True, _semantic=_semantic)


@core.extern
def double2ull_rd(arg0, _semantic=None):
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("fp64"), ): ("Unsupported.double2ull_rd", core.dtype("int64")),
    }, is_pure=True, _semantic=_semantic)


@core.extern
def double2ull_ru(arg0, _semantic=None):
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("fp64"), ): ("Unsupported.double2ull_ru", core.dtype("int64")),
    }, is_pure=True, _semantic=_semantic)


@core.extern
def ll2float_rn(arg0, _semantic=None):
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("int64"), ): ("Unsupported.ll2float_rn", core.dtype("fp32")),
    }, is_pure=True, _semantic=_semantic)


@core.extern
def ll2float_rz(arg0, _semantic=None):
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("int64"), ): ("Unsupported.ll2float_rz", core.dtype("fp32")),
    }, is_pure=True, _semantic=_semantic)


@core.extern
def ll2float_rd(arg0, _semantic=None):
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("int64"), ): ("Unsupported.ll2float_rd", core.dtype("fp32")),
    }, is_pure=True, _semantic=_semantic)


@core.extern
def ll2float_ru(arg0, _semantic=None):
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("int64"), ): ("Unsupported.ll2float_ru", core.dtype("fp32")),
    }, is_pure=True, _semantic=_semantic)


@core.extern
def ull2float_rn(arg0, _semantic=None):
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("uint64"), ): ("Unsupported.ull2float_rn", core.dtype("fp32")),
    }, is_pure=True, _semantic=_semantic)


@core.extern
def ull2float_rz(arg0, _semantic=None):
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("uint64"), ): ("Unsupported.ull2float_rz", core.dtype("fp32")),
    }, is_pure=True, _semantic=_semantic)


@core.extern
def ull2float_rd(arg0, _semantic=None):
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("uint64"), ): ("Unsupported.ull2float_rd", core.dtype("fp32")),
    }, is_pure=True, _semantic=_semantic)


@core.extern
def ull2float_ru(arg0, _semantic=None):
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("uint64"), ): ("Unsupported.ull2float_ru", core.dtype("fp32")),
    }, is_pure=True, _semantic=_semantic)


@core.extern
def ll2double_rn(arg0, _semantic=None):
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("int64"), ): ("Unsupported.ll2double_rn", core.dtype("fp64")),
    }, is_pure=True, _semantic=_semantic)


@core.extern
def ll2double_rz(arg0, _semantic=None):
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("int64"), ): ("Unsupported.ll2double_rz", core.dtype("fp64")),
    }, is_pure=True, _semantic=_semantic)


@core.extern
def ll2double_rd(arg0, _semantic=None):
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("int64"), ): ("Unsupported.ll2double_rd", core.dtype("fp64")),
    }, is_pure=True, _semantic=_semantic)


@core.extern
def ll2double_ru(arg0, _semantic=None):
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("int64"), ): ("Unsupported.ll2double_ru", core.dtype("fp64")),
    }, is_pure=True, _semantic=_semantic)


@core.extern
def ull2double_rn(arg0, _semantic=None):
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("uint64"), ): ("Unsupported.ull2double_rn", core.dtype("fp64")),
    }, is_pure=True, _semantic=_semantic)


@core.extern
def ull2double_rz(arg0, _semantic=None):
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("uint64"), ): ("Unsupported.ull2double_rz", core.dtype("fp64")),
    }, is_pure=True, _semantic=_semantic)


@core.extern
def ull2double_rd(arg0, _semantic=None):
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("uint64"), ): ("Unsupported.ull2double_rd", core.dtype("fp64")),
    }, is_pure=True, _semantic=_semantic)


@core.extern
def ull2double_ru(arg0, _semantic=None):
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("uint64"), ): ("Unsupported.ull2double_ru", core.dtype("fp64")),
    }, is_pure=True, _semantic=_semantic)


@core.extern
def int_as_float(arg0, _semantic=None):
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("int32"), ): ("Unsupported.int_as_float", core.dtype("fp32")),
    }, is_pure=True, _semantic=_semantic)


@core.extern
def float_as_int(arg0, _semantic=None):
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("fp32"), ): ("Unsupported.float_as_int", core.dtype("int32")),
    }, is_pure=True, _semantic=_semantic)


@core.extern
def uint_as_float(arg0, _semantic=None):
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("uint32"), ): ("Unsupported.uint_as_float", core.dtype("fp32")),
    }, is_pure=True, _semantic=_semantic)


@core.extern
def float_as_uint(arg0, _semantic=None):
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("fp32"), ): ("Unsupported.float_as_uint", core.dtype("int32")),
    }, is_pure=True, _semantic=_semantic)


@core.extern
def longlong_as_double(arg0, _semantic=None):
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("int64"), ): ("Unsupported.longlong_as_double", core.dtype("fp64")),
    }, is_pure=True, _semantic=_semantic)


@core.extern
def double_as_longlong(arg0, _semantic=None):
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("fp64"), ): ("Unsupported.double_as_longlong", core.dtype("int64")),
    }, is_pure=True, _semantic=_semantic)


@core.extern
def fast_sinf(arg0, _semantic=None):
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("fp32"), ): ("Unsupported.fast_sinf", core.dtype("fp32")),
    }, is_pure=True, _semantic=_semantic)


@core.extern
def fast_cosf(arg0, _semantic=None):
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("fp32"), ): ("Unsupported.fast_cosf", core.dtype("fp32")),
    }, is_pure=True, _semantic=_semantic)


@core.extern
def fast_log2f(arg0, _semantic=None):
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("fp32"), ): ("Unsupported.fast_log2f", core.dtype("fp32")),
    }, is_pure=True, _semantic=_semantic)


@core.extern
def fast_logf(arg0, _semantic=None):
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("fp32"), ): ("Unsupported.fast_logf", core.dtype("fp32")),
    }, is_pure=True, _semantic=_semantic)


@core.extern
def fast_expf(arg0, _semantic=None):
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("fp32"), ): ("libdevice.fast_expf", core.dtype("fp32")),
    }, is_pure=True, _semantic=_semantic)


@core.extern
def fast_tanf(arg0, _semantic=None):
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("fp32"), ): ("Unsupported.fast_tanf", core.dtype("fp32")),
    }, is_pure=True, _semantic=_semantic)


@core.extern
def fast_exp10f(arg0, _semantic=None):
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("fp32"), ): ("Unsupported.fast_exp10f", core.dtype("fp32")),
    }, is_pure=True, _semantic=_semantic)


@core.extern
def fast_log10f(arg0, _semantic=None):
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("fp32"), ): ("Unsupported.fast_log10f", core.dtype("fp32")),
    }, is_pure=True, _semantic=_semantic)


@core.extern
def fast_powf(arg0, arg1, _semantic=None):
    return core.extern_elementwise("", "", [arg0, arg1], {
        (core.dtype("fp32"), core.dtype("fp32")): ("Unsupported.fast_powf", core.dtype("fp32")),
    }, is_pure=True, _semantic=_semantic)


@core.extern
def hadd(arg0, arg1, _semantic=None):
    return core.extern_elementwise(
        "", "", [arg0, arg1], {
            (core.dtype("int32"), core.dtype("int32")): ("Unsupported.hadd", core.dtype("int32")),
            (core.dtype("uint32"), core.dtype("uint32")): ("Unsupported.hadd", core.dtype("uint32")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def rhadd(arg0, arg1, _semantic=None):
    return core.extern_elementwise(
        "", "", [arg0, arg1], {
            (core.dtype("int32"), core.dtype("int32")): ("Unsupported.rhadd", core.dtype("int32")),
            (core.dtype("uint32"), core.dtype("uint32")): ("Unsupported.rhadd", core.dtype("uint32")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def sub_rn(arg0, arg1, _semantic=None):
    return core.extern_elementwise(
        "", "", [arg0, arg1], {
            (core.dtype("fp32"), core.dtype("fp32")): ("Unsupported.sub_rn", core.dtype("fp32")),
            (core.dtype("fp64"), core.dtype("fp64")): ("Unsupported.sub_rn", core.dtype("fp64")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def sub_rz(arg0, arg1, _semantic=None):
    return core.extern_elementwise(
        "", "", [arg0, arg1], {
            (core.dtype("fp32"), core.dtype("fp32")): ("Unsupported.sub_rz", core.dtype("fp32")),
            (core.dtype("fp64"), core.dtype("fp64")): ("Unsupported.sub_rz", core.dtype("fp64")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def sub_rd(arg0, arg1, _semantic=None):
    return core.extern_elementwise(
        "", "", [arg0, arg1], {
            (core.dtype("fp32"), core.dtype("fp32")): ("Unsupported.sub_rd", core.dtype("fp32")),
            (core.dtype("fp64"), core.dtype("fp64")): ("Unsupported.sub_rd", core.dtype("fp64")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def sub_ru(arg0, arg1, _semantic=None):
    return core.extern_elementwise(
        "", "", [arg0, arg1], {
            (core.dtype("fp32"), core.dtype("fp32")): ("Unsupported.sub_ru", core.dtype("fp32")),
            (core.dtype("fp64"), core.dtype("fp64")): ("Unsupported.sub_ru", core.dtype("fp64")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def rsqrt_rn(arg0, _semantic=None):
    return core.extern_elementwise("", "", [
        arg0,
    ], {
        (core.dtype("fp32"), ): ("Unsupported.rsqrt_rn", core.dtype("fp32")),
    }, is_pure=True, _semantic=_semantic)


@core.extern
def ffs(arg0, _semantic=None):
    # Position of the least significant set bit (1-based); 0 for arg0 == 0.
    # xpu3 already ships _ZN3xpu3ffsEi but xpu4/xpu5 do not, so implement the
    # same branch-free cascade inline to stay architecture independent.
    x = _semantic.to_tensor(arg0)
    if x.dtype != core.dtype("int32"):
        # Key the placeholder on the operand's own type, like every other
        # fallback in this table.  A hard-coded (int32,) key looks right for
        # this branch but is the wrong key whenever it is taken: ffs on an
        # int64 tensor asks extern_elementwise for (int64,) and gets a bare
        # KeyError out of the frontend instead of the named diagnostic.
        return core.extern_elementwise("", "", [x], {
            (x.dtype, ): ("Unsupported.ffs", core.dtype("int32")),
        }, is_pure=True, _semantic=_semantic)
    y = x
    pos = _semantic.full(x.shape, 1, x.dtype)
    for width, mask in ((16, 0xFFFF), (8, 0xFF), (4, 0xF), (2, 0x3), (1, 0x1)):
        miss = _semantic.equal(_semantic.and_(y, mask), 0)
        y = _semantic.where(miss, _semantic.ashr(y, width), y)
        pos = _semantic.where(miss, _semantic.add(pos, width, sanitize_overflow=False), pos)
    return _semantic.where(_semantic.equal(x, 0), _semantic.full(x.shape, 0, x.dtype), pos)


@core.extern
def rint(arg0, _semantic=None):
    return core.extern_elementwise(
        "", "", [
            arg0,
        ], {
            (core.dtype("fp32"), ): ("_ZN3xpu4rintEf", core.dtype("fp32")),
            (core.dtype("fp64"), ): ("Unsupported.rint", core.dtype("fp64")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def llrint(arg0, _semantic=None):
    return core.extern_elementwise(
        "", "", [
            arg0,
        ], {
            (core.dtype("fp32"), ): ("Unsupported.llrint", core.dtype("int64")),
            (core.dtype("fp64"), ): ("Unsupported.llrint", core.dtype("int64")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def nearbyint(arg0, _semantic=None):
    # Round to nearest, ties to even (C's current rounding mode), which is what
    # PyTorch's quantize_per_channel relies on for its value*inv_scale step.
    #
    # fp32 keeps the device symbol: it is exact there and one call.  fp64 has
    # no device implementation (`libdevice-xpu{3,4,5}.bc` carries no fp64
    # transcendental symbols at all), so it is spelled out in fp64 Triton
    # arithmetic instead.  That is exact rounding rather than a polynomial
    # approximation of it, so fp64 does not lose precision the way routing an
    # fp64 call through the fp32 polynomials would.
    x = _semantic.to_tensor(arg0)
    if x.dtype == core.dtype("fp64"):
        # No fp64 `rint`/`floor` on this device, and a select-chain spelling
        # bloats every kernel that calls this (pow, sin/cos) enough to make
        # canonicalize take minutes.  `_f64_round` gets ties-to-even from the
        # hardware by adding a signed 2**52 and subtracting it back.
        fp64 = core.dtype("fp64")
        i64 = core.dtype("int64")
        zero = _semantic.full(x.shape, 0.0, fp64)
        result = _f64_round(x, _semantic)
        # The add loses the sign of a zero result (nearbyint(-0.5) is -0.0,
        # not +0.0).  Take the sign bit from x by bitcast; `x * 0.0` folds to
        # +0.0 under the backend's fast-math flags.  NaN / zero inputs pass
        # straight through.
        sign_bits = _semantic.and_(_semantic.bitcast(x, i64), _semantic.full(x.shape, -(1 << 63), i64))
        signed_zero = _semantic.bitcast(sign_bits, fp64)
        result = _semantic.where(_semantic.equal(result, zero), signed_zero, result)
        is_nan = _semantic.not_equal(x, x)
        is_zero = _semantic.equal(x, zero)
        return _semantic.where(is_nan, x, _semantic.where(is_zero, x, result))
    if x.dtype != _FP32:
        return core.extern_elementwise("", "", [x], {
            tuple(a.dtype for a in [x]): ("Unsupported.nearbyint", x.dtype),
        }, is_pure=True, _semantic=_semantic)
    return core.extern_elementwise("", "", [
        x,
    ], {
        (_FP32, ): ("_ZN3xpu9nearbyintEf", _FP32),
    }, is_pure=True, _semantic=_semantic)


@core.extern
def isnan(arg0, _semantic=None):
    return core.extern_elementwise(
        "", "", [
            arg0,
        ], {
            (core.dtype("fp16"), ): ("_ZN3xpu6hisnanEDF16_", core.dtype("int32")),
            (core.dtype("fp32"), ): ("_ZN3xpu5isnanEf", core.dtype("int32")),
            (core.dtype("fp64"), ): ("Unsupported.isnan", core.dtype("int32")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def signbit(arg0, _semantic=None):
    return core.extern_elementwise(
        "", "", [
            arg0,
        ], {
            (core.dtype("fp32"), ): ("_ZN3xpu10__signbitfEf", core.dtype("int32")),
            (core.dtype("fp64"), ): ("Unsupported.signbit", core.dtype("int32")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def copysign(arg0, arg1, _semantic=None):
    # |arg0| with the sign bit of arg1, including the +-0 / NaN / inf cases.
    x = _semantic.to_tensor(arg0)
    if x.dtype != _FP32:
        return core.extern_elementwise("", "", [x, _semantic.to_tensor(arg1)], {
            tuple(a.dtype for a in [x, _semantic.to_tensor(arg1)]): (f"Unsupported.copysign", _FP32),
        }, is_pure=True, _semantic=_semantic)
    ua = _semantic.bitcast(x, core.dtype("uint32"))
    ub = _semantic.bitcast(_semantic.to_tensor(arg1), core.dtype("uint32"))
    r = _semantic.xor_(_semantic.and_(ua, 0x7FFFFFFF), _semantic.and_(ub, 0x80000000))
    return _semantic.bitcast(r, _FP32)


@core.extern
def finitef(arg0, _semantic=None):
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp16"), ): ("_ZN3xpu7hfiniteEDF16_", core.dtype("int16")),
            (core.dtype("fp32"), ): ("_ZN3xpu7finitefEf", core.dtype("int32")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def isinf(arg0, _semantic=None):
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp16"), ): ("_ZN3xpu4hsinEDF16_", core.dtype("int32")),
            (core.dtype("fp32"), ): ("_ZN3xpu5isinfEf", core.dtype("int32")),
            (core.dtype("fp64"), ): ("Unsupported.isinf", core.dtype("int32")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def nextafter(arg0, arg1, _semantic=None):
    # IEEE 754 nextafter on the fp32 bit pattern. A NaN operand returns that
    # operand's bits; equal operands return arg0; the +0/-0 crossing uses the
    # 0x80000001 constant. Verified bit-exact against np.nextafter.
    x = _semantic.to_tensor(arg0)
    y = _semantic.to_tensor(arg1)
    if x.dtype != _FP32:
        return core.extern_elementwise("", "", [x, y], {
            tuple(a.dtype for a in [x, y]): (f"Unsupported.nextafter", _FP32),
        }, is_pure=True, _semantic=_semantic)
    xi = _semantic.bitcast(x, core.dtype("int32"))
    yi = _semantic.bitcast(y, core.dtype("int32"))
    exp_mask, frac_mask = 2139095040, 8388607  # 0x7F800000 / 0x007FFFFF
    sign_bit, cross = -2147483648, -2147483647  # 0x80000000 / 0x80000001
    xnan = _semantic.logical_and(_semantic.equal(_semantic.and_(xi, exp_mask), exp_mask),
                                 _semantic.not_equal(_semantic.and_(xi, frac_mask), 0))
    ynan = _semantic.logical_and(_semantic.equal(_semantic.and_(yi, exp_mask), exp_mask),
                                 _semantic.not_equal(_semantic.and_(yi, frac_mask), 0))
    # IEEE 754: x == y returns y. The comparison must be on values, not bit
    # patterns, so that nextafter(+0, -0) yields -0 instead of a subnormal.
    is_equal = _semantic.equal(x, y)
    is_positive = _semantic.cast(_semantic.equal(_semantic.and_(xi, sign_bit), 0), core.dtype("int32"))
    x_is_zero = _semantic.cast(_semantic.equal(xi, 0), core.dtype("int32"))
    x_is_zm = _semantic.cast(_semantic.equal(xi, sign_bit), core.dtype("int32"))
    kx = _semantic.where(_semantic.less_than(xi, 0), _semantic.xor_(_semantic.invert(xi), sign_bit), xi)
    ky = _semantic.where(_semantic.less_than(yi, 0), _semantic.xor_(_semantic.invert(yi), sign_bit), yi)
    is_going_up = _semantic.cast(_semantic.less_than(kx, ky), core.dtype("int32"))
    inc = _semantic.mul(
        _semantic.sub(_semantic.mul(is_going_up, 2, sanitize_overflow=False), 1, sanitize_overflow=False),
        _semantic.sub(_semantic.mul(is_positive, 2, sanitize_overflow=False), 1, sanitize_overflow=False),
        sanitize_overflow=False)
    zc = _semantic.add(
        _semantic.mul(
            _semantic.mul(is_positive, _semantic.sub(1, is_going_up, sanitize_overflow=False), sanitize_overflow=False),
            x_is_zero, sanitize_overflow=False),
        _semantic.mul(
            _semantic.mul(_semantic.sub(1, is_positive, sanitize_overflow=False), is_going_up, sanitize_overflow=False),
            x_is_zm, sanitize_overflow=False), sanitize_overflow=False)
    step = _semantic.where(_semantic.equal(zc, 1), _semantic.add(xi, cross, sanitize_overflow=False),
                           _semantic.add(xi, inc, sanitize_overflow=False))
    r = _semantic.where(is_equal, yi, step)
    r = _semantic.where(ynan, yi, r)
    r = _semantic.where(xnan, xi, r)
    return _semantic.bitcast(r, _FP32)


# ---------------------------------------------------------------------------
# fp64 elementary functions.
#
# The device library carries no fp64 transcendental at all, and its one fp64
# helper that looks usable, `xpu_floor(double)`, is `return floor(input)` --
# an external `floor` the XPU device link does not provide.  So fp64 sin / cos /
# atan2 / pow are built here from fp64 arithmetic, comparisons, int64 casts and
# `ldexp`, all of which the backend does emit.
#
# Everything is written as a single expression tree per function so the
# argument reductions stay branch-free.  Accuracy is that of the polynomials
# below over the reduced ranges; the reductions themselves assume the caller is
# not passing arguments whose magnitude already destroys the reduction (beyond
# |x| ~ 2**20 for sin/cos the Cody-Waite split runs out of bits, which is the
# same limitation every two-term reduction has).
# ---------------------------------------------------------------------------

_FP64 = core.dtype("fp64")
_I64 = core.dtype("int64")


def _f64(value, like, _semantic):
    return _semantic.full(like.shape, value, _FP64)


def _f64_round(x, _semantic):
    """Round an fp64 to the nearest integer, ties to even.

    x + copysign(2**52, x) lands in the fp64 binade [2**52, 2**53) whose ulp
    is exactly 1 for |x| < 2**52, so the add rounds x to an integer with the
    hardware's ties-to-even mode; subtracting the same constant back is
    exact (a single 1.5*2**52 magic only covers |x| <= 2**51 and would
    mis-round the half-integers in [2**51, 2**52)).  One add, one subtract
    and one sign select -- a select chain here made pow/atan2/sin/cos
    kernels take minutes in canonicalize.  |x| >= 2**52 is already integral
    and passes through.
    """
    zero = _f64(0.0, x, _semantic)
    magic = _f64(4503599627370496.0, x, _semantic)  # 2**52
    neg_magic = _semantic.sub(zero, magic, sanitize_overflow=False)
    shift = _semantic.where(_semantic.less_than(x, zero), neg_magic, magic)
    rounded = _semantic.sub(_semantic.add(x, shift, sanitize_overflow=False), shift, sanitize_overflow=False)
    outside = _semantic.logical_or(_semantic.greater_equal(x, magic), _semantic.less_equal(x, neg_magic))
    return _semantic.where(outside, x, rounded)


def _f64_horner(t2, coeffs, start, _semantic):
    """Evaluate start + t2*(c0 + t2*(c1 + ...)) in fp64."""
    acc = _f64(coeffs[-1], t2, _semantic)
    for c in reversed(coeffs[:-1]):
        acc = _semantic.add(_semantic.mul(acc, t2, sanitize_overflow=False), _f64(c, t2, _semantic),
                            sanitize_overflow=False)
    return _semantic.add(_semantic.mul(acc, t2, sanitize_overflow=False), _f64(start, t2, _semantic),
                         sanitize_overflow=False)


def _f64_sin_cos(x, _semantic):
    """(sin x, cos x) by reducing x to [-pi/4, pi/4] and re-selecting by quadrant."""
    inv_pio2 = _semantic.mul(x, _f64(0.63661977236758134308, x, _semantic), sanitize_overflow=False)
    k = _f64_round(inv_pio2, _semantic)
    # Cody-Waite: subtract k*(pi/2) in two pieces so each product stays exact.
    r = _semantic.sub(x, _semantic.mul(k, _f64(1.5707963267948966, x, _semantic), sanitize_overflow=False),
                      sanitize_overflow=False)
    r = _semantic.sub(r, _semantic.mul(k, _f64(6.123233995736766e-17, x, _semantic), sanitize_overflow=False),
                      sanitize_overflow=False)
    r = _semantic.sub(r, _semantic.mul(k, _f64(-1.4973849048591698e-33, x, _semantic), sanitize_overflow=False),
                      sanitize_overflow=False)
    r2 = _semantic.mul(r, r, sanitize_overflow=False)
    # sin(r) = r*(1 + r2*c0 + r2^2*c1 + ...), cos(r) = 1 + r2*d0 + ...
    s_poly = _semantic.mul(
        r,
        _f64_horner(r2, [
            -1.6666666666666666e-01, 8.3333333333333333e-03, -1.9841269841269841e-04, 2.7557319223985891e-06,
            -2.5052108385441719e-08, 1.6059043836821613e-10, -7.6471637318198165e-13, 2.8114572543455208e-15
        ], 1.0, _semantic), sanitize_overflow=False)
    c_poly = _f64_horner(r2, [
        -5.0000000000000000e-01, 4.1666666666666667e-02, -1.3888888888888889e-03, 2.4801587301587302e-05,
        -2.7557319223985891e-07, 2.0876756987868099e-09, -1.1470745597729725e-11, 4.7794773323873853e-14
    ], 1.0, _semantic)
    q = _semantic.and_(_semantic.cast(k, _I64), 3)
    # quadrant 0: (s, c)  1: (c, -s)  2: (-s, -c)  3: (-c, s)
    neg_s = _semantic.sub(_f64(0.0, x, _semantic), s_poly, sanitize_overflow=False)
    neg_c = _semantic.sub(_f64(0.0, x, _semantic), c_poly, sanitize_overflow=False)
    sin_x = _semantic.where(
        _semantic.equal(q, 0), s_poly,
        _semantic.where(_semantic.equal(q, 1), c_poly, _semantic.where(_semantic.equal(q, 2), neg_s, neg_c)))
    cos_x = _semantic.where(
        _semantic.equal(q, 0), c_poly,
        _semantic.where(_semantic.equal(q, 1), neg_s, _semantic.where(_semantic.equal(q, 2), neg_c, s_poly)))
    return sin_x, cos_x


def _f64_log2(x, _semantic):
    """log2(x) for x > 0, from the exponent field plus an atanh series."""
    one = _f64(1.0, x, _semantic)
    bits = _semantic.bitcast(x, _I64)
    # Biased exponent minus the bias, and the mantissa renormalised into [1, 2)
    # by overwriting the exponent field with the bias.  Subnormal inputs leave
    # the field at 0, so their log2 loses accuracy; that is the same trade every
    # field-extraction log makes.
    e = _semantic.sub(_semantic.and_(_semantic.ashr(bits, 52), _semantic.full(x.shape, 0x7FF, _I64)),
                      _semantic.full(x.shape, 1023, _I64), sanitize_overflow=False)
    mant_bits = _semantic.or_(_semantic.and_(bits, _semantic.full(x.shape, 0x000FFFFFFFFFFFFF, _I64)),
                              _semantic.full(x.shape, 0x3FF0000000000000, _I64))
    m = _semantic.bitcast(mant_bits, _FP64)
    # m in [1, 2); t = (m-1)/(m+1) lies in [0, 1/3] and
    # log(m) = 2*(t + t^3/3 + t^5/5 + ...).
    num = _semantic.sub(m, one, sanitize_overflow=False)
    den = _semantic.add(m, one, sanitize_overflow=False)
    t = _semantic.truediv(num, den)
    t2 = _semantic.mul(t, t, sanitize_overflow=False)
    series = _f64_horner(t2, [
        3.3333333333333333e-01, 2.0000000000000000e-01, 1.4285714285714286e-01, 1.1111111111111111e-01,
        9.0909090909090909e-02, 7.6923076923076923e-02, 6.6666666666666667e-02, 5.8823529411764706e-02,
        5.2631578947368421e-02, 4.7619047619047619e-02, 4.3478260869565217e-02, 4.0000000000000000e-02,
        3.7037037037037037e-02, 3.4482758620689655e-02, 3.2258064516129032e-02, 3.0303030303030303e-02,
        2.8571428571428571e-02, 2.7027027027027027e-02, 2.5641025641025641e-02, 2.4390243902439024e-02
    ], 1.0, _semantic)
    log_m = _semantic.mul(_semantic.mul(t, _f64(2.0, x, _semantic), sanitize_overflow=False), series,
                          sanitize_overflow=False)
    log2_m = _semantic.mul(log_m, _f64(1.4426950408889634074, x, _semantic), sanitize_overflow=False)
    return _semantic.add(_semantic.cast(e, _FP64), log2_m, sanitize_overflow=False)


def _f64_exp2(z, _semantic):
    """2**z as a polynomial on the fractional part times an exact ldexp."""
    zero = _f64(0.0, z, _semantic)
    one = _f64(1.0, z, _semantic)
    pos_inf = _semantic.truediv(one, zero)
    n = _f64_round(z, _semantic)
    f = _semantic.sub(z, n, sanitize_overflow=False)  # in [-0.5, 0.5]
    # 2**f = exp(f*ln2), Taylor to f**12.
    u = _semantic.mul(f, _f64(0.69314718055994530942, z, _semantic), sanitize_overflow=False)
    acc = _f64(1.0 / 479001600.0, z, _semantic)
    for c in (1.0 / 39916800.0, 1.0 / 3628800.0, 1.0 / 362880.0, 1.0 / 40320.0, 1.0 / 5040.0, 1.0 / 720.0, 1.0 / 120.0,
              1.0 / 24.0, 1.0 / 6.0, 1.0 / 2.0, 1.0, 1.0):
        acc = _semantic.add(_semantic.mul(acc, u, sanitize_overflow=False), _f64(c, z, _semantic),
                            sanitize_overflow=False)
    # ldexp takes an int32 exponent: fptosi of NaN or of a value outside
    # int32 is poison, which is exactly what pow(x, +-inf) and pow(x, huge)
    # feed here (inf - inf = NaN for the fractional part).  Clamp before the
    # cast, then put the non-finite / out-of-range answers back from z.
    lo = _f64(-2147483648.0, z, _semantic)
    hi = _f64(2147483647.0, z, _semantic)
    n_safe = _semantic.where(_semantic.not_equal(n, n), zero, n)
    n_safe = _semantic.where(_semantic.greater_than(n_safe, hi), hi, n_safe)
    n_safe = _semantic.where(_semantic.less_than(n_safe, lo), lo, n_safe)
    result = ldexp(acc, _semantic.cast(n_safe, core.dtype("int32")), _semantic=_semantic)
    result = _semantic.where(_semantic.greater_than(z, hi), pos_inf, result)
    result = _semantic.where(_semantic.less_than(z, lo), zero, result)
    return _semantic.where(_semantic.not_equal(z, z), z, result)


def _f64_atan(x, _semantic):
    """atan on fp64: |x| halved into [0, tan(pi/12)] by two pi/6 steps."""
    sqrt3 = _f64(1.7320508075688772935, x, _semantic)
    one = _f64(1.0, x, _semantic)
    zero = _f64(0.0, x, _semantic)
    neg = _semantic.less_than(x, zero)
    a = _semantic.where(neg, _semantic.sub(zero, x, sanitize_overflow=False), x)
    # Bring |x| into [0, 1] first; two pi/6 steps then reach [0, tan(pi/12)].
    # Without this, |x| > 1 needs one pi/6 step per factor of ~3.7 and two are
    # nowhere near enough.
    one = _semantic.full(x.shape, 1.0, _FP64)
    half_pi = _f64(1.57079632679489661923, x, _semantic)
    invert = _semantic.greater_than(a, one)
    a = _semantic.where(invert, _semantic.truediv(one, a), a)
    tan_pi_12 = _f64(0.26794919243112270647, x, _semantic)
    pi_6 = _f64(0.52359877559829887308, x, _semantic)
    shift = zero
    for _ in range(2):
        hit = _semantic.greater_than(a, tan_pi_12)
        num = _semantic.sub(_semantic.mul(a, sqrt3, sanitize_overflow=False), one, sanitize_overflow=False)
        den = _semantic.add(sqrt3, a, sanitize_overflow=False)
        a = _semantic.where(hit, _semantic.truediv(num, den), a)
        shift = _semantic.where(hit, _semantic.add(shift, pi_6, sanitize_overflow=False), shift)
    # |a| <= tan(pi/12); the odd Taylor series converges to fp64 there.
    a2 = _semantic.mul(a, a, sanitize_overflow=False)
    # atan(a) = a*(1 - a^2/3 + a^4/5 - ...), so the Horner coefficients start
    # at -1/3 and alternate.
    inv = [-1.0 / 3.0]
    sign = 1.0
    for n in range(5, 28, 2):
        inv.append(sign / n)
        sign = -sign
    series = _f64_horner(a2, inv, 1.0, _semantic)
    atan_a = _semantic.mul(a, series, sanitize_overflow=False)
    out = _semantic.add(shift, atan_a, sanitize_overflow=False)
    out = _semantic.where(invert, _semantic.sub(half_pi, out, sanitize_overflow=False), out)
    return _semantic.where(neg, _semantic.sub(zero, out, sanitize_overflow=False), out)


@core.extern
def sin(arg0, _semantic=None):
    x = _semantic.to_tensor(arg0)
    if x.dtype == _FP64:
        return _f64_sin_cos(x, _semantic)[0]
    if x.dtype != _FP32:
        return core.extern_elementwise("", "", [x], {
            tuple(a.dtype for a in [x]): ("Unsupported.sin", x.dtype),
        }, is_pure=True, _semantic=_semantic)
    return core.extern_elementwise("", "", [x], {
        (_FP32, ): ("_ZN3xpu4sinfEf", _FP32),
    }, is_pure=True, _semantic=_semantic)


@core.extern
def cos(arg0, _semantic=None):
    x = _semantic.to_tensor(arg0)
    if x.dtype == _FP64:
        return _f64_sin_cos(x, _semantic)[1]
    if x.dtype != _FP32:
        return core.extern_elementwise("", "", [x], {
            tuple(a.dtype for a in [x]): ("Unsupported.cos", x.dtype),
        }, is_pure=True, _semantic=_semantic)
    return core.extern_elementwise("", "", [x], {
        (_FP32, ): ("_ZN3xpu4cosfEf", _FP32),
    }, is_pure=True, _semantic=_semantic)


@core.extern
def sinpi(arg0, _semantic=None):
    # sin(pi * x) evaluated on the distance to the nearest integer. This keeps
    # integer arguments exactly 0 (sin(pi*x) would give ~1e-6 there) and lets
    # the |pi*r| < 0.5 region use a Taylor series, which is both faster and
    # dodges a backend defect where tl.sin returns 0 for tiny arguments on
    # narrow tiles (recorded in FlagGems _kunlunxin/sinc.py).
    x = _semantic.to_tensor(arg0)
    if x.dtype != _FP32:
        return core.extern_elementwise("", "", [x], {
            tuple(a.dtype for a in [x]): (f"Unsupported.sinpi", _FP32),
        }, is_pure=True, _semantic=_semantic)
    half = _semantic.full(x.shape, 0.5, _FP32)
    n = _math.floor(_semantic.add(x, half, sanitize_overflow=False), _semantic=_semantic)
    r = _semantic.sub(x, n, sanitize_overflow=False)
    odd = _semantic.sub(
        n,
        _semantic.mul(_math.floor(_semantic.mul(n, half, sanitize_overflow=False), _semantic=_semantic),
                      _semantic.full(x.shape, 2.0, _FP32), sanitize_overflow=False), sanitize_overflow=False)
    p = _semantic.mul(_semantic.full(x.shape, 3.141592653589793, _FP32), r, sanitize_overflow=False)
    p2 = _semantic.mul(p, p, sanitize_overflow=False)
    _sp = _semantic.full(p2.shape, -2.505210838544172e-08, p2.dtype)
    _sp = _semantic.add(_semantic.mul(_sp, p2, sanitize_overflow=False), 2.7557319223985893e-06,
                        sanitize_overflow=False)
    _sp = _semantic.add(_semantic.mul(_sp, p2, sanitize_overflow=False), -0.0001984126984126984,
                        sanitize_overflow=False)
    _sp = _semantic.add(_semantic.mul(_sp, p2, sanitize_overflow=False), 0.008333333333333333, sanitize_overflow=False)
    _sp = _semantic.add(_semantic.mul(_sp, p2, sanitize_overflow=False), -0.16666666666666666, sanitize_overflow=False)
    _sp = _semantic.add(_semantic.mul(_sp, p2, sanitize_overflow=False), 1.0, sanitize_overflow=False)
    series = _semantic.mul(p, _sp, sanitize_overflow=False)
    s = _semantic.where(_semantic.less_than(p2, _semantic.full(x.shape, 0.25, _FP32)), series,
                        _math.sin(p, _semantic=_semantic))
    return _semantic.where(_semantic.equal(odd, 0), s, _semantic.minus(s))


@core.extern
def cospi(arg0, _semantic=None):
    # cos(pi * x); same nearest-integer reduction as sinpi. The series branch
    # keeps half-integer arguments exact (cos(pi/2) = 0 instead of ~1e-8).
    x = _semantic.to_tensor(arg0)
    if x.dtype != _FP32:
        return core.extern_elementwise("", "", [x], {
            tuple(a.dtype for a in [x]): (f"Unsupported.cospi", _FP32),
        }, is_pure=True, _semantic=_semantic)
    half = _semantic.full(x.shape, 0.5, _FP32)
    n = _math.floor(_semantic.add(x, half, sanitize_overflow=False), _semantic=_semantic)
    r = _semantic.sub(x, n, sanitize_overflow=False)
    odd = _semantic.sub(
        n,
        _semantic.mul(_math.floor(_semantic.mul(n, half, sanitize_overflow=False), _semantic=_semantic),
                      _semantic.full(x.shape, 2.0, _FP32), sanitize_overflow=False), sanitize_overflow=False)
    p = _semantic.mul(_semantic.full(x.shape, 3.141592653589793, _FP32), r, sanitize_overflow=False)
    p2 = _semantic.mul(p, p, sanitize_overflow=False)
    _cp = _semantic.full(p2.shape, -2.755731922398589e-07, p2.dtype)
    _cp = _semantic.add(_semantic.mul(_cp, p2, sanitize_overflow=False), 2.48015873015873e-05, sanitize_overflow=False)
    _cp = _semantic.add(_semantic.mul(_cp, p2, sanitize_overflow=False), -0.001388888888888889, sanitize_overflow=False)
    _cp = _semantic.add(_semantic.mul(_cp, p2, sanitize_overflow=False), 0.041666666666666664, sanitize_overflow=False)
    _cp = _semantic.add(_semantic.mul(_cp, p2, sanitize_overflow=False), -0.5, sanitize_overflow=False)
    _cp = _semantic.add(_semantic.mul(_cp, p2, sanitize_overflow=False), 1.0, sanitize_overflow=False)
    c = _semantic.where(_semantic.less_than(p2, _semantic.full(x.shape, 0.25, _FP32)), _cp,
                        _math.cos(p, _semantic=_semantic))
    return _semantic.where(_semantic.equal(odd, 0), c, _semantic.minus(c))


@core.extern
def tan(arg0, _semantic=None):
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp32"), ): ("_ZN3xpu4tanfEf", core.dtype("fp32")),
            (core.dtype("fp64"), ): ("Unsupported.tan", core.dtype("fp64")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def log2(arg0, _semantic=None):
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp16"), ): ("_ZN3xpu5hlog2EDF16_", core.dtype("fp16")),
            (core.dtype("fp32"), ): ("_ZN3xpu5log2fEf", core.dtype("fp32")),
            (core.dtype("fp64"), ): ("Unsupported.log2", core.dtype("fp64")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def exp(arg0, _semantic=None):
    arg0 = _semantic.to_tensor(arg0)
    if arg0.dtype == core.dtype("fp32"):
        return _math.exp(arg0, _semantic=_semantic)
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("fp64"), ): ("Unsupported.exp", core.dtype("fp64")),
    }, is_pure=True, _semantic=_semantic)


@core.extern
def exp10(arg0, _semantic=None):
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp32"), ): ("Unsupported.exp10", core.dtype("fp32")),
            (core.dtype("fp64"), ): ("Unsupported.exp10", core.dtype("fp64")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def cosh(arg0, _semantic=None):
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp32"), ): ("_ZN3xpu5coshfEf", core.dtype("fp32")),
            (core.dtype("fp64"), ): ("Unsupported.cosh", core.dtype("fp64")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def sinh(arg0, _semantic=None):
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp32"), ): ("_ZN3xpu5sinhfEf", core.dtype("fp32")),
            (core.dtype("fp64"), ): ("Unsupported.sinh", core.dtype("fp64")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def tanh(arg0, _semantic=None):
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp16"), ): ("_ZN3xpu5htanhEDF16_", core.dtype("fp16")),
            (core.dtype("fp32"), ): ("_ZN3xpu5tanhfEf", core.dtype("fp32")),
            (core.dtype("fp64"), ): ("Unsupported.tanh", core.dtype("fp64")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def atan2(arg0, arg1, _semantic=None):
    y = _semantic.to_tensor(arg0)
    x = _semantic.to_tensor(arg1)
    if x.dtype == _FP64 or y.dtype == _FP64:
        y = y.to(_FP64, _semantic=_semantic)
        x = x.to(_FP64, _semantic=_semantic)
        # atan2(y, x) = atan(y/x) with the half-plane fixed up, and the two
        # axes handled separately because y/x is 0 or infinite there.  Both
        # axes distinguish +0.0 from -0.0 (atan2(-0.0, x<0) is -pi, not +pi),
        # so the sign comes from the bit pattern: `>= 0` cannot tell them apart.
        zero = _f64(0.0, y, _semantic)
        one = _f64(1.0, y, _semantic)
        pos_inf = _semantic.truediv(one, zero)
        pi = _f64(3.14159265358979323846, y, _semantic)
        half_pi = _f64(1.57079632679489661923, y, _semantic)
        neg_pi = _semantic.sub(zero, pi, sanitize_overflow=False)
        neg_half_pi = _semantic.sub(zero, half_pi, sanitize_overflow=False)
        sign_mask = _semantic.full(y.shape, -(1 << 63), _I64)
        zero_i64 = _semantic.full(y.shape, 0, _I64)
        y_neg = _semantic.less_than(_semantic.bitcast(y, _I64), zero_i64)
        x_neg = _semantic.less_than(_semantic.bitcast(x, _I64), zero_i64)
        # y/x is NaN when both are infinite, but atan2(+-inf, +-inf) is
        # +-pi/4 or +-3pi/4; handle that pair from the sign bits.
        abs_y = _semantic.where(y_neg, _semantic.sub(zero, y, sanitize_overflow=False), y)
        abs_x = _semantic.where(x_neg, _semantic.sub(zero, x, sanitize_overflow=False), x)
        both_inf = _semantic.logical_and(_semantic.equal(abs_y, pos_inf), _semantic.equal(abs_x, pos_inf))
        ratio = _semantic.truediv(y, x)
        base = _f64_atan(ratio, _semantic)
        neg_side = _semantic.logical_or(_semantic.less_than(x, zero), x_neg)
        base = _semantic.where(neg_side, _semantic.add(base, _semantic.where(y_neg, neg_pi, pi),
                                                       sanitize_overflow=False), base)
        on_axis = _semantic.equal(x, zero)
        # A zero result must keep y's sign (atan2(+-0.0, +0.0) is +-0.0).
        y_signed_zero = _semantic.bitcast(_semantic.and_(_semantic.bitcast(y, _I64), sign_mask), _FP64)
        axis = _semantic.where(
            _semantic.greater_than(y, zero), half_pi,
            _semantic.where(_semantic.less_than(y, zero), neg_half_pi,
                            _semantic.where(x_neg, _semantic.where(y_neg, neg_pi, pi), y_signed_zero)))
        result = _semantic.where(on_axis, axis, base)
        # atan2(+-inf, +-inf): pi/4 or 3pi/4 by the x sign, negated by the y
        # sign.  Inserted after the axis fix-up so it overrides the NaN ratio.
        quarter_pi = _semantic.mul(pi, _f64(0.25, y, _semantic), sanitize_overflow=False)
        inf_angle = _semantic.where(x_neg, _semantic.mul(quarter_pi, _f64(3.0, y, _semantic), sanitize_overflow=False),
                                    quarter_pi)
        inf_angle = _semantic.where(y_neg, _semantic.sub(zero, inf_angle, sanitize_overflow=False), inf_angle)
        result = _semantic.where(both_inf, inf_angle, result)
        # atan2(+-0.0, +x) is +-0.0: _f64_atan works on |y/x| and loses the
        # sign of a zero result, so restore it from y.
        result = _semantic.where(_semantic.equal(result, zero), y_signed_zero, result)
        return _semantic.where(_semantic.not_equal(y, y), y, result)
    if x.dtype != _FP32 or y.dtype != _FP32:
        return core.extern_elementwise("", "", [y, x], {
            tuple(a.dtype for a in [y, x]): ("Unsupported.atan2", x.dtype),
        }, is_pure=True, _semantic=_semantic)
    return core.extern_elementwise("", "", [y, x], {
        (_FP32, _FP32): ("_ZN3xpu6atan2fEff", _FP32),
    }, is_pure=True, _semantic=_semantic)


@core.extern
def atan(arg0, _semantic=None):
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp32"), ): ("_ZN3xpu5atanfEf", core.dtype("fp32")),
            (core.dtype("fp64"), ): ("Unsupported.atan", core.dtype("fp64")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def asin(arg0, _semantic=None):
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp32"), ): ("_ZN3xpu5asinfEf", core.dtype("fp32")),
            (core.dtype("fp64"), ): ("Unsupported.asin", core.dtype("fp64")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def acos(arg0, _semantic=None):
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp32"), ): ("_ZN3xpu5acosfEf", core.dtype("fp32")),
            (core.dtype("fp64"), ): ("Unsupported.acos", core.dtype("fp64")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def log(arg0, _semantic=None):
    arg0 = _semantic.to_tensor(arg0)
    if arg0.dtype == core.dtype("fp32"):
        return _math.log(arg0, _semantic=_semantic)
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("fp64"), ): ("Unsupported.log", core.dtype("fp64")),
    }, is_pure=True, _semantic=_semantic)


@core.extern
def log10(arg0, _semantic=None):
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp32"), ): ("_ZN3xpu6log10fEf", core.dtype("fp32")),
            (core.dtype("fp64"), ): ("Unsupported.log10", core.dtype("fp64")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def log1p(arg0, _semantic=None):
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp32"), ): ("_ZN3xpu6log1pfEf", core.dtype("fp32")),
            (core.dtype("fp64"), ): ("Unsupported.log1p", core.dtype("fp64")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def acosh(arg0, _semantic=None):
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp32"), ): ("_ZN3xpu6acoshfEf", core.dtype("fp32")),
            (core.dtype("fp64"), ): ("Unsupported.acosh", core.dtype("fp64")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def asinh(arg0, _semantic=None):
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp32"), ): ("_ZN3xpu6asinhfEf", core.dtype("fp32")),
            (core.dtype("fp64"), ): ("Unsupported.asinh", core.dtype("fp64")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def atanh(arg0, _semantic=None):
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp32"), ): ("_ZN3xpu6atanhfEf", core.dtype("fp32")),
            (core.dtype("fp64"), ): ("Unsupported.atanh", core.dtype("fp64")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def expm1(arg0, _semantic=None):
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp32"), ): ("_ZN3xpu6expm1fEf", core.dtype("fp32")),
            (core.dtype("fp64"), ): ("Unsupported.expm1", core.dtype("fp64")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def hypot(arg0, arg1, _semantic=None):
    # sqrt(x*x + y*y) computed on max(|x|,|y|) * sqrt(1 + (min/max)**2) so the
    # intermediate product cannot overflow/underflow. A subnormal max falls
    # back to the max itself, bounding that unavoidable case by sqrt(2).
    x = _semantic.to_tensor(arg0)
    y = _semantic.to_tensor(arg1)
    if x.dtype != _FP32:
        return core.extern_elementwise("", "", [x, y], {
            tuple(a.dtype for a in [x, y]): (f"Unsupported.hypot", _FP32),
        }, is_pure=True, _semantic=_semantic)
    ax = _math.abs(x, _semantic=_semantic)
    ay = _math.abs(y, _semantic=_semantic)
    hi = _semantic.maximum(ax, ay, core.PropagateNan.NONE)
    lo = _semantic.minimum(ax, ay, core.PropagateNan.NONE)
    nz = _semantic.greater_than(hi, 0)
    safe = _semantic.where(nz, hi, _semantic.full(hi.shape, 1.0, hi.dtype))
    rr = _semantic.truediv(lo, safe)
    one = _semantic.full(hi.shape, 1.0, hi.dtype)
    res = _semantic.mul(
        hi,
        _math.sqrt(_semantic.add(one, _semantic.mul(rr, rr, sanitize_overflow=False), sanitize_overflow=False),
                   _semantic=_semantic), sanitize_overflow=False)
    res = _semantic.where(nz, res, hi)
    has_nan = _semantic.logical_or(_semantic.not_equal(x, x), _semantic.not_equal(y, y))
    has_inf = _semantic.equal(hi, float("inf"))
    # Order matters: C99 Annex F.10.4.3 specifies hypot(+-inf, y) as +inf "even
    # if y is a NaN", so the inf branch has to be applied last. hi comes from
    # maxnumf, which returns the non-NaN operand, so an (inf, NaN) pair still
    # sets has_inf. Verified against glibc hypotf and numpy.hypot.
    res = _semantic.where(has_nan, _semantic.full(hi.shape, float("nan"), hi.dtype), res)
    res = _semantic.where(has_inf, _semantic.full(hi.shape, float("inf"), hi.dtype), res)
    return res


@core.extern
def rhypot(arg0, arg1, _semantic=None):
    # 1 / hypot(x, y); the reciprocal of the overflow-safe form above.
    x = _semantic.to_tensor(arg0)
    y = _semantic.to_tensor(arg1)
    if x.dtype != _FP32:
        return core.extern_elementwise("", "", [x, y], {
            tuple(a.dtype for a in [x, y]): (f"Unsupported.rhypot", _FP32),
        }, is_pure=True, _semantic=_semantic)
    return _semantic.truediv(_semantic.full(x.shape, 1.0, _FP32), hypot(x, y, _semantic=_semantic))


@core.extern
def norm3d(arg0, arg1, arg2, _semantic=None):
    return core.extern_elementwise(
        "", "", [arg0, arg1, arg2], {
            (core.dtype("fp32"), core.dtype("fp32"), core.dtype("fp32")): ("Unsupported.norm3d", core.dtype("fp32")),
            (core.dtype("fp64"), core.dtype("fp64"), core.dtype("fp64")): ("Unsupported.norm3d", core.dtype("fp64")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def rnorm3d(arg0, arg1, arg2, _semantic=None):
    return core.extern_elementwise(
        "", "", [arg0, arg1, arg2], {
            (core.dtype("fp32"), core.dtype("fp32"), core.dtype("fp32")): ("Unsupported.rnorm3d", core.dtype("fp32")),
            (core.dtype("fp64"), core.dtype("fp64"), core.dtype("fp64")): ("Unsupported.rnorm3d", core.dtype("fp64")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def norm4d(arg0, arg1, arg2, arg3, _semantic=None):
    return core.extern_elementwise(
        "", "", [arg0, arg1, arg2, arg3], {
            (core.dtype("fp32"), core.dtype("fp32"), core.dtype("fp32"), core.dtype("fp32")):
            ("Unsupported.norm4d", core.dtype("fp32")),
            (core.dtype("fp64"), core.dtype("fp64"), core.dtype("fp64"), core.dtype("fp64")):
            ("Unsupported.norm4d", core.dtype("fp64")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def rnorm4d(arg0, arg1, arg2, arg3, _semantic=None):
    return core.extern_elementwise(
        "", "", [arg0, arg1, arg2, arg3], {
            (core.dtype("fp32"), core.dtype("fp32"), core.dtype("fp32"), core.dtype("fp32")):
            ("Unsupported.rnorm4d", core.dtype("fp32")),
            (core.dtype("fp64"), core.dtype("fp64"), core.dtype("fp64"), core.dtype("fp64")):
            ("Unsupported.rnorm4d", core.dtype("fp64")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def cbrt(arg0, _semantic=None):
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp32"), ): ("Unsupported.cbrt", core.dtype("fp32")),
            (core.dtype("fp64"), ): ("Unsupported.cbrt", core.dtype("fp64")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def rcbrt(arg0, _semantic=None):
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp32"), ): ("Unsupported.rcbrt", core.dtype("fp32")),
            (core.dtype("fp64"), ): ("Unsupported.rcbrt", core.dtype("fp64")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def j0(arg0, _semantic=None):
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp32"), ): ("_ZN3xpu3j0fEf", core.dtype("fp32")),
            (core.dtype("fp64"), ): ("Unsupported.j0", core.dtype("fp64")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def j1(arg0, _semantic=None):
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp32"), ): ("_ZN3xpu3j1fEf", core.dtype("fp32")),
            (core.dtype("fp64"), ): ("Unsupported.j1", core.dtype("fp64")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def y0(arg0, _semantic=None):
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp32"), ): ("_ZN3xpu3y0fEf", core.dtype("fp32")),
            (core.dtype("fp64"), ): ("Unsupported.y0", core.dtype("fp64")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def y1(arg0, _semantic=None):
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp32"), ): ("_ZN3xpu3y1fEf", core.dtype("fp32")),
            (core.dtype("fp64"), ): ("Unsupported.y1", core.dtype("fp64")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def yn(arg0, arg1, _semantic=None):
    return core.extern_elementwise(
        "", "", [arg0, arg1], {
            (core.dtype("int32"), core.dtype("fp32")): ("Unsupported.yn", core.dtype("fp32")),
            (core.dtype("int32"), core.dtype("fp64")): ("Unsupported.yn", core.dtype("fp64")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def jn(arg0, arg1, _semantic=None):
    return core.extern_elementwise(
        "", "", [arg0, arg1], {
            (core.dtype("int32"), core.dtype("fp32")): ("Unsupported.jn", core.dtype("fp32")),
            (core.dtype("int32"), core.dtype("fp64")): ("Unsupported.jn", core.dtype("fp64")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def cyl_bessel_i0(arg0, _semantic=None):
    # Modified Bessel function of the first kind, order 0. Polynomial form for
    # |x| <= 3.75 (A&S 9.8.1), exp(x)/sqrt(x) times a 1/x polynomial above it
    # (A&S 9.8.2). Measured max rel error 6.9e-7 vs float64 over [-90, 90].
    # I0 is even, so only |x| is evaluated.
    x = _semantic.to_tensor(arg0)
    if x.dtype != _FP32:
        return core.extern_elementwise("", "", [x], {
            tuple(a.dtype for a in [x]): (f"Unsupported.cyl_bessel_i0", _FP32),
        }, is_pure=True, _semantic=_semantic)
    a = _math.abs(x, _semantic=_semantic)
    t = _semantic.truediv(a, _semantic.full(x.shape, 3.75, _FP32))
    t2 = _semantic.mul(t, t, sanitize_overflow=False)
    _sm = _semantic.full(t2.shape, 0.0045813, t2.dtype)
    _sm = _semantic.add(_semantic.mul(_sm, t2, sanitize_overflow=False), 0.0360768, sanitize_overflow=False)
    _sm = _semantic.add(_semantic.mul(_sm, t2, sanitize_overflow=False), 0.2659732, sanitize_overflow=False)
    _sm = _semantic.add(_semantic.mul(_sm, t2, sanitize_overflow=False), 1.2067492, sanitize_overflow=False)
    _sm = _semantic.add(_semantic.mul(_sm, t2, sanitize_overflow=False), 3.0899424, sanitize_overflow=False)
    _sm = _semantic.add(_semantic.mul(_sm, t2, sanitize_overflow=False), 3.5156229, sanitize_overflow=False)
    _sm = _semantic.add(_semantic.mul(_sm, t2, sanitize_overflow=False), 1.0, sanitize_overflow=False)
    inv = _semantic.truediv(_semantic.full(x.shape, 1.0, _FP32), t)
    _lg = _semantic.full(inv.shape, 0.00392377, inv.dtype)
    _lg = _semantic.add(_semantic.mul(_lg, inv, sanitize_overflow=False), -0.01647633, sanitize_overflow=False)
    _lg = _semantic.add(_semantic.mul(_lg, inv, sanitize_overflow=False), 0.02635537, sanitize_overflow=False)
    _lg = _semantic.add(_semantic.mul(_lg, inv, sanitize_overflow=False), -0.02057706, sanitize_overflow=False)
    _lg = _semantic.add(_semantic.mul(_lg, inv, sanitize_overflow=False), 0.00916281, sanitize_overflow=False)
    _lg = _semantic.add(_semantic.mul(_lg, inv, sanitize_overflow=False), -0.00157565, sanitize_overflow=False)
    _lg = _semantic.add(_semantic.mul(_lg, inv, sanitize_overflow=False), 0.00225319, sanitize_overflow=False)
    _lg = _semantic.add(_semantic.mul(_lg, inv, sanitize_overflow=False), 0.01328592, sanitize_overflow=False)
    _lg = _semantic.add(_semantic.mul(_lg, inv, sanitize_overflow=False), 0.39894228, sanitize_overflow=False)
    large = _semantic.mul(_semantic.truediv(_math.exp(a, _semantic=_semantic), _math.sqrt(a, _semantic=_semantic)), _lg,
                          sanitize_overflow=False)
    return _semantic.where(_semantic.less_equal(a, _semantic.full(x.shape, 3.75, _FP32)), _sm, large)


@core.extern
def cyl_bessel_i1(arg0, _semantic=None):
    # Modified Bessel function of the first kind, order 1 (A&S 9.8.3 / 9.8.4).
    # Measured max rel error 8.2e-7 vs float64 over [-90, 90]. I1 is odd, so
    # the magnitude is computed and the sign of x restored at the end.
    x = _semantic.to_tensor(arg0)
    if x.dtype != _FP32:
        return core.extern_elementwise("", "", [x], {
            tuple(a.dtype for a in [x]): (f"Unsupported.cyl_bessel_i1", _FP32),
        }, is_pure=True, _semantic=_semantic)
    a = _math.abs(x, _semantic=_semantic)
    t = _semantic.truediv(a, _semantic.full(x.shape, 3.75, _FP32))
    t2 = _semantic.mul(t, t, sanitize_overflow=False)
    _sm = _semantic.full(t2.shape, 0.00032411, t2.dtype)
    _sm = _semantic.add(_semantic.mul(_sm, t2, sanitize_overflow=False), 0.00301532, sanitize_overflow=False)
    _sm = _semantic.add(_semantic.mul(_sm, t2, sanitize_overflow=False), 0.02658733, sanitize_overflow=False)
    _sm = _semantic.add(_semantic.mul(_sm, t2, sanitize_overflow=False), 0.15084934, sanitize_overflow=False)
    _sm = _semantic.add(_semantic.mul(_sm, t2, sanitize_overflow=False), 0.51498869, sanitize_overflow=False)
    _sm = _semantic.add(_semantic.mul(_sm, t2, sanitize_overflow=False), 0.87890594, sanitize_overflow=False)
    _sm = _semantic.add(_semantic.mul(_sm, t2, sanitize_overflow=False), 0.5, sanitize_overflow=False)
    small = _semantic.mul(a, _sm, sanitize_overflow=False)
    inv = _semantic.truediv(_semantic.full(x.shape, 1.0, _FP32), t)
    _lg = _semantic.full(inv.shape, -0.00420059, inv.dtype)
    _lg = _semantic.add(_semantic.mul(_lg, inv, sanitize_overflow=False), 0.01787654, sanitize_overflow=False)
    _lg = _semantic.add(_semantic.mul(_lg, inv, sanitize_overflow=False), -0.02895312, sanitize_overflow=False)
    _lg = _semantic.add(_semantic.mul(_lg, inv, sanitize_overflow=False), 0.02282967, sanitize_overflow=False)
    _lg = _semantic.add(_semantic.mul(_lg, inv, sanitize_overflow=False), -0.01031555, sanitize_overflow=False)
    _lg = _semantic.add(_semantic.mul(_lg, inv, sanitize_overflow=False), 0.00163801, sanitize_overflow=False)
    _lg = _semantic.add(_semantic.mul(_lg, inv, sanitize_overflow=False), -0.00362018, sanitize_overflow=False)
    _lg = _semantic.add(_semantic.mul(_lg, inv, sanitize_overflow=False), -0.03988024, sanitize_overflow=False)
    _lg = _semantic.add(_semantic.mul(_lg, inv, sanitize_overflow=False), 0.39894228, sanitize_overflow=False)
    large = _semantic.mul(_semantic.truediv(_math.exp(a, _semantic=_semantic), _math.sqrt(a, _semantic=_semantic)), _lg,
                          sanitize_overflow=False)
    mag = _semantic.where(_semantic.less_equal(a, _semantic.full(x.shape, 3.75, _FP32)), small, large)
    return _semantic.where(_semantic.less_than(x, 0), _semantic.minus(mag), mag)


@core.extern
def erf(arg0, _semantic=None):
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp32"), ): ("_ZN3xpu3erfEf", core.dtype("fp32")),
            (core.dtype("fp64"), ): ("Unsupported.erf", core.dtype("fp64")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def erfinv(arg0, _semantic=None):
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp32"), ): ("_ZN3xpu6erfinvEf", core.dtype("fp32")),
            (core.dtype("fp64"), ): ("Unsupported.erfinv", core.dtype("fp64")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def erfc(arg0, _semantic=None):
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp32"), ): ("_ZN3xpu4erfcEf", core.dtype("fp32")),
            (core.dtype("fp64"), ): ("Unsupported.erfc", core.dtype("fp64")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def erfcx(arg0, _semantic=None):
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp32"), ): ("Unsupported.erfcx", core.dtype("fp32")),
            (core.dtype("fp64"), ): ("Unsupported.erfcx", core.dtype("fp64")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def erfcinv(arg0, _semantic=None):
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp32"), ): ("Unsupported.erfcinv", core.dtype("fp32")),
            (core.dtype("fp64"), ): ("Unsupported.erfcinv", core.dtype("fp64")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def normcdfinv(arg0, _semantic=None):
    # Inverse of the standard normal CDF (Abramowitz & Stegun 26.2.23).
    # Central rational approximation for |p - 0.5| <= 0.425, two tail branches
    # otherwise; the upper tail reuses 1 - p so both directions share one path.
    # Measured max rel error 3.9e-7 vs float64 across p in [1e-30, 1-1e-7].
    p = _semantic.to_tensor(arg0)
    if p.dtype != _FP32:
        return core.extern_elementwise("", "", [p], {
            tuple(a.dtype for a in [p]): (f"Unsupported.normcdfinv", _FP32),
        }, is_pure=True, _semantic=_semantic)
    half = _semantic.full(p.shape, 0.5, _FP32)
    q = _semantic.sub(p, half, sanitize_overflow=False)
    aq = _math.abs(q, _semantic=_semantic)
    rc = _semantic.sub(_semantic.full(p.shape, 0.180625, _FP32), _semantic.mul(q, q, sanitize_overflow=False),
                       sanitize_overflow=False)
    _nc = _semantic.full(rc.shape, 59.10937472, rc.dtype)
    _nc = _semantic.add(_semantic.mul(_nc, rc, sanitize_overflow=False), 159.29113202, sanitize_overflow=False)
    _nc = _semantic.add(_semantic.mul(_nc, rc, sanitize_overflow=False), 50.434271938, sanitize_overflow=False)
    _nc = _semantic.add(_semantic.mul(_nc, rc, sanitize_overflow=False), 3.3871327179, sanitize_overflow=False)
    _dc = _semantic.full(rc.shape, 67.1875636, rc.dtype)
    _dc = _semantic.add(_semantic.mul(_dc, rc, sanitize_overflow=False), 78.757757664, sanitize_overflow=False)
    _dc = _semantic.add(_semantic.mul(_dc, rc, sanitize_overflow=False), 17.895169469, sanitize_overflow=False)
    _dc = _semantic.add(_semantic.mul(_dc, rc, sanitize_overflow=False), 1.0, sanitize_overflow=False)
    x_central = _semantic.mul(q, _semantic.truediv(_nc, _dc), sanitize_overflow=False)
    pt = _semantic.where(_semantic.less_than(q, 0), p, _semantic.sub(half, q, sanitize_overflow=False))
    r = _math.sqrt(
        _semantic.sub(_semantic.full(p.shape, 0.0, _FP32), _math.log(pt, _semantic=_semantic), sanitize_overflow=False),
        _semantic=_semantic)
    r1 = _semantic.sub(r, _semantic.full(p.shape, 1.6, _FP32), sanitize_overflow=False)
    _n1 = _semantic.full(r1.shape, 0.17023821103, r1.dtype)
    _n1 = _semantic.add(_semantic.mul(_n1, r1, sanitize_overflow=False), 1.3067284816, sanitize_overflow=False)
    _n1 = _semantic.add(_semantic.mul(_n1, r1, sanitize_overflow=False), 2.75681539, sanitize_overflow=False)
    _n1 = _semantic.add(_semantic.mul(_n1, r1, sanitize_overflow=False), 1.4234372777, sanitize_overflow=False)
    _d1 = _semantic.full(r1.shape, 0.12021132975, r1.dtype)
    _d1 = _semantic.add(_semantic.mul(_d1, r1, sanitize_overflow=False), 0.7370016425, sanitize_overflow=False)
    _d1 = _semantic.add(_semantic.mul(_d1, r1, sanitize_overflow=False), 1.0, sanitize_overflow=False)
    # Clamp r to 1e16 arithmetically: for finite r in [0.83, 9.4] the extra
    # 1e-16 is nine orders below an fp32 ulp of 1/r (a no-op), while r = +inf
    # maps to 1e16 so the tail branch yields +inf instead of inf/inf = NaN.
    r2 = _semantic.sub(
        _semantic.truediv(
            _semantic.full(p.shape, 1.0, _FP32),
            _semantic.add(_semantic.truediv(_semantic.full(p.shape, 1.0, _FP32), r),
                          _semantic.full(p.shape, 1e-16, _FP32), sanitize_overflow=False),
        ), _semantic.full(p.shape, 5.0, _FP32), sanitize_overflow=False)
    _n2 = _semantic.full(r2.shape, 0.017337203997, r2.dtype)
    _n2 = _semantic.add(_semantic.mul(_n2, r2, sanitize_overflow=False), 0.42868294337, sanitize_overflow=False)
    _n2 = _semantic.add(_semantic.mul(_n2, r2, sanitize_overflow=False), 3.081226386, sanitize_overflow=False)
    _n2 = _semantic.add(_semantic.mul(_n2, r2, sanitize_overflow=False), 6.657905115, sanitize_overflow=False)
    _d2 = _semantic.full(r2.shape, 0.012258202635, r2.dtype)
    _d2 = _semantic.add(_semantic.mul(_d2, r2, sanitize_overflow=False), 0.24197894225, sanitize_overflow=False)
    _d2 = _semantic.add(_semantic.mul(_d2, r2, sanitize_overflow=False), 1.0, sanitize_overflow=False)
    x_tail = _semantic.mul(
        _semantic.truediv(q, aq),
        _semantic.where(_semantic.greater_than(r, _semantic.full(p.shape, 5.0, _FP32)), _semantic.truediv(_n2, _d2),
                        _semantic.truediv(_n1, _d1)), sanitize_overflow=False)
    return _semantic.where(_semantic.less_equal(aq, _semantic.full(p.shape, 0.425, _FP32)), x_central, x_tail)


@core.extern
def normcdf(arg0, _semantic=None):
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp32"), ): ("Unsupported.normcdf", core.dtype("fp32")),
            (core.dtype("fp64"), ): ("Unsupported.normcdf", core.dtype("fp64")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def lgamma(arg0, _semantic=None):
    # log|Gamma(x)| via the Lanczos approximation (g = 7.5, 9 coefficients).
    # For x < 0.5 apply the reflection formula
    #   log|Gamma(x)| = log(pi) - log|sin(pi*x)| - log|Gamma(1-x)|.
    # The sine argument is reduced around the nearest integer first; feeding
    # pi*x straight in loses ~6e-2 absolute near the negative poles because
    # the fp32 phase error is amplified by 1/|sin|.
    x = _semantic.to_tensor(arg0)
    if x.dtype != _FP32:
        return core.extern_elementwise("", "", [x], {
            tuple(a.dtype for a in [x]): (f"Unsupported.lgamma", _FP32),
        }, is_pure=True, _semantic=_semantic)
    reflect = _semantic.less_than(x, 0.5)
    one = _semantic.full(x.shape, 1.0, _FP32)
    y = _semantic.where(reflect, _semantic.sub(one, x, sanitize_overflow=False), x)
    z = _semantic.sub(y, one, sanitize_overflow=False)
    t = _semantic.add(z, _semantic.full(x.shape, 7.5, _FP32), sanitize_overflow=False)
    # Lanczos series: c0 + sum_k c_k / (z + k). This is a sum of reciprocals,
    # not a polynomial in z.
    _h = _semantic.full(z.shape, 0.9999999999998099, z.dtype)
    _h = _semantic.add(
        _h,
        _semantic.truediv(_semantic.full(z.shape, 676.5203681218851, _FP32),
                          _semantic.add(z, _semantic.full(z.shape, 1.0, _FP32), sanitize_overflow=False)),
        sanitize_overflow=False)
    _h = _semantic.add(
        _h,
        _semantic.truediv(_semantic.full(z.shape, -1259.1392167224028, _FP32),
                          _semantic.add(z, _semantic.full(z.shape, 2.0, _FP32), sanitize_overflow=False)),
        sanitize_overflow=False)
    _h = _semantic.add(
        _h,
        _semantic.truediv(_semantic.full(z.shape, 771.3234287776531, _FP32),
                          _semantic.add(z, _semantic.full(z.shape, 3.0, _FP32), sanitize_overflow=False)),
        sanitize_overflow=False)
    _h = _semantic.add(
        _h,
        _semantic.truediv(_semantic.full(z.shape, -176.6150291621406, _FP32),
                          _semantic.add(z, _semantic.full(z.shape, 4.0, _FP32), sanitize_overflow=False)),
        sanitize_overflow=False)
    _h = _semantic.add(
        _h,
        _semantic.truediv(_semantic.full(z.shape, 12.507343278686905, _FP32),
                          _semantic.add(z, _semantic.full(z.shape, 5.0, _FP32), sanitize_overflow=False)),
        sanitize_overflow=False)
    _h = _semantic.add(
        _h,
        _semantic.truediv(_semantic.full(z.shape, -0.13857109526572012, _FP32),
                          _semantic.add(z, _semantic.full(z.shape, 6.0, _FP32), sanitize_overflow=False)),
        sanitize_overflow=False)
    _h = _semantic.add(
        _h,
        _semantic.truediv(_semantic.full(z.shape, 9.984369578019572e-06, _FP32),
                          _semantic.add(z, _semantic.full(z.shape, 7.0, _FP32), sanitize_overflow=False)),
        sanitize_overflow=False)
    _h = _semantic.add(
        _h,
        _semantic.truediv(_semantic.full(z.shape, 1.5056327351493116e-07, _FP32),
                          _semantic.add(z, _semantic.full(z.shape, 8.0, _FP32), sanitize_overflow=False)),
        sanitize_overflow=False)
    log_a = _math.log(_h, _semantic=_semantic)
    half = _semantic.full(x.shape, 0.5, _FP32)
    lg_pos = _semantic.add(
        _semantic.sub(
            _semantic.add(
                _semantic.full(x.shape, 0.9189385332046727, _FP32),
                _semantic.mul(_semantic.add(z, half, sanitize_overflow=False), _math.log(t, _semantic=_semantic),
                              sanitize_overflow=False), sanitize_overflow=False), t, sanitize_overflow=False), log_a,
        sanitize_overflow=False)
    reduced = _semantic.sub(x, _math.floor(_semantic.add(x, half, sanitize_overflow=False), _semantic=_semantic),
                            sanitize_overflow=False)
    angle = _semantic.mul(_semantic.full(x.shape, 3.141592653589793, _FP32), reduced, sanitize_overflow=False)
    angle2 = _semantic.mul(angle, angle, sanitize_overflow=False)
    sin_taylor = _semantic.mul(
        angle,
        _semantic.add(
            one,
            _semantic.mul(
                angle2,
                _semantic.add(
                    _semantic.full(x.shape, -1.0 / 6.0, _FP32),
                    _semantic.mul(angle2, _semantic.full(x.shape, 1.0 / 120.0, _FP32), sanitize_overflow=False),
                    sanitize_overflow=False), sanitize_overflow=False), sanitize_overflow=False),
        sanitize_overflow=False)
    sin_val = _math.abs(
        _semantic.where(_semantic.less_than(angle2, _semantic.full(x.shape, 1e-4, _FP32)), sin_taylor,
                        _math.sin(angle, _semantic=_semantic)), _semantic=_semantic)
    lg_neg = _semantic.sub(
        _semantic.sub(_semantic.full(x.shape, 1.1447298858494002, _FP32), _math.log(sin_val, _semantic=_semantic),
                      sanitize_overflow=False), lg_pos, sanitize_overflow=False)
    out = _semantic.where(reflect, lg_neg, lg_pos)
    is_pole = _semantic.logical_and(_semantic.less_equal(x, 0), _semantic.equal(x, _math.floor(x, _semantic=_semantic)))
    return _semantic.where(is_pole, _semantic.full(x.shape, float("inf"), _FP32), out)


@core.extern
def ldexp(arg0, arg1, _semantic=None):
    return core.extern_elementwise(
        "", "", [arg0, arg1], {
            (core.dtype("fp32"), core.dtype("int32")): ("ldexpf", core.dtype("fp32")),
            (core.dtype("fp64"), core.dtype("int32")): ("ldexp", core.dtype("fp64")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def scalbn(arg0, arg1, _semantic=None):
    return core.extern_elementwise(
        "", "", [arg0, arg1], {
            (core.dtype("fp32"), core.dtype("int32")): ("Unsupported.scalbn", core.dtype("fp32")),
            (core.dtype("fp64"), core.dtype("int32")): ("Unsupported.scalbn", core.dtype("fp64")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def fmod(arg0, arg1, _semantic=None):
    return core.extern_elementwise(
        "", "", [arg0, arg1], {
            (core.dtype("fp32"), core.dtype("fp32")): ("_ZN3xpu5fmodfEff", core.dtype("fp32")),
            (core.dtype("fp64"), core.dtype("fp64")): ("Unsupported.fmod", core.dtype("fp64")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def remainder(arg0, arg1, _semantic=None):
    return core.extern_elementwise(
        "", "", [arg0, arg1], {
            (core.dtype("fp32"), core.dtype("fp32")): ("Unsupported.remainder", core.dtype("fp32")),
            (core.dtype("fp64"), core.dtype("fp64")): ("Unsupported.remainder", core.dtype("fp64")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def fma(arg0, arg1, arg2, _semantic=None):
    return core.extern_elementwise(
        "", "", [arg0, arg1, arg2], {
            (core.dtype("fp32"), core.dtype("fp32"), core.dtype("fp32")): ("_ZN3xpu3fmaEfff", core.dtype("fp32")),
            (core.dtype("fp64"), core.dtype("fp64"), core.dtype("fp64")): ("Unsupported.fma", core.dtype("fp64")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def pow(arg0, arg1, _semantic=None):
    arg0 = _semantic.to_tensor(arg0)
    arg1 = _semantic.to_tensor(arg1)
    if arg1.dtype.is_int():
        # C's pow(float, int) and pow(double, int) convert the exponent to the
        # base's floating type first; without this the fp64 base below would
        # fall through to the placeholder for the (fp64, int) pair.
        if arg0.dtype == core.dtype("fp32"):
            arg1 = arg1.to(core.dtype("fp32"), _semantic=_semantic)
        elif arg0.dtype == _FP64:
            arg1 = arg1.to(_FP64, _semantic=_semantic)
    if arg0.dtype == _FP64 and arg1.dtype == _FP64:
        # exp2(y * log2(|x|)), with a sign from y for a negative base and
        # integral y (C's pow), and the cases log2 cannot express peeled off
        # first: a zero base, and a negative base with non-integral y (NaN).
        zero = _f64(0.0, arg0, _semantic)
        one = _f64(1.0, arg0, _semantic)
        minus_one = _f64(-1.0, arg0, _semantic)
        pos_inf = _semantic.truediv(one, zero)
        nan = _semantic.truediv(zero, zero)
        is_one_base = _semantic.equal(arg0, one)
        negative = _semantic.less_than(arg0, zero)
        # Integral-exponent test and parity, shared by the negative-base and
        # the -0.0 cases.  `_f64_round` passes |y| >= 2**52 through, and the
        # parity of a huge (or infinite) exponent must not reach the int64
        # cast: fptosi of an out-of-range value is poison.  Treat that band
        # as non-integral (negative base -> NaN).
        y_int = _f64_round(arg1, _semantic)
        limit = _f64(4503599627370496.0, arg0, _semantic)  # 2**52
        abs_y = _semantic.where(_semantic.less_than(arg1, zero), _semantic.sub(zero, arg1, sanitize_overflow=False),
                                arg1)
        in_range = _semantic.less_equal(abs_y, limit)
        y_safe = _semantic.where(in_range, y_int, zero)
        is_int = _semantic.logical_and(_semantic.equal(arg1, y_int), in_range)
        odd = _semantic.not_equal(_semantic.and_(_semantic.cast(y_safe, _I64), 1), 0)
        odd_int = _semantic.logical_and(is_int, odd)

        abs_base = _semantic.where(negative, _semantic.sub(zero, arg0, sanitize_overflow=False), arg0)
        out = _f64_exp2(_semantic.mul(arg1, _f64_log2(abs_base, _semantic), sanitize_overflow=False), _semantic)
        out = _semantic.where(is_one_base, one, out)
        # pow(+-0, y): 0 for y > 0, 1 for y == 0, +inf for y < 0, NaN for a
        # NaN exponent.  A -0.0 base with an odd integral y flips the sign of
        # the zero/inf.
        is_nan_exp = _semantic.not_equal(arg1, arg1)
        zero_base = _semantic.where(_semantic.greater_than(arg1, zero), zero,
                                    _semantic.where(_semantic.equal(arg1, zero), one, pos_inf))
        zero_base = _semantic.where(is_nan_exp, nan, zero_base)
        zero_i64 = _semantic.full(arg0.shape, 0, _I64)
        neg_zero = _semantic.logical_and(_semantic.equal(arg0, zero),
                                         _semantic.less_than(_semantic.bitcast(arg0, _I64), zero_i64))
        zero_base = _semantic.where(_semantic.logical_and(neg_zero, odd_int),
                                    _semantic.mul(zero_base, minus_one, sanitize_overflow=False), zero_base)
        out = _semantic.where(_semantic.equal(arg0, zero), zero_base, out)
        # pow(+inf, y): +inf for y > 0, +0 for y < 0, 1 for y == 0, NaN for a
        # NaN exponent.  _f64_log2 reads +inf's exponent field as 1024, so the
        # generic exp2(y*log2(x)) path would return a finite value.
        is_pos_inf = _semantic.equal(arg0, pos_inf)
        inf_base = _semantic.where(
            _semantic.greater_than(arg1, zero), pos_inf,
            _semantic.where(_semantic.less_than(arg1, zero), zero, _semantic.where(is_nan_exp, nan, one)))
        out = _semantic.where(is_pos_inf, inf_base, out)
        # pow(x < 0, y) is real only for integral y: (-1)^y * |x|^y.
        signed = _semantic.where(odd_int, _semantic.mul(out, minus_one, sanitize_overflow=False), out)
        out = _semantic.where(negative, _semantic.where(is_int, signed, nan), out)
        # pow(x, +-inf) depends only on |x|: 1 for |x| == 1, +inf/0 for
        # |x| > 1, 0/+inf for |x| < 1.  The generic exp2(y*log2|x|) path gets
        # NaN from inf*0 and inf-inf, and the negative-base rule above would
        # turn a finite negative base into NaN, so override both.
        is_inf_exp = _semantic.equal(abs_y, pos_inf)
        inf_exp = _semantic.where(
            _semantic.greater_than(abs_base, one), _semantic.where(_semantic.greater_than(arg1, zero), pos_inf, zero),
            _semantic.where(_semantic.less_than(abs_base, one),
                            _semantic.where(_semantic.greater_than(arg1, zero), zero, pos_inf), one))
        out = _semantic.where(is_inf_exp, inf_exp, out)
        out = _semantic.where(_semantic.not_equal(arg0, arg0), arg0, out)
        # pow(anything, +-0) is 1, including a NaN base.
        return _semantic.where(_semantic.equal(arg1, zero), one, out)
    if arg0.dtype == _FP64 or arg1.dtype == _FP64:
        return core.extern_elementwise("", "", [arg0, arg1], {
            tuple(a.dtype for a in [arg0, arg1]): ("Unsupported.pow", _FP64),
        }, is_pure=True, _semantic=_semantic)
    return core.extern_elementwise(
        "", "", [arg0, arg1], {
            (_FP32, _FP32): ("_ZN3xpu3powEff", _FP32),
            (core.dtype("fp16"), core.dtype("fp16")): ("_ZN3xpu4hpowEDF16_DF16_", core.dtype("fp16")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def tgamma(arg0, _semantic=None):
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp32"), ): ("Unsupported.tgamma", core.dtype("fp32")),
            (core.dtype("fp64"), ): ("Unsupported.tgamma", core.dtype("fp64")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def round(arg0, _semantic=None):
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp32"), ): ("Unsupported.round", core.dtype("fp32")),
            (core.dtype("fp64"), ): ("Unsupported.round", core.dtype("fp64")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def llround(arg0, _semantic=None):
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp32"), ): ("Unsupported.llround", core.dtype("int64")),
            (core.dtype("fp64"), ): ("Unsupported.llround", core.dtype("int64")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def fdim(arg0, arg1, _semantic=None):
    return core.extern_elementwise(
        "", "", [arg0, arg1], {
            (core.dtype("fp32"), core.dtype("fp32")): ("Unsupported.fdim", core.dtype("fp32")),
            (core.dtype("fp64"), core.dtype("fp64")): ("Unsupported.fdim", core.dtype("fp64")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def ilogb(arg0, _semantic=None):
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp32"), ): ("Unsupported.ilogb", core.dtype("int32")),
            (core.dtype("fp64"), ): ("Unsupported.ilogb", core.dtype("int32")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def logb(arg0, _semantic=None):
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp32"), ): ("Unsupported.logb", core.dtype("fp32")),
            (core.dtype("fp64"), ): ("Unsupported.logb", core.dtype("fp64")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def isfinited(arg0, _semantic=None):
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("fp64"), ): ("Unsupported.isfinited", core.dtype("int32")),
    }, is_pure=True, _semantic=_semantic)


@core.extern
def xpu_trunc_div(arg0, arg1, _semantic=None):
    return core.extern_elementwise(
        "", "", [arg0, arg1], {
            (core.dtype("fp32"), core.dtype("fp32")): ("_ZN3xpu9xpu_truncEff", core.dtype("fp32")),
            (core.dtype("fp64"), core.dtype("fp64")): ("Unsupported.xpu_trunc_div", core.dtype("fp64")),
        }, is_pure=True, _semantic=_semantic)
