from triton.language import core

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
    if arg1 == None:
        return core.extern_elementwise(
            "", "", [arg0], {
                (core.dtype("bf16"), ): ("__hmf_flipDhb", core.dtype("bf16")),
                (core.dtype("fp16"), ): ("__hmf_flipDh", core.dtype("fp16")),
                (core.dtype("fp32"), ): ("__hmf_flipf", core.dtype("fp32")),
                (core.dtype("int8"), ): ("__hmf_flipi8", core.dtype("int8")),
                (core.dtype("int16"), ): ("__hmf_flipi16", core.dtype("int16")),
                (core.dtype("int32"), ): ("__hmf_flipi32", core.dtype("int32")),
                (core.dtype("uint32"), ): ("__hmf_flipui32", core.dtype("uint32")),
                (core.dtype("int64"), ): ("__hmf_flipi64", core.dtype("int64")),
            }, is_pure=True, _builder=_builder)

    return core.extern_elementwise(
        "", "", [arg0, arg1], {
            (core.dtype("bf16"), core.dtype("int32")): ("__hmf_flipDhb", core.dtype("bf16")),
            (core.dtype("fp16"), core.dtype("int32")): ("__hmf_flipDh", core.dtype("fp16")),
            (core.dtype("fp32"), core.dtype("int32")): ("__hmf_flipf", core.dtype("fp32")),
            (core.dtype("int8"), core.dtype("int32")): ("__hmf_flipi8", core.dtype("int8")),
            (core.dtype("int16"), core.dtype("int32")): ("__hmf_flipi16", core.dtype("int16")),
            (core.dtype("int32"), core.dtype("int32")): ("__hmf_flipi32", core.dtype("int32")),
            (core.dtype("uint32"), core.dtype("int32")): ("__hmf_flipui32", core.dtype("uint32")),
            (core.dtype("int64"), core.dtype("int32")): ("__hmf_flipi64", core.dtype("int64")),
        }, is_pure=True, _builder=_builder)

@core.extern
def atan2(arg0, _builder=None):
    core.static_print("tl.atan2 is unsupported for now. Use libdevice.atan2 instead.")
    core.static_assert(False)

@core.extern
def div_rz(arg0, arg1, _builder=None):
    core.static_print("tl.div_rz is unsupported for now. Use libdevice.div_rz instead.")
    core.static_assert(False)

@core.extern
def fmod(arg0, arg1, _builder=None):
    core.static_print("tl.fmod is unsupported for now. Use libdevice.fmod instead.")
    core.static_assert(False)

@core.extern
def trunc(arg0, _builder=None):
    core.static_print("tl.trunc is unsupported for now. Use libdevice.trunc instead.")
    core.static_assert(False)

@core.extern
def round(arg0, _builder=None):
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp32"), ): ("__hmf_roundf", core.dtype("fp32")),            
        }, is_pure=True, _builder=_builder)