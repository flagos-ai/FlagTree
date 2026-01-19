from triton.language import core

@core.extern
def erf(arg0, _semantic=None):
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp32"), ): ("__ocml_erf_f32", core.dtype("fp32")),
            (core.dtype("fp64"), ): ("__ocml_erf_f32", core.dtype("fp64")),
        }, is_pure=True, _semantic=_semantic)

@core.extern
def pow(arg0, arg1, _semantic=None):
    return core.extern_elementwise(
        "", "", [arg0, arg1], {
            (core.dtype("fp32"), core.dtype("int32")): ("__ocml_pown_f32", core.dtype("fp32")),
            (core.dtype("fp64"), core.dtype("int32")): ("__ocml_pown_f32", core.dtype("fp64")),
            (core.dtype("fp16"), core.dtype("fp16")): ("__ocml_pow_f16", core.dtype("fp16")),
            (core.dtype("fp32"), core.dtype("fp32")): ("__ocml_pow_f32", core.dtype("fp32")),
            (core.dtype("fp64"), core.dtype("fp64")): ("__ocml_pow_f32", core.dtype("fp64")),
        }, is_pure=True, _semantic=_semantic)

@core.extern
def tanh(arg0, _semantic=None):
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp32"), ): ("__ocml_tanh_f32", core.dtype("fp32")),
            (core.dtype("fp64"), ): ("__ocml_tanh_f32", core.dtype("fp64")),
        }, is_pure=True, _semantic=_semantic)

@core.extern
def atan2(arg0, arg1, _semantic=None):
    return core.extern_elementwise(
        "", "", [arg0, arg1], {
            (core.dtype("fp32"), core.dtype("fp32")): ("__ocml_atan2_f32", core.dtype("fp32")),
            (core.dtype("fp64"), core.dtype("fp64")): ("__ocml_atan2_f32", core.dtype("fp64")),
        }, is_pure=True, _semantic=_semantic)

@core.extern
def asin(arg0, _semantic=None):
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp32"), ): ("__ocml_asin_f32", core.dtype("fp32")),
            (core.dtype("fp64"), ): ("__ocml_asin_f32", core.dtype("fp64")),
        }, is_pure=True, _semantic=_semantic)

@core.extern
def div_rd(arg0, arg1, _semantic=None):
    return core.extern_elementwise(
        "", "", [arg0, arg1], {
            (core.dtype("fp32"), core.dtype("fp32")): ("llvm.stvm.div.rm.f32", core.dtype("fp32")),
            (core.dtype("fp64"), core.dtype("fp64")): ("llvm.stvm.div.rm.f32", core.dtype("fp64")),
        }, is_pure=True, _semantic=_semantic)

@core.extern
def div_rz(arg0, arg1, _semantic=None):
    return core.extern_elementwise(
        "", "", [arg0, arg1], {
            (core.dtype("fp32"), core.dtype("fp32")): ("llvm.stvm.div.rz.f32", core.dtype("fp32")),
            (core.dtype("fp64"), core.dtype("fp64")): ("llvm.stvm.div.rz.f32", core.dtype("fp64")),
        }, is_pure=True, _semantic=_semantic)

@core.extern
def rsqrt(arg0, _semantic=None):
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp32"), ): ("llvm.stvm.rsqrt.f32", core.dtype("fp32")),
            (core.dtype("fp64"), ): ("llvm.stvm.rsqrt.f32", core.dtype("fp64")),
        }, is_pure=True, _semantic=_semantic)

@core.extern
def isinf(arg0, _semantic=None):
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp32"), ): ("llvm.stvm.testp.f32.inf", core.dtype("int32")),
            (core.dtype("fp64"), ): ("llvm.stvm.testp.f32.inf", core.dtype("int32")),
        }, is_pure=True, _semantic=_semantic)

@core.extern
def isnan(arg0, _semantic=None):
    return core.extern_elementwise(
        "", "", [
            arg0,
        ], {
            (core.dtype("fp32"), ): ("llvm.stvm.testp.f32.not", core.dtype("int32")),
            (core.dtype("fp64"), ): ("llvm.stvm.testp.f32.not", core.dtype("int32")),
        }, is_pure=True, _semantic=_semantic)

@core.extern
def erf(arg0, _semantic=None):
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp32"), ): ("__ocml_erf_f32", core.dtype("fp32")),
            (core.dtype("fp64"), ): ("__ocml_erf_f32", core.dtype("fp64")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def exp(arg0, _semantic=None):
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp32"), ): ("__ocml_exp_f32", core.dtype("fp32")),
            (core.dtype("fp64"), ): ("__ocml_exp_f32", core.dtype("fp64")),
        }, is_pure=True, _semantic=_semantic)

@core.extern
def exp2(arg0, _semantic=None):
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp32"), ): ("llvm.stvm.exp2.f32", core.dtype("fp32")),
            (core.dtype("fp64"), ): ("llvm.stvm.exp2.f32", core.dtype("fp64")),
        }, is_pure=True, _semantic=_semantic)

@core.extern
def div_rn(arg0, arg1, _semantic=None):
    return core.extern_elementwise(
        "", "", [arg0, arg1], {
            (core.dtype("fp32"), core.dtype("fp32")): ("llvm.stvm.div.rn.f32", core.dtype("fp32")),
            (core.dtype("fp64"), core.dtype("fp64")): ("llvm.stvm.div.rn.f32", core.dtype("fp64")),
        }, is_pure=True, _semantic=_semantic)

@core.extern
def trunc(arg0, _semantic=None):
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp32"), ): ("llvm.stvm.trunc.f", core.dtype("fp32")),
            (core.dtype("fp64"), ): ("llvm.stvm.trunc.f", core.dtype("fp64")),
        }, is_pure=True, _semantic=_semantic)

@core.extern
def fmod(arg0, arg1, _semantic=None):
    return core.extern_elementwise(
        "", "", [arg0, arg1], {
            (core.dtype("fp32"), core.dtype("fp32")): ("__ocml_fmod_f32", core.dtype("fp32")),
            (core.dtype("fp64"), core.dtype("fp64")): ("__ocml_fmod_f32", core.dtype("fp64")),
        }, is_pure=True, _semantic=_semantic)

@core.extern
def isfinited(arg0, _semantic=None):
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("fp64"), ): ("__ocml_isfinite_f64", core.dtype("int32")),
    }, is_pure=True, _semantic=_semantic).to(core.int1, _semantic=_semantic)

@core.extern
def finitef(arg0, _semantic=None):
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("fp32"), ): ("__ocml_isfinite_f32", core.dtype("int32")),
    }, is_pure=True, _semantic=_semantic).to(core.int1, _semantic=_semantic)

@core.extern
def rint(arg0, _semantic=None):
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("fp32"), ): ("llvm.rint.f32", core.dtype("fp32")),
        (core.dtype("fp64"), ): ("llvm.rint.f32", core.dtype("fp64")),
    }, is_pure=True, _semantic=_semantic)