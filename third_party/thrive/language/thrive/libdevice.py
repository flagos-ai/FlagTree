from triton.language import core
from .thrive_semantic import get_thrive_semantic


@core.extern
def j0(arg0, _semantic=None):
    raise RuntimeError("j0 is not supported on thrive backend")


@core.extern
def j1(arg0, _semantic=None):
    raise RuntimeError("j1 is not supported on thrive backend")


@core.extern
def y0(arg0, _semantic=None):
    raise RuntimeError("y0 is not supported on thrive backend")


@core.extern
def y1(arg0, _semantic=None):
    raise RuntimeError("y1 is not supported on thrive backend")


@core.extern
def cyl_bessel_i0(arg0, _semantic=None):
    raise RuntimeError("cyl_bessel_i0 is not supported on thrive backend")


@core.extern
def cyl_bessel_i1(arg0, _semantic=None):
    raise RuntimeError("cyl_bessel_i1 is not supported on thrive backend")


@core.extern
def fast_dividef(arg0, arg1, _semantic=None):
    raise RuntimeError("fast_dividef is not supported on thrive backend")


@core.extern
def asin(arg0, _semantic=None):
    semantic = get_thrive_semantic(_semantic)
    return semantic.asin(arg0)


@core.extern
def asinh(arg0, _semantic=None):
    semantic = get_thrive_semantic(_semantic)
    return semantic.asinh(arg0)


@core.extern
def acos(arg0, _semantic=None):
    semantic = get_thrive_semantic(_semantic)
    return semantic.acos(arg0)


@core.extern
def acosh(arg0, _semantic=None):
    semantic = get_thrive_semantic(_semantic)
    return semantic.acosh(arg0)


@core.extern
def atan(arg0, _semantic=None):
    semantic = get_thrive_semantic(_semantic)
    return semantic.atan(arg0)


@core.extern
def atanh(arg0, _semantic=None):
    semantic = get_thrive_semantic(_semantic)
    return semantic.atanh(arg0)


@core.extern
def sin(arg0, _semantic=None):
    semantic = get_thrive_semantic(_semantic)
    return semantic.sin(arg0)


@core.extern
def sinh(arg0, _semantic=None):
    semantic = get_thrive_semantic(_semantic)
    return semantic.sinh(arg0)


@core.extern
def cos(arg0, _semantic=None):
    semantic = get_thrive_semantic(_semantic)
    return semantic.cos(arg0)


@core.extern
def cosh(arg0, _semantic=None):
    semantic = get_thrive_semantic(_semantic)
    return semantic.cosh(arg0)


@core.extern
def tan(arg0, _semantic=None):
    semantic = get_thrive_semantic(_semantic)
    return semantic.tan(arg0)


@core.extern
def tanh(arg0, _semantic=None):
    semantic = get_thrive_semantic(_semantic)
    return semantic.tanh(arg0)


@core.extern
def exp(arg0, _semantic=None):
    semantic = get_thrive_semantic(_semantic)
    return semantic.exp(arg0)


@core.extern
def exp2(arg0, _semantic=None):
    semantic = get_thrive_semantic(_semantic)
    return semantic.exp2(arg0)


@core.extern
def log(arg0, _semantic=None):
    semantic = get_thrive_semantic(_semantic)
    return semantic.log(arg0)


@core.extern
def log2(arg0, _semantic=None):
    semantic = get_thrive_semantic(_semantic)
    return semantic.log2(arg0)


@core.extern
def rsqrt(arg0, _semantic=None):
    semantic = get_thrive_semantic(_semantic)
    return semantic.rsqrt(arg0)


@core.extern
def pow(arg0, arg1, _semantic=None):
    semantic = get_thrive_semantic(_semantic)
    return semantic.pow(arg0, arg1)
