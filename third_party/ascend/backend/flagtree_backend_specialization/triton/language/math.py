import triton.language as language
from . import standard
softmax = standard.softmax
sigmoid = standard.sigmoid
argmax = standard.argmax
argmin = standard.argmin
umulhi = language.extra.ascend.libdevice.umulhi
exp = language.extra.ascend.libdevice.exp
exp2 = language.extra.ascend.libdevice.exp2
log = language.extra.ascend.libdevice.log
log2 = language.extra.ascend.libdevice.log2
cos = language.extra.ascend.libdevice.cos
sin = language.extra.ascend.libdevice.sin
sqrt = language.extra.ascend.libdevice.sqrt
sqrt_rn = language.extra.ascend.libdevice.sqrt_rn
rsqrt = language.extra.ascend.libdevice.rsqrt
div_rn = language.extra.ascend.libdevice.div_rn
erf = language.extra.ascend.libdevice.erf
tanh = language.extra.ascend.libdevice.tanh
floor = language.extra.ascend.libdevice.floor
ceil = language.extra.ascend.libdevice.ceil
fma = language.extra.ascend.libdevice.fma
_check_dtype = language.extra.ascend.libdevice._check_dtype
cdiv = language.extra.ascend.libdevice.cdiv

isnan = language.extra.ascend.libdevice.isnan
isinf = language.extra.ascend.libdevice.isinf
reciprocal = language.extra.ascend.libdevice.reciprocal
relu = language.extra.ascend.libdevice.relu
log1p = language.extra.ascend.libdevice.log1p
tan = language.extra.ascend.libdevice.tan
atan = language.extra.ascend.libdevice.atan
tanh = language.extra.ascend.libdevice.tanh
ilogb = language.extra.ascend.libdevice.ilogb
ldexp = language.extra.ascend.libdevice.ldexp
pow = language.extra.ascend.libdevice.pow
flip = standard.flip
atan2 = standard.atan2
rint = standard.rint
finitef = standard.finitef
isfinited = standard.isfinited
div_rz = language.extra.ascend.libdevice.div_rz
fmod = language.extra.ascend.libdevice.fmod
trunc = language.extra.ascend.libdevice.trunc
round = language.extra.ascend.libdevice.round

math_ext_base_func_list = [
    "umulhi", "exp", "exp2", "log", "log2", "cos",
    "sin", "sqrt", "sqrt_rn", "rsqrt", "div_rn", "erf",
    "tanh", "floor", "ceil", "fma", "_check_dtype", "softmax", "sigmoid",
    "cdiv", "argmax", "argmin"
]
math_ext_spec_func_list = [
    "isnan", "isinf", "reciprocal", "relu", "log1p", "tan",
    "atan", "tanh", "ilogb", "ldexp", "pow", "flip", "atan2",
    "div_rz", "fmod", "trunc", "round", "rint", "finitef", "isfinited"
]
