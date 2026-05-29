from triton import jit
import triton.language as tl
from triton.language.core import builtin


@jit
def _vnni_decode(arg0):
    tl.static_assert(len(arg0.shape) == 2)
    tmp = arg0.reshape((arg0.shape[0], arg0.shape[1] // 2, 2))
    tmp1, tmp2 = tl.split(tmp)
    return tl.join(tmp1.T, tmp2.T).reshape((arg0.shape[1] // 2, arg0.shape[0] * 2)).T


@builtin
def vnni_decode(arg0, _builder=None, _generator=None):
    bitwidth = arg0.dtype.primitive_bitwidth
    if bitwidth > 16:
        raise ValueError("Expected 8-bit or 16-bit values for vnni_decode")
    decoded = _generator.call_JitFunction(_vnni_decode, (arg0, ), kwargs={})
    if bitwidth == 8:
        decoded = _generator.call_JitFunction(_vnni_decode, (decoded, ), kwargs={})
    return decoded
