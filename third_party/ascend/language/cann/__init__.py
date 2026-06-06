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

from triton.language import math
from triton.backends.ascend.utils import triton_enable_libdevice_simt

from . import libdevice
from . import extension

extension.parallel = extension.aux_ops.parallel
if not triton_enable_libdevice_simt():
    libdevice.atan2 = extension.math_ops.atan2
libdevice.isfinited = extension.math_ops.isfinited
libdevice.finitef = extension.math_ops.finitef
libdevice.flip = extension.flip

libdevice.umulhi = math.umulhi
libdevice.exp = math.exp
libdevice.exp2 = math.exp2
libdevice.log = math.log
libdevice.log2 = math.log2
libdevice.cos = math.cos
libdevice.sin = math.sin
libdevice.sqrt = math.sqrt
libdevice.sqrt_rn = math.sqrt_rn
libdevice.rsqrt = math.rsqrt
libdevice.div_rn = math.div_rn
libdevice.erf = math.erf
libdevice.floor = math.floor
libdevice.ceil = math.ceil
libdevice.fdiv = math.fdiv
libdevice.fma = math.fma
libdevice.abs = math.abs
math.tanh = libdevice.tanh

__all__ = ["libdevice", "extension"]

# flagtree language extension
from triton.flagtree_spec import bind_language_extension_symbols_to_tl

bind_language_extension_symbols_to_tl(extension)
