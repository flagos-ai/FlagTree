# Copyright 2018-2020 Philippe Tillet
# Copyright 2020-2022 OpenAI
# Copyright 2025-     FlagOS Contributors
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import triton.language.math as tl_math
from ._core import builtin

umulhi = builtin(tl_math.umulhi)
exp = builtin(tl_math.exp)
exp2 = builtin(tl_math.exp2)
fma = builtin(tl_math.fma)
log = builtin(tl_math.log)
log2 = builtin(tl_math.log2)
cos = builtin(tl_math.cos)
rsqrt = builtin(tl_math.rsqrt)
sin = builtin(tl_math.sin)
sqrt = builtin(tl_math.sqrt)
sqrt_rn = builtin(tl_math.sqrt_rn)
abs = builtin(tl_math.abs)
fdiv = builtin(tl_math.fdiv)
div_rn = builtin(tl_math.div_rn)
erf = builtin(tl_math.erf)
floor = builtin(tl_math.floor)
ceil = builtin(tl_math.ceil)
