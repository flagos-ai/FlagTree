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

import triton
import triton.language as tl


@triton.jit
def _masked_compaction(Yv, Yi, BitMask, stride_bm, stride_bn, RetYv, RetYi, sentinel, K: tl.constexpr):
    pid_m = tl.program_id(0)
    yv = tl.load(Yv + pid_m * K + tl.arange(0, K))
    yi = tl.load(Yi + pid_m * K + tl.arange(0, K))
    div = yi // 32
    rem = yi % 32
    active_bits = (tl.load(BitMask + pid_m * stride_bm + div * stride_bn) >> rem) & 1
    exc_cumsum = tl.cumsum(active_bits, 0) - active_bits
    active_flags = active_bits.to(tl.int1)
    rev_arange = tl.where(active_flags, 0, K - 1 - tl.arange(0, K))
    write_indx = exc_cumsum + rev_arange
    yv = tl.where(active_flags, yv, sentinel)
    yi = tl.where(active_flags, yi, sentinel)
    tl.store(RetYv + pid_m * K + write_indx, yv)
    tl.store(RetYi + pid_m * K + write_indx, yi)
