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

from dataclasses import dataclass
import triton
import triton.language as tl
from .base import Layout

NON_K_PRESHUFFLE_BLOCK_SIZE = 32


@dataclass
class CDNA4MXScaleLayout(Layout):
    name: str = "CDNA4_SCALE"

    def __init__(self, shape) -> None:
        super().__init__(shape)

    def swizzle_data(self, data):
        block_shape = data.shape
        SCALE_K = block_shape[-2]
        N = block_shape[-1]
        data = data.transpose(-1, -2)
        data = data.view(-1, N // NON_K_PRESHUFFLE_BLOCK_SIZE, 2, 16, SCALE_K // 8, 2, 4, 1)
        data = data.permute(0, 1, 4, 6, 3, 5, 2, 7).contiguous()
        if len(block_shape) == 3:
            E = block_shape[0]
            data = data.reshape(E, N // 32, SCALE_K * 32)
        else:
            assert len(block_shape) == 2
            data = data.reshape(N // 32, SCALE_K * 32)
        return data.transpose(-1, -2)

    def unswizzle_data(self, data):
        raise NotImplementedError()

    def swizzle_block_shape(self, block_shape):
        SCALE_K = block_shape[-2]
        N = block_shape[-1]
        return block_shape[:-2] + [N // 32, SCALE_K * 32]


@triton.jit
def unswizzle_mx_scale_cdna4(x, BLOCK_N: tl.constexpr, MX_SCALE_BLOCK_K: tl.constexpr,
                             N_PRESHUFFLE_FACTOR: tl.constexpr = NON_K_PRESHUFFLE_BLOCK_SIZE):
    x = x.reshape(BLOCK_N // N_PRESHUFFLE_FACTOR, MX_SCALE_BLOCK_K // 8, 4, 16, 2, 2, 1)
    x = x.permute(0, 5, 3, 1, 4, 2, 6)
    x = x.reshape(BLOCK_N, MX_SCALE_BLOCK_K)
    return x
