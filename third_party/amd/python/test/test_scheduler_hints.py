# Copyright 2026 FlagOS Contributors
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

import pytest
import torch
import triton
import triton.language as tl

from triton._internal_testing import is_hip

if not is_hip():
    pytest.skip(allow_module_level=True)


def test_schedule_hint(device):

    @triton.jit
    def kernel(X, Y, Z, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
        off_m = tl.arange(0, BLOCK_M)
        off_n = tl.arange(0, BLOCK_N)
        off_k = tl.arange(0, BLOCK_K)
        Xs = X + off_m[:, None] * BLOCK_K + off_k[None, :] * 1
        Ys = Y + off_k[:, None] * 1 + off_n[None, :] * BLOCK_K
        z_offset = off_m[:, None] * BLOCK_N + off_n[None, :] * 1
        Zs = Z + z_offset
        x = tl.load(Xs)
        y = tl.load(Ys)
        z = tl.dot(x, y)
        # additional computations to give more diverse context to backend scheduler
        z += z_offset
        tl.store(Zs, z)

    M = 128
    N = 128
    K = 128

    pgm_default = kernel.warmup(torch.float32, torch.float32, torch.float32, M, N, K, grid=(1, ))
    pgm_custom = kernel.warmup(torch.float32, torch.float32, torch.float32, M, N, K,
                               schedule_hint="memory-bound-attention", grid=(1, ))

    # check that option affects only llvm backend
    listing_default = pgm_default.asm["llir"].split("\n")
    listing_custom = pgm_custom.asm["llir"].split("\n")

    # check that llir is identical except some possible differences in attributes
    assert len(listing_custom) == len(listing_default)
    for lineId in range(len(listing_custom)):
        if "attribute" in listing_custom[lineId] and "attribute" in listing_default[lineId]:
            continue
        assert listing_custom[lineId] == listing_default[lineId]
    assert pgm_default.asm["amdgcn"] != pgm_custom.asm["amdgcn"]
