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

import torch
import torch_npu
import triton
import triton.language as tl
import numpy as np
import test_common
import pytest

@pytest.mark.skip(reason="waiting for the compiler to support.")
@pytest.mark.parametrize("src_shape, indices_shape, axis", [
    ([4, 4], [8, 4], 0),
    ([4, 64], [4, 32], 1),
    ([128, 64], [128, 128], 1),
])
def test_gather(src_shape, indices_shape, axis):

    @triton.jit
    def gather_kernel(src_ptr, idx_ptr, out_ptr, axis: tl.constexpr, src_dim0: tl.constexpr, src_dim1: tl.constexpr,
                      src_stride0: tl.constexpr, src_stride1: tl.constexpr, idx_dim0: tl.constexpr,
                      idx_dim1: tl.constexpr, idx_stride0: tl.constexpr, idx_stride1: tl.constexpr,
                      out_dim0: tl.constexpr, out_dim1: tl.constexpr, out_stride0: tl.constexpr,
                      out_stride1: tl.constexpr):
        src_offs = (tl.arange(0, src_dim0)[:, None] * src_stride0 + tl.arange(0, src_dim1)[None, :] * src_stride1)
        src = tl.load(src_ptr + src_offs)

        idx_offs = (tl.arange(0, idx_dim0)[:, None] * idx_stride0 + tl.arange(0, idx_dim1)[None, :] * idx_stride1)
        idx = tl.load(idx_ptr + idx_offs)

        out = tl.gather(src, idx, axis)

        out_offs = (tl.arange(0, out_dim0)[:, None] * out_stride0 + tl.arange(0, out_dim1)[None, :] * out_stride1)
        tl.store(out_ptr + out_offs, out)

    def triton_gather(src: torch.Tensor, axis: int, indices: torch.Tensor):
        output = torch.empty(indices.shape, dtype=src.dtype, device=src.device)
        gather_kernel[(1, )](src, indices, output, axis,
                             src.shape[0], src.shape[1],
                             src.stride(0), src.stride(1),
                             indices.shape[0], indices.shape[1],
                             indices.stride(0), indices.stride(1),
                             output.shape[0], output.shape[1],
                             output.stride(0), output.stride(1))
        return output

    DEV = "npu"
    src = torch.randn(src_shape, device=DEV)
    indices = torch.randint(0, src.shape[axis], indices_shape, device=DEV)
    ref = torch.gather(src, axis, indices)
    result = triton_gather(src, axis, indices)
    torch.testing.assert_close(result, ref, rtol=0, atol=0)

if __name__ == "__main__":
    test_gather([4, 64], [4, 32], 1)
    print("success: test_gather")
