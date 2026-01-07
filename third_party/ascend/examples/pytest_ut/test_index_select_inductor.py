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

import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

import torch
import torch_npu

import test_common


@triton.jit
def triton_unk_fused_embedding_0(in_ptr0, in_ptr1, out_ptr0, y0_numel, x1_numel, Y0BLOCK: tl.constexpr, Y0BLOCK_SUB: tl.constexpr):
    x1_numel = 64
    X1BLOCK_SUB: tl.constexpr = 64
    y0_offset = tl.program_id(0) * Y0BLOCK
    base_y0 = tl.arange(0, Y0BLOCK_SUB)
    loops_y0 = (Y0BLOCK + Y0BLOCK_SUB - 1) // Y0BLOCK_SUB
    base_x1 = tl.arange(0, X1BLOCK_SUB)
    for loop_y0 in range(loops_y0):
        y0 = y0_offset + (loop_y0 * Y0BLOCK_SUB) + base_y0[:, None]
        y0_mask = y0 < min(Y0BLOCK + y0_offset, y0_numel)
        x1 = base_x1[None, :]
        tmp0 = tl.load(in_ptr0 + (y0), y0_mask)
        tmp6 = tl.load(in_ptr1 + (x1 + 64 * tmp0))
        tl.store(out_ptr0 + (x1 + 64 * y0), tmp6, y0_mask)


def torch_embedding_impl(in_ptr0, in_ptr1):
    indices_flat = in_ptr0.flatten() # [6400]
    vocab_size = 1353406
    out_flat = torch.embedding(in_ptr1, indices_flat)  # [6400, 16]
    out = out_flat.view(128, 50, 64)
    return out


def test_kernel():
    arg0_1 = torch.randint(0, 1353406, (128, 50), device='npu', dtype=torch.int64)
    arg1_1 = torch.randn((1353406, 64), device='npu', dtype=torch.float32)
    buf0 = torch.empty((128, 50, 64), device='npu', dtype=torch.float32)

    y0_numel = 128 * 50  
    x1_numel = 64
    Y0BLOCK = 256 
    Y0BLOCK_SUB = 16 

    grid = (triton.cdiv(y0_numel, Y0BLOCK),)

    print(f"Grid size: {grid}")
    print(f"Total programs: {grid[0]}")
    print(f"y0_numel: {y0_numel}, Y0BLOCK: {Y0BLOCK}")

    triton_unk_fused_embedding_0[grid](
        arg0_1,
        arg1_1,
        buf0,
        y0_numel,
        x1_numel,
        Y0BLOCK=Y0BLOCK,
        Y0BLOCK_SUB=Y0BLOCK_SUB,
    )

    print(f"Output shape: {buf0.shape}")
    print(f"Output sample: {buf0[0, 0, :]}")

    expected = torch_embedding_impl(arg0_1, arg1_1)

    test_common.validate_cmp("float32", buf0, expected)


if __name__ == "__main__":
    test_kernel()