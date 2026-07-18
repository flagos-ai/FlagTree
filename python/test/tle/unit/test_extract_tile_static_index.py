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

import torch
import triton
import triton.language as tl
import triton.experimental.tle.language as tle
import pytest


@triton.jit
def extract_tile_kernel(x_ptr, out_ptr, M: tl.constexpr, N: tl.constexpr):
    # Set M, N as input matrix dimensions
    offs_m = tl.arange(0, M)
    offs_n = tl.arange(0, N)
    x = tl.load(x_ptr + offs_m[:, None] * N + offs_n[None, :])

    # Extract a 128x128 tile starting from index [1, 1]
    # Note: index refers to the tile position (e.g., index [1, 1] for 128x128 tiles
    # starts at row 128 and column 128)
    tile = tle.extract_tile(x, index=[1, 1], tile_shape=[128, 128])

    # Store the 128x128 extracted tile into the output pointer
    out_offs_m = tl.arange(0, 128)
    out_offs_n = tl.arange(0, 128)
    tl.store(out_ptr + out_offs_m[:, None] * 128 + out_offs_n[None, :], tile)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for this test")
def test_extract_tile_kernel():
    # Set matrix dimensions
    M, N = 512, 512
    # Create input tensor with sequential values
    x = torch.arange(M * N, device='cuda', dtype=torch.float32).reshape(M, N)
    # Prepare output buffer for a 128x128 result
    out = torch.zeros((128, 128), device='cuda', dtype=torch.float32)

    # Launch kernel with a single program (grid size 1)
    extract_tile_kernel[(1, )](x, out, M, N)

    # Verification:
    # Since index=[1, 1] and tile_shape=[128, 128], the extraction starts at
    # row 1 * 128 and column 1 * 128.
    expected = x[128:256, 128:256]

    assert torch.allclose(out, expected), "The extracted tile does not match the expected slice!"
