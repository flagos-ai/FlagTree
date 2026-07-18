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

# fmt: off

import numpy as np
import pytest
import torch
import triton
import triton.language as tl

from triton.language.extra import libdevice


# -----------------------
# test extern functions
# -----------------------


@triton.jit
def tanh_kernel(
    x_ptr,
    y_ptr,
    n_elements,
    direct_import: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)

    if direct_import:
        y = libdevice.tanh(x)
    else:
        y = tl.extra.libdevice.tanh(x)

    tl.store(y_ptr + offsets, y, mask=mask)


@pytest.mark.parametrize("direct_import", [False, True])
@pytest.mark.parametrize("dtype_str", ['float32', 'float64'])
def test_math_extern(dtype_str, direct_import):

    if not torch.cuda.is_available():
        pytest.skip("Test requires CUDA target.")
        return

    torch.manual_seed(42)

    x = torch.randn((100,), dtype=getattr(torch, dtype_str), device="cuda")

    y_tri = torch.empty_like(x)
    tanh_kernel[(1, )](x, y_tri, x.shape[0], direct_import, BLOCK_SIZE=128)

    y_ref = torch.tanh(x)
    np.testing.assert_allclose(y_ref.cpu().numpy(), y_tri.cpu().numpy(), rtol=0, atol=1.0e-6)
