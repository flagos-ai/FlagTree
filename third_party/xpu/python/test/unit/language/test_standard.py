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

import triton
import pytest
import torch
import triton.language as tl

from test_core import _test_binary, int_dtypes, uint_dtypes, float_dtypes, numpy_random

# ---------------
# test maximum/minimum ops
# ---------------


# TODO: Tests with unsigned integers failed at compilation stage.
@pytest.mark.interpreter
@pytest.mark.parametrize("dtype", int_dtypes + uint_dtypes + float_dtypes + ["bfloat16"])
@pytest.mark.parametrize("op", ["maximum", "minimum"])
def test_maximum_minium(dtype, op, device):
    expr = f'tl.{op}(x, y)'
    numpy_expr = f'np.{op}(x, y)'
    _test_binary(dtype, dtype, expr, numpy_expr, device=device)


# ---------------
# test sort op
# ---------------


@pytest.mark.skip("Skip for kunlunxin")
@pytest.mark.interpreter
@pytest.mark.parametrize("M, N", [[1, 512], [8, 64], [256, 16], [512, 8]])
@pytest.mark.parametrize("descending", [False, True])
@pytest.mark.parametrize("dtype_str", ['int32', 'float16', 'float32'])
def test_sort(M, N, descending, dtype_str, device):

    @triton.jit
    def sort_kernel(X, Z, N: tl.constexpr, M: tl.constexpr, descending: tl.constexpr):
        offx = tl.arange(0, M)
        offy = tl.arange(0, N) * M
        off2d = offx[None, :] + offy[:, None]
        x = tl.load(X + off2d)
        x = tl.sort(x, descending=descending)
        tl.store(Z + off2d, x)

    x = numpy_random((N, M), dtype_str=dtype_str)
    x = torch.from_numpy(x).to(device)
    y = torch.sort(x, descending=descending)[0]
    z = torch.empty_like(x)
    sort_kernel[(1, )](x, z, N, M, descending, num_warps=8)
    assert (y == z).all(), (y, z)


# ---------------
# test flip op
# ---------------


@pytest.mark.skip("Skip for kunlunxin")
@pytest.mark.interpreter
@pytest.mark.parametrize("M, N", [[1, 512], [8, 64], [256, 16], [512, 8]])
@pytest.mark.parametrize("dtype_str", ['int32', 'float16', 'float32'])
def test_flip(M, N, dtype_str, device):

    @triton.jit
    def flip_kernel(X, Z, N: tl.constexpr, M: tl.constexpr):
        offx = tl.arange(0, M)
        offy = tl.arange(0, N) * M
        off2d = offx[None, :] + offy[:, None]
        x = tl.load(X + off2d)
        x = tl.flip(x)
        tl.store(Z + off2d, x)

    x = numpy_random((N, M), dtype_str=dtype_str)
    x = torch.from_numpy(x).to(device)
    y = torch.flip(x, (1, ))
    z = torch.empty_like(x, device=device)
    flip_kernel[(1, )](x, z, N, M, num_warps=8)
    assert (y == z).all(), (y, z)
