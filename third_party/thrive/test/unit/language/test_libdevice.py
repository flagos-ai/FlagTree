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

from triton.language.extra import libdevice
from triton.language.extra.libdevice import fast_dividef as my_fast_dividef


@pytest.mark.skip(reason="[Thrive] custom implementations of bessel functions is not supported on thrive")
@pytest.mark.parametrize("dtype_str", ["float32", "float64"])
@pytest.mark.parametrize(
    "libdevice_fn, torch_special_fn",
    [
        ("j0", "bessel_j0"),
        ("j1", "bessel_j1"),
        ("y0", "bessel_y0"),
        ("y1", "bessel_y1"),
        ("cyl_bessel_i0", "i0"),
        ("cyl_bessel_i1", "i1"),
    ],
)
def test_bessel(dtype_str, libdevice_fn, torch_special_fn, device):
    SIZE = 128
    dtype = getattr(torch, dtype_str)

    torch.manual_seed(42)
    x = torch.randn((SIZE, ), dtype=dtype).to(device)
    y_exp = torch.empty((SIZE, ), dtype=dtype).to(device)
    y_ref = getattr(torch.special, torch_special_fn)(x)

    @triton.jit
    def kernel(in_p, out_p, fn: tl.constexpr, SIZE: tl.constexpr):
        off = tl.arange(0, SIZE)
        x = tl.load(in_p + off)
        res = getattr(libdevice, fn)(x)
        tl.store(out_p + off, res)

    kernel[(1, )](x, y_exp, fn=libdevice_fn, SIZE=SIZE, num_warps=4, num_ctas=1)

    torch.testing.assert_close(y_ref, y_exp, equal_nan=True)


def test_libdevice_rename(device):
    # mark the import as used by this test
    _ = my_fast_dividef

    @triton.jit
    def triton_copy(in_ptr, out_ptr, BLOCK_SIZE: tl.constexpr):
        offsets = tl.arange(0, BLOCK_SIZE)
        data = tl.load(in_ptr + offsets)
        tl.store(out_ptr + offsets, data)

    BLOCK_SIZE = 256
    inp = torch.randn(BLOCK_SIZE).to(device)
    out = torch.empty_like(inp)

    triton_copy[(1, )](inp, out, BLOCK_SIZE)
