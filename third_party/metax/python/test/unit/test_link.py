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

import sys

import pytest
import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

from triton._C.libtriton import llvm


@triton.jit(noinline=True)
def add_one(x_ptr, SQRT: tl.constexpr) -> None:
    x = tl.load(x_ptr)
    if SQRT:
        x = libdevice.sqrt(x)
    tl.store(x_ptr, x + 1.0)


@triton.jit
def add_one_indirect(x_ptr, SQRT: tl.constexpr) -> None:
    add_one(x_ptr, SQRT)


@pytest.mark.parametrize("use_libdevice", (False, True))
@pytest.mark.parametrize("kernel", (add_one, add_one_indirect))
def test_link_extern_libs(use_libdevice, kernel):
    link_called: bool = False

    def callback(frame, event, arg):
        nonlocal link_called
        if event == "c_call" and arg is llvm.link_extern_libs:
            link_called = True

    x = torch.ones((1, ), device="cuda")
    prior_callback = sys.getprofile()
    try:
        sys.setprofile(callback)
        with (compilation := triton.knobs.compilation).scope():
            compilation.always_compile = True
            kernel[(1, )](x, SQRT=use_libdevice)
    finally:
        sys.setprofile(prior_callback)

    assert (link_called == use_libdevice)
