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

import triton
import triton.language as tl
from triton.compiler import ASTSource
import triton.extension.buffer.language as bl
import triton.language.extra.cann.extension as al

target = triton.runtime.driver.active.get_current_target()


@triton.jit
def to_buffer():
    a = tl.full((32, 2, 4), 0, dtype=tl.int64)
    a_buf = bl.to_buffer(a)
    b = tl.full((32, 2, 4), 0, dtype=tl.int64)
    b_buf = bl.to_buffer(b, al.ascend_address_space.UB)
    c = tl.full((32, 2, 4), 0, dtype=tl.int64)
    c_buf = bl.to_buffer(c, al.ascend_address_space.L1)
    d = tl.full((32, 2, 4), 0, dtype=tl.int64)
    d_buf = bl.to_buffer(d, al.ascend_address_space.L0A)
    e = tl.full((32, 2, 4), 0, dtype=tl.int64)
    e_buf = bl.to_buffer(e, al.ascend_address_space.L0B)
    f = tl.full((32, 2, 4), 0, dtype=tl.int64)
    f_buf = bl.to_buffer(f, al.ascend_address_space.L0C)


def test_to_buffer():
    src = ASTSource(
        fn=to_buffer,
        constants={},
        signature={},
    )
    triton.compile(src=src, target=target)
