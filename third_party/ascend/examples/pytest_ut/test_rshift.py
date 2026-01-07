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

import pytest
import triton
import triton.language as tl
import time
import test_common
import torch
import torch_npu


def standard_unary(x0, dtype):
    res = x0 >> 2
    return res


def standard_binary(x0, y0, dtype):
    res = x0 + y0
    return res


@triton.jit
def triton_elementwise_unary(in_ptr0, out_ptr0, N: tl.constexpr, NUMEL: tl.constexpr):
    idx_block = tl.arange(0, NUMEL)
    x = tl.load(in_ptr0 + idx_block, mask=idx_block < N)
    tmp = tl.cast(2, tl.int8)
    ret = x >> tmp
    tl.store(out_ptr0 + idx_block, ret, mask=idx_block < N)


@triton.jit
def triton_elementwise_binary(in_ptr0, in_ptr1, out_ptr0, N: tl.constexpr, NUMEL: tl.constexpr):
    idx_block = tl.arange(0, NUMEL)
    x = tl.load(in_ptr0 + idx_block, mask=idx_block < N)
    y = tl.load(in_ptr1 + idx_block, mask=idx_block < N)
    ret = x + y
    tl.store(out_ptr0 + idx_block, ret, mask=idx_block < N)


types = [
    # (torch.float32, 'float32'),
    # (torch.float16, 'float16'),
    # (torch.bfloat16, 'bfloat16'),
    (torch.int8, 'int8'),
    # (torch.int16, 'int16'),
    # (torch.int32, 'int32'),
    # (torch.int64, 'int64'),
]

shapes = [
    (3, 32),
    (-32, 32),
    (37, 64),
    (-256, 256),
    (781, 1024),
]

map_for_64_t = {37: 31}


@pytest.mark.parametrize('dtype,sigtype', types)
@pytest.mark.parametrize('N,NUMEL', shapes)
def test_elementwsie_common(dtype, sigtype, N, NUMEL):
    N = (-N) // torch.tensor(0, dtype=dtype).element_size() if N < 0 else N

    if sigtype == "int64":
        N = map_for_64_t[N] if N in map_for_64_t else N

    print(f"elementwise : ({N},) {dtype} {sigtype}")

    x0 = test_common.generate_tensor(shape=(N,), dtype=sigtype)

    ans = standard_unary(x0, dtype)
    x0 = x0.npu()
    # print(ans)

    out = torch.zeros((N,), dtype=dtype).npu()
    triton_elementwise_unary[1, 1, 1](x0, out, N=N, NUMEL=NUMEL, debug=True)
    # print(out)

    test_common.validate_cmp(sigtype, out, ans)
