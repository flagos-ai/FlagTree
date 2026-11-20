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

import time

import pytest
import torch
import torch_npu

import triton
import triton.language as tl
from triton.runtime.libentry import libentry

DEV = "npu"
DTYPE = torch.float32
SEQ_LEN = 2 * 1024
device = torch.npu.current_device()
stream = torch.npu.current_stream(device)


def benchmark(func):
    warmup = 10
    repeat = 100

    def wrapper(*args, **kwargs):
        #
        for _ in range(warmup):
            result = func(*args, **kwargs)
        stream.synchronize()
        #
        start_time = time.perf_counter_ns()
        for _ in range(repeat):
            result = func(*args, **kwargs)
        stream.synchronize()
        end_time = time.perf_counter_ns()
        #
        start_time = start_time * 1e-3
        end_time = end_time * 1e-3
        elapsed_time = (end_time - start_time) / repeat
        return (result, elapsed_time)

    return wrapper


@libentry()
@triton.jit
def softmax_kernel(
    output_ptr,
    input_ptr,
    input_row_stride,
    output_row_stride,
    n_rows,
    n_cols,
    XBLOCK: tl.constexpr,
    XBLOCK_SUB: tl.constexpr,
    RBLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    row_start = pid * XBLOCK
    rblk_idx = tl.arange(0, XBLOCK_SUB)
    col_idx = tl.arange(0, RBLOCK)
    for row_idx in tl.range(0, XBLOCK, XBLOCK_SUB):
        row_offsets = row_start + row_idx + rblk_idx[:, None]
        col_offsets = col_idx[None, :]
        xmask = row_offsets < n_rows
        ymask = col_offsets < n_cols
        mask = xmask & ymask
        input_idx = row_offsets * input_row_stride + col_offsets
        input_ptrs = input_ptr + input_idx
        row = tl.load(input_ptrs, mask=mask, other=-float('inf'))
        row_minus_max = row - tl.max(row, axis=1).reshape(XBLOCK_SUB, 1)
        numerator = tl.exp(row_minus_max)
        denominator = tl.sum(numerator, axis=1).reshape(XBLOCK_SUB, 1)
        softmax_output = numerator / denominator
        output_ptrs = output_ptr + (row_offsets * output_row_stride + col_offsets)
        tl.store(output_ptrs, softmax_output, mask=mask)


@benchmark
def torch_func(x0: torch.Tensor):
    m = torch.nn.Softmax(dim=1)
    return m(x0)


@benchmark
def triton_func(y0: torch.Tensor, x0: torch.Tensor):
    n_rows, n_cols = x0.shape
    ncore = 40
    xs = (n_rows + ncore - 1) // ncore
    xss = min(xs, 5)
    softmax_kernel[(ncore, 1, 1)](
        y0,
        x0,
        x0.stride(0),
        y0.stride(0),
        n_rows,
        n_cols,
        XBLOCK=xs,
        XBLOCK_SUB=xss,
        RBLOCK=n_cols,
    )
    return y0


@pytest.mark.parametrize("batch", [1000 * x for x in range(1, 16 + 1)])
def test_demo_libentry_softmax(batch):
    torch.manual_seed(0)
    x = torch.rand((batch, SEQ_LEN), dtype=DTYPE, device=DEV)
    y = torch.empty_like(x)

    torch_out, _ = torch_func(x)
    triton_out, _ = triton_func(y, x)

    torch.testing.assert_close(triton_out, torch_out)
