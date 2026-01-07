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

"""
Fused Softmax
=============
"""

import os

import torch
import torch_npu
import triton
import triton.language as tl
from triton.testing import do_bench_npu


# split_params={"x": "XBLOCK"}, tiling_params={"x": "XBLOCK_SUB"}, low_dims=["y"]
# persistent_reduction=False, dual_reduction=False
@triton.autotune(
    configs=[],
    hints={"enable_ascend_autotune": True},
    key=["n_rows", "n_cols"],
)
@triton.jit
def softmax_kernel(
    output_ptr,
    input_ptr,
    input_row_stride,
    output_row_stride,
    n_rows,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
    XBLOCK: tl.constexpr,
    XBLOCK_SUB: tl.constexpr,
):
    # starting row of the program
    row_start = tl.program_id(0) * XBLOCK
    for row_idx in tl.range(0, XBLOCK, XBLOCK_SUB):
        # The stride represents how much we need to increase the pointer to advance 1 row
        row_offsets = row_start + row_idx + tl.arange(0, XBLOCK_SUB)[:, None]
        col_offsets = tl.arange(0, BLOCK_SIZE)[None, :]
        xmask = row_offsets < n_rows
        ymask = col_offsets < n_cols
        mask = xmask & ymask
        input_ptrs = input_ptr + (row_offsets * input_row_stride + col_offsets)
        # Load the row into SRAM, using a mask since BLOCK_SIZE may be > than n_cols
        row = tl.load(input_ptrs, mask=mask, other=-float("inf"))
        # Subtract maximum for numerical stability
        row_minus_max = row - tl.max(row, axis=1).reshape(XBLOCK_SUB, 1).broadcast_to(
            XBLOCK_SUB, BLOCK_SIZE
        )
        # Note that exponentiation in Triton is fast but approximate (i.e., think __expf in CUDA)
        numerator = tl.exp(row_minus_max)
        denominator = (
            tl.sum(numerator, axis=1)
            .reshape(XBLOCK_SUB, 1)
            .broadcast_to(XBLOCK_SUB, BLOCK_SIZE)
        )
        softmax_output = numerator / denominator
        # Write back output to DRAM
        output_ptrs = output_ptr + (row_offsets * output_row_stride + col_offsets)
        tl.store(output_ptrs, softmax_output, mask=mask)


def softmax_torch(x):
    return torch.softmax(x, axis=-1)


def softmax_autotune(x):
    n_rows, n_cols = x.shape
    BLOCK_SIZE = n_cols

    # Allocate output
    y = torch.empty_like(x)
    grid = lambda meta: (triton.cdiv(n_rows, meta["XBLOCK"]), 1, 1)
    # Create a number of persistent programs.
    softmax_kernel[grid](
        y, x, x.stride(0), y.stride(0), n_rows, n_cols, BLOCK_SIZE=BLOCK_SIZE
    )
    return y


def test_softmax(shape, dtype):
    os.environ["TRITON_BENCH_METHOD"] = (
        "npu"  # use torch_npu.profiler to get calculating time
    )
    x = torch.randn(shape, dtype=dtype, device="npu")

    y_torch = softmax_torch(x)
    y_triton = softmax_autotune(x)
    assert torch.allclose(y_triton, y_torch)

    time_eager = do_bench_npu(lambda: softmax_torch(x))
    time_triton = do_bench_npu(lambda: softmax_autotune(x))
    assert (time_eager / time_triton) >= 0.8
    print(f"Fused Softmax {shape} {dtype} PASSED!")


if __name__ == "__main__":
    test_softmax((16896, 1024), torch.float32)
