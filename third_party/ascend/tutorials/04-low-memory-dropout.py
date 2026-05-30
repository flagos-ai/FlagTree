# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
# Copyright 2018-2020 Philippe Tillet
# Copyright 2020-2022 OpenAI
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
Low-Memory Dropout
==================
"""

import tabulate
import torch
import torch_npu

import triton
import triton.language as tl

DEV = "npu"


@triton.jit
def _dropout(
    x_ptr,  # pointer to the input
    x_keep_ptr,  # pointer to a mask of 0s and 1s
    output_ptr,  # pointer to the output
    n_elements,  # number of elements in the `x` tensor
    p,  # probability that an element of `x` is changed to zero
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    # Load data
    x = tl.load(x_ptr + offsets, mask=mask)
    x_keep = tl.load(x_keep_ptr + offsets, mask=mask)
    # The line below is the crucial part, described in the paragraph above!
    output = tl.where(x_keep != 0, x / (1 - p), 0.0)
    # Write-back output
    tl.store(output_ptr + offsets, output, mask=mask)


def dropout(x, x_keep, p):
    output = torch.empty_like(x)
    assert x.is_contiguous()
    n_elements = x.numel()

    def grid(meta):
        return (triton.cdiv(n_elements, meta['BLOCK_SIZE']), )

    _dropout[grid](x, x_keep, output, n_elements, p, BLOCK_SIZE=1024)
    return output


@triton.jit
def _seeded_dropout(
    x_ptr,
    output_ptr,
    n_elements,
    p,
    seed,
    BLOCK_SIZE: tl.constexpr,
):
    # compute memory offsets of elements handled by this instance
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    # load data from x
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    # randomly prune it
    random = tl.rand(seed, offsets)
    x_keep = random > p
    # write-back
    output = tl.where(x_keep, x / (1 - p), 0.0)
    tl.store(output_ptr + offsets, output, mask=mask)


def seeded_dropout(x, p, seed):
    output = torch.empty_like(x)
    assert x.is_contiguous()
    n_elements = x.numel()

    def grid(meta):
        return (triton.cdiv(n_elements, meta['BLOCK_SIZE']), )

    _seeded_dropout[grid](x, output, n_elements, p, seed, BLOCK_SIZE=1024)
    return output


def test():
    # Input tensor
    x = torch.randn(size=(10, ), device=DEV)
    # Dropout mask
    p = 0.5
    x_keep = (torch.rand(size=(10, ), device=DEV) > p).to(torch.int32)
    #
    output = dropout(x, x_keep=x_keep, p=p)
    print(tabulate.tabulate([
        ["input"] + x.tolist(),
        ["keep mask"] + x_keep.tolist(),
        ["output"] + output.tolist(),
    ]))

    x = torch.randn(size=(10, ), device=DEV)
    # Compare this to the baseline - dropout mask is never instantiated!
    output = seeded_dropout(x, p=0.5, seed=123)
    output2 = seeded_dropout(x, p=0.5, seed=123)
    output3 = seeded_dropout(x, p=0.5, seed=512)

    print(
        tabulate.tabulate([
            ["input"] + x.tolist(),
            ["output (seed = 123)"] + output.tolist(),
            ["output (seed = 123)"] + output2.tolist(),
            ["output (seed = 512)"] + output3.tolist(),
        ]))


if __name__ == "__main__":
    test()
