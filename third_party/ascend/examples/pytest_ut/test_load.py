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

import triton
import triton.language as tl
import numpy as np
import torch
import pytest
import test_common

# eg: pytest -v test.py::test_add
#############################

@triton.jit
def triton_load_store(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr, XBLOCK_SUB: tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    for xoffset_sub in range(0, XBLOCK, XBLOCK_SUB):
        xindex = xoffset + xoffset_sub + tl.arange(0, XBLOCK_SUB)[:]
        xmask = xindex < xnumel
        x0 = xindex
        tmp0 = tl.load(in_ptr0 + (x0), xmask)
        tmp2 = tmp0
        tl.store(out_ptr0 + (xindex), tmp2, xmask)


# require: all data (4d and 5d) can be placed into but without ub overflow
@triton.jit
def triton_load_store_multi_d(
    in_ptr0, out_ptr0, 
    BLOCK_0: tl.constexpr, BLOCK_1: tl.constexpr, BLOCK_2: tl.constexpr, BLOCK_3: tl.constexpr, BLOCK_4: tl.constexpr,
    SHAPE_0: tl.constexpr, SHAPE_1: tl.constexpr, SHAPE_2: tl.constexpr, SHAPE_3: tl.constexpr, SHAPE_4: tl.constexpr,
    STRIDE_0: tl.constexpr, STRIDE_1: tl.constexpr, STRIDE_2: tl.constexpr, STRIDE_3: tl.constexpr, STRIDE_4: tl.constexpr
):
    offsets = tl.program_id(0)

    offsets = offsets + tl.arange(0, BLOCK_0) * STRIDE_0
    masks = tl.arange(0, BLOCK_0) < SHAPE_0
    if (BLOCK_1 * BLOCK_2 * BLOCK_3 * BLOCK_4) > 1:
        offsets = offsets[:, None] + tl.arange(0, BLOCK_1)[None, :] * STRIDE_1
        masks = masks[:, None] & (tl.arange(0, BLOCK_1)[None, :] < SHAPE_1)
    if (BLOCK_2 * BLOCK_3 * BLOCK_4) > 1:
        offsets = offsets[:, :, None] + tl.arange(0, BLOCK_2)[None, None, :] * STRIDE_2
        masks = masks[:, :, None] & (tl.arange(0, BLOCK_2)[None, None, :] < SHAPE_2)
    if (BLOCK_3 * BLOCK_4) > 1:
        offsets = offsets[:, :, :, None] + tl.arange(0, BLOCK_3)[None, None, None, :] * STRIDE_3
        masks = masks[:, :, :, None] & (tl.arange(0, BLOCK_3)[None, None, None, :] < SHAPE_3)
    if BLOCK_4 > 1:
        offsets = offsets[:, :, :, :, None] + tl.arange(0, BLOCK_4)[None, None, None, None, :] * STRIDE_4
        masks = masks[:, :, :, :, None] & (tl.arange(0, BLOCK_4)[None, None, None, None, :] < SHAPE_4)

    tmp_in = tl.load(in_ptr0 + offsets, masks)
    tmp_out = tmp_in
    tl.store(out_ptr0 + offsets, tmp_out, masks)


@pytest.mark.parametrize('param_list',
                         [
                             ['float32', (2, 4096, 8), 2, 32768, 1024],
                             ['float16', (2, 4096, 8), 2, 32768, 1024],
                             ['int8', (2, 4096, 8), 2, 32768, 1024],
                             ['float32', (8, 8, 4), 2, 128, 64],
                             ['float16', (8, 8, 4), 2, 128, 64],
                             ['int8', (8, 8, 4), 2, 128, 64],
                             ['int8', (8, 7, 4), 2, 128, 64],

                         ]
                         )
def test_load_store(param_list):
    dtype, shape, ncore, xblock, xblock_sub = param_list
    x0 = test_common.generate_tensor(shape, dtype).npu()
    y_ref = x0
    y_cal = test_common.generate_tensor(shape, dtype).npu()
    triton_load_store[(ncore, )](x0, y_cal, x0.numel(), xblock, xblock_sub)
    test_common.validate_cmp(dtype, y_cal, y_ref)


@pytest.mark.parametrize('param_list',
                         [
                             ['float32', (8, 4, 16, 16)],
                             ['float16', (8, 4, 16, 16)],
                             ['int8', (8, 4, 16, 16)],
                             ['float32', (8, 8, 4, 4)],
                             ['float16', (8, 8, 4, 4)],
                             ['int8', (8, 8, 4, 4)],
                             ['float32', (3, 8, 2, 16, 16)],
                             ['float16', (3, 8, 2, 16, 16)],
                             ['int8', (9, 8, 8, 16, 16)],
                             ['float32', (11, 8, 8, 4, 4)],
                             ['float16', (11, 8, 8, 4, 4)],
                             ['int8', (11, 8, 8, 4, 4)],
                         ]
                         )
def test_load_store_multi_d(param_list):
    dtype, shape = param_list
    x0 = test_common.generate_tensor(shape, dtype).npu()
    y_expect = x0
    y_actual = test_common.generate_tensor(shape, dtype).npu()

    blocks = list(x0.size())
    shapes = list(x0.stride())
    while len(blocks) < 5:
        blocks.append(1)
        shapes.append(1)
    triton_load_store_multi_d[(1, )](x0, y_actual, *blocks, *blocks, *shapes)
    test_common.validate_cmp(dtype, y_actual, y_expect)
