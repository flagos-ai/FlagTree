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
import pytest
import test_common
import itertools
import random, time


@triton.jit
def triton_store_new_mask(
    in_ptr0,
    out_ptr0,
    mask_ptr,
    cache_modifier: tl.constexpr,
    eviction_policy: tl.constexpr,
    BLOCK_0: tl.constexpr,
    BLOCK_1: tl.constexpr,
    BLOCK_2: tl.constexpr,
    BLOCK_3: tl.constexpr,
    BLOCK_4: tl.constexpr,
    BLOCK_5: tl.constexpr,
    BLOCK_6: tl.constexpr,
    BLOCK_7: tl.constexpr,
    SHAPE_0: tl.constexpr,
    SHAPE_1: tl.constexpr,
    SHAPE_2: tl.constexpr,
    SHAPE_3: tl.constexpr,
    SHAPE_4: tl.constexpr,
    SHAPE_5: tl.constexpr,
    SHAPE_6: tl.constexpr,
    SHAPE_7: tl.constexpr,
    STRIDE_0: tl.constexpr,
    STRIDE_1: tl.constexpr,
    STRIDE_2: tl.constexpr,
    STRIDE_3: tl.constexpr,
    STRIDE_4: tl.constexpr,
    STRIDE_5: tl.constexpr,
    STRIDE_6: tl.constexpr,
    STRIDE_7: tl.constexpr,
):
    offsets = tl.program_id(0)

    offsets = offsets + tl.arange(0, BLOCK_0) * STRIDE_0
    masks = tl.arange(0, BLOCK_0) < SHAPE_0

    if (BLOCK_1 * BLOCK_2 * BLOCK_3 * BLOCK_4 * BLOCK_5 * BLOCK_6 * BLOCK_7) > 1:
        offsets = offsets[:, None] + tl.arange(0, BLOCK_1)[None, :] * STRIDE_1
        masks = masks[:, None] & (tl.arange(0, BLOCK_1)[None, :] < SHAPE_1)
    if (BLOCK_2 * BLOCK_3 * BLOCK_4 * BLOCK_5 * BLOCK_6 * BLOCK_7) > 1:
        offsets = offsets[:, :, None] + tl.arange(0, BLOCK_2)[None, None, :] * STRIDE_2
        masks = masks[:, :, None] & (tl.arange(0, BLOCK_2)[None, None, :] < SHAPE_2)
    if (BLOCK_3 * BLOCK_4 * BLOCK_5 * BLOCK_6 * BLOCK_7) > 1:
        offsets = offsets[:, :, :, None] + tl.arange(0, BLOCK_3)[None, None, None, :] * STRIDE_3
        masks = masks[:, :, :, None] & (tl.arange(0, BLOCK_3)[None, None, None, :] < SHAPE_3)
    if (BLOCK_4 * BLOCK_5 * BLOCK_6 * BLOCK_7) > 1:
        offsets = offsets[:, :, :, :, None] + tl.arange(0, BLOCK_4)[None, None, None, None, :] * STRIDE_4
        masks = masks[:, :, :, :, None] & (tl.arange(0, BLOCK_4)[None, None, None, None, :] < SHAPE_4)
    if (BLOCK_5 * BLOCK_6 * BLOCK_7) > 1:
        offsets = offsets[:, :, :, :, :, None] + tl.arange(0, BLOCK_5)[None, None, None, None, None, :] * STRIDE_5
        masks = masks[:, :, :, :, :, None] & (tl.arange(0, BLOCK_5)[None, None, None, None, None, :] < SHAPE_5)
    if (BLOCK_6 * BLOCK_7) > 1:
        offsets = offsets[:, :, :, :, :, :,
                          None] + tl.arange(0, BLOCK_6)[None, None, None, None, None, None, :] * STRIDE_6
        masks = masks[:, :, :, :, :, :, None] & (tl.arange(0, BLOCK_6)[None, None, None, None, None, None, :] < SHAPE_6)
    if BLOCK_7 > 1:
        offsets = offsets[:, :, :, :, :, :, :,
                          None] + tl.arange(0, BLOCK_7)[None, None, None, None, None, None, None, :] * STRIDE_7
        masks = masks[:, :, :, :, :, :, :, None] & (tl.arange(0, BLOCK_7)[None, None, None, None, None, None, None, :]
                                                    < SHAPE_7)

    mask = tl.load(mask_ptr + offsets) != 0
    tmp_in = tl.load(in_ptr0 + offsets)
    tmp_out = tmp_in
    tl.store(out_ptr0 + offsets, tmp_out, mask=mask, cache_modifier=cache_modifier, eviction_policy=eviction_policy)


testlist = [
    (15, 2, 2, 2, 3, 2, 1),
]

cache_modifier = [None, ".wb", ".cs", ".cg", ".wt"]
eviction_policy = [None, "evict_first", "evict_last"]

seed = int(time.strftime('%Y%m%d%H'))
random.seed(seed)
all_back_combinations = list(itertools.product(cache_modifier, eviction_policy))
testlist1 = [random.choice(all_back_combinations)]


@pytest.mark.parametrize('shape', testlist)
@pytest.mark.parametrize('dtype', ["bfloat16"])
@pytest.mark.parametrize('cache_modifier, eviction_policy', testlist1)
def test_store_new_mask(shape, dtype, cache_modifier, eviction_policy):
    mask = test_common.generate_tensor(shape, "bool").npu()
    x0 = test_common.generate_tensor(shape, dtype).npu()
    y_actual = test_common.generate_tensor(shape, dtype).npu()
    blocks = list(x0.size()) + [1] * (8 - len(x0.size()))
    triton_shape = list(shape) + [1] * (8 - len(shape))
    strides = list(x0.stride()) + [1] * (8 - len(x0.stride()))
    triton_store_new_mask[(1, )](x0, y_actual, mask, cache_modifier, eviction_policy, *blocks, *triton_shape, *strides)
