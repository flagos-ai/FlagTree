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

import logging

import pytest
import triton
import triton.language as tl
import torch
import torch_npu
import test_common
from test_common import TestUtils
import math


def torch_erf(x0):
    res = torch.erf(x0)
    return res


@triton.jit
def fn_npu_(output_ptr, x_ptr, y_ptr, z_ptr,
            XB: tl.constexpr, YB: tl.constexpr, ZB: tl.constexpr,
            XNUMEL: tl.constexpr, YNUMEL: tl.constexpr, ZNUMEL: tl.constexpr):
    xoffs = tl.program_id(0) * XB
    yoffs = tl.program_id(1) * YB
    zoffs = tl.program_id(2) * ZB

    xidx = tl.arange(0, XB) + xoffs
    yidx = tl.arange(0, YB) + yoffs
    zidx = tl.arange(0, ZB) + zoffs

    idx = xidx[:, None, None] * YNUMEL * ZNUMEL + yidx[None, :, None] * ZNUMEL + zidx[None, None, :]

    X = tl.load(x_ptr + idx)
    Y = tl.load(y_ptr + idx)

    ret = tl.erf(X)

    tl.store(output_ptr + idx, ret)


@triton.jit
def triton_erf_4d_5d(
        output_ptr, x_ptr,
        BLOCK_0: tl.constexpr, BLOCK_1: tl.constexpr, BLOCK_2: tl.constexpr, BLOCK_3: tl.constexpr,
        BLOCK_4: tl.constexpr,
        SHAPE_0: tl.constexpr, SHAPE_1: tl.constexpr, SHAPE_2: tl.constexpr, SHAPE_3: tl.constexpr,
        SHAPE_4: tl.constexpr,
        STRIDE_0: tl.constexpr, STRIDE_1: tl.constexpr, STRIDE_2: tl.constexpr, STRIDE_3: tl.constexpr,
        STRIDE_4: tl.constexpr,
        BLOCK_TOTAL: tl.constexpr
):

    pid = tl.program_id(0)
    start_idx = pid * BLOCK_TOTAL
    local_idx = tl.arange(0, BLOCK_TOTAL)
    global_idx = start_idx + local_idx
    total_elements = SHAPE_0 * SHAPE_1 * SHAPE_2 * SHAPE_3 * SHAPE_4
    masks = global_idx < total_elements

    dim1_base = SHAPE_1 * SHAPE_2 * SHAPE_3 * SHAPE_4
    dim2_base = SHAPE_2 * SHAPE_3 * SHAPE_4
    dim3_base = SHAPE_3 * SHAPE_4
    dim4_base = SHAPE_4

    idx_0 = (global_idx // dim1_base) % SHAPE_0
    idx_1 = (global_idx // dim2_base) % SHAPE_1
    idx_2 = (global_idx // dim3_base) % SHAPE_2
    idx_3 = (global_idx // dim4_base) % SHAPE_3
    idx_4 = global_idx % SHAPE_4

    offsets = idx_0 * STRIDE_0 + idx_1 * STRIDE_1 + idx_2 * STRIDE_2 + idx_3 * STRIDE_3 + idx_4 * STRIDE_4

    x_val = tl.load(x_ptr + offsets, mask=masks)
    ret = tl.erf(x_val)
    tl.store(output_ptr + offsets, ret, mask=masks)


@pytest.mark.parametrize('shape', TestUtils.full_shape)
@pytest.mark.parametrize('dtype',
                         ['float32', 'float16', 'bfloat16'])
def test_case2(dtype, shape):
    # 生成数据
    x = test_common.generate_tensor(shape, dtype).npu()
    y = test_common.generate_tensor(shape, dtype).npu()
    z = test_common.generate_tensor(shape, dtype).npu()
    new_shape = shape

    output = torch.randint(1, new_shape, dtype=eval('torch.' + dtype)).npu()
    output1 = output
    logging.debug(f"output.dtype={output.dtype}")

    ans = torch_erf(x)

    if len(shape) == 1:
        XB = 1
        xnumel = 1
        YB = 1
        ynumel = 1
        ZB = shape[0]
        znumel = shape[0]
    elif len(shape) == 2:
        XB = 1
        xnumel = 1
        YB = shape[0]
        ynumel = shape[0]
        ZB = shape[1]
        znumel = shape[1]
    else:
        XB = shape[0]
        xnumel = shape[0]
        YB = shape[1]
        ynumel = shape[1]
        ZB = shape[2]
        znumel = shape[2]

    grid = (1, 1, 1)
    if x.numel() * x.element_size() >= 8192:
        grid = (1, 1, ZB)
        ZB = 1

    fn_npu_[grid](output, x, y, z, XB, YB, ZB, xnumel, ynumel, znumel)

    test_common.validate_cmp(dtype, ans, output)


@pytest.mark.parametrize('shape', TestUtils.test_shape4d + TestUtils.test_shape5d)
@pytest.mark.parametrize('dtype', ['float32', 'float16', 'bfloat16'])
def test_erf_4d_5d(shape, dtype):
    logging.debug(f"Testing erf for shape={shape}, dtype={dtype}")

    x = test_common.generate_tensor(shape, dtype).npu()
    output = torch.empty_like(x)

    ans = torch_erf(x)

    shape_5d = list(shape)
    strides_5d = list(x.stride())
    while len(shape_5d) < 5:
        shape_5d.append(1)
        strides_5d.append(1)

    MAX_BLOCK_ELEMENTS = 1024
    total_elements = x.numel()

    block_5d = [1] * 5
    for i in reversed(range(5)):
        if shape_5d[i] == 0:
            continue
        max_block_i = min(shape_5d[i], MAX_BLOCK_ELEMENTS // (torch.prod(torch.tensor(block_5d)).item()))
        block_5d[i] = max_block_i
        if torch.prod(torch.tensor(block_5d)).item() >= MAX_BLOCK_ELEMENTS:
            break
    block_total = torch.prod(torch.tensor(block_5d)).item()

    grid = (triton.cdiv(total_elements, block_total),)
    logging.debug(f"Grid={grid}, block_5d={block_5d}, block_total={block_total}")

    triton_erf_4d_5d[grid](
        output, x,
        *block_5d,
        *shape_5d,
        *strides_5d,
        block_total
    )

    test_common.validate_cmp(dtype, ans, output)
