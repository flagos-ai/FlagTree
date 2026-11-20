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
import random
import pytest

import triton
import triton.language as tl
import torch
import test_common
import numpy as np
import triton.language.extra.cann.extension as extension


def gen_1d_cat_shapes(min_val=1, max_val=4096):
    shape1 = random.randint(min_val, max_val)
    shape2 = random.randint(min_val, max_val)
    return (shape1, ), (shape2, ), 0


def gen_2d_cat_shapes(dim=0, min_val=1, max_val=4096):
    if dim == 0:
        common_col = random.randint(min_val, max_val)
        row1 = random.randint(min_val, max_val)
        row2 = random.randint(min_val, max_val)
        shape1 = (row1, common_col)
        shape2 = (row2, common_col)
    elif dim == 1:
        common_row = random.randint(min_val, max_val)
        col1 = random.randint(min_val, max_val)
        col2 = random.randint(min_val, max_val)
        shape1 = (common_row, col1)
        shape2 = (common_row, col2)
    else:
        raise ValueError("2d shape only support dim=0 or dim=1")
    return shape1, shape2, dim


def gen_3d_cat_shapes(dim=0, min_val=1, max_val=4096):
    if dim not in [0, 1, 2]:
        raise ValueError("3d shape only support dim=0/1/2")

    if dim == 0:
        common_d1 = random.randint(min_val, max_val)
        common_d2 = random.randint(min_val, max_val)
        d0_1 = random.randint(min_val, max_val)
        d0_2 = random.randint(min_val, max_val)
        shape1 = (d0_1, common_d1, common_d2)
        shape2 = (d0_2, common_d1, common_d2)

    elif dim == 1:
        common_d0 = random.randint(min_val, max_val)
        common_d2 = random.randint(min_val, max_val)
        d1_1 = random.randint(min_val, max_val)
        d1_2 = random.randint(min_val, max_val)
        shape1 = (common_d0, d1_1, common_d2)
        shape2 = (common_d0, d1_2, common_d2)

    else:  # dim == 2
        common_d0 = random.randint(min_val, max_val)
        common_d1 = random.randint(min_val, max_val)
        d2_1 = random.randint(min_val, max_val)
        d2_2 = random.randint(min_val, max_val)
        shape1 = (common_d0, common_d1, d2_1)
        shape2 = (common_d0, common_d1, d2_2)

    return shape1, shape2, dim


def gen_100_cat_shapes(num_groups=100, mix_ratio=(0.3, 0.3, 0.4), min_val=1, max_val=4096):

    shape_list = []
    num_1d = int(num_groups * mix_ratio[0])
    num_2d = int(num_groups * mix_ratio[1])
    num_3d = num_groups - num_1d - num_2d

    for _ in range(num_1d):
        shape_list.append(gen_1d_cat_shapes(min_val, max_val))

    for _ in range(num_2d):
        dim = random.choice([0, 1])
        shape_list.append(gen_2d_cat_shapes(dim, min_val, max_val))

    for _ in range(num_3d):
        dim = random.choice([0, 1, 2])
        shape_list.append(gen_3d_cat_shapes(dim, min_val, max_val))

    random.shuffle(shape_list)
    return shape_list


full_shape = gen_100_cat_shapes(num_groups=100, mix_ratio=(0.3, 0.4, 0.3), min_val=1, max_val=4096)


@triton.jit
def _cat_helper_func_2D_1(
    in_ptr0,
    in_ptr1,
    out_ptr0,
    in0_x: tl.constexpr,
    in1_x: tl.constexpr,
    y0_numel,
    x1_numel,
    Y0BLOCK: tl.constexpr,
    Y0BLOCK_SUB: tl.constexpr,
):
    y0_offset = tl.program_id(0) * Y0BLOCK_SUB
    base_y0 = tl.arange(0, Y0BLOCK_SUB)
    loops_y0 = (Y0BLOCK + Y0BLOCK_SUB - 1) // Y0BLOCK_SUB
    base_input0_x1 = tl.arange(0, in0_x)[None, :]
    base_input1_x1 = tl.arange(0, in1_x)[None, :]
    x1 = tl.arange(0, in0_x + in1_x)[None, :]

    for loop in range(loops_y0):
        y0 = y0_offset + (loop * Y0BLOCK_SUB) + base_y0[:, None]
        y0_mask = y0 < min(Y0BLOCK + y0_offset, y0_numel)
        x1_mask = x1 < x1_numel
        tmp0 = tl.load(in_ptr0 + (base_input0_x1 + in0_x * y0), y0_mask)
        tmp1 = tl.load(in_ptr1 + (base_input1_x1 + in1_x * y0), y0_mask)
        tmp2 = tl.zeros((Y0BLOCK_SUB, in0_x + in1_x), dtype=tmp0.dtype)
        tmp3 = extension.insert_slice(tmp2, tmp0, [0, 0], [Y0BLOCK_SUB, in0_x], [1, 1])
        tmp4 = extension.insert_slice(tmp3, tmp1, [0, in0_x], [Y0BLOCK_SUB, in1_x], [1, 1])
        tl.store(out_ptr0 + (x1 + (in0_x + in1_x) * y0), tmp4, x1_mask & y0_mask)


@triton.jit
def triton_unk_fused_cat_dim0_sameshape(output_ptr, x_ptr, y_ptr, y0_numel, x1_numel, Y0BLOCK: tl.constexpr,
                                        Y0BLOCK_SUB: tl.constexpr, X1BLOCK_SUB: tl.constexpr):
    y0_offset = tl.program_id(0) * Y0BLOCK
    base_y0 = tl.arange(0, Y0BLOCK_SUB)
    loops_y0 = (Y0BLOCK + Y0BLOCK_SUB - 1) // Y0BLOCK_SUB
    base_x1 = tl.arange(0, X1BLOCK_SUB)
    loops_x1 = (x1_numel + X1BLOCK_SUB - 1) // X1BLOCK_SUB
    for loop_y0 in range(loops_y0):
        y0 = y0_offset + (loop_y0 * Y0BLOCK_SUB) + base_y0[:, None]
        y0_mask = y0 < min(Y0BLOCK + y0_offset, y0_numel)
        for loop_x1 in range(loops_x1):
            x1 = (loop_x1 * X1BLOCK_SUB) + base_x1[None, :]
            x1_mask = x1 < x1_numel

            tmp0 = tl.load(x_ptr + (x1 + x1_numel * y0), x1_mask & y0_mask)
            tmp8 = tl.load(y_ptr + (x1 + x1_numel * y0), x1_mask & y0_mask)
            tmp10 = tl.zeros((2 * Y0BLOCK_SUB, X1BLOCK_SUB), dtype=tmp0.dtype)
            tmp11 = extension.insert_slice(tmp10, tmp0, [0, 0], [Y0BLOCK_SUB, X1BLOCK_SUB], [1, 1])
            tmp12 = extension.insert_slice(tmp11, tmp8, [Y0BLOCK_SUB, 0], [Y0BLOCK_SUB, X1BLOCK_SUB], [1, 1])
            tmp13 = tl.reshape(tmp12, (2, Y0BLOCK_SUB, X1BLOCK_SUB))

            new_base_x2 = tl.arange(0, X1BLOCK_SUB)
            new_x2 = (loop_x1 * X1BLOCK_SUB) + new_base_x2[None, None, :]
            new_base_y1 = tl.arange(0, Y0BLOCK_SUB)
            new_y1 = y0_offset + (loop_y0 * Y0BLOCK_SUB) + new_base_y1[None, :, None]
            new_z0 = tl.arange(0, 2)[:, None, None]
            new_x2_mask = new_x2 < x1_numel
            new_y1_mask = new_y1 < y0_numel
            tl.store(output_ptr + (new_x2 + x1_numel * (new_y1 + y0_numel * new_z0)), tmp13, new_x2_mask & new_y1_mask)


@triton.jit
def triton_unk_fused_cat_dim0_diffshape(output_ptr, x_ptr, y_ptr, y0_numel, y1_numel, x1_numel, YBLOCK: tl.constexpr,
                                        YBLOCK_2: tl.constexpr, YBLOCK_SUB: tl.constexpr, X1BLOCK_SUB: tl.constexpr):
    y0_offset = tl.program_id(0) * YBLOCK
    base_y0 = tl.arange(0, YBLOCK_SUB)
    loops_y0 = (YBLOCK + YBLOCK_SUB - 1) // YBLOCK_SUB
    base_x1 = tl.arange(0, X1BLOCK_SUB)
    loops_x1 = (x1_numel + X1BLOCK_SUB - 1) // X1BLOCK_SUB
    min_numel = 0
    max_numel = 0
    clone_numel = 0
    if y0_numel < y1_numel:
        min_numel = y0_numel
        max_numel = y1_numel
        clone_numel = y1_numel - y0_numel
    else:
        min_numel = y1_numel
        max_numel = y0_numel
        clone_numel = y0_numel - y1_numel

    for loops_y in range(loops_y0):
        y0 = y0_offset + (loops_y * YBLOCK_SUB) + base_y0[:, None]
        y0_mask = y0 < min(YBLOCK + y0_offset, min_numel)
        for loop_x1 in range(loops_x1):
            x1 = (loop_x1 * X1BLOCK_SUB) + base_x1[None, :]
            x1_mask = x1 < x1_numel

            tmp0 = tl.load(x_ptr + (x1 + x1_numel * y0), x1_mask & y0_mask)
            tmp8 = tl.load(y_ptr + (x1 + x1_numel * y0), x1_mask & y0_mask)
            tmp10 = tl.zeros((2 * YBLOCK_SUB, X1BLOCK_SUB), dtype=tmp0.dtype)
            tmp11 = extension.insert_slice(tmp10, tmp0, [0, 0], [YBLOCK_SUB, X1BLOCK_SUB], [1, 1])
            tmp12 = extension.insert_slice(tmp11, tmp8, [YBLOCK_SUB, 0], [YBLOCK_SUB, X1BLOCK_SUB], [1, 1])
            tmp13 = tl.reshape(tmp12, (2, YBLOCK_SUB, X1BLOCK_SUB))

            new_base_x2 = tl.arange(0, X1BLOCK_SUB)
            new_x2 = (loop_x1 * X1BLOCK_SUB) + new_base_x2[None, None, :]
            new_base_y1 = tl.arange(0, YBLOCK_SUB)
            new_y1 = y0_offset + (loops_y * YBLOCK_SUB) + new_base_y1[None, :, None]
            new_z0 = tl.arange(0, 2)[:, None, None]
            new_x2_mask = new_x2 < x1_numel
            new_y1_mask = new_y1 < min_numel
            tl.store(output_ptr + (new_x2 + x1_numel * new_y1 + x1_numel * y0_numel * new_z0), tmp13,
                     new_x2_mask & new_y1_mask)

    loops_y1 = (YBLOCK_2 + YBLOCK_SUB - 1) // YBLOCK_SUB
    y2_offset = tl.program_id(0) * YBLOCK_2 + min_numel
    if y0_numel < y1_numel:
        for loops_y1 in range(loops_y1):
            y0 = y2_offset + (loops_y1 * YBLOCK_SUB) + base_y0[:, None]
            y0_mask = y0 < min(YBLOCK_2 + y2_offset, y1_numel)
            for loop_x1 in range(loops_x1):
                x1 = (loop_x1 * X1BLOCK_SUB) + base_x1[None, :]
                x1_mask = x1 < x1_numel

                tmp8 = tl.load(y_ptr + (x1 + x1_numel * y0), x1_mask & y0_mask)
                new_base_x2 = tl.arange(0, X1BLOCK_SUB)
                new_x2 = (loop_x1 * X1BLOCK_SUB) + new_base_x2[None, :]
                new_base_y1 = tl.arange(0, YBLOCK_SUB)
                new_y1 = y2_offset + y0_numel + (loops_y1 * YBLOCK_SUB) + new_base_y1[:, None]
                sum_numel = y0_numel + y1_numel
                new_x2_mask = new_x2 < x1_numel
                new_y1_mask = new_y1 < sum_numel
                tl.store(output_ptr + (new_x2 + x1_numel * new_y1), tmp8, new_x2_mask & new_y1_mask)
    else:
        for loops_y1 in range(loops_y1):
            y0 = y2_offset + (loops_y1 * YBLOCK_SUB) + base_y0[:, None]
            y0_mask = y0 < min(YBLOCK_2 + y2_offset, y0_numel)
            for loop_x1 in range(loops_x1):
                x1 = (loop_x1 * X1BLOCK_SUB) + base_x1[None, :]
                x1_mask = x1 < x1_numel

                tmp8 = tl.load(x_ptr + (x1 + x1_numel * y0), x1_mask & y0_mask)
                new_base_x2 = tl.arange(0, X1BLOCK_SUB)
                new_x2 = (loop_x1 * X1BLOCK_SUB) + new_base_x2[None, :]
                new_base_y1 = tl.arange(0, YBLOCK_SUB)
                new_y1 = y2_offset + (loops_y1 * YBLOCK_SUB) + new_base_y1[:, None]
                new_x2_mask = new_x2 < x1_numel
                new_y1_mask = new_y1 < y0_numel
                tl.store(output_ptr + (new_x2 + x1_numel * new_y1), tmp8, new_x2_mask & new_y1_mask)


@triton.jit
def triton_unk_fused_cat_dim1_sameshape(output_ptr, x_ptr, y_ptr, y0_numel, x1_numel, Y0BLOCK: tl.constexpr,
                                        Y0BLOCK_SUB: tl.constexpr, X1BLOCK_SUB: tl.constexpr):
    y0_offset = tl.program_id(0) * Y0BLOCK
    base_y0 = tl.arange(0, Y0BLOCK_SUB)
    loops_y0 = (Y0BLOCK + Y0BLOCK_SUB - 1) // Y0BLOCK_SUB
    base_x1 = tl.arange(0, X1BLOCK_SUB)
    loops_x1 = (x1_numel + X1BLOCK_SUB - 1) // X1BLOCK_SUB
    for loop_y0 in range(loops_y0):
        y0 = y0_offset + (loop_y0 * Y0BLOCK_SUB) + base_y0[:, None]
        y0_mask = y0 < min(Y0BLOCK + y0_offset, y0_numel)
        for loop_x1 in range(loops_x1):
            x1 = (loop_x1 * X1BLOCK_SUB) + base_x1[None, :]
            x1_mask = x1 < x1_numel

            tmp0 = tl.load(x_ptr + (x1 + x1_numel * y0), x1_mask & y0_mask)
            tmp8 = tl.load(y_ptr + (x1 + x1_numel * y0), x1_mask & y0_mask)
            tmp10 = tl.zeros((Y0BLOCK_SUB, 2 * X1BLOCK_SUB), dtype=tmp0.dtype)
            tmp11 = extension.insert_slice(tmp10, tmp0, [0, 0], [Y0BLOCK_SUB, X1BLOCK_SUB], [1, 1])
            tmp12 = extension.insert_slice(tmp11, tmp8, [0, X1BLOCK_SUB], [Y0BLOCK_SUB, X1BLOCK_SUB], [1, 1])
            tmp13 = tl.reshape(tmp12, (Y0BLOCK_SUB, 2, X1BLOCK_SUB))

            new_base_x2 = tl.arange(0, X1BLOCK_SUB)
            new_x2 = (loop_x1 * X1BLOCK_SUB) + new_base_x2[None, None, :]
            new_base_y1 = tl.arange(0, Y0BLOCK_SUB)
            new_y1 = y0_offset + (loop_y0 * Y0BLOCK_SUB) + new_base_y1[:, None, None]
            new_z0 = tl.arange(0, 2)[None, :, None]
            new_x2_mask = new_x2 < x1_numel
            new_y1_mask = new_y1 < y0_numel
            tl.store(output_ptr + (new_x2 + 2 * x1_numel * new_y1 + x1_numel * new_z0), tmp13,
                     new_x2_mask & new_y1_mask)


@triton.jit
def triton_unk_fused_cat_dim1_diffshape(output_ptr, x_ptr, y_ptr, y0_numel, x0_numel, x1_numel, Y0BLOCK: tl.constexpr,
                                        Y0BLOCK_SUB: tl.constexpr, XBLOCK_SUB: tl.constexpr):
    y0_offset = tl.program_id(0) * Y0BLOCK
    base_y0 = tl.arange(0, Y0BLOCK_SUB)
    loops_y0 = (Y0BLOCK + Y0BLOCK_SUB - 1) // Y0BLOCK_SUB
    base_x = tl.arange(0, XBLOCK_SUB)
    min_numel = 0
    max_numel = 0
    clone_numel = 0
    if x0_numel < x1_numel:
        min_numel = x0_numel
        max_numel = x1_numel
        clone_numel = x1_numel - x0_numel
    else:
        min_numel = x1_numel
        max_numel = x0_numel
        clone_numel = x0_numel - x1_numel
    loops_x = (min_numel + XBLOCK_SUB - 1) // XBLOCK_SUB
    loops_x2 = (clone_numel + XBLOCK_SUB - 1) // XBLOCK_SUB
    for loop_y0 in range(loops_y0):
        y0 = y0_offset + (loop_y0 * Y0BLOCK_SUB) + base_y0[:, None]
        y0_mask = y0 < min(Y0BLOCK + y0_offset, y0_numel)
        for loop_x in range(loops_x):
            x = (loop_x * XBLOCK_SUB) + base_x[None, :]
            x_mask = x < min_numel

            tmp0 = tl.load(x_ptr + (x + x0_numel * y0), x_mask & y0_mask)
            tmp8 = tl.load(y_ptr + (x + x1_numel * y0), x_mask & y0_mask)
            tmp10 = tl.zeros((Y0BLOCK_SUB, 2 * XBLOCK_SUB), dtype=tmp0.dtype)
            tmp11 = extension.insert_slice(tmp10, tmp0, [0, 0], [Y0BLOCK_SUB, XBLOCK_SUB], [1, 1])
            tmp12 = extension.insert_slice(tmp11, tmp8, [0, XBLOCK_SUB], [Y0BLOCK_SUB, XBLOCK_SUB], [1, 1])
            tmp13 = tl.reshape(tmp12, (Y0BLOCK_SUB, 2, XBLOCK_SUB))

            new_base_x2 = tl.arange(0, XBLOCK_SUB)
            new_x2 = (loop_x * XBLOCK_SUB) + new_base_x2[None, None, :]
            new_base_y1 = tl.arange(0, Y0BLOCK_SUB)
            new_y1 = y0_offset + (loop_y0 * Y0BLOCK_SUB) + new_base_y1[:, None, None]
            new_z0 = tl.arange(0, 2)[None, :, None]
            new_x2_mask = new_x2 < min_numel
            new_y1_mask = new_y1 < y0_numel
            sum_numel = x0_numel + x1_numel
            tl.store(output_ptr + (new_x2 + sum_numel * new_y1 + x0_numel * new_z0), tmp13, new_x2_mask & new_y1_mask)

    if x0_numel < x1_numel:
        for loop_y0 in range(loops_y0):
            y0 = y0_offset + (loop_y0 * Y0BLOCK_SUB) + base_y0[:, None]
            y0_mask = y0 < min(Y0BLOCK + y0_offset, y0_numel)
            for loop_x2 in range(loops_x2):
                x = (loop_x2 * XBLOCK_SUB) + base_x[None, :] + min_numel
                x_mask = x < x1_numel

                tmp8 = tl.load(y_ptr + (x + x1_numel * y0), x_mask & y0_mask)
                new_base_x2 = tl.arange(0, XBLOCK_SUB)
                new_x2 = x0_numel + min_numel + (loop_x2 * XBLOCK_SUB) + new_base_x2[None, :]
                new_base_y1 = tl.arange(0, Y0BLOCK_SUB)
                new_y1 = y0_offset + (loop_y0 * Y0BLOCK_SUB) + new_base_y1[:, None]
                sum_numel = x0_numel + x1_numel
                new_x2_mask = new_x2 < sum_numel
                new_y1_mask = new_y1 < y0_numel
                tl.store(output_ptr + (new_x2 + sum_numel * new_y1), tmp8, new_x2_mask & new_y1_mask)
    else:
        for loop_y0 in range(loops_y0):
            y0 = y0_offset + (loop_y0 * Y0BLOCK_SUB) + base_y0[:, None]
            y0_mask = y0 < min(Y0BLOCK + y0_offset, y0_numel)
            for loop_x2 in range(loops_x2):
                x = (loop_x2 * XBLOCK_SUB) + base_x[None, :] + min_numel
                x_mask = x < x0_numel

                tmp8 = tl.load(x_ptr + (x + x0_numel * y0), x_mask & y0_mask)
                new_base_x2 = tl.arange(0, XBLOCK_SUB)
                new_x2 = min_numel + (loop_x2 * XBLOCK_SUB) + new_base_x2[None, :]
                new_base_y1 = tl.arange(0, Y0BLOCK_SUB)
                new_y1 = y0_offset + (loop_y0 * Y0BLOCK_SUB) + new_base_y1[:, None]
                sum_numel = x0_numel + x1_numel
                new_x2_mask = new_x2 < x0_numel
                new_y1_mask = new_y1 < y0_numel
                tl.store(output_ptr + (new_x2 + sum_numel * new_y1), tmp8, new_x2_mask & new_y1_mask)


@triton.jit
def triton_unk_fused_cat_3d_dim0(output_ptr, x_ptr, y_ptr, z0_numel, z1_numel, y1_numel, x1_numel, ZBLOCK: tl.constexpr,
                                 ZBLOCK_2: tl.constexpr, ZBLOCK_SUB: tl.constexpr, X1BLOCK_SUB: tl.constexpr):
    z0_offset = tl.program_id(0) * ZBLOCK
    base_z0 = tl.arange(0, ZBLOCK_SUB)
    loops_z0 = (ZBLOCK + ZBLOCK_SUB - 1) // ZBLOCK_SUB
    xy_numel = x1_numel * y1_numel
    base_x1 = tl.arange(0, X1BLOCK_SUB)
    loops_x1 = (xy_numel + X1BLOCK_SUB - 1) // X1BLOCK_SUB
    min_numel = 0
    max_numel = 0
    clone_numel = 0
    if z0_numel < z1_numel:
        min_numel = z0_numel
        max_numel = z1_numel
        clone_numel = z1_numel - z0_numel
    else:
        min_numel = z1_numel
        max_numel = z0_numel
        clone_numel = z0_numel - z1_numel

    for loops_z in range(loops_z0):
        z0 = z0_offset + (loops_z * ZBLOCK_SUB) + base_z0[:, None]
        z0_mask = z0 < min(ZBLOCK + z0_offset, min_numel)
        for loop_x1 in range(loops_x1):
            x1 = (loop_x1 * X1BLOCK_SUB) + base_x1[None, :]
            x1_mask = x1 < xy_numel

            tmp0 = tl.load(x_ptr + (x1 + xy_numel * z0), x1_mask & z0_mask)
            tmp8 = tl.load(y_ptr + (x1 + xy_numel * z0), x1_mask & z0_mask)
            tmp10 = tl.zeros((2 * ZBLOCK_SUB, X1BLOCK_SUB), dtype=tmp0.dtype)
            tmp11 = extension.insert_slice(tmp10, tmp0, [0, 0], [ZBLOCK_SUB, X1BLOCK_SUB], [1, 1])
            tmp12 = extension.insert_slice(tmp11, tmp8, [ZBLOCK_SUB, 0], [ZBLOCK_SUB, X1BLOCK_SUB], [1, 1])
            tmp13 = tl.reshape(tmp12, (2, ZBLOCK_SUB, X1BLOCK_SUB))

            new_base_x2 = tl.arange(0, X1BLOCK_SUB)
            new_x2 = (loop_x1 * X1BLOCK_SUB) + new_base_x2[None, None, :]
            new_base_z1 = tl.arange(0, ZBLOCK_SUB)
            new_z1 = z0_offset + (loops_z * ZBLOCK_SUB) + new_base_z1[None, :, None]
            new_z0 = tl.arange(0, 2)[:, None, None]
            new_x2_mask = new_x2 < xy_numel
            new_z1_mask = new_z1 < min_numel
            tl.store(output_ptr + (new_x2 + xy_numel * new_z1 + xy_numel * z0_numel * new_z0), tmp13,
                     new_x2_mask & new_z1_mask)

    loops_z1 = (ZBLOCK_2 + ZBLOCK_SUB - 1) // ZBLOCK_SUB
    z2_offset = tl.program_id(0) * ZBLOCK_2 + min_numel
    if z0_numel < z1_numel:
        for loops_z1 in range(loops_z1):
            z0 = z2_offset + (loops_z1 * ZBLOCK_SUB) + base_z0[:, None]
            z0_mask = z0 < min(ZBLOCK_2 + z2_offset, z1_numel)
            for loop_x1 in range(loops_x1):
                x1 = (loop_x1 * X1BLOCK_SUB) + base_x1[None, :]
                x1_mask = x1 < xy_numel

                tmp8 = tl.load(y_ptr + (x1 + xy_numel * z0), x1_mask & z0_mask)
                new_base_x2 = tl.arange(0, X1BLOCK_SUB)
                new_x2 = (loop_x1 * X1BLOCK_SUB) + new_base_x2[None, :]
                new_base_z1 = tl.arange(0, ZBLOCK_SUB)
                new_z1 = z2_offset + z0_numel + (loops_z1 * ZBLOCK_SUB) + new_base_z1[:, None]
                sum_numel = z0_numel + z1_numel
                new_x2_mask = new_x2 < xy_numel
                new_z1_mask = new_z1 < sum_numel
                tl.store(output_ptr + (new_x2 + xy_numel * new_z1), tmp8, new_x2_mask & new_z1_mask)
    else:
        for loops_z1 in range(loops_z1):
            z0 = z2_offset + (loops_z1 * ZBLOCK_SUB) + base_z0[:, None]
            z0_mask = z0 < min(ZBLOCK_2 + z2_offset, z0_numel)
            for loop_x1 in range(loops_x1):
                x1 = (loop_x1 * X1BLOCK_SUB) + base_x1[None, :]
                x1_mask = x1 < xy_numel

                tmp8 = tl.load(x_ptr + (x1 + xy_numel * z0), x1_mask & z0_mask)
                new_base_x2 = tl.arange(0, X1BLOCK_SUB)
                new_x2 = (loop_x1 * X1BLOCK_SUB) + new_base_x2[None, :]
                new_base_z1 = tl.arange(0, ZBLOCK_SUB)
                new_z1 = z2_offset + (loops_z1 * ZBLOCK_SUB) + new_base_z1[:, None]
                new_x2_mask = new_x2 < xy_numel
                new_z1_mask = new_z1 < z0_numel
                tl.store(output_ptr + (new_x2 + xy_numel * new_z1), tmp8, new_x2_mask & new_z1_mask)


@triton.jit
def triton_unk_fused_cat_3d_dim1(output_ptr, x_ptr, y_ptr, z0_numel, y0_numel, y1_numel, x0_numel,
                                 Z0BLOCK: tl.constexpr, Z0BLOCK_SUB: tl.constexpr, XBLOCK_SUB: tl.constexpr):
    z0_offset = tl.program_id(0) * Z0BLOCK
    base_z0 = tl.arange(0, Z0BLOCK_SUB)
    loops_z0 = (Z0BLOCK + Z0BLOCK_SUB - 1) // Z0BLOCK_SUB
    base_x = tl.arange(0, XBLOCK_SUB)
    min_numel = 0
    max_numel = 0
    clone_numel = 0
    if y0_numel < y1_numel:
        min_numel = y0_numel * x0_numel
        max_numel = y1_numel * x0_numel
        clone_numel = (y1_numel - y0_numel) * x0_numel
    else:
        min_numel = y1_numel * x0_numel
        max_numel = y0_numel * x0_numel
        clone_numel = (y0_numel - y1_numel) * x0_numel
    loops_x = (min_numel + XBLOCK_SUB - 1) // XBLOCK_SUB
    loops_x2 = (clone_numel + XBLOCK_SUB - 1) // XBLOCK_SUB
    for loop_z0 in range(loops_z0):
        z0 = z0_offset + (loop_z0 * Z0BLOCK_SUB) + base_z0[:, None]
        z0_mask = z0 < min(Z0BLOCK + z0_offset, z0_numel)
        for loop_x in range(loops_x):
            x = (loop_x * XBLOCK_SUB) + base_x[None, :]
            x_mask = x < min_numel

            tmp0 = tl.load(x_ptr + (x + x0_numel * y0_numel * z0), x_mask & z0_mask)
            tmp8 = tl.load(y_ptr + (x + x0_numel * y1_numel * z0), x_mask & z0_mask)
            tmp10 = tl.zeros((Z0BLOCK_SUB, 2 * XBLOCK_SUB), dtype=tmp0.dtype)
            tmp11 = extension.insert_slice(tmp10, tmp0, [0, 0], [Z0BLOCK_SUB, XBLOCK_SUB], [1, 1])
            tmp12 = extension.insert_slice(tmp11, tmp8, [0, XBLOCK_SUB], [Z0BLOCK_SUB, XBLOCK_SUB], [1, 1])
            tmp13 = tl.reshape(tmp12, (Z0BLOCK_SUB, 2, XBLOCK_SUB))

            new_base_x2 = tl.arange(0, XBLOCK_SUB)
            new_x2 = (loop_x * XBLOCK_SUB) + new_base_x2[None, None, :]
            new_base_z1 = tl.arange(0, Z0BLOCK_SUB)
            new_z1 = z0_offset + (loop_z0 * Z0BLOCK_SUB) + new_base_z1[:, None, None]
            new_z0 = tl.arange(0, 2)[None, :, None]
            new_x2_mask = new_x2 < min_numel
            new_z1_mask = new_z1 < z0_numel
            sum_numel = min_numel + max_numel
            tl.store(output_ptr + (new_x2 + sum_numel * new_z1 + x0_numel * y0_numel * new_z0), tmp13,
                     new_x2_mask & new_z1_mask)

    if y0_numel == y1_numel:
        return

    if y0_numel < y1_numel:
        for loop_z0 in range(loops_z0):
            z0 = z0_offset + (loop_z0 * Z0BLOCK_SUB) + base_z0[:, None]
            z0_mask = z0 < min(Z0BLOCK + z0_offset, z0_numel)
            for loop_x2 in range(loops_x2):
                x = (loop_x2 * XBLOCK_SUB) + base_x[None, :] + min_numel
                x_mask = x < y1_numel * x0_numel

                tmp8 = tl.load(y_ptr + (x + x0_numel * y1_numel * z0), x_mask & z0_mask)
                new_base_x2 = tl.arange(0, XBLOCK_SUB)
                new_x2 = x0_numel * y0_numel + min_numel + (loop_x2 * XBLOCK_SUB) + new_base_x2[None, :]
                new_base_z1 = tl.arange(0, Z0BLOCK_SUB)
                new_z1 = z0_offset + (loop_z0 * Z0BLOCK_SUB) + new_base_z1[:, None]
                sum_numel = min_numel + max_numel
                new_x2_mask = new_x2 < sum_numel
                new_z1_mask = new_z1 < z0_numel
                tl.store(output_ptr + (new_x2 + sum_numel * new_z1), tmp8, new_x2_mask & new_z1_mask)
    else:
        for loop_z0 in range(loops_z0):
            z0 = z0_offset + (loop_z0 * Z0BLOCK_SUB) + base_z0[:, None]
            z0_mask = z0 < min(Z0BLOCK + z0_offset, z0_numel)
            for loop_x2 in range(loops_x2):
                x = (loop_x2 * XBLOCK_SUB) + base_x[None, :] + min_numel
                x_mask = x < x0_numel * y0_numel

                tmp8 = tl.load(x_ptr + (x + x0_numel * y0_numel * z0), x_mask & z0_mask)
                new_base_x2 = tl.arange(0, XBLOCK_SUB)
                new_x2 = min_numel + (loop_x2 * XBLOCK_SUB) + new_base_x2[None, :]
                new_base_z1 = tl.arange(0, Z0BLOCK_SUB)
                new_z1 = z0_offset + (loop_z0 * Z0BLOCK_SUB) + new_base_z1[:, None]
                sum_numel = min_numel + max_numel
                new_x2_mask = new_x2 < x0_numel * y0_numel
                new_z1_mask = new_z1 < z0_numel
                tl.store(output_ptr + (new_x2 + sum_numel * new_z1), tmp8, new_x2_mask & new_z1_mask)


@triton.jit
def triton_unk_fused_cat_3d_dim2(output_ptr, x_ptr, y_ptr, z0_numel, y0_numel, x0_numel, x1_numel,
                                 Y0BLOCK: tl.constexpr, Y0BLOCK_SUB: tl.constexpr, XBLOCK_SUB: tl.constexpr):
    y0_offset = tl.program_id(0) * Y0BLOCK
    base_y0 = tl.arange(0, Y0BLOCK_SUB)
    loops_y0 = (Y0BLOCK + Y0BLOCK_SUB - 1) // Y0BLOCK_SUB
    base_x = tl.arange(0, XBLOCK_SUB)
    min_numel = 0
    max_numel = 0
    clone_numel = 0
    zy_numel = z0_numel * y0_numel
    if x0_numel < x1_numel:
        min_numel = x0_numel
        max_numel = x1_numel
        clone_numel = x1_numel - x0_numel
    else:
        min_numel = x1_numel
        max_numel = x0_numel
        clone_numel = x0_numel - x1_numel
    loops_x = (min_numel + XBLOCK_SUB - 1) // XBLOCK_SUB
    loops_x2 = (clone_numel + XBLOCK_SUB - 1) // XBLOCK_SUB
    for loop_y0 in range(loops_y0):
        y0 = y0_offset + (loop_y0 * Y0BLOCK_SUB) + base_y0[:, None]
        y0_mask = y0 < min(Y0BLOCK + y0_offset, zy_numel)
        for loop_x in range(loops_x):
            x = (loop_x * XBLOCK_SUB) + base_x[None, :]
            x_mask = x < min_numel

            tmp0 = tl.load(x_ptr + (x + x0_numel * y0), x_mask & y0_mask)
            tmp8 = tl.load(y_ptr + (x + x1_numel * y0), x_mask & y0_mask)
            tmp10 = tl.zeros((Y0BLOCK_SUB, 2 * XBLOCK_SUB), dtype=tmp0.dtype)
            tmp11 = extension.insert_slice(tmp10, tmp0, [0, 0], [Y0BLOCK_SUB, XBLOCK_SUB], [1, 1])
            tmp12 = extension.insert_slice(tmp11, tmp8, [0, XBLOCK_SUB], [Y0BLOCK_SUB, XBLOCK_SUB], [1, 1])
            tmp13 = tl.reshape(tmp12, (Y0BLOCK_SUB, 2, XBLOCK_SUB))

            new_base_x2 = tl.arange(0, XBLOCK_SUB)
            new_x2 = (loop_x * XBLOCK_SUB) + new_base_x2[None, None, :]
            new_base_y1 = tl.arange(0, Y0BLOCK_SUB)
            new_y1 = y0_offset + (loop_y0 * Y0BLOCK_SUB) + new_base_y1[:, None, None]
            new_z0 = tl.arange(0, 2)[None, :, None]
            new_x2_mask = new_x2 < min_numel
            new_y1_mask = new_y1 < zy_numel
            sum_numel = x0_numel + x1_numel
            tl.store(output_ptr + (new_x2 + sum_numel * new_y1 + x0_numel * new_z0), tmp13, new_x2_mask & new_y1_mask)

    if x0_numel == x1_numel:
        return

    if x0_numel < x1_numel:
        for loop_y0 in range(loops_y0):
            y0 = y0_offset + (loop_y0 * Y0BLOCK_SUB) + base_y0[:, None]
            y0_mask = y0 < min(Y0BLOCK + y0_offset, zy_numel)
            for loop_x2 in range(loops_x2):
                x = (loop_x2 * XBLOCK_SUB) + base_x[None, :] + min_numel
                x_mask = x < x1_numel

                tmp8 = tl.load(y_ptr + (x + x1_numel * y0), x_mask & y0_mask)
                new_base_x2 = tl.arange(0, XBLOCK_SUB)
                new_x2 = x0_numel + min_numel + (loop_x2 * XBLOCK_SUB) + new_base_x2[None, :]
                new_base_y1 = tl.arange(0, Y0BLOCK_SUB)
                new_y1 = y0_offset + (loop_y0 * Y0BLOCK_SUB) + new_base_y1[:, None]
                sum_numel = x0_numel + x1_numel
                new_x2_mask = new_x2 < sum_numel
                new_y1_mask = new_y1 < zy_numel
                tl.store(output_ptr + (new_x2 + sum_numel * new_y1), tmp8, new_x2_mask & new_y1_mask)
    else:
        for loop_y0 in range(loops_y0):
            y0 = y0_offset + (loop_y0 * Y0BLOCK_SUB) + base_y0[:, None]
            y0_mask = y0 < min(Y0BLOCK + y0_offset, zy_numel)
            for loop_x2 in range(loops_x2):
                x = (loop_x2 * XBLOCK_SUB) + base_x[None, :] + min_numel
                x_mask = x < x0_numel

                tmp8 = tl.load(x_ptr + (x + x0_numel * y0), x_mask & y0_mask)
                new_base_x2 = tl.arange(0, XBLOCK_SUB)
                new_x2 = min_numel + (loop_x2 * XBLOCK_SUB) + new_base_x2[None, :]
                new_base_y1 = tl.arange(0, Y0BLOCK_SUB)
                new_y1 = y0_offset + (loop_y0 * Y0BLOCK_SUB) + new_base_y1[:, None]
                sum_numel = x0_numel + x1_numel
                new_x2_mask = new_x2 < x0_numel
                new_y1_mask = new_y1 < zy_numel
                tl.store(output_ptr + (new_x2 + sum_numel * new_y1), tmp8, new_x2_mask & new_y1_mask)


testlist = [
    # ===================== 1D场景（15组，dim=0） =====================
    ((3, ), (3, ), 0),
    ((7, ), (9, ), 0),
    ((13, ), (11, ), 0),
    ((2047, ), (2047, ), 0),
    ((2701, ), (3003, ), 0),
    ((4093, ), (3095, ), 0),

    # ===================== 2D场景（20组，dim0/dim1） =====================
    # dim0（行拼接，列维度一致）
    ((3, 5), (3, 5), 0),
    ((1005, 300), (2007, 300), 0),
    ((1307, 400), (309, 400), 0),
    ((303, 500), (303, 500), 0),
    # dim1（列拼接，行维度一致）
    ((7, 9), (7, 9), 1),
    ((100, 1001), (100, 2003), 1),
    ((200, 2005), (200, 207), 1),
    ((300, 707), (300, 707), 1),

    # ===================== 3D场景（15组，dim0/dim1/dim2） =====================
    # dim0（第0维拼接，d1/d2一致）
    ((378, 200, 300), (101, 200, 300), 0),
    ((378, 70, 50), (601, 70, 50), 0),
    # dim1（第1维拼接，d0/d2一致）
    ((100, 452, 300), (100, 201, 300), 1),
    ((65, 1735, 57), (65, 2001, 57), 1),
    # dim2（第2维拼接，d0/d1一致）
    ((87, 200, 387), (87, 200, 501), 2),
    ((20, 337, 543), (20, 337, 401), 2),
]


@pytest.mark.parametrize('testlists', testlist)
@pytest.mark.parametrize('dtype', ['bool', 'int8', 'int16', 'int32', 'int64', 'float16', 'float32', 'bfloat16'])
def test_cat_bigshape(testlists, dtype):
    torch_dtype = eval('torch.' + dtype)
    np_x0 = test_common.generate_numpy(testlists[0], dtype)
    np_x1 = test_common.generate_numpy(testlists[1], dtype)
    cat_dim = testlists[2]

    x0 = torch.from_numpy(np_x0).to(torch_dtype).npu()
    x1 = torch.from_numpy(np_x1).to(torch_dtype).npu()

    if len(x0.shape) > 3:
        pytest.skip("dim > 3 for 3D+ tensor, skipping.")

    torch_res = torch.cat([x0, x1], dim=cat_dim)
    triton_res = torch.zeros_like(torch_res)
    num_core = 32
    if len(x0.shape) == 3:
        if cat_dim == 0:
            ZBLOCK = (min(x0.shape[0], x1.shape[0]) + num_core - 1) // num_core
            ZBLOCK_2 = (max(x0.shape[0], x1.shape[0]) - min(x0.shape[0], x1.shape[0]) + num_core - 1) // num_core
            triton_unk_fused_cat_3d_dim0[num_core, 1, 1](triton_res, x0, x1, x0.shape[0], x1.shape[0], x0.shape[1],
                                                         x0.shape[2], ZBLOCK, ZBLOCK_2, 1, 256)
        elif cat_dim == 1:
            Z0BLOCK = (x0.shape[0] + num_core - 1) // num_core
            triton_unk_fused_cat_3d_dim1[num_core, 1, 1](triton_res, x0, x1, x0.shape[0], x0.shape[1], x1.shape[1],
                                                         x1.shape[2], Z0BLOCK, 1, 256)
        else:
            Y0BLOCK = (x0.shape[0] * x0.shape[1] + num_core - 1) // num_core
            triton_unk_fused_cat_3d_dim2[num_core, 1, 1](triton_res, x0, x1, x0.shape[0], x0.shape[1], x0.shape[2],
                                                         x1.shape[2], Y0BLOCK, 1, 256)
        test_common.validate_cmp(dtype, torch_res, triton_res)
        return
    numel_large = torch_res.numel() > 512 and len(x0.shape) < 3
    if numel_large or (cat_dim == 0 and len(x0.shape) == 2):
        squeeze_flag = False
        if len(x0.shape) == 1:
            squeeze_flag = True
            x0 = torch.unsqueeze(x0, dim=0)
            x1 = torch.unsqueeze(x1, dim=0)
            triton_res = torch.unsqueeze(triton_res, dim=0)
            cat_dim = 1
        if cat_dim == 1:
            Y0BLOCK = (x0.shape[0] + num_core - 1) // num_core
            if x0.shape[1] == x1.shape[1]:
                triton_unk_fused_cat_dim1_sameshape[num_core, 1, 1](triton_res, x0, x1, x0.shape[0], x0.shape[1],
                                                                    Y0BLOCK, 1, 256)
            else:
                triton_unk_fused_cat_dim1_diffshape[num_core, 1, 1](triton_res, x0, x1, x0.shape[0], x0.shape[1],
                                                                    x1.shape[1], Y0BLOCK, 1, 256)
        else:
            if x0.shape[0] == x1.shape[0]:
                Y0BLOCK = (x0.shape[0] + num_core - 1) // num_core
                triton_unk_fused_cat_dim0_sameshape[num_core, 1, 1](triton_res, x0, x1, x0.shape[0], x0.shape[1],
                                                                    Y0BLOCK, 1, 256)
            else:
                YBLOCK = (min(x0.shape[0], x1.shape[0]) + num_core - 1) // num_core
                YBLOCK_2 = (max(x0.shape[0], x1.shape[0]) - min(x0.shape[0], x1.shape[0]) + num_core - 1) // num_core
                triton_unk_fused_cat_dim0_diffshape[num_core, 1, 1](triton_res, x0, x1, x0.shape[0], x1.shape[0],
                                                                    x1.shape[1], YBLOCK, YBLOCK_2, 1, 256)
        if squeeze_flag:
            triton_res = triton_res.squeeze()
    else:
        squeeze_flag = False
        if len(x0.shape) == 1:
            squeeze_flag = True
            x0 = torch.unsqueeze(x0, dim=0)
            x1 = torch.unsqueeze(x1, dim=0)
            triton_res = torch.unsqueeze(triton_res, dim=0)
        _cat_helper_func_2D_1[num_core, 1, 1](x0, x1, triton_res, x0.shape[1], x1.shape[1], x0.shape[0],
                                              x0.shape[1] + x1.shape[1], 256, 16)
        if squeeze_flag:
            triton_res = triton_res.squeeze()

    test_common.validate_cmp(dtype, torch_res, triton_res)
