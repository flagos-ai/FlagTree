# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
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

import triton.language as tl
from test_common import check_axes_parse_res, mock_autotuner


def test_triton_max_last_dim_case1(mock_autotuner):
    import triton.backends.ascend.runtime

    @triton.autotune(configs=[], key=["x0_numel", "r1_numel"])
    @triton.jit
    def triton_max_last_dim1(
        in_ptr0,
        out_ptr0,
        x0_numel,
        r1_numel,
        X0BLOCK: tl.constexpr,
        X0BLOCK_SUB: tl.constexpr,
        R1BLOCK_SUB: tl.constexpr,
    ):
        x0_offset = tl.program_id(0) * X0BLOCK
        base_x0 = tl.arange(0, X0BLOCK_SUB)
        loops_x0 = (X0BLOCK + X0BLOCK_SUB - 1) // X0BLOCK_SUB
        base_r1 = tl.arange(0, R1BLOCK_SUB)
        loops_r1 = (r1_numel + R1BLOCK_SUB - 1) // R1BLOCK_SUB
        for loop_x0 in range(loops_x0):
            x0 = x0_offset + (loop_x0 * X0BLOCK_SUB) + base_x0[:, None]
            x0_mask = x0 < min(X0BLOCK + x0_offset, x0_numel)
            block_val = tl.full([X0BLOCK_SUB, R1BLOCK_SUB], float("-inf"), tl.float32)
            for loop_r1 in range(loops_r1):
                r1 = (loop_r1 * R1BLOCK_SUB) + base_r1[None, :]
                r1_mask = r1 < r1_numel
                tmp = tl.load(in_ptr0 + (r1 + r1_numel * x0), r1_mask & x0_mask, other=float("-inf"))
                block_val = tl.maximum(block_val, tmp)
            # Reduce along axis = 1 (the last dimension in this 2D tensor)
            block_res = tl.max(block_val, axis=1)[:, None]  # <- explicit positive axis index
            tl.store(out_ptr0 + x0, block_res, x0_mask)

    ref_res = {
        "keys": {"x": "x0_numel", "ry": "r1_numel"},
        "split_params": {"x": "X0BLOCK"},
        "tiling_params": {"x": "X0BLOCK_SUB", "ry": "R1BLOCK_SUB"},
        "low_dim_axes": ["ry"],
        "reduction_axes": ["ry"],
    }
    grid = lambda meta: (meta["X0BLOCK"], )
    act_res = triton_max_last_dim1[grid]()

    check_axes_parse_res(act_res, ref_res)


def test_triton_max_last_dim_case2(mock_autotuner):
    import triton.backends.ascend.runtime

    @triton.autotune(configs=[], key=["x0_numel", "r1_numel"])
    @triton.jit
    def triton_max_last_dim2(
        in_ptr0,
        out_ptr0,
        x0_numel,
        r1_numel,
        X0BLOCK: tl.constexpr,
        X0BLOCK_SUB: tl.constexpr,
        R1BLOCK_SUB: tl.constexpr,
    ):
        x0_offset = tl.program_id(0) * X0BLOCK
        base_x0 = tl.arange(0, X0BLOCK_SUB)
        loops_x0 = (X0BLOCK + X0BLOCK_SUB - 1) // X0BLOCK_SUB
        base_r1 = tl.arange(0, R1BLOCK_SUB)
        loops_r1 = (r1_numel + R1BLOCK_SUB - 1) // R1BLOCK_SUB
        for loop_x0 in range(loops_x0):
            x0 = x0_offset + (loop_x0 * X0BLOCK_SUB) + base_x0[:, None]
            x0_mask = x0 < min(X0BLOCK + x0_offset, x0_numel)
            block_val = tl.full([X0BLOCK_SUB, R1BLOCK_SUB], float("-inf"), tl.float32)
            for loop_r1 in range(loops_r1):
                r1 = (loop_r1 * R1BLOCK_SUB) + base_r1[None, :]
                r1_mask = r1 < r1_numel
                tmp = tl.load(in_ptr0 + (r1 + r1_numel * x0), r1_mask & x0_mask, other=float("-inf"))
                block_val = tl.maximum(block_val, tmp)
            # Reduce along axis=-1 (the last dimension, equivalent to axis=1 in 2D)
            block_res = tl.max(block_val, axis=-1)[:, None]  # <- negative axis index (last dim)
            tl.store(out_ptr0 + x0, block_res, x0_mask)

    ref_res = {
        "keys": {"x": "x0_numel", "ry": "r1_numel"},
        "split_params": {"x": "X0BLOCK"},
        "tiling_params": {"x": "X0BLOCK_SUB", "ry": "R1BLOCK_SUB"},
        "low_dim_axes": ["ry"],
        "reduction_axes": ["ry"],
    }
    grid = lambda meta: (meta["X0BLOCK"], )
    act_res = triton_max_last_dim2[grid]()

    check_axes_parse_res(act_res, ref_res)


def test_triton_max_last_dim_case3(mock_autotuner):
    import triton.backends.ascend.runtime

    @triton.autotune(configs=[], key=["x0_numel", "r1_numel"])
    @triton.jit
    def triton_max_last_dim3(
        in_ptr0,
        out_ptr0,
        x0_numel,
        r1_numel,
        X0BLOCK: tl.constexpr,
        X0BLOCK_SUB: tl.constexpr,
        R1BLOCK_SUB: tl.constexpr,
    ):
        x0_offset = tl.program_id(0) * X0BLOCK
        base_x0 = tl.arange(0, X0BLOCK_SUB)
        loops_x0 = (X0BLOCK + X0BLOCK_SUB - 1) // X0BLOCK_SUB
        base_r1 = tl.arange(0, R1BLOCK_SUB)
        loops_r1 = (r1_numel + R1BLOCK_SUB - 1) // R1BLOCK_SUB
        for loop_x0 in range(loops_x0):
            x0 = x0_offset + (loop_x0 * X0BLOCK_SUB) + base_x0[:, None]
            x0_mask = x0 < min(X0BLOCK + x0_offset, x0_numel)
            block_val = tl.full([X0BLOCK_SUB, R1BLOCK_SUB], float("-inf"), tl.float32)
            for loop_r1 in range(loops_r1):
                r1 = (loop_r1 * R1BLOCK_SUB) + base_r1[None, :]
                r1_mask = r1 < r1_numel
                tmp = tl.load(in_ptr0 + (r1 + r1_numel * x0), r1_mask & x0_mask, other=float("-inf"))
                block_val = tl.maximum(block_val, tmp)
            # Reduce along axis=1, passed as a positional argument (not keyword `axis=...`)
            block_res = tl.max(block_val, 1)[:, None]  # <- explicit positive axis index
            tl.store(out_ptr0 + x0, block_res, x0_mask)

    ref_res = {
        "keys": {"x": "x0_numel", "ry": "r1_numel"},
        "split_params": {"x": "X0BLOCK"},
        "tiling_params": {"x": "X0BLOCK_SUB", "ry": "R1BLOCK_SUB"},
        "low_dim_axes": ["ry"],
        "reduction_axes": ["ry"],
    }
    grid = lambda meta: (meta["X0BLOCK"], )
    act_res = triton_max_last_dim3[grid]()

    check_axes_parse_res(act_res, ref_res)
