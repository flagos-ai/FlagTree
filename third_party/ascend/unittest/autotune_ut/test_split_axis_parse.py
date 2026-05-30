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

import unittest.mock as mock

import triton
import triton.language as tl

from test_common import check_axes_parse_res, mock_autotuner


def test_split_axis_parse_base_case1(mock_autotuner):
    import triton.backends.ascend.runtime

    @triton.autotune(configs=[], key=["n_elements"])
    @triton.jit
    def triton_split_axis_parse_base_case1(x_ptr, y_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
        pid = tl.program_id(axis=0)
        block_start = pid * BLOCK_SIZE  # <- Separate assignment

        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements

        x = tl.load(x_ptr + offsets, mask=mask)
        y = tl.load(y_ptr + offsets, mask=mask)
        output = x + y

        tl.store(output_ptr + offsets, output, mask=mask)

    ref_res = {
        "keys": {"x": "n_elements"},
        "split_params": {"x": "BLOCK_SIZE"},
        "tiling_params": {},
        "low_dim_axes": ["x"],
        "reduction_axes": [],
    }
    grid = lambda meta: (meta["BLOCK_SIZE"], )
    act_res = triton_split_axis_parse_base_case1[grid]()

    check_axes_parse_res(act_res, ref_res)


def test_split_axis_parse_base_case2(mock_autotuner):
    import triton.backends.ascend.runtime

    @triton.autotune(configs=[], key=["n_elements"])
    @triton.jit
    def triton_split_axis_parse_base_case2(x_ptr, y_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
        block_start = tl.program_id(axis=0) * BLOCK_SIZE  # <- Computed inline but still named

        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements

        x = tl.load(x_ptr + offsets, mask=mask)
        y = tl.load(y_ptr + offsets, mask=mask)
        output = x + y

        tl.store(output_ptr + offsets, output, mask=mask)

    ref_res = {
        "keys": {"x": "n_elements"},
        "split_params": {"x": "BLOCK_SIZE"},
        "tiling_params": {},
        "low_dim_axes": ["x"],
        "reduction_axes": [],
    }
    grid = lambda meta: (meta["BLOCK_SIZE"], )
    act_res = triton_split_axis_parse_base_case2[grid]()

    check_axes_parse_res(act_res, ref_res)


def test_split_axis_parse_base_case3(mock_autotuner):
    import triton.backends.ascend.runtime

    @triton.autotune(configs=[], key=["n_elements"])
    @triton.jit
    def triton_split_axis_parse_base_case3(x_ptr, y_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
        offsets = tl.program_id(axis=0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)  # <- Fully fused
        mask = offsets < n_elements

        x = tl.load(x_ptr + offsets, mask=mask)
        y = tl.load(y_ptr + offsets, mask=mask)
        output = x + y

        tl.store(output_ptr + offsets, output, mask=mask)

    ref_res = {
        "keys": {"x": "n_elements"},
        "split_params": {"x": "BLOCK_SIZE"},
        "tiling_params": {},
        "low_dim_axes": ["x"],
        "reduction_axes": [],
    }
    grid = lambda meta: (meta["BLOCK_SIZE"], )
    act_res = triton_split_axis_parse_base_case3[grid]()

    check_axes_parse_res(act_res, ref_res)


def test_split_axis_parse_base_case4(mock_autotuner):
    import triton.backends.ascend.runtime

    @triton.autotune(configs=[], key=["n_elements"])
    @triton.jit
    def triton_split_axis_parse_base_case4(x_ptr, y_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
        offsets = tl.program_id(axis=0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)  # <- Fully fused
        mask = offsets < n_elements

        x = tl.load(x_ptr + offsets, mask=mask)
        y = tl.load(y_ptr + offsets, mask=mask)
        output_x = x[:, None].to(tl.float32) * 1  # left=Call(func=Attribute(value=Subscipt(...),),))
        output_y = 1 * y[None, :].to(tl.float32)

        output_offsets = tl.program_id(axis=0) * BLOCK_SIZE * BLOCK_SIZE + \
                         tl.arange(0, BLOCK_SIZE)[:, None] * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)[None, :]
        tl.store(output_ptr + output_offsets, output_x + output_y, mask=mask)

    ref_res = {
        "keys": {"x": "n_elements"},
        "split_params": {"x": "BLOCK_SIZE"},
        "tiling_params": {},
        "low_dim_axes": ["x"],
        "reduction_axes": [],
    }
    grid = lambda meta: (meta["BLOCK_SIZE"], )
    act_res = triton_split_axis_parse_base_case4[grid]()

    check_axes_parse_res(act_res, ref_res)


def test_grid_stride_loop_block_only_tiling_semantics(mock_autotuner):
    import triton.backends.ascend.runtime

    @triton.autotune(configs=[], key=["N", "index_len"])
    @triton.jit
    def triton_grid_stride_loop_block_only_tiling_semantics(
        input_ptr,
        output_ptr,
        index_ptr,
        N: tl.constexpr,
        index_len: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        pid_x = tl.program_id(axis=0)
        pid_y = tl.program_id(axis=1)
        grid_x = tl.num_programs(axis=0)
        grid_y = tl.num_programs(axis=1)
        for x in range(pid_x * BLOCK_M, index_len, grid_x * BLOCK_M):
            row_offsets = x + tl.arange(0, BLOCK_M)
            indices = tl.load(index_ptr + row_offsets, mask=row_offsets < index_len, other=0)
            for y in range(pid_y * BLOCK_N, N, grid_y * BLOCK_N):
                col_offsets = y + tl.arange(0, BLOCK_N)
                col_mask = col_offsets < N
                inp_offset = indices[:, None] * N + col_offsets[None, :]
                out_offset = row_offsets[:, None] * N + col_offsets[None, :]
                selected = tl.load(input_ptr + inp_offset, mask=col_mask[None, :], other=0.0)
                tl.store(output_ptr + out_offset, selected, mask=col_mask[None, :])

    act_res = triton_grid_stride_loop_block_only_tiling_semantics[(1, 1)]()
    assert act_res["split_params"] == {}
    assert act_res["tiling_params"] == {"y": "BLOCK_M", "x": "BLOCK_N"}
