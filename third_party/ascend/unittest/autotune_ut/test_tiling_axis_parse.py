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

import pytest
import triton
import triton.language as tl
from test_common import check_axes_parse_res, mock_autotuner


def test_tiling_axis_parse_base_case1(mock_autotuner):
    import triton.backends.ascend.runtime

    @triton.autotune(configs=[], key=["n_elements"])
    @triton.jit
    def triton_tiling_axis_parse_base_case1(x_ptr, y_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr,
                                            BLOCK_SUB: tl.constexpr):
        offset = tl.program_id(axis=0) * BLOCK_SIZE
        base = tl.arange(0, BLOCK_SUB)
        loops = (BLOCK_SIZE + BLOCK_SUB - 1) // BLOCK_SUB  # <-
        for loop in range(loops):
            offsets = offset + (loop * BLOCK_SUB) + base
            mask = offsets < min(BLOCK_SIZE + offset, n_elements)

            x = tl.load(x_ptr + offsets, mask=mask)
            y = tl.load(y_ptr + offsets, mask=mask)
            output = x + y

            tl.store(output_ptr + offsets, output, mask=mask)

    ref_res = {
        "keys": {"x": "n_elements"},
        "split_params": {"x": "BLOCK_SIZE"},
        "tiling_params": {"x": "BLOCK_SUB"},
        "low_dim_axes": ["x"],
        "reduction_axes": [],
    }
    grid = lambda meta: (meta["BLOCK_SIZE"], )
    act_res = triton_tiling_axis_parse_base_case1[grid]()

    check_axes_parse_res(act_res, ref_res)


@pytest.mark.skip
def test_tiling_axis_parse_base_case2(mock_autotuner):
    import triton.backends.ascend.runtime

    @triton.autotune(configs=[], key=["n_elements"])
    @triton.jit
    def triton_tiling_axis_parse_base_case2(x_ptr, y_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr,
                                            BLOCK_SUB: tl.constexpr):
        offset = tl.program_id(axis=0) * BLOCK_SIZE
        base = tl.arange(0, BLOCK_SUB)
        for offset_sub in range(0, BLOCK_SIZE, BLOCK_SUB):
            offsets = offset + offset_sub + base[:]  # <-
            mask = offsets < min(BLOCK_SIZE + offset, n_elements)

            x = tl.load(x_ptr + offsets, mask=mask)
            y = tl.load(y_ptr + offsets, mask=mask)
            output = x + y

            tl.store(output_ptr + offsets, output, mask=mask)

    ref_res = {
        "keys": {"x": "n_elements"},
        "split_params": {"x": "BLOCK_SIZE"},
        "tiling_params": {"x": "BLOCK_SUB"},
        "low_dim_axes": ["x"],
        "reduction_axes": [],
    }
    grid = lambda meta: (meta["BLOCK_SIZE"], )
    act_res = triton_tiling_axis_parse_base_case2[grid]()

    check_axes_parse_res(act_res, ref_res)


@pytest.mark.skip
def test_tiling_axis_parse_base_case3(mock_autotuner):
    import triton.backends.ascend.runtime

    @triton.autotune(configs=[], key=["n_elements"])
    @triton.jit
    def triton_tiling_axis_parse_base_case3(x_ptr, y_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr,
                                            BLOCK_SUB: tl.constexpr):
        offset = tl.program_id(axis=0) * BLOCK_SIZE
        base = tl.arange(0, BLOCK_SUB)[:]  # <-
        for offset_sub in range(0, BLOCK_SIZE, BLOCK_SUB):
            offsets = offset + offset_sub + base
            mask = offsets < min(BLOCK_SIZE + offset, n_elements)

            x = tl.load(x_ptr + offsets, mask=mask)
            y = tl.load(y_ptr + offsets, mask=mask)
            output = x + y

            tl.store(output_ptr + offsets, output, mask=mask)

    ref_res = {
        "keys": {"x": "n_elements"},
        "split_params": {"x": "BLOCK_SIZE"},
        "tiling_params": {"x": "BLOCK_SUB"},
        "low_dim_axes": ["x"],
        "reduction_axes": [],
    }
    grid = lambda meta: (meta["BLOCK_SIZE"], )
    act_res = triton_tiling_axis_parse_base_case3[grid]()

    check_axes_parse_res(act_res, ref_res)
