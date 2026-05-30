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


def _mock_single_reduction_axis(self):
    if "x" in self.keys and "rx" not in self.keys:
        self.keys["rx"] = self.keys.pop("x")
    return ["rx"]


def test_low_dim_axis_parse_base_case1(mock_autotuner):
    import triton.backends.ascend.runtime

    @triton.autotune(configs=[], key=["n_elements"])
    @triton.jit
    def triton_low_dim_axis_parse_base_case1(x_ptr, y_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
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
    act_res = triton_low_dim_axis_parse_base_case1[grid]()

    check_axes_parse_res(act_res, ref_res)


def test_low_dim_axis_parse_empty_is_non_fatal(mock_autotuner):
    import triton.backends.ascend.runtime

    @triton.autotune(configs=[], key=["n_elements"])
    @triton.jit
    def triton_low_dim_axis_parse_empty_case(x_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
        pid = tl.program_id(axis=0)
        block_start = pid * BLOCK_SIZE
        mask = block_start < n_elements

        _ = tl.load(x_ptr + block_start, mask=mask, other=0)

    ref_res = {
        "keys": {"x": "n_elements"},
        "split_params": {"x": "BLOCK_SIZE"},
        "tiling_params": {},
        "low_dim_axes": [],
        "reduction_axes": [],
    }
    grid = lambda meta: (meta["BLOCK_SIZE"], )
    act_res = triton_low_dim_axis_parse_empty_case[grid]()

    check_axes_parse_res(act_res, ref_res)


def test_low_dim_axis_parse_empty_no_persistent_reduction_index_error(mock_autotuner):
    import triton.backends.ascend.runtime
    from triton.backends.ascend.runtime.autotuner import AutoTilingTuner

    @triton.autotune(configs=[], key=["n_elements"])
    @triton.jit
    def triton_low_dim_axis_parse_guard_case(x_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
        pid = tl.program_id(axis=0)
        block_start = pid * BLOCK_SIZE
        mask = block_start < n_elements

        _ = tl.load(x_ptr + block_start, mask=mask, other=0)

    with mock.patch.object(AutoTilingTuner, "_autoparse_reduction_axes", return_value=["rx"]), \
         mock.patch.object(AutoTilingTuner, "_autoparse_low_dim_axes", return_value=[]), \
         mock.patch.object(AutoTilingTuner, "_autoparse_split_params", return_value={}), \
         mock.patch.object(AutoTilingTuner, "_autoparse_tiling_params", return_value={}):
        act_res = triton_low_dim_axis_parse_guard_case[(1, )]()

    assert act_res["low_dim_axes"] == []
    assert act_res["reduction_axes"] == ["rx"]


def test_persistent_reduction_inner_axis_threshold(mock_autotuner):
    import triton.backends.ascend.runtime
    from triton.backends.ascend.runtime.autotuner import AutoTilingTuner

    @triton.autotune(configs=[], key=["n_elements"])
    @triton.jit
    def triton_persistent_reduction_inner_axis_case(x_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
        pid = tl.program_id(axis=0)
        block_start = pid * BLOCK_SIZE
        mask = block_start < n_elements
        _ = tl.load(x_ptr + block_start, mask=mask, other=0)

    with mock.patch.object(AutoTilingTuner, "_autoparse_reduction_axes", new=_mock_single_reduction_axis), \
         mock.patch.object(AutoTilingTuner, "_autoparse_low_dim_axes", return_value=["rx"]), \
         mock.patch.object(AutoTilingTuner, "_autoparse_split_params", return_value={}), \
         mock.patch.object(AutoTilingTuner, "_autoparse_tiling_params", return_value={}):
        act_res = triton_persistent_reduction_inner_axis_case[(1, )](None, 1024)

    assert act_res["persistent_reduction"] is True


def test_persistent_reduction_outer_axis_threshold(mock_autotuner):
    import triton.backends.ascend.runtime
    from triton.backends.ascend.runtime.autotuner import AutoTilingTuner

    @triton.autotune(configs=[], key=["n_elements"])
    @triton.jit
    def triton_persistent_reduction_outer_axis_case(x_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
        pid = tl.program_id(axis=0)
        block_start = pid * BLOCK_SIZE
        mask = block_start < n_elements
        _ = tl.load(x_ptr + block_start, mask=mask, other=0)

    with mock.patch.object(AutoTilingTuner, "_autoparse_reduction_axes", new=_mock_single_reduction_axis), \
         mock.patch.object(AutoTilingTuner, "_autoparse_low_dim_axes", return_value=[]), \
         mock.patch.object(AutoTilingTuner, "_autoparse_split_params", return_value={}), \
         mock.patch.object(AutoTilingTuner, "_autoparse_tiling_params", return_value={}):
        act_res = triton_persistent_reduction_outer_axis_case[(1, )](None, 64)

    assert act_res["persistent_reduction"] is True


def test_persistent_reduction_outer_axis_over_threshold(mock_autotuner):
    import triton.backends.ascend.runtime
    from triton.backends.ascend.runtime.autotuner import AutoTilingTuner

    @triton.autotune(configs=[], key=["n_elements"])
    @triton.jit
    def triton_persistent_reduction_outer_axis_over_threshold_case(x_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
        pid = tl.program_id(axis=0)
        block_start = pid * BLOCK_SIZE
        mask = block_start < n_elements
        _ = tl.load(x_ptr + block_start, mask=mask, other=0)

    with mock.patch.object(AutoTilingTuner, "_autoparse_reduction_axes", new=_mock_single_reduction_axis), \
         mock.patch.object(AutoTilingTuner, "_autoparse_low_dim_axes", return_value=[]), \
         mock.patch.object(AutoTilingTuner, "_autoparse_split_params", return_value={}), \
         mock.patch.object(AutoTilingTuner, "_autoparse_tiling_params", return_value={}):
        act_res = triton_persistent_reduction_outer_axis_over_threshold_case[(1, )](None, 65)

    assert act_res["persistent_reduction"] is False
