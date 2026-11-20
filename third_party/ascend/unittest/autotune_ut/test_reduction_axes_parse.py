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
from test_common import check_axes_parse_res, mock_autotuner

import triton
import triton.language as tl


def assert_outward_semantic_axes_state(act_res, ref_res):

    def normalize_axes(axis_names, keys):
        normalized = []
        for axis in axis_names:
            if axis in keys:
                normalized.append(axis)
            elif isinstance(axis, str) and axis.startswith("r") and axis[1:] in keys:
                normalized.append(axis[1:])
            else:
                normalized.append(axis)
        return normalized

    assert act_res["keys"] == ref_res["keys"]
    assert act_res["split_params"] == ref_res["split_params"]
    assert act_res["tiling_params"] == ref_res["tiling_params"]
    assert act_res["low_dim_axes"] == ref_res["low_dim_axes"]
    assert normalize_axes(act_res["reduction_axes"],
                          act_res["keys"]) == normalize_axes(ref_res["reduction_axes"], ref_res["keys"])


def assert_vv_parser_semantic_axes_state(act_res, ref_res):
    vv_parse_result = act_res["vv_parse_result_v2"]
    assert vv_parse_result is not None
    assert vv_parse_result.axis_length_exprs == ref_res["keys"]
    assert vv_parse_result.split_params == ref_res["split_params"]
    assert vv_parse_result.tiling_params == ref_res["tiling_params"]
    assert vv_parse_result.low_dim_axes == ref_res["low_dim_axes"]
    assert vv_parse_result.reduction_axes == ref_res["reduction_axes"]


def test_triton_max_last_dim_case1(mock_autotuner):
    import triton.backends.ascend.runtime

    @triton.autotune(
        configs=[],
        key=["x0_numel", "r1_numel"],
    )
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
        "keys": {"x": "x0_numel", "y": "r1_numel"},
        "split_params": {"x": "X0BLOCK"},
        "tiling_params": {"x": "X0BLOCK_SUB", "y": "R1BLOCK_SUB"},
        "low_dim_axes": ["y"],
        "reduction_axes": ["y"],
    }
    grid = lambda meta: (meta["X0BLOCK"], )
    act_res = triton_max_last_dim1[grid]()

    assert_outward_semantic_axes_state(act_res, ref_res)
    assert_vv_parser_semantic_axes_state(act_res, ref_res)
    check_axes_parse_res(act_res, ref_res)


def test_triton_max_last_dim_case2(mock_autotuner):
    import triton.backends.ascend.runtime

    @triton.autotune(
        configs=[],
        key=["x0_numel", "r1_numel"],
    )
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
        "keys": {"x": "x0_numel", "y": "r1_numel"},
        "split_params": {"x": "X0BLOCK"},
        "tiling_params": {"x": "X0BLOCK_SUB", "y": "R1BLOCK_SUB"},
        "low_dim_axes": ["y"],
        "reduction_axes": ["y"],
    }
    grid = lambda meta: (meta["X0BLOCK"], )
    act_res = triton_max_last_dim2[grid]()

    assert_outward_semantic_axes_state(act_res, ref_res)
    assert_vv_parser_semantic_axes_state(act_res, ref_res)
    check_axes_parse_res(act_res, ref_res)


def test_triton_max_last_dim_case3(mock_autotuner):
    import triton.backends.ascend.runtime

    @triton.autotune(
        configs=[],
        key=["x0_numel", "r1_numel"],
    )
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
        "keys": {"x": "x0_numel", "y": "r1_numel"},
        "split_params": {"x": "X0BLOCK"},
        "tiling_params": {"x": "X0BLOCK_SUB", "y": "R1BLOCK_SUB"},
        "low_dim_axes": ["y"],
        "reduction_axes": ["y"],
    }
    grid = lambda meta: (meta["X0BLOCK"], )
    act_res = triton_max_last_dim3[grid]()

    assert_outward_semantic_axes_state(act_res, ref_res)
    assert_vv_parser_semantic_axes_state(act_res, ref_res)
    check_axes_parse_res(act_res, ref_res)


@pytest.mark.parametrize("kernel_type", ["vector", "auto"])
def test_reduction_axes_parse_kernel_type_vector_auto_consistency(mock_autotuner, kernel_type):
    import triton.backends.ascend.runtime

    @triton.autotune(
        configs=[],
        key=["x0_numel", "r1_numel"],
        hints={"kernel_type": kernel_type},
    )
    @triton.jit
    def triton_reduction_axes_parse_kernel_type_vector_auto_consistency(
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
            block_res = tl.max(block_val, axis=1)[:, None]
            tl.store(out_ptr0 + x0, block_res, x0_mask)

    ref_res = {
        "keys": {"x": "x0_numel", "y": "r1_numel"},
        "split_params": {"x": "X0BLOCK"},
        "tiling_params": {"x": "X0BLOCK_SUB", "y": "R1BLOCK_SUB"},
        "low_dim_axes": ["y"],
        "reduction_axes": ["y"],
    }
    grid = lambda meta: (meta["X0BLOCK"], )
    act_res = triton_reduction_axes_parse_kernel_type_vector_auto_consistency[grid]()
    assert_outward_semantic_axes_state(act_res, ref_res)
    assert_vv_parser_semantic_axes_state(act_res, ref_res)
    check_axes_parse_res(act_res, ref_res)
