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


def _axis_bases(axes):
    return [axis[1:] if isinstance(axis, str) and axis.startswith("r") else axis for axis in axes]


def _run_autoparse_and_generate(tuner, all_args):
    tuner.nargs = {}
    tuner.is_simt_mode = False
    tuner.user_specified_multibuffer = None
    tuner.user_specified_num_stages = None
    tuner.user_specified_warps = None
    dtype = all_args["x_ptr"].dtype
    tuner._autoparse_axis_params(all_args)
    tuner._gen_tile_configs({"x": all_args["n_cols"]}, dtype, all_args)
    return {
        "persistent_reduction": tuner.persistent_reduction,
        "gen_configs": [dict(config.kwargs) for config in tuner.gen_configs],
        "reduction_axes": list(tuner.reduction_axes or []),
        "low_dim_axes": list(tuner.low_dim_axes or []),
        "tiling_params": dict(tuner.tiling_params or {}),
    }


class _FakeTensor:

    def __init__(self, dtype):
        self.dtype = dtype


def test_vector_parse_state_resets_persistent_reduction_between_shapes():
    import torch
    import triton.backends.ascend.runtime

    @triton.autotune(
        configs=[],
        key=["n_cols"],
        hints={"auto_gen_config": True},
    )
    @triton.jit
    def poly_norm_like_kernel(
        x_ptr,
        y_ptr,
        n_cols,
        BLOCK_SIZE: tl.constexpr,
    ):
        acc = tl.zeros([], dtype=tl.float32)
        for start in range(0, n_cols, BLOCK_SIZE):
            offsets = start + tl.arange(0, BLOCK_SIZE)
            mask = offsets < n_cols
            x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
            acc += tl.sum(x * x, axis=0)
        scale = acc / n_cols
        for start in range(0, n_cols, BLOCK_SIZE):
            offsets = start + tl.arange(0, BLOCK_SIZE)
            mask = offsets < n_cols
            x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
            tl.store(y_ptr + offsets, x * scale, mask=mask)

    tuner = poly_norm_like_kernel
    fake_x = _FakeTensor(torch.bfloat16)
    fake_y = _FakeTensor(torch.bfloat16)

    with mock.patch(
            "triton.backends.ascend.runtime.autotuner.get_byte_per_numel",
            lambda dtype: 2,
    ):
        small = _run_autoparse_and_generate(
            tuner,
            {"x_ptr": fake_x, "y_ptr": fake_y, "n_cols": 1024},
        )
        large = _run_autoparse_and_generate(
            tuner,
            {"x_ptr": fake_x, "y_ptr": fake_y, "n_cols": 32768},
        )

    assert small["persistent_reduction"] is True
    assert large["tiling_params"] == {"x": "BLOCK_SIZE"}
    assert large["low_dim_axes"] == ["x"]
    assert _axis_bases(large["reduction_axes"]) == ["x"]
    assert large["persistent_reduction"] is False
    assert large["gen_configs"]
    assert any(config.get("BLOCK_SIZE", 0) <= 8192 for config in large["gen_configs"])
