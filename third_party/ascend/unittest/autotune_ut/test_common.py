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
import pytest
import torch


def MockAutoTilingTunerRun(self, *args, **kwargs):
    self.nargs = dict(zip(self.arg_names, args))

    # generate key
    all_args = {**self.nargs, **kwargs}
    try:
        self._autoparse_axis_params(all_args)
    except ValueError as e:
        if "Missing required arguments" in str(e):
            pass
        else:
            raise
    return {
        "keys": self.keys,
        "split_params": self.split_params,
        "tiling_params": self.tiling_params,
        "low_dim_axes": self.low_dim_axes,
        "reduction_axes": self.reduction_axes,
        "persistent_reduction": self.persistent_reduction,
    }


def check_axes_parse_res(act: dict, ref: dict):
    """
    Compare two axes parse results that may use different symbolic axis names,
    but map to the same semantic dimensions via the 'keys' field.
    """
    ref_keys = ref["keys"]
    act_keys = act["keys"]

    assert set(ref_keys.values()) == set(act_keys.values()), \
        f"Semantic dimensions mismatch: ref={set(ref_keys.values())}, act={set(act_keys.values())}"

    def normalize_param_dict(param_dict: dict, sym_to_sem: dict) -> dict:
        """Convert {symbol: value} -> {semantic: value}"""
        return {sym_to_sem[sym]: value for sym, value in param_dict.items()}

    ref_split = normalize_param_dict(ref["split_params"], ref_keys)
    act_split = normalize_param_dict(act["split_params"], act_keys)

    ref_tiling = normalize_param_dict(ref["tiling_params"], ref_keys)
    act_tiling = normalize_param_dict(act["tiling_params"], act_keys)

    def normalize_axis_list(axis_list: list, sym_to_sem: dict) -> list:
        return sorted(sym_to_sem[sym] for sym in axis_list)

    ref_low = normalize_axis_list(ref["low_dim_axes"], ref_keys)
    act_low = normalize_axis_list(act["low_dim_axes"], act_keys)

    ref_red = normalize_axis_list(ref["reduction_axes"], ref_keys)
    act_red = normalize_axis_list(act["reduction_axes"], act_keys)

    # Compare normalized structures
    assert ref_split == act_split, f"split_params mismatch: {ref_split} vs {act_split}"
    assert ref_tiling == act_tiling, f"tiling_params mismatch: {ref_tiling} vs {act_tiling}"
    assert ref_low == act_low, f"low_dim_axes mismatch: {ref_low} vs {act_low}"
    assert ref_red == act_red, f"reduction_axes mismatch: {ref_red} vs {act_red}"


@pytest.fixture
def mock_autotuner():
    with mock.patch("triton.backends.ascend.runtime.autotuner.AutoTilingTuner.run", new=MockAutoTilingTunerRun):
        yield


def generate_tensor(shape, dtype):
    if dtype == 'float32' or dtype == 'float16' or dtype == 'bfloat16':
        return torch.randn(size=shape, dtype=eval('torch.' + dtype))
    elif dtype == 'int32' or dtype == 'int64' or dtype == 'int16':
        return torch.randint(low=0, high=2000, size=shape, dtype=eval('torch.' + dtype))
    elif dtype == 'int8':
        return torch.randint(low=0, high=127, size=shape, dtype=eval('torch.' + dtype))
    elif dtype == 'bool':
        return torch.randint(low=0, high=2, size=shape).bool()
    elif dtype == 'uint8':
        return torch.randint(low=0, high=255, size=shape, dtype=torch.uint8)
    else:
        raise ValueError('Invalid parameter \"dtype\" is found : {}'.format(dtype))
