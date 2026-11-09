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


def _serialize_vector_axes(vector_axes):
    if vector_axes is None:
        return {
            "axis_length_exprs": {},
            "fixed_tiling_exprs": {},
            "split_params": {},
            "tiling_params": {},
            "low_dim_axes": [],
            "reduction_axes": [],
        }
    return {
        "axis_length_exprs": dict(getattr(vector_axes, "axis_length_exprs", {}) or {}),
        "fixed_tiling_exprs": dict(getattr(vector_axes, "fixed_tiling_exprs", {}) or {}),
        "split_params": dict(getattr(vector_axes, "split_params", {}) or {}),
        "tiling_params": dict(getattr(vector_axes, "tiling_params", {}) or {}),
        "low_dim_axes": list(getattr(vector_axes, "low_dim_axes", []) or []),
        "reduction_axes": list(getattr(vector_axes, "reduction_axes", []) or []),
    }


def _get_semantic_axes_state(result: dict):
    vector_axes = result.get("vector_axes") or {}
    if vector_axes.get("axis_length_exprs"):
        return {
            "keys": dict(vector_axes["axis_length_exprs"]),
            "split_params": dict(vector_axes.get("split_params", {}) or {}),
            "tiling_params": dict(vector_axes.get("tiling_params", {}) or {}),
            "low_dim_axes": list(vector_axes.get("low_dim_axes", []) or []),
            "reduction_axes": list(vector_axes.get("reduction_axes", []) or []),
        }

    keys = result.get("keys")
    return {
        "keys": dict(keys or {}) if isinstance(keys, dict) else {},
        "split_params": dict(result.get("split_params", {}) or {}),
        "tiling_params": dict(result.get("tiling_params", {}) or {}),
        "low_dim_axes": list(result.get("low_dim_axes", []) or []),
        "reduction_axes": list(result.get("reduction_axes", []) or []),
    }


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
    vector_axes = _serialize_vector_axes(getattr(self, "vector_axes", None))
    has_semantic_axes = bool(vector_axes["axis_length_exprs"])
    semantic_state = _get_semantic_axes_state({"keys": getattr(self, "keys", None), "vector_axes": vector_axes})
    return {
        "keys": semantic_state["keys"] if has_semantic_axes else self.keys,
        "vector_axes": vector_axes,
        "split_params": semantic_state["split_params"] if has_semantic_axes else self.split_params,
        "tiling_params": semantic_state["tiling_params"] if has_semantic_axes else self.tiling_params,
        "low_dim_axes": semantic_state["low_dim_axes"] if has_semantic_axes else self.low_dim_axes,
        "reduction_axes": semantic_state["reduction_axes"] if has_semantic_axes else self.reduction_axes,
        "persistent_reduction": self.persistent_reduction,
        "cv_parse_result": getattr(self, "cv_parse_result", None),
        "vv_parse_result_v2": getattr(self, "vv_parse_result_v2", None),
        "vv_adapter_result_v2": getattr(self, "vv_adapter_result_v2", None),
    }


def check_axes_parse_res(act: dict, ref: dict):
    """
    Compare two axes parse results that may use different symbolic axis names,
    but map to the same semantic dimensions via the semantic axis view.
    """
    ref_state = _get_semantic_axes_state(ref)
    act_state = _get_semantic_axes_state(act)
    ref_axis_lengths = ref_state["keys"]
    act_axis_lengths = act_state["keys"]

    assert set(ref_axis_lengths.values()) == set(act_axis_lengths.values()), \
        f"Semantic dimensions mismatch: ref={set(ref_axis_lengths.values())}, act={set(act_axis_lengths.values())}"

    def resolve_symbol(sym: str, sym_to_sem: dict) -> str:
        if sym in sym_to_sem:
            return sym
        if isinstance(sym, str) and sym.startswith("r") and sym[1:] in sym_to_sem:
            return sym[1:]
        raise KeyError(sym)

    def normalize_param_dict(param_dict: dict, sym_to_sem: dict) -> dict:
        """Convert {symbol: value} -> {semantic: value}"""
        return {sym_to_sem[resolve_symbol(sym, sym_to_sem)]: value for sym, value in param_dict.items()}

    ref_split = normalize_param_dict(ref_state["split_params"], ref_axis_lengths)
    act_split = normalize_param_dict(act_state["split_params"], act_axis_lengths)

    ref_tiling = normalize_param_dict(ref_state["tiling_params"], ref_axis_lengths)
    act_tiling = normalize_param_dict(act_state["tiling_params"], act_axis_lengths)

    def normalize_axis_list(axis_list: list, sym_to_sem: dict) -> list:
        return sorted(sym_to_sem[resolve_symbol(sym, sym_to_sem)] for sym in axis_list)

    ref_low = normalize_axis_list(ref_state["low_dim_axes"], ref_axis_lengths)
    act_low = normalize_axis_list(act_state["low_dim_axes"], act_axis_lengths)

    ref_red = normalize_axis_list(ref_state["reduction_axes"], ref_axis_lengths)
    act_red = normalize_axis_list(act_state["reduction_axes"], act_axis_lengths)

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
