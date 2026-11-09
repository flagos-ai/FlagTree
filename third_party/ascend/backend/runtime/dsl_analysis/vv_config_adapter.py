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

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

_DEFAULT_KEY_AXES = ("x", "y", "z", "w", "v", "t")


@dataclass(frozen=True)
class VvConfigAdapterResult:
    split_params: Dict[str, str]
    tiling_params: Dict[str, str]
    low_dim_axes: List[str] = field(default_factory=list)
    reduction_axes: List[str] = field(default_factory=list)
    axis_length_exprs: Dict[str, str] = field(default_factory=dict)
    fixed_tiling_exprs: Dict[str, str] = field(default_factory=dict)
    axis_dynamic_sources: Dict[str, str] = field(default_factory=dict)
    axis_pid_dims: Dict[str, int] = field(default_factory=dict)
    inferred_keys: Dict[str, str] = field(default_factory=dict)
    status: str = "ok"
    diagnostics: List[str] = field(default_factory=list)


def _dedup_tunable_by_first_axis(axis_tunable_pairs: Sequence[Tuple[int, str]], ) -> List[Tuple[int, str]]:
    ordered: List[Tuple[int, str]] = []
    seen = set()
    for axis_index, tunable_name in axis_tunable_pairs:
        if tunable_name in seen:
            continue
        ordered.append((axis_index, tunable_name))
        seen.add(tunable_name)
    return ordered


def _resolve_key_axis_name(ordered_key_axes: List[str], axis_index: int) -> str:
    if not ordered_key_axes:
        if axis_index < 0:
            return _DEFAULT_KEY_AXES[0]
        if axis_index >= len(_DEFAULT_KEY_AXES):
            return _DEFAULT_KEY_AXES[-1]
        return _DEFAULT_KEY_AXES[axis_index]
    if axis_index < 0:
        return ordered_key_axes[0]
    if axis_index >= len(ordered_key_axes):
        return ordered_key_axes[-1]
    return ordered_key_axes[axis_index]


def adapt_vv_v2_to_vector_inputs(
    key_axis_to_param: Dict[str, str],
    vv_parse_result: object,
) -> Optional[VvConfigAdapterResult]:
    if vv_parse_result is None:
        return None

    direct_split = dict(getattr(vv_parse_result, "split_params", {}) or {})
    direct_tiling = dict(getattr(vv_parse_result, "tiling_params", {}) or {})
    direct_low_dim = list(getattr(vv_parse_result, "low_dim_axes", []) or [])
    direct_reduction = list(getattr(vv_parse_result, "reduction_axes", []) or [])
    direct_axis_length_exprs = dict(getattr(vv_parse_result, "axis_length_exprs", {}) or {})
    direct_fixed_tiling_exprs = dict(getattr(vv_parse_result, "fixed_tiling_exprs", {}) or {})
    direct_axis_dynamic_sources = dict(getattr(vv_parse_result, "axis_dynamic_sources", {}) or {})
    direct_axis_pid_dims = dict(getattr(vv_parse_result, "axis_pid_dims", {}) or {})
    direct_inferred_keys = dict(getattr(vv_parse_result, "inferred_keys", {}) or {})
    direct_status = getattr(vv_parse_result, "status", "ok")
    direct_diagnostics = list(getattr(vv_parse_result, "diagnostics", []) or [])

    if direct_status == "failed":
        return None

    if (direct_split or direct_tiling or direct_low_dim or direct_reduction or direct_axis_length_exprs
            or direct_fixed_tiling_exprs or direct_axis_dynamic_sources or direct_axis_pid_dims
            or direct_inferred_keys):
        return VvConfigAdapterResult(
            split_params=direct_split,
            tiling_params=direct_tiling,
            low_dim_axes=direct_low_dim,
            reduction_axes=direct_reduction,
            axis_length_exprs=direct_axis_length_exprs,
            fixed_tiling_exprs=direct_fixed_tiling_exprs,
            axis_dynamic_sources=direct_axis_dynamic_sources,
            axis_pid_dims=direct_axis_pid_dims,
            inferred_keys=direct_inferred_keys,
            status=direct_status,
            diagnostics=direct_diagnostics,
        )

    axes = list(getattr(vv_parse_result, "axes", []))
    if not axes:
        return None

    axis_tunable_pairs: List[Tuple[int, str]] = []
    for item in axes:
        split_param = getattr(item, "split_param", None)
        if isinstance(split_param, str) and split_param:
            axis_index = getattr(item, "axis_index", None)
            if isinstance(axis_index, int):
                axis_tunable_pairs.append((axis_index, split_param))
            continue
        tunable = getattr(item, "tunable_param", None)
        if not isinstance(tunable, str) or not tunable:
            continue
        axis_index = getattr(item, "axis_index", None)
        if not isinstance(axis_index, int):
            continue
        axis_tunable_pairs.append((axis_index, tunable))

    axis_tunable_pairs = _dedup_tunable_by_first_axis(axis_tunable_pairs)
    if not axis_tunable_pairs:
        if direct_status != "ok":
            return VvConfigAdapterResult(
                split_params={},
                tiling_params={},
                low_dim_axes=direct_low_dim,
                reduction_axes=direct_reduction,
                axis_length_exprs=direct_axis_length_exprs,
                fixed_tiling_exprs=direct_fixed_tiling_exprs,
                axis_dynamic_sources=direct_axis_dynamic_sources,
                axis_pid_dims=direct_axis_pid_dims,
                inferred_keys=direct_inferred_keys,
                status=direct_status,
                diagnostics=direct_diagnostics,
            )
        return None

    ordered_key_axes = list(key_axis_to_param.keys())
    split_params: Dict[str, str] = {}
    tiling_params: Dict[str, str] = {}
    used_axes = set()

    for idx, (axis_index, tunable_name) in enumerate(axis_tunable_pairs):
        key_axis = _resolve_key_axis_name(ordered_key_axes, axis_index)
        if idx == 0:
            split_params[key_axis] = tunable_name
            used_axes.add(key_axis)
            continue
        if key_axis in used_axes:
            continue
        tiling_params[key_axis] = tunable_name
        used_axes.add(key_axis)

    if not split_params and not tiling_params:
        if direct_status != "ok":
            return VvConfigAdapterResult(
                split_params={},
                tiling_params={},
                low_dim_axes=direct_low_dim,
                reduction_axes=direct_reduction,
                axis_length_exprs=direct_axis_length_exprs,
                fixed_tiling_exprs=direct_fixed_tiling_exprs,
                axis_dynamic_sources=direct_axis_dynamic_sources,
                axis_pid_dims=direct_axis_pid_dims,
                inferred_keys=direct_inferred_keys,
                status=direct_status,
                diagnostics=direct_diagnostics,
            )
        return None
    return VvConfigAdapterResult(
        split_params=split_params,
        tiling_params=tiling_params,
        low_dim_axes=direct_low_dim,
        reduction_axes=direct_reduction,
        axis_length_exprs=direct_axis_length_exprs,
        fixed_tiling_exprs=direct_fixed_tiling_exprs,
        axis_dynamic_sources=direct_axis_dynamic_sources,
        axis_pid_dims=direct_axis_pid_dims,
        inferred_keys=direct_inferred_keys,
        status=direct_status,
        diagnostics=direct_diagnostics,
    )
