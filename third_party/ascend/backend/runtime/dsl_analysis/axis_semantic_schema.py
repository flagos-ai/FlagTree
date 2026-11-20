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

from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass(frozen=True)
class AxisExtent:
    expr: Optional[str]
    state: str
    const_value: Optional[int]
    source: str
    confidence: float
    dynamic_source: str = "none"


@dataclass(frozen=True)
class AxisSplit:
    param: Optional[str]
    pid_dim: Optional[int]
    source: str
    confidence: float


@dataclass(frozen=True)
class AxisTiling:
    param: Optional[str]
    loop_var: Optional[str]
    source: str
    confidence: float
    fixed_expr: Optional[str] = None


@dataclass(frozen=True)
class AxisSemanticInfo:
    axis_name: str
    extent: AxisExtent
    split: AxisSplit
    tiling: AxisTiling
    is_low_dim: bool
    is_reduction: bool
    diagnostics: List[str]


@dataclass(frozen=True)
class AxisSemanticResult:
    axes: Dict[str, AxisSemanticInfo]
    axis_length_exprs: Dict[str, str]
    fixed_tiling_exprs: Dict[str, str]
    axis_pid_dims: Dict[str, int]
    inferred_keys: Dict[str, str]
    split_params: Dict[str, str]
    tiling_params: Dict[str, str]
    low_dim_axes: List[str]
    reduction_axes: List[str]
    status: str
    diagnostics: List[str]
