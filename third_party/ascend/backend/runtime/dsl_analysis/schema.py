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
from typing import (Any, Dict, List, Literal, Mapping, Optional, Protocol, Sequence, Set)


@dataclass(frozen=True)
class ParameterSpec:
    name: str
    is_constexpr: bool = False
    has_default: bool = False


@dataclass(frozen=True)
class SignatureInfo:
    parameters: Sequence[ParameterSpec]

    def parameter_names(self) -> List[str]:
        return [param.name for param in self.parameters]

    def constexpr_names(self) -> List[str]:
        return [param.name for param in self.parameters if param.is_constexpr]

    def is_constexpr(self, parameter_name: str) -> bool:
        for param in self.parameters:
            if param.name == parameter_name:
                return param.is_constexpr
        return False

    def has_default(self, parameter_name: str) -> bool:
        for param in self.parameters:
            if param.name == parameter_name:
                return param.has_default
        return False


@dataclass(frozen=True)
class MissingTunableContext:
    arg_names: Sequence[str]
    provided_args: Mapping[str, object]
    signature: SignatureInfo
    split_params: Mapping[str, str]
    tiling_params: Mapping[str, str]
    launch_info: Optional[Any] = None

    def missing_arg_names(self) -> List[str]:
        provided = set(self.provided_args.keys())
        return [name for name in self.arg_names if name not in provided]

    def fixed_tunable_names(self) -> Set[str]:
        return set(self.split_params.values()) | set(self.tiling_params.values())


class MissingTunablePolicy(Protocol):

    def select_missing_tunable(self, context: MissingTunableContext) -> List[str]:
        ...


AXIS_LENGTH_STATE_TUNABLE = "tunable"
AXIS_LENGTH_STATE_FIXED_COMPILE_TIME = "fixed_compile_time"
AXIS_LENGTH_STATE_RUNTIME_NON_TUNABLE = "runtime_non_tunable"


@dataclass(frozen=True)
class AxisLengthStateInfo:
    axis_symbol: str
    length_symbol: Optional[str]
    state: str
    value: Optional[int] = None


@dataclass(frozen=True)
class AxisLengthStateResult:
    axis_states: Dict[str, AxisLengthStateInfo]


CV_PARSER_VERSION_V1 = "cv-parse-v1"
CV_PARSE_STATUS_OK = "ok"
CV_PARSE_STATUS_PARTIAL = "partial"
CV_PARSE_STATUS_FAILED = "failed"

CvParserMode = Literal["cube", "mix"]
CvParseStatus = Literal["ok", "partial", "failed"]
CvAxisRole = Literal["M", "N", "K", "OTHER"]
CvTunableSource = Literal["signature", "axis_length", "launch_semantics"]
CvAxisLengthState = Literal[
    AXIS_LENGTH_STATE_TUNABLE,
    AXIS_LENGTH_STATE_FIXED_COMPILE_TIME,
    AXIS_LENGTH_STATE_RUNTIME_NON_TUNABLE,
    "unknown",
]
CvAxisDynamicSource = Literal["none", "dynamic_from_load"]


@dataclass(frozen=True)
class CvAxisLengthInfo:
    axis_symbol: Optional[str]
    length_expr: Optional[str]
    state: CvAxisLengthState
    tunable_param: Optional[str]
    const_value: Optional[int] = None
    dynamic_source: CvAxisDynamicSource = "none"


@dataclass(frozen=True)
class CvDotSiteInfo:
    site_id: int
    lhs: Optional[str]
    rhs: Optional[str]
    m: CvAxisLengthInfo
    n: CvAxisLengthInfo
    k: CvAxisLengthInfo
    scope: Optional[str] = None


@dataclass(frozen=True)
class CvTunableParamInfo:
    name: str
    role: CvAxisRole
    source: CvTunableSource


@dataclass(frozen=True)
class CvParseResult:
    version: str
    mode: CvParserMode
    status: CvParseStatus
    signature_params: Sequence[str]
    dot_sites: List[CvDotSiteInfo]
    tunable_params: List[CvTunableParamInfo]
    missing_tunable_params: List[str]
    diagnostics: List[str]
