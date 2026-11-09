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

from .entry import (SignatureMissingTunableResult, DotSiteMNKResult, analyze_axis_length_state, analyze_dot_site_mnk,
                    analyze_signature_and_missing_tunable, resolve_missing_tunable_policy)
from .axis_length_resolver import classify_length_symbol, resolve_axis_length_states
from .axis_semantic_schema import (AxisExtent, AxisSemanticInfo, AxisSemanticResult, AxisSplit, AxisTiling)
from .call_graph import (CallSite, FunctionMeta, build_call_sites, build_function_index, find_function_name_by_node)
from .cv_param_parser import parse_cv_params
from .dot_role_annotator import TensorRole, annotate_tensor_roles
from .kernel_classifier import classify_kernel_type_from_dsl, resolve_kernel_type
from .launch_analyzer import (LaunchSemantics, analyze_launch_semantics, looks_like_cube_mix_tunable)
from .missing_tunable_detector import MissingTunableDetector
from .missing_tunable_policies import CubeMixPolicy, VectorPolicy
from .schema import (AXIS_LENGTH_STATE_FIXED_COMPILE_TIME, AXIS_LENGTH_STATE_RUNTIME_NON_TUNABLE,
                     AXIS_LENGTH_STATE_TUNABLE, CV_PARSE_STATUS_FAILED, CV_PARSE_STATUS_OK, CV_PARSE_STATUS_PARTIAL,
                     CV_PARSER_VERSION_V1, AxisLengthStateInfo, AxisLengthStateResult, CvAxisLengthInfo, CvDotSiteInfo,
                     CvParseResult, CvTunableParamInfo, MissingTunableContext, ParameterSpec, SignatureInfo)
from .shape_inference import (DotSiteMNK, infer_dot_sites_mnk, infer_dot_sites_mnk_interprocedural)
from .signature_analyzer import (extract_constexpr_name_set, extract_constexpr_names, extract_signature_info)
from .symbolic_expr import SymbolicExpr
from .vv_config_adapter import (VvConfigAdapterResult, adapt_vv_v2_to_vector_inputs)
from .vv_parser_options import (resolve_vv_parser_v2_enabled, resolve_vv_parser_v2_mode)
from .vv_param_parser_v2 import (VvAxisInfoV2, VvAxisParseResultV2, parse_vv_axis_semantic_v2, parse_vv_axis_info_v2)

__all__ = [
    "CubeMixPolicy",
    "CallSite",
    "CvAxisLengthInfo",
    "CvDotSiteInfo",
    "CvParseResult",
    "CvTunableParamInfo",
    "DotSiteMNK",
    "DotSiteMNKResult",
    "FunctionMeta",
    "VectorPolicy",
    "LaunchSemantics",
    "AxisExtent",
    "AxisSemanticInfo",
    "AxisSemanticResult",
    "AxisSplit",
    "AxisTiling",
    "AXIS_LENGTH_STATE_FIXED_COMPILE_TIME",
    "AXIS_LENGTH_STATE_RUNTIME_NON_TUNABLE",
    "AXIS_LENGTH_STATE_TUNABLE",
    "CV_PARSER_VERSION_V1",
    "CV_PARSE_STATUS_OK",
    "CV_PARSE_STATUS_PARTIAL",
    "CV_PARSE_STATUS_FAILED",
    "AxisLengthStateInfo",
    "AxisLengthStateResult",
    "MissingTunableContext",
    "MissingTunableDetector",
    "ParameterSpec",
    "SignatureInfo",
    "SignatureMissingTunableResult",
    "SymbolicExpr",
    "TensorRole",
    "VvConfigAdapterResult",
    "VvAxisInfoV2",
    "VvAxisParseResultV2",
    "analyze_axis_length_state",
    "analyze_launch_semantics",
    "analyze_dot_site_mnk",
    "analyze_signature_and_missing_tunable",
    "annotate_tensor_roles",
    "build_call_sites",
    "build_function_index",
    "classify_kernel_type_from_dsl",
    "classify_length_symbol",
    "extract_constexpr_name_set",
    "extract_constexpr_names",
    "extract_signature_info",
    "infer_dot_sites_mnk",
    "infer_dot_sites_mnk_interprocedural",
    "find_function_name_by_node",
    "looks_like_cube_mix_tunable",
    "adapt_vv_v2_to_vector_inputs",
    "parse_cv_params",
    "resolve_axis_length_states",
    "resolve_missing_tunable_policy",
    "resolve_kernel_type",
    "resolve_vv_parser_v2_enabled",
    "resolve_vv_parser_v2_mode",
    "parse_vv_axis_semantic_v2",
    "parse_vv_axis_info_v2",
]
