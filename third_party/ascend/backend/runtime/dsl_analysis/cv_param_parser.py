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

import ast
from typing import Dict, List, Mapping, Optional, Sequence, Set

from .dynamic_source_utils import resolve_dynamic_source
from .entry import (analyze_axis_length_state, analyze_dot_site_mnk, analyze_signature_and_missing_tunable)
from .load_dependency_analyzer import collect_load_derived_symbols
from .schema import (AXIS_LENGTH_STATE_TUNABLE, CV_PARSE_STATUS_FAILED, CV_PARSE_STATUS_OK, CV_PARSE_STATUS_PARTIAL,
                     CV_PARSER_VERSION_V1, CvAxisLengthInfo, CvAxisRole, CvDotSiteInfo, CvParseResult, CvParserMode,
                     CvTunableParamInfo)


def _resolve_axis_state_info(axis_symbol: Optional[str], axis_states: Mapping[str, object]):
    if axis_symbol is None:
        return None

    state_info = axis_states.get(axis_symbol, None)
    if state_info is not None:
        return state_info

    if "|" not in axis_symbol:
        return None
    for symbol in axis_symbol.split("|"):
        state_info = axis_states.get(symbol, None)
        if state_info is not None:
            return state_info
    return None


def _normalize_tunable_param_symbol(symbol: Optional[str]) -> Optional[str]:
    if symbol is None:
        return None
    if not isinstance(symbol, str):
        return None
    if not symbol.isidentifier():
        if "|" not in symbol:
            return None
        candidates = []
        for item in symbol.split("|"):
            token = item.strip()
            if not token or not token.isidentifier():
                continue
            if token in candidates:
                continue
            candidates.append(token)
        if len(candidates) == 1:
            return candidates[0]
        return None
    return symbol


def _to_cv_axis_length_info(
    axis_symbol: Optional[str],
    axis_states: Mapping[str, object],
    *,
    override_length_expr: Optional[str] = None,
    load_derived_symbols: Optional[Set[str]] = None,
) -> CvAxisLengthInfo:
    load_derived_symbols = load_derived_symbols or set()
    state_info = _resolve_axis_state_info(axis_symbol, axis_states)
    if state_info is None:
        length_expr = override_length_expr
        return CvAxisLengthInfo(
            axis_symbol=axis_symbol,
            length_expr=length_expr,
            state="unknown",
            tunable_param=None,
            const_value=None,
            dynamic_source=resolve_dynamic_source(
                axis_symbol=axis_symbol,
                length_expr=length_expr,
                load_derived_symbols=load_derived_symbols,
            ),
        )

    state = getattr(state_info, "state", "unknown")
    base_tunable_symbol = _normalize_tunable_param_symbol(getattr(state_info, "length_symbol", None))
    tunable_param = base_tunable_symbol if state == AXIS_LENGTH_STATE_TUNABLE else None
    length_expr = None
    has_effective_override = (override_length_expr is not None and override_length_expr != axis_symbol)
    # If we cannot infer a separate/global length expression and the axis symbol
    # equals the tunable split parameter itself (e.g. BLOCK_M/BLOCK_N/BLOCK_K),
    # keep length_expr as None to let downstream fallback logic decide defaults.
    suppress_axis_self_length = (not has_effective_override and axis_symbol is not None and tunable_param is not None
                                 and axis_symbol == tunable_param)
    if override_length_expr is not None and override_length_expr != axis_symbol:
        length_expr = override_length_expr
    if length_expr is None and not suppress_axis_self_length:
        length_expr = getattr(state_info, "length_symbol", None)
    value = getattr(state_info, "value", None)
    # Keep tunable parameter as the original tunable length symbol even when
    # we override length_expr to show logical/global axis length.
    return CvAxisLengthInfo(
        axis_symbol=axis_symbol,
        length_expr=length_expr,
        state=state,
        tunable_param=tunable_param,
        const_value=value,
        dynamic_source=resolve_dynamic_source(
            axis_symbol=axis_symbol,
            length_expr=length_expr,
            load_derived_symbols=load_derived_symbols,
        ),
    )


def _collect_role_param_names(dot_sites: Sequence[CvDotSiteInfo]) -> Dict[CvAxisRole, Set[str]]:
    role_map: Dict[CvAxisRole, Set[str]] = {"M": set(), "N": set(), "K": set(), "OTHER": set()}
    for site in dot_sites:
        if site.m.tunable_param:
            role_map["M"].add(site.m.tunable_param)
        if site.n.tunable_param:
            role_map["N"].add(site.n.tunable_param)
        if site.k.tunable_param:
            role_map["K"].add(site.k.tunable_param)
    return role_map


def _infer_role(name: str, role_param_names: Mapping[CvAxisRole, Set[str]]) -> CvAxisRole:
    if name in role_param_names["M"]:
        return "M"
    if name in role_param_names["N"]:
        return "N"
    if name in role_param_names["K"]:
        return "K"
    return "OTHER"


def parse_cv_params(
    func_ast: ast.AST,
    parser_mode: str,
    arg_names: Sequence[str],
    provided_args: Mapping[str, object],
    explicit_tunable_params: Optional[Sequence[str]] = None,
    module_ast: Optional[ast.AST] = None,
    entry_function_name: Optional[str] = None,
) -> CvParseResult:
    mode = (parser_mode or "").lower()
    if mode not in ("cube", "mix"):
        raise ValueError("parse_cv_params only supports parser_mode='cube' or 'mix'.")

    diagnostics = []
    signature_params: List[str] = []
    missing_tunable_params: List[str] = []
    load_derived_symbols: Set[str] = set()

    try:
        sig_result = analyze_signature_and_missing_tunable(
            func_ast,
            arg_names=arg_names,
            provided_args=dict(provided_args),
            explicit_tunable_params=explicit_tunable_params,
            parser_mode=mode,
        )
        signature_params = list(sig_result.signature.parameter_names())
        missing_tunable_params = list(sig_result.missing_tunable)
    except ValueError:
        raise
    except Exception as exc:
        diagnostics.append("signature/missing_tunable analysis failed: {}".format(type(exc).__name__))

    dot_sites_raw = []
    try:
        dot_result = analyze_dot_site_mnk(
            func_ast,
            parser_mode=mode,
            module_ast=module_ast,
            entry_function_name=entry_function_name,
        )
        dot_sites_raw = list(dot_result.dot_sites)
    except Exception as exc:
        diagnostics.append("dot-site analysis failed: {}".format(type(exc).__name__))

    axis_states = {}
    try:
        axis_state_result = analyze_axis_length_state(
            func_ast=func_ast,
            parser_mode=mode,
            arg_names=arg_names,
            provided_args=dict(provided_args),
            module_ast=module_ast,
            entry_function_name=entry_function_name,
        )
        axis_states = dict(axis_state_result.axis_states)
    except Exception as exc:
        diagnostics.append("axis-length-state analysis failed: {}".format(type(exc).__name__))

    try:
        analysis_root = module_ast if module_ast is not None else func_ast
        load_derived_symbols = collect_load_derived_symbols(
            analysis_root,
            provided_args=dict(provided_args),
            entry_function_name=entry_function_name,
        )
    except Exception as exc:
        diagnostics.append("load-dependency analysis failed: {}".format(type(exc).__name__))

    dot_sites = []
    for site in dot_sites_raw:
        dot_sites.append(
            CvDotSiteInfo(
                site_id=int(getattr(site, "site_id", len(dot_sites))),
                lhs=getattr(site, "lhs", None),
                rhs=getattr(site, "rhs", None),
                m=_to_cv_axis_length_info(
                    getattr(site, "m", None),
                    axis_states,
                    override_length_expr=getattr(site, "m_total", None),
                    load_derived_symbols=load_derived_symbols,
                ),
                n=_to_cv_axis_length_info(
                    getattr(site, "n", None),
                    axis_states,
                    override_length_expr=getattr(site, "n_total", None),
                    load_derived_symbols=load_derived_symbols,
                ),
                k=_to_cv_axis_length_info(
                    getattr(site, "k", None),
                    axis_states,
                    override_length_expr=getattr(site, "k_total", None),
                    load_derived_symbols=load_derived_symbols,
                ),
                scope=None,
            ))

    if not dot_sites:
        diagnostics.append("no tl.dot/tl.conv site resolved in cube/mix parser.")

    role_param_names = _collect_role_param_names(dot_sites)
    tunable_params: List[CvTunableParamInfo] = []
    seen = set()
    for name in missing_tunable_params:
        role = _infer_role(name, role_param_names)
        tunable_params.append(CvTunableParamInfo(name=name, role=role, source="signature"))
        seen.add(name)

    for role in ("M", "N", "K"):
        for name in sorted(role_param_names[role]):
            if name in seen:
                continue
            tunable_params.append(CvTunableParamInfo(name=name, role=role, source="axis_length"))
            seen.add(name)

    status = CV_PARSE_STATUS_OK
    if diagnostics:
        status = CV_PARSE_STATUS_PARTIAL
    if not signature_params and not dot_sites and not tunable_params:
        status = CV_PARSE_STATUS_FAILED

    return CvParseResult(
        version=CV_PARSER_VERSION_V1,
        mode=mode,  # type: ignore[arg-type]
        status=status,  # type: ignore[arg-type]
        signature_params=signature_params,
        dot_sites=dot_sites,
        tunable_params=tunable_params,
        missing_tunable_params=missing_tunable_params,
        diagnostics=diagnostics,
    )
