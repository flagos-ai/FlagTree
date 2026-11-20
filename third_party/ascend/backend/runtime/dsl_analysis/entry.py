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
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

from .axis_length_resolver import resolve_axis_length_states
from .launch_analyzer import analyze_launch_semantics
from .missing_tunable_detector import MissingTunableDetector
from .missing_tunable_policies import CubeMixPolicy, VectorPolicy
from .schema import (AxisLengthStateResult, MissingTunableContext, MissingTunablePolicy, SignatureInfo)
from .shape_inference import (DotSiteMNK, infer_dot_sites_mnk, infer_dot_sites_mnk_interprocedural)
from .signature_analyzer import extract_signature_info


@dataclass(frozen=True)
class SignatureMissingTunableResult:
    signature: SignatureInfo
    missing_tunable: List[str]


@dataclass(frozen=True)
class DotSiteMNKResult:
    dot_sites: List[DotSiteMNK]


def resolve_missing_tunable_policy(parser_mode: Optional[str]) -> MissingTunablePolicy:
    mode = (parser_mode or "vector").lower()
    if mode in ("cube", "mix"):
        return CubeMixPolicy()
    return VectorPolicy()


def analyze_signature_and_missing_tunable(
    func_ast: ast.AST,
    arg_names: Sequence[str],
    provided_args: Dict[str, object],
    split_params: Optional[Dict[str, str]] = None,
    tiling_params: Optional[Dict[str, str]] = None,
    explicit_tunable_params: Optional[Sequence[str]] = None,
    parser_mode: Optional[str] = None,
    policy: Optional[MissingTunablePolicy] = None,
) -> SignatureMissingTunableResult:
    split_params = split_params or {}
    tiling_params = tiling_params or {}
    mode = (parser_mode or "vector").lower()

    signature = extract_signature_info(func_ast)
    launch_info = analyze_launch_semantics(func_ast)
    context = MissingTunableContext(
        arg_names=arg_names,
        provided_args=provided_args,
        signature=signature,
        split_params=split_params,
        tiling_params=tiling_params,
        launch_info=launch_info,
    )

    resolved_policy = policy or resolve_missing_tunable_policy(parser_mode)
    detector = MissingTunableDetector(resolved_policy)
    missing_tunable = detector.detect(context)

    explicit_candidates = list(explicit_tunable_params or [])
    if explicit_candidates:
        signature_names = set(signature.parameter_names())
        invalid_names = [
            name for name in explicit_candidates if not isinstance(name, str) or name not in signature_names
        ]
        if invalid_names:
            raise ValueError("hints.tunable_parameter contains unknown parameters: {}".format(invalid_names))
        non_constexpr = [name for name in explicit_candidates if not signature.is_constexpr(name)]
        if non_constexpr:
            raise ValueError(
                "hints.tunable_parameter must reference constexpr parameters only: {}".format(non_constexpr))
        provided = set(provided_args.keys())
        for name in explicit_candidates:
            if name in provided:
                continue
            if name in missing_tunable:
                continue
            missing_tunable.append(name)

    return SignatureMissingTunableResult(
        signature=signature,
        missing_tunable=missing_tunable,
    )


def analyze_dot_site_mnk(
    func_ast: ast.AST,
    parser_mode: Optional[str] = None,
    module_ast: Optional[ast.AST] = None,
    entry_function_name: Optional[str] = None,
) -> DotSiteMNKResult:
    mode = (parser_mode or "vector").lower()
    if mode not in ("cube", "mix"):
        return DotSiteMNKResult(dot_sites=[])
    if module_ast is not None and entry_function_name:
        dot_sites = infer_dot_sites_mnk_interprocedural(
            module_ast=module_ast,
            entry_function_name=entry_function_name,
        )
        return DotSiteMNKResult(dot_sites=dot_sites)
    dot_sites = infer_dot_sites_mnk(func_ast)
    return DotSiteMNKResult(dot_sites=dot_sites)


def analyze_axis_length_state(
    func_ast: ast.AST,
    parser_mode: Optional[str],
    arg_names: Sequence[str],
    provided_args: Dict[str, object],
    module_ast: Optional[ast.AST] = None,
    entry_function_name: Optional[str] = None,
) -> AxisLengthStateResult:
    del arg_names
    mode = (parser_mode or "vector").lower()
    if mode not in ("cube", "mix"):
        return AxisLengthStateResult(axis_states={})

    dot_result = analyze_dot_site_mnk(
        func_ast=func_ast,
        parser_mode=mode,
        module_ast=module_ast,
        entry_function_name=entry_function_name,
    )
    dot_sites = dot_result.dot_sites
    axis_symbols = []
    for site in dot_sites:
        if site.m is not None:
            axis_symbols.append(site.m)
        if site.n is not None:
            axis_symbols.append(site.n)
        if site.k is not None:
            axis_symbols.append(site.k)

    signature = extract_signature_info(func_ast)
    axis_states = resolve_axis_length_states(
        func_ast=func_ast,
        axis_symbols=axis_symbols,
        signature=signature,
        provided_args=provided_args,
    )
    return AxisLengthStateResult(axis_states=axis_states)
