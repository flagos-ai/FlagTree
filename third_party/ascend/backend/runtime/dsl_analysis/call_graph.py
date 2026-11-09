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
from typing import Dict, List, Optional, Sequence, Union

FunctionLike = Union[ast.FunctionDef, ast.AsyncFunctionDef]


@dataclass(frozen=True)
class FunctionMeta:
    name: str
    node: FunctionLike
    param_names: Sequence[str]


@dataclass(frozen=True)
class CallSite:
    caller: str
    callee: str
    args: Sequence[Optional[ast.AST]]
    lineno: int


def _extract_param_names(function_node: FunctionLike) -> List[str]:
    args_node = function_node.args
    ordered_args = list(args_node.posonlyargs) + list(args_node.args) + list(args_node.kwonlyargs)
    return [arg.arg for arg in ordered_args if isinstance(arg, ast.arg)]


def build_function_index(tree: ast.AST) -> Dict[str, FunctionMeta]:
    functions = {}
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if node.name in functions:
            continue
        functions[node.name] = FunctionMeta(
            name=node.name,
            node=node,
            param_names=_extract_param_names(node),
        )
    return functions


def build_call_sites(functions: Dict[str, FunctionMeta]) -> List[CallSite]:
    call_sites = []
    for caller_meta in functions.values():
        for node in ast.walk(caller_meta.node):
            if not isinstance(node, ast.Call):
                continue
            if not isinstance(node.func, ast.Name):
                continue
            callee_name = node.func.id
            if callee_name not in functions:
                continue
            callee_meta = functions[callee_name]
            bound_args: List[Optional[ast.AST]] = [None] * len(callee_meta.param_names)
            for idx, arg_node in enumerate(node.args):
                if idx >= len(bound_args):
                    break
                bound_args[idx] = arg_node
            keyword_map = {kw.arg: kw.value for kw in node.keywords if kw.arg is not None}
            for idx, param_name in enumerate(callee_meta.param_names):
                if bound_args[idx] is not None:
                    continue
                if param_name not in keyword_map:
                    continue
                bound_args[idx] = keyword_map[param_name]
            call_sites.append(
                CallSite(
                    caller=caller_meta.name,
                    callee=callee_name,
                    args=bound_args,
                    lineno=getattr(node, "lineno", -1),
                ))
    call_sites.sort(key=lambda site: (site.caller, site.lineno, site.callee))
    return call_sites


def find_function_name_by_node(functions: Dict[str, FunctionMeta], node: ast.AST) -> Optional[str]:
    for name, meta in functions.items():
        if meta.node is node:
            return name
    return None
