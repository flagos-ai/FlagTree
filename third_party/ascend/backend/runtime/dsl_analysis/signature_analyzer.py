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
from typing import List, Set, Union

from .schema import ParameterSpec, SignatureInfo


def _is_tl_constexpr(annotation: ast.AST) -> bool:
    return (isinstance(annotation, ast.Attribute) and isinstance(annotation.value, ast.Name)
            and annotation.value.id == "tl" and annotation.attr == "constexpr")


def _find_function_node(func_ast: ast.AST) -> Union[ast.FunctionDef, ast.AsyncFunctionDef]:
    if isinstance(func_ast, (ast.FunctionDef, ast.AsyncFunctionDef)):
        return func_ast

    for node in ast.walk(func_ast):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            return node

    raise ValueError("Unable to locate function definition in DSL AST.")


def _extract_default_flags(function_node: Union[ast.FunctionDef, ast.AsyncFunctionDef]) -> dict:
    args_node = function_node.args
    positional_args = list(args_node.posonlyargs) + list(args_node.args)
    kwonly_args = list(args_node.kwonlyargs)

    default_flags = {}
    positional_defaults = list(args_node.defaults or [])
    positional_default_count = len(positional_defaults)
    positional_arg_count = len(positional_args)
    positional_default_start = positional_arg_count - positional_default_count
    for idx, arg in enumerate(positional_args):
        default_flags[arg.arg] = idx >= positional_default_start

    kw_defaults = list(args_node.kw_defaults or [])
    for idx, arg in enumerate(kwonly_args):
        has_default = idx < len(kw_defaults) and kw_defaults[idx] is not None
        default_flags[arg.arg] = has_default
    return default_flags


def extract_signature_info(func_ast: ast.AST) -> SignatureInfo:
    function_node = _find_function_node(func_ast)
    args_node = function_node.args
    if not isinstance(args_node, ast.arguments):
        return SignatureInfo(parameters=[])

    ordered_args = list(args_node.posonlyargs) + list(args_node.args) + list(args_node.kwonlyargs)
    default_flags = _extract_default_flags(function_node)
    parameters = []
    for arg in ordered_args:
        if not isinstance(arg, ast.arg):
            continue
        parameters.append(
            ParameterSpec(
                name=arg.arg,
                is_constexpr=_is_tl_constexpr(arg.annotation),
                has_default=bool(default_flags.get(arg.arg, False)),
            ))
    return SignatureInfo(parameters=parameters)


def extract_constexpr_names(func_ast: ast.AST) -> List[str]:
    return extract_signature_info(func_ast).constexpr_names()


def extract_constexpr_name_set(func_ast: ast.AST) -> Set[str]:
    return set(extract_constexpr_names(func_ast))
