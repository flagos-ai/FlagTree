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
from typing import Any, Dict, Optional

SUPPORTED_KERNEL_TYPES = {"vector", "cube", "mix", "auto"}
_MIX_OPERATOR_ATTRS = {"dot"}


def _get_hint_kernel_type(hints: Optional[Dict[str, Any]]) -> str:
    if not hints:
        return "auto"

    kernel_type = hints.get("kernel_type", "auto")
    if isinstance(kernel_type, str):
        kernel_type = kernel_type.strip().lower()

    if kernel_type not in SUPPORTED_KERNEL_TYPES:
        raise ValueError(f"Unsupported kernel_type '{kernel_type}', expected one of: "
                         f"{sorted(SUPPORTED_KERNEL_TYPES)}")
    return kernel_type


def _base_name(expr: ast.AST) -> Optional[str]:
    node = expr
    while isinstance(node, ast.Attribute):
        node = node.value
    if isinstance(node, ast.Name):
        return node.id
    return None


def classify_kernel_type_from_dsl(func_ast: Optional[ast.AST]) -> str:
    if func_ast is None:
        return "vector"
    try:
        for node in ast.walk(func_ast):
            if not isinstance(node, ast.Call):
                continue
            if not isinstance(node.func, ast.Attribute):
                continue
            if node.func.attr not in _MIX_OPERATOR_ATTRS:
                continue
            if _base_name(node.func) == "tl":
                return "mix"
    except Exception:
        return "vector"
    return "vector"


def resolve_kernel_type(
    hints: Optional[Dict[str, Any]],
    func_ast: Optional[ast.AST] = None,
) -> str:
    kernel_type = _get_hint_kernel_type(hints)
    if kernel_type != "auto":
        return kernel_type
    return classify_kernel_type_from_dsl(func_ast)
