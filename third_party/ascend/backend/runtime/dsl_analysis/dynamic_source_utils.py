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
from typing import Optional, Set


def extract_expr_names_from_text(expr_text: Optional[str]) -> Set[str]:
    if not expr_text:
        return set()
    try:
        parsed = ast.parse(expr_text, mode="eval")
    except Exception:
        try:
            parsed = ast.parse(expr_text, mode="exec")
        except Exception:
            return set()
    return {node.id for node in ast.walk(parsed) if isinstance(node, ast.Name)}


def resolve_dynamic_source(
    axis_symbol: Optional[str],
    length_expr: Optional[str],
    load_derived_symbols: Set[str],
) -> str:
    if not load_derived_symbols:
        return "none"

    names = extract_expr_names_from_text(length_expr)
    if not names and axis_symbol is not None:
        names = extract_expr_names_from_text(axis_symbol)
        if not names and axis_symbol.isidentifier():
            names = {axis_symbol}

    if names.intersection(load_derived_symbols):
        return "dynamic_from_load"
    return "none"
