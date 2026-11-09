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
from typing import Set

_CUBE_MIX_TUNABLE_KEYWORDS = (
    "BLOCK",
    "TILE",
    "CHUNK",
    "SPLIT",
    "GROUP",
    "WARP",
    "STAGE",
)


@dataclass(frozen=True)
class LaunchSemantics:
    launch_related_names: Set[str]


def analyze_launch_semantics(func_ast: ast.AST) -> LaunchSemantics:
    launch_related_names = set()
    for node in ast.walk(func_ast):
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
            launch_related_names.add(node.id)
    return LaunchSemantics(launch_related_names=launch_related_names)


def looks_like_cube_mix_tunable(name: str) -> bool:
    upper_name = name.upper()
    if upper_name in {"M", "N", "K"}:
        return True
    return any(keyword in upper_name for keyword in _CUBE_MIX_TUNABLE_KEYWORDS)
