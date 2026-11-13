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

from typing import List

from .launch_analyzer import looks_like_cube_mix_tunable
from .schema import MissingTunableContext


def _select_missing_constexpr(context: MissingTunableContext) -> List[str]:
    missing = set(context.missing_arg_names())
    fixed = context.fixed_tunable_names()
    candidates = []
    for name in context.signature.constexpr_names():
        if name not in missing:
            continue
        if context.signature.has_default(name):
            continue
        if name in fixed:
            continue
        candidates.append(name)
    return candidates


class VectorPolicy:

    def select_missing_tunable(self, context: MissingTunableContext) -> List[str]:
        return _select_missing_constexpr(context)


class CubeMixPolicy:

    def select_missing_tunable(self, context: MissingTunableContext) -> List[str]:
        candidates = _select_missing_constexpr(context)
        launch_related_names = set()
        if context.launch_info is not None:
            launch_related_names = set(getattr(context.launch_info, "launch_related_names", set()))

        filtered = []
        for name in candidates:
            if looks_like_cube_mix_tunable(name):
                filtered.append(name)
                continue
            if name in launch_related_names:
                filtered.append(name)
        return filtered
