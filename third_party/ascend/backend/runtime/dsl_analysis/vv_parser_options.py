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

from typing import Any, Mapping, Optional

_VV_PARSER_V2_MODES = ("observe", "assist", "authoritative")


def resolve_vv_parser_v2_mode(hints: Optional[Mapping[str, Any]]) -> str:
    """
    Resolve vv parser v2 mode from hints.

    Priority:
    1. hints["vv_parser_v2_mode"] in {"observe", "assist", "authoritative"}
    2. hints missing -> "assist"
    3. invalid mode -> "off"
    """
    if not hints:
        return "assist"

    raw_mode = hints.get("vv_parser_v2_mode")
    if isinstance(raw_mode, str):
        mode = raw_mode.strip().lower()
        if mode in _VV_PARSER_V2_MODES:
            return mode

    if "vv_parser_v2_mode" not in hints:
        return "assist"

    return "off"


def resolve_vv_parser_v2_enabled(hints: Optional[Mapping[str, Any]]) -> bool:
    return resolve_vv_parser_v2_mode(hints) != "off"
