# Copyright 2025-     FlagOS Contributors
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from __future__ import annotations

import hashlib
from typing import Dict, Optional

_PENDING_RAW_SOURCES: Dict[str, dict] = {}


def register_source(
    *,
    region_dialect: str,
    extern_func_name: str | None,
    source: str,
    hint: str = "",
    extra: Optional[dict] = None,
) -> str:
    payload = f"{region_dialect}\0{extern_func_name or ''}\0{source}".encode()
    source_id = hashlib.sha256(payload).hexdigest()
    entry = {
        "region_dialect": region_dialect,
        "extern_func_name": extern_func_name,
        "source": source,
        "hint": hint,
    }
    if extra:
        entry.update(extra)
    _PENDING_RAW_SOURCES[source_id] = entry
    return source_id


def get_source(source_id: str) -> dict | None:
    return _PENDING_RAW_SOURCES.get(source_id)


def list_pending_sources() -> Dict[str, dict]:
    return dict(_PENDING_RAW_SOURCES)


def clear_pending_sources() -> None:
    _PENDING_RAW_SOURCES.clear()
