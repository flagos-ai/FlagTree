"""Process-wide store for `tle.raw` payloads whose compilation is deferred.

At trace time `tle.raw.call` does not know which backend/architecture will
compile the kernel, so a deferred
payload only records a content-addressed id in the IR
(`triton_xpu.raw_source_id`). The backend later compiles every pending payload
for the arch it is building for and hands the results to
`tritonxpu-materialize-deferred-raw`.

Entries are keyed by a hash of (dialect, callee, source), so two kernels sharing
a payload register once and the compiled LLVM IR is reused.
"""

from __future__ import annotations

import hashlib
from typing import Any, Dict

__all__ = [
    "register_source",
    "get_source",
    "list_pending_sources",
    "clear_pending_sources",
]

_PENDING_RAW_SOURCES: Dict[str, dict] = {}


def source_id(*, dialect: str, callee: str, source: str) -> str:
    payload = f"{dialect}\0{callee}\0{source}".encode()
    return hashlib.sha256(payload).hexdigest()


def register_source(*, dialect: str, callee: str, source: str, handle: Any = None, **extra) -> str:
    """Record a payload for later compilation and return its id."""
    sid = source_id(dialect=dialect, callee=callee, source=source)
    entry = {"dialect": dialect, "callee": callee, "source": source, "handle": handle}
    entry.update(extra)
    _PENDING_RAW_SOURCES[sid] = entry
    return sid


def get_source(sid: str) -> dict | None:
    return _PENDING_RAW_SOURCES.get(sid)


def list_pending_sources() -> Dict[str, dict]:
    return dict(_PENDING_RAW_SOURCES)


def clear_pending_sources() -> None:
    _PENDING_RAW_SOURCES.clear()
