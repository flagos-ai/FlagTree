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
