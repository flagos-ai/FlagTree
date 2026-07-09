from __future__ import annotations

import hashlib
import inspect
import json
from pathlib import Path
from typing import Any, Optional, Union

# Protocol attribute checked by triton.runtime.jit.DependenciesFinder.
# Value may be a str or a zero-arg callable returning the current key fragment.
TLE_RAW_SOURCE_CACHE_KEY_ATTR = "__triton_tle_raw_source_cache_key__"

_DIALECT_KWARG_KEYS = (
    "name",
    "compiler",
    "target",
    "extern_func_name",
    "deferred",
)
_DIALECT_PATH_KEYS = (
    "file",
    "extern_file",
)


def _read_source(path: Union[str, Path]) -> str:
    return Path(path).read_text()


def _normalize_dialect_kwargs(dialect_kwargs: dict) -> dict:
    normalized: dict[str, Any] = {}
    for key in _DIALECT_KWARG_KEYS:
        if key not in dialect_kwargs:
            continue
        value = dialect_kwargs[key]
        if value is not None:
            normalized[key] = value
    for key in _DIALECT_PATH_KEYS:
        if key not in dialect_kwargs:
            continue
        value = dialect_kwargs[key]
        if value is not None:
            normalized[key] = str(Path(value).resolve())
    if "extra_source_files" in dialect_kwargs:
        extra = dialect_kwargs["extra_source_files"]
        if extra:
            normalized["extra_source_files"] = [str(Path(path).resolve()) for path in extra]
    return normalized


def _serialize_dialect_kwargs(dialect_kwargs: dict) -> str:
    return json.dumps(dialect_kwargs, sort_keys=True, separators=(",", ":"))


def _collect_source_file_paths(dialect_kwargs: dict) -> list[str]:
    paths: list[str] = []
    for key in _DIALECT_PATH_KEYS:
        path = dialect_kwargs.get(key)
        if path is not None:
            paths.append(str(path))
    for path in dialect_kwargs.get("extra_source_files", ()):
        paths.append(str(path))
    return sorted(set(paths))


def _read_inline_edsl_source(edsl: Any) -> Optional[str]:
    fn = getattr(edsl, "fn", None)
    if fn is None:
        return None
    return inspect.getsource(fn)


def compute_tle_raw_source_cache_key(
    dialect_kwargs: dict,
    *,
    edsl: Any | None = None,
) -> str:
    """Hash @dialect kwargs plus the contents of referenced source files."""
    normalized = _normalize_dialect_kwargs(dialect_kwargs)
    hasher = hashlib.sha256()
    hasher.update(_serialize_dialect_kwargs(normalized).encode())

    source_paths = _collect_source_file_paths(normalized)
    for path in source_paths:
        hasher.update(path.encode())
        hasher.update(_read_source(path).encode())

    if not source_paths and edsl is not None:
        inline_source = _read_inline_edsl_source(edsl)
        if inline_source is not None:
            hasher.update(b"inline_edsl_source")
            hasher.update(inline_source.encode())

    return hasher.hexdigest()


def bind_tle_raw_source_cache_key(edsl: Any, **dialect_kwargs) -> None:
    """Attach __triton_tle_raw_source_cache_key__ to a @dialect edsl object."""
    if getattr(edsl, TLE_RAW_SOURCE_CACHE_KEY_ATTR, None) is not None:
        return

    normalized = _normalize_dialect_kwargs(dialect_kwargs)
    if not normalized and getattr(edsl, "fn", None) is None:
        return

    def tle_raw_source_cache_key() -> str:
        return compute_tle_raw_source_cache_key(normalized, edsl=edsl)

    setattr(edsl, TLE_RAW_SOURCE_CACHE_KEY_ATTR, tle_raw_source_cache_key)
