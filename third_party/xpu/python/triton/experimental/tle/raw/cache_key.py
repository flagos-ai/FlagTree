"""Make `tle.raw` payload sources part of the kernel's compile cache key.

A payload handle produced by `@tle.raw.dialect(...)` is referenced from the
kernel body as a plain global, so `DependenciesFinder` would otherwise ignore
its `.xpu` source: editing the payload would silently reuse a stale binary.
Handles therefore expose `__triton_tle_raw_source_cache_key__`, which
`DependenciesFinder.record_reference` folds into the kernel hash.

The attribute is a callable so the payload file is read when the hash is
computed rather than when the handle is created. Note that
`JITFunction.cache_key` memoizes its result, so a payload edited *within* a
running process is not picked up -- the invalidation this provides is across
processes, which is what the on-disk compile cache needs.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Callable, Optional, Sequence

# Protocol attribute read by triton.runtime.jit.DependenciesFinder.
TLE_RAW_SOURCE_CACHE_KEY_ATTR = "__triton_tle_raw_source_cache_key__"

__all__ = [
    "TLE_RAW_SOURCE_CACHE_KEY_ATTR",
    "compute_source_cache_key",
    "bind_source_cache_key",
]


def compute_source_cache_key(
        *,
        dialect: str,
        callee: str,
        source: str,
        arch: Any = None,
        flags: Sequence[str] = (),
        file: Optional[Path] = None,
) -> str:
    hasher = hashlib.sha256()
    for part in (dialect, callee, str(arch), str(list(flags)), str(file or ""), source):
        hasher.update(part.encode("utf-8"))
        hasher.update(b"\0")
    return hasher.hexdigest()


def bind_source_cache_key(handle: Any, key_fn: Callable[[], str]) -> None:
    setattr(handle, TLE_RAW_SOURCE_CACHE_KEY_ATTR, key_fn)
