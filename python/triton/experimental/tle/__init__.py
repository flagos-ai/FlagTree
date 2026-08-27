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

# flagtree tle
from __future__ import annotations

import importlib
from collections.abc import Callable
from typing import Any

_TLE_LANGUAGE_PREFIX = "triton.experimental.tle.language."
_TLE_PRIMITIVE_PREFIXES = (
    "triton.experimental.tle.",
    "tle.",
    "language.",
)


class TLEUnsupportedPrimitiveError(RuntimeError):
    """Raised when a backend does not declare support for a TLE primitive."""


def _callable_primitive_name(primitive: Callable[..., Any]) -> str:
    module = getattr(primitive, "__module__", "")
    qualname = getattr(primitive, "__qualname__", "")
    if module.startswith(_TLE_LANGUAGE_PREFIX):
        module_parts = set(module[len(_TLE_LANGUAGE_PREFIX):].split("."))
        candidates = {
            name for name in TLE_PRIMITIVES
            if name == qualname or name.endswith(f".{qualname}")
        }
        scoped = {
            name for name in candidates
            if "." in name and name.split(".", 1)[0] in module_parts
        }
        if len(scoped) == 1:
            return scoped.pop()
        if qualname in candidates:
            return qualname
        if len(candidates) == 1:
            return candidates.pop()
        raise ValueError(f"{primitive!r} is not listed in TLE_PRIMITIVES")

    name = getattr(primitive, "__name__", "")
    if name not in TLE_PRIMITIVES:
        raise ValueError(f"{primitive!r} is not listed in TLE_PRIMITIVES")

    import triton.language as tl

    language_extensions = getattr(tl, "ext", None)
    try:
        extension = getattr(language_extensions, name) if language_extensions is not None else None
    except (AttributeError, RuntimeError):
        extension = None
    if extension is primitive:
        return name
    raise ValueError(f"{primitive!r} is not a registered backend language extension")


def primitive_name(primitive: str | Callable[..., Any]) -> str:
    """Return the stable public whitelist name for a TLE primitive."""
    if isinstance(primitive, str):
        name = primitive
        for prefix in _TLE_PRIMITIVE_PREFIXES:
            if name.startswith(prefix):
                name = name[len(prefix):]
                break
        if name.startswith("language."):
            name = name[len("language."):]
    elif callable(primitive):
        return _callable_primitive_name(primitive)
    else:
        raise TypeError(f"TLE primitive must be a string or callable, got {type(primitive).__name__}")

    aliases = (
        ("gpu.core.", "gpu."),
        ("gpu.types.", "gpu."),
        ("raw.core.", "raw."),
        ("core.", ""),
        ("pipe.", ""),
        ("distributed.", ""),
    )
    for prefix, replacement in aliases:
        if name.startswith(prefix):
            name = replacement + name[len(prefix):]
            break
    if not name or name.startswith(".") or name.endswith("."):
        raise ValueError(f"invalid TLE primitive name: {primitive!r}")
    return name


def _backend_config_module(backend_name: str):
    from triton.backends import backends

    backend = backends.get(backend_name)
    if backend is not None:
        package = backend.compiler.__module__.rsplit(".", 1)[0]
    else:
        package = f"triton.backends.{backend_name}"
    config_module = f"{package}.tle_supported"
    try:
        return importlib.import_module(config_module)
    except ModuleNotFoundError as exc:
        if exc.name != config_module:
            raise
        return None


def get_supported_primitives(backend_name: str) -> frozenset[str]:
    """Read and validate a backend's strict TLE primitive whitelist."""
    module = _backend_config_module(backend_name)
    configured = getattr(module, "TLE_SUPPORTED_PRIMITIVES", []) if module is not None else []
    if not isinstance(configured, list):
        raise TypeError(f"backend {backend_name!r} TLE_SUPPORTED_PRIMITIVES must be a list of strings, "
                        f"got {type(configured).__name__}")
    if not all(isinstance(item, str) for item in configured):
        raise TypeError(f"backend {backend_name!r} TLE_SUPPORTED_PRIMITIVES must contain only strings")
    return frozenset(primitive_name(item) for item in configured)


def is_primitive_supported(backend_name: str, primitive: str | Callable[..., Any]) -> bool:
    name = primitive_name(primitive)
    return name not in TLE_PRIMITIVES or name in get_supported_primitives(backend_name)


def require_tle(backend_name: str, primitive: str | Callable[..., Any]) -> None:
    name = primitive_name(primitive)
    if name in TLE_PRIMITIVES and name not in get_supported_primitives(backend_name):
        raise TLEUnsupportedPrimitiveError(f"backend {backend_name!r} does not support TLE primitive {name!r}")


from . import language
from .language.primitives import TLE_PRIMITIVES

try:
    from . import raw
except ModuleNotFoundError:
    raw = None

__all__ = [
    "language",
    "TLE_PRIMITIVES",
    "TLEUnsupportedPrimitiveError",
    "get_supported_primitives",
    "is_primitive_supported",
    "primitive_name",
    "require_tle",
]

if raw is not None:
    __all__.append("raw")
