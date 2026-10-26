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
import os
from collections.abc import Callable, Iterable
from typing import Any, Union

from triton import _flagtree_backend

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
        candidates = {name for name in TLE_PRIMITIVES if name == qualname or name.endswith(f".{qualname}")}
        scoped = {name for name in candidates if "." in name and name.split(".", 1)[0] in module_parts}
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
    else:  # callable(primitive)
        return _callable_primitive_name(primitive)

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


def supported_primitives(backend_name: str) -> frozenset[str]:
    """Read and validate one backend's ``tle_supported.TLE_SUPPORTED_PRIMITIVES``."""
    module = _backend_config_module(backend_name)
    configured = getattr(module, "TLE_SUPPORTED_PRIMITIVES", []) if module is not None else []
    if not isinstance(configured, list):
        raise TypeError(f"backend {backend_name!r}: TLE_SUPPORTED_PRIMITIVES must be a list of strings, "
                        f"got {type(configured).__name__}")
    if not all(isinstance(item, str) for item in configured):
        raise TypeError(f"backend {backend_name!r}: TLE_SUPPORTED_PRIMITIVES must contain only strings")
    return frozenset(primitive_name(item) for item in configured)


def _routable_backend_names() -> tuple[str, ...]:
    """Backends a kernel can be routed to in this build and environment.

    TileIR builds leave ``FLAGTREE_BACKEND`` empty; TileIR is only routable
    when ``FLAGTREE_USE_TILEIR=1``, matching the TileIR router opt-in.
    """
    from triton.backends import backends

    backend_name = _flagtree_backend.FLAGTREE_BACKEND
    if backend_name and backend_name != "tileir":
        return (backend_name, )
    if "tileir" in backends and os.environ.get("FLAGTREE_USE_TILEIR", "0") == "1":
        return ("tileir", "nvidia")
    return ("nvidia", )


def _route_backend_name(names: frozenset[str]) -> str:
    """Mirror the TileIR router: TileIR only if every TLE name is in its whitelist."""
    *preferred, fallback = _routable_backend_names()
    for backend_name in preferred:
        if names <= supported_primitives(backend_name):
            return backend_name
    return fallback


def get_supported_primitives() -> frozenset[str]:
    """Union of ``tle_supported`` whitelists of the currently routable backends."""
    return frozenset().union(*(supported_primitives(name) for name in _routable_backend_names()))


_PrimitiveArg = Union[str, Callable[..., Any]]


def _unsupported_primitives(primitives: _PrimitiveArg | Iterable[_PrimitiveArg]) -> tuple[str, list[str]]:
    """Return the backend these primitives route to and the ones it rejects."""
    if isinstance(primitives, str) or callable(primitives):
        primitives = [primitives]
    names = [name for name in map(primitive_name, primitives) if name in TLE_PRIMITIVES]
    backend_name = _route_backend_name(frozenset(names))
    supported = supported_primitives(backend_name)
    return backend_name, [name for name in names if name not in supported]


# For pytest skip or gems implementation routing
def is_tle_supported(primitives: _PrimitiveArg | Iterable[_PrimitiveArg]) -> bool:
    """Whether a kernel using all of ``primitives`` is supported by the backend it routes to."""
    return not _unsupported_primitives(primitives)[1]


# For assertion in gems implementation
def require_tle(primitives: _PrimitiveArg | Iterable[_PrimitiveArg]) -> None:
    backend_name, unsupported = _unsupported_primitives(primitives)
    if unsupported:
        raise TLEUnsupportedPrimitiveError(f"backend {backend_name!r} does not support TLE primitive(s) "
                                           f"{', '.join(repr(name) for name in unsupported)}")


# For the compiler guard, which must check the backend actually generating code
# rather than every backend this build could route to.
def _require_tle_for_backend(backend_name: str, primitive: str | Callable[..., Any]) -> None:
    name = primitive_name(primitive)
    if name in TLE_PRIMITIVES and name not in supported_primitives(backend_name):
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
    "is_tle_supported",
    "primitive_name",
    "require_tle",
]

if raw is not None:
    __all__.append("raw")
