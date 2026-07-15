"""Backend capability registry for TLE.

Backends register the TLE features they do **not** support so the Python
frontend can raise a clear error before MLIR-level legalization fails.

The active backend is resolved via the compile options carried on the IR
builder (``builder.options.backend_name``), which the code generator wires
up in ``triton/compiler/code_generator.py``.
"""

from typing import Iterable, Optional

# backend_name -> set of feature identifiers that backend does NOT support
_UNSUPPORTED: dict = {}


def register_unsupported(backend_name: str, features: Iterable[str]) -> None:
    """Record TLE features that ``backend_name`` does not support.

    Subsequent calls overwrite the previous set for the same backend.
    Backends should call this from their package ``__init__.py`` so the set
    is populated as soon as the backend is discovered.
    """
    _UNSUPPORTED[backend_name] = set(features)


def _active_backend(*, semantic=None, builder=None, options=None) -> Optional[str]:
    if options is None:
        if builder is None and semantic is not None:
            builder = getattr(semantic, "builder", None)
        if builder is not None:
            options = getattr(builder, "options", None)
    if options is None:
        return None
    return getattr(options, "backend_name", None)


def check_supported(feature: str, *, semantic=None, builder=None, options=None) -> None:
    """Raise ``NotImplementedError`` if ``feature`` is unsupported by the
    currently-active backend. Silently returns when no backend or no
    registration is found (so non-PPU backends and bare-MLIR tests keep
    working unchanged)."""
    backend = _active_backend(semantic=semantic, builder=builder, options=options)
    if backend is None:
        return
    unsupported = _UNSUPPORTED.get(backend)
    if unsupported is not None and feature in unsupported:
        raise NotImplementedError(
            f"TLE feature '{feature}' is not supported on the '{backend}' backend"
        )
