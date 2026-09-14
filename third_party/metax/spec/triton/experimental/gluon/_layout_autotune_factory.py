from __future__ import annotations

import threading
from typing import Any, Callable


_DEFAULT_PLUGIN = "parallel-compile"
_FACTORIES: dict[str, Callable[..., Any]] = {}
_FACTORY_LOCK = threading.RLock()


def register_layout_autotuner_factory(
    name: str, factory: Callable[..., Any]
) -> None:
    """Register one explicit runtime layout-autotuner implementation."""

    if not name:
        raise ValueError("layout autotuner factory name must be non-empty")
    if not callable(factory):
        raise TypeError("layout autotuner factory must be callable")
    with _FACTORY_LOCK:
        previous = _FACTORIES.get(name)
        if previous is not None and previous is not factory:
            raise RuntimeError(f"layout autotuner factory {name!r} is already registered")
        _FACTORIES[name] = factory


def create_layout_autotuner(fn: Any, *, plugin: str = _DEFAULT_PLUGIN, **kwargs):
    """Create the selected plugin without coupling the JIT runtime to its class."""

    if plugin == _DEFAULT_PLUGIN:
        # Importing the implementation performs its local registration. Keep
        # this lazy so the front door owns only the factory contract.
        from . import _parallel_compile_autotuner  # noqa: F401

    with _FACTORY_LOCK:
        factory = _FACTORIES.get(plugin)
    if factory is None:
        raise RuntimeError(f"unknown Gluon layout autotuner plugin {plugin!r}")
    return factory(fn=fn, **kwargs)


__all__ = [
    "create_layout_autotuner",
    "register_layout_autotuner_factory",
]
