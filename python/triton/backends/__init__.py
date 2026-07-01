import importlib
import os
import inspect
import sys
from dataclasses import dataclass
from typing import Type, TypeVar, Union
from types import ModuleType
from .driver import DriverBase
from .compiler import BaseBackend

if sys.version_info >= (3, 10):
    from importlib.metadata import entry_points
else:
    from importlib_metadata import entry_points

T = TypeVar("T", bound=Union[BaseBackend, DriverBase])


def _find_concrete_subclasses(module: ModuleType, base_class: Type[T]) -> Type[T]:
    ret: list[Type[T]] = []
    for attr_name in dir(module):
        attr = getattr(module, attr_name)
        if isinstance(attr, type) and issubclass(attr, base_class) and not inspect.isabstract(attr):
            ret.append(attr)
    if len(ret) == 0:
        raise RuntimeError(f"Found 0 concrete subclasses of {base_class} in {module}: {ret}")
    if len(ret) > 1:
        raise RuntimeError(f"Found >1 concrete subclasses of {base_class} in {module}: {ret}")
    return ret[0]


@dataclass(frozen=True)
class Backend:
    compiler: Type[BaseBackend]
    driver: Type[DriverBase]


def _discover_backends() -> dict[str, Backend]:
    backends = dict()
    active_backend = os.environ.get("FLAGTREE_BACKEND", "")
    if not active_backend:
        from triton._flagtree_backend import FLAGTREE_BACKEND  # type: ignore

        active_backend = FLAGTREE_BACKEND

    # Fast path: optionally skip entry point discovery (which can be slow) and
    # discover only in-tree backends under the `triton.backends` namespace.
    skip_entrypoints_env = os.environ.get("TRITON_BACKENDS_IN_TREE", "")

    if skip_entrypoints_env == "1":
        root = os.path.dirname(__file__)
        for name in os.listdir(root):
            if not os.path.isdir(os.path.join(root, name)):
                continue
            if name.startswith('__'):
                continue
            compiler = importlib.import_module(f"triton.backends.{name}.compiler")
            driver = importlib.import_module(f"triton.backends.{name}.driver")
            backends[name] = Backend(_find_concrete_subclasses(compiler, BaseBackend),
                                     _find_concrete_subclasses(driver, DriverBase))
        return backends

    # Default path: discover via entry points for out-of-tree/downstream plugins.
    for ep in entry_points().select(group="triton.backends"):
        # ==================== FLAGTREE XPU SYNC MARK ====================
        # Editable XPU development environments can also have a reference
        # `triton` package installed. Its entry points advertise amd/nvidia/xcn
        # backends that are not present in this FlagTree checkout, so restrict
        # discovery to the active FlagTree backend when one is selected.
        # ==================== FLAGTREE XPU SYNC MARK ====================
        if active_backend and ep.name != active_backend:
            continue
        compiler = importlib.import_module(f"{ep.value}.compiler")
        driver = importlib.import_module(f"{ep.value}.driver")
        backends[ep.name] = Backend(_find_concrete_subclasses(compiler, BaseBackend),  # type: ignore
                                    _find_concrete_subclasses(driver, DriverBase))  # type: ignore
    return backends


backends: dict[str, Backend] = _discover_backends()
