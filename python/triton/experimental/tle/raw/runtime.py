from __future__ import annotations

from typing import Any

from .cache_key import bind_tle_raw_source_cache_key


class RawJITFunction:
    """Shared @dialect state and default LLVM region materialization."""

    def __init__(self, fn: Any, **kwargs) -> None:
        self.fn = fn
        self.extern_func_name = kwargs.get("extern_func_name", "")
        self.deferred = kwargs.get("deferred", False)
        self.library = kwargs.get("library", "") or ""
        self.__triton_builtin__ = True

    def create_region_by_llvm(self, builder, llvm: str, handles, alias_indices, hint: str = "",
                              extern_func_name: str = ""):
        return builder.create_tle_raw_region_by_llvm_func(
            llvm,
            self.region_dialect,
            self.arg_dialect,
            handles,
            alias_indices,
            hint,
            extern_func_name,
        )


registry = {}

try:
    from .cuda import CUDAJITFunction
    registry["cuda"] = CUDAJITFunction
except ImportError:
    pass

try:
    from .mlir import MLIRJITFunction
    registry["mlir"] = MLIRJITFunction
except ModuleNotFoundError as exc:
    if exc.name != "mlir":
        raise

try:
    from .tops import TOPSJITFunction, TOPSMLIRJITFunction
    registry["tops"] = TOPSJITFunction
    registry["tops_mlir"] = TOPSMLIRJITFunction
except ImportError:
    pass


def dialect(
    *,
    name: str,
    **kwargs,
):

    def decorator(fn):
        if name not in registry:
            if name == "cuda":
                from .cuda import CUDAJITFunction
                registry[name] = CUDAJITFunction
            elif name == "mlir":
                from .mlir import MLIRJITFunction
                registry[name] = MLIRJITFunction
        edsl = registry[name](fn, **kwargs)
        bind_tle_raw_source_cache_key(edsl, name=name, **kwargs)
        return edsl

    return decorator
