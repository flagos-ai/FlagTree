from .cache_key import bind_tle_raw_source_cache_key

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
