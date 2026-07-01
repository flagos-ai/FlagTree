from .cuda import CUDAJITFunction

registry = {"cuda": CUDAJITFunction}

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
        if name == "mlir" and name not in registry:
            from .mlir import MLIRJITFunction
            registry[name] = MLIRJITFunction
        edsl = registry[name](fn, **kwargs)
        return edsl

    return decorator
