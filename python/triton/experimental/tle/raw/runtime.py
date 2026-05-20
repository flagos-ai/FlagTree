from .cuda import CUDAJITFunction
from .mlir import MLIRJITFunction

registry = {"cuda": CUDAJITFunction, "mlir": MLIRJITFunction}

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
        edsl = registry[name](fn, **kwargs)
        return edsl

    return decorator
