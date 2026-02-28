from .cuda import CUDAJITFunction
from .mlir import MLIRJITFunction

registry = {"cuda": CUDAJITFunction, "mlir": MLIRJITFunction}


def dialect(
    *,
    name: str,
    **kwargs,
):

    def decorator(fn):
        edsl = registry[name](fn, **kwargs)
        return edsl

    return decorator
