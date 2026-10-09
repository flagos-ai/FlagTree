registry = {}

try:
    from .cuda import CUDAJITFunction
    registry["cuda"] = CUDAJITFunction
except Exception:
    pass

try:
    from .mlir import MLIRJITFunction
    registry["mlir"] = MLIRJITFunction
except Exception:
    pass

try:
    from .cann import CANNJITFunction
    registry["cann"] = CANNJITFunction
except Exception:
    pass


def dialect(
    *,
    name: str,
    **kwargs,
):

    if name not in registry:
        raise ValueError(f"unknown or unavailable dialect {name!r}; available: {sorted(registry)}")

    def decorator(fn):
        edsl = registry[name](fn, **kwargs)
        return edsl

    return decorator
