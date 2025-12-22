from .mlir import EdslMLIRJITFunction
from typing import Any, Dict, Final, List, Optional
registry = {"mlir": EdslMLIRJITFunction}


def dialect(*, name: str, pipeline: Optional[List[str]] | None = None):

    def decorator(fn):
        edsl = registry[name](fn, pipeline=pipeline)
        setattr(edsl, "__triton_builtin__", True)
        return edsl

    return decorator
