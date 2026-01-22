from .mlir import EdslMLIRJITFunction
from typing import List

registry = {"mlir": EdslMLIRJITFunction}


def dialect(*, name: str, pipeline: List[str] = [
    "convert-scf-to-cf", "finalize-memref-to-llvm", "convert-arith-to-llvm", "convert-cf-to-llvm",
    "convert-func-to-llvm", "convert-index-to-llvm", "convert-nvvm-to-llvm", "cse"
]):

    def decorator(fn):
        edsl = registry[name](fn, pipeline=pipeline)
        return edsl

    return decorator
