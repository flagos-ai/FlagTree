from .mlir import EdslMLIRJITFunction

registry = {"mlir": EdslMLIRJITFunction}


def dialect(*, name: str):

    def decorator(fn):
        edsl = registry[name](fn)
        setattr(edsl, "__triton_builtin__", True)
        return edsl

    return decorator
