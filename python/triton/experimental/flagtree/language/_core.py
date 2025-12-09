from triton.language.core import builtin


@builtin
def call(func, operands, _semantic=None):
    return _semantic.call(func, operands)
