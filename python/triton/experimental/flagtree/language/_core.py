from triton.language.core import builtin


@builtin
def call(func, outputs, inputs, _semantic=None):
    return _semantic.call(func, outputs, inputs)
