import sys

def spec_libdevice_math_func():
    import triton.language as language
    spec_func = language.extra.ascend.libdevice.relu
    setattr(sys.modules["triton.language"], spec_func.__name__, spec_func)
    setattr(sys.modules["triton.language.math"], spec_func.__name__, spec_func)
