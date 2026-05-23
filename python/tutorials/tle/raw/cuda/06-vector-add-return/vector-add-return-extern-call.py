import triton.language as tl
import triton.language.core as core

@core.extern
def vector_add_return(C, A, B, N, _semantic=None):
    return core.extern_call(
        "",
        "",
        [C, A, B, N],
        {
            (
                core.pointer_type(core.dtype("fp32")),
                core.pointer_type(core.dtype("fp32")),
                core.pointer_type(core.dtype("fp32")),
                core.dtype("int32"),
            ) : ("vector_add_float_return", (core.dtype("uint64"))),
        },
        is_pure=False,
        _semantic=_semantic,
    )