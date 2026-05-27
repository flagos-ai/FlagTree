import triton.language as tl
import triton.language.core as core

@core.extern
def vector_add(C, A, B, N, _semantic=None):
    return core.extern_call(
        "",
        "",
        [C, A, B, N],
        {
            (
                core.pointer_type(core.dtype("int32")),
                core.pointer_type(core.dtype("int32")),
                core.pointer_type(core.dtype("int32")),
                core.dtype("int32"),
            ) : ("vector_add_int", ()),
            (
                core.pointer_type(core.dtype("fp32")),
                core.pointer_type(core.dtype("fp32")),
                core.pointer_type(core.dtype("fp32")),
                core.dtype("int32"),
            ) : ("vector_add_float", ()),
        },
        is_pure=False,
        _semantic=_semantic,
    )