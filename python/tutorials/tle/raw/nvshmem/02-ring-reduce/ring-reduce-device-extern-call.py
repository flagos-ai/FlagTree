import triton.language as tl
import triton.language.core as core

@core.extern
def ring_reduce(dst, src, nreduce, signal, chunk_size, _semantic=None):
    return core.extern_call(
        "",
        "",
        [dst, 
         src, 
         tl.cast(nreduce, tl.int32, _semantic=_semantic), 
         signal, 
         tl.cast(chunk_size, tl.int32, _semantic=_semantic)],
        {
            (
                core.pointer_type(core.dtype("int32")), 
                core.pointer_type(core.dtype("int32")), 
                core.dtype("int32"), 
                core.pointer_type(core.dtype("int64")),
                core.dtype("int32"), 
            ) : ("ring_reduce", ()),
        },
        is_pure=False,
        _semantic=_semantic,
    )
