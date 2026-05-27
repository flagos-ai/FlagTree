from triton.language import core
import triton.language as tl


def _pointer_type_hash(self):
    return hash((self.name, self.element_ty, "tt_ptr"))


def patch_hash_method_for_pointer_type():
    elem_dtype_list = tl.core.dtype.SINT_TYPES + tl.core.dtype.UINT_TYPES + tl.core.dtype.FP_TYPES + tl.core.dtype.OTHER_TYPES
    for elem_dtype in elem_dtype_list:
        ptr_ty = type(tl.core.pointer_type(tl.core.dtype(elem_dtype)))
        ptr_ty.__hash__ = _pointer_type_hash


patch_hash_method_for_pointer_type()

# 01-simple-shift
@core.extern
def simple_shift(dst, _semantic=None):
    return core.extern_call(
        "",  # libname
        "",  # libpath
        [dst],  # args
        {
            (
                core.pointer_type(core.dtype("int32")),  # arg_type_symbol_dict
            ): ("simple_shift", ()),  # void return type
        },
        is_pure=False,
        _semantic=_semantic,
    )

# 02-ring-reduce
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
    
# 03-ring-bcast
@core.extern
def ring_bcast(data, nelem, root, psync, _semantic=None):
    return core.extern_call(
        "",
        "",
        [data, nelem, root, psync],
        {
            (
                core.pointer_type(core.dtype("int32")),  
                core.dtype("int32"), 
                core.dtype("int32"), 
                core.pointer_type(core.dtype("uint64")),
            ) : ("ring_bcast", ()),
        },
        is_pure=False,
        _semantic=_semantic,
    )

# 04-on-stream
@core.extern
def accumulate(input, partial_sum, _semantic=None):
    return core.extern_call(
        "",  # libname
        "",  # libpath
        [input, partial_sum],
        {
            (
                core.pointer_type(core.dtype("int32")), 
                core.pointer_type(core.dtype("int32")),
            ): ("accumulate", ()),
        },
        is_pure=False,
        _semantic=_semantic,
    )

@core.extern
def correct_accumulate(input, partial_sum, full_sum, _semantic=None):
    return core.extern_call(
        "",  # libname
        "",  # libpath
        [input, partial_sum, full_sum],
        {
            (
                core.pointer_type(core.dtype("int32")), 
                core.pointer_type(core.dtype("int32")),
                core.pointer_type(core.dtype("int32")),
            ): ("correct_accumulate", ()),
        },
        is_pure=False,
        _semantic=_semantic,
    )
    
# 05-put-block
@core.extern
def set_and_shift(send_data, recv_data, num_elems, mype, npes, _semantic=None):
    return core.extern_call(
        "",  # libname
        "",  # libpath
        [send_data, recv_data, num_elems, mype, npes],
        {
            (
                core.pointer_type(core.dtype("fp32")), 
                core.pointer_type(core.dtype("fp32")),
                core.dtype("int32"),
                core.dtype("int32"),
                core.dtype("int32"),
            ): ("set_and_shift", ()),
        },
        is_pure=False,
        _semantic=_semantic,
    )

# 00-gemm-allreduce
@core.extern
def tiled_gemm(C, A, B, m, n, k, _semantic=None):
    return core.extern_call(
        "",
        "",
        [
            C, A, B, 
            m, n, k
        ],
        {
            (
                core.pointer_type(core.dtype("fp32")),
                core.pointer_type(core.dtype("fp32")),
                core.pointer_type(core.dtype("fp32")),
                core.dtype("int32"),
                core.dtype("int32"), 
                core.dtype("int32"), 
            ) : ("tiled_gemm", ()),
        },
        is_pure=False,
        _semantic=_semantic,
    )

@core.extern
def per_token_group_quant_8bit(x_q_ptr, x_s_ptr, x_ptr, group_size, num_groups, groups_per_block, eps, fp8_min, fp8_max,
                               _semantic=None):
    return core.extern_call(
        "",
        "",
        [
            tl.cast(x_q_ptr, tl.pointer_type(core.dtype("void")), _semantic=_semantic), 
            x_s_ptr,
            x_ptr,
            group_size, 
            num_groups,
            groups_per_block, 
            eps, 
            fp8_min, 
            fp8_max
        ],
        {
            (
                core.pointer_type(core.dtype("void")),
                core.pointer_type(core.dtype("fp32")),
                core.pointer_type(core.dtype("fp32")),
                core.dtype("int32"),
                core.dtype("int32"),
                core.dtype("int32"),
                core.dtype("fp32"),
                core.dtype("fp32"),
                core.dtype("fp32"),
            ): ("per_token_group_quant_8bit", ()),
        },
        is_pure=False,
        _semantic=_semantic,
    )

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