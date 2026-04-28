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


@core.extern
def simple_shift(dst, _semantic=None):
    return core.extern_call(
        "",  # libname
        "",  # libpath
        [dst],  # args
        {(
            core.pointer_type(core.dtype("int32")),  # arg_type_symbol_dict
        ): ("simple_shift", ()),  # void return type
         },
        is_pure=False,
        _semantic=_semantic,
    )
