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
def per_token_group_quant_8bit(x_ptr, x_q_ptr, x_s_ptr, group_size, num_groups, groups_per_block, eps, fp8_min, fp_max,
                               _semantic=None):
    return core.extern_call(
        "",
        "",
        [
            x_ptr,
            tl.cast(x_q_ptr, tl.pointer_type(core.dtype("void")), _semantic=_semantic), x_s_ptr, group_size, num_groups,
            groups_per_block, eps, fp8_min, fp_max
        ],
        {
            (
                core.pointer_type(core.dtype("fp32")),
                core.pointer_type(core.dtype("void")),
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
