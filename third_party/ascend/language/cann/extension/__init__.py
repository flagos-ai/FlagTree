try:
    import acl
    is_compile_on_910_95 = acl.get_soc_name().startswith("Ascend910_95")
except Exception as e:
    is_compile_on_910_95 = False

from .core import (
    ascend_address_space,
    builtin,
    CORE,
    copy_from_ub_to_l1,
    debug_barrier,
    fixpipe,
    FixpipeDMAMode,
    FixpipeDualDstMode,
    FixpipePreQuantMode,
    FixpipePreReluMode,
    int64,
    is_builtin,
    MODE,
    PIPE,
    sub_vec_id,
    sub_vec_num,
    sync_block_all,
    sync_block_set,
    sync_block_wait,
    SYNC_IN_VF,
)

from .scope import scope

from .custom_op import (
    custom,
    custom_semantic,
    register_custom_op,
)

from . import builtin_custom_ops

from .math_ops import (atan2, isfinited, finitef)

from .aux_ops import (
    parallel,
    compile_hint,
    multibuffer,
)

from .vec_ops import (
    insert_slice,
    extract_slice,
    get_element,
    sort,
    flip,
    cast,
)

from .mem_ops import (
    index_select,
    index_put,
    gather_out_to_ub,
    scatter_ub_to_out,
    index_select_simd,
)

__all__ = [
    # core
    "builtin",
    "copy_from_ub_to_l1",
    "CORE",
    "debug_barrier",
    "fixpipe",
    "FixpipeDMAMode",
    "FixpipeDualDstMode",
    "FixpipePreQuantMode",
    "FixpipePreReluMode",
    "int64",
    "is_builtin",
    "MODE",
    "PIPE",
    "sub_vec_id",
    "sub_vec_num",
    "sync_block_all",
    "SYNC_IN_VF",

    # address space
    "ascend_address_space",

    # scope
    "scope",

    # custom op
    "custom",
    "custom_semantic",
    "register_custom_op",

    # math ops
    "atan2",
    "isfinited",
    "finitef",

    # aux ops
    "sync_block_set",
    "sync_block_wait",
    "parallel",
    "compile_hint",
    "multibuffer",

    # vec ops
    "insert_slice",
    "extract_slice",
    "get_element",
    "sort",
    "flip",
    "cast",

    # mem ops
    "index_select",
    "index_put",
    "gather_out_to_ub",
    "scatter_ub_to_out",
    "index_select_simd",
]
