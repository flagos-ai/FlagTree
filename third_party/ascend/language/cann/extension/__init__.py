from triton.tools.get_ascend_devices import is_compile_on_910_95
from triton._C.libtriton.ascend import ir as _ascend_ir

# MLIR affine bindings (same objects as triton._C.libtriton.ascend.ir).
affine_expr = _ascend_ir.affine_expr
affine_constant_expr = _ascend_ir.affine_constant_expr
affine_dim_expr = _ascend_ir.affine_dim_expr
affine_symbol_expr = _ascend_ir.affine_symbol_expr
affine_binary_op_expr = _ascend_ir.affine_binary_op_expr
affine_map = _ascend_ir.affine_map

AffineExpr = affine_expr
AffineConstantExpr = affine_constant_expr
AffineDimExpr = affine_dim_expr
AffineSymbolExpr = affine_symbol_expr
AffineBinaryOpExpr = affine_binary_op_expr
AffineMap = affine_map

from .core import (
    ascend_address_space,
    builtin,
    CORE,
    copy_from_ub_to_l1,
    copy,
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
    IteratorType,
    sub_vec_id,
    sub_vec_num,
    sync_block_all,
    sync_block_set,
    sync_block_wait,
    SYNC_IN_VF,
    conv1d,
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
    index_put,
    gather_out_to_ub,
    scatter_ub_to_out,
    index_select_simd,
)

__all__ = [
    # core
    "builtin",
    "copy_from_ub_to_l1",
    "copy",
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
    "IteratorType",
    "sub_vec_id",
    "sub_vec_num",
    "sync_block_all",
    "SYNC_IN_VF",
    "conv1d",

    # address space
    "ascend_address_space",

    # ascend IR affine (MLIR)
    "affine_expr",
    "affine_constant_expr",
    "affine_dim_expr",
    "affine_symbol_expr",
    "affine_binary_op_expr",
    "affine_map",
    "AffineExpr",
    "AffineConstantExpr",
    "AffineDimExpr",
    "AffineSymbolExpr",
    "AffineBinaryOpExpr",
    "AffineMap",

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
    "index_put",
    "gather_out_to_ub",
    "scatter_ub_to_out",
    "index_select_simd",
]
