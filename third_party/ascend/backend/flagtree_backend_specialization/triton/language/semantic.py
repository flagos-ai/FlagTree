from typing import List
import triton.language as tl
from triton._C.libtriton import ir
from triton.language.semantic import to_tensor, bitcast, wrap_tensor, cast
from triton.language._utils import TRITON_MAX_TENSOR_NUMEL
from .tensor_descriptor import (
    _unwrap_if_constexpr,
    _unwrap_shape,
    block_type,
    tensor_descriptor
)

def arange_disable_check_power_of_two():
    return True

def check_arange_less_than_max_numel(range):
    if range > TRITON_MAX_TENSOR_NUMEL:
        raise ValueError(f"end - start must be less than or equal to TRITON_MAX_TENSOR_NUMEL = {TRITON_MAX_TENSOR_NUMEL}")

def is_cast_src_dst_scalar_type_equal(src_sca_ty, dst_sca_ty):
    if src_sca_ty == dst_sca_ty:
        return True
    return False

def check_unsupported_fp8_fp64(src_sca_ty, dst_sca_ty):
    if (src_sca_ty.is_fp8() or dst_sca_ty.is_fp8()) or (src_sca_ty.is_fp64() or dst_sca_ty.is_fp64()):
        raise ValueError("[fp8, fp64] is unsupported on Ascend for now."
                         "Source scalar type is " + str(src_sca_ty) + " and destination type is " + str(dst_sca_ty))

def ext_dot_operand_types():
    return (tl.int1,)

def dot_check_hf32_input_precision(input_precision, ir, lhs, rhs, ret_scalar_ty):
    if (input_precision == getattr(ir.INPUT_PRECISION, "HF32")):
        if (not lhs.dtype.is_fp32() or not rhs.dtype.is_fp32() or not ret_scalar_ty.is_fp32()):
            raise ValueError("input_precision = 'hf32' must be used with f32 * f32 = f32 on Ascend")

def dot_disable_check_max_num_imprecise_acc():
    return True

def reset_dot_max_num_imprecise_acc():
    return 0

def check_was_bool_to_int8_dtype(input):
    if hasattr(input, 'was_bool_to_int8'):
        if input.type.scalar.is_int8():
            raise TypeError(f"unexpected type bool")

def cast_bool_to_specified_dtype(input, builder):
    if hasattr(input, 'was_bool_to_int8'):
        assert input.type.scalar.is_int8(), "input wat bool to int8. However, input.type is not int8."
        return cast(input, tl.int1, builder)
    if input.type.scalar.is_floating():  # NOTE: Only in not_? What about logical_and/logical_or?
        raise TypeError(f"unexpected type {input.type.scalar}")
    return input

def check_unexpected_dtype_float(input):
    if input.type.scalar.is_floating():
        raise TypeError(f"unexpected type {input.type.scalar}")

def check_unexpected_dtype_bool(dtype):
    if dtype.is_bool():
        raise TypeError(f"Unexpected dtype {dtype}")

def set_load_legacy_other_input(other, care_padding, builder):
    if other is None and care_padding == True:
        return to_tensor(0, builder)
    return other

def disable_cast_back_when_load_legacy_ptr_is_bool():
    return True

def set_attr_was_bool_to_int8(ret, is_bool):
    if is_bool:
        ret.was_bool_to_int8 = True

def atomic_disable_original_check():
    return True

def ext_atomic_element_typechecking(element_ty, op):
    # Add `tl.int64` restriction for NPU
    if element_ty in [tl.int1, tl.float16, tl.float32, tl.float64, tl.bfloat16] and op in ['or', 'xor']:
        raise ValueError(f"atomic_{op} does not support {str(element_ty)}. "
                         "All support dtypes are int8, int16, int32, int64.")
    if element_ty in [tl.int1, tl.float64, tl.bfloat16] and op == 'xchg':
        raise ValueError(f"atomic_{op} does not support {str(element_ty)}. "
                         "All support dtypes are int8, int16, int32, int64, float16, float32.")
    if element_ty in [tl.int1, tl.int64, tl.float64] and op in ['add', 'max', 'min']:
        raise ValueError(f"atomic_{op} does not support {str(element_ty)}. "
                         "All support dtypes are int8, int16, int32, float16, float32, bfloat16.")
    if element_ty in [tl.int1, tl.float64]:
        raise ValueError(f"atomic_{op} does not support {str(element_ty)}. "
                         "All support dtypes are int8, int16, int32, int64, float16, float32, bfloat16.")

def atomic_cas_disable_element_bitwidth_check():
    return True

def ext_atomic_cas_element_typechecking(element_ty):
    if element_ty in [tl.int1, tl.int8, tl.float64, tl.bfloat16]:
        raise ValueError(f"atomic_cas does not support {str(element_ty)}. "
                        "All support dtypes are int16, int32, int64, float16, float32.")

def is_atomic_max_no_bitcast():
    return True

def is_atomic_min_no_bitcast():
    return True

def atomic_max_returning_tensor(ir, ptr, val, mask, sem, scope, builder):
    return tl.tensor(
        builder.create_atomic_rmw(ir.ATOMIC_OP.MAX, ptr.handle, val.handle, mask.handle, sem, scope), val.type)

def atomic_min_returning_tensor(ir, ptr, val, mask, sem, scope, builder):
    return tl.tensor(
        builder.create_atomic_rmw(ir.ATOMIC_OP.MIN, ptr.handle, val.handle, mask.handle, sem, scope), val.type)

def is_float_format_support_bf16():
    return True

def is_float_format_support_fp16():
    return True

def ext_dot_scaled_validate_lhs_dtype(lhs):
    assert lhs.dtype == tl.bfloat16 or lhs.dtype == tl.float16, f"lhs matrix dtype must be bf16 or fp16"

def ext_dot_scaled_validate_rhs_dtype(rhs):
    assert rhs.dtype == tl.bfloat16 or rhs.dtype == tl.float16, f"rhs matrix dtype must be bf16 or fp16"

def ext_dot_scaled_check_same_dtype(lhs, rhs):
    assert lhs.dtype == rhs.dtype, f"lhs rhs matrix must get same dtype"

def dot_scaled_disable_original_check():
    return True

def ext_dot_scaled_check_lhs_rhs_format(lhs_format, rhs_format):
    lhs_format: str = lhs_format.value
    rhs_format: str = rhs_format.value
    allowed_formats = {"bf16", "fp16"} # unsupported fp8/4 dtype: "e2m1", "e4m3", "e5m2"
    assert lhs_format in allowed_formats, f"NYI: lhs_format {lhs_format}"
    assert rhs_format in allowed_formats, f"NYI: rhs_format {rhs_format}"

def dot_scaled_recheck_rhs_scale_is_none(rhs_scale, rhs_scale_is_none):
    rhs_scale_is_none = rhs_scale is None or (isinstance(rhs_scale, tl.constexpr) and rhs_scale.value is None)
    return rhs_scale_is_none

def dot_scaled_check_lhs_scale_is_none(lhs_scale):
    lhs_scale_is_none = lhs_scale is None or (isinstance(lhs_scale, tl.constexpr) and lhs_scale.value is None)
    return lhs_scale_is_none

def is_dot_scaled_support_rhs_scale():
    return True

def check_dot_scaled_lhs_scale_dtype(lhs_scale):
    assert isinstance(lhs_scale, tl.tensor) and lhs_scale.dtype == tl.int8, f"lhs_scale must be int8 tensor"

def check_dot_scaled_rhs_scale_dtype(rhs_scale, rhs_scale_is_none):
    if not rhs_scale_is_none:
        assert isinstance(rhs_scale, tl.tensor) and rhs_scale.dtype == tl.int8, f"rhs_scale must be int8 tensor"

def _bitcast_to_fp_type(val, float_format, builder):
    triton_ty = {"e5m2": tl.float8e5, "e4m3": tl.float8e4nv, "bf16": tl.bfloat16, "fp16": tl.float16}.get(float_format)
    if triton_ty is None:
        assert float_format == "e2m1", f"Internal Error: Unexpected float format: {float_format}"
        assert val.dtype == tl.uint8, f"e2m1 format must be packed as uint8. Got {val.dtype}"
        return val
    if val.dtype == triton_ty:
        return val
    else:
        unsigned_ty = {"e5m2": tl.uint8, "e4m3": tl.uint8, "bf16": tl.uint16, "fp16": tl.uint16}[float_format]
        assert val.dtype == unsigned_ty, f"Unexpected dtype for {float_format}. Got {val.dtype}"
        return bitcast(val, triton_ty, builder)

def dot_scaled_lhs_bitcast_to_fp_type(lhs, lhs_format, builder):
    lhs_format: str = lhs_format.value
    lhs = _bitcast_to_fp_type(lhs, lhs_format, builder)
    return lhs

def dot_scaled_rhs_bitcast_to_fp_type(rhs, rhs_format, builder):
    rhs_format: str = rhs_format.value
    rhs = _bitcast_to_fp_type(rhs, rhs_format, builder)
    return rhs

def check_dot_scaled_dimension(lhs, rhs):
    assert lhs.type.shape[-1] == rhs.type.shape[-2], (
        f"lhs last dimension (columns) {lhs.shape[-1]} "
        f"must equal rhs penultimate dimension (rows) {rhs.shape[-2]}"
    )

def check_dot_scaled_pack_size(PACKED_A, K, lhs_format, lhs, rhs):
    lhs_format: str = lhs_format.value
    PACKED_B = 2 if lhs_format == "e2m1" else 1
    assert K * PACKED_B == PACKED_A * lhs.type.shape[
        -1], f"Reduction dimension should pack the same number of elements; (lhs: {lhs.shape} vs rhs: {rhs.shape})"

def set_dot_scaled_lhs_scale_handle(lhs_scale, lhs_scale_is_none):
    return None if lhs_scale_is_none else lhs_scale.handle

def ext_semantic_gather(src: tl.tensor, index: tl.tensor, axis: int, builder: ir.builder) -> tl.tensor:
    assert index.dtype.is_int(), "index must be an integer tensor"
    if not src.dtype.is_floating():
        raise ValueError(f"Expected dtype fp16/fp32/bf16, but got {src.dtype}")

    rank = len(src.type.shape)
    assert len(index.type.shape) == rank, "source and index tensors must have the same rank"

    assert -rank <= axis < rank, f"gather axis {axis} must be < source rank ({rank})"
    if axis < 0:
        axis += rank

    for d in range(rank):
        if d == axis:
            continue
        assert index.type.shape[d] == src.type.shape[d], f"index dim {axis} must match the corresponding source dim"

    gather = builder.create_gather(src.handle, index.handle, axis)
    return wrap_tensor(gather, src.type.scalar, index.type.shape)

def ext_semantic_insert_slice(ful: tl.tensor, sub: tl.tensor, offsets: List[tl.tensor], sizes: List[int], strides: List[int], builder: ir.builder) -> tl.tensor:
    assert(len(ful.shape) == len(offsets))
    assert(len(ful.shape) == len(sizes))
    assert(len(ful.shape) == len(strides))
    assert(all([s>=1 for s in sizes]))
    assert(all([s>=0 for s in strides]))
    new_offsets = [o.handle for o in offsets]
    ret_type = tl.block_type(ful.type.scalar, ful.shape)
    out = builder.create_insert_slice(ful.handle, sub.handle, new_offsets, sizes, strides)
    return tl.tensor(out, ret_type)

def ext_semantic_extract_slice(ful: tl.tensor, offsets: List[tl.tensor], sizes: List[int], strides: List[int], builder: ir.builder) -> tl.tensor:
    assert(len(ful.shape) == len(offsets))
    assert(len(ful.shape) == len(sizes))
    assert(len(ful.shape) == len(strides))
    assert(all([s>=1 for s in sizes]))
    assert(all([s>=0 for s in strides]))
    new_offsets = [o.handle for o in offsets]
    ret_type = tl.block_type(ful.type.scalar, sizes)
    out = builder.create_extract_slice(ful.handle, new_offsets, sizes, strides)
    return tl.tensor(out, ret_type)

def ext_semantic_get_element(src: tl.tensor, indice: List[tl.tensor], builder: ir.builder):
    if len(src.shape) != len(indice):
        raise ValueError("Indice's rank must be equal to src tensor's rank")

    new_indice = [i.handle for i in indice]
    result = builder.create_extract_scalar(src.handle, new_indice)
    return wrap_tensor(result, src.type.scalar, None)

def ext_semantic_compile_hint(ptr: tl.tensor, hint_name: str, hint_val, builder: ir.builder):
    if not hint_val:
        hint_val = builder.get_unit_attr()
    elif isinstance(hint_val, bool):
        hint_val = builder.get_bool_attr(hint_val)
    elif isinstance(hint_val, int):
        hint_val = builder.get_int32_attr(hint_val)
    elif isinstance(hint_val, tl.constexpr):
        hint_val = builder.get_str_attr(hint_val.value)
    elif isinstance(hint_val, list):
        # only support i64 array attr for now
        hint_val = builder.get_i64_array_attr(hint_val)
    else:
        raise ValueError(f"Unsupported hint value type: {type(hint_val)}")
    builder.create_annotation(ptr.handle, hint_name, hint_val)

def ext_semantic_custom_op(builder: ir.builder, op_name: str, **kwargs):
    if op_name == "sync_block_all":
        return builder.create_custom_op_for_inter_core_sync(op_name, kwargs["mode"], kwargs["event_id"])

    elif op_name == "sync_block_set":
        return builder.create_custom_op_for_inter_core_sync(op_name, kwargs["sender"], kwargs["event_id"])

    elif op_name == "sync_block_wait":
        return builder.create_custom_op_for_inter_core_sync(op_name, kwargs["sender"], kwargs["event_id"])

    raise ValueError(f"Unsupported custom op: {op_name}")

def ext_semantic_sort(ptr: tl.tensor, dim: int, descending, builder: ir.builder):
    """
    Triton sort 操作

    参数：
        ptr: tl.tensor，输入张量
        dim: int，排序维度，必须是尾轴（最后一维）
        descending: bool 或 constexpr，是否降序
        builder: ir.builder，底层 IR 构建器
    返回：
        values: tl.tensor，排序后的值（类型与输入一致）
    """

    allowed_types = {tl.int8, tl.int16, tl.bfloat16, tl.float16, tl.float32}
    base_ty = ptr.type.scalar if hasattr(ptr.type, "scalar") else ptr.type
    if base_ty not in allowed_types:
        raise TypeError(
            f"tt.sort only supports int8, int16, bfloat16, float16, float32, "
            f"but got {ptr.type}"
        )

    shape = getattr(ptr, "shape", None)
    if shape is None or shape == ():
        shape = getattr(getattr(ptr, "type", None), "shape", None)

    rank = None
    if shape is not None:
        try:
            rank = len(shape)
        except Exception:
            rank = len(list(shape))

    if rank is not None:
        if rank < 1:
            raise ValueError("tt.sort requires tensor rank >= 1")
        last_dim = rank - 1
        norm_dim = dim if dim >= 0 else dim + rank
        if norm_dim != last_dim:
            raise ValueError(
                f"tt.sort only supports sorting along the last dimension "
                f"(dim={last_dim} or -1) for shape {tuple(shape)}, but got dim={dim}"
            )
        dim = last_dim
    else:
        if dim != -1:
            raise ValueError(
                "tt.sort only supports the last dimension; when rank is unknown "
                "you must pass dim=-1"
            )

    if hasattr(descending, "value"):
        descending = bool(descending.value)
    else:
        descending = bool(descending)

    sorted_vals = builder.create_sort(ptr.handle, dim, descending)

    values = tl.tensor(sorted_vals, type=ptr.type)

    return values

def ext_semantic_scalar_constant(value, dtype: tl.dtype, builder: ir.builder) -> tl.tensor:
    if dtype is None:
        raise ValueError("dtype must be specified when value is not a tensor")
    if value == 0:
        value = builder.get_null_value(dtype.to_ir(builder))
    else:
        get_value_fn = getattr(builder, f"get_{dtype.name}")
        value = get_value_fn(value)
    return tl.tensor(value, dtype)

def ext_semantic_make_scalar(value, dtype: tl.dtype, builder: ir.builder) -> tl.tensor:
    if isinstance(value, tl.tensor):
        assert value.numel.value == 1, "only accepts size-1 tensor"
        return cast(value, dtype, builder)
    return ext_semantic_scalar_constant(value, dtype, builder)

def ext_semantic_make_tensor_descriptor(
    base: tl.tensor,
    shape: List[tl.tensor],
    strides: List[tl.tensor],
    block_shape: List[tl.constexpr],
    builder: ir.builder
) -> tensor_descriptor:
    ndim = len(shape)
    if not (1 <= ndim <= 5):
        raise ValueError(f"Expected 1 <= ndim <= 5 but got {ndim} dimensions")
    if len(strides) != ndim:
        raise ValueError(f"Expected {ndim} strides but got {len(strides)}")
    if len(block_shape) != ndim:
        raise ValueError(f"Expected block_shape to have {ndim} dimensions but got {len(strides)}")
    assert isinstance(base.dtype, tl.pointer_type)
    primitive_bitwidth = base.dtype.element_ty.primitive_bitwidth
    if primitive_bitwidth == 1:
        raise ValueError("int1 type is not supported for make_tensor_descriptor yet")
    elem_size = primitive_bitwidth // 8
    contig_dim_size = _unwrap_if_constexpr(block_shape[-1])
    if contig_dim_size * elem_size < 16:
        raise ValueError(
            f"Descriptor block shape must have at least 16 bytes in the last dimension, but got {contig_dim_size} * {elem_size} = {contig_dim_size * elem_size} bytes"
        )

    strides[-1] = _unwrap_if_constexpr(strides[-1])
    if strides[-1] != 1:
        raise ValueError(f"Tensor descriptor last dim must be 1 but got {strides[-1]}")

    shape = [ext_semantic_make_scalar(x, tl.int32, builder) for x in shape]
    strides = [ext_semantic_make_scalar(x, tl.int64, builder) for x in strides]

    block_shape = _unwrap_shape(block_shape)

    assert isinstance(base.type, tl.pointer_type)
    desc_block_type = block_type(base.type.element_ty, block_shape)
    base_handle = base.handle
    is_signed_int = base.type.element_ty.is_int_signed()

    handle = builder.create_make_tensor_descriptor(base_handle, [s.handle for s in shape],
                                                    [s.handle for s in strides], block_shape, is_signed_int)
    return tensor_descriptor(handle, shape, strides, desc_block_type)

def ext_semantic_gather_load(
    src: tl.tensor,
    gather_dim: int,
    gather_indices: tl.tensor,
    src_shape: List[tl.tensor],  # Can be int or tl.tensor
    src_offset: List[tl.tensor],  # Can be int or tl.tensor
    read_shape: List[int],  # Can be int or tl.tensor
    builder: ir.builder
) -> tl.tensor:
    """
    Gather load operation that loads data from multiple indices along a dimension.
    Args:
        src: Source tensor pointer (in GM)
        gather_dim: Dimension along which to gather
        gather_indices: 1D tensor of indices to gather (in UB)
        src_shape: Complete shape of source tensor (can be int or tensor)
        src_offset: Starting offset for reading (can be int or tensor)
        read_shape: Size to read (tile shape, can be int or tensor)
        builder: IR builder

    Returns:
        Result tensor in UB

    Constraints:
        - read_shape[gather_dim] must be -1
        - src_offset[gather_dim] can be -1 (ignored)
    """
    # Validate inputs
    ndim = len(src_shape)
    assert len(src_offset) == ndim, \
        f"src_offset length {len(src_offset)} must match src_shape length {ndim}"
    assert len(read_shape) == ndim, \
        f"read_shape length {len(read_shape)} must match src_shape length {ndim}"
    assert 0 <= gather_dim < ndim, \
        f"gather_dim={gather_dim} must be in range [0, {ndim})"
    assert len(gather_indices.shape) == 1, \
        f"gather_indices must be 1D tensor, got {len(gather_indices.shape)}D"

    newsrc_shape = [o.handle for o in src_shape]
    newsrc_offset = [o.handle for o in src_offset]
    # Create output type
    return_shape = [
        gather_indices.shape[0] if i == gather_dim else read_shape[i]
        for i in range(ndim)
    ]
    element_ty = src.type.element_ty
    output_ty = tl.block_type(element_ty, return_shape)
    out = builder.create_gather_load(src.handle, gather_indices.handle, gather_dim, newsrc_shape, newsrc_offset, read_shape, return_shape)
    return tl.tensor(out, output_ty)

semantic_ext_spec_func_list = [
    "gather", "insert_slice", "extract_slice", "get_element", "compile_hint",
    "custom_op", "sort", "scalar_constant", "make_scalar", "make_tensor_descriptor", "gather_load"
]
