# insert_slice
# extract_slice
# get_element
# sort
# flip
# gather

import triton.language as tl
from triton.language import semantic, core, standard
from triton.language.core import (_unwrap_if_constexpr, _tensor_member_fn, _unwrap_iterable, builtin, constexpr, dtype,
                                  tensor, check_bit_width, _unwrap_if_constexpr, range)

from . import is_compile_on_910_95
from .aux_ops import compile_hint_impl

from typing import Optional, Tuple, List, overload
from triton._C.libtriton import ir


@_tensor_member_fn
@builtin
def insert_slice(ful, sub, offsets, sizes, strides, _semantic=None, _generator=None) -> tensor:
    """
    Insert a tensor to another tensor as specified by the operation’s offsets, sizes and strides arguments.

    :param ful: The tensor to receive tensor.
    :type ful: Tensor
    :param sub: The tensor to be inserted.
    :type sub: Tensor
    :param offsets:
    :type offsets: tuple of ints
    :param sizes:
    :type sizes: tuple of ints
    :param strides:
    :type strides: tuple of ints
    """

    def insert_slice_impl(ful: tensor, sub: tensor, offsets: List[tensor], sizes: List[int], strides: List[int],
                          builder: ir.builder) -> tensor:
        assert (len(ful.shape) == len(offsets))
        assert (len(ful.shape) == len(sizes))
        assert (len(ful.shape) == len(strides))
        assert (all([s >= 1 for s in sizes]))
        assert (all([s >= 0 for s in strides]))
        # Handle both tensor and int offsets (for interpreter mode)
        new_offsets = []
        for o in offsets:
            if isinstance(o, tensor):
                new_offsets.append(o.handle)
            elif isinstance(o, int):
                # For interpreter mode: keep as int
                new_offsets.append(o)
            else:
                new_offsets.append(o.handle if hasattr(o, 'handle') else o)
        ret_type = tl.block_type(ful.type.scalar, ful.shape)
        out = builder.create_insert_slice(ful.handle, sub.handle, new_offsets, sizes, strides)
        return tensor(out, ret_type)

    assert len(ful.shape) > 0
    assert len(ful.shape) == len(sub.shape)
    new_offsets = [_semantic.to_tensor(o) if isinstance(o, constexpr) else o for o in offsets]
    out = insert_slice_impl(ful, sub, new_offsets, sizes, strides, _semantic.builder)
    return out


@_tensor_member_fn
@builtin
def extract_slice(ful, offsets, sizes, strides, _semantic=None, _generator=None) -> tensor:
    """
    Extract a tensor from another tensor as specified by the operation’s offsets, sizes and strides arguments.

    :param ful: The tensor to split.
    :type ful: Tensor
    :param offsets:
    :type offsets: tuple of ints
    :param sizes:
    :type sizes: tuple of ints
    :param strides:
    :type strides: tuple of ints
    """

    def extract_slice_impl(ful: tensor, offsets: List[tensor], sizes: List[int], strides: List[int],
                           builder: ir.builder) -> tensor:
        assert (len(ful.shape) == len(offsets))
        assert (len(ful.shape) == len(sizes))
        assert (len(ful.shape) == len(strides))
        assert (all([s >= 1 for s in sizes]))
        assert (all([s >= 0 for s in strides]))
        # Handle both tensor and int offsets (for interpreter mode)
        new_offsets = []
        for o in offsets:
            if isinstance(o, tensor):
                new_offsets.append(o.handle)
            elif isinstance(o, int):
                # For interpreter mode: keep as int
                new_offsets.append(o)
            else:
                new_offsets.append(o.handle if hasattr(o, 'handle') else o)
        ret_type = tl.block_type(ful.type.scalar, sizes)
        out = builder.create_extract_slice(ful.handle, new_offsets, sizes, strides)
        return tensor(out, ret_type)

    assert len(ful.shape) > 0
    new_offsets = [_semantic.to_tensor(o) if isinstance(o, constexpr) else o for o in offsets]
    sub = extract_slice_impl(ful, new_offsets, sizes, strides, _semantic.builder)
    return sub


@_tensor_member_fn
@builtin
def get_element(src, indice, _semantic=None, _generator=None):
    """
    get_element op reads a ranked tensor and returns one element as specified by the given indices.
    The result of the op is a value with the same type as the elements of the tensor.
    The arity of indices must match the rank of the accessed value.

    :param src: The tensor to be accessed.
    :type src: Tensor
    :param indice:
    :type indice: tuple of ints
    """

    def get_element_impl(src: tensor, indice: List[tensor], builder: ir.builder):
        if len(src.shape) != len(indice):
            raise ValueError("Indice's rank must be equal to src tensor's rank")

        # Handle both tensor and int indices (for interpreter mode)
        new_indice = []
        for i in indice:
            if isinstance(i, tensor):
                new_indice.append(i.handle)
            elif isinstance(i, int):
                # For interpreter mode: convert int to TensorHandle
                new_indice.append(i)
            else:
                # Try to use .handle attribute if available
                new_indice.append(i.handle if hasattr(i, 'handle') else i)

        result = builder.create_extract_scalar(src.handle, new_indice)
        return _semantic.wrap_tensor(result, src.type.scalar, None)

    assert len(src.shape) > 0
    new_indice = [_semantic.to_tensor(i) if isinstance(i, constexpr) else i for i in indice]
    return get_element_impl(src, new_indice, _semantic.builder)


@builtin
def flip(ptr, dim=-1, _semantic=None, _generator=None):

    def flip_impl(ptr: tensor, dim: int, builder: ir.builder, generator=None):
        """
        Flips a tensor `ptr` along the dimension `dim`.

        :param ptr: the first input tensor
        :type ptr: tensor
        :param dim: the dimension to flip along
        :type dim: int
        :param generator: the code generator (required for reduce operations)
        :type generator: generator object
        """

        def _get_flip_dim(dim, shape):
            dim = _unwrap_if_constexpr(dim)
            shape = _unwrap_if_constexpr(shape)
            if dim is None:
                dim = len(shape) - 1
            if dim < 0:  # flip doesn't work if dim < 0 because the xor-swap for loop will start/end at the wrong index
                dim += len(shape)
            return constexpr(dim)

        def _log2(i: core.constexpr):
            log2 = 0
            n = core.constexpr(i).value
            while n > 1:
                n >>= 1
                log2 += 1
            return core.constexpr(log2)

        def flip_simd(ptr: tensor, dim: int, builder: ir.builder):
            """
            Triton flip operation for simd

            Args:
                ptr: tensor, input tensor
                dim: int, dimension to flip (can be negative, normalized here)
                builder: ir.builder, underlying IR builder
            Returns:
                flipped: tensor, same type and shape as input
            """

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
                    raise ValueError("ascend.flip requires tensor rank >= 1")
                norm_dim = dim if dim >= 0 else dim + rank
                if not (0 <= norm_dim < rank):
                    raise ValueError(f"ascend.flip got invalid dim={dim} for shape {tuple(shape)}")
                dim = norm_dim
            else:
                if dim < 0:
                    raise ValueError("ascend.flip with unknown rank requires non-negative dim")

            flipped_vals = builder.create_flip(ptr.handle, dim)
            flipped = tensor(flipped_vals, type=ptr.type)
            return flipped

        # If compile_mode is not simt, use the simd implementation
        if not builder.is_simt_mode():
            return flip_simd(ptr, dim, builder)
        core.static_assert(-len(ptr.shape) <= dim and dim < len(ptr.shape), _builder=builder)
        _dim: core.constexpr = _get_flip_dim(dim, ptr.shape)
        core.static_assert(standard._is_power_of_two(ptr.shape[_dim]), _builder=builder)
        steps: core.constexpr = _log2(ptr.shape[_dim])
        # If steps is 0, return the original tensor
        if steps == 0:
            return ptr
        # reshape the swap dimension to (2, 2, ..., 2)
        idtype = core.get_int_dtype(bitwidth=ptr.dtype.primitive_bitwidth, signed=True)
        y = core.reshape(
            ptr.to(idtype, bitcast=True, _builder=builder),
            ptr.shape.__getitem__(slice(None, _dim)) + [2] * steps + ptr.shape.__getitem__(slice(_dim + 1, None)),
            _builder=builder)
        for i in static_range(steps):
            y = y.__xor__(standard.xor_sum(y, _dim + i, True, _builder=builder, _generator=generator), _builder=builder)
        ptr = core.reshape(y, ptr.shape, _builder=builder).to(ptr.dtype, bitcast=True, _builder=builder)
        return ptr

    try:
        dim = int(dim.value) if hasattr(dim, "value") else int(dim)
    except Exception as e:
        raise TypeError(f"dim must be an integer (or tl.constexpr int), got {dim!r}") from e

    dim = len(ptr.shape) - 1 if dim == -1 else dim
    return flip_impl(ptr, dim, _semantic.builder, _generator)


class static_range:
    """
    Iterator for non-JIT Python functions that need to iterate over constexpr values.
    This is used in functions like flip that are called during compilation.
    """

    def __init__(self, arg1, arg2=None, step=None):
        if step is None:
            self.step = core.constexpr(1)
        else:
            self.step = step
        if arg2 is None:
            self.start = core.constexpr(0)
            self.end = arg1
        else:
            self.start = arg1
            self.end = arg2

    def __iter__(self):
        # Extract actual values from constexpr objects for iteration
        start_val = core._unwrap_if_constexpr(self.start)
        end_val = core._unwrap_if_constexpr(self.end)
        step_val = core._unwrap_if_constexpr(self.step)
        # Store as regular Python integers for iteration
        self._current = start_val
        self._end = end_val
        self._step = step_val
        return self

    def __next__(self):
        if self._current >= self._end:
            raise StopIteration
        value = self._current
        self._current += self._step
        return value


@builtin
def sort(ptr, dim=-1, descending=False, _semantic=None):
    """
    sort the input tensor along 'dim'

    param:
        ptr: tensor, input tensor
        dim: int or tl.constexpr[int], dimension to sort
        descending: bool or tl.constexpr[bool], the result is descending or not
        _builder: ir.builder
    return:
        values: tensor, the sorted tensor
    """

    def sort_impl(ptr: tensor, dim: int, descending, builder: ir.builder):
        allowed_types = {
            tl.int8, tl.int16, tl.bfloat16, tl.float16, tl.float32, tl.int32, tl.int64, tl.float8e4nv, tl.float8e5
        }
        base_ty = ptr.type.scalar if hasattr(ptr.type, "scalar") else ptr.type
        if base_ty not in allowed_types:
            raise TypeError(
                f"ascend.sort only supports int8, int16, bfloat16, float16, float32, int32, int64, float8e4nv, float8e5"
                f"but got {ptr.type}")

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
                raise ValueError("ascend.sort requires tensor rank >= 1")
            last_dim = rank - 1
            norm_dim = dim if dim >= 0 else dim + rank
            if norm_dim != last_dim:
                raise ValueError(f"ascend.sort only supports sorting along the last dimension "
                                 f"(dim={last_dim} or -1) for shape {tuple(shape)}, but got dim={dim}")
            dim = last_dim
        else:
            if dim != -1:
                raise ValueError("ascend.sort only supports the last dimension; when rank is unknown "
                                 "you must pass dim=-1")

        if hasattr(descending, "value"):
            descending = bool(descending.value)
        else:
            descending = bool(descending)

        sorted_vals = builder.create_sort(ptr.handle, dim, descending)

        values = tensor(sorted_vals, type=ptr.type)

        return values

    try:
        dim = int(dim.value) if hasattr(dim, "value") else int(dim)
    except Exception as e:
        raise TypeError(f"dim must be an integer (or tl.constexpr int), got {dim!r}. Error: {str(e)}") from e

    if hasattr(descending, "value"):
        descending = bool(descending.value)
    else:
        descending = bool(descending)

    ret = sort_impl(ptr, dim, descending, _semantic.builder)
    # interpreter mode not support compile_hint overflow_mode, direct return
    from triton.runtime.interpreter import InterpreterBuilder
    if isinstance(_semantic.builder, InterpreterBuilder):
        return ret
    base_ty = ptr.type.scalar if hasattr(ptr.type, "scalar") else ptr.type
    if base_ty.is_int8() or base_ty.is_int16():
        compile_hint_impl(ret, "overflow_mode", constexpr("saturate"), _semantic.builder)
    return ret


def ascend_cast_impl(input: tensor, dst_ty: dtype, _semantic=None, fp_downcast_rounding: Optional[str] = None,
                     overflow_mode: Optional[str] = None) -> tensor:
    src_ty = input.type
    if isinstance(dst_ty, tl.constexpr):
        dst_ty = dst_ty.value
    if isinstance(fp_downcast_rounding, tl.constexpr):
        fp_downcast_rounding = fp_downcast_rounding.value
    if src_ty.is_block():
        dst_ty = tl.block_type(dst_ty.scalar, input.type.get_block_shapes())
    if src_ty == dst_ty:
        return input

    src_sca_ty = src_ty.scalar
    dst_sca_ty = dst_ty.scalar
    if src_sca_ty == dst_sca_ty:
        return input

    # For fp downcasting default rounding mode should be RTNE, for all other conversions it should
    # not be set
    fp_downcast_rounding = _semantic._str_to_rounding_mode(fp_downcast_rounding)
    use_custom_rounding = False
    if dst_sca_ty.is_floating() and src_sca_ty.is_floating(
    ) and dst_sca_ty.primitive_bitwidth < src_sca_ty.primitive_bitwidth:
        if fp_downcast_rounding is None: fp_downcast_rounding = ir.ROUNDING_MODE.RTNE
        elif fp_downcast_rounding != ir.ROUNDING_MODE.RTNE: use_custom_rounding = True
    else:
        if fp_downcast_rounding is not None:
            raise ValueError("fp_downcast_rounding should be set only for truncating fp conversions. "
                             "Source scalar type is " + str(src_sca_ty) + " and destination type is " + str(dst_sca_ty))
    if not is_compile_on_910_95:
        if (src_sca_ty.is_fp8() or dst_sca_ty.is_fp8()) or (src_sca_ty.is_fp64() or dst_sca_ty.is_fp64()):
            raise ValueError("[fp8, fp64] is unsupported on Ascend for now."
                             "Source scalar type is " + str(src_sca_ty) + " and destination type is " + str(dst_sca_ty))
    if (src_sca_ty.is_fp8e4b15() or dst_sca_ty.is_fp8e4b15()):
        assert _semantic.builder.codegen_fns.get(
            "convert_custom_types") is not None, "target doesn't provide conversion for this type."
        return _semantic.builder.codegen_fns["convert_custom_types"](input, dst_ty, fp_downcast_rounding,
                                                                     _semantic=_semantic)
    # Casting with customized floating types involved: fp8 <=> bf16, fp16, fp32, fp64
    # and non-default rounding modes for downcasting
    if (src_sca_ty.is_fp8() and dst_sca_ty.is_floating()) or \
       (src_sca_ty.is_floating() and dst_sca_ty.is_fp8()) or \
       use_custom_rounding:
        return tensor(
            _semantic.builder.create_fp_to_fp(input.handle, dst_ty.to_ir(_semantic.builder), fp_downcast_rounding),
            dst_ty)

    # bf16 <=> (not fp32)
    if (src_sca_ty.is_fp16() and not dst_sca_ty.is_fp32()) or \
       (src_sca_ty.is_bf16() and not dst_sca_ty.is_fp32()):
        return ascend_cast_impl(ascend_cast_impl(input, tl.float32, _semantic), dst_sca_ty, _semantic)

    # Standard floating types' casting: truncation
    #   fp64 => fp32, fp16, bf16
    #   fp32 => fp16, bf16
    truncate_fp = src_sca_ty.is_floating() and \
        dst_sca_ty.is_floating() and \
        src_sca_ty.primitive_bitwidth > dst_sca_ty.primitive_bitwidth
    if truncate_fp:
        return tensor(_semantic.builder.create_fp_trunc(input.handle, dst_ty.to_ir(_semantic.builder)), dst_ty)

    # Standard floating types' casting: extension
    #   fp32 => fp64
    #   fp16 => fp32, fp64
    #   bf16 => fp32, fp64
    ext_fp = src_sca_ty.is_floating() and \
        dst_sca_ty.is_floating() and \
        src_sca_ty.primitive_bitwidth < dst_sca_ty.primitive_bitwidth
    if ext_fp:
        return tensor(_semantic.builder.create_fp_ext(input.handle, dst_ty.to_ir(_semantic.builder)), dst_ty)

    # Casting between integer types
    if src_sca_ty.is_int() and dst_sca_ty.is_int() and \
       (src_sca_ty.int_bitwidth != dst_sca_ty.int_bitwidth or src_sca_ty.int_signedness != dst_sca_ty.int_signedness):
        sign_extend = src_sca_ty.is_int_signed() and not src_sca_ty.is_bool()
        if dst_sca_ty.is_bool():
            ty = input.dtype.to_ir(_semantic.builder)
            _0 = tensor(_semantic.builder.get_null_value(ty), input.dtype)
            return _semantic.not_equal(input, _0)
        elif overflow_mode == "saturate" and \
             (src_sca_ty.is_int_unsigned() or dst_sca_ty.is_int_unsigned()) and \
             src_sca_ty.int_bitwidth >= dst_sca_ty.int_bitwidth:
            if is_compile_on_910_95:
                result = tensor(
                    _semantic.builder.create_int_cast(input.handle, dst_ty.to_ir(_semantic.builder), sign_extend),
                    dst_ty)
                compile_hint_impl(result, "saturate_src_unsigned", src_sca_ty.is_int_unsigned(), _semantic.builder)
                compile_hint_impl(result, "saturate_dst_unsigned", dst_sca_ty.is_int_unsigned(), _semantic.builder)
                return result
            else:
                return ascend_cast_impl(ascend_cast_impl(input, tl.float32, _semantic), dst_sca_ty, _semantic)
        return tensor(_semantic.builder.create_int_cast(input.handle, dst_ty.to_ir(_semantic.builder), sign_extend),
                      dst_ty)

    # Casting standard floating types to integer types
    if src_sca_ty.is_standard_floating() and dst_sca_ty.is_int():
        if dst_sca_ty.is_bool():
            ty = input.dtype.to_ir(_semantic.builder)
            _0 = tensor(_semantic.builder.get_null_value(ty), input.dtype)
            return _semantic.not_equal(input, _0)
        elif dst_sca_ty.is_int_signed():
            return tensor(_semantic.builder.create_fp_to_si(input.handle, dst_ty.to_ir(_semantic.builder)), dst_ty)
        else:
            return tensor(_semantic.builder.create_fp_to_ui(input.handle, dst_ty.to_ir(_semantic.builder)), dst_ty)

    # Casting integer types to standard floating types
    if src_sca_ty.is_int() and dst_sca_ty.is_standard_floating():
        if src_sca_ty.is_bool() or not src_sca_ty.is_int_signed():
            return tensor(_semantic.builder.create_ui_to_fp(input.handle, dst_ty.to_ir(_semantic.builder)), dst_ty)
        else:
            return tensor(_semantic.builder.create_si_to_fp(input.handle, dst_ty.to_ir(_semantic.builder)), dst_ty)

    # Casting pointer types to integer types
    if src_sca_ty.is_ptr() and dst_sca_ty.is_int():
        bitwidth = dst_sca_ty.int_bitwidth
        if bitwidth == 64:
            return tensor(_semantic.builder.create_ptr_to_int(input.handle, dst_ty.to_ir(_semantic.builder)), dst_ty)
        if bitwidth == 1:
            return _semantic.not_equal(ascend_cast_impl(input, tl.int64, _semantic),
                                       tensor(_semantic.builder.get_int64(0), tl.int64))

    # Casting integer types to pointer types
    if src_sca_ty.is_int() and dst_sca_ty.is_ptr():
        return tensor(_semantic.builder.create_int_to_ptr(input.handle, dst_ty.to_ir(_semantic.builder)), dst_ty)

    # Casting pointer types to pointer types
    if src_sca_ty.is_ptr() and dst_sca_ty.is_ptr():
        return tensor(_semantic.builder.create_bitcast(input.handle, dst_ty.to_ir(_semantic.builder)), dst_ty)

    assert False, f'cannot cast {input} to {dst_ty}'


@_tensor_member_fn
@builtin
def cast(input, dtype: dtype, fp_downcast_rounding: Optional[str] = None, bitcast: bool = False,
         overflow_mode: Optional[str] = None, _semantic=None):
    """
    Casts a tensor to the given :code:`dtype`.

    :param dtype: The target data type.
    :type dtype: dtype
    :param fp_downcast_rounding: The rounding mode for downcasting
        floating-point values. This parameter is only used when self is a
        floating-point tensor and dtype is a floating-point type with a
        smaller bitwidth. Supported values are :code:`"rtne"` (round to
        nearest, ties to even) and :code:`"rtz"` (round towards zero).
    :type fp_downcast_rounding: str, optional
    :param bitcast: If true, the tensor is bitcasted to the given
        :code:`dtype`, instead of being numerically casted.
    :type bitcast: bool, optional
    :param overflow_mode: When overflow_mode is not set or is "trunc",
        truncation (cut-off) will be used to handle overflow. When
        overflow_mode is "sautrate", the maximum value of the data type
        will be used to handle overflow.
    :type overflow_mode: string, optional
    """
    overflow_modes = ["trunc", "saturate"]
    input = _semantic.to_tensor(input)
    if isinstance(bitcast, constexpr):
        bitcast = bitcast.value
    if bitcast:
        return _semantic.bitcast(input, dtype)
    ret = ascend_cast_impl(input, dtype, _semantic, fp_downcast_rounding, overflow_mode)
    if overflow_mode is not None:
        if overflow_mode in overflow_modes:
            from triton.runtime.interpreter import InterpreterBuilder
            if isinstance(_semantic.builder, InterpreterBuilder):
                overflow_mode = constexpr(overflow_mode)
            compile_hint_impl(ret, "overflow_mode", overflow_mode, _semantic.builder)
        else:
            raise ValueError(f"Unknown overflow_mode:{overflow_mode} is found.")
    return ret
