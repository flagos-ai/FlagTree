from typing import List, Sequence, Union
from triton._C.libtriton import ir
import triton.language.semantic as semantic
from triton.language.core import (
    _unwrap_iterable,
    _constexpr_to_value,
    constexpr,
    tensor,
    check_bit_width,
    _unwrap_if_constexpr,
    add,
    sub,
    mul,
)

from triton.language.tensor_descriptor import tensor_descriptor, tensor_descriptor_base

def ext_cast_set_overflow_modes():
    return ["trunc", "saturate"]

def ext_cast_check_overflow_mode(overflow_mode, overflow_modes, ret, _builder):
    if overflow_mode is not None:
        if overflow_mode in overflow_modes:
            semantic.compile_hint(ret, "overflow_mode", overflow_mode, _builder)
        else:
            raise ValueError(f"Unknown overflow_mode:{overflow_mode} is found.")

def ext_trans_unwrap_iterable(dims):
    return _unwrap_iterable(dims)

def check_dot_deprecated_param_allow_tf32(allow_tf32):
    assert (
        not allow_tf32
    ), "allow_tf32 is deprecated, please use input_precision='hf32' on Ascend instead."

def check_dot_invalid_input_precision(input_precision):
    assert input_precision not in [
            "tf32",
            "tf32x3",
        ], "input_precision == tf32 or tf32x3 is invalid, please use input_precision='hf32' on Ascend instead."

def ext_core_gather(src, index, axis, _builder=None):
    """Gather from a tensor along a given dimension.
    :param src: the source tensor
    :type src: Tensor
    :param index: the index tensor
    :type index: Tensor
    :param axis: the dimension to gather along
    :type axis: int
    """
    axis = _constexpr_to_value(axis)
    return semantic.gather(src, index, axis, _builder)

def ext_core_insert_slice(ful, sub, offsets, sizes, strides, _builder=None, _generator=None) -> tensor:
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
    assert len(ful.shape) > 0
    assert len(ful.shape) == len(sub.shape)
    new_offsets = [
        semantic.to_tensor(o, _builder) if isinstance(o, constexpr) else o
        for o in offsets
    ]
    out = semantic.insert_slice(ful, sub, new_offsets, sizes, strides, _builder)
    return out

def ext_core_extract_slice(ful, offsets, sizes, strides, _builder=None, _generator=None) -> tensor:
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
    assert len(ful.shape) > 0
    new_offsets = [
        semantic.to_tensor(o, _builder) if isinstance(o, constexpr) else o
        for o in offsets
    ]
    sub = semantic.extract_slice(ful, new_offsets, sizes, strides, _builder)
    return sub

def ext_core_get_element(src, indice, _builder=None, _generator=None):
    """
    get_element op reads a ranked tensor and returns one element as specified by the given indices.
    The result of the op is a value with the same type as the elements of the tensor.
    The arity of indices must match the rank of the accessed value.

    :param src: The tensor to be accessed.
    :type src: Tensor
    :param indice:
    :type indice: tuple of ints
    """
    assert len(src.shape) > 0
    new_indice = [
        semantic.to_tensor(i, _builder) if isinstance(i, constexpr) else i
        for i in indice
    ]
    return semantic.get_element(src, new_indice, _builder)

def ext_core_add(self, other, _builder=None):
    return add(self, other, sanitize_overflow=False, _builder=_builder)

def ext_core_radd(self, other, _builder=None):
    return add(other, self, sanitize_overflow=False, _builder=_builder)

def ext_core_sub(self, other, _builder=None):
    return sub(self, other, sanitize_overflow=False, _builder=_builder)

def ext_core_rsub(self, other, _builder=None):
    return sub(other, self, sanitize_overflow=False, _builder=_builder)

def ext_core_mul(self, other, _builder=None):
    return mul(self, other, sanitize_overflow=False, _builder=_builder)

def ext_core_rmul(self, other, _builder=None):
    return mul(other, self, sanitize_overflow=False, _builder=_builder)

def ext_core_lshift(self, other, _builder=None):
    if self.type.scalar.is_floating():
        raise TypeError(f"unexpected type {self.type.scalar}")
    check_bit_width(self, other)
    other = _unwrap_if_constexpr(other)
    return semantic.shl(self, other, _builder)

def ext_core_rshift(self, other, _builder=None):
    if self.type.scalar.is_floating():
        raise TypeError(f"unexpected type {self.type.scalar}")
    other = _unwrap_if_constexpr(other)
    check_bit_width(self, other)
    if self.dtype.is_int_signed():
        return semantic.ashr(self, other, _builder)
    else:
        return semantic.lshr(self, other, _builder)

def ext_core_compile_hint(ptr, hint_name, hint_val=None, _builder=None):
    def _unwrap(val):
        return _unwrap_if_constexpr(val) if val else val

    hint_name = _constexpr_to_value(hint_name)
    assert isinstance(hint_name, str), f"hint name: {hint_name} is not string"
    if isinstance(hint_val, list):
        hint_val = [_unwrap(val) for val in hint_val]
    else:
        hint_val = _unwrap(hint_val)
    hint_val = _unwrap_if_constexpr(hint_val) if hint_val else hint_val
    semantic.compile_hint(ptr, hint_name, hint_val, _builder)

def ext_core_sort(ptr, dim=-1, descending=False, _builder=None):
    """
    Triton sort 前端接口

    参数：
        ptr: tl.tensor，输入张量
        dim: int 或 tl.constexpr[int]，排序维度
        descending: bool 或 tl.constexpr[bool]，是否降序
        _builder: ir.builder，底层 IR 构建器
    返回：
        values: tl.tensor，排序后的值（类型与输入一致）
    """

    try:
        dim = int(dim.value) if hasattr(dim, "value") else int(dim)
    except Exception as e:
        raise TypeError(f"dim must be an integer (or tl.constexpr int), got {dim!r}. Error: {str(e)}") from e

    if hasattr(descending, "value"):
        descending = bool(descending.value)
    else:
        descending = bool(descending)

    ret = semantic.sort(ptr, dim, descending, _builder)
    base_ty = ptr.type.scalar if hasattr(ptr.type, "scalar") else ptr.type
    if base_ty.is_int8() or base_ty.is_int16():
        semantic.compile_hint(ret, "overflow_mode", constexpr("saturate"), _builder)
    return ret

def ext_core_multibuffer(src: tensor, size, _builder=None):
    """
    Set multi_buffer for an existing tensor
    :src: tensor set to bufferize multiple time
    :size: number of copies
    """
    buffer_size = _constexpr_to_value(size)
    assert isinstance(buffer_size, int) and buffer_size == 2, f"only support bufferize equals 2"
    semantic.compile_hint(src, "multi_buffer", buffer_size, _builder)

def ext_core_sync_block_all(mode, event_id, _builder=None):
    mode = _constexpr_to_value(mode)
    event_id = _constexpr_to_value(event_id)
    assert isinstance(mode, str), f"mode: {mode} is not string"
    assert isinstance(event_id, int) and (event_id >= 0) and (event_id < 16), f"event_id: {event_id} should be 0 ~ 15"
    assert mode == "all_cube" or mode == "all_vector" or mode == "all", f"ERROR: mode = {mode}, only supports all_cube/all_vector/all"
    semantic.custom_op(_builder, "sync_block_all", mode=mode, event_id=event_id)

def ext_core_sync_block_set(sender, receiver, event_id, _builder=None):
    sender = _constexpr_to_value(sender)
    receiver = _constexpr_to_value(receiver)
    event_id = _constexpr_to_value(event_id)
    assert isinstance(sender, str) and (sender == "cube" or sender == "vector"), f"ERROR: sender = {sender}, only supports cube/vector"
    assert isinstance(receiver, str) and (receiver == "cube" or receiver == "vector"), f"ERROR: receiver = {receiver}, only supports cube/vector"
    assert isinstance(event_id, int) and (event_id >= 0) and (event_id < 16), f"event_id: {event_id} should be 0 ~ 15"
    if sender == receiver:
        raise ValueError(f'Unexpected pair: {sender} -> {receiver}, only supports cube -> vector or vector -> cube')
    semantic.custom_op(_builder, "sync_block_set", sender=sender, event_id=event_id)

def ext_core_sync_block_wait(sender, receiver, event_id, _builder=None):
    sender = _constexpr_to_value(sender)
    receiver = _constexpr_to_value(receiver)
    event_id = _constexpr_to_value(event_id)
    assert isinstance(sender, str) and (sender == "cube" or sender == "vector"), f"ERROR: sender = {sender}, only supports cube/vector"
    assert isinstance(receiver, str) and (receiver == "cube" or receiver == "vector"), f"ERROR: receiver = {receiver}, only supports cube/vector"
    assert isinstance(event_id, int) and (event_id >= 0) and (event_id < 16), f"event_id: {event_id} should be 0 ~ 15"
    if sender == receiver:
        raise ValueError(f'Unexpected pair: {sender} -> {receiver}, only supports cube -> vector or vector -> cube')
    semantic.custom_op(_builder, "sync_block_wait", sender=sender, event_id=event_id)

def ext_core_load_tensor_descriptor(desc: tensor_descriptor_base, offsets: Sequence[Union[constexpr, tensor]],
                                    _builder=None) -> tensor:
    """Load a block of data from a tensor descriptor."""
    return desc.load(offsets, _builder=_builder)

def ext_core_store_tensor_descriptor(desc: tensor_descriptor_base, offsets: Sequence[Union[constexpr, tensor]], value: tensor,
                                     _builder=None) -> tensor:
    """Store a block of data to a tensor descriptor."""
    return desc.store(offsets, value, _builder=_builder)

def ext_core_make_tensor_descriptor(
    base: tensor,
    shape: List[tensor],
    strides: List[tensor],
    block_shape: List[constexpr],
    _builder=None,
) -> tensor_descriptor:
    """Make a tensor descriptor object

    :param base: the base pointer of the tensor, must be 16-byte aligned
    :param shape: A list of non-negative integers representing the tensor shape
    :param strides: A list of tensor strides. Leading dimensions must be multiples
        of 16-byte strides and the last dimension must be contiguous.
    :param block_shape: The shape of block to be loaded/stored from global memory

    Notes
    *****
    On NVIDIA GPUs with TMA support, this will result in a TMA descriptor object
    and loads and stores from the descriptor will be backed by the TMA hardware.

    Currently only 2-5 dimensional tensors are supported.

    Example
    *******
    .. code-block:: python

        @triton.jit
        def inplace_abs(in_out_ptr, M, N, M_BLOCK: tl.constexpr, N_BLOCK: tl.constexpr):
            desc = tl.make_tensor_descriptor(
                in_out_ptr,
                shape=[M, N],
                strides=[N, 1],
                block_shape=[M_BLOCK, N_BLOCK],
            )

            moffset = tl.program_id(0) * M_BLOCK
            noffset = tl.program_id(1) * N_BLOCK

            value = desc.load([moffset, noffset])
            desc.store([moffset, noffset], tl.abs(value))

        # TMA descriptors require a global memory allocation
        def alloc_fn(size: int, alignment: int, stream: Optional[int]):
            return torch.empty(size, device="cuda", dtype=torch.int8)

        triton.set_allocator(alloc_fn)

        M, N = 256, 256
        x = torch.randn(M, N, device="cuda")
        M_BLOCK, N_BLOCK = 32, 32
        grid = (M // M_BLOCK, N // N_BLOCK)
        inplace_abs[grid](x, M, N, M_BLOCK, N_BLOCK)

    """
    return semantic.make_tensor_descriptor(base, shape, strides, block_shape, _builder)

def ext_core_dtype_to_ir(self, builder: ir.builder) -> ir.type:
    if self.name.startswith("fp8"):
        raise ValueError(f'unexpected type fp8.')

    if self.name == 'void':
        return builder.get_void_ty()
    elif self.name == 'int1':
        return builder.get_int1_ty()
    elif self.name in ('int8', 'uint8'):
        return builder.get_int8_ty()
    elif self.name in ('int16', 'uint16'):
        return builder.get_int16_ty()
    elif self.name in ('int32', 'uint32'):
        return builder.get_int32_ty()
    elif self.name in ('int64', 'uint64'):
        return builder.get_int64_ty()
    elif self.name == 'fp8e5':
        return builder.get_fp8e5_ty()
    elif self.name == 'fp8e5b16':
        return builder.get_fp8e5b16_ty()
    elif self.name == 'fp8e4nv':
        return builder.get_fp8e4nv_ty()
    elif self.name == 'fp8e4b8':
        return builder.get_fp8e4b8_ty()
    elif self.name == 'fp8e4b15':
        return builder.get_fp8e4b15_ty()
    elif self.name == 'fp16':
        return builder.get_half_ty()
    elif self.name == 'bf16':
        return builder.get_bf16_ty()
    elif self.name == 'fp32':
        return builder.get_float_ty()
    elif self.name == 'fp64':
        return builder.get_double_ty()
    raise ValueError(f'fail to convert {self} to ir type')
