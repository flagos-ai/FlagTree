from typing import List, Sequence, Union
from triton._C.libtriton import ir
import triton.language.semantic as semantic
from . import semantic as semantic_spec
from triton.language.core import (
    builtin,
    _tensor_member_fn,
    _unwrap_iterable,
    _constexpr_to_value,
    constexpr,
    tensor,
    range,
    check_bit_width,
    _unwrap_if_constexpr,
    add,
    sub,
    mul,
)

from .tensor_descriptor import tensor_descriptor, tensor_descriptor_base

def enable_care_padding_load():
    return True

def ext_cast_set_overflow_modes():
    return ["trunc", "saturate"]

def ext_cast_check_overflow_mode(overflow_mode, overflow_modes, ret, _builder):
    if overflow_mode is not None:
        if overflow_mode in overflow_modes:
            semantic_spec.ext_semantic_compile_hint(ret, "overflow_mode", overflow_mode, _builder)
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

@_tensor_member_fn
@builtin
def gather(src, index, axis, _builder=None):
    """Gather from a tensor along a given dimension.
    :param src: the source tensor
    :type src: Tensor
    :param index: the index tensor
    :type index: Tensor
    :param axis: the dimension to gather along
    :type axis: int
    """
    axis = _constexpr_to_value(axis)
    return semantic_spec.ext_semantic_gather(src, index, axis, _builder)

@_tensor_member_fn
@builtin
def insert_slice(ful, sub, offsets, sizes, strides, _builder=None, _generator=None) -> tensor:
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
    out = semantic_spec.ext_semantic_insert_slice(ful, sub, new_offsets, sizes, strides, _builder)
    return out

@_tensor_member_fn
@builtin
def extract_slice(ful, offsets, sizes, strides, _builder=None, _generator=None) -> tensor:
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
    sub = semantic_spec.ext_semantic_extract_slice(ful, new_offsets, sizes, strides, _builder)
    return sub

@_tensor_member_fn
@builtin
def get_element(src, indice, _builder=None, _generator=None):
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
    return semantic_spec.ext_semantic_get_element(src, new_indice, _builder)

@builtin
def __add__(self, other, _builder=None):
    return add(self, other, sanitize_overflow=False, _builder=_builder)

@builtin
def __radd__(self, other, _builder=None):
    return add(other, self, sanitize_overflow=False, _builder=_builder)

@builtin
def __sub__(self, other, _builder=None):
    return sub(self, other, sanitize_overflow=False, _builder=_builder)

@builtin
def __rsub__(self, other, _builder=None):
    return sub(other, self, sanitize_overflow=False, _builder=_builder)

@builtin
def __mul__(self, other, _builder=None):
    return mul(self, other, sanitize_overflow=False, _builder=_builder)

@builtin
def __rmul__(self, other, _builder=None):
    return mul(other, self, sanitize_overflow=False, _builder=_builder)

@builtin
def __lshift__(self, other, _builder=None):
    if self.type.scalar.is_floating():
        raise TypeError(f"unexpected type {self.type.scalar}")
    check_bit_width(self, other)
    other = _unwrap_if_constexpr(other)
    return semantic.shl(self, other, _builder)

@builtin
def __rshift__(self, other, _builder=None):
    if self.type.scalar.is_floating():
        raise TypeError(f"unexpected type {self.type.scalar}")
    other = _unwrap_if_constexpr(other)
    check_bit_width(self, other)
    if self.dtype.is_int_signed():
        return semantic.ashr(self, other, _builder)
    else:
        return semantic.lshr(self, other, _builder)

@builtin
def compile_hint(ptr, hint_name, hint_val=None, _builder=None):
    def _unwrap(val):
        return _unwrap_if_constexpr(val) if val else val

    hint_name = _constexpr_to_value(hint_name)
    assert isinstance(hint_name, str), f"hint name: {hint_name} is not string"
    if isinstance(hint_val, list):
        hint_val = [_unwrap(val) for val in hint_val]
    else:
        hint_val = _unwrap(hint_val)
    hint_val = _unwrap_if_constexpr(hint_val) if hint_val else hint_val
    semantic_spec.ext_semantic_compile_hint(ptr, hint_name, hint_val, _builder)

@builtin
def sort(ptr, dim=-1, descending=False, _builder=None):
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

    ret = semantic_spec.ext_semantic_sort(ptr, dim, descending, _builder)
    base_ty = ptr.type.scalar if hasattr(ptr.type, "scalar") else ptr.type
    if base_ty.is_int8() or base_ty.is_int16():
        semantic_spec.ext_semantic_compile_hint(ret, "overflow_mode", constexpr("saturate"), _builder)
    return ret

@builtin
def multibuffer(src: tensor, size, _builder=None):
    """
    Set multi_buffer for an existing tensor
    :src: tensor set to bufferize multiple time
    :size: number of copies
    """
    buffer_size = _constexpr_to_value(size)
    assert isinstance(buffer_size, int) and buffer_size == 2, f"only support bufferize equals 2"
    semantic_spec.ext_semantic_compile_hint(src, "multi_buffer", buffer_size, _builder)

@builtin
def sync_block_all(mode, event_id, _builder=None):
    mode = _constexpr_to_value(mode)
    event_id = _constexpr_to_value(event_id)
    assert isinstance(mode, str), f"mode: {mode} is not string"
    assert isinstance(event_id, int) and (event_id >= 0) and (event_id < 16), f"event_id: {event_id} should be 0 ~ 15"
    assert mode == "all_cube" or mode == "all_vector" or mode == "all", f"ERROR: mode = {mode}, only supports all_cube/all_vector/all"
    semantic_spec.ext_semantic_custom_op(_builder, "sync_block_all", mode=mode, event_id=event_id)

@builtin
def sync_block_set(sender, receiver, event_id, _builder=None):
    sender = _constexpr_to_value(sender)
    receiver = _constexpr_to_value(receiver)
    event_id = _constexpr_to_value(event_id)
    assert isinstance(sender, str) and (sender == "cube" or sender == "vector"), f"ERROR: sender = {sender}, only supports cube/vector"
    assert isinstance(receiver, str) and (receiver == "cube" or receiver == "vector"), f"ERROR: receiver = {receiver}, only supports cube/vector"
    assert isinstance(event_id, int) and (event_id >= 0) and (event_id < 16), f"event_id: {event_id} should be 0 ~ 15"
    if sender == receiver:
        raise ValueError(f'Unexpected pair: {sender} -> {receiver}, only supports cube -> vector or vector -> cube')
    semantic_spec.ext_semantic_custom_op(_builder, "sync_block_set", sender=sender, event_id=event_id)

@builtin
def sync_block_wait(sender, receiver, event_id, _builder=None):
    sender = _constexpr_to_value(sender)
    receiver = _constexpr_to_value(receiver)
    event_id = _constexpr_to_value(event_id)
    assert isinstance(sender, str) and (sender == "cube" or sender == "vector"), f"ERROR: sender = {sender}, only supports cube/vector"
    assert isinstance(receiver, str) and (receiver == "cube" or receiver == "vector"), f"ERROR: receiver = {receiver}, only supports cube/vector"
    assert isinstance(event_id, int) and (event_id >= 0) and (event_id < 16), f"event_id: {event_id} should be 0 ~ 15"
    if sender == receiver:
        raise ValueError(f'Unexpected pair: {sender} -> {receiver}, only supports cube -> vector or vector -> cube')
    semantic_spec.ext_semantic_custom_op(_builder, "sync_block_wait", sender=sender, event_id=event_id)

@builtin
def load_tensor_descriptor(desc: tensor_descriptor_base, offsets: Sequence[Union[constexpr, tensor]],
                                    _builder=None) -> tensor:
    """Load a block of data from a tensor descriptor."""
    return desc.load(offsets, _builder=_builder)

@builtin
def store_tensor_descriptor(desc: tensor_descriptor_base, offsets: Sequence[Union[constexpr, tensor]], value: tensor,
                                     _builder=None) -> tensor:
    """Store a block of data to a tensor descriptor."""
    return desc.store(offsets, value, _builder=_builder)

@builtin
def make_tensor_descriptor(
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
    return semantic_spec.ext_semantic_make_tensor_descriptor(base, shape, strides, block_shape, _builder)

@builtin
def gather_load(
    src,
    gather_dim,
    gather_indices,
    src_shape,
    src_offset,
    read_shape,
    _builder=None
) -> tensor:
    """
    Parallel gather load operation from Global Memory to Unified Buffer.

    Gathers data from multiple indices along a specified dimension and loads
    them as tiles from GM directly to UB with zero-copy semantics.

    :param src: Source tensor pointer (in GM)
    :type src: tensor (pointer type)
    :param gather_dim: The dimension along which to gather
    :type gather_dim: int or constexpr
    :param gather_indices: 1D tensor of indices to gather (in UB)
    :type gather_indices: tensor
    :param src_shape: Complete shape of the source tensor (can be int or tensor)
    :type src_shape: List[Union[int, tensor]]
    :param src_offset: Starting offset for reading (can be int or tensor)
    :type src_offset: List[Union[int, tensor]]
    :param read_shape: Size to read (tile shape, can be int or tensor)
    :type read_shape: List[Union[int, tensor]]

    **Constraints:**

    - ``read_shape[gather_dim]`` must be ``-1``
    - ``src_offset[gather_dim]`` can be ``-1`` (will be ignored)
    - Boundary handling: ``src_offset + read_shape > src_shape`` automatically
      truncates to ``src_shape`` boundary
    - Does not check if ``gather_indices`` contains out-of-bounds values

    **Example:**

    .. code-block:: python

        @triton.jit
        def kernel(src_ptr, output_ptr, indices_ptr, M, N, D, ...):
            # Load indices (e.g., [5, 10, 15, 20])
            indices = tl.load(indices_ptr + tl.arange(0, 4))

            # Example 1: Static shapes (constants)
            # Gather load from dimension 1
            # src: [8, 100, 256], gather at dim=1
            # Read: [4, ?, 128] starting from [4, ?, 128]
            result = tl.gather_load(
                src_ptr,
                gather_dim=1,
                gather_indices=indices,
                src_shape=[8, 100, 256],
                src_offset=[4, -1, 128],
                read_shape=[4, -1, 128]
            )
            # result shape: [4, 4, 128]

            # Example 2: Dynamic shapes (variables)
            result2 = tl.gather_load(
                src_ptr,
                gather_dim=1,
                gather_indices=indices,
                src_shape=[M, N, D],
                src_offset=[4, -1, 128],
                read_shape=[4, -1, 128]
            )

            tl.store(output_ptr + ..., result)

    :return: Result tensor in UB with shape where ``gather_dim`` is replaced
        by the length of ``gather_indices``
    :rtype: tensor
    """
    gather_dim = _constexpr_to_value(gather_dim)

    # Process shape parameters: convert constexpr to values, keep tensors as-is
    def process_param(val):
        """Convert constexpr to value, keep tensor or int as-is"""
        if isinstance(val, tensor):
            return val
        else:
            return _constexpr_to_value(val)

    newsrc_shape = [
        semantic.to_tensor(o, _builder) if isinstance(o, constexpr) else o
        for o in src_shape
    ]
    newsrc_offset = [
        semantic.to_tensor(o, _builder) if isinstance(o, constexpr) else o
        for o in src_offset
    ]
    assert len(gather_indices.shape) == 1, "gather_indices must be a 1D tensor"

    return semantic_spec.ext_semantic_gather_load(
        src, gather_dim, gather_indices, newsrc_shape, newsrc_offset, read_shape, _builder
    )

def dtype_to_ir(self, builder: ir.builder) -> ir.type:
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


class parallel(range):
    """
    Iterator that counts upward forever, with parallel execution semantics.

    This is a special iterator used to implement similar semantics to Python's :code:`range` in the context of
    :code:`triton.jit` functions. In addition, it allows user to pass extra attributes to the compiler.
    :param bind_sub_block: Tells the compiler if multiple vector cores participate in the loop.
        This is used in the mixed cube-vector kernel on 910B. The number of vector cores is determined by the number of
        iteration in this loop. Currently on 910B, max 2 vector cores could be used.
    """
    def __init__(self, arg1, arg2=None, step=None, num_stages=None, loop_unroll_factor=None, bind_sub_block: bool = False):
        super().__init__(arg1, arg2, step, num_stages, loop_unroll_factor)
        self.bind_sub_block = bind_sub_block

core_ext_spec_func_list = [
    "gather", "insert_slice", "extract_slice", "get_element", "__add__",
    "__radd__", "__sub__", "__rsub__", "__mul__", "__rmul__", "__lshift__",
    "__rshift__", "compile_hint", "sort", "multibuffer", "sync_block_all",
    "sync_block_set", "sync_block_wait", "load_tensor_descriptor",
    "store_tensor_descriptor", "make_tensor_descriptor", "gather_load", "dtype_to_ir", "parallel"
]
