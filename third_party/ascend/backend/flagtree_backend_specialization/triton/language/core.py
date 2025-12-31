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
    float32,
    check_bit_width,
    _unwrap_if_constexpr,
    add,
    sub,
    mul,
)

from .tensor_descriptor import tensor_descriptor, tensor_descriptor_base

try:
    import acl
    is_compile_on_910_95 = acl.get_soc_name().startswith("Ascend910_95")
except Exception as e:
    is_compile_on_910_95 = False

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
def __mod__(self, other, _builder=None):
    other = _unwrap_if_constexpr(other)
    return semantic.mod(self, other, _builder)

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
def flip(ptr, dim=-1, _builder=None, _generator=None):
    try:
        dim = int(dim.value) if hasattr(dim, "value") else int(dim)
    except Exception as e:
        raise TypeError(f"dim must be an integer (or tl.constexpr int), got {dim!r}") from e

    dim = len(ptr.shape) - 1 if dim == -1 else dim
    return semantic_spec.ext_semantic_flip(ptr, dim, _builder, _generator)

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
def index_select(src: tensor, idx: tensor, bound, lstdim_blksiz, offsets, numels, _builder=None):
    """
    Embedding
    :src_ptr:
    :idx:
    """
    bound = _constexpr_to_value(bound)
    lstdim_blksiz = _constexpr_to_value(lstdim_blksiz)
    return semantic_spec.ext_semantic_embedding_gather(src, idx, bound, lstdim_blksiz, offsets, numels, _builder)

@builtin
def index_put(
    ptr: tensor,
    index: tensor,
    value: tensor,
    dim: int,
    index_boundary: int,
    end_offset: tuple,
    start_offset: tuple,
    dst_stride: tuple,
    _builder=None
):
    """
    Index put values from a tensor into a destination tensor.

    Index put operation for different tensor ranks:
    1. 2D index scatter (0 <= dim < 1):
        1.1 dim = 0
        out[index[i]][start_offset[1]:end_offset[1]] = value[i][0:end_offset[1]-start_offset[1]]
    2. 3D index scatter (0 <= dim < 2):
        2.1 dim = 0
            out[index[i]][start_offset[1]:end_offset[1]][start_offset[2]:end_offset[2]]
                = value[i][0:end_offset[1]-start_offset[1]][0:end_offset[2]-start_offset[2]]
        2.2 dim = 1
            out[start_offset[0]:end_offset[0]][index[j]][start_offset[2]:end_offset[2]]
                = value[0:end_offset[0]-start_offset[0]][j][0:end_offset[2]-start_offset[2]]


    :param ptr: pointer type, the destination tensor pointer (in GM)
    :param index: tensor, a index to scatter (in UB)
    :param value: tensor, a value to store (in UB)
    :param dim: int32, the dimension to scatter along
    :param index_boundary: int64, the upper boundary for index values
    :param end_offset: tuple of int, the offsets of each dimension for the end of the scatter region
    :param start_offset: tuple of int, the offsets of each dimension for the start of the scatter region
    :param dst_stride: tuple of int, the stride of each dimension of destination tensor

    Constraints
    ***********
    - `ptr` and `value` must have the same rank.
    - `ptr.dtype` only supports `float16`, `bfloat16`, `float32` currently.
    - `index` must be an integer tensor. If `index.rank` != 1, it will be reshaped to 1D.
    - `index.numel` must equal `value.shape[dim]`.
    - `value` support 2~5D tensors.
    - `dim` must be valid (0 <= dim < rank(value) - 1).

    Example
    *******
    .. code-block:: python

        import torch
        import triton
        import triton.language as tl
        from triton.language.extra.ascend.libdevice import index_put

        @triton.jit
        def simple_index_put_kernel(value_ptr, index_ptr, dst_ptr):
            # index tile shape: [2]
            index_local = tl.arange(0, 2)
            x1_local = tl.arange(0, 2)[None, :]  # shape=(1,2)

            index_tile = tl.load(index_ptr + index_local)
            value_tile = tl.load(value_ptr + index_local[:, None]*2 + x1_local)

            index_put(
                ptr=dst_ptr,
                index=index_tile,
                value=value_tile,
                dim=0,
                index_boundary=4,
                end_offset=(2, 2),
                start_offset=(0, 0),
                dst_stride=(2, 1)
            )

        dst = torch.zeros((4,2), device='npu', dtype=torch.float32)
        value = torch.tensor([[1.,2.], [3.,4.]], device='npu')
        index = torch.tensor([2, 0], device='npu')

        simple_index_put_kernel[(1,)](value, index, dst)
        print("IndexPut result:", dst) # ref:[[3.,4.], [0.,0.], [1.,2.], [0.,0.]]
    """
    dim = _constexpr_to_value(dim)
    index_boundary = _constexpr_to_value(index_boundary)
    return semantic_spec.ext_semantic_index_put(ptr, index, value, dim, index_boundary,
                              end_offset, start_offset, dst_stride, _builder)

@builtin
def gather_out_to_ub(
    src: tensor,
    index: tensor,
    index_boundary: int,
    dim: int,
    src_stride: tuple,
    end_offset: tuple,
    start_offset: tuple,
    other=None,
    _builder=None
):
    """
    Gather from a source tensor in Global Memory (GM) to Unified Buffer (UB)
    along a specified dimension with out-of-bound handling.

    Gather operation for different tensor ranks:
    1. 1D index gather:
        out[i] = src[start_offset[0] + index[i]]
    2. 2D index gather (0 <= dim < 2):
        2.1 dim = 0
            out[i][j] = src[start_offset[0] + index[i][j]][start_offset[1] + j]
        2.2 dim = 1
            out[i][j] = src[start_offset[0] + i][start_offset[1] + index[i][j]]
    3. 3D index gather (0 <= dim < 3):
        3.1 dim = 0
            out[i][j][k] = src[start_offset[0] + index[i][j][k]][start_offset[1] + j][start_offset[2] + k]
        3.2 dim = 1
            out[i][j][k] = src[start_offset[0] + i][start_offset[1] + index[i][j][k]][start_offset[2] + k]
        3.3 dim = 2
            out[i][j][k] = src[start_offset[0] + i][start_offset[1] + j][start_offset[2] + index[i][j][k]]

    :param src: pointer type, the source tensor pointer (in GM)
    :param index: tensor, a tensor to gather (in UB)
    :param index_boundary: int64, the upper boundary for index values
    :param dim: int32, the dimension to gather along
    :param src_stride: tuple of int64, the stride of each dimension of src tensor
    :param end_offset: tuple of int32, the end offsets of each dimension for index tensor
    :param start_offset: tuple of int32, the start offsets of each dimension for index tensor
    :param other(Optional): scalar value, the default value when index is out of boundary (in UB)
    :return: tensor, with the same shape as `index.shape` (in UB)

    Constraints
    ***********
    - `src` and `index` must have the same rank.
    - `src.dtype` only supports `float16`, `bfloat16`, `float32` currently.
    - `index` must be an integer tensor, with rank between 1 and 5.
    - `dim` must be valid (0 <= dim < rank(index)).
    - `other` must be a scalar value.
    - For every dimension `i` not equal to `dim`, `index.size[i]` <= `src.size[i]`.
    - The output shape is the same as `index.shape`. If `index` is None, \
        the output tensor will be an empty tensor with the same shape as `index`.

    Example
    *******
    .. code-block:: python

        import torch
        import triton
        import triton.language as tl
        from triton.language.extra.ascend.libdevice import gather_out_to_ub

        @triton.jit
        def simple_gather_kernel(src_ptr, index_ptr, out_ptr):
            # index tile shape: [2,2]
            y0_local = tl.arange(0, 2)[:, None]  # [0,1] rows
            x1_local = tl.arange(0, 2)[None, :]  # [0,1] cols
            mask = (y0_local < 2) & (x1_local < 2)

            # Load index tile to UB
            index = tl.load(index_ptr + y0_local*2 + x1_local, mask)

            # Call gather_out_to_ub: gather values from src along dim=0
            gathered = gather_out_to_ub(
                src=src_ptr,
                index=index,
                index_boundary=4,
                dim=0,
                src_stride=(2, 1),
                end_offset=(2, 2),
                start_offset=(0, 0)
            )

            tl.store(out_ptr + y0_local*2 + x1_local, gathered, mask)

        src = torch.tensor([[1.,2.], [3.,4.], [5.,6.], [7.,8.]], device='npu')
        index = torch.tensor([[0,1], [2,3]], device='npu')
        out = torch.empty((2,2), device='npu', dtype=torch.float32)

        simple_gather_kernel[(1,)](src, index, out)
        print("Gather result:", out)  # ref: [[1.,4.], [5.,8.]]
    """
    dim = _constexpr_to_value(dim)
    index_boundary = _constexpr_to_value(index_boundary)
    return semantic_spec.ext_semantic_gather_out_to_ub(
        src, index, index_boundary, dim,
        src_stride, end_offset, start_offset, other, _builder
    )

@builtin
def scatter_ub_to_out(
    ptr: tensor,
    value: tensor,
    index: tensor,
    index_boundary: int,
    dim: int,
    dst_stride: tuple,
    end_offset: tuple,
    start_offset: tuple,
    _builder=None
):
    """
    Scatter a tile from Unified Buffer (UB) into a destination tensor in Global Memory (GM)
    along a specified dimension, with index-boundary checking.

    Scatter operation for different tensor ranks:
    1. 1D index scatter:
        out[start_offset[0] + index[i]] = value[i]
    2. 2D index scatter (0 <= dim < 2):
        2.1 dim = 0
            out[start_offset[0] + index[i][j]][start_offset[1] + j] = value[i][j]
        2.2 dim = 1
            out[start_offset[0] + i][start_offset[1] + index[i][j]] = value[i][j]
    3. 3D index scatter (0 <= dim < 3):
        3.1 dim = 0
            out[start_offset[0] + index[i][j][k]][start_offset[1] + j][start_offset[2] + k] = value[i][j][k]
        3.2 dim = 1
            out[start_offset[0] + i][start_offset[1] + index[i][j][k]][start_offset[2] + k] = value[i][j][k]
        3.3 dim = 2
            out[start_offset[0] + i][start_offset[1] + j][start_offset[2] + index[i][j][k]] = value[i][j][k]

    :param ptr: pointer type, the destination tensor pointer (in GM)
    :param value: tensor, a tile value to store (in UB)
    :param index: tensor, a index to scatter (in UB)
    :param index_boundary: int64, the upper boundary for index values
    :param dim: int32, the dimension to scatter along
    :param dst_stride: tuple of int64, the stride of each dimension of destination tensor
    :param end_offset: tuple of int32, the end offsets of each dimension for index tensor
    :param start_offset: tuple of int32, the start offsets of each dimension for index tensor

    Constraints
    ***********
    - `ptr`, `index` and `value` must have the same rank.
    - `ptr.dtype` only supports `float16`, `bfloat16`, `float32` currently.
    - `index` must be an integer tensor, with rank between 1 and 5.
    - `dim` must be valid (0 <= dim < rank(index)).
    - For every dimension `i` not equal to `dim`, `index.size[i]` <= `ptr.size[i]`.
    - The output shape is the same as `index.shape`. If `index` is None, \
        the output tensor will be an empty tensor with the same shape as `index`.

    Example
    *******
    .. code-block:: python

        import torch
        import triton
        import triton.language as tl
        from triton.language.extra.ascend.libdevice import scatter_ub_to_out

        @triton.jit
        def simple_scatter_kernel(value_ptr, index_ptr, dst_ptr):
            # index tile shape: [2,2]
            y0_local = tl.arange(0, 2)[:, None]  # [0,1] rows
            x1_local = tl.arange(0, 2)[None, :]  # [0,1] cols
            mask = (y0_local < 2) & (x1_local < 2)

            value = tl.load(value_ptr + y0_local*2 + x1_local, mask)
            index = tl.load(index_ptr + y0_local*2 + x1_local, mask)

            scatter_ub_to_out(
                ptr=dst_ptr,
                value=value,
                index=index,
                index_boundary=4,
                dim=0,
                dst_stride=(2, 1),
                end_offset=(2, 2),
                start_offset=(0, 0)
            )

        dst = torch.zeros((4,2), device='npu', dtype=torch.float32)
        value = torch.tensor([[1.,2.], [3.,4.]], device='npu')
        index = torch.tensor([[1,2], [3,0]], device='npu')

        simple_scatter_kernel[(1,)](value, index, dst)
        print("Scatter result:", dst)  # ref:[[0.,4.], [1.,0.], [0.,2.], [3.,0.]]
    """
    dim = _constexpr_to_value(dim)
    index_boundary = _constexpr_to_value(index_boundary)
    return semantic_spec.ext_semantic_scatter_ub_to_out(
        ptr, value, index, index_boundary, dim,
        dst_stride, end_offset, start_offset, _builder
    )

@builtin
def index_select_simd(
    src,
    dim,
    index,
    src_shape,
    src_offset,
    read_shape,
    _builder=None
) -> tensor:
    """
    Parallel index_select operation from Global Memory to Unified Buffer (SIMD version).

    Selects data from multiple indices along a specified dimension and loads
    them as tiles from GM directly to UB with zero-copy semantics.

    :param src: Source tensor pointer (in GM)
    :type src: tensor (pointer type)
    :param dim: The dimension along which to select indices
    :type dim: int or constexpr
    :param index: 1D tensor of indices to select (in UB)
    :type index: tensor
    :param src_shape: Complete shape of the source tensor (can be int or tensor)
    :type src_shape: List[Union[int, tensor]]
    :param src_offset: Starting offset for reading (can be int or tensor)
    :type src_offset: List[Union[int, tensor]]
    :param read_shape: Size to read (tile shape, can be int or tensor)
    :type read_shape: List[Union[int, tensor]]

    **Constraints:**

    - ``read_shape[dim]`` must be ``-1``
    - ``src_offset[dim]`` can be ``-1`` (will be ignored)
    - Boundary handling: ``src_offset + read_shape > src_shape`` automatically
      truncates to ``src_shape`` boundary
    - Does not check if ``index`` contains out-of-bounds values

    **Example:**

    .. code-block:: python

        @triton.jit
        def kernel(src_ptr, output_ptr, indices_ptr, M, N, D, ...):
            # Load indices (e.g., [5, 10, 15, 20])
            indices = tl.load(indices_ptr + tl.arange(0, 4))

            # Example 1: Static shapes (constants)
            # Index select from dimension 1
            # src: [8, 100, 256], index_select at dim=1
            # Read: [4, ?, 128] starting from [4, ?, 128]
            result = libdevice.index_select_simd(
                src_ptr,
                dim=1,
                index=indices,
                src_shape=[8, 100, 256],
                src_offset=[4, -1, 128],
                read_shape=[4, -1, 128]
            )
            # result shape: [4, 4, 128]

            # Example 2: Dynamic shapes (variables)
            result2 = libdevice.index_select_simd(
                src_ptr,
                dim=1,
                index=indices,
                src_shape=[M, N, D],
                src_offset=[4, -1, 128],
                read_shape=[4, -1, 128]
            )

            tl.store(output_ptr + ..., result)

    :return: Result tensor in UB with shape where ``dim`` is replaced
        by the length of ``index``
    :rtype: tensor
    """
    dim = _constexpr_to_value(dim)

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
    assert len(index.shape) == 1, "index must be a 1D tensor"

    return semantic_spec.ext_semantic_index_select_simd(
        src, dim, index, newsrc_shape, newsrc_offset, read_shape, _builder
    )

def dtype_to_ir(self, builder: ir.builder) -> ir.type:
    if not is_compile_on_910_95:
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

@builtin
def dot_scaled(lhs, lhs_scale, lhs_format, rhs, rhs_scale, rhs_format, acc=None, out_dtype=float32, lhs_k_pack=True, rhs_k_pack=True, _builder=None):
    """
    Returns the matrix product of two blocks in microscaling format.
    lhs and rhs use microscaling formats described here:
    https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf
    :param lhs: The first tensor to be multiplied.
    :type lhs: 2D tensor of f8, f6 or f4 format packed in int32 format.
    :param lhs_scale: Scale factor for lhs tensor.
    :type lhs_scale: ue8m0 float8 type (currently represented as an int8 tensor).
    :param lhs_format: format of the lhs tensor, available formats: {:code:`e4m3`, :code: `e5m2`, :code:`e2m3`, :code:`e3m2`, :code:`e2m1`}.
    :param rhs: The second tensor to be multiplied.
    :type rhs: 2D tensor of f8, f6 or f4 format packed in int32 format.
    :param rhs_scale: Scale factor for rhs tensor.
    :type rhs_scale: ue8m0 float8 type (currently represented as an int8 tensor).
    :param rhs_format: format of the rhs tensor, available formats: {:code:`e4m3`, :code: `e5m2`, :code:`e2m3`, :code:`e3m2`, :code:`e2m1`}.
    :param acc: The accumulator tensor. If not None, the result is added to this tensor.
    """
    out_dtype = _constexpr_to_value(out_dtype)
    assert out_dtype == float32, "Only float32 is supported for out_dtype at the moment"
    return semantic.dot_scaled(lhs, lhs_scale, lhs_format, rhs, rhs_scale, rhs_format, acc, out_dtype, lhs_k_pack, rhs_k_pack, _builder)

class range():
    """
    Iterator that counts upward forever.

    .. highlight:: python
    .. code-block:: python

        @triton.jit
        def kernel(...):
            for i in tl.range(10, num_stages=3):
                ...
    :note: This is a special iterator used to implement similar semantics to Python's :code:`range` in the context of
        :code:`triton.jit` functions. In addition, it allows user to pass extra attributes to the compiler.
    :param arg1: the start value.
    :param arg2: the end value.
    :param step: the step value.
    :param num_stages: pipeline the loop into this many stages (so there are
        :code:`num_stages` iterations of the loop in flight at once).

        Note this is subtly different than passing :code:`num_stages` as a
        kernel argument.  The kernel argument only pipelines loads that feed
        into :code:`dot` operations, while this attribute tries to pipeline most
        (though not all) loads in this loop.
    :param loop_unroll_factor: Tells the Triton IR level loop unroller how many
        times to unroll a for loop that this range is used with. Less than 2 for
        this value implies no unrolling.
    :param disallow_acc_multi_buffer: If true, prevent the accumulator of the dot
        operation in the loop to be multi-buffered, if applicable.
    :param flatten: automatically flatten the loop nest starting at this loop to
        create a single flattened loop. The compiler will try to pipeline the
        flattened loop which can avoid stage stalling.
    :param warp_specialize: Enable automatic warp specialization on the loop.
        The compiler will attempt to partition memory, MMA, and vector
        operations in the loop into separate async partitions. This will
        increase the total number of warps required by the kernel.
    :param disable_licm: Tells the compiler it shouldn't hoist loop invariant
        code outside the loop. This is often useful to avoid creating long liveranges
        within a loop.

        Note that warp specialization is only supported on Blackwell GPUs and
        only works on simple matmul loops. Support for arbitrary loops will be
        expanded over time.
    """

    def __init__(self, arg1, arg2=None, step=None, num_stages=None, loop_unroll_factor=None,
                 disallow_acc_multi_buffer=False, flatten=False, warp_specialize=False, disable_licm=False):
        if step is None:
            self.step = constexpr(1)
        else:
            self.step = step
        if arg2 is None:
            self.start = constexpr(0)
            self.end = arg1
        else:
            self.start = arg1
            self.end = arg2
        self.num_stages = num_stages
        self.loop_unroll_factor = loop_unroll_factor
        self.disallow_acc_multi_buffer = disallow_acc_multi_buffer
        self.flatten = flatten
        self.warp_specialize = warp_specialize
        self.disable_licm = disable_licm

    def __iter__(self):
        raise RuntimeError("tl.range can only be used in @triton.jit'd functions")

    def __next__(self):
        raise RuntimeError("tl.range can only be used in @triton.jit'd functions")

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
    "__radd__", "__sub__", "__rsub__", "__mul__", "__rmul__", "__mod__", "__lshift__",
    "__rshift__", "compile_hint", "sort", "multibuffer", "sync_block_all",
    "sync_block_set", "sync_block_wait", "load_tensor_descriptor",
    "store_tensor_descriptor", "make_tensor_descriptor", "index_select_simd", "dtype_to_ir", "parallel",
    "index_select", "index_put", "gather_out_to_ub", "scatter_ub_to_out", "dot_scaled", "range"
]
