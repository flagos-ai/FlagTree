import numbers
import triton.language as tl
from triton.language.core import (
    _unwrap_if_constexpr,
    _tensor_member_fn,
    _unwrap_iterable,
    builtin,
    constexpr,
    dtype,
    tensor,
    check_bit_width,
    _unwrap_if_constexpr,
)

from typing import Optional, Tuple, List, overload, Union
from triton._C.libtriton import ir

from ._utils import _convert_elem_to_ir_value


@_tensor_member_fn
@builtin
def index_put(ptr: tensor, index: tensor, value: tensor, dim: int, index_boundary: int, end_offset: tuple,
              start_offset: tuple, dst_stride: tuple, _semantic=None):
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
        from triton.language.extra.cann.extension import index_put

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

    def index_put_impl(ptr: tl.tensor, index: tl.tensor, value: tl.tensor, dim: int, index_boundary: int,
                       end_offset: Tuple, start_offset: Tuple, dst_stride: Tuple, _builder: ir.builder):
        assert index.dtype.is_int(), "index must be an integer tensor"
        if not ptr.dtype.element_ty.is_floating():
            raise ValueError(f"Expected dtype fp16/fp32/bf16, but got {ptr.dtype.element_ty}")
        if not isinstance(dim, int):
            raise ValueError("dim must be of type tl.constexpr")

        v_rank = len(value.shape)
        idx_rank = len(index.shape)
        if v_rank < 2 or v_rank > 5:
            raise ValueError(f"value rank must be in [2, 5], got value rank={v_rank}")
        if dim < 0 or dim >= v_rank - 1:
            raise ValueError(f"dim must satisfy 0<=dim<value.rank-1 ({v_rank-1}), got dim={dim}")

        if idx_rank != 1:
            # flatten index to 1D, shape (index.numel,)
            flat_numel = index.numel
            index = _semantic.reshape(index, (flat_numel, ), True)
            idx_rank = 1

        if value.shape[dim] != index.shape[0]:
            raise ValueError(f"index.numel must equal value.shape[dim], "
                             f"but got index.numel={index.numel.value}, value.shape[dim]={value.shape[dim].value}")

        require_i64 = index.dtype.is_int64()
        end_offset = [_convert_elem_to_ir_value(_builder, elem, require_i64) for elem in end_offset]
        start_offset = [_convert_elem_to_ir_value(_builder, elem, require_i64) for elem in start_offset]
        dst_stride = [_convert_elem_to_ir_value(_builder, elem, require_i64) for elem in dst_stride]

        if len(end_offset) != v_rank or len(start_offset) != v_rank or len(dst_stride) != v_rank:
            raise ValueError(f"len(end_offset)==len(start_offset)==len(dst_stride)==value.rank required, "
                             f"got {len(end_offset)}, {len(start_offset)}, {len(dst_stride)}, {v_rank}")

        return tl.tensor(
            _builder.create_index_put(ptr.handle, index.handle, value.handle, dim, index_boundary, end_offset,
                                      start_offset, dst_stride), tl.void)

    dim = _unwrap_if_constexpr(dim)
    index_boundary = _unwrap_if_constexpr(index_boundary)

    return index_put_impl(ptr, index, value, dim, index_boundary, end_offset, start_offset, dst_stride,
                          _semantic.builder)


@_tensor_member_fn
@builtin
def gather_out_to_ub(src: tensor, index: tensor, index_boundary: int, dim: int, src_stride: tuple, end_offset: tuple,
                     start_offset: tuple, other=None, _semantic=None):
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
        from triton.language.extra.cann.extension import gather_out_to_ub

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

    def gather_out_to_ub_impl(src: tl.tensor, index: tl.tensor, index_boundary: int, dim: int, src_stride: Tuple,
                              end_offset: Tuple, start_offset: Tuple, other: Optional[numbers.Number] = None,
                              _builder: ir.builder = None):
        assert index.dtype.is_int(), "index must be an integer tensor"
        if not src.dtype.element_ty.is_floating():
            raise ValueError(f"Expected dtype fp16/fp32/bf16, but got {src.dtype.element_ty}")

        if not isinstance(index_boundary, int):
            raise ValueError("index_boundary must be of type tl.constexpr")
        if not isinstance(dim, int):
            raise ValueError("dim must be of type tl.constexpr")

        idx_rank = len(index.shape)
        if idx_rank < 1 or idx_rank > 5:
            raise ValueError(f"index rank must be in [1, 5], got rank={idx_rank}")
        if dim < 0 or dim >= idx_rank:
            raise ValueError(f"dim must satisfy 0<=dim<index.rank ({idx_rank}), got dim={dim}")

        if other is not None:
            other = _semantic.cast(other, src.dtype.element_ty)

        # src stride need to be i64
        src_stride = [_convert_elem_to_ir_value(_builder, elem, True) for elem in src_stride]
        # end offset and start offset need to be i32
        end_offset = [_convert_elem_to_ir_value(_builder, elem, False) for elem in end_offset]
        start_offset = [_convert_elem_to_ir_value(_builder, elem, False) for elem in start_offset]

        if len(src_stride) != idx_rank or len(end_offset) != idx_rank or len(start_offset) != idx_rank:
            raise ValueError(f"len(src_stride)==len(end_offset)==len(start_offset)==index.rank required, "
                             f"got {len(src_stride)}, {len(end_offset)}, {len(start_offset)}, {idx_rank}")

        ret = _builder.create_gather_out_to_ub(src.handle, index.handle, index_boundary, dim, src_stride, end_offset,
                                               start_offset, other if other else None)
        ret_shape = [_unwrap_if_constexpr(s) for s in index.shape]
        return _semantic.wrap_tensor(ret, src.dtype.element_ty, ret_shape)

    dim = _unwrap_if_constexpr(dim)
    index_boundary = _unwrap_if_constexpr(index_boundary)
    return gather_out_to_ub_impl(src, index, index_boundary, dim, src_stride, end_offset, start_offset, other,
                                 _semantic.builder)


@_tensor_member_fn
@builtin
def scatter_ub_to_out(ptr: tensor, value: tensor, index: tensor, index_boundary: int, dim: int, dst_stride: tuple,
                      end_offset: tuple, start_offset: tuple, _semantic=None):
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
        from triton.language.extra.cann.extension import scatter_ub_to_out

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

    def scatter_ub_to_out_impl(ptr: tl.tensor, value: tl.tensor, index: tl.tensor, index_boundary: int, dim: int,
                               dst_stride: tuple, end_offset: tuple, start_offset: tuple, _builder=None):
        assert index.dtype.is_int(), "index must be an integer tensor"
        if not ptr.dtype.element_ty.is_floating():
            raise ValueError(f"Expected dtype fp16/fp32/bf16, but got {ptr.dtype.element_ty}")

        if not isinstance(index_boundary, int):
            raise ValueError("index_boundary must be of type tl.constexpr")
        if not isinstance(dim, int):
            raise ValueError("dim must be of type tl.constexpr")

        idx_rank = len(index.shape)
        if idx_rank < 1 or idx_rank > 5:
            raise ValueError(f"index rank must be in [1, 5], got rank={idx_rank}")
        if dim < 0 or dim >= idx_rank:
            raise ValueError(f"dim must satisfy 0<=dim<index.rank (index.rank={idx_rank}), got dim={dim}")

        # dst stride are always i64
        dst_stride = [_convert_elem_to_ir_value(_builder, elem, True) for elem in dst_stride]
        # end offset and start offset need to be i32
        end_offset = [_convert_elem_to_ir_value(_builder, elem, False) for elem in end_offset]
        start_offset = [_convert_elem_to_ir_value(_builder, elem, False) for elem in start_offset]

        if len(dst_stride) != idx_rank or len(end_offset) != idx_rank or len(start_offset) != idx_rank:
            raise ValueError(f"len(dst_stride)==len(end_offset)==len(start_offset)==index.rank required, "
                             f"got {len(dst_stride)}, {len(end_offset)}, {len(start_offset)}, {idx_rank}")

        return tl.tensor(
            _builder.create_scatter_ub_to_out(ptr.handle, value.handle, index.handle, index_boundary, dim, dst_stride,
                                              end_offset, start_offset), tl.void)

    def _is_ranked_tensor(x):
        return isinstance(x, tensor) and x.shape and len(x.shape) > 0

    dim = _unwrap_if_constexpr(dim)
    index_boundary = _unwrap_if_constexpr(index_boundary)
    value = _unwrap_if_constexpr(value)

    if not _is_ranked_tensor(value) or isinstance(value, constexpr):
        element_ty = ptr.type.scalar.element_ty
        value = _semantic.full(index.shape, value, element_ty)
    return scatter_ub_to_out_impl(ptr, value, index, index_boundary, dim, dst_stride, end_offset, start_offset,
                                  _semantic.builder)


@_tensor_member_fn
@builtin
def index_select_simd(src, dim, index, src_shape, src_offset, read_shape, _semantic=None) -> tensor:
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
            result = extension.index_select_simd(
                src_ptr,
                dim=1,
                index=indices,
                src_shape=[8, 100, 256],
                src_offset=[4, -1, 128],
                read_shape=[4, -1, 128]
            )
            # result shape: [4, 4, 128]

            # Example 2: Dynamic shapes (variables)
            result2 = extension.index_select_simd(
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

    def index_select_simd_impl(src: tl.tensor, dim: int, index: tl.tensor, src_shape: List[Union[int, tl.tensor]],
                               src_offset: List[Union[int, tl.tensor]], read_shape: List[Union[int, tl.tensor]],
                               _builder: ir.builder) -> tl.tensor:
        # Validate inputs
        ndim = len(src_shape)
        assert len(src_offset) == ndim, \
            f"src_offset length {len(src_offset)} must match src_shape length {ndim}"
        assert len(read_shape) == ndim, \
            f"read_shape length {len(read_shape)} must match src_shape length {ndim}"
        assert 0 <= dim < ndim, \
            f"dim={dim} must be in range [0, {ndim})"
        assert len(index.shape) == 1, \
            f"index must be 1D tensor, got {len(index.shape)}D"
        assert dim < ndim - 1, \
            f"index_select_simd cannot support trailing dimension as dim={dim}, ndim={ndim}"
        # Handle both tensor and int offsets (for interpreter mode)
        newsrc_shape = []
        for s in src_shape:
            if isinstance(s, tensor):
                newsrc_shape.append(s.handle)
            elif isinstance(s, int):
                # For interpreter mode: keep as int
                newsrc_shape.append(s)
            else:
                newsrc_shape.append(s.handle if hasattr(s, 'handle') else s)
        newsrc_offset = []
        for s in src_offset:
            if isinstance(s, tensor):
                newsrc_offset.append(s.handle)
            elif isinstance(s, int):
                # For interpreter mode: keep as int
                newsrc_offset.append(s)
            else:
                newsrc_offset.append(s.handle if hasattr(s, 'handle') else s)

        # Create output type
        return_shape = [index.shape[0] if i == dim else read_shape[i] for i in range(ndim)]
        element_ty = src.type.element_ty
        output_ty = tl.block_type(element_ty, return_shape)
        out = _builder.create_index_select_simd(src.handle, index.handle, dim, newsrc_shape, newsrc_offset, read_shape,
                                                return_shape)
        return tl.tensor(out, output_ty)

    dim = _unwrap_if_constexpr(dim)

    # Process shape parameters: convert constexpr to values, keep tensors as-is
    def process_param(val):
        """Convert constexpr to value, keep tensor or int as-is"""
        if isinstance(val, tensor):
            return val
        else:
            return _unwrap_if_constexpr(val)

    newsrc_shape = [_semantic.to_tensor(o) if isinstance(o, constexpr) else o for o in src_shape]
    newsrc_offset = [_semantic.to_tensor(o) if isinstance(o, constexpr) else o for o in src_offset]
    assert len(index.shape) == 1, "index must be a 1D tensor"

    return index_select_simd_impl(src, dim, index, newsrc_shape, newsrc_offset, read_shape, _semantic.builder)
