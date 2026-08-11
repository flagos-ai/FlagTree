# Copyright 2026- Xcoresigma Technology Co., Ltd

import triton.language.core as tl
import triton.language as dl
import triton
from triton.language.core import builtin, tensor
from typing import List

_SHMEM_DTYPE_MAP = {
    tl.float16: "half",
    tl.float32: "float",
    tl.bfloat16: "bfloat16",
    tl.int8: "int8_t",
    tl.int16: "int16_t",
    tl.int32: "int32_t",
    tl.int64: "int64_t",
    tl.uint8: "uint8_t",
    tl.uint16: "uint16_t",
    tl.uint32: "uint32_t",
    tl.uint64: "uint64_t",
}


def _shmem_dtype(dtype):
    if hasattr(dtype, 'element_ty'):
        return _SHMEM_DTYPE_MAP.get(dtype.element_ty, str(dtype.element_ty))
    return _SHMEM_DTYPE_MAP.get(dtype, str(dtype))


def _scalar_dtype(t):
    d = t.dtype if hasattr(t, 'dtype') else t
    return d.element_ty if hasattr(d, 'element_ty') else d


def _dispatch(
    func,
    lib_name: str,
    lib_path: str,
    args: list,
    arg_type_symbol_dict: dict,
    is_pure: bool,
    _semantic=None,
):
    if len(arg_type_symbol_dict) == 0:
        raise ValueError("arg_type_symbol_dict is empty")

    num_args = len(list(arg_type_symbol_dict.keys())[0])
    if len(args) != num_args:
        raise ValueError(f"length of input args does not match."
                         f"Expect {len(args)}, got {num_args}")

    arg_types = []
    arg_list = []
    for arg in args:
        if isinstance(arg, tensor):
            dtype = arg.dtype
            if hasattr(dtype, 'element_ty'):
                dtype = dtype.element_ty
            arg_types.append(dtype)
            arg_list.append(arg.handle)
        else:
            arg_types.append(type(arg))
            arg_list.append(arg)
    arg_types = tuple(arg_types)

    if arg_types not in arg_type_symbol_dict:
        raise ValueError(f"input arg type does not match."
                         f"Expect one of {arg_type_symbol_dict.keys()}, got {arg_types}")
    else:
        symbol = arg_type_symbol_dict[arg_types][0]
        ret_types = arg_type_symbol_dict[arg_types][1]
        if not isinstance(ret_types, (List, tuple)):
            ret_types = [ret_types]

        if symbol == "":
            raise ValueError("Symbol can not be empty")
        call = func(
            lib_name,
            lib_path,
            symbol,
            arg_list,
            [ret_type.to_ir(_semantic.builder) for ret_type in ret_types],
            is_pure,
        )

        if len(ret_types) == 0:
            return tensor(call, tl.void)
        if len(ret_types) == 1:
            return tensor(call.get_result(0), ret_types[0])
        return tuple(tensor(call.get_result(i), ty) for i, ty in enumerate(ret_types))


def _extern_call(
    lib_name: str,
    lib_path: str,
    args: list,
    arg_type_symbol_dict: dict,
    is_pure: bool,
    _semantic=None,
):
    """Dispatch an external function call. Supports ptr/block arguments."""
    dispatch_args = args.copy()
    for i in range(len(dispatch_args)):
        dispatch_args[i] = _semantic.to_tensor(dispatch_args[i])

    if len(arg_type_symbol_dict) == 0:
        raise ValueError("arg_type_symbol_dict is empty")

    num_args = len(list(arg_type_symbol_dict.keys())[0])
    if len(args) != num_args:
        raise ValueError(f"length of input args does not match."
                         f"Expect {len(args)}, got {num_args}")

    func = _semantic.builder.create_extern_call
    return _dispatch(
        func,
        lib_name,
        lib_path,
        dispatch_args,
        arg_type_symbol_dict,
        is_pure,
        _semantic,
    )


class MeshConfig:
    """Lightweight mesh configuration for Ascend.

    Only ``device`` level is implemented; ``node``, ``block_cluster``,
    ``block`` accept values but trigger assert at usage time.
    """

    def __init__(self, device=None, node=None, block_cluster=None, block=None):
        self.device = device
        self.node = node
        self.block_cluster = block_cluster
        self.block = block

    def __repr__(self):
        return (f"MeshConfig(device={self.device}, node={self.node}, "
                f"block_cluster={self.block_cluster}, block={self.block})")


class device_mesh:
    """Lightweight logical view of physical device topology."""

    def __init__(self, topology: MeshConfig):
        if not isinstance(topology, MeshConfig):
            raise TypeError(f"topology must be MeshConfig, got {type(topology).__name__}")
        self._mesh_config = topology
        # Only device level is implemented
        assert topology.node is None, "TODO: node level not yet implemented"
        assert topology.block_cluster is None, "TODO: block_cluster level not yet implemented"
        assert topology.block is None, "TODO: block level not yet implemented"
        assert topology.device is not None and topology.device > 0, "device count must be > 0"

    @property
    def device_count(self) -> int:
        return self._mesh_config.device

    @property
    def shape(self) -> tuple:
        return (self.device_count, )

    def __repr__(self):
        return f"DeviceMesh(device={self.device_count})"


@builtin
def remote(
    tensor,
    shard_id=None,
    scope=None,
    space: str = None,
    dtype: tl.dtype = None,
    offset: int = None,
    _semantic=None,
):
    """Distributed pointer shuffle: remote pointer access by peer rank.

    Signature aligned with distributed.py::remote().
    Currently only ``device`` level is supported.
    """
    scope = tl._unwrap_if_constexpr(scope)
    if scope is not None:
        assert isinstance(scope, device_mesh), f"scope must be device_mesh, got {type(scope).__name__}"

    space = tl._unwrap_if_constexpr(space)
    if space is not None and not isinstance(space, str):
        raise TypeError(f"space must be str or None, got {type(space).__name__}")
    assert space is None or space == "device", \
        f"TODO: space='{space}' not yet implemented (only 'device' is supported)"

    dtype = tl._unwrap_if_constexpr(dtype)
    if dtype is not None:
        assert dtype in _SHMEM_DTYPE_MAP, \
            f"dtype {dtype} is not in the supported SHMEM type map: {list(_SHMEM_DTYPE_MAP.keys())}"

    offset = tl._unwrap_if_constexpr(offset)
    if offset is not None and not isinstance(offset, (int, tl.tensor)):
        raise TypeError(f"offset must be int, tl.tensor, or None, got {type(offset).__name__}")

    assert not tensor.type.is_block() and tensor.type.is_ptr(), "only support scalar pointer"
    rank = _semantic._convert_elem_to_ir_value(shard_id, require_i64=False)
    remote_ptr = tl.tensor(_semantic.builder.create_symm_at(tensor.handle, rank), tensor.type)

    # offset is accepted for interface compatibility but not yet applied at
    # the HIVM level — the TT addptr is lost during TT→HIVM bufferization.
    # Callers should use `remote_ptr + offset` at the call site instead.
    assert offset is None, "offset not applied at the HIVM level"
    # TODO: wire offset into symm_at / HIVM custom op

    return remote_ptr


@builtin
def shard_id(
    mesh=None,
    axis=-1,
    device_dptr=None,
    _semantic=None,
):
    """Return current shard coordinate.

    Signature aligned with distributed.py::shard_id().
    Currently only ``device`` axis is supported.
    """
    mesh = tl._unwrap_if_constexpr(mesh)
    if mesh is not None:
        assert isinstance(mesh, device_mesh), f"mesh must be device_mesh, got {type(mesh).__name__}"

    axis = tl._unwrap_if_constexpr(axis)
    if isinstance(axis, str):
        assert axis == "device", f"TODO: axis='{axis}' not yet implemented (only 'device' is supported)"
        axis = 0
        if device_dptr is not None:
            assert isinstance(device_dptr, tl.tensor) and device_dptr.dtype.is_ptr(), \
                "device_dptr must be a pointer tensor"
    elif not isinstance(axis, int):
        raise TypeError(f"axis must be int or str, got {type(axis).__name__}")
    else:
        assert axis == -1 or axis == 0, f"TODO: axis={axis} not yet implemented (only 'device'/0/-1)"

    axis = _semantic._convert_elem_to_ir_value(axis, require_i64=False)
    return tl.tensor(_semantic.builder.create_get_rank(axis), tl.int32)


@builtin
def distributed_barrier(
    mesh=None,
    device_dptr=None,
    space: str = None,
    group_kind=None,
    barrier_kind=None,
    order=None,
    index: int = None,
    _semantic=None,
):
    """
    Distributed barrier across all ranks.
    Currently only ``device`` level is supported.
    """
    mesh = tl._unwrap_if_constexpr(mesh)
    if mesh is not None:
        assert isinstance(mesh, device_mesh), f"mesh must be device_mesh, got {type(mesh).__name__}"

    space = tl._unwrap_if_constexpr(space)
    if space is not None and not isinstance(space, str):
        raise TypeError(f"space must be str or None, got {type(space).__name__}")
    assert space is None or space == "device", \
        f"TODO: space='{space}' not yet implemented (only 'device' is supported)"

    index = tl._unwrap_if_constexpr(index)
    if index is not None and not isinstance(index, int):
        raise TypeError(f"index must be int or None, got {type(index).__name__}")

    if device_dptr is not None:
        assert isinstance(device_dptr, tl.tensor) and device_dptr.dtype.is_ptr(), \
            "device_dptr must be a pointer tensor"

    return _extern_call(
        "libshmem_device",
        "",
        [],
        {
            (): ("aclshmem_barrier_all", ()),
        },
        is_pure=False,
        _semantic=_semantic,
    )


@triton.jit
def swizzle2d_Nz(
    iter_id,
    rank_size,
    data_row_shape,
    data_col_shape,
    tile_row_shape,
    tile_col_shape,
    comm_npu_split=1,
):
    """Ascend Nz-format 2D swizzle for communication tiles."""
    data_row_loop_num = dl.cdiv(data_row_shape, tile_row_shape)
    data_col_loop_num = dl.cdiv(data_col_shape, tile_col_shape)
    data_loop_num = data_row_loop_num * data_col_loop_num
    rank_stride = rank_size // comm_npu_split
    swizzle_offset = comm_npu_split
    rank_loop_num = dl.cdiv(rank_size, swizzle_offset)
    rank_tile_idx = iter_id // (swizzle_offset * data_loop_num)
    data_rank_tile_idx = iter_id % (swizzle_offset * data_loop_num)
    rank_tile_size = swizzle_offset
    if rank_tile_idx == rank_loop_num - 1:
        rank_tile_size = rank_size - swizzle_offset * rank_tile_idx
    data_tile_idx = data_rank_tile_idx // rank_tile_size
    rank_idx = rank_tile_idx * swizzle_offset + data_rank_tile_idx % rank_tile_size
    rank_idx = (rank_idx * rank_stride) % rank_size + (rank_idx * rank_stride) // rank_size
    rank_idx = (rank_idx + data_tile_idx) % rank_size
    data_row_idx = data_tile_idx // data_col_loop_num
    data_col_idx = data_tile_idx % data_col_loop_num
    comm_row_size = dl.minimum(data_row_shape - data_row_idx * tile_row_shape, tile_row_shape)
    comm_col_size = dl.minimum(data_col_shape - data_col_idx * tile_col_shape, tile_col_shape)
    return data_row_idx, data_col_idx, rank_idx, comm_row_size, comm_col_size


@triton.jit
def gemm_swizzle2d_Nz(
    iter_id,
    data_row_shape,
    data_col_shape,
    tile_row_shape,
    tile_col_shape,
    swizzle_offset=7,
):
    """Ascend Nz-format 2D swizzle for GEMM compute tiles."""
    data_row_loop_num = dl.cdiv(data_row_shape, tile_row_shape)
    data_col_loop_num = dl.cdiv(data_col_shape, tile_col_shape)
    col_loop_num = dl.cdiv(data_col_loop_num, swizzle_offset)
    n_tile_idx = iter_id // (swizzle_offset * data_row_loop_num)
    m_n_tile_idx = iter_id % (swizzle_offset * data_row_loop_num)
    n_tile_size = swizzle_offset
    if n_tile_idx == col_loop_num - 1:
        n_tile_size = data_col_loop_num - swizzle_offset * n_tile_idx
    data_row_idx = m_n_tile_idx // n_tile_size
    data_col_idx = n_tile_idx * swizzle_offset + m_n_tile_idx % n_tile_size
    if n_tile_idx % 2 == 1:
        data_row_idx = data_row_loop_num - data_row_idx - 1
    return data_row_idx, data_col_idx
