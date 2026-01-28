from typing import Optional, overload, Tuple

import triton.language.core as tl
from triton._C.libtriton import ir

from . import types as tlx
from .mma_ops import require_nv_mma_shared_layout
from .types import storage_kind
from .utility import cuda_parse_arch


def _assert_blackwell_for_tmem(arch):
    capability = int(cuda_parse_arch(arch))
    assert capability >= 100, "tmem is only available on Blackwell"


@tl.builtin
def storage_alias_spec(
    storage: tlx.storage_kind = tlx.storage_kind.smem,
    buffer_size_bytes: Optional[tl.constexpr] = None,
    _semantic=None,
) -> tlx.storage_alias_spec:
    """
    Create a storage alias specification.

    This function creates a storage alias specification that can be referenced by
    multiple `local_alloc` calls via the `reuse` parameter. Unlike directly
    passing a `buffered_tensor` to `reuse`, using a `storage_alias_spec` makes
    all referencing allocations equal peers with no primary owner.

    The storage alias spec can be either unsized or sized:

    - **Unsized (default)**: The compiler sets the buffer size to accommodate
      the largest allocation that references it.
    - **Sized**: The user specifies an explicit size, and the compiler verifies
      all referencing allocations fit within this size.

    All attributes of the returned object are immutable after construction.

    Args:
        storage: The storage kind for this buffer. Must be `smem` or `tmem`.
            All `local_alloc` calls that reference this `storage_alias_spec`
            must use the same storage kind. `smemCluster` is not supported.
        buffer_size_bytes: Optional explicit size in bytes. If provided, must
            be a compile-time constant (`tl.constexpr`). The compiler will
            verify that all referencing allocations fit within this size.
            This value is immutable after construction.
        _semantic: Internal parameter for Triton semantics.

    Returns:
        A `storage_alias_spec` object that can be passed to `local_alloc` via
        the `reuse` parameter.

    Raises:
        ValueError: If storage is not a valid `storage_kind`.
        ValueError: If storage is `smemCluster` (not supported).
        ValueError: If buffer_size_bytes is not a compile-time constant.
        ValueError: If buffer_size_bytes is not positive.

    Example:
        # Create an unsized storage alias spec (size determined by largest user)
        alias_spec = tlx.storage_alias_spec(storage=tlx.storage_kind.smem)

        # Create a sized storage alias spec with explicit size
        alias_spec = tlx.storage_alias_spec(
            storage=tlx.storage_kind.tmem,
            buffer_size_bytes=16384,
        )

        # Use with local_alloc (Phase 2 - not yet implemented)
        # buf_a = tlx.local_alloc(..., reuse=alias_spec)
        # buf_b = tlx.local_alloc(..., reuse=alias_spec)
    """
    # Validate storage kind
    if not isinstance(storage, tlx.storage_kind):
        raise ValueError(f"storage must be a tlx.storage_kind, got {type(storage)}")

    # smemCluster is not supported
    if storage == tlx.storage_kind.smemCluster:
        raise ValueError("smemCluster storage is not supported for storage_alias_spec")

    # Validate and unwrap buffer_size_bytes if provided
    unwrapped_size = None
    if buffer_size_bytes is not None:
        unwrapped_size = tl._unwrap_if_constexpr(buffer_size_bytes)
        if unwrapped_size <= 0:
            raise ValueError(f"buffer_size_bytes must be positive, got {unwrapped_size}")

    # Create IR operation
    handle = _semantic.builder.create_storage_alias_spec(
        storage.value,
        unwrapped_size,
    )

    # Return wrapper object (immutable)
    return tlx.storage_alias_spec(
        handle=handle,
        storage=storage,
        buffer_size_bytes=unwrapped_size,
    )


def _create_tmem_compatible_tensor_layout_encoding(
    builder,
    tensor: tlx.buffered_tensor,
):
    return builder.make_dummy_register_layout_attr(list(tensor.shape), tensor.dtype.to_ir(builder), True)


@tl.builtin
def local_alloc(
    shape: tuple,
    dtype: tl.dtype,
    num: tl.constexpr,
    storage: tlx.storage_kind = tlx.storage_kind.smem,
    reuse: Optional[tlx.buffered_tensor] = None,
    layout: Optional[tlx.shared_layout_encoding] = None,
    _semantic=None,
) -> tlx.buffered_tensor:
    """
    Allocates buffer in shared memory and return a view of the buffer.
    """
    if storage == tlx.storage_kind.tmem:
        _assert_blackwell_for_tmem(_semantic.builder.options.arch)

    if not isinstance(num, tl.constexpr):
        user_error = """
`num` must be a constexpr without introducing any `ast.Assign` nodes,
otherwise its value will be wrapped as `tensor.handle`.
For example, following will fail because `num` will be promoted to tl.tensor by semantics.py
in visit_Assign
    num = tl.constexpr(2)
    local_alloc(..., num=num)

To bypass, rewrite it to `local_alloc(..., num=tl.constexpr(2))` or `local_alloc(..., num=2)`
        """
        raise ValueError(user_error)

    unwrapped_shape = [tl._unwrap_if_constexpr(dim) for dim in shape]
    unwrapped_num = tl._unwrap_if_constexpr(num)
    full_shape = [unwrapped_num] + unwrapped_shape
    dtype = tl._unwrap_if_constexpr(dtype)
    elem_type = dtype.to_ir(_semantic.builder)
    if layout is None:
        if storage == tlx.storage_kind.smem:
            if len(shape) == 1:
                layout = tlx.swizzled_shared_layout_encoding.make_default(rank=len(shape))
                layout_handle = _semantic.builder.make_swizzled_shared_encoding_attr(
                    layout.vectorSize,
                    layout.perPhase,
                    layout.maxPhase,
                    layout.order,
                    layout.numCTAsPerCGA,
                    layout.numCTASplit,
                    layout.numCTAOrder,
                )
            else:
                layout = tlx.nv_mma_shared_layout_encoding.make_default(shape, dtype)
                layout_handle = _semantic.builder.make_nv_mma_shared_encoding_attr(
                    [int(x) for x in layout.shape],
                    layout.order,
                    layout.elemType.to_ir(_semantic.builder),
                    layout.numCTAsPerCGA,
                    layout.numCTASplit,
                    layout.numCTAOrder,
                    layout.fp4Padded,
                    layout.swizzled,
                )
        else:
            # For 8-bit element types (uint8/int8), use a dummy TMEM layout that will
            # be resolved during layout propagation. This is used for scales in
            # scaled MMA operations where the final layout depends on usage context.
            if dtype == tl.uint8 or dtype == tl.int8:
                layout = None  # Will be resolved by layout propagation
                layout_handle = _semantic.builder.make_dummy_tmem_layout_attr()
            else:
                layout = tlx.tensor_memory_layout_encoding.make_default(shape)
                layout_handle = _semantic.builder.make_tensor_memory_encoding_attr(
                    layout.blockM,
                    layout.blockN,
                    layout.unpacked,
                    layout.CTASplitM,
                    layout.CTASplitN,
                )
    else:
        raise NotImplementedError("User-specified layout encoding not yet implemented.")

    alias_handle = None
    if reuse:
        # reuse tensor has to be a buffered tensor
        if not isinstance(reuse, tlx.buffered_tensor):
            raise ValueError("reuse tensor has to be a buffered tensor")
        # verify that the reuse tensor has the same storage
        if reuse.type.storage != storage:
            raise ValueError("reuse tensor has different storage")
        alias_handle = reuse.handle

    if storage == tlx.storage_kind.smem:
        tensor_handle = _semantic.builder.create_local_alloc(full_shape, elem_type, layout_handle, alias_handle)
    else:
        tensor_handle = _semantic.builder.create_tmem_alloc(full_shape, elem_type, layout_handle, alias_handle)

    return tlx.buffered_tensor(tensor_handle, dtype, unwrapped_shape, unwrapped_num, storage, layout)


# overload declarations just to make linter happy
@overload
def local_view(
    local_allocated_buffers: tlx.buffered_tensor,
    buffer_idx: int,
    _semantic=None,
) -> tlx.buffered_tensor:
    ...


@overload
def local_view(
    local_allocated_buffers: tlx.mbarrier,
    buffer_idx: int,
    _semantic=None,
) -> tlx.mbarrier:
    ...


@overload
def local_view(
    local_allocated_buffers: tlx.clc_response,
    buffer_idx: int,
    _builder=None,
) -> tlx.clc_response:
    ...


@tl.builtin
def local_view(
    local_allocated_buffers: tlx.buffered_tensor | tlx.mbarrier | tlx.clc_response,
    buffer_idx: int,
    _semantic=None,
) -> tlx.buffered_tensor | tlx.mbarrier | tlx.clc_response:
    """
    Returns a subview of the buffer.
    """
    buffer_idx = _semantic._convert_elem_to_ir_value(buffer_idx, require_i64=False)
    view_handle = _semantic.builder.create_memdesc_subview(local_allocated_buffers.handle, buffer_idx)
    if isinstance(local_allocated_buffers, tlx.mbarrier):
        return tlx.mbarrier(view_handle, 0, local_allocated_buffers.type.layout)
    elif isinstance(local_allocated_buffers, tlx.clc_response):
        return tlx.clc_response(view_handle, 0, local_allocated_buffers.type.layout)
    else:
        # Calculate the correct shape for the subview according to create_memdesc_subview logic
        original_shape = local_allocated_buffers.shape
        if local_allocated_buffers.type.num == 0:
            if len(original_shape) == 1:
                # For 1D tensors, subview creates a single element view with shape [1]
                new_shape = [1]
            else:
                # For multi-dimensional tensors, drop the first dimension
                new_shape = original_shape[1:]
        else:
            new_shape = original_shape

        return tlx.buffered_tensor(
            view_handle,
            local_allocated_buffers.type.scalar,
            new_shape,
            0,
            local_allocated_buffers.type.storage,
            local_allocated_buffers.type.layout,
        )


@tl.builtin
def _buffered_tensor_getitem(self, buffer_idx, _semantic=None):
    return local_view(self, buffer_idx, _semantic=_semantic)


def _get_remote_cta_rank_handle(remote_cta_rank, _semantic):
    """
    Convert remote_cta_rank to MLIR Value handle.

    Handles multiple input types:
    - tl.constexpr or int: Converted via _convert_elem_to_ir_value
    - tl.tensor: Extract .handle attribute
    """
    if isinstance(remote_cta_rank, tl.constexpr) or isinstance(remote_cta_rank, int):
        remote_cta_rank_handle = _semantic._convert_elem_to_ir_value(tl._unwrap_if_constexpr(remote_cta_rank),
                                                                     require_i64=False)
    else:
        assert isinstance(remote_cta_rank, tl.tensor), (
            f"`remote_cta_rank` is in type {type(remote_cta_rank)} (must be either `tl.tensor` or `tl.constexpr`)")
        remote_cta_rank_handle = remote_cta_rank.handle
    return remote_cta_rank_handle


@tl.builtin
def remote_view(
    local_allocated_buffer: tlx.mbarrier,
    remote_cta_rank: int | tl.constexpr | tl.tensor,
    _semantic=None,
) -> tlx.mbarrier:
    """
    Returns a remote view of the buffer. This returns a remote buf handle living in a CTA in the same CTA cluster with the
    executing CTA.
    :arg local_allocated_buffer: the local buffer handle we start with
    :arg remote_cta_rank: unique ID of the remote CTA within the CTA cluster. This ID is across all dims, so e.g. for
    a cluster of shape [2, 4] a valid unique ID could be 0~7, including the executing CTA itself
    :returns: a remote view of the buffer, located at the same relative location, but just in a possibly different CTA
    """
    assert isinstance(local_allocated_buffer, tlx.mbarrier), ("remote_view only supports barrier for now")
    assert local_allocated_buffer.type.storage == storage_kind.smem, "remote_view requires local smem as input"
    remote_cta_rank_handle = _get_remote_cta_rank_handle(remote_cta_rank, _semantic)
    remote_buf_handle = _semantic.builder.create_map_to_remote_buffer(local_allocated_buffer.handle,
                                                                      remote_cta_rank_handle)
    if isinstance(local_allocated_buffer, tlx.mbarrier):
        return tlx.mbarrier(
            remote_buf_handle,
            0,
            local_allocated_buffer.type.layout,
            storage_kind.smemCluster,
        )
    else:
        raise ValueError("Unsupported type for local_allocated_buffer")


@tl.builtin
def remote_shmem_store(
    dst: tlx.buffered_tensor,
    src: tl.tensor,
    remote_cta_rank: int | tl.constexpr,
    _semantic=None,
) -> tl.tensor:
    """
    Store a distributed tensor into a buffer into the remote shared memory of a cluster.
    """
    storage = dst.type.storage
    assert storage == tlx.storage_kind.smem, (
        "remote_shmem_store only supports local smem for dst. dst will be internally mapped to remote_cta_rank's shmem")
    assert remote_cta_rank is not None, "remote_cta_rank is required for remote_shmem_store"
    remote_cta_rank_handle = _get_remote_cta_rank_handle(remote_cta_rank, _semantic)
    return tl.tensor(
        _semantic.builder.create_remote_store(dst.handle, src.handle, remote_cta_rank_handle),
        tl.void,
    )


@tl.builtin
def async_remote_shmem_store(
    dst: tlx.buffered_tensor,
    src: tl.tensor,
    remote_cta_rank: int | tl.constexpr,
    barrier: tlx.mbarrier,
    _semantic=None,
) -> tl.tensor:
    """
    Store a distributed tensor into a buffer into the remote shared memory of a cluster asynchronously.
    Signals the provided mbarrier when the store completes.

    Args:
        dst: The destination buffer in local shared memory (will be internally mapped to remote CTA)
        src: The source tensor to store
        remote_cta_rank: The rank of the remote CTA within the cluster
        barrier: mbarrier to signal when the store completes
    """
    storage = dst.type.storage
    if storage == tlx.storage_kind.smemCluster:
        print("tlx.async_remote_shmem_store only supports smem dst, it internally calls mapa(dst)")
    assert storage == tlx.storage_kind.smem, (
        "async_remote_shmem_store only supports local smem for dst. dst will be internally mapped to remote_cta_rank's shmem"
    )
    assert remote_cta_rank is not None, "remote_cta_rank is required for async_remote_shmem_store"
    assert barrier is not None, "barrier is required for async_remote_shmem_store"
    remote_cta_rank_handle = _get_remote_cta_rank_handle(remote_cta_rank, _semantic)
    return tl.tensor(
        _semantic.builder.create_async_remote_store(dst.handle, src.handle, remote_cta_rank_handle, barrier.handle),
        tl.void,
    )


@tl.builtin
def _tensor_descriptor_ptr_getitem(self, index, _semantic=None):
    """
    Index into the tensor descriptor pointer array.
    Returns a pointer to the descriptor at the given index.
    Advances by descriptor_size bytes per index.

    :param index: The index into the descriptor array (can be int, constexpr, or tensor)
    :return: A new tensor_descriptor_ptr pointing to the indexed descriptor
    """
    descriptor_size = self.descriptor_size

    # Convert index to IR value
    if isinstance(index, tl.tensor):
        # If it's a tensor, use its handle directly
        index_handle = index.handle
    elif isinstance(index, int) or isinstance(index, tl.constexpr):
        index_val = tl._unwrap_if_constexpr(index)
        index_handle = _semantic.builder.get_int32(index_val)
    else:
        raise TypeError(f"Index must be int, constexpr, or tensor, got {type(index)}")

    # Multiply index by descriptor_size to get byte offset
    size_handle = _semantic.builder.get_int32(descriptor_size)
    offset_handle = _semantic.builder.create_mul(index_handle, size_handle)

    # Create addptr to advance by index * descriptor_size bytes
    indexed_handle = _semantic.builder.create_addptr(self.handle, offset_handle)

    # Return a new tensor_descriptor_ptr, preserving the original num and descriptor_size
    # This allows proper bounds tracking across the entire array
    return tlx.tensor_descriptor_ptr(indexed_handle, self.num, descriptor_size)


tlx.buffered_tensor.__getitem__ = _buffered_tensor_getitem
tlx.mbarrier.__getitem__ = _buffered_tensor_getitem
tlx.clc_response.__getitem__ = _buffered_tensor_getitem
tlx.tensor_descriptor_ptr.__getitem__ = _tensor_descriptor_ptr_getitem


@tl.builtin
def subslice(
    local_allocated_buffer: tlx.buffered_tensor,
    offset: int,
    size: int,
    _semantic=None,
) -> tlx.buffered_tensor:
    """
    Returns a subslice of the buffer (in TMEM). The source has to be 128xN and the slicing is
    along the innermost dimension.

    :param local_allocated_buffer: the source buffer
    :param offset: the start offset of the subslice, in terms of number of elements
    :param size: the size of the subslice, in terms of number of elements
    """
    # this is for TMEM subslice
    assert local_allocated_buffer.type.storage == tlx.storage_kind.tmem, "subslice is only supported for tmem"
    assert isinstance(local_allocated_buffer.type, tl.block_type), "subslice src is not block type"
    subslice_shape = [dim for dim in local_allocated_buffer.type.shape[:-1]] + [size]
    return tlx.buffered_tensor(
        _semantic.builder.create_tmem_subslice(local_allocated_buffer.handle, offset, size),
        local_allocated_buffer.type.element_ty,
        subslice_shape,
        local_allocated_buffer.type.num,
        local_allocated_buffer.type.storage,
        local_allocated_buffer.type.layout,
    )


@tl.builtin
def local_slice(
    buffer: tlx.buffered_tensor,
    offset: list[int],
    shape: list[int],
    _semantic=None,
) -> tlx.buffered_tensor:
    if buffer.type.storage == tlx.storage_kind.tmem:
        # TMEM can only slice along the innermost dimension
        assert len(offset) == 2 and len(shape) == 2
        assert offset[0] == 0
        assert shape[0] == buffer.type.shape[0]
        return subslice(buffer, offset[1], shape[1], _semantic=_semantic)
    else:
        slice_handle = _semantic.builder.create_memdesc_subslice(buffer.handle, offset, shape)
        return tlx.buffered_tensor(
            slice_handle,
            buffer.type.scalar,
            shape,
            0,
            buffer.type.storage,
            buffer.type.layout,
        )


@tl.builtin
def async_load(
    src: tl.tensor,
    result: tlx.buffered_tensor,
    mask: Optional[tl.tensor] = None,
    other: Optional[tl.tensor] = None,
    cache_modifier: str = "",
    eviction_policy: str = "",
    is_volatile: bool = False,
    _semantic=None,
) -> tlx.async_token:
    """
    Loads buffer from global to local memory asynchronously.
    """
    # Unwrap constexpr and convert to tensor (same as tl.load)
    mask = tl._unwrap_if_constexpr(mask)
    other = tl._unwrap_if_constexpr(other)
    if mask is not None:
        mask = _semantic.to_tensor(mask)
    if other is not None:
        other = _semantic.to_tensor(other)

    if src.type.is_ptr() and src.type.element_ty.is_block():
        # Load by a block pointer: `pointer_type<block_type<>>`
        # unsupported for now
        raise NotImplementedError("async_load by block pointer is not supported yet")
    else:
        # Load by a tensor of pointers or a pointer of scalar: `block_type<pointer_type<>>` or `pointer_type<>`
        _, src, mask, other = _semantic._prepare_legacy_load(src, mask, other, None, None)

    cache = _semantic._str_to_load_cache_modifier(cache_modifier)
    eviction = _semantic._str_to_eviction_policy(eviction_policy)
    return tlx.async_token(
        _semantic.builder.create_async_load(
            src.handle,
            result.handle,
            mask.handle if mask else None,
            other.handle if other else None,
            cache,
            eviction,
            is_volatile,
        ))


@tl.builtin
def async_load_commit_group(
    tokens: list[tlx.async_token] = [],
    _semantic=None,
) -> tlx.async_token:
    """
    Commits all prior initiated but uncommitted async_load ops an async group.
    Each token represents a tracked async load operation.
    """
    handles = [t.handle for t in tokens]
    return tlx.async_token(_semantic.builder.create_async_commit_group(handles))


@tl.builtin
def async_load_wait_group(
    pendings: tl.constexpr,
    tokens: list[tlx.async_token] = [],
    _semantic=None,
) -> tlx.async_token:
    """
    Wait for completion of prior asynchronous copy operations.
    Each token represents a tracked async commit group operation.
    """
    pendings = tl._unwrap_if_constexpr(pendings)
    handles = [t.handle for t in tokens]
    return tlx.async_token(_semantic.builder.create_async_wait(handles, pendings))


@tl.builtin
def local_load(
    src: tlx.buffered_tensor,
    token: tlx.async_token = None,
    _semantic=None,
) -> tl.tensor:
    """
    Loads buffer from local or tensor memory into a distributed tensor.
    """
    block_type = tl.block_type(src.type.element_ty, src.type.shape)
    storage = src.type.storage
    if storage == tlx.storage_kind.tmem:
        _assert_blackwell_for_tmem(_semantic.builder.options.arch)
        tmem_compatible_layout_encoding = _create_tmem_compatible_tensor_layout_encoding(_semantic.builder, src)
        load_handle = _semantic.builder.create_tmem_load(src.handle, tmem_compatible_layout_encoding,
                                                         token.handle if token else None)
        output = _semantic.builder.create_release_layout(load_handle)
        return tl.tensor(output, block_type)
    else:
        output = _semantic.builder.create_local_load(src.handle, token.handle if token else None)
        return tl.tensor(output, block_type)


@tl.builtin
def local_store(
    dst: tlx.buffered_tensor,
    src: tl.tensor,
    _semantic=None,
) -> tl.tensor:
    """
    Store a distributed tensor into a buffer in local or tensor memory.
    """
    storage = dst.type.storage
    if storage == tlx.storage_kind.tmem:
        _assert_blackwell_for_tmem(_semantic.builder.options.arch)
        tmem_compatible_layout_encoding = _create_tmem_compatible_tensor_layout_encoding(_semantic.builder, dst)
        src_handle = _semantic.builder.create_require_layout(src.handle, tmem_compatible_layout_encoding)
        return tl.tensor(_semantic.builder.create_tmem_store(dst.handle, src_handle), tl.void)

    return tl.tensor(_semantic.builder.create_local_store(dst.handle, src.handle), tl.void)


@tl.builtin
def tmem_copy(
    src: tlx.buffered_tensor,
    dst: tlx.buffered_tensor,
    _semantic=None,
) -> None:
    """
    Start an asynchronous copy from shared memory to tensor memory.

    This maps directly to NVIDIA Blackwell's tcgen05.cp instruction,
    enabling efficient data movement from SMEM to TMEM without going
    through registers.

    Args:
        src: Source buffer in shared memory (SMEM).
        dst: Destination buffer in tensor memory (TMEM).

    Note:
        The current semantics of the instruction are not well defined and
        the API may change in the future. Use at your own risk.
    """
    assert isinstance(src, tlx.buffered_tensor), "source must be a buffered tensor"
    assert isinstance(dst, tlx.buffered_tensor), "destination must be a buffered tensor"
    assert src.type.storage == tlx.storage_kind.smem, "source must be in shared memory"
    assert dst.type.storage == tlx.storage_kind.tmem, "destination must be in tensor memory"
    _assert_blackwell_for_tmem(_semantic.builder.options.arch)
    _semantic.builder.create_tmem_copy(src.handle, dst.handle)


@tl.builtin
def local_trans(input: tlx.buffered_tensor, dims: Tuple[int] = (1, 0), _semantic=None) -> tlx.buffered_tensor:
    """
    Permutes the dimensions of a tensor.

    If the parameter :code:`dims` is not specified, the function defaults to a (1,0) permutation,
    effectively transposing a 2D tensor.

    :param input: The input tensor.
    :param dims: The desired ordering of dimensions.  For example,
        :code:`(2, 1, 0)` reverses the order dims in a 3D tensor.
    """
    if len(input.type.shape) != len(dims):
        raise ValueError("permute dims must have the same length as input shape")
    if sorted(tl._unwrap_if_constexpr(d) for d in dims) != list(range(len(dims))):
        raise ValueError(f"permute dims must be a permutation of 0, 1, ..., n-1, but were {dims}")

    permuted_handle = _semantic.builder.create_memdesc_trans(input.handle, dims)
    return input.make_permute(permuted_handle, dims)


@tl.builtin
def local_reinterpret(
    src: tlx.buffered_tensor,
    dtype: tl.dtype,
    shape: list[tl.constexpr] = None,
    _semantic=None,
) -> tlx.buffered_tensor:
    """
    Reinterpret the dtype and shape of a buffered tensor. Layout is preserved.
    """
    if shape is None:
        shape = src.type.shape
    else:
        assert isinstance(src, tlx.buffered_tensor) and src.type.storage == tlx.storage_kind.smem, (
            "TLX local_reinterpret with reshaping only supports SMEM")

    reinterpreted_value_handle = _semantic.builder.create_memdesc_reinterpret(src.handle,
                                                                              dtype.to_ir(_semantic.builder), shape)
    return tlx.buffered_tensor(
        reinterpreted_value_handle,
        dtype,
        shape,
        src.type.num,
        src.type.storage,
        src.type.layout,
    )


@tl.builtin
def async_descriptor_load(
    desc: tl.tensor_descriptor_base,
    result: tlx.buffered_tensor,
    offsets: list[tl.tensor],
    barrier: tlx.mbarrier,
    pred: tl.tensor = None,
    cache_modifier: str = "",
    eviction_policy: str = "",
    multicast_targets: list[tl.tensor] = [],
    _semantic=None,
) -> None:
    assert isinstance(desc, tl.tensor_descriptor_base)
    ndim = len(desc.block_shape)
    assert len(offsets) == ndim, f"expected {ndim} offsets, but got {len(offsets)}"
    result_handle = require_nv_mma_shared_layout(result, True, _semantic.builder)
    multicast_targets = _semantic._convert_to_ir_values(multicast_targets, require_i64=False)
    offsets = _semantic._convert_to_ir_values(offsets, require_i64=False)
    cache = _semantic._str_to_load_cache_modifier(cache_modifier)
    eviction = _semantic._str_to_eviction_policy(eviction_policy)
    if pred is None:
        pred_handle = _semantic.builder.get_int1(True)
    else:
        pred_handle = pred.handle
    _semantic.builder.create_async_TMA_load(
        multicast_targets,
        desc.handle,
        offsets,
        barrier.handle,
        pred_handle,
        result_handle,
        cache,
        eviction,
        False,
    )


@tl.builtin
def async_descriptor_store(
    desc: tl.tensor_descriptor_base,
    source: tlx.buffered_tensor,
    offsets: list[tl.tensor],
    _semantic=None,
) -> None:
    assert isinstance(desc, tl.tensor_descriptor_base)
    ndim = len(desc.block_shape)
    assert len(offsets) == ndim, f"expected {ndim} offsets, but got {len(offsets)}"
    source_handle = require_nv_mma_shared_layout(source, True, _semantic.builder)
    offsets = _semantic._convert_to_ir_values(offsets, require_i64=False)
    _semantic.builder.create_async_TMA_store(desc.handle, offsets, source_handle)


@tl.builtin
def async_descriptor_store_wait(
    pendings: tl.constexpr,
    _semantic=None,
) -> None:
    """
    Wait for completion of prior asynchronous TMA store operations.
    """
    pendings = tl._unwrap_if_constexpr(pendings)
    _semantic.builder.create_async_TMA_store_wait(pendings)


@tl.builtin
def fence_async_shared(_semantic=None, ) -> None:
    """
    Order memory operations that go through the shared memory.
    """
    _semantic.builder.create_fence_async_shared(False)


@tl.builtin
def allocate_tensor_descriptor(
    num: tl.constexpr,
    _semantic=None,
) -> tlx.tensor_descriptor_ptr:
    """
    Allocates buffer in global memory for tensor descriptor storage with builtin parameters
    (nbytes=128, alignment=128) and returns a tensor descriptor pointer.
    The returned pointer advances by 128 bytes when incremented by 1 (ptr + 1).
    Supports indexing operation: ptr[i] to access the i-th descriptor.

    :param num: Number of tensor descriptors to allocate
    :return: A tensor_descriptor_ptr with 128-byte stride semantics and num tracking
    """
    if not isinstance(num, tl.constexpr):
        raise ValueError("`num` must be a constexpr")

    # Use builtin values for tensor descriptor allocation
    unwrapped_num = tl._unwrap_if_constexpr(num)
    descriptor_size = 128
    nbytes = descriptor_size * unwrapped_num
    alignment = 128

    tensor_handle = _semantic.builder.create_global_scratch_alloc(nbytes, alignment)

    # Return a tensor_descriptor_ptr which has built-in 128-byte stride semantics
    # Pass num and descriptor_size so the type knows how many descriptors it can access
    return tlx.tensor_descriptor_ptr(tensor_handle, unwrapped_num, descriptor_size)


@tl.builtin
def make_tensor_descriptor(
    desc_ptr: tlx.tensor_descriptor_ptr | None,
    base: tl.tensor,
    shape: list[tl.tensor],
    strides: list[tl.tensor],
    block_shape: list[tl.constexpr],
    padding_option="zero",
    _semantic=None,
) -> tl.tensor_descriptor_base:
    """
    Create a TMA descriptor on device for loading/storing data from global memory.

    This function creates a tt.make_tensor_descriptor operation that can be used with
    async TMA operations for efficient data movement.

    .. note::
        The `desc_ptr` parameter is optional. If provided, the descriptor will use the
        provided tensor descriptor pointer (from tlx.allocate_tensor_descriptor). If None, the
        compiler will automatically allocate global scratch memory for the descriptor.

    :param desc_ptr: Optional tensor_descriptor_ptr for descriptor storage (from tlx.allocate_tensor_descriptor). Pass None to auto-allocate.
    :param base: Base pointer to the tensor in global memory
    :param shape: List of tensor dimensions (dynamic, runtime values)
    :param strides: List of tensor strides (dynamic, runtime values)
    :param block_shape: Shape of the block to be loaded/stored (compile-time constants)
    :param padding_option: Padding option for out-of-bounds accesses (default: "zero")

    Example:
    --------
    .. code-block:: python

        # Allocate storage for descriptors
        desc_ptrs = tlx.allocate_tensor_descriptor(num=2)

        # Create a 2D tensor descriptor at index 0
        tlx.make_tensor_descriptor(
            desc_ptr=desc_ptrs[0],
            base=tensor_ptr,
            shape=[M, N],
            strides=[N, tl.constexpr(1)],
            block_shape=[64, 64],
        )

        # Reinterpret the descriptor for TMA operations
        desc = tlx.reinterpret_tensor_descriptor(
            desc_ptr=desc_ptrs[0],
            block_shape=[64, 64],
            dtype=tl.float16,
        )

        # Use with async TMA load
        tlx.async_descriptor_load(desc, buffer, offsets=[m_offset, n_offset], barrier=mbar)
    """
    # Type check desc_ptr
    if desc_ptr is not None and not isinstance(desc_ptr, tlx.tensor_descriptor_ptr):
        raise TypeError(f"desc_ptr must be None or tlx.tensor_descriptor_ptr, got {type(desc_ptr)}. "
                        f"Use tlx.allocate_tensor_descriptor() to allocate descriptor storage.")
    ndim = len(shape)
    if not (1 <= ndim <= 5):
        raise ValueError(f"Expected 1 <= ndim <= 5 but got {ndim} dimensions")
    if len(strides) != ndim:
        raise ValueError(f"Expected {ndim} strides but got {len(strides)}")
    if len(block_shape) != ndim:
        raise ValueError(f"Expected block_shape to have {ndim} dimensions but got {len(strides)}")
    assert isinstance(base.dtype, tl.pointer_type)
    elem_size = base.dtype.element_ty.primitive_bitwidth // 8
    contig_dim_size = tl._unwrap_if_constexpr(block_shape[-1])
    if contig_dim_size * elem_size < 16:
        raise ValueError(
            f"Descriptor block shape must have at least 16 bytes in the last dimension, but got {contig_dim_size} * {elem_size} = {contig_dim_size * elem_size} bytes"
        )

    last_stride = tl._unwrap_if_constexpr(strides[-1])
    if last_stride != 1:
        raise ValueError(f"Tensor descriptor last dim must be 1 but got {last_stride}")

    shape = [_semantic.make_scalar(x, tl.int32) for x in shape]
    strides = [_semantic.make_scalar(tl._unwrap_if_constexpr(x), tl.int64) for x in strides]

    # Check whether `block_shape` is static
    block_shape = tl._unwrap_shape(block_shape)

    assert isinstance(base.type, tl.pointer_type)
    block_type = tl.block_type(base.type.element_ty, block_shape)
    base_handle = base.handle
    is_signed_int = base.type.element_ty.is_int_signed()

    padding = _semantic._str_to_padding_option(padding_option)

    if base.type.element_ty.is_int() and padding == ir.PADDING_OPTION.PAD_NAN:
        raise ValueError("Padding option `nan` is not supported for integer blocks")

    desc_handle = desc_ptr.handle if desc_ptr is not None else None
    if desc_handle:
        handle = _semantic.builder.create_make_tensor_descriptor(
            base_handle,
            [s.handle for s in shape],
            [s.handle for s in strides],
            desc_handle,
            block_shape,
            is_signed_int,
            padding,
        )
    else:
        handle = _semantic.builder.create_make_tensor_descriptor(
            base_handle,
            [s.handle for s in shape],
            [s.handle for s in strides],
            block_shape,
            is_signed_int,
            padding,
        )
    return tl.tensor_descriptor(handle, shape, strides, block_type)


@tl.builtin
def reinterpret_tensor_descriptor(
    desc_ptr: tlx.tensor_descriptor_ptr,
    block_shape: list[tl.constexpr],
    dtype: tl.dtype,
    _semantic=None,
) -> tl.tensor_descriptor_base:
    """
    Reinterpret a tensor descriptor pointer as a TMA-backed tensor descriptor object.

    This function creates a tensor descriptor from a tensor_descriptor_ptr
    (e.g., from tlx.allocate_tensor_descriptor). This is useful when you have
    allocated descriptor storage and need to convert it to a tensor descriptor
    for use with TMA operations.

    :param desc_ptr: A tensor_descriptor_ptr pointing to the TMA descriptor
    :param block_shape: Shape of the block to be loaded/stored (compile-time constants)
    :param dtype: Data type of the tensor elements

    Example:
    --------
    .. code-block:: python

        # Allocate storage for 4 tensor descriptors
        desc_ptrs = tlx.allocate_tensor_descriptor(num=4)

        # Reinterpret the first descriptor
        desc = tlx.reinterpret_tensor_descriptor(
            desc_ptr=desc_ptrs[0],
            block_shape=[64],
            dtype=tl.int16,
        )

        # Now you can use desc with TMA operations
        tlx.async_descriptor_load(desc, buffer, offsets=[0], barrier=mbar)
    """
    # Type check desc_ptr
    if not isinstance(desc_ptr, tlx.tensor_descriptor_ptr):
        raise TypeError(f"desc_ptr must be tlx.tensor_descriptor_ptr, got {type(desc_ptr)}. "
                        f"Use tlx.allocate_tensor_descriptor() to allocate descriptor storage.")

    # Extract the IR handle from the tensor_descriptor_ptr
    # Create a tl.tensor wrapper for compatibility with reinterpret_tensor_descriptor
    ptr_type = tl.pointer_type(tl.int8)
    tensor_wrapper = tl.tensor(desc_ptr.handle, ptr_type)

    block_ty = tl.block_type(tl._unwrap_if_constexpr(dtype), block_shape)
    return _semantic.reinterpret_tensor_descriptor(tensor_wrapper, block_ty)
