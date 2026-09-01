"""Thrive vendor implementation of tle.chiplet.*."""
import triton.language.core as tl
from triton.language.extra.thrive.thrive_semantic import get_thrive_semantic

SIGNAL_SET = 0
SIGNAL_ADD = 1
CMP_EQ = 0


def _as_uint64(value, _semantic):
    if isinstance(value, int):
        value = tl.constexpr(value)
    if isinstance(value, tl.constexpr):
        return _semantic.tensor(_semantic.builder.get_uint64(value), tl.uint64)
    if isinstance(value, tl.tensor):
        return value
    raise TypeError("Only support int/constexpr/tensor type of pe/sig_val/cmp_val")


def _as_uint32(value, _semantic):
    if isinstance(value, int):
        return _semantic.tensor(_semantic.builder.get_uint32(value), tl.uint32)
    raise TypeError("invalid input type of sig_op/cmp_op")


def _element_byte_size(element_ty):
    sizes = {
        'fp16': 2,
        'bf16': 2,
        'fp32': 4,
        'fp64': 8,
        'i8': 1,
        'i16': 2,
        'i32': 4,
        'i64': 8,
        'u8': 1,
        'u16': 2,
        'u32': 4,
        'u64': 8,
    }
    return sizes.get(str(element_ty), 4)


def _size_of(nelements, element_ty, _semantic):
    elem_size = _element_byte_size(element_ty)
    if isinstance(nelements, tl.constexpr):
        return _semantic.tensor(_semantic.builder.get_uint32(nelements * elem_size), tl.uint32)
    if isinstance(nelements, tl.tensor):
        size_ir = _semantic.builder.get_uint32(elem_size)
        return _semantic.tensor(_semantic.builder.create_mul(nelements.handle, size_ir), nelements.type)
    raise TypeError("Only support constexpr and tensor type of size")


def putmem_impl(dest, source, nelements, rank, blocking=True, _semantic=None):
    semantic = get_thrive_semantic(_semantic)
    size_input = _size_of(nelements, source.type.element_ty, _semantic)
    pe_input = _as_uint64(rank, _semantic)
    suffix = "" if blocking else "_nbi"
    semantic.extern_call(
        f"__shmem_putmem{suffix}_block",
        [dest, source, size_input, pe_input],
        [tl.void],
        False,
    )


def getmem_impl(dest, source, nelements, rank, blocking=True, _semantic=None):
    semantic = get_thrive_semantic(_semantic)
    size_input = _size_of(nelements, source.type.element_ty, _semantic)
    pe_input = _as_uint64(rank, _semantic)
    suffix = "" if blocking else "_nbi"
    semantic.extern_call(
        f"__shmem_getmem{suffix}_block",
        [dest, source, size_input, pe_input],
        [tl.void],
        False,
    )


def wait_impl(barrier_ptr, wait_value=1, _semantic=None):
    semantic = get_thrive_semantic(_semantic)
    cmp_input = _as_uint32(CMP_EQ, _semantic)
    cmp_val_input = _as_uint64(wait_value, _semantic)
    return semantic.extern_call(
        "__shmem_signal_wait_until_block",
        [barrier_ptr, cmp_input, cmp_val_input],
        [tl.void],
        False,
    )


def notify_impl(ptr, rank, signal=1, sig_op="set", _semantic=None):
    semantic = get_thrive_semantic(_semantic)
    sig_op_code = {"set": SIGNAL_SET, "add": SIGNAL_ADD}[sig_op]
    signal_input = _as_uint64(signal, _semantic)
    sig_op_input = _as_uint32(sig_op_code, _semantic)
    pe_input = _as_uint64(rank, _semantic)
    semantic.extern_call(
        "__shmem_signal_op_nbi_block",
        [ptr, signal_input, sig_op_input, pe_input],
        [tl.void],
        False,
    )


def fence_impl(_semantic=None):
    semantic = get_thrive_semantic(_semantic)
    semantic.extern_call("__shmem_fence", [], [tl.void], False)


def sync_impl(_semantic=None):
    semantic = get_thrive_semantic(_semantic)
    semantic.extern_call("__shmem_quiet", [], [tl.void], False)


def _sharding_spec_to_pe_tensor_spec(spec, tensor_ndim):
    if spec.split:
        assert len(
            spec.split) == tensor_ndim, (f"split rank ({len(spec.split)}) must match tensor rank ({tensor_ndim})")

    if "chiplet" not in spec.mesh.dim_names:
        raise ValueError('Thrive backend requires "chiplet" axis in mesh')

    from torch_thrive.backend.parallel_info import (
        PEMesh,
        PEPlacements,
        PEShard,
        PEReplicate,
        PEPartial,
        ParallelInfo,
    )

    tp_placement = PEReplicate()
    sharded_dim_count = 0
    for dim_idx, axes in enumerate(spec.split):
        for axis in axes:
            if axis == "chiplet":
                tp_placement = PEShard(dim_idx)
                sharded_dim_count += 1
            else:
                raise ValueError(f"Thrive backend does not support sharding on '{axis}' axis")
    if sharded_dim_count > 1:
        raise ValueError("Thrive backend supports at most one split dim along 'chiplet' axis")
    if "chiplet" in spec.partial:
        tp_placement = PEPartial()
    cluster_size = spec.mesh.shape[spec.mesh.dim_names.index("chiplet")]
    pe_mesh = PEMesh(TP=cluster_size, SP=1, DP=1, PP=1)
    placements = PEPlacements(
        TP=tp_placement,
        SP=PEReplicate(),
        DP=PEReplicate(),
        PP=PEReplicate(),
    )
    return ParallelInfo(placements, pe_mesh)


def make_sharded_tensor_impl(handle, sharding, shape=None):
    from torch_thrive.backend.tade import update_tade_context
    pe_spec = _sharding_spec_to_pe_tensor_spec(sharding, handle.ndim)
    update_tade_context(pe_spec)
    if handle.device.type == "thrive":
        raise RuntimeError("handle already on thrive; use torch.empty_like(..., device='thrive') to consume spec")
    return handle.to("thrive")
