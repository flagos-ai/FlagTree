"""Host-side Thrive sharding helpers for tle.dsa.thrive (not kernel builtins)."""

from .core import CMP_EQ, SIGNAL_ADD, SIGNAL_SET  # noqa: F401  (re-exported vendor constants)


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
