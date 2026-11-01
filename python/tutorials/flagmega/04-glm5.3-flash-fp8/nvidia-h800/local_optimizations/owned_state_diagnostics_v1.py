"""Explicit host-only assembly for diagnostic compact-per-chip state exports."""
from itertools import product
from math import prod
import torch
from triton.flagmega.ir.distributed_storage import DistributedBufferStorageKind
from triton.flagmega.ir.distributed_type import block_owner_axes
from triton.flagmega.ir.local_shard import local_shard_descriptor, unravel_placement_index
from triton.flagmega.runtime.state_storage import tensor_state_storage_type


def assemble(descriptor, pieces):
    """Assemble all chips, checking coverage and any replicated coordinates.

    This function performs no collectives, device transfers or runtime state
    writes. Callers explicitly collect CPU snapshots outside timed requests.
    """
    assert descriptor.distributed_storage_kind is DistributedBufferStorageKind.COMPACT_PER_CHIP
    kind = descriptor.distributed_type
    assert kind.partial is None and kind.exclusive is None
    mesh = kind.placement
    chips = tuple(axis for axis, level in enumerate(mesh.hierarchy_levels) if level in {"c", "d"})
    blocks = block_owner_axes(kind)
    assert len(pieces) == prod(mesh.hierarchy[axis] for axis in chips)
    physical = tuple(d.fixed_value for d in tensor_state_storage_type(descriptor).shape)
    expected_dtype = getattr(torch, descriptor.dtype.value)
    assert all(value.device.type == "cpu" and value.is_contiguous() and tuple(value.shape) == physical
               and value.dtype == expected_dtype for value in pieces)
    whole = torch.empty(descriptor.shape, dtype=expected_dtype)
    covered = torch.zeros(descriptor.shape, dtype=torch.bool)
    for chip, piece in enumerate(pieces):
        coords = [0] * mesh.rank
        for axis, value in zip(chips, unravel_placement_index(chip, tuple(mesh.hierarchy[a] for a in chips)), strict=True):
            coords[axis] = value
        for owner, block in enumerate(product(*(range(mesh.hierarchy[a]) for a in blocks))):
            for axis, value in zip(blocks, block, strict=True):
                coords[axis] = value
            local = local_shard_descriptor(kind, tuple(coords))
            active = tuple(a.active_extent.fixed_value for a in local.axes)
            if not all(active):
                continue
            source = piece[owner][tuple(slice(0, n) for n in active)]
            slices = tuple(axis.local_slice for axis in local.axes)
            if all(part is not None for part in slices):
                target, mask = whole[slices], covered[slices]
                if bool(mask.any()):
                    assert torch.equal(target[mask].contiguous().view(torch.uint8),
                                       source[mask].contiguous().view(torch.uint8)), "different replicated state"
                target.copy_(source)
                mask.fill_(True)
            else:
                indices = tuple(torch.tensor([axis.map_local_to_global(i).fixed_value for i in range(n)])
                                for axis, n in zip(local.axes, active, strict=True))
                grid = torch.meshgrid(*indices, indexing="ij")
                mask = covered[grid]
                if bool(mask.any()):
                    assert torch.equal(whole[grid][mask].contiguous().view(torch.uint8),
                                       source[mask].contiguous().view(torch.uint8)), "different replicated state"
                whole[grid] = source
                covered[grid] = True
    assert bool(covered.all()), "state has uncovered global coordinates"
    return whole


def collect_kda(runtime, state, *, group):
    """Explicit diagnostic all-gather and global assembly on rank zero only."""
    import torch.distributed as dist
    local = {layer: (conv.cpu(), rec.cpu()) for layer, (conv, rec) in state.tensor_states.items()}
    pieces = [None] * dist.get_world_size(group=group)
    dist.all_gather_object(pieces, local, group=group)
    if dist.get_rank(group=group) != 0:
        return None
    function = runtime.buffer_plan.function_map[runtime.ir_module.entry]
    bindings = dict(function.parameters)
    result = {}
    for record in runtime.ir_module.metadata["composite_abi"]["tensor_states"]:
        layer = record["state_id"]
        fields = []
        for index, input_id in enumerate(record["input"]):
            descriptor = runtime.buffer_plan.buffer_map[bindings[input_id][0]]
            fields.append(assemble(descriptor, [value[layer][index] for value in pieces]))
        result[layer] = tuple(fields)
    return result
