# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Verified device byte permutations shared by FlagMega renderers."""

from collections.abc import Mapping
from math import prod

from triton.flagmega.errors import ArtifactError
from triton.flagmega.ir import (
    DistributedBufferStorageKind, block_owner_count, verify_buffer_plan,
)

READONLY_DEVICE_LAYOUT_SCHEMA = "flagmega.readonly-device-layout/v1"


def readonly_roots(plan, buffer, seen=()):
    """Resolve every closed caller of a formal to canonical readonly roots."""
    if buffer.storage == "rdata":
        return (buffer,)
    if buffer.storage != "parameter" or buffer.function in seen:
        return ()
    callers = tuple(c for c in plan.call_map.values() if c.callee == buffer.function)
    if not callers:
        return ()
    roots = []
    for call in callers:
        actual = plan.resolve_call_buffer(call.call, buffer.id)
        resolved = readonly_roots(plan, actual, (*seen, buffer.function))
        if not resolved:
            return ()
        roots.extend(resolved)
    return tuple(roots)


def apply_readonly_layout(pool, contract, module, *, schema=READONLY_DEVICE_LAYOUT_SCHEMA):
    """Validate all ranges, then permute fresh canonical bytes before any use."""
    import torch
    if not isinstance(contract, Mapping) or contract.get("schema") != schema:
        raise ArtifactError("Unknown readonly device layout schema.")
    buffers = verify_buffer_plan(module).buffer_map
    records = contract.get("buffers")
    if not isinstance(records, (list, tuple)):
        raise ArtifactError("Readonly device layouts must be a list.")
    if pool.dtype != torch.uint8 or pool.ndim != 1 or not pool.is_contiguous():
        raise ArtifactError("Readonly device layouts require a contiguous byte pool.")
    end = 0
    for record in records:
        if not isinstance(record, Mapping):
            raise ArtifactError("Readonly layouts must contain buffer records.")
        buffer = buffers.get(record.get("buffer"))
        shape, axes = record.get("shape"), record.get("axes")
        if (buffer is None or buffer.storage != "rdata"
                or buffer.distributed_storage_kind is not DistributedBufferStorageKind.CANONICAL_GLOBAL
                or record.get("offset") != buffer.offset or record.get("nbytes") != buffer.nbytes
                or not isinstance(shape, (list, tuple)) or len(shape) != 4
                or any(type(dim) is not int or dim <= 0 for dim in shape)
                or shape[0] != buffer.shape[0]
                or buffer.distributed_type is None
                or shape[1] != block_owner_count(buffer.distributed_type)
                or shape[3] != 2 or axes != [1, 0, 2, 3]
                or prod(shape) != buffer.nbytes or record["offset"] < end
                or record["offset"] + buffer.nbytes > pool.numel()):
            raise ArtifactError("Readonly layout does not match its canonical buffer byte range.")
        end = record["offset"] + buffer.nbytes
    for record in records:
        buffer = buffers[record["buffer"]]
        view = pool.narrow(0, buffer.offset, buffer.nbytes).view(*record["shape"])
        # Even a unit owner dimension must not alias the source during copy.
        reordered = view.permute(*record["axes"]).clone(memory_format=torch.contiguous_format)
        view.reshape(-1).copy_(reordered.reshape(-1))
    return pool
