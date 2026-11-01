# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Relocate readonly allocations while preserving typed views and bank rows."""

from dataclasses import replace

from triton.flagmega.errors import ArtifactError
from triton.flagmega.ir import dim, verify_buffer_plan, verify_module
from triton.flagmega.artifacts.rdata import _aliases_rdata_owner


def relocate_readonly(module, offsets, *, nbytes):
    plan = verify_buffer_plan(module)
    owners = {b.id: b for b in plan.buffers if b.storage == "rdata" and not _aliases_rdata_owner(b)}
    if set(offsets) != set(owners) or type(nbytes) is not int or nbytes < 0:
        raise ArtifactError("Readonly relocation requires every owner and a valid pool extent.")
    physical_offsets, group_deltas = {}, {}
    for identity, buffer in owners.items():
        offset = offsets[identity]
        if type(offset) is not int or offset < 0 or offset % buffer.alignment:
            raise ArtifactError("Readonly relocation violates an allocation's alignment.")
        start = offset - buffer.mem_span.start.fixed_value
        prior = physical_offsets.setdefault(buffer.physical_id, start)
        if prior != start:
            raise ArtifactError("Readonly aliases require one consistent physical relocation.")
        if buffer.rdata_group:
            delta = offset - buffer.offset
            if group_deltas.setdefault(buffer.rdata_group, delta) != delta:
                raise ArtifactError("Readonly relocation must preserve all parameter-bank row offsets.")
    physical = tuple(replace(p, start=dim(physical_offsets[p.id]))
                     if p.id in physical_offsets else p for p in plan.physical_buffers)
    occupied = sorted((p.offset, p.offset + p.nbytes) for p in physical if p.memory_space == "rdata")
    if any(b > c for (_, b), (c, _) in zip(occupied, occupied[1:])):
        raise ArtifactError("Readonly relocation has overlapping allocations.")
    by_id = {p.id: p for p in physical}
    buffers = tuple(replace(b, mem_span=replace(b.mem_span, buffer=by_id[b.physical_id])) for b in plan.buffers)
    relocated = replace(plan, physical_buffers=physical, buffers=buffers, rdata_bytes=nbytes)
    result = replace(module, metadata={**module.metadata, "buffer_plan": relocated.to_data()})
    verify_buffer_plan(result)
    return verify_module(result)


__all__ = ["relocate_readonly"]
