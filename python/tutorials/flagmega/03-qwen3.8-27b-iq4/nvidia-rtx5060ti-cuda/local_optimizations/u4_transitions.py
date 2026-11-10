# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Complete-value stream edges and preserved fixed96 coordinates.

IR aliases and core physical spans are checked here. CPU coordinate proofs
cannot establish native stores, publication or capture behavior.
"""
from __future__ import annotations
from dataclasses import dataclass, replace
from triton.flagmega.ir import DistributedType, Placement, SBP, tensor_type
from triton.flagmega.ir.distributed_type import SBPBroadCast
from .u4_q8_abi import integer, require
from .u4_views import (
    PhysicalView, validate_live_views, validate_gdn_z, validate_q_gate_roots,
    validate_frames, cyclic32_coordinate, q_read_coordinate, gate_read_coordinate,
    gdn_z_coordinate, q_gate_coordinate,
)


def make_canonical_reader_view(source, reader_type, *, name):
    """Construct G0's real ShardedView with D0 preconditions after inference."""
    require(source.metadata.get("canonical_global") is True, "reader requires canonical producer")
    require(isinstance(source.type, DistributedType), "canonical producer needs inferred unit-block owners")
    require(all(n == 1 for n, level in zip(source.type.placement.hierarchy,
                                           source.type.placement.hierarchy_levels) if level == "b"),
            "ordinary canonical source retains fixed-block owners")
    require(all(isinstance(p, SBPBroadCast) for p in source.type.axis_policies)
            and source.type.exclusive is None and source.type.partial is None,
            "canonical producer must be a complete chip value")
    require(isinstance(reader_type, DistributedType) and reader_type.tensor == source.type.tensor,
            "reader must preserve actual dtype, lanes and full logical extent")
    require(tuple(n for n, level in zip(reader_type.placement.hierarchy,
                                        reader_type.placement.hierarchy_levels) if level == "b") == (4, 24),
            "reader must retain fixed96 owner mesh")
    from .u4_regions import make_canonical_reader_view as construct
    view = construct(source, reader_type, name=name)
    return replace(view, metadata={**view.metadata, "u4_publication": "same-stream-complete-value",
                                   "u4_read_only": True, "u4_preserve_canonical_strides": True})


def gdn_z_reader_type(source_type, *, fullhead48=False):
    """Typed destination-owner requirement, consumed during I0 inference.

    This is not a zero-copy view: K0 must gather into a distinct private root.
    """
    tensor = source_type.tensor if isinstance(source_type, DistributedType) else source_type
    require(tensor == tensor_type("bfloat16", (1, 6144)), "GDN Z requires scalar BF16[1,6144]")
    require(type(fullhead48) is bool, "fullhead48 must be explicitly selected")
    return DistributedType(tensor, (SBP.broadcast(), SBP.split_contiguous((0, 1), 128 if fullhead48 else 64)),
                           Placement((4, 12) if fullhead48 else (4, 24), "yx", "bb"))


def validate_canonical_cyclic32_buffers(source, reader, plan):
    """Check regenerated canonical root, actual BC32 type and view-only path.

    This is a physical-plan guard. Native indexing/publication still requires
    the emitted program and device validation; zero copy alone is insufficient.
    """
    from triton.flagmega.ir.bufferization import BufferDescriptor, BufferPlan
    from .u4_execution_plan import cyclic32_suffix_type
    require(isinstance(source, BufferDescriptor) and isinstance(reader, BufferDescriptor)
            and isinstance(plan, BufferPlan), "cyclic32 boundary needs actual core buffer descriptors")
    require(plan.buffer_map.get(source.id) == source and plan.buffer_map.get(reader.id) == reader,
            "cyclic32 descriptors are not in the supplied plan")
    require(source.distributed_type is not None and source.distributed_type.partial is None
            and source.distributed_type.exclusive is None
            and source.distributed_type.placement == Placement((1, 1), "yx", "bb")
            and all(isinstance(p, SBPBroadCast) for p in source.distributed_type.axis_policies),
            "cyclic32 writer is not a complete ordinary chip value")
    require(reader.distributed_type == cyclic32_suffix_type(source.distributed_type,
                                             Placement((4, 24), "yx", "bb")),
            "canonical reader lost frozen96 cyclic32 ownership")
    require(source.shape == reader.shape == (1, 5120)
            and source.strides == reader.strides == (5120, 1)
            and source.storage == reader.storage == "workspace"
            and source.distributed_storage_kind.exposes_logical_coordinates
            and reader.distributed_storage_kind.exposes_logical_coordinates
            and source.mem_span == reader.mem_span,
            "cyclic32 view changed canonical root/strides/coordinate storage")
    chain, cursor = [], reader
    while cursor.id != source.id:
        require(cursor.id not in chain and cursor.alias_of in plan.buffer_map
                and cursor.mem_span == source.mem_span,
                "canonical cyclic32 transition contains a materializing copy or alias cycle")
        chain.append(cursor.id)
        cursor = plan.buffer_map[cursor.alias_of]
    require(bool(chain) and reader.live_end <= source.mem_span.buffer.live_end,
            "canonical root does not cover its reader")
    return dict(source=source.id, reader=reader.id, physical_root=source.physical_id,
                span=source.mem_span.to_data(), alias_chain=chain,
                axis_policy=reader.distributed_type.axis_policies[1].to_data(),
                scalar_storage_strides=list(reader.strides), copied_bytes=0)


@dataclass(frozen=True)
class CompleteValueTransition:
    producer: int
    consumer: int
    source: PhysicalView
    reader: PhysicalView
    producer_stream: int = 0
    consumer_stream: int = 0
    complete: bool = True
    publication: str = "same-stream-kernel-completion"


def validate_complete_value_transition(edge):
    require(isinstance(edge, CompleteValueTransition), "expected typed transition")
    require(integer(edge.producer) and integer(edge.consumer) and edge.producer < edge.consumer,
            "consumer must follow complete producer entry")
    require(type(edge.producer_stream) is int and edge.producer_stream == 0
            and type(edge.consumer_stream) is int and edge.consumer_stream == 0,
            "U4 supports one caller stream only")
    require(edge.complete is True and edge.publication == "same-stream-kernel-completion",
            "enqueue return/partial value is not publication")
    source, reader = edge.source, edge.reader
    require(isinstance(source, PhysicalView) and isinstance(reader, PhysicalView), "missing physical values")
    require(source.coordinate_space == reader.coordinate_space == "canonical_global"
            and source.writable and not reader.writable and reader.alias_of == source.name,
            "transition needs one complete canonical writer and explicit read-only alias")
    require(source.span == reader.span and source.layout == reader.layout
            and source.epoch == reader.epoch and source.frame == reader.frame,
            "transition changed root/extent/lanes/frame/epoch")
    require(source.live_start <= edge.producer < edge.consumer <= source.live_end
            and reader.live_start <= edge.consumer <= reader.live_end,
            "root/read view expires before complete consumer")
    validate_live_views((source, reader))
    return dict(producer=edge.producer, consumer=edge.consumer, source=source.name,
                reader=reader.name, physical_buffer=source.root_id, epoch=source.epoch,
                stream_index=0, publication=edge.publication, copied_bytes=0)


def validate_fixed96_z_gather(source, destination, *, fused_into_entry):
    validate_gdn_z(source, destination)
    require(integer(fused_into_entry), "Z gather must name owning fixed entry ordinal")
    require(source.epoch == destination.epoch and source.frame == destination.frame
            and source.live_start <= fused_into_entry <= source.live_end
            and destination.live_start <= fused_into_entry <= destination.live_end,
            "Z gather escapes source/private frame epoch")
    return dict(kind="fixed96-fused-z-gather", fixed_entry=fused_into_entry,
                source_root=source.root_id, destination_root=destination.root_id,
                scalars_per_cta=64, ctas=96, read_nbytes=12288, write_nbytes=12288,
                extra_launches=0, publication="retain existing GDN block/grid barriers",
                coordinate_map="64*cta+local", destination="parent_shard_local")


def validate_gdn_z_buffers(source, destination, plan):
    """Check actual core pool assignment; own-root local and parent view differ.

    A fresh gather root uses compact_local coordinates. An alias over a larger
    backing shard can use parent_shard_local. Neither may expose canonical
    addressing. This does not approve any change to the retained GDN reader.
    """
    from triton.flagmega.ir.bufferization import BufferDescriptor, BufferPlan
    from triton.flagmega.ir.distributed_storage import DistributedBufferStorageKind
    require(isinstance(source, BufferDescriptor) and isinstance(destination, BufferDescriptor)
            and isinstance(plan, BufferPlan), "gather needs actual core descriptors and buffer plan")
    require(source.dtype.value == "bfloat16" and source.shape == (1, 6144)
            and source.distributed_storage_kind.exposes_logical_coordinates
            and source.strides == (6144, 1), "gather input is not complete canonical BF16")
    fullhead48 = destination.distributed_type.placement == Placement((4, 12), "yx", "bb")
    values = 128 if fullhead48 else 64
    require(destination.distributed_type == gdn_z_reader_type(source.distributed_type, fullhead48=fullhead48)
            and destination.strides == (values, 1)
            and destination.distributed_storage_kind is DistributedBufferStorageKind.COMPACT_LOCAL
            and destination.physical_id != source.physical_id and destination.mem_span.nbytes == values * 2,
            "gather destination lost private64 root/type/stride")
    space = plan.memory_space_map[destination.mem_span.buffer.memory_space]
    require(space.name == "block_local_data" and space.kind == "device"
            and space.sharing_scope.value == "block" and space.allocation_scope.value == "function",
            "gather must use a global-backed block-scoped function pool")
    from triton.flagmega.codegen.triton.call_abi import describe_local_buffer_abi
    abi = describe_local_buffer_abi(destination, plan)
    require(abi["coordinate_space"] in ("local", "parent_shard_local")
            and tuple(abi["local_capacity_shape"]) == (1, values), "gather coordinate backing differs")
    return dict(coordinate_space=abi["coordinate_space"], physical_buffer=destination.physical_id,
                storage=space.name, local_capacity=[1, values], scalar_strides=[values, 1], scope="block")


def validate_fixed96_q_gate(q, gate, derived_q):
    validate_q_gate_roots(q, gate, derived_q)
    require(len(derived_q) == 3, "enumerate statistics, q_norm and RoPE private roots")
    require(all(v.epoch == q.epoch == gate.epoch and v.frame == q.frame == gate.frame for v in derived_q),
            "Q/gate derived value escaped frame epoch")
    return dict(q_read_multiplicity=4, gate_read_multiplicity=1, canonical_writers_per_scalar=1,
                canonical_q_modified=False, derived_q_roots=[v.root_id for v in derived_q])


def validate_all_call_frames(execution, frames, *, expected_call_paths):
    from .u4_execution_plan import flatten_execution
    _, calls = flatten_execution(execution)
    paths = tuple("/".join(path) for path, _ in calls)
    require(paths == tuple(expected_call_paths), "frame order differs from typed callers")
    validate_frames(frames, expected_call_paths=paths)
    by_path = {frame.call_path: frame for frame in frames}
    for path, callee in calls:
        require(by_path["/".join(path)].function == callee, "caller binds another function's frame")
    for before, after in zip(paths, paths[1:]):
        require(by_path[before].live_end < by_path[after].live_start, "serial layer frame epochs overlap")
    return dict(calls=len(paths), functions=sorted({f.function for f in frames}),
                epochs=[by_path[p].epoch for p in paths])


def retained_gdn_owner(cta, local):
    """Retained c64 owner facts; internal head Q8/Shared storage is unchanged."""
    require(integer(cta) and cta < 96 and integer(local) and local < 64, "GDN coordinate out of bounds")
    value = 64 * cta + local
    return dict(head=cta // 2, value=value, state_start=value * 128, state_stop=(value + 1) * 128,
                beta_qnorm_knorm_thread=0, decay_thread=32, inactive_slot=1)


__all__ = [
    "make_canonical_reader_view", "gdn_z_reader_type", "CompleteValueTransition", "validate_complete_value_transition",
    "validate_fixed96_z_gather", "validate_fixed96_q_gate", "validate_all_call_frames",
    "validate_gdn_z_buffers", "validate_canonical_cyclic32_buffers",
    "retained_gdn_owner", "cyclic32_coordinate", "q_read_coordinate", "gate_read_coordinate",
    "gdn_z_coordinate", "q_gate_coordinate",
]



def fullhead48_gdn_owner(cta, warp, item):
    """The A55 owner map, checked against emitted result descriptors separately."""
    require(integer(cta) and cta < 48 and integer(warp) and warp < 8
            and integer(item) and item < 16, "fullhead48 GDN coordinate out of bounds")
    value = 128 * cta + warp + 8 * item
    return dict(head=cta, value=value, state_start=value * 128, state_stop=(value + 1) * 128,
                local_value=warp + 8 * item, q8_group=4 * cta + warp if warp < 4 and cta < 40 else None)
