# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Explicitly commute an elementwise widening with a broadcast storage view."""
from dataclasses import replace

from triton.flagmega.errors import IRSchemaError
from triton.flagmega.ir import DType, DistributedType, SBPBroadCast, VectorType, verify_module
from triton.flagmega.ir.op_fusion import has_ops
from triton.flagmega.rules.neutral._utility import make_node


def commute_vectorized_widening_before_broadcast(module, *, cast_id):
    """Preserve a BF16-to-F32 same-lane cast while evaluating it on its owners.

    This is opt-in and creates a real cast followed by a real sharded view.
    The original cast's identity and result type remain at the view, so every
    existing value consumer retains its logical coordinates and scalar dtype.
    No arithmetic-bearing view, partial value, or lane repacking is accepted.
    """
    verify_module(module)
    cast = module.node_map.get(cast_id)
    if (cast is None or cast.op != 'ntt.vectorized_cast' or len(cast.inputs) != 1
            or not cast.effect.is_pure or has_ops(cast.attrs)):
        raise IRSchemaError('Widening commute requires an unfused pure vectorized cast.')
    view = module.node_map[cast.inputs[0]]
    if (view.op != 'distributed.sharded_view' or len(view.inputs) != 1
            or not view.effect.is_pure or has_ops(view.attrs)):
        raise IRSchemaError('Widening commute requires an unfused pure sharded view.')
    source = module.node_map[view.inputs[0]]
    types = (source.type, view.type, cast.type)
    if (any(not isinstance(t, DistributedType) or t.partial is not None or t.exclusive is not None
            for t in types)
            or source.type.placement != view.type.placement or cast.type.placement != view.type.placement
            or source.type.tensor != view.type.tensor
            or source.type.axis_policies == view.type.axis_policies
            or not all(isinstance(p, SBPBroadCast) for p in view.type.axis_policies)
            or cast.type.axis_policies != view.type.axis_policies):
        raise IRSchemaError('Widening commute requires the same nonpartial tensor and mesh before broadcast.')
    before, after = source.type.tensor, cast.type.tensor
    if (not isinstance(before.dtype, VectorType) or not isinstance(after.dtype, VectorType)
            or before.dtype.elem_type is not DType.BFLOAT16 or after.dtype.elem_type is not DType.FLOAT32
            or before.dtype.lanes != after.dtype.lanes
            or before.shape != after.shape or before.layout != after.layout):
        raise IRSchemaError('Widening commute only supports same-lane BF16-to-F32 conversion.')
    owner_id = cast_id + '.owner_widening'
    if owner_id in module.node_map:
        raise IRSchemaError('Widening commute owner identity already exists.')
    metadata = {'introduced_by': 'CommuteVectorizedWideningBeforeBroadcast', 'source_cast': cast_id}
    owner = make_node(cast.op, owner_id, (source,), cast.attrs, metadata)
    expected = replace(source.type, tensor=cast.type.tensor)
    if owner.type != expected:
        raise IRSchemaError('Widening commute changed owner coordinates or layout.')
    output = make_node('distributed.sharded_view', cast_id, (owner,), {'new_type': cast.type}, metadata)
    if output.type != cast.type:
        raise IRSchemaError('Widening commute changed the original result ABI.')
    nodes = tuple(item for node in module.nodes
                  for item in ((owner, output) if node.id == cast_id else (node,)))
    points = tuple(p for p in module.selection_points if p.owner != cast_id)
    point_ids = {p.id for p in points}
    return verify_module(replace(module, nodes=nodes, selection_points=points,
                                 selections=tuple(s for s in module.selections if s.point_id in point_ids)))


__all__ = ['commute_vectorized_widening_before_broadcast']
