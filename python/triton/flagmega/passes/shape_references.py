# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Explicitly rebind closed shape-only aliases to an equal-geometry value."""
from dataclasses import replace

from triton.flagmega.errors import IRSchemaError
from triton.flagmega.ir import DistributedType, VectorType, verify_module
from triton.flagmega.ir.distributed_inference import tensor_of
from triton.flagmega.ir.memory_effect import MemoryEffect
from triton.flagmega.ir.op_fusion import has_ops
from triton.flagmega.ir.ops.core import get_definition
from triton.flagmega.rules.neutral._utility import make_node


def rebind_shape_only_view_users(module, *, view_id, replacement_id):
    """Redirect only schema-declared no-value reads, preserving call effects.

    A pure cast/view dependency may move before the earliest reference only
    when all other dependencies already dominate it. The effectful references
    themselves retain their identity, order, output type and effect. A shared
    real-value reader or exported alias is an explicit refusal.
    """
    verify_module(module)
    nodes = module.node_map
    view, replacement = nodes.get(view_id), nodes.get(replacement_id)
    if (view is None or view.op != 'distributed.sharded_view' or not view.effect.is_pure
            or has_ops(view.attrs) or replacement is None or not replacement.effect.is_pure
            or has_ops(replacement.attrs) or view.id == replacement.id):
        raise IRSchemaError('Shape-reference rebinding requires distinct pure unfused views/values.')
    if any(view.id in (*f.parameters, *f.outputs) for f in module.functions):
        raise IRSchemaError('An exported view cannot be rebound as a private shape reference.')
    def geometry(ty):
        tensor = tensor_of(ty)
        lanes = tensor.dtype.lanes if isinstance(tensor.dtype, VectorType) else ()
        return (tensor.shape, tensor.layout, lanes,
                (ty.placement, ty.axis_policies) if isinstance(ty, DistributedType) else None)
    if (geometry(view.type) != geometry(replacement.type)
            or any(getattr(t, 'partial', None) is not None or getattr(t, 'exclusive', None) is not None
                   for t in (view.type, replacement.type))):
        raise IRSchemaError('Shape-reference rebinding requires equal complete tensor coordinates.')
    users = tuple(n for n in module.nodes if view.id in n.inputs)
    if not users:
        raise IRSchemaError('Shape-reference rebinding requires a live closed reference set.')
    updated = {}
    for user in users:
        if has_ops(user.attrs):
            raise IRSchemaError('A shape reference with fused arithmetic cannot be rebound.')
        definition = get_definition(user.op)
        positions = {p.input_index for p in definition.input_parameters
                     if not p.variadic and p.memory_effect == MemoryEffect.NONE}
        if any(i not in positions for i, key in enumerate(user.inputs) if key == view.id):
            raise IRSchemaError('Shared view has a real-value reader; shape rebinding is illegal.')
        inputs = tuple(replacement if key == view.id else nodes[key] for key in user.inputs)
        candidate = make_node(user.op, user.id, inputs, user.attrs, user.metadata)
        if candidate.type != user.type or candidate.effect != user.effect:
            raise IRSchemaError('Shape-reference rebinding changed consumer output type or effect.')
        updated[user.id] = candidate
    order = {n.id: i for i, n in enumerate(module.nodes)}
    anchor = min(users, key=lambda n: order[n.id]).id
    moving, active = set(), set()
    allowed = {'ntt.vectorized_cast', 'tensors.cast', 'distributed.sharded_view', 'tensors.bitcast'}
    def dependencies(key):
        if key == view.id:
            raise IRSchemaError('The replacement depends on the old shape view.')
        if order[key] < order[anchor] or key in moving:
            return
        if key in active:
            raise IRSchemaError('Cyclic replacement dependencies cannot be moved.')
        node = nodes[key]
        if node.op not in allowed or not node.effect.is_pure or has_ops(node.attrs):
            raise IRSchemaError('Only pure cast/view dependencies may precede a shape reference.')
        active.add(key)
        for source in node.inputs:
            dependencies(source)
        active.remove(key)
        moving.add(key)
    dependencies(replacement.id)
    moved = tuple(n for n in module.nodes if n.id in moving)
    result = []
    for node in module.nodes:
        if node.id == anchor:
            result.extend(moved)
        if node.id != view.id and node.id not in moving:
            result.append(updated.get(node.id, node))
    invalidated = {view.id, *updated}
    points = tuple(p for p in module.selection_points if p.owner not in invalidated)
    point_ids = {p.id for p in points}
    return verify_module(replace(module, nodes=tuple(result), selection_points=points,
                                 selections=tuple(s for s in module.selections if s.point_id in point_ids)))


__all__ = ['rebind_shape_only_view_users']
