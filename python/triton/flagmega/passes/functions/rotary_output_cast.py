# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Absorb closed rotary tuple rounding demands at the original state read."""
from collections import defaultdict
from dataclasses import replace

from triton.flagmega.ir import DType, TensorType, TupleType, VectorType, verify_module
from triton.flagmega.rules.neutral._utility import make_node


def absorb_rotary_output_casts(module):
    users = defaultdict(list)
    for node in module.nodes:
        for source in node.inputs:
            users[source].append(node)
    public = {name for function in module.functions for name in (*function.parameters, *function.outputs)}
    changed, removed = {}, set()
    for producer in module.nodes:
        if (producer.op != 'nn.rotary_embedding' or len(producer.inputs) != 2 or producer.id in public
                or producer.attrs.get('post_ops') or producer.attrs.get('pre_ops')
                or not isinstance(producer.type, TupleType) or len(producer.type.fields) != 2
                or any(not isinstance(field, TensorType) for field in producer.type.fields)):
            continue
        projections = users[producer.id]
        if (not projections
                or any(node.op != 'builtin.get_item' or node.id in public or node.attrs['index'] not in (0, 1)
                       for node in projections)):
            continue
        casts = [cast for projection in projections for cast in users[projection.id]]
        if (not casts or {projection.attrs['index']
                          for projection in projections} != {0, 1}
                or any(not users[projection.id] for projection in projections)):
            continue
        target = casts[0].type
        if not isinstance(target, TensorType):
            continue
        dtype = target.dtype.elem_type if isinstance(target.dtype, VectorType) else target.dtype
        if dtype is not DType.BFLOAT16 or any(cast.type != target for cast in casts):
            continue
        valid = True
        for cast in casts:
            if cast.op not in {'tensors.cast', 'ntt.vectorized_cast'} or not cast.effect.is_pure:
                valid = False
                break
            if cast.op == 'ntt.vectorized_cast':
                rank = module.node_map[cast.inputs[0]].type.rank
                if any(axis % rank != rank - 1 for axis in cast.attrs['vectorize_axes']):
                    valid = False
                    break
        if not valid:
            continue
        lanes = target.dtype.lanes if isinstance(target.dtype, VectorType) else ()
        candidate = make_node(producer.op, producer.id, tuple(module.node_map[name] for name in producer.inputs),
                              {**producer.attrs, 'output_dtype': DType.BFLOAT16.value, 'output_lanes': lanes},
                              producer.metadata)
        if candidate.type != TupleType((target, target)) or candidate.effect != producer.effect:
            continue
        # The complete raw tuple is private. Update its type and every live
        # field boundary atomically; no FP32 restoration or duplicate read.
        changed[producer.id] = candidate
        for projection in projections:
            removed.add(projection.id)
            for cast in users[projection.id]:
                changed[cast.id] = make_node('builtin.get_item', cast.id, (candidate, ),
                                             {'index': projection.attrs['index']}, cast.metadata)
    if not changed:
        return module
    nodes = tuple(changed.get(node.id, node) for node in module.nodes if node.id not in removed)
    owners = set(changed) | removed
    points = tuple(point for point in module.selection_points if point.owner not in owners)
    identities = {point.id for point in points}
    result = replace(module, nodes=nodes, selection_points=points,
                     selections=tuple(selection for selection in module.selections if selection.point_id in identities))
    assert [(node.id, node.effect) for node in result.nodes if not node.effect.is_pure] == [(node.id, node.effect)
                                                                                            for node in module.nodes
                                                                                            if not node.effect.is_pure]
    return verify_module(result)


__all__ = ['absorb_rotary_output_casts']
