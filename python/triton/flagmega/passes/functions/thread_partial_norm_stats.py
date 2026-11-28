# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Opt-in owner-local statistics across a closed reusable-function ABI.

Unlike sinking an already selected boxing, this transformation chooses local
additive statistics before a broadcast value view. Unordered NormStats permits
the resulting reassociation. Ordered reductions are deliberately rejected.
"""
from dataclasses import replace

from triton.flagmega.errors import IRSchemaError
from triton.flagmega.ir import DistributedType, SBPBroadCast, TupleType, verify_module
from triton.flagmega.ir.ops.core import get_definition
from triton.flagmega.ir.ops.nn.norm_stats import NormStats
from triton.flagmega.ir.op_fusion import has_ops


def thread_partial_norm_stats_across_function_boundaries(
    module,
    *,
    function_name,
    stats_parameter_index,
    stats_output_index,
):
    """Retype one closed internal statistics channel and fuse its consumers.

    The caller explicitly identifies the channel; no default pipeline changes.
    Every actual must originate from a compatible local NormStats or another
    call's selected statistics result. Every use must remain in this channel or
    be a NormApply statistics operand. Public scalar statistics are not changed.
    """
    verify_module(module)
    node_map = module.node_map
    function = next((f for f in module.functions if f.name == function_name), None)
    if function is None or function.name == module.entry:
        raise IRSchemaError("Partial statistics requires an existing internal function.")
    if not (0 <= stats_parameter_index < len(function.parameters) and 0 <= stats_output_index < len(function.outputs)):
        raise IRSchemaError("Partial statistics function channel index is out of range.")
    parameter_id = function.parameters[stats_parameter_index]
    output_id = function.outputs[stats_output_index]
    old_type = node_map[parameter_id].type
    if not isinstance(old_type, DistributedType):
        raise IRSchemaError("Partial statistics requires a distributed statistics channel.")
    if old_type.partial is not None:
        # A completed invocation is a fixed point, with ordinary module
        # verification checking the formal/actual and result types.
        if node_map[output_id].type == old_type:
            return module
        raise IRSchemaError("Partially transformed statistics function ABI.")
    if any(not isinstance(p, SBPBroadCast) for p in old_type.axis_policies):
        raise IRSchemaError("Original statistics channel must be broadcast.")
    if node_map[output_id].type != old_type:
        raise IRSchemaError("Statistics formal and result must have identical types.")
    calls = tuple(n for n in module.nodes if n.op == "builtin.call" and n.attrs.get("callee") == function_name)
    if not calls or any(not isinstance(n.type, TupleType) for n in calls):
        raise IRSchemaError("Partial statistics requires explicit tuple-returning calls.")
    call_ids = {n.id for n in calls}
    users = {n.id: [] for n in module.nodes}
    for node in module.nodes:
        for index, value in enumerate(node.inputs):
            users[value].append((node, index))
    for call in calls:
        if has_ops(call.attrs):
            raise IRSchemaError("Fused call arithmetic cannot change its statistics channel.")
        if any(n.op != "builtin.get_item" for n, _ in users[call.id]):
            raise IRSchemaError("The call tuple must have explicit field projections only.")
        if any(call.id in f.outputs for f in module.functions):
            raise IRSchemaError("An exported call tuple cannot change its statistics field.")

    def local_stats(node):
        if node.op != "nn.norm_stats" or node.type != old_type:
            raise IRSchemaError("Statistics source must be an explicit broadcast NormStats.")
        if has_ops(node.attrs):
            raise IRSchemaError("Fused statistics arithmetic cannot be reassociated.")
        attrs = NormStats.normalize_attrs(node.attrs)
        if attrs.get("reduction_group_size", 1) != 1 or attrs.get("reduction_order", "unspecified") != "unspecified":
            raise IRSchemaError("Ordered complete-vector statistics cannot be reassociated.")
        source = node_map[node.inputs[0]]
        seen = set()
        while isinstance(source.type, DistributedType):
            inferred = NormStats.infer_type((source, ), attrs)
            if inferred.partial is not None:
                if replace(inferred, partial=None) != old_type:
                    raise IRSchemaError("Local statistics has incompatible tensor or mesh.")
                return source, inferred, attrs
            if (source.id in seen or source.op not in {"distributed.sharded_view", "tensors.reshape"}
                    or len(source.inputs) != 1 or not source.effect.is_pure or has_ops(source.attrs)):
                break
            seen.add(source.id)
            previous = node_map[source.inputs[0]]
            if (not isinstance(previous.type, DistributedType) or previous.type.tensor != source.type.tensor
                    or previous.type.placement != source.type.placement):
                break
            source = previous
        raise IRSchemaError("NormStats has no same-tensor owner-local source before its broadcast view.")

    _, partial_type, _ = local_stats(node_map[output_id])
    changed = {parameter_id: replace(node_map[parameter_id], type=partial_type)}
    active = set()

    def trace(value):
        if value in changed:
            return
        if value in active:
            raise IRSchemaError("Cyclic statistics dataflow is unsupported.")
        active.add(value)
        node = node_map[value]
        if has_ops(node.attrs):
            raise IRSchemaError("Fused statistics-channel arithmetic cannot be retyped as partial.")
        if node.op == "nn.norm_stats":
            source, ty, attrs = local_stats(node)
            if ty != partial_type:
                raise IRSchemaError("All call statistics require the same owner layout.")
            prepared = NormStats.prepare((source, ), attrs)
            changed[value] = replace(node, inputs=(source.id, ), type=ty, attrs=prepared.attrs, effect=prepared.effect)
        elif node.op == "builtin.get_item" and node.inputs[0] in call_ids and node.attrs["index"] == stats_output_index:
            changed[value] = replace(node, type=partial_type)
        elif (node.op == "tensors.reshape" and node.type == old_type and node_map[node.inputs[0]].type == old_type):
            trace(node.inputs[0])
            changed[value] = replace(node, type=partial_type)
        else:
            raise IRSchemaError("Statistics actual is not a closed local-statistics/call-result chain.")
        active.remove(value)

    trace(output_id)
    for call in calls:
        trace(call.inputs[stats_parameter_index])
        fields = list(call.type.fields)
        if fields[stats_output_index] != old_type:
            raise IRSchemaError("Call statistics result disagrees with the function ABI.")
        fields[stats_output_index] = partial_type
        changed[call.id] = replace(call, type=TupleType(tuple(fields)))
        for user, _ in users[call.id]:
            if user.attrs["index"] == stats_output_index:
                trace(user.id)
    # Identity aliases are part of the same logical channel, not new owners.
    while True:
        extra = [
            n.id
            for n in module.nodes
            if n.op == "tensors.reshape" and n.inputs[0] in changed and n.type == old_type and n.id not in changed
        ]
        if not extra:
            break
        for value in extra:
            trace(value)

    channel_ids = {name for name, node in changed.items() if node.type == partial_type}
    for value in channel_ids:
        if any(value in f.outputs for f in module.functions if f.name != function_name):
            raise IRSchemaError("Public or unrelated function statistics output cannot change type.")
        if value in function.outputs and value != output_id:
            raise IRSchemaError("Statistics is also exported through an unselected result.")
        for user, index in users[value]:
            if user.op == "nn.norm_apply" and index == 1:
                if has_ops(user.attrs):
                    raise IRSchemaError("Fused NormApply arithmetic must remain with its original ABI.")
                attrs = {**dict(user.attrs), "materialized_stats_type": old_type, "has_bias": True}
                arguments = (changed[value], *(node_map[k] for k in (user.inputs[0], user.inputs[2], user.inputs[3])))
                prepared = get_definition("ntt.gather_reduce_norm_apply").prepare(arguments, attrs)
                if prepared.result_type != user.type:
                    raise IRSchemaError("Partial-statistics apply changed the normalization result type.")
                changed[user.id] = replace(user, op="ntt.gather_reduce_norm_apply",
                                           inputs=tuple(n.id for n in arguments), attrs=prepared.attrs,
                                           effect=prepared.effect)
            elif user.id in call_ids and index == stats_parameter_index:
                pass
            elif user.op == "tensors.reshape" and user.id in channel_ids:
                pass
            else:
                raise IRSchemaError("Statistics has a shared consumer outside the selected channel.")

    points = tuple(p for p in module.selection_points if p.owner not in changed)
    point_ids = {p.id for p in points}
    result = replace(module, nodes=tuple(changed.get(n.id, n) for n in module.nodes), selection_points=points,
                     selections=tuple(s for s in module.selections if s.point_id in point_ids))
    return verify_module(result)


__all__ = ["thread_partial_norm_stats_across_function_boundaries"]
