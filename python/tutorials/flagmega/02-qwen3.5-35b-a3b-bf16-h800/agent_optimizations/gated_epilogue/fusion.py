"""Fuse an explicit rounding chain without deleting shared or exported values."""

from dataclasses import replace
from triton.flagmega import ir as fm
from triton.flagmega.errors import IRSchemaError, StageError
from triton.flagmega.ir.distributed_inference import tensor_of
from triton.flagmega.ir.ops.builtin.get_item import GetItem
from triton.flagmega.ir.ops.distributed.sharded_view import ShardedView
from triton.flagmega.ir.ops.tensors.bitcast import Bitcast
from triton.flagmega.ir.ops.tensors.slice_to_shape import SliceToShape
from triton.flagmega.ir.ops.nn._sparse_experts import element_type, lanes, scale_policy
from triton.flagmega.ir.types import data_type
from triton.flagmega.passes.tir.fuse_gather_reduce_norm_apply import _users
from agent_optimizations.scalar_scale.op import ScalarScale
from agent_optimizations.gated_epilogue.op import GatedResidualNormStats


def fuse_gated_epilogue(module):
    if module.stage not in {"frozen_constants", "tuple_boxing_lowered"}:
        raise StageError("Gated epilogue fusion requires a complete pre-TIR module.", stage=module.stage)
    nodes, users = module.node_map, _users(module)
    inserted, replacements, removable = {}, {}, set()
    known_ids = set(nodes)

    def peel(value, matched):
        while value.op == "distributed.sharded_view" and value.effect.is_pure:
            matched.add(value.id)
            value = nodes[value.inputs[0]]
        return value

    def add(value):
        return (value.effect.is_pure and len(value.inputs) == 2
                and (value.op == "math.add" or value.op == "math.vectorized_binary" and value.attrs["binary_op"] == "add"))

    def cast_source(value, matched):
        value = peel(value, matched)
        if (value.op != "ntt.vectorized_cast" or not value.effect.is_pure
                or element_type(data_type(value.attrs["new_type"])) != fm.DType.FLOAT32
                or any(axis not in (1, -1) for axis in value.attrs["vectorize_axes"])):
            return None
        source = nodes[value.inputs[0]]
        if element_type(tensor_of(source.type).dtype) != fm.DType.BFLOAT16:
            return None
        matched.add(value.id)
        return peel(source, matched)

    for stats in module.nodes:
        if stats.op != "nn.norm_stats" or stats.attrs["axis"] not in (1, -1) or not stats.effect.is_pure:
            continue
        matched = {stats.id}
        stats_source = nodes[stats.inputs[0]]
        final_add = peel(stats_source, matched)
        if not add(final_add) or final_add.id in replacements:
            continue
        merged = peel(nodes[final_add.inputs[1]], matched)
        if not add(merged):
            continue
        routed = cast_source(nodes[merged.inputs[0]], matched)
        product = cast_source(nodes[merged.inputs[1]], matched)
        if routed is None or product is None or product.op != ScalarScale.op_name or not product.effect.is_pure:
            continue
        shared = peel(nodes[ScalarScale.value.read(product.inputs)], matched)
        sigmoid = nodes[ScalarScale.scale.read(product.inputs)]
        gate_views = ()
        if sigmoid.op == "tensors.slice_to_shape" and sigmoid.effect.is_pure:
            scalar_view = nodes[sigmoid.inputs[0]]
            if scalar_view.op != "tensors.bitcast" or not scalar_view.effect.is_pure:
                continue
            compute = nodes[scalar_view.inputs[0]]
            if (compute.op != "math.vectorized_unary" or compute.attrs["unary_op"] != "sigmoid"
                    or not compute.effect.is_pure
                    or element_type(tensor_of(compute.type).dtype) != tensor_of(scalar_view.type).dtype):
                continue
            # A pointwise sigmoid commutes with this scalar prefix projection.
            # Feed the unactivated projected logit to the existing fused ABI;
            # neither discarded vector tails nor BF16 rounding are changed.
            gate = nodes[compute.inputs[0]]
            gate_views = (scalar_view, sigmoid)
            matched.update((scalar_view.id, compute.id))
        elif sigmoid.op == "math.sigmoid" and sigmoid.effect.is_pure:
            gate = nodes[sigmoid.inputs[0]]
        else:
            continue
        residual = peel(nodes[final_add.inputs[0]], matched)
        matched.update((final_add.id, merged.id, product.id, sigmoid.id))
        desired = stats_source.type
        prefix = final_add.id + ".gated_epilogue"
        additions = []

        def make(definition, arguments, attrs, identity):
            if identity in known_ids:
                raise ValueError(f"Gated epilogue node name collides: {identity}")
            prepared = definition.prepare(tuple(arguments), attrs)
            node = fm.Node(identity, definition.op_name, tuple(n.id for n in arguments), prepared.result_type,
                           attrs=prepared.attrs, effect=prepared.effect)
            additions.append(node)
            return node

        def view(value, name):
            if not isinstance(desired, fm.DistributedType):
                return value
            policies = (desired.axis_policies[0], scale_policy(
                desired.axis_policies[1], lanes(tensor_of(desired).dtype), lanes(tensor_of(value.type).dtype)))
            target = fm.DistributedType(tensor_of(value.type), policies, desired.placement)
            return value if value.type == target else make(ShardedView, (value,), {"new_type": target}, prefix + "." + name)

        try:
            if gate_views:
                gate = make(Bitcast, (gate,), gate_views[0].attrs, prefix + ".gate_view")
                gate = make(SliceToShape, (gate,), gate_views[1].attrs, prefix + ".gate_slice")
            arguments = (view(routed, "routed"), view(shared, "shared"), gate, view(residual, "residual"))
            fused = make(GatedResidualNormStats, arguments, {"use_mean": stats.attrs["use_mean"]}, prefix)
            value = make(GetItem, (fused,), {"index": 0}, prefix + ".value")
            if fused.type.fields[1] != stats.type or tensor_of(value.type) != tensor_of(final_add.type):
                continue
            if value.type == final_add.type:
                restored = replace(final_add, op=GetItem.op_name, inputs=(fused.id,), attrs={"index": 0})
                additions.pop()  # The preserved public ID is the tuple projection.
            else:
                prepared = ShardedView.prepare((value,), {"new_type": final_add.type})
                restored = replace(final_add, op=ShardedView.op_name, inputs=(value.id,),
                                   attrs=prepared.attrs, effect=prepared.effect)
        except IRSchemaError:
            continue
        inserted[final_add.id] = tuple(additions)
        known_ids.update(n.id for n in additions)
        replacements[final_add.id] = restored
        replacements[stats.id] = replace(stats, op=GetItem.op_name, inputs=(fused.id,), attrs={"index": 1})
        removable.update(matched - {final_add.id, stats.id})

    # Retain a shared intermediate and recursively retain anything it needs.
    while True:
        retained = {name for name in removable
                    if any(user not in removable and user not in replacements for user in users[name])}
        if not retained:
            break
        removable.difference_update(retained)
    changed = removable | replacements.keys()
    obsolete = {p.id for p in module.selection_points if p.owner in changed}
    result = replace(module,
                     nodes=tuple(n for original in module.nodes if original.id not in removable
                                 for n in (*inserted.get(original.id, ()), replacements.get(original.id, original))),
                     selection_points=tuple(p for p in module.selection_points if p.id not in obsolete),
                     selections=tuple(s for s in module.selections if s.point_id not in obsolete))
    fm.verify_module(result)
    return result
