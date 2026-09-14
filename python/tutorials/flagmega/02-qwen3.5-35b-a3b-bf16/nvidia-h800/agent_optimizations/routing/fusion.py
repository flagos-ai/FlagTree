"""Post-distribution SSA fusion; preserve exported/shared numerical stages."""

from dataclasses import replace

from triton.flagmega.errors import IRSchemaError, StageError
from triton.flagmega.ir.axis import normalize_axis
from triton.flagmega.ir.distributed_inference import tensor_of
from triton.flagmega.passes.tir.fuse_gather_reduce_norm_apply import _users
from .op import StagedRouting


def fuse_staged_routing(module):
    if module.stage not in {"frozen_constants", "tuple_boxing_lowered"}:
        raise StageError("Routing fusion must run before TIR candidate proposal.", stage=module.stage)
    users = _users(module)
    nodes = module.node_map
    removed, replacements = set(), {}
    for division in module.nodes:
        if division.op != "math.div":
            continue
        values, broadcast = (nodes[n] for n in division.inputs)
        if values.op != "builtin.get_item" or values.attrs["index"] != 0 or broadcast.op != "tensors.broadcast_to":
            continue
        reduction = nodes[broadcast.inputs[0]]
        topk = nodes[values.inputs[0]]
        if reduction.op != "math.reduce_sum" or reduction.inputs != (values.id, ) or topk.op != "tensors.top_k":
            continue
        softmax = nodes[topk.inputs[0]]
        if softmax.op != "nn.softmax" or not topk.attrs["largest"] or not topk.attrs["sorted"]:
            continue
        rank = tensor_of(values.type).rank
        axis = normalize_axis(topk.attrs["axis"], rank)
        if (normalize_axis(softmax.attrs["axis"], rank) != axis
                or tuple(normalize_axis(a, rank) for a in reduction.attrs["axes"]) != (axis, )
                or not reduction.attrs["keep_dims"] or broadcast.type != values.type):
            continue
        topk_users = [nodes.get(n) for n in users[topk.id]]
        indices = [n for n in topk_users if n is not None and n.op == "builtin.get_item" and n.attrs["index"] == 1]
        if len(topk_users) != 2 or len(indices) != 1:
            continue
        if (users[softmax.id] != (topk.id, ) or set(users[values.id]) != {division.id, reduction.id}
                or users[reduction.id] != (broadcast.id, ) or users[broadcast.id] != (division.id, )):
            continue
        source = nodes[softmax.inputs[0]]
        extent = tensor_of(source.type).shape[axis]
        # Explicit single-tile implementation contract, not a model-name test.
        if not extent.is_fixed or not 0 < extent.fixed_value <= 256:
            continue
        attrs = {"k": topk.attrs["k"], "axis": axis, "index_dtype": topk.attrs["index_dtype"]}
        try:
            prepared = StagedRouting.prepare((source, ), attrs)
        except IRSchemaError:
            continue
        if prepared.result_type.fields != (division.type, indices[0].type):
            continue
        replacements[topk.id] = replace(topk, op=StagedRouting.op_name, inputs=(source.id, ),
                                         type=prepared.result_type, effect=prepared.effect, attrs=prepared.attrs,
                                         metadata={**topk.metadata, "staged_routing": (softmax.id, division.id)})
        replacements[division.id] = replace(division, op="builtin.get_item", inputs=(topk.id, ), attrs={"index": 0})
        removed.update((softmax.id, values.id, reduction.id, broadcast.id))
    if not replacements:
        return module
    # Retire obsolete node-owned decisions, not hashes or downstream plans.
    changed = removed | replacements.keys()
    obsolete = {p.id for p in module.selection_points if p.owner in changed}
    return replace(module, nodes=tuple(replacements.get(n.id, n) for n in module.nodes if n.id not in removed),
                   selection_points=tuple(p for p in module.selection_points if p.id not in obsolete),
                   selections=tuple(s for s in module.selections if s.point_id not in obsolete))
