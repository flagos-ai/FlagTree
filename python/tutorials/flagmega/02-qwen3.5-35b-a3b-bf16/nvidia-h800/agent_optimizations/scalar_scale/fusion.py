"""Remove only private pure scalar broadcasts and scalar-preserving relayouts."""

from dataclasses import replace
from triton.flagmega.errors import IRSchemaError, StageError
from triton.flagmega.ir.distributed_inference import tensor_of
from triton.flagmega.ir.types import VectorType
from triton.flagmega.passes.tir.fuse_gather_reduce_norm_apply import _users
from .op import ScalarScale


def fuse_scalar_scale(module):
    if module.stage not in {"frozen_constants", "tuple_boxing_lowered"}:
        raise StageError("ScalarScale fusion must precede TIR proposal.", stage=module.stage)
    nodes, users = module.node_map, _users(module)
    removed, replacements = set(), {}
    for product in module.nodes:
        if not (product.op == "math.mul" or
                product.op == "math.vectorized_binary" and product.attrs["binary_op"] == "mul"):
            continue
        if not product.effect.is_pure:
            continue
        for index in (1, 0):
            current = nodes[product.inputs[index]]
            consumer = product.id
            chain = []
            while current.op in {"tensors.pack", "tensors.unpack", "tensors.bitcast", "distributed.sharded_view"}:
                if len(current.inputs) != 1 or not current.effect.is_pure or users[current.id] != (consumer,):
                    break
                if current.op == "tensors.bitcast":
                    source_dtype = tensor_of(nodes[current.inputs[0]].type).dtype
                    result_dtype = tensor_of(current.type).dtype
                    source_element = source_dtype.elem_type if isinstance(source_dtype, VectorType) else source_dtype
                    result_element = result_dtype.elem_type if isinstance(result_dtype, VectorType) else result_dtype
                    if source_element != result_element:
                        break
                chain.append(current.id)
                consumer, current = current.id, nodes[current.inputs[0]]
            if (current.op != "tensors.broadcast_to" or not current.effect.is_pure
                    or users[current.id] != (consumer,)):
                continue
            scalar = nodes[current.inputs[0]]
            # Packing or sharding a splat must not change its element type.
            result_dtype, source_dtype = tensor_of(current.type).dtype, tensor_of(scalar.type).dtype
            result_element = result_dtype.elem_type if isinstance(result_dtype, VectorType) else result_dtype
            source_element = source_dtype.elem_type if isinstance(source_dtype, VectorType) else source_dtype
            if result_element != source_element:
                continue
            value = nodes[product.inputs[1 - index]]
            try:
                prepared = ScalarScale.prepare((value, scalar), {})
            except IRSchemaError:
                continue
            if prepared.result_type != product.type:
                continue
            replacements[product.id] = replace(
                product, op=ScalarScale.op_name, inputs=(value.id, scalar.id), attrs=prepared.attrs,
                type=prepared.result_type, effect=prepared.effect,
                metadata={**product.metadata, "fused_scalar_materializations": (*chain, current.id)},
            )
            removed.update((*chain, current.id))
            break
    if not replacements:
        return module
    changed = removed | replacements.keys()
    obsolete = {point.id for point in module.selection_points if point.owner in changed}
    return replace(module, nodes=tuple(replacements.get(n.id, n) for n in module.nodes if n.id not in removed),
                   selection_points=tuple(p for p in module.selection_points if p.id not in obsolete),
                   selections=tuple(s for s in module.selections if s.point_id not in obsolete))
