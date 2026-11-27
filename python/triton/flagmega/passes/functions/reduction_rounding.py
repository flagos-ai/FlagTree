# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Keep BF16 reduction boundaries in the reduction and materialization owners."""

from dataclasses import replace
from math import prod

from triton.flagmega.ir import DType, DistributedType, Node, SBP, TensorType, VectorType, verify_module
from triton.flagmega.ir.axis import normalize_axis
from triton.flagmega.ir.distributed_inference import tensor_of
from triton.flagmega.ir.distributed_type import local_tensor_type
from triton.flagmega.ir.ops.ntt.vectorized_cast import cast_vector_axes
from triton.flagmega.ir.ops.tensors.bitcast import Bitcast
from triton.flagmega.passes.functions.remove_unused import remove_unused_functions
from triton.flagmega.rules.neutral._utility import make_node


def _scalar_kind(kind):
    return tensor_of(kind).dtype


def _bf16(kind):
    tensor = tensor_of(kind)
    converted = replace(tensor, dtype=DType.BFLOAT16)
    return replace(kind, tensor=converted) if isinstance(kind, DistributedType) else converted


def _materialized(kind):
    return replace(kind, partial=None) if isinstance(kind, DistributedType) else kind


def _scalarized_bf16(kind):
    # Type proof only: no byte-view operation is emitted into compute.
    source = Node("proof", "builtin.var", (), _materialized(kind))
    return Bitcast.infer_type((source,), {"dtype": "bfloat16"})


def _widening_path(node, nodes):
    """Find an exact BF16->FP32 widening with final-axis component order."""
    if node.op == "tensors.cast" and _scalar_kind(node.type) == DType.FLOAT32:
        source = nodes[node.inputs[0]]
        if _scalar_kind(source.type) == DType.BFLOAT16:
            return source, (node,)
        return None
    if node.op != "tensors.bitcast" or _scalar_kind(node.type) != DType.FLOAT32:
        return None
    cast = nodes[node.inputs[0]]
    if cast.op != "ntt.vectorized_cast":
        return None
    source = nodes[cast.inputs[0]]
    original, widened = tensor_of(source.type), tensor_of(cast.type)
    if (not isinstance(original.dtype, VectorType) or not isinstance(widened.dtype, VectorType)
            or original.dtype.elem_type != DType.BFLOAT16 or widened.dtype.elem_type != DType.FLOAT32):
        return None
    source_axes, result_axes = cast_vector_axes(original.dtype.lanes, widened.dtype.lanes,
                                               cast.attrs["vectorize_axes"], original.rank)
    if any(axis != original.rank - 1 for axis in (*source_axes, *result_axes)):
        return None
    if _scalarized_bf16(source.type) != _bf16(node.type):
        return None
    return source, (cast, node)


def fuse_materialized_reduction_rounding(module):
    """Absorb private widening/reduce/(Boxing)/narrowing chains pre-TIR.

    A local complete FP32 reduction may write its final BF16 result directly;
    a subsequent gather only moves that result. Across a Sum partial, the
    BF16 local result is exact only with one reduced element per physical
    owner. Existing Boxing then widens BF16 partials for its FP32 sum and
    rounds once to BF16. Other local fan-in is deliberately left unchanged.

    Packed results may replace a scalar narrowing only when every user is a
    supported MhcPost value input and the value is not a public result. No
    Pack/Unpack/Bitcast adapter is introduced to restore a missing consumer.
    """
    if module.kernel_definitions:
        raise ValueError("Reduction rounding fusion requires materialized distributed graph IR before TIR.")
    nodes = module.node_map
    users = {node.id: [] for node in module.nodes}
    for node in module.nodes:
        for value in node.inputs:
            users[value].append(node)
    exports = {value for function in module.functions for value in function.outputs}
    replacements, changed, evidence = {}, set(), []
    for narrow in module.nodes:
        if (narrow.op != "tensors.cast" or _scalar_kind(narrow.type) != DType.BFLOAT16
                or len(narrow.inputs) != 1):
            continue
        tail = nodes[narrow.inputs[0]]
        boxing = tail if tail.op in {"distributed.boxing", "distributed.force_boxing"} else None
        reduction = nodes[tail.inputs[0]] if boxing is not None else tail
        if (reduction.op != "math.reduce_sum" or _scalar_kind(reduction.type) != DType.FLOAT32
                or len(reduction.inputs) != 1 or _scalar_kind(tail.type) != DType.FLOAT32):
            continue
        widening = _widening_path(nodes[reduction.inputs[0]], nodes)
        if widening is None:
            continue
        source, adapters = widening
        # The pipeline may retain its high_level dialect name after selecting
        # distributed types. Actual owner contracts, not that label, prove
        # that the local fan-in used below is already materialized.
        if not isinstance(source.type, DistributedType) or not isinstance(reduction.type, DistributedType):
            continue
        chain = (*adapters, reduction, *((boxing,) if boxing is not None else ()), narrow)
        if any(not node.effect.is_pure or "pre_ops" in node.attrs or "post_ops" in node.attrs for node in chain):
            continue
        if any(node.id in exports or users[node.id] != [following]
               for node, following in zip(chain[:-1], chain[1:], strict=True)):
            continue
        if isinstance(source.type, DistributedType) and (source.type.partial is not None or source.type.exclusive is not None):
            continue
        partial = reduction.type.partial if isinstance(reduction.type, DistributedType) else None
        if isinstance(tail.type, DistributedType) and (tail.type.partial is not None or tail.type.exclusive is not None):
            continue
        if partial is not None:
            if boxing is None or partial != SBP.partial(partial.axes):
                continue
            local = local_tensor_type(source.type)
            axes = tuple(normalize_axis(axis, local.rank) for axis in reduction.attrs["axes"])
            if any(not local.shape[axis].is_fixed for axis in axes) or prod(local.shape[axis].fixed_value for axis in axes) != 1:
                continue
        source_tensor = tensor_of(source.type)
        axes = tuple(normalize_axis(axis, source_tensor.rank) for axis in reduction.attrs["axes"])
        if isinstance(source_tensor.dtype, VectorType) and source_tensor.rank - 1 in axes:
            # The old scalar reduction would also consume packet components.
            continue
        identity = narrow.id + ".rounded_sum" if boxing is not None else narrow.id
        if boxing is not None and identity in nodes:
            continue
        reduced = make_node("math.reduce_sum", identity, (source,), dict(reduction.attrs),
                            {**reduction.metadata, "formed_by": "FuseMaterializedReductionRounding"})
        if (_scalarized_bf16(reduced.type) != _materialized(_bf16(reduction.type))
                or (reduced.type.partial if isinstance(reduced.type, DistributedType) else None) != partial):
            continue
        if boxing is not None:
            if isinstance(boxing.type, DistributedType):
                if any(policy != SBP.broadcast() for policy in boxing.type.axis_policies):
                    continue
                target = replace(boxing.type, tensor=tensor_of(reduced.type))
            elif isinstance(boxing.type, TensorType):
                target = tensor_of(reduced.type)
            else:
                continue
            result = make_node(boxing.op, narrow.id, (reduced,), {**boxing.attrs, "new_type": target},
                               {**boxing.metadata, "formed_by": "FuseMaterializedReductionRounding"})
            new_nodes = (reduced, result)
        else:
            result, new_nodes = reduced, (reduced,)
        if _scalarized_bf16(result.type) != narrow.type:
            continue
        type_changed = result.type != narrow.type
        if type_changed:
            if narrow.id in exports or not users[narrow.id]:
                continue
            if any(user.op != "nn.mhc_post" or user.inputs[0] != narrow.id
                   or user.inputs.count(narrow.id) != 1 for user in users[narrow.id]):
                continue
            # Validate the owning consumer's full inferred result contract.
            for user in users[narrow.id]:
                rebuilt = make_node(user.op, user.id, tuple(result if value == narrow.id else nodes[value]
                                                            for value in user.inputs), dict(user.attrs), dict(user.metadata))
                if rebuilt.type != user.type or rebuilt.effect != user.effect:
                    break
            else:
                changed.update(user.id for user in users[narrow.id])
                replacements[narrow.id] = new_nodes
                changed.update(node.id for node in chain)
                evidence.append({"narrow": narrow.id, "reduction": reduction.id, "partial_axes": partial.axes if partial else (),
                                 "packed_result": True, "owner_reduced_capacity": 1 if partial else None})
            continue
        replacements[narrow.id] = new_nodes
        changed.update(node.id for node in chain)
        evidence.append({"narrow": narrow.id, "reduction": reduction.id, "partial_axes": partial.axes if partial else (),
                         "packed_result": False, "owner_reduced_capacity": 1 if partial else None})
    if not replacements:
        return module
    points = tuple(point for point in module.selection_points
                   if not (point.kind == "distribution" and point.owner in changed))
    point_ids = {point.id for point in points}
    result = replace(module, nodes=tuple(value for node in module.nodes for value in replacements.get(node.id, (node,))),
                     selection_points=points, selections=tuple(record for record in module.selections if record.point_id in point_ids),
                     metadata={**module.metadata, "materialized_reduction_rounding": tuple(evidence)})
    return verify_module(remove_unused_functions(result))
