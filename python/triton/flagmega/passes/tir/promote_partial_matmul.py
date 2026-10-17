# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Keep matmul accumulators wide until their closed owner sums materialize."""

from collections import defaultdict
from dataclasses import replace

from triton.flagmega.ir import DType, DistributedType, NoneType, VectorType, logical_type, verify_module
from triton.flagmega.ir.op_fusion import has_ops
from triton.flagmega.rules.neutral._utility import make_node


def promote_partial_matmul(module):
    """Move one BF16 rounding boundary after each complete owner reduction.

    A partial represents components of one logical result. Rounding the local
    dot products before their sum changes that result, especially under
    cancellation. Closed matmul -> Boxing edges can retain FP32 components
    without changing any observable output type or the chosen K partition.
    """
    source = module.node_map
    users = defaultdict(list)
    for node in module.nodes:
        for value in node.inputs:
            users[value].append(node)
    escaped = {value for function in module.functions for value in function.outputs}
    promoted, reductions = {}, {}
    used_ids = set(source)

    def fresh_id(prefix):
        result = prefix
        ordinal = 1
        while result in used_ids:
            result = f"{prefix}.{ordinal}"
            ordinal += 1
        used_ids.add(result)
        return result

    for node in module.nodes:
        if (node.op not in {"math.matmul", "ntt.packed_matmul"}
                or not isinstance(node.type, DistributedType) or node.type.partial is None
                or node.id in escaped or has_ops(node.attrs)):
            continue
        dtype = node.type.tensor.dtype
        scalar = dtype.elem_type if isinstance(dtype, VectorType) else dtype
        if scalar != DType.BFLOAT16:
            continue
        if node.op == "ntt.packed_matmul" and any(
                not isinstance(source[value].type, NoneType) for value in node.inputs[2:]):
            # Scale/addend rounding is a separate numerical boundary.
            continue
        consumers = users[node.id]
        if not consumers or any(
                consumer.op not in {"distributed.boxing", "distributed.force_boxing"}
                or (isinstance(consumer.type, DistributedType) and consumer.type.partial is not None)
                or has_ops(consumer.attrs) for consumer in consumers):
            continue
        wide = make_node(node.op, node.id, tuple(source[value] for value in node.inputs),
                         {**node.attrs, "output_data_type": "float32"}, node.metadata)
        promoted[node.id] = wide
        for consumer in consumers:
            target = consumer.type
            wide_tensor = replace(logical_type(target), dtype=wide.type.tensor.dtype)
            wide_target = replace(target, tensor=wide_tensor) if isinstance(target, DistributedType) else wide_tensor
            reduced = make_node(consumer.op, fresh_id(consumer.id + ".accumulator"), (wide,),
                                {**consumer.attrs, "new_type": wide_target}, consumer.metadata)
            output_dtype = logical_type(target).dtype
            if isinstance(output_dtype, VectorType):
                narrowed = make_node("ntt.vectorized_cast", consumer.id, (reduced,),
                                     {"new_type": output_dtype, "vectorize_axes": (-1,)}, consumer.metadata)
            else:
                narrowed = make_node("tensors.cast", consumer.id, (reduced,),
                                     {"dtype": output_dtype}, consumer.metadata)
            assert narrowed.type == target
            reductions[consumer.id] = (reduced, narrowed)
    if not promoted:
        return module
    nodes = []
    for node in module.nodes:
        if node.id in promoted:
            nodes.append(promoted[node.id])
        elif node.id in reductions:
            nodes.extend(reductions[node.id])
        else:
            nodes.append(node)
    changed = promoted.keys() | reductions.keys()
    invalid = {point.id for point in module.selection_points if point.owner in changed}
    return verify_module(replace(
        module, nodes=tuple(nodes),
        selection_points=tuple(point for point in module.selection_points if point.id not in invalid),
        selections=tuple(selection for selection in module.selections if selection.point_id not in invalid),
    ))
