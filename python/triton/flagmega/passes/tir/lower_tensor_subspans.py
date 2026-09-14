# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Choose contiguous storage views against preplanned semantic TIR ABIs."""

from dataclasses import replace

from triton.flagmega.ir import DistributedType, verify_module
from triton.flagmega.errors import StageError
from triton.flagmega.ir.distributed_type import local_shape
from triton.flagmega.ir.op_fusion import has_ops
from triton.flagmega.ir.ops.tir.buffer_subspan import dense_subspan_offset, slice_subspan_attrs
from triton.flagmega.ir.tir import kernel_dispatch_for_call, kernel_dispatch_of
from triton.flagmega.passes.tir.bufferize.alignment import storage_alignment_requirements, validate_storage_alignments
from triton.flagmega.passes.tir.plan_function_memory import MEMORY_SPACE_METADATA


def _preserves_snapshot(module, view_id):
    """Mandatory writers must not turn an owned Slice value into a live view."""
    views = {"tir.buffer_view", "tir.buffer_subspan", "distributed.sharded_view", "builtin.get_item", "builtin.tuple"}
    roots = {}
    for node in module.nodes:
        roots[node.id] = (set().union(*(roots[v] for v in node.inputs)) if node.op in views else {node.id})
    indices = {node.id: index for index, node in enumerate(module.nodes)}
    dependent = {view_id}
    last_use = indices[view_id]
    for node in module.nodes:
        if dependent.intersection(node.inputs):
            last_use = max(last_use, indices[node.id])
            if node.op in views:
                dependent.add(node.id)
    storage = roots[view_id]
    for node in module.nodes[indices[view_id] + 1:last_use + 1]:
        dispatch = kernel_dispatch_for_call(module, node)
        if dispatch is not None:
            writes = [value for name, value in zip(dispatch.arguments, node.inputs) if name in dispatch.writes]
        elif node.op == "tir.call" and node.effect.kind.value in {"write", "read_write"}:
            writes = node.inputs
        else:
            continue
        if any(storage.intersection(roots[value]) for value in writes):
            return False
    return True


def lower_tensor_subspans(module):
    module = verify_module(module)
    if module.execution_functions or "buffer_plan" in module.metadata:
        raise StageError("Subspan storage planning must precede bufferization; resume from semantic TIR.",
                         stage=module.stage)
    if any(parameter.alignment_bytes is None for function in module.kernel_callable_map.values()
           if kernel_dispatch_of(function) is not None
           for parameter in function.parameters if parameter.role.value != "metadata"):
        raise StageError("Subspan planning requires declared TIR alignment contracts; run plan-tir-alignments first.",
                         stage=module.stage)
    validate_storage_alignments(module)
    escapes = {value for function in module.functions for value in function.outputs}
    # Partial graph-function parameters currently have a dense component ABI;
    # non-partial arguments can acquire canonical backing before binding.
    escapes.update(value for node in module.nodes if node.op == "tir.call"
                   and node.attrs.get("callee") in module.function_map for value in node.inputs
                   if isinstance(module.node_map[value].type, DistributedType)
                   and module.node_map[value].type.partial is not None)
    pending = list(escapes)
    while pending:
        node = module.node_map[pending.pop()]
        if node.op not in {"builtin.tuple", "builtin.get_item", "tir.buffer_view", "distributed.sharded_view"}:
            continue
        for value in node.inputs:
            if value not in escapes:
                escapes.add(value)
                pending.append(value)

    replacements = {}
    retired_callees = set()
    current = module
    # Consumers first: propagate the fixed ABI through aliases, while a real
    # materialization remains an independently aligned allocation boundary.
    for node in reversed(module.nodes):
        dispatch = kernel_dispatch_for_call(module, node)
        if (node.id in escapes or dispatch is None
                or dispatch.semantic_op not in {"tensors.slice", "tensors.slice_to_shape"}
                or has_ops(dispatch.semantic_attrs) or not node.effect.is_pure
                or MEMORY_SPACE_METADATA in node.metadata):
            continue
        semantic = replace(node, op=dispatch.semantic_op, attrs=dispatch.semantic_attrs)
        attrs = slice_subspan_attrs(semantic, current)
        if attrs is None:
            continue
        view = replace(node, op="tir.buffer_subspan", attrs=attrs,
                       metadata={**node.metadata, "lowered_from": dispatch.semantic_op})
        proposed = replace(current, nodes=tuple(view if value.id == node.id else value for value in current.nodes))
        if not _preserves_snapshot(proposed, node.id):
            continue
        alignment = storage_alignment_requirements(proposed).get(node.id, 1)
        source = module.node_map[node.inputs[0]].type
        tensor = getattr(source, "tensor", source)
        shapes = [(tuple(d.fixed_value for d in tensor.shape), attrs["shape"])]
        if isinstance(source, DistributedType):
            shapes.append((tuple(d.fixed_value for d in local_shape(source)),
                           tuple(d.fixed_value for d in local_shape(node.type))))
        if any(dense_subspan_offset(parent, result, attrs["offsets"], tensor.dtype.itemsize) % alignment
               for parent, result in shapes):
            continue
        replacements[node.id] = view
        retired_callees.add(str(node.attrs["callee"]))
        current = proposed

    if not replacements:
        return module
    validate_storage_alignments(current)
    callees = {node.attrs.get("callee") for node in current.nodes if node.op == "tir.call"}
    from triton.flagmega.ir.tir.visitor import iter_tir_children
    pending = list(module.kernel_callable_map.values())
    while pending:
        value = pending.pop()
        callee = getattr(value, "callee", None)
        if isinstance(callee, str):
            callees.add(callee)
        pending.extend(iter_tir_children(value))
    retired_callees -= callees
    points = tuple(point for point in module.selection_points
                   if point.owner not in replacements and point.owner not in retired_callees)
    point_ids = {point.id for point in points}
    # Only retire leaf definitions that belonged to the removed copy calls.
    return verify_module(replace(
        current, selection_points=points,
        selections=tuple(record for record in module.selections if record.point_id in point_ids),
        kernel_definitions=tuple(kernel for kernel in module.kernel_definitions
                                 if kernel.name not in retired_callees),
    ))


__all__ = ["lower_tensor_subspans"]
