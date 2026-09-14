# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Interprocedural propagation of declared storage alignment contracts."""

from __future__ import annotations

from triton.flagmega.errors import IRVerificationError
from triton.flagmega.ir import DistributedType, IRModule, kernel_dispatch_of


def storage_alignment_requirements(module: IRModule) -> dict[str, int]:
    """Read semantic TIR ABI requirements without consulting microkernels."""
    return _propagate(module, _declared_requirements(module))


def _declared_requirements(module):
    requirements = {}
    for node in module.nodes:
        if node.op != "tir.call":
            continue
        function = module.kernel_callable_map.get(str(node.attrs.get("callee", "")))
        if function is None:
            continue
        for parameter, value in zip(function.runtime_parameters, node.inputs, strict=True):
            if parameter.role.value != "metadata" and parameter.alignment_bytes is not None:
                _add(requirements, value, parameter.alignment_bytes)
        for parameter in function.output_parameters:
            if parameter.alignment_bytes is not None:
                _add(requirements, node.id, parameter.alignment_bytes)
    return requirements


def transfer_source_alignment_requirements(module: IRModule) -> dict[str, int]:
    """Allocation requirements, including legacy checkpoints without contracts.

    Planned ABIs are authoritative. A selected implementation may not increase
    them. Old checkpoints lacking the field retain their existing transfer ABI.
    """

    requirements = _declared_requirements(module)
    for node in module.nodes:
        if node.op != "tir.call":
            continue
        function = module.kernel_callable_map.get(str(node.attrs.get("callee", "")))
        dispatch = None if function is None else kernel_dispatch_of(function)
        selection = None if dispatch is None else dispatch.microkernel
        pipeline = None if selection is None else selection.transfer_pipeline
        if pipeline is None:
            continue
        for channel in pipeline.channels:
            for argument_index in channel.source_argument_indices:
                if argument_index >= len(node.inputs):
                    raise IRVerificationError(
                        f"TIR call {node.id!r} transfer channel {channel.name!r} "
                        f"references missing argument {argument_index}.",
                        stage=module.stage,
                        node_id=node.id,
                    )
                parameter = function.runtime_parameters[argument_index]
                if parameter.alignment_bytes is not None and channel.source_alignment_bytes > parameter.alignment_bytes:
                    raise IRVerificationError(
                        f"Microkernel on {node.id!r} exceeds the declared alignment contract of {parameter.name!r}.",
                        stage=module.stage, node_id=node.id)
                _add(
                    requirements,
                    node.inputs[argument_index],
                    channel.source_alignment_bytes,
                )

    return _propagate(module, requirements)


def _propagate(module, requirements):
    changed = True
    while changed:
        changed = False
        for node in module.nodes:
            required = requirements.get(node.id)
            if required is not None and node.op in {
                "builtin.get_item",
                "builtin.tuple",
                "distributed.sharded_view",
                "tir.buffer_view",
                "tir.buffer_subspan",
            }:
                for input_id in node.inputs:
                    changed |= _add(requirements, input_id, required)
            if node.op not in {"builtin.call", "tir.call"}:
                continue
            callee = module.function_map.get(str(node.attrs.get("callee", "")))
            if callee is None:
                continue
            for formal_id, actual_id in zip(callee.parameters, node.inputs):
                required = requirements.get(formal_id)
                if required is not None:
                    changed |= _add(requirements, actual_id, required)
            result_alignment = max((requirements.get(value, 1) for value in callee.outputs), default=1)
            result_alignment = max(result_alignment, requirements.get(node.id, 1))
            changed |= _add(requirements, node.id, result_alignment)
            for value in callee.outputs:
                changed |= _add(requirements, value, result_alignment)
    return requirements


def validate_storage_alignments(module: IRModule) -> None:
    """An edited view must still satisfy its already-declared consumer ABI."""
    from triton.flagmega.ir.distributed_type import local_shape
    from triton.flagmega.ir.ops.tir.buffer_subspan import dense_subspan_offset

    requirements = storage_alignment_requirements(module)
    for node in module.nodes:
        if node.op != "tir.buffer_subspan":
            continue
        source = module.node_map[node.inputs[0]].type
        tensor = getattr(source, "tensor", source)
        shapes = [(tensor.shape, getattr(node.type, "tensor", node.type).shape)]
        if isinstance(source, DistributedType):
            shapes.append((local_shape(source), local_shape(node.type)))
        alignment = requirements.get(node.id, 1)
        for parent, result in shapes:
            offset = dense_subspan_offset(tuple(d.fixed_value for d in parent), tuple(d.fixed_value for d in result),
                                           node.attrs["offsets"], tensor.dtype.itemsize)
            if offset % alignment:
                raise IRVerificationError(
                    f"Subspan {node.id!r} offset {offset} violates its {alignment}-byte storage alignment contract.",
                    stage=module.stage, node_id=node.id)


def _add(requirements: dict[str, int], value_id: str, alignment: int) -> bool:
    previous = requirements.get(value_id, 1)
    if alignment <= previous:
        return False
    requirements[value_id] = alignment
    return True


__all__ = ["storage_alignment_requirements", "transfer_source_alignment_requirements", "validate_storage_alignments"]
