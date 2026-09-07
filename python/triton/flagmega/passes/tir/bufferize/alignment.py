# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Interprocedural transfer-source alignment requirements."""

from __future__ import annotations

from triton.flagmega.errors import IRVerificationError
from triton.flagmega.ir import IRModule, kernel_dispatch_of


def transfer_source_alignment_requirements(module: IRModule) -> dict[str, int]:
    """Map graph values to minimum base-address alignment.

    Requirements originate on semantic argument indexes in a selected
    microkernel.  They are propagated through graph function calls and
    zero-copy views before allocation, matching nncase's physical-buffer
    alignment rewrite without relying on a target-specific metadata key.
    """

    requirements: dict[str, int] = {}
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
                _add(
                    requirements,
                    node.inputs[argument_index],
                    channel.source_alignment_bytes,
                )

    changed = True
    while changed:
        changed = False
        for node in module.nodes:
            required = requirements.get(node.id)
            if required is not None and node.op in {
                "builtin.get_item",
                "distributed.sharded_view",
                "tir.buffer_view",
            }:
                for input_id in node.inputs:
                    changed |= _add(requirements, input_id, required)
            if node.op != "builtin.call":
                continue
            callee = module.function_map.get(str(node.attrs.get("callee", "")))
            if callee is None:
                continue
            for formal_id, actual_id in zip(callee.parameters, node.inputs):
                required = requirements.get(formal_id)
                if required is not None:
                    changed |= _add(requirements, actual_id, required)
    return requirements


def _add(requirements: dict[str, int], value_id: str, alignment: int) -> bool:
    previous = requirements.get(value_id, 1)
    if alignment <= previous:
        return False
    requirements[value_id] = alignment
    return True


__all__ = ["transfer_source_alignment_requirements"]
