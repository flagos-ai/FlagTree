# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Validate and materialize typed target-private microkernel resources."""

from __future__ import annotations

from triton.flagmega.errors import IRVerificationError
from triton.flagmega.ir.bufferization import MemSpan, PhysicalBuffer
from triton.flagmega.ir.dim_expr import dim
from triton.flagmega.ir.tir import Buffer, KernelDispatch, PrimFunction
from triton.flagmega.ir.tir.microkernel_selection import TIRMicroKernelSelection


def validate_transfer_pipeline(
    function: PrimFunction,
    dispatch: KernelDispatch,
    selection: TIRMicroKernelSelection,
    *,
    stage: str,
) -> None:
    pipeline = selection.transfer_pipeline
    if pipeline is None:
        return
    for channel in pipeline.channels:
        for argument_index in channel.source_argument_indices:
            if argument_index >= len(dispatch.arguments):
                raise IRVerificationError(
                    f"TIR microkernel {selection.family}/{selection.variant} for "
                    f"@{function.name} transfer channel {channel.name!r} declares "
                    f"invalid source operand {argument_index}.",
                    stage=stage,
                )
            argument = dispatch.arguments[argument_index]
            if argument not in dispatch.reads or argument in dispatch.writes:
                raise IRVerificationError(
                    f"TIR microkernel {selection.family}/{selection.variant} for "
                    f"@{function.name} transfer channel {channel.name!r} declares "
                    f"non-read-only source operand {argument_index} ({argument!r}).",
                    stage=stage,
                )


def materialize_shared_workspace_buffers(
    function: PrimFunction,
    selection: TIRMicroKernelSelection,
) -> tuple[Buffer, ...]:
    parameter_names = set(function.parameter_map)
    result = []
    for index, descriptor in enumerate(selection.shared_workspaces):
        if descriptor.name in parameter_names:
            raise IRVerificationError(
                f"TIR microkernel {selection.family}/{selection.variant} shared "
                f"workspace {descriptor.name!r} conflicts with @{function.name} ABI."
            )
        physical = PhysicalBuffer(
            id=f"shared:{function.name}:{index}:{descriptor.name}",
            memory_space="shared",
            size=dim(descriptor.maximum_nbytes),
            alignment=descriptor.alignment_bytes,
            function=function.name,
            role="microkernel_shared_workspace",
        )
        result.append(Buffer(
            descriptor.name,
            descriptor.type.dtype,
            MemSpan(physical),
            descriptor.type.shape,
            _dense_strides(descriptor.type.shape),
        ))
    return tuple(result)


def _dense_strides(shape):
    result = []
    current = dim(1)
    for extent in reversed(shape):
        result.append(current)
        current = (current * extent).simplify()
    return tuple(reversed(result))


__all__ = [
    "materialize_shared_workspace_buffers",
    "validate_transfer_pipeline",
]
