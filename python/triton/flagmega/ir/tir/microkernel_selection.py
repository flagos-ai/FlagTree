# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""First-class target implementation selected for one semantic TIR op."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType

from triton.flagmega.errors import IRSchemaError
from triton.flagmega.ir.tir.base import TIRNode, tir_node
from triton.flagmega.ir.tir.shared_workspace_descriptor import (
    TIRSharedWorkspaceDescriptor,
)
from triton.flagmega.ir.tir.transfer_pipeline_contract import (
    TIRTransferPipelineContract,
)


@tir_node("microkernel_selection")
@dataclass(frozen=True)
class TIRMicroKernelSelection(TIRNode):
    """Editable result of target microkernel selection.

    The semantic operation and its attributes remain on ``KernelDispatch``.
    This object owns only the concrete implementation identity, physical
    schedule parameters, target requirements, and implementation facts.  It
    mirrors nncase's ``Call.Metadata.TIRMicroKernel`` boundary while remaining
    ordinary Python IR that can be edited and resumed.
    """

    implementation: str
    family: str
    variant: str
    parameters: Mapping[str, object] = field(default_factory=dict)
    facts: Mapping[str, object] = field(default_factory=dict)
    requires: Sequence[str] = field(default_factory=tuple)
    shared_workspaces: Sequence[TIRSharedWorkspaceDescriptor] = field(
        default_factory=tuple
    )
    transfer_pipeline: TIRTransferPipelineContract | None = None

    def __post_init__(self) -> None:
        if not self.implementation or not self.family or not self.variant:
            raise IRSchemaError(
                "TIRMicroKernelSelection requires implementation, family, and variant."
            )
        requirements = (
            (self.requires,)
            if isinstance(self.requires, str)
            else tuple(str(value) for value in self.requires)
        )
        if any(not value for value in requirements) or len(set(requirements)) != len(
            requirements
        ):
            raise IRSchemaError(
                "TIRMicroKernelSelection requirements must be non-empty and unique."
            )
        object.__setattr__(self, "parameters", _freeze_mapping(self.parameters))
        object.__setattr__(self, "facts", _freeze_mapping(self.facts))
        object.__setattr__(self, "requires", requirements)
        workspaces = tuple(self.shared_workspaces)
        if any(
            not isinstance(value, TIRSharedWorkspaceDescriptor)
            for value in workspaces
        ):
            raise IRSchemaError(
                "TIRMicroKernelSelection shared workspaces must be typed descriptors."
            )
        names = tuple(value.name for value in workspaces)
        if len(set(names)) != len(names):
            raise IRSchemaError(
                "TIRMicroKernelSelection shared workspace names must be unique."
            )
        pipeline = self.transfer_pipeline
        if pipeline is not None:
            if not isinstance(pipeline, TIRTransferPipelineContract):
                raise IRSchemaError(
                    "TIRMicroKernelSelection transfer pipeline must be a typed contract."
                )
            owned = tuple(sorted((
                *pipeline.shared_workspace_indices,
                *pipeline.consumer_shared_workspace_indices,
            )))
            expected = tuple(range(len(workspaces)))
            if owned != expected:
                raise IRSchemaError(
                    "A transfer pipeline must assign every shared workspace to "
                    "exactly one transfer channel or the consumer role."
                )
        object.__setattr__(self, "shared_workspaces", workspaces)


def _freeze_mapping(value: Mapping[str, object]) -> Mapping[str, object]:
    return MappingProxyType(
        {str(key): _freeze(item) for key, item in sorted(value.items())}
    )


def _freeze(value: object) -> object:
    if isinstance(value, Mapping):
        return _freeze_mapping(value)
    if isinstance(value, (tuple, list)):
        return tuple(_freeze(item) for item in value)
    return value


__all__ = ["TIRMicroKernelSelection"]
