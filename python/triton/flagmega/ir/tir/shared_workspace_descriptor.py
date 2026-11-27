# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Typed target-private shared-memory workspace requirements."""

from __future__ import annotations

from dataclasses import dataclass
from math import prod

from triton.flagmega.errors import IRSchemaError
from triton.flagmega.ir.model import TensorType
from triton.flagmega.ir.tir.base import TIRNode, tir_node


@tir_node("shared_workspace_descriptor")
@dataclass(frozen=True)
class TIRSharedWorkspaceDescriptor(TIRNode):
    """One bounded shared-memory allocation required by a microkernel.

    This is the Python counterpart of nncase's
    ``TIRSharedWorkspaceDescriptor``.  It describes target-private storage;
    it is deliberately distinct from :class:`WorkspaceRequirement`, whose
    storage is allocated by the graph caller in device memory.
    """

    name: str
    type: TensorType
    alignment_bytes: int
    matrix_compatible: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise IRSchemaError(
                "TIRSharedWorkspaceDescriptor requires a non-empty name."
            )
        if not isinstance(self.type, TensorType):
            raise IRSchemaError(
                "TIRSharedWorkspaceDescriptor requires a TensorType."
            )
        alignment = self.alignment_bytes
        if (
            isinstance(alignment, bool)
            or not isinstance(alignment, int)
            or alignment <= 0
            or alignment & (alignment - 1)
        ):
            raise IRSchemaError(
                "Shared workspace alignment must be a positive power of two."
            )
        if alignment < self.type.dtype.itemsize:
            raise IRSchemaError(
                "Shared workspace alignment must be at least its element size."
            )
        if not isinstance(self.matrix_compatible, bool):
            raise IRSchemaError(
                "Shared workspace matrix_compatible must be boolean."
            )
        if self.maximum_nbytes <= 0:
            raise IRSchemaError(
                "Shared workspace requires a finite positive maximum size."
            )

    @property
    def maximum_nbytes(self) -> int:
        extents = tuple(dimension.maximum for dimension in self.type.shape)
        if any(value is None or value < 0 for value in extents):
            raise IRSchemaError(
                "Shared workspace requires a finite positive maximum size."
            )
        return prod((int(value) for value in extents), start=1) * self.type.dtype.itemsize


__all__ = ["TIRSharedWorkspaceDescriptor"]
