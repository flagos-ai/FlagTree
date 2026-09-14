# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Owner-local row snapshots consumed by an in-place state update."""

from dataclasses import dataclass

from triton.flagmega.errors import IRSchemaError, IRVerificationError
from triton.flagmega.ir.memory_effect import MemoryEffect, expand_memory_effect
from triton.flagmega.ir.model import RefType, TensorType
from triton.flagmega.ir.tir.base import TIRNode, tir_node


@tir_node("inplace_transfer_partition")
@dataclass(frozen=True)
class TIRInplaceTransferPartition(TIRNode):
    """Partition a dense state into rows owned exactly as one output axis.

    Prefix dimensions of the selected source leaf flatten to rows; all
    remaining dimensions and vector lanes flatten to columns. Producer and
    consumer traverse the output's local rows in tiles of ``tile_rows``.
    Each row is snapshotted once, updated only after the snapshot is ready,
    and never accessed after releasing that tile. Inactive rows/columns must
    not access global state. Cross-call publication remains a caller duty.
    """

    source_field_path: tuple[str, ...]
    source_row_rank: int
    output_index: int
    output_axis: int
    tile_rows: int

    def __post_init__(self):
        if not isinstance(self.source_field_path, (tuple, list)):
            raise IRSchemaError("Inplace transfer field path must be a sequence of field names.")
        path = tuple(self.source_field_path)
        if any(not isinstance(value, str) or not value for value in path):
            raise IRSchemaError("Inplace transfer fields must be non-empty names.")
        for name in ("source_row_rank", "output_index", "output_axis", "tile_rows"):
            value = getattr(self, name)
            minimum = 1 if name in {"source_row_rank", "tile_rows"} else 0
            if type(value) is not int or value < minimum:
                raise IRSchemaError(f"Inplace transfer {name} must be an integer >= {minimum}.")
        object.__setattr__(self, "source_field_path", path)

    def source_leaf(self, value_type):
        """Resolve the field path and its physical ABI leaf index together."""
        index = 0
        for name in self.source_field_path:
            if not isinstance(value_type, RefType):
                raise IRVerificationError("Inplace transfer field path must traverse Ref fields.")
            for field_name, field_type in value_type.fields:
                if field_name == name:
                    value_type = field_type
                    break
                index += len(expand_memory_effect(field_type, MemoryEffect.NONE))
            else:
                raise IRVerificationError(f"Inplace transfer has unknown source field {name!r}.")
        if not isinstance(value_type, TensorType):
            raise IRVerificationError("Inplace transfer source must resolve to a dense tensor leaf.")
        return index, value_type


__all__ = ["TIRInplaceTransferPartition"]
