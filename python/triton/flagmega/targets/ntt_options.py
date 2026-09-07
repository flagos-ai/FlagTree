# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Backend-independent NTT rule and topology options."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from triton.flagmega.errors import IRSchemaError
from triton.flagmega.ir import Placement


@dataclass(frozen=True)
class NttTargetOptions:
    """Parameters consumed by NTT rules, never inferred from a vendor name."""

    placements: tuple[Placement, ...]
    vector_lane_bytes: int
    vector_max_axes: int
    packing_vector_bytes: int
    packing_k_pack: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "placements", tuple(self.placements))
        if not self.placements:
            raise IRSchemaError("NTT target options require at least one placement.")
        for name in (
            "vector_lane_bytes",
            "vector_max_axes",
            "packing_vector_bytes",
            "packing_k_pack",
        ):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise IRSchemaError(f"NTT target option {name} must be a positive integer.")

    def to_data(self) -> dict[str, object]:
        return {
            "schema": "flagmega.ntt-target-options/v4",
            "placements": [placement.to_data() for placement in self.placements],
            "vector_lane_bytes": self.vector_lane_bytes,
            "vector_max_axes": self.vector_max_axes,
            "packing_vector_bytes": self.packing_vector_bytes,
            "packing_k_pack": self.packing_k_pack,
        }

    @classmethod
    def from_data(cls, data: Mapping[str, object]) -> NttTargetOptions:
        if data.get("schema") == "flagmega.pyntt-target-options/v1":
            return PyNttTargetOptions.from_data(data)
        if data.get("schema") not in {
            "flagmega.ntt-target-options/v2",
            "flagmega.ntt-target-options/v3",
            "flagmega.ntt-target-options/v4",
        }:
            raise IRSchemaError(
                f"Unsupported NTT target-options schema {data.get('schema')!r}."
            )
        placements = data.get("placements")
        if not isinstance(placements, Sequence) or isinstance(placements, (str, bytes)):
            raise IRSchemaError("NTT target-options placements must be a sequence.")
        try:
            parsed = tuple(
                Placement(
                    tuple(value["hierarchy"]),
                    value["name"],
                    value["hierarchy_levels"],
                )
                for value in placements
                if isinstance(value, Mapping)
            )
            if len(parsed) != len(placements):
                raise IRSchemaError("Malformed NTT target-options placement entry.")
            return cls(
                placements=parsed,
                vector_lane_bytes=data["vector_lane_bytes"],
                vector_max_axes=data["vector_max_axes"],
                packing_vector_bytes=data["packing_vector_bytes"],
                packing_k_pack=data["packing_k_pack"],
            )
        except (KeyError, TypeError, ValueError) as error:
            raise IRSchemaError("Malformed NTT target-options payload.") from error


@dataclass(frozen=True)
class PyNttTargetOptions(NttTargetOptions):
    """PyNTT physical-distribution options, independent of a vendor machine."""

    block_cyclic_block_bytes: int = 128

    def __post_init__(self) -> None:
        super().__post_init__()
        value = self.block_cyclic_block_bytes
        if (
            not isinstance(value, int)
            or isinstance(value, bool)
            or value <= 0
            or value & (value - 1)
        ):
            raise IRSchemaError(
                "PyNTT block_cyclic_block_bytes must be a positive power of two."
            )

    def to_data(self) -> dict[str, object]:
        data = super().to_data()
        data.update({
            "schema": "flagmega.pyntt-target-options/v1",
            "block_cyclic_block_bytes": self.block_cyclic_block_bytes,
        })
        return data

    @classmethod
    def from_ntt_options(
        cls,
        options: NttTargetOptions,
        *,
        block_cyclic_block_bytes: int = 128,
    ) -> PyNttTargetOptions:
        if isinstance(options, cls):
            return options
        return cls(
            placements=options.placements,
            vector_lane_bytes=options.vector_lane_bytes,
            vector_max_axes=options.vector_max_axes,
            packing_vector_bytes=options.packing_vector_bytes,
            packing_k_pack=options.packing_k_pack,
            block_cyclic_block_bytes=block_cyclic_block_bytes,
        )

    @classmethod
    def from_data(cls, data: Mapping[str, object]) -> PyNttTargetOptions:
        if data.get("schema") != "flagmega.pyntt-target-options/v1":
            raise IRSchemaError(
                f"Unsupported PyNTT target-options schema {data.get('schema')!r}."
            )
        base_data = dict(data)
        base_data["schema"] = "flagmega.ntt-target-options/v4"
        base = NttTargetOptions.from_data(base_data)
        try:
            return cls.from_ntt_options(
                base,
                block_cyclic_block_bytes=int(data["block_cyclic_block_bytes"]),
            )
        except (KeyError, TypeError, ValueError) as error:
            raise IRSchemaError("Malformed PyNTT target-options payload.") from error


__all__ = ["NttTargetOptions", "PyNttTargetOptions"]
