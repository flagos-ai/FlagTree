# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Scalar and nncase-style vector data types."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from math import prod
from typing import Any, Mapping, TypeAlias

from triton.flagmega.errors import IRSchemaError


class DType(str, Enum):
    BOOL = "bool"
    INT32 = "int32"
    INT64 = "int64"
    BFLOAT16 = "bfloat16"
    FLOAT16 = "float16"
    FLOAT32 = "float32"
    FLOAT8_E4M3FN = "float8_e4m3fn"

    @property
    def itemsize(self) -> int:
        return {
            DType.BOOL: 1,
            DType.INT32: 4,
            DType.INT64: 8,
            DType.BFLOAT16: 2,
            DType.FLOAT16: 2,
            DType.FLOAT32: 4,
            DType.FLOAT8_E4M3FN: 1,
        }[self]


@dataclass(frozen=True)
class PointerType:
    """A target ABI pointer; FlagMega's supported targets use 64-bit addresses."""

    elem_type: object

    def __post_init__(self) -> None:
        object.__setattr__(self, "elem_type", data_type(self.elem_type))

    @property
    def itemsize(self) -> int:
        return 8

    @property
    def value(self) -> str:
        return f"*{self.elem_type.value}"

    def to_data(self) -> dict[str, object]:
        return {"kind": "pointer", "elem_type": data_type_to_data(self.elem_type)}

    def __str__(self) -> str:
        return self.value


class MaskVectorStyle(str, Enum):
    UNKNOWN = "unknown"
    FAT = "fat"
    SLIM = "slim"


@dataclass(frozen=True)
class MaskVectorType:
    """Boolean lane mask with an explicit physical element width and style."""

    style: MaskVectorStyle
    element_bits: int
    lanes: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "style", MaskVectorStyle(self.style))
        if (
            isinstance(self.element_bits, bool)
            or self.element_bits <= 0
            or self.element_bits % 8
            or isinstance(self.lanes, bool)
            or self.lanes <= 0
        ):
            raise IRSchemaError("MaskVectorType requires byte-sized positive elements and lanes.")

    @property
    def itemsize(self) -> int:
        return self.element_bits * self.lanes // 8

    @property
    def value(self) -> str:
        return f"mask<{self.style.value},{self.element_bits},{self.lanes}>"

    def to_data(self) -> dict[str, object]:
        return {
            "kind": "mask_vector",
            "style": self.style.value,
            "element_bits": self.element_bits,
            "lanes": self.lanes,
        }

    def __str__(self) -> str:
        return self.value


@dataclass(frozen=True)
class VectorType:
    """A scalar element type with one or more physical vector lane axes.

    This follows nncase's ``VectorType(ElemType, Lanes)`` model. Tensor shape
    stores the outer logical extents while ``lanes`` describes the fixed-size
    element payload, so ``shape * lanes * elem_type.itemsize`` is the exact
    physical byte size.
    """

    elem_type: DType
    lanes: tuple[int, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "elem_type", DType(self.elem_type))
        object.__setattr__(self, "lanes", tuple(int(lane) for lane in self.lanes))
        if self.elem_type == DType.BOOL:
            raise IRSchemaError("Boolean is not supported as a VectorType element; use a mask type.")
        if not self.lanes or any(isinstance(lane, bool) or lane <= 0 for lane in self.lanes):
            raise IRSchemaError(f"VectorType lanes must be non-empty positive integers, got {self.lanes!r}.")

    @property
    def lane_count(self) -> int:
        return prod(self.lanes)

    @property
    def itemsize(self) -> int:
        return self.elem_type.itemsize * self.lane_count

    @property
    def value(self) -> str:
        return f"{self.elem_type.value}<{','.join(str(lane) for lane in self.lanes)}>"

    def to_data(self) -> dict[str, object]:
        return {
            "kind": "vector",
            "elem_type": self.elem_type.value,
            "lanes": list(self.lanes),
        }

    def __str__(self) -> str:
        return self.value


DataType: TypeAlias = DType | VectorType | PointerType | MaskVectorType


def vector_type(elem_type: DType | str, lanes: int | tuple[int, ...] | list[int], *more_lanes: int) -> VectorType:
    if isinstance(lanes, int):
        normalized = (lanes, *more_lanes)
    else:
        if more_lanes:
            raise TypeError("Additional vector lanes require the first lane to be an integer.")
        normalized = tuple(lanes)
    return VectorType(DType(elem_type), normalized)


def data_type(value: DataType | str | Mapping[str, Any]) -> DataType:
    if isinstance(value, (DType, VectorType, PointerType, MaskVectorType)):
        return value
    if isinstance(value, Mapping):
        return data_type_from_data(value)
    return DType(value)


def data_type_to_data(value: DataType) -> str | dict[str, object]:
    return value.value if isinstance(value, DType) else value.to_data()


def data_type_from_data(value: object) -> DataType:
    if isinstance(value, str):
        if value.endswith(">") and "<" in value:
            element, encoded_lanes = value[:-1].split("<", 1)
            try:
                lanes = tuple(int(lane) for lane in encoded_lanes.split(","))
            except ValueError as error:
                raise IRSchemaError(f"Invalid vector data type spelling {value!r}.") from error
            return VectorType(DType(element), lanes)
        return DType(value)
    if isinstance(value, Mapping) and value.get("kind") == "vector":
        return VectorType(
            DType(str(value["elem_type"])),
            tuple(int(lane) for lane in value.get("lanes", ())),
        )
    if isinstance(value, Mapping) and value.get("kind") == "pointer":
        return PointerType(data_type_from_data(value["elem_type"]))
    if isinstance(value, Mapping) and value.get("kind") == "mask_vector":
        return MaskVectorType(
            MaskVectorStyle(str(value["style"])),
            int(value["element_bits"]),
            int(value["lanes"]),
        )
    raise IRSchemaError(f"Unknown data type encoding {value!r}.")


__all__ = [
    "DType",
    "DataType",
    "MaskVectorStyle",
    "MaskVectorType",
    "PointerType",
    "VectorType",
    "data_type",
    "data_type_from_data",
    "data_type_to_data",
    "vector_type",
]
