# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT

from dataclasses import dataclass

from triton.flagmega.errors import IRSchemaError
from triton.flagmega.ir.model import TensorType
from triton.flagmega.ir.tir.base import TIRValue, tir_node


@tir_node("scalar_var")
@dataclass(frozen=True)
class ScalarVar(TIRValue):
    name: str
    value_type: TensorType

    def __post_init__(self) -> None:
        if not self.name or not isinstance(self.value_type, TensorType) or self.value_type.rank != 0:
            raise IRSchemaError("TIR ScalarVar requires a name and rank-zero TensorType.")

    @property
    def type(self) -> TensorType:
        return self.value_type


__all__ = ["ScalarVar"]
