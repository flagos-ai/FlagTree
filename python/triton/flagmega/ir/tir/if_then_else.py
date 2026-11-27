# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT

from dataclasses import dataclass

from triton.flagmega.errors import IRSchemaError
from triton.flagmega.ir.types import DType
from triton.flagmega.ir.tir.base import TIRStmt, TIRValue, tir_node
from triton.flagmega.ir.tir.sequential import Sequential


@tir_node("if_then_else")
@dataclass(frozen=True)
class IfThenElse(TIRStmt):
    condition: TIRValue
    then_body: Sequential
    else_body: Sequential = Sequential()

    def __post_init__(self) -> None:
        if self.condition.type.rank != 0 or self.condition.type.dtype != DType.BOOL:
            raise IRSchemaError("TIR IfThenElse condition must be a scalar bool.")


__all__ = ["IfThenElse"]
