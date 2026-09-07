# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT

from dataclasses import dataclass

from triton.flagmega.errors import IRSchemaError
from triton.flagmega.ir.tir.base import TIRCost, TIRValue, tir_node


@tir_node("binary")
@dataclass(frozen=True)
class Binary(TIRValue):
    op: str
    lhs: TIRValue
    rhs: TIRValue

    def __post_init__(self) -> None:
        if self.op not in {"add", "sub", "mul", "div", "floordiv", "mod", "min", "max"}:
            raise IRSchemaError(f"Unsupported TIR binary operator {self.op!r}.")
        if self.lhs.type != self.rhs.type:
            raise IRSchemaError("TIR Binary operands must have identical types.")

    @property
    def type(self):
        return self.lhs.type

    @property
    def local_cost(self) -> TIRCost:
        return TIRCost(flops=1)


__all__ = ["Binary"]
