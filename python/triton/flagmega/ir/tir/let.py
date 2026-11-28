# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT

from dataclasses import dataclass

from triton.flagmega.errors import IRSchemaError
from triton.flagmega.ir.tir.base import TIRStmt, TIRValue, tir_node
from triton.flagmega.ir.tir.scalar_var import ScalarVar
from triton.flagmega.ir.tir.sequential import Sequential


@tir_node("let")
@dataclass(frozen=True)
class Let(TIRStmt):
    var: ScalarVar
    value: TIRValue
    body: Sequential

    def __post_init__(self) -> None:
        if self.var.type != self.value.type:
            raise IRSchemaError("TIR Let variable and value types must match.")


__all__ = ["Let"]
