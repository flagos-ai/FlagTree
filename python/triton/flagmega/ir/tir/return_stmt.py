# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT

from dataclasses import dataclass

from triton.flagmega.ir.tir.base import TIRNode, TIRStmt, TIRValue, tir_node
from triton.flagmega.ir.tir.buffer import Buffer


@tir_node("return_binding")
@dataclass(frozen=True)
class ReturnBinding(TIRNode):
    value: TIRValue | Buffer
    storage: str

    def __post_init__(self) -> None:
        if not self.storage:
            raise ValueError("TIR ReturnBinding storage parameter cannot be empty.")

    @property
    def type(self):
        return self.value.type


@tir_node("return")
@dataclass(frozen=True)
class Return(TIRStmt):
    values: tuple[ReturnBinding, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "values", tuple(self.values))


__all__ = ["Return", "ReturnBinding"]
