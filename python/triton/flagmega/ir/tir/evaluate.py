# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT

from dataclasses import dataclass

from triton.flagmega.ir.tir.base import TIRStmt, TIRValue, tir_node


@tir_node("evaluate")
@dataclass(frozen=True)
class Evaluate(TIRStmt):
    value: TIRValue


__all__ = ["Evaluate"]
