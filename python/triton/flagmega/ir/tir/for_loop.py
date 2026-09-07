# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT

from dataclasses import dataclass
from enum import Enum

from triton.flagmega.ir.dim_expr import DimVar
from triton.flagmega.ir.tir.base import TIRStmt, tir_node
from triton.flagmega.ir.tir.range import Range
from triton.flagmega.ir.tir.sequential import Sequential


class LoopMode(str, Enum):
    SERIAL = "serial"
    PARALLEL = "parallel"
    VECTORIZED = "vectorized"
    UNROLLED = "unrolled"
    THREAD = "thread"


class LoopPartition(str, Enum):
    UNPARTITIONED = "unpartitioned"
    PROLOGUE = "prologue"
    STEADY = "steady"
    EPILOGUE = "epilogue"


@tir_node("for_loop")
@dataclass(frozen=True)
class For(TIRStmt):
    loop_var: DimVar
    domain: Range
    mode: LoopMode
    body: Sequential
    partition: LoopPartition = LoopPartition.UNPARTITIONED

    def __post_init__(self) -> None:
        object.__setattr__(self, "mode", LoopMode(self.mode))
        object.__setattr__(self, "partition", LoopPartition(self.partition))


__all__ = ["For", "LoopMode", "LoopPartition"]
