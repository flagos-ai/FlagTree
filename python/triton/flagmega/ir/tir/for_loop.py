# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT

from dataclasses import dataclass
from enum import Enum

from triton.flagmega.ir.dim_expr import DimVar
from triton.flagmega.ir.tir.base import TIRStmt, tir_node
from triton.flagmega.ir.tir.range import Range
from triton.flagmega.ir.tir.sequential import Sequential
from triton.flagmega.ir.tir.value_ref import ValueRef


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
    iter_args: tuple[ValueRef, ...] = ()
    yield_values: tuple[ValueRef, ...] = ()
    body_function: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "mode", LoopMode(self.mode))
        object.__setattr__(self, "partition", LoopPartition(self.partition))
        object.__setattr__(self, "iter_args", tuple(self.iter_args))
        object.__setattr__(self, "yield_values", tuple(self.yield_values))
        if self.body_function is not None and not self.body_function:
            raise ValueError("TIR For body_function cannot be empty.")
        if len(self.iter_args) != len(self.yield_values):
            raise ValueError("TIR For iter_args and yield_values must have equal arity.")


__all__ = ["For", "LoopMode", "LoopPartition"]
