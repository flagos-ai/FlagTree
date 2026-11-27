# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT

from dataclasses import dataclass

from triton.flagmega.errors import IRSchemaError
from triton.flagmega.ir.dim_expr import Dimension, dim
from triton.flagmega.ir.tir.base import TIRNode, tir_node


@tir_node("range")
@dataclass(frozen=True)
class Range(TIRNode):
    start: Dimension
    stop: Dimension
    step: Dimension = dim(1)

    def __post_init__(self) -> None:
        object.__setattr__(self, "start", dim(self.start).simplify())
        object.__setattr__(self, "stop", dim(self.stop).simplify())
        object.__setattr__(self, "step", dim(self.step).simplify())
        if self.step.maximum is not None and self.step.maximum <= 0:
            raise IRSchemaError("TIR Range step must be positive.")

    @property
    def extent(self) -> Dimension:
        return ((self.stop - self.start + self.step - 1) // self.step).simplify()


__all__ = ["Range"]
