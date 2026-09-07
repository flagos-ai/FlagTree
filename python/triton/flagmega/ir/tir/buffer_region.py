# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT

from dataclasses import dataclass

from triton.flagmega.errors import IRSchemaError
from triton.flagmega.ir.tir.base import TIRNode, tir_node
from triton.flagmega.ir.tir.buffer import Buffer
from triton.flagmega.ir.tir.range import Range


@tir_node("buffer_region")
@dataclass(frozen=True)
class BufferRegion(TIRNode):
    buffer: Buffer
    region: tuple[Range, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "region", tuple(self.region))
        if len(self.region) != self.buffer.rank:
            raise IRSchemaError("TIR BufferRegion rank must equal its Buffer rank.")


__all__ = ["BufferRegion"]
