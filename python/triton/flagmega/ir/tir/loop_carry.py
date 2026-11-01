# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Physical tensor recurrence: init, parallel stage/commit, and publication.

A For initializes each current slot from its initial value. After the body,
all yielded leaves are staged before any current slot is overwritten. It then
commits the stages to the current slots. On exit, including zero trips, current
slots are published to the results. Slots belong to the ordinary BufferPlan.
"""
from dataclasses import dataclass
from triton.flagmega.ir.tir.base import TIRNode,tir_node


@tir_node('loop_carry')
@dataclass(frozen=True)
class LoopCarry(TIRNode):
    formal: str
    initial: str
    current: str
    staging: str
    yielded: str
    result: str

    def __post_init__(self):
        from triton.flagmega.ir.bufferization.loop_carry import LoopCarryBinding
        LoopCarryBinding(**{name:getattr(self,name) for name in ('formal','initial','current','staging','yielded','result')})
