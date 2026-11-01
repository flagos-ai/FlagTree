# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Caller-owned storage implementing one tensor leaf of an SSA loop phi."""
from dataclasses import dataclass
from triton.flagmega.errors import IRSchemaError


@dataclass(frozen=True)
class LoopCarryBinding:
    formal: str
    initial: str
    current: str
    staging: str
    yielded: str
    result: str

    def __post_init__(self):
        if any(not isinstance(v,str) or not v for v in self.to_data().values()):
            raise IRSchemaError('Loop carry bindings require named physical buffers.')
        if self.current==self.staging:
            raise IRSchemaError('Loop carry current and staging storage must differ.')

    def to_data(self):
        return {name:getattr(self,name) for name in ('formal','initial','current','staging','yielded','result')}

    @classmethod
    def from_data(cls,data):
        return cls(**{name:str(data[name]) for name in ('formal','initial','current','staging','yielded','result')})
