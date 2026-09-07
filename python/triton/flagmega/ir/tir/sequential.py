# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from triton.flagmega.ir.tir.base import TIRStmt, tir_node


@tir_node("sequential")
@dataclass(frozen=True)
class Sequential(TIRStmt):
    fields: tuple[TIRStmt, ...] = ()
    trace_scope_name: str | None = None
    preserve_codegen_boundary: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "fields", tuple(self.fields))
        if self.trace_scope_name == "":
            raise ValueError("TIR Sequential trace_scope_name cannot be empty.")

    @property
    def can_flatten(self) -> bool:
        return self.trace_scope_name is None and not self.preserve_codegen_boundary

    @classmethod
    def flatten(cls, fields: Iterable[TIRStmt]) -> Sequential:
        result: list[TIRStmt] = []
        for field in fields:
            if isinstance(field, Sequential) and field.can_flatten:
                result.extend(field.fields)
            else:
                result.append(field)
        return cls(tuple(result))


__all__ = ["Sequential"]
