# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Base contracts for first-class FlagMega TIR expressions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar


_NODE_TYPES: dict[str, type[TIRNode]] = {}


def tir_node(kind: str):
    """Register one explicit TIR node class for deterministic serialization."""

    if not kind or kind in _NODE_TYPES:
        raise ValueError(f"Invalid or duplicate TIR node kind {kind!r}.")

    def decorate(cls: type[TIRNode]) -> type[TIRNode]:
        cls.kind = kind
        _NODE_TYPES[kind] = cls
        return cls

    return decorate


@dataclass(frozen=True)
class TIRCost:
    flops: int | None = 0
    bytes_read: int | None = 0
    bytes_written: int | None = 0
    synchronizations: int | None = 0

    def __post_init__(self) -> None:
        if any(
            value is not None and (isinstance(value, bool) or value < 0)
            for value in (
                self.flops,
                self.bytes_read,
                self.bytes_written,
                self.synchronizations,
            )
        ):
            raise ValueError("TIRCost factors must be non-negative integers or None.")

    def __add__(self, other: TIRCost) -> TIRCost:
        return TIRCost(
            _add_factor(self.flops, other.flops),
            _add_factor(self.bytes_read, other.bytes_read),
            _add_factor(self.bytes_written, other.bytes_written),
            _add_factor(self.synchronizations, other.synchronizations),
        )

    @property
    def unknown_factors(self) -> tuple[str, ...]:
        return tuple(
            name
            for name, value in (
                ("flops", self.flops),
                ("bytes_read", self.bytes_read),
                ("bytes_written", self.bytes_written),
                ("synchronizations", self.synchronizations),
            )
            if value is None
        )


class TIRNode:
    kind: ClassVar[str]

    @property
    def local_cost(self) -> TIRCost:
        return TIRCost()

    def to_data(self) -> dict[str, object]:
        from triton.flagmega.ir.tir.serialization import tir_to_data

        return tir_to_data(self)


class TIRValue(TIRNode):
    @property
    def type(self):
        raise NotImplementedError


class TIRStmt(TIRNode):
    pass


def node_type(kind: str) -> type[TIRNode]:
    return _NODE_TYPES[kind]


def _add_factor(lhs: int | None, rhs: int | None) -> int | None:
    return None if lhs is None or rhs is None else lhs + rhs


__all__ = ["TIRCost", "TIRNode", "TIRStmt", "TIRValue", "node_type", "tir_node"]
