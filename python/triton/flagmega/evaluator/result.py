# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Structured reference-evaluator results."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping


@dataclass(frozen=True)
class EvaluationResult:
    """Outputs plus the immutable value trace produced by one evaluator run."""

    outputs: tuple[Any, ...]
    trace: Mapping[str, Any]

    def __post_init__(self) -> None:
        object.__setattr__(self, "outputs", tuple(self.outputs))
        object.__setattr__(self, "trace", MappingProxyType(dict(self.trace)))


__all__ = ["EvaluationResult"]
