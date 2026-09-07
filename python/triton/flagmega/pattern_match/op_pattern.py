# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Pattern for an operation target and its stored attributes."""

from __future__ import annotations

from typing import Callable, Mapping

from triton.flagmega.ir.model import Node
from triton.flagmega.pattern_match.pattern import Pattern


class OpPattern(Pattern):
    def __init__(
        self,
        op_name: str,
        condition: Callable[[Node], bool] | None = None,
        name: str | None = None,
        *,
        attributes: Mapping[str, object] | None = None,
    ) -> None:
        super().__init__(name)
        if not op_name:
            raise ValueError("OpPattern requires a non-empty op name.")
        self.op_name = op_name
        self.condition = condition or (lambda node: True)
        self.attributes = dict(attributes or {})

    def match_leaf(self, node: Node) -> bool:
        return (
            super().match_leaf(node)
            and node.op == self.op_name
            and all(key in node.attrs and node.attrs[key] == value for key, value in self.attributes.items())
            and bool(self.condition(node))
        )


def is_op(
    op_name: str,
    name: str | None = None,
    condition: Callable[[Node], bool] | None = None,
    *,
    attributes: Mapping[str, object] | None = None,
) -> OpPattern:
    return OpPattern(op_name, condition, name, attributes=attributes)


__all__ = ["OpPattern", "is_op"]
