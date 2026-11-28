# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""A terminal pattern behind zero or more unary steps, with path captures."""

from triton.flagmega.pattern_match.pattern import Pattern


class UnaryChainPattern(Pattern):
    """Match terminal or step(...step(terminal)); capture outer-to-inner steps.

    Each step has a fresh match scope. Reusing the terminal pattern still
    enforces shared producer identity across independently matched chains.
    """

    def __init__(self, terminal: Pattern, step: Pattern, name: str | None = None):
        super().__init__(name)
        self.terminal = terminal
        self.step = step


def is_unary_chain(terminal: Pattern, step: Pattern, *, name: str | None = None) -> UnaryChainPattern:
    return UnaryChainPattern(terminal, step, name)
