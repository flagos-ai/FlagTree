# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Stateless local rewrite rule protocol."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Callable

from triton.flagmega.ir import IRModule, Node
from triton.flagmega.pattern_match import MatchResult, Pattern, try_match_root


RuleMatch = Callable[[Node, IRModule], bool]
@dataclass(frozen=True)
class RewriteResult:
    """One local replacement plus topologically ordered helper nodes.

    The replacement keeps the matched root id. This makes function outputs,
    selection owners and editable dump names stable while still allowing a
    rule to materialize representation operations such as Pack/Unpack.
    """

    replacement: Node
    prefix_nodes: tuple[Node, ...] = ()
    # Dataflow-only, atomic region edits. A removed id may be explicitly
    # relocated into prefix_nodes. Other users are updated with stable ids.
    extra_replacements: tuple[Node, ...] = ()
    removed_ids: tuple[str, ...] = ()
    # Region helpers may publish at an earlier matched boundary, e.g. the
    # cache write preceding the attention root. None means before the root.
    insertion_before: str | None = None


@dataclass(frozen=True)
class RewriteRedirect:
    """Remove the matched node and redirect every use to an earlier node."""

    target_id: str


RuleOutput = Node | RewriteResult | RewriteRedirect
RuleRewrite = Callable[[Node, IRModule], RuleOutput]
PatternRewrite = Callable[[MatchResult, IRModule], RuleOutput]


class RewriteEffectPolicy(str, Enum):
    """Whether a dataflow rule may explicitly change side-effect semantics."""

    PRESERVE = "preserve"
    ALLOW = "allow"


class RewriteRule:
    """A rewrite driven by a structural ``Pattern`` or legacy predicate."""

    def __init__(
        self,
        name: str,
        matches: RuleMatch | Pattern | None = None,
        rewrite: RuleRewrite | PatternRewrite | None = None,
        *,
        pattern: Pattern | None = None,
        effect_policy: RewriteEffectPolicy | str = RewriteEffectPolicy.PRESERVE,
        supports_fusion: bool = False,
    ) -> None:
        if not name:
            raise ValueError("RewriteRule requires a non-empty name.")
        if isinstance(matches, Pattern):
            if pattern is not None:
                raise TypeError("RewriteRule pattern was provided twice.")
            pattern, matches = matches, None
        if (pattern is None) == (matches is None):
            raise TypeError("RewriteRule requires exactly one Pattern or predicate.")
        if rewrite is None:
            raise TypeError("RewriteRule requires a rewrite callback.")
        self.name = name
        self.pattern = pattern
        self.matches = matches
        self.rewrite = rewrite
        self.effect_policy = RewriteEffectPolicy(effect_policy)
        self.supports_fusion = supports_fusion

    def apply(self, node: Node, module: IRModule) -> RuleOutput | None:
        from triton.flagmega.ir.op_fusion import has_ops
        if not self.supports_fusion and has_ops(node.attrs):
            return None
        if self.pattern is not None:
            result = try_match_root(node, self.pattern, module)
            return None if result is None else self.apply_match(result, module)
        assert self.matches is not None
        if not self.supports_fusion and any(has_ops(module.node_map[value].attrs) for value in node.inputs):
            return None
        return self.rewrite(node, module) if self.matches(node, module) else None  # type: ignore[arg-type]

    def apply_match(self, result: MatchResult, module: IRModule) -> RuleOutput | None:
        """Invoke a pattern rule with a match supplied by another provider.

        The ordinary data-flow provider obtains matches from ``Node.inputs``;
        the e-graph provider supplies matches found through child e-classes.
        Keeping callback invocation here avoids a second, concrete-DAG match
        that would discard equality information.
        """

        if self.pattern is None:
            raise TypeError("Predicate rewrite rules do not accept MatchResult values.")
        from triton.flagmega.ir.op_fusion import has_ops
        # Existing rules describe naked operators. A fused call has additional
        # semantic expressions, even though its base op name is unchanged.
        if not self.supports_fusion and any(isinstance(value, Node) and has_ops(value.attrs)
                                            for value in result.values()):
            return None
        return self.rewrite(result, module)  # type: ignore[arg-type]


class RuleRegistry:
    """An explicit registry; importing a module never scans for rules."""

    def __init__(self) -> None:
        self._rules: dict[str, RewriteRule] = {}

    def add(self, rule: RewriteRule) -> None:
        if rule.name in self._rules:
            raise ValueError(f"Rewrite rule {rule.name!r} is already registered.")
        self._rules[rule.name] = rule

    def get(self, name: str) -> RewriteRule:
        return self._rules[name]

    def values(self) -> tuple[RewriteRule, ...]:
        return tuple(self._rules[name] for name in sorted(self._rules))
