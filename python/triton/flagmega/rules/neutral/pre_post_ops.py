# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Pattern rules for composing private unary expressions at call boundaries.

The backend supplies a capability predicate. These rules contain no target,
model, dtype preference, or implementation-name policy.
"""

from dataclasses import replace

from triton.flagmega.ir.fusion import Fusion
from triton.flagmega.ir.ops.core import get_definition
from triton.flagmega.pattern_match import CallPattern, OpPattern, VArgsPattern, is_call, wildcard
from triton.flagmega.rules import RewriteResult, RewriteRule

BOUNDARY_OPS = ("tensors.cast", "ntt.vectorized_cast", "math.sigmoid", "math.silu", "math.vectorized_unary")


def pre_post_ops_rules(op_names, can_fuse):
    """Build schema-addressed rules for explicitly supported base operators."""
    rules = []
    for op_name in sorted(op_names):
        definition = get_definition(op_name)
        if any(parameter.variadic for parameter in definition.input_parameters):
            continue
        for boundary in BOUNDARY_OPS:
            rules.append(_post_rule(definition, boundary, can_fuse))
            for parameter in definition.input_parameters:
                rules.append(_pre_rule(definition, parameter, boundary, can_fuse))
    return tuple(rules)


def _pre_rule(definition, parameter, boundary, can_fuse):
    source = is_call(OpPattern(boundary), wildcard("value"), name="source").with_user_count(1)
    arguments = tuple(source if item is parameter else wildcard() for item in definition.input_parameters)
    pattern = is_call(OpPattern(definition.op_name), *arguments, name="root")

    def rewrite(match, module):
        root, source, value = match["root"], match["source"], match["value"]
        if not root.effect.is_pure or not source.effect.is_pure:
            return root
        pre = dict(root.attrs.get("pre_ops", {}))
        inputs = list(root.inputs)
        # User count counts consuming calls, not operand occurrences. One
        # private value may occupy several arguments of this same call.
        for item in definition.input_parameters:
            if inputs[item.input_index] != source.id:
                continue
            body = Fusion.from_call(source, value)
            if item.name in pre:
                body = body.then(pre[item.name])
            pre[item.name] = body
            inputs[item.input_index] = value.id
        candidate = replace(root, inputs=tuple(inputs), attrs={**root.attrs, "pre_ops": pre})
        # The private source disappears in the same transaction as the root
        # changes. Its old materialized selections no longer describe either
        # call; region rewriting invalidates only those two owners.
        return RewriteResult(candidate, removed_ids=(source.id, )) if can_fuse(candidate, module) else root

    return RewriteRule(f"FusePreOps.{definition.op_name}.{parameter.name}.{boundary}", pattern, rewrite,
                       supports_fusion=True)


def _post_rule(definition, boundary, can_fuse):
    arguments = VArgsPattern(lambda nodes: tuple(wildcard() for _ in nodes))
    producer = CallPattern(OpPattern(definition.op_name), arguments, "producer").with_user_count(1)
    pattern = is_call(OpPattern(boundary), producer, name="root")

    def rewrite(match, module):
        root, producer = match["root"], match["producer"]
        if not producer.effect.is_pure or not root.effect.is_pure:
            return root
        body = Fusion.from_call(root, producer)
        post = producer.attrs.get("post_ops", ())
        if post:
            if len(post) != 1:
                return root
            if post[0] is not None:
                body = post[0].then(body)
        candidate = replace(producer, id=root.id, type=root.type, attrs={**producer.attrs, "post_ops": (body, )})
        return RewriteResult(candidate, removed_ids=(producer.id, )) if can_fuse(candidate, module) else root

    return RewriteRule(f"FusePostOps.{definition.op_name}.{boundary}", pattern, rewrite, supports_fusion=True)


__all__ = ["pre_post_ops_rules"]
