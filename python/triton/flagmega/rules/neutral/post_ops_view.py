# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Fuse a private numeric boundary at its producer before a pure view."""

from dataclasses import replace
from triton.flagmega.errors import IRSchemaError
from triton.flagmega.ir import Fusion, VectorType
from triton.flagmega.ir.distributed_inference import tensor_of
from triton.flagmega.ir.memory_effect import MemoryEffect
from triton.flagmega.ir.ops.core import get_definition
from triton.flagmega.pattern_match import CallPattern, OpPattern, VArgsPattern, is_call, wildcard
from triton.flagmega.rules import RewriteResult, RewriteRule
from triton.flagmega.rules.neutral._utility import make_node


def shape_only_replacements(module, source, replacement, boundary_id):
    """Prove every extra operand occurrence is a representation-independent reference.

    References stay at their original positions. Only the pure replacement is
    published before the first reference, preserving state and effect order.
    """
    public = {name for function in module.functions for name in (*function.parameters, *function.outputs)}
    if source.id in public:
        return None
    extras = []
    for user in module.nodes:
        if user.id == boundary_id or source.id not in user.inputs:
            continue
        definition = get_definition(user.op)
        positions = {parameter.input_index for parameter in definition.input_parameters
                     if not parameter.variadic and parameter.memory_effect == MemoryEffect.NONE}
        if any(index not in positions for index, value in enumerate(user.inputs) if value == source.id):
            return None
        inputs = tuple(replacement if value == source.id else module.node_map[value] for value in user.inputs)
        try:
            updated = make_node(user.op, user.id, inputs, user.attrs, user.metadata)
        except IRSchemaError:
            return None
        if updated.type != user.type or updated.effect != user.effect:
            return None
        extras.append(updated)
    order = {node.id: index for index, node in enumerate(module.nodes)}
    anchor = min((boundary_id, *(user.id for user in extras)), key=order.get)
    return tuple(extras), anchor


def post_ops_through_view_rules(op_names, supported, *, views=("distributed.sharded_view", "tensors.bitcast")):
    return tuple(
        _rule(op, view, cast, supported)
        for op in sorted(op_names)
        for view in views
        for cast in ("tensors.cast", "ntt.vectorized_cast")
    )


def _rule(op, view_op, cast_op, supported):
    arguments = VArgsPattern(lambda nodes: tuple(wildcard() for _ in nodes))
    producer = CallPattern(OpPattern(op), arguments, "producer").with_user_count(1)
    view = is_call(OpPattern(view_op), producer, name="view")
    pattern = is_call(OpPattern(cast_op), view, name="root")

    def rewrite(match, module):
        root, view, producer = match["root"], match["view"], match["producer"]
        public = {name for function in module.functions for name in (*function.parameters, *function.outputs)}
        if producer.id in public or view.id in public:
            return root
        other_users = tuple(node for node in module.nodes if view.id in node.inputs and node.id != root.id)
        if any(
            not n.effect.is_pure
            or getattr(n.type, "partial", None) is not None
            or getattr(n.type, "exclusive", None) is not None
            for n in (root, view, producer)
        ):
            return root
        try:
            if view_op == "distributed.sharded_view":
                lifted = make_node(root.op, root.id, (producer,), root.attrs, {})
                view_attrs = {"new_type": root.type}
            else:
                source_dtype = tensor_of(producer.type).dtype
                view_dtype = tensor_of(view.type).dtype
                output_dtype = tensor_of(root.type).dtype

                def element(dtype):
                    return dtype.elem_type if isinstance(dtype, VectorType) else dtype

                # Only numeric-element-preserving lane regrouping commutes
                # with conversion. Reinterpreting scalar bits does not.
                if element(source_dtype) != element(view_dtype):
                    return root
                dtype = (
                    VectorType(element(output_dtype), source_dtype.lanes)
                    if isinstance(source_dtype, VectorType)
                    else element(output_dtype)
                )
                lifted = make_node("tensors.cast", root.id, (producer,), {"dtype": dtype}, {})
                view_attrs = {"dtype": output_dtype}
            body = Fusion.from_call(lifted, producer)
            post = producer.attrs.get("post_ops", ())
            if post:
                if len(post) != 1:
                    return root
                if post[0] is not None:
                    body = post[0].then(body)
            candidate = make_node(
                producer.op,
                producer.id,
                tuple(module.node_map[i] for i in producer.inputs),
                {**producer.attrs, "post_ops": (body,)},
                producer.metadata,
            )
            replacement = make_node(view.op, root.id, (candidate,), view_attrs, view.metadata)
        except IRSchemaError:
            return root
        if replacement.type != root.type or not supported(candidate, module):
            return root
        if other_users:
            # A shape-only reference may share the old view. Its schema must
            # promise no data access, and substituting the widened view must
            # preserve both its output type and effects. Publish the pure
            # producer/view before the earliest such reference; do not move
            # or duplicate the effectful consumer itself.
            shadow_id = root.id + ".shape_reference"
            if shadow_id in module.node_map:
                return root
            shadow = make_node(view.op, shadow_id, (candidate,), view_attrs, view.metadata)
            proof = shape_only_replacements(module, view, shadow, root.id)
            if proof is None:
                return root
            extras, anchor = proof
            return RewriteResult(replacement, (candidate, shadow), extra_replacements=tuple(extras),
                                 removed_ids=(producer.id, view.id), insertion_before=anchor)
        if candidate.type == root.type:
            return RewriteResult(replace(candidate, id=root.id), removed_ids=(producer.id, view.id))
        return RewriteResult(replacement, (candidate,), removed_ids=(producer.id, view.id))

    return RewriteRule(f"FusePostOpsThroughView.{op}.{view_op}.{cast_op}", pattern, rewrite, supports_fusion=True)


__all__ = ["post_ops_through_view_rules"]
