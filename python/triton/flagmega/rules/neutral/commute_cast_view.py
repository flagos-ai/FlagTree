# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Expose private numeric casts behind value-preserving coordinate views."""

from triton.flagmega.errors import IRSchemaError
from triton.flagmega.ir import DType, VectorType, get_definition
from triton.flagmega.ir.distributed_inference import tensor_of
from triton.flagmega.ir.ops.tensors.pack import normalize_axes
from triton.flagmega.pattern_match import OpPattern, is_call, wildcard
from triton.flagmega.rules import RewriteResult, RewriteRule
from triton.flagmega.rules.neutral._utility import make_node


def _element(dtype):
    return dtype.elem_type if isinstance(dtype, VectorType) else dtype


def _with_element(dtype, element):
    return VectorType(element, dtype.lanes) if isinstance(dtype, VectorType) else element


def commute_cast_view_rules():
    return tuple(_rule(view, cast) for view in ("distributed.sharded_view", "tensors.bitcast")
                 for cast in ("tensors.cast", "ntt.vectorized_cast"))


def _rule(view_op, cast_op):
    cast = is_call(OpPattern(cast_op), wildcard("value"), name="cast").with_user_count(1)
    pattern = is_call(OpPattern(view_op), cast, name="view")

    def rewrite(match, module):
        view, cast, value = match["view"], match["cast"], match["value"]
        if (not view.effect.is_pure or not cast.effect.is_pure
                or any(cast.id in function.outputs for function in module.functions)
                or any(getattr(node.type, "partial", None) is not None for node in (view, cast, value))):
            return view
        source_type, cast_type, result_type = (tensor_of(node.type) for node in (value, cast, view))
        if any(not isinstance(_element(t.dtype), DType) for t in (source_type, cast_type, result_type)):
            return view
        if cast_op == "tensors.cast" and getattr(source_type.dtype, "lanes", ()) != getattr(cast_type.dtype, "lanes", ()):
            return view
        identity = f"{view.id}.cast_input"
        while identity in module.node_map:
            identity += ".view"
        metadata = {**cast.metadata, **view.metadata, "formed_by": "CommuteCastView"}
        try:
            if view_op == "distributed.sharded_view":
                inverse_attrs = ({"new_type": source_type.dtype, "vectorize_axes": cast.attrs["vectorize_axes"]}
                                 if cast_op == "ntt.vectorized_cast" else {"dtype": source_type.dtype})
                input_type = get_definition(cast_op).prepare((view,), inverse_attrs).result_type
                source_view = (value if value.type == input_type else make_node(
                    view_op, identity, (value,), {"new_type": input_type}, {"formed_by": "CommuteCastView"}))
                replacement = make_node(cast_op, view.id, (source_view,), cast.attrs, metadata)
            else:
                # Bit reinterpretation of scalar values is not a coordinate
                # view. Only lane regrouping along the physical trailing axis
                # commutes with a numeric conversion here.
                if _element(cast_type.dtype) != _element(result_type.dtype):
                    return view
                if cast_op == "ntt.vectorized_cast" and set(normalize_axes(
                    cast.attrs["vectorize_axes"], cast_type.rank)) != {cast_type.rank - 1}:
                    return view
                narrow_dtype = _with_element(result_type.dtype, _element(source_type.dtype))
                source_view = make_node(view_op, identity, (value,), {"dtype": narrow_dtype},
                                        {"formed_by": "CommuteCastView"})
                if source_view.type == value.type:
                    source_view = value
                metadata = {key: item for key, item in metadata.items()
                            if key not in {"selected_vectorization", "selected_vector_axes", "selected_vector_lanes"}}
                if isinstance(result_type.dtype, VectorType):
                    replacement = make_node("ntt.vectorized_cast", view.id, (source_view,), {
                        "new_type": result_type.dtype, "vectorize_axes": (result_type.rank - 1,),
                    }, metadata)
                else:
                    replacement = make_node("tensors.cast", view.id, (source_view,), {"dtype": result_type.dtype}, metadata)
        except IRSchemaError:
            return view
        if replacement.type != view.type:
            return view
        return RewriteResult(replacement, () if source_view is value else (source_view,), removed_ids=(cast.id,))

    return RewriteRule(f"CommuteCastView.{view_op}.{cast_op}", pattern, rewrite)


__all__ = ["commute_cast_view_rules"]
