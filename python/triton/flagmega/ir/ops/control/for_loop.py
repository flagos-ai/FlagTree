# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Typed structured recurrence with explicit captures and a closed body."""

from triton.flagmega.errors import IRSchemaError, IRVerificationError
from triton.flagmega.ir.model import DType, Effect, TensorType, TupleType, logical_type
from triton.flagmega.ir.ops.core import (
    NodeRef,
    OpDefinition,
    PythonCall,
    attribute_parameter,
    input_parameter,
    op_definition,
    variadic_input_parameter,
)
from triton.flagmega.ir.region import is_region, region_effect
from triton.flagmega.ir.type_pattern import is_ir_type


@op_definition("control.for", namespace="control", functional_name="for_loop", display_name="For")
class ForLoop(OpDefinition):
    supports_broadcast_lifting = False
    lower = input_parameter(is_ir_type(), parameter_kind="attribute")
    upper = input_parameter(is_ir_type(), parameter_kind="attribute")
    step = input_parameter(is_ir_type(), parameter_kind="attribute")
    arguments = variadic_input_parameter(is_ir_type())
    callee = attribute_parameter()
    iter_count = attribute_parameter()
    effect_kind = attribute_parameter(default="pure")
    effect_resource = attribute_parameter(default=None)

    @classmethod
    def normalize_attrs(cls, attrs):
        result = super().normalize_attrs(attrs)
        if not isinstance(result["callee"], str) or not result["callee"]:
            raise IRSchemaError("For requires a named body Region.")
        if type(result["iter_count"]) is not int or result["iter_count"] < 0:
            raise IRSchemaError("For iter_count must be a nonnegative integer.")
        return result

    @classmethod
    def infer_type(cls, inputs, attrs):
        count = int(attrs["iter_count"])
        if len(inputs) < 3 + count:
            raise IRSchemaError("For is missing initial iter arguments.")
        return TupleType(tuple(value.type for value in inputs[3:3 + count]))

    @classmethod
    def infer_effect(cls, inputs, attrs):
        return Effect(attrs["effect_kind"], attrs["effect_resource"])

    @classmethod
    def ir_attrs(cls, attrs):
        return {
            "callee": attrs["callee"],
            "iter_count": attrs["iter_count"],
            "effect_kind": attrs["effect_kind"],
            "effect_resource": attrs["effect_resource"],
        }

    @classmethod
    def verify(cls, node, module):
        cls.verify_arity(node)
        if set(node.attrs) != {"callee", "iter_count", "effect_kind", "effect_resource"}:
            raise IRVerificationError("For has invalid semantic attributes.", node_id=node.id)
        inputs = tuple(module.node_map[value] for value in node.inputs)
        for value in inputs[:3]:
            if logical_type(value.type) != TensorType(DType.INT64, ()):
                raise IRVerificationError("For bounds and step must be scalar i64.", node_id=node.id)
        step = inputs[2]
        if step.op not in {"builtin.scalar_const", "tir.scalar_const"} or int(step.attrs["value"]) <= 0:
            raise IRVerificationError("For step requires a proven positive constant.", node_id=node.id)
        region = module.function_map.get(str(node.attrs["callee"]))
        count = node.attrs["iter_count"]
        if region is None and module.metadata.get("_dump_function_fragment") is True:
            # A per-function dump view holds one function, so the body Region
            # lives in its own sibling checkpoint. Signature, effect and arity
            # agreement are verified on the merged module.
            if cls.infer_type(inputs, node.attrs) != node.type:
                raise IRVerificationError("For result type differs from its iter arguments.", node_id=node.id)
            return
        if region is None or not is_region(region) or region.attrs["iter_count"] != count:
            raise IRVerificationError("For body must reference a Region with matching iter arguments.", node_id=node.id)
        expected = tuple(module.node_map[value].type for value in region.parameters)
        actual = (TensorType(DType.INT64, ()), *(value.type for value in inputs[3:]))
        if actual != expected or cls.infer_type(inputs, node.attrs) != node.type:
            raise IRVerificationError("For capture/iter argument types differ from the body signature.", node_id=node.id)
        if node.effect != region_effect(module, region):
            raise IRVerificationError("For effect differs from its body Region.", node_id=node.id)

    @classmethod
    def python_call(cls, node):
        count = node.attrs["iter_count"]
        return PythonCall("F.control.for_loop", tuple(NodeRef(value) for value in node.inputs[:3 + count]),
                          {"body": node.attrs["callee"],
                           "captures": tuple(NodeRef(value) for value in node.inputs[3 + count:]),
                           "effect_kind": node.effect.kind.value, "effect_resource": node.effect.resource,
                           "name": node.id, **({"metadata": node.metadata} if node.metadata else {})})


def static_trip_count(node, module):
    values = tuple(module.node_map[value] for value in node.inputs[:3])
    if any(value.op not in {"builtin.scalar_const", "tir.scalar_const"} for value in values):
        return None
    lower, upper, step = (int(value.attrs["value"]) for value in values)
    if step <= 0:
        raise IRVerificationError("For step must be positive.", node_id=node.id)
    return max(0, (upper - lower + step - 1) // step)


__all__ = ["ForLoop", "static_trip_count"]
