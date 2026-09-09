# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""FP32 decay and update gates, separate from normalization and state updates.

Compute g = -fast_exp(a_log) * stable_softplus(a + dt_bias), then alpha =
exp(g), beta = fast_sigmoid(b). The outer exponential is an explicit
fast/accurate numerical choice; it need not share the inner exp implementation.
The CPU evaluator is an arithmetic reference, not an exact fast-intrinsic oracle.
"""

import math
from itertools import product

from triton.flagmega.errors import IRSchemaError
from triton.flagmega.ir.distributed_inference import placement_of, tensor_of
from triton.flagmega.ir.model import DistributedType, TupleType, tensor_type
from triton.flagmega.ir.ops.core import OpCost, OpDefinition, attribute_parameter, input_parameter, op_definition, tensor_nbytes
from triton.flagmega.ir.type_pattern import has_rank, is_tensor
from triton.flagmega.ir.types import DType


@op_definition("nn.delta_rule_gates", namespace="nn", functional_name="delta_rule_gates",
               display_name="NN.DeltaRuleGates")
class DeltaRuleGates(OpDefinition):
    supports_broadcast_lifting = False
    a = input_parameter(is_tensor() & has_rank(2))
    b = input_parameter(is_tensor() & has_rank(2))
    a_log = input_parameter(is_tensor() & has_rank(1))
    dt_bias = input_parameter(is_tensor() & has_rank(1))
    softplus_threshold = attribute_parameter(default=20.)
    alpha_exp_mode = attribute_parameter(default="accurate")

    @classmethod
    def normalize_attrs(cls, attributes):
        attrs = super().normalize_attrs(attributes)
        threshold = attrs["softplus_threshold"]
        if (isinstance(threshold, bool) or not isinstance(threshold, (int, float))
                or not math.isfinite(threshold) or threshold <= 0):
            raise IRSchemaError("DeltaRuleGates softplus_threshold must be finite and positive.")
        if attrs["alpha_exp_mode"] not in ("fast", "accurate"):
            raise IRSchemaError("DeltaRuleGates alpha_exp_mode must be fast or accurate.")
        return {**attrs, "softplus_threshold": float(threshold)}

    @classmethod
    def infer_type(cls, inputs, attrs):
        types = tuple(parameter.type_of(inputs) for parameter in cls.input_parameters)
        a, b, a_log, bias = tuple(tensor_of(value) for value in types)
        if any(value.dtype not in (DType.BFLOAT16, DType.FLOAT32) for value in (a, b, a_log, bias)):
            raise IRSchemaError("DeltaRuleGates requires scalar BF16/FP32 elements.")
        if a.shape != b.shape or a_log.shape != a.shape[1:] or bias.shape != a_log.shape:
            raise IRSchemaError("DeltaRuleGates token/head dimensions and parameter heads must match.")
        if a.shape[1].is_fixed and a.shape[1].fixed_value <= 0:
            raise IRSchemaError("DeltaRuleGates head extent must be positive.")
        result = tensor_type(DType.FLOAT32, a.shape)
        placement = placement_of(*types)
        if placement is not None:
            if not all(isinstance(value, DistributedType) for value in types):
                raise IRSchemaError("DeltaRuleGates requires every input to name its placement.")
            policies = types[0].axis_policies
            if (any(value.partial is not None for value in types) or types[1].axis_policies != policies
                    or any(value.axis_policies != policies[1:] for value in types[2:])):
                raise IRSchemaError("DeltaRuleGates requires materialized, matching token/head ownership.")
            result = DistributedType(result, policies, placement)
        return TupleType((result, result))

    @classmethod
    def distributed_input_type_tuples(cls, choices, attrs):
        # a determines the only compatible b/parameter policies. Intersect
        # exact available types; do not invent a reshard or prune valid plans.
        indexes = []
        plain = []
        for values in choices:
            by_policy = {}
            plain.append(tuple(value for value in values if not isinstance(value, DistributedType)))
            for value in values:
                if isinstance(value, DistributedType) and value.partial is None:
                    by_policy.setdefault((value.placement, value.axis_policies), []).append(value)
            indexes.append(by_policy)
        for a in choices[cls.a.input_index]:
            if not isinstance(a, DistributedType):
                yield from ((a, *others) for others in product(*plain[1:]))
                continue
            if a.partial is not None:
                continue
            others_by_parameter = []
            for parameter in (cls.b, cls.a_log, cls.dt_bias):
                policies = a.axis_policies if parameter is cls.b else a.axis_policies[1:]
                others_by_parameter.append(indexes[parameter.input_index].get((a.placement, policies), ()))
            for others in product(*others_by_parameter):
                yield (a, *others)

    @classmethod
    def evaluate(cls, node, arguments, context):
        torch = context.torch
        a, b, a_log, bias = (parameter.read(arguments).float() for parameter in cls.input_parameters)
        x = a + bias
        softplus = torch.where(x > 0, x + torch.log(1. + torch.exp(-x)), torch.log(1. + torch.exp(x)))
        softplus = torch.where(x <= node.attrs["softplus_threshold"], softplus, x)
        g = -a_log.exp() * softplus
        return g.exp(), b.sigmoid()

    @classmethod
    def cost(cls, node):
        return OpCost(bytes_written=sum(tensor_nbytes(tensor_of(value)) for value in node.type.fields),
                      notes=("fp32-stable-softplus-and-gates", "explicit-outer-exp"))


__all__ = ["DeltaRuleGates"]
