# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""A single-token KDA step with an explicit request phase."""

from dataclasses import replace

from triton.flagmega.errors import IRSchemaError
from triton.flagmega.ir.distributed_inference import tensor_of
from triton.flagmega.ir.model import DistributedType
from triton.flagmega.ir.ops.core import (
    OpDefinition,
    attribute_parameter,
    input_parameter,
    op_definition,
)
from triton.flagmega.ir.ops.nn.kda_gated_recurrent_core import KdaGatedRecurrentCore
from triton.flagmega.ir.type_pattern import is_tensor
from triton.flagmega.ir.types import DType


def _phase_matches(phase, query):
    """The phase is shared by every head, including across device owners."""
    if not isinstance(query, DistributedType):
        return not isinstance(phase, DistributedType)
    return (isinstance(phase, DistributedType) and phase.placement == query.placement
            and phase.partial is None and phase.exclusive is None)


@op_definition("nn.kda_gated_step", namespace="nn", functional_name="kda_gated_step",
               display_name="NN.KdaGatedStep")
class KdaGatedStep(OpDefinition):
    """Dispatch the explicit single-token chunk/recurrent numerical contracts.

    A scalar BOOL ``prefill`` selects ``chunk_bf16`` when true and the FP32
    recurrence when false. It is an ordinary input, independent of the contents
    of the state. Both paths support nonzero initial state. The state remains
    FP32 storage, while the chunk path retains its BF16 conversion boundaries.
    Existing static KDA operations and their default arithmetic are unchanged.
    """

    const_evaluable = True
    query = input_parameter(is_tensor())
    key = input_parameter(is_tensor())
    value = input_parameter(is_tensor())
    gate = input_parameter(is_tensor())
    decay_rate = input_parameter(is_tensor())
    gate_bias = input_parameter(is_tensor())
    beta_logits = input_parameter(is_tensor())
    state = input_parameter(is_tensor())
    prefill = input_parameter(is_tensor())
    lower_bound = attribute_parameter()
    epsilon = attribute_parameter(default=1e-6)
    l2norm_in_kernel = attribute_parameter(default=True)
    query_scale = attribute_parameter(default=1.0)
    output_lanes = attribute_parameter(default=())
    decay_input = attribute_parameter(default="rate")

    @classmethod
    def normalize_attrs(cls, attributes):
        attrs = super().normalize_attrs(attributes)
        normalized = KdaGatedRecurrentCore.normalize_attrs({**attrs, "arithmetic": "chunk_bf16"})
        return {key: value for key, value in normalized.items() if key != "arithmetic"}

    @classmethod
    def ir_attrs(cls, attrs):
        return KdaGatedRecurrentCore.ir_attrs(attrs)

    @classmethod
    def infer_type(cls, inputs, attrs):
        # Validate both branches: the chunk path's Dk=Dv=128 requirement is
        # stronger than the recurrent path, so a runtime phase cannot evade it.
        result = KdaGatedRecurrentCore.infer_type(inputs[:8], {**attrs, "arithmetic": "chunk_bf16"})
        phase = cls.prefill.type_of(inputs)
        logical = tensor_of(phase)
        if logical.rank != 0 or logical.dtype != DType.BOOL:
            raise IRSchemaError("KdaGatedStep prefill must be a scalar BOOL tensor.")
        if not _phase_matches(phase, cls.query.type_of(inputs)):
            raise IRSchemaError("KdaGatedStep prefill must be broadcast on the operands' placement.")
        return result

    @classmethod
    def distributed_input_type_tuples(cls, choices, attrs):
        for operands in KdaGatedRecurrentCore.distributed_input_type_tuples(choices[:8], attrs):
            for phase in choices[8]:
                if _phase_matches(phase, operands[0]):
                    yield (*operands, phase)

    @classmethod
    def distributed_output_type_candidates(cls, choices, output_type, attrs):
        return KdaGatedRecurrentCore.distributed_output_type_candidates(choices[:8], output_type, attrs)

    @classmethod
    def infer_distributed_input_types(cls, output_type, logical_input_types, attrs):
        for operands in KdaGatedRecurrentCore.infer_distributed_input_types(
                output_type, logical_input_types[:8], attrs):
            yield (*operands, DistributedType(logical_input_types[8], (), operands[0].placement))

    @classmethod
    def infer_effect(cls, inputs, attrs):
        return KdaGatedRecurrentCore.infer_effect(inputs[:8], attrs)

    @classmethod
    def evaluate(cls, node, arguments, context):
        prefill = bool(cls.prefill.read(arguments).item())
        static = replace(node, attrs={**node.attrs, "arithmetic": "chunk_bf16" if prefill else "recurrent"})
        return KdaGatedRecurrentCore.evaluate(static, arguments[:8], context)

    @classmethod
    def cost(cls, node):
        return KdaGatedRecurrentCore.cost(node)


__all__ = ["KdaGatedStep"]
