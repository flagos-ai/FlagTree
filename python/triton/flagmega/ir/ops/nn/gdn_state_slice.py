# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""A one-layer, non-owning view of persistent Gated DeltaNet state.

The dynamic layer operand is part of the reusable decoder ABI. Both result
fields alias the corresponding input field: field byte offset equals layer_id
times the field's single-layer byte size. Lowering must preserve these views,
not allocate or copy new state. This op itself does not read/write field data.
"""

from dataclasses import replace

from triton.flagmega.errors import EvaluationError, IRSchemaError
from triton.flagmega.ir.memory_effect import MemoryEffect
from triton.flagmega.ir.model import DType, RefType, TensorType, tensor_type
from triton.flagmega.ir.ops.core import OpCost, OpDefinition, ParameterKind, input_parameter, op_definition
from triton.flagmega.ir.ops.nn._gdn_state import GatedDeltaNetState, GatedDeltaNetStateDimKind
from triton.flagmega.ir.type_pattern import has_dtype, has_rank, is_ref, is_tensor
from triton.flagmega.ir.types import VectorType


@op_definition("nn.gdn_state_slice", namespace="nn", functional_name="gated_delta_net_state_slice",
               display_name="NN.GatedDeltaNetStateSlice")
class GatedDeltaNetStateSlice(OpDefinition):
    state = input_parameter(is_ref(), parameter_kind=ParameterKind.ATTRIBUTE, memory_effect=MemoryEffect.NONE)
    layer_id = input_parameter(is_tensor() & has_rank(0) & has_dtype(DType.INT32),
                               parameter_kind=ParameterKind.ATTRIBUTE)

    @classmethod
    def infer_type(cls, inputs, attrs):
        source = cls.state.type_of(inputs)
        if source.name != "qwen3_5_gated_delta_net_state":
            raise IRSchemaError("GatedDeltaNetStateSlice requires a configured GDN state reference.")
        fields = dict(source.fields)
        if set(fields) != {"convolution", "recurrent"}:
            raise IRSchemaError("GatedDeltaNetStateSlice requires convolution/recurrent fields.")
        result = []
        for name, rank, dtype in (("convolution", 3, DType.BFLOAT16), ("recurrent", 4, DType.FLOAT32)):
            field = fields[name]
            if (not isinstance(field, TensorType) or field.rank != rank or not isinstance(field.dtype, VectorType)
                    or field.dtype.elem_type != dtype
                    or any(not dim.is_fixed or dim.fixed_value <= 0 for dim in field.shape)):
                raise IRSchemaError(f"GatedDeltaNetStateSlice {name} requires the static packed state ABI.")
            result.append((name, tensor_type(field.dtype, (1, *field.shape[1:]), layout=field.layout)))
        if fields["convolution"].shape[0] != fields["recurrent"].shape[0]:
            raise IRSchemaError("GatedDeltaNetStateSlice fields must have the same layer extent.")
        return RefType(source.name, tuple(result))

    @classmethod
    def evaluate(cls, node, arguments, context):
        state = cls.state.read(arguments)
        if not isinstance(state, GatedDeltaNetState):
            raise EvaluationError("GatedDeltaNetStateSlice requires GatedDeltaNetState backing.")
        layer_id = int(cls.layer_id.read(arguments).item())
        state._validate_layer(layer_id)
        if any(layout[0] != GatedDeltaNetStateDimKind.NUM_LAYERS
               for layout in (state.config.convolution_layout, state.config.recurrent_layout)):
            raise EvaluationError("GatedDeltaNetStateSlice requires a leading layer axis.")
        return GatedDeltaNetState(state.convolution[layer_id:layer_id + 1], state.recurrent[layer_id:layer_id + 1],
                                  replace(state.config, num_layers=1))

    @classmethod
    def cost(cls, node):
        return OpCost(flops=0, bytes_read=0, bytes_written=0, notes=("non-owning-state-layer-view", ))
