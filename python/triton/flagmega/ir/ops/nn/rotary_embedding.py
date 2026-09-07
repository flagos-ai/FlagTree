# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Position-dependent cosine/sine tensors for rotary embeddings."""

from __future__ import annotations

from typing import Mapping, Sequence

from triton.flagmega.errors import EvaluationError, IRSchemaError
from triton.flagmega.ir.distributed_inference import tensor_of
from triton.flagmega.ir.memory_effect import MemoryEffect
from triton.flagmega.ir.model import (
    DType,
    Effect,
    IRType,
    Node,
    TupleType,
    effect,
    tensor_type,
)
from triton.flagmega.ir.ops.core import (
    OpCost,
    OpDefinition,
    attribute_parameter,
    input_parameter,
    op_definition,
)
from triton.flagmega.ir.ops.nn._paged_attention_state import PagedAttentionState
from triton.flagmega.ir.type_pattern import has_rank, is_ref, is_tensor


@op_definition(
    "nn.rotary_embedding",
    namespace="nn",
    functional_name="rotary_embedding",
    display_name="NN.RotaryEmbedding",
)
class RotaryEmbedding(OpDefinition):
    """Create the nncase importer helper's decode-time cos/sin pair.

    nncase spells this helper as primitive GetPositionIds/arithmetic/unsqueeze
    operations.  FlagMega keeps it as one target-independent semantic op until
    those general broadcasting primitives are present; RoPE itself remains a
    separate operation and the pair is editable in Python IR.
    """

    reference = input_parameter(
        is_tensor() & has_rank(2), memory_effect=MemoryEffect.NONE
    )
    state = input_parameter(is_ref(), memory_effect=MemoryEffect.CHIP_READ)
    head_dim = attribute_parameter()
    theta = attribute_parameter()
    attention_scaling = attribute_parameter(default=1.0)

    @classmethod
    def normalize_attrs(cls, attributes: Mapping[str, object]) -> dict[str, object]:
        attrs = super().normalize_attrs(attributes)
        head_dim = attrs["head_dim"]
        if isinstance(head_dim, bool) or not isinstance(head_dim, int) or head_dim <= 0:
            raise IRSchemaError("RotaryEmbedding head_dim must be a positive integer.")
        if head_dim % 2:
            raise IRSchemaError("RotaryEmbedding head_dim must be even.")
        theta = float(attrs["theta"])
        scaling = float(attrs["attention_scaling"])
        if theta <= 0 or scaling <= 0:
            raise IRSchemaError("RotaryEmbedding theta and attention_scaling must be positive.")
        return {
            "head_dim": head_dim,
            "theta": theta,
            "attention_scaling": scaling,
        }

    @classmethod
    def infer_type(cls, inputs: Sequence[Node], attrs: Mapping[str, object]) -> IRType:
        reference = tensor_of(cls.reference.type_of(inputs))
        result = tensor_type(
            DType.FLOAT32,
            (reference.shape[0], 1, int(attrs["head_dim"])),
        )
        return TupleType((result, result))

    @classmethod
    def infer_effect(cls, inputs: Sequence[Node], attrs: Mapping[str, object]) -> Effect:
        return effect("read", "paged_attention_kv_cache")

    @classmethod
    def evaluate(cls, node, arguments, context):
        reference = cls.reference.read(arguments)
        state = cls.state.read(arguments)
        if not isinstance(state, PagedAttentionState):
            raise EvaluationError(
                "RotaryEmbedding state must evaluate to PagedAttentionState.")
        state.validate()
        sequence = int(reference.shape[0])
        head_dim = int(node.attrs["head_dim"])
        positions = context.torch.arange(
            state.sequence_length,
            state.sequence_length + sequence,
            dtype=context.torch.float32,
            device=reference.device,
        )
        indices = context.torch.arange(
            0,
            head_dim,
            2,
            dtype=context.torch.float32,
            device=reference.device,
        )
        inverse_frequency = float(node.attrs["theta"]) ** (-indices / head_dim)
        angles = context.torch.outer(positions, inverse_frequency)
        frequency = context.torch.cat((angles, angles), dim=-1).unsqueeze(1)
        scale = float(node.attrs["attention_scaling"])
        return frequency.cos() * scale, frequency.sin() * scale

    @classmethod
    def cost(cls, node: Node) -> OpCost:
        return OpCost(notes=("position-dependent-rotary-cos-sin",))


__all__ = ["RotaryEmbedding"]
