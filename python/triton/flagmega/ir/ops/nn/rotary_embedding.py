# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Position-dependent cosine/sine tensors for rotary embeddings."""

from __future__ import annotations

from typing import Mapping, Sequence
from math import isfinite, prod

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
    variadic_input_parameter,
)
from triton.flagmega.ir.ops.nn._paged_attention_state import PagedAttentionState
from triton.flagmega.ir.type_pattern import has_rank, is_ref, is_tensor
from triton.flagmega.ir.types import VectorType


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
    cache = variadic_input_parameter(is_tensor() & has_rank(3), memory_effect=MemoryEffect.CHIP_READ)
    head_dim = attribute_parameter()
    theta = attribute_parameter()
    attention_scaling = attribute_parameter(default=1.0)
    output_lanes = attribute_parameter(default=())
    output_dtype = attribute_parameter(default=DType.FLOAT32)
    math_mode = attribute_parameter(default="fast")
    inverse_frequencies = attribute_parameter(default=())

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
        lanes = tuple(attrs["output_lanes"])
        if any(isinstance(lane, bool) or not isinstance(lane, int) or lane <= 0 for lane in lanes):
            raise IRSchemaError("RotaryEmbedding output_lanes must contain positive integers.")
        if head_dim % prod(lanes):
            raise IRSchemaError("RotaryEmbedding head_dim must be divisible by output_lanes.")
        try:
            output_dtype = DType(attrs["output_dtype"])
        except (ValueError, TypeError) as error:
            raise IRSchemaError("RotaryEmbedding output_dtype must be BF16/FP32.") from error
        if output_dtype not in {DType.BFLOAT16, DType.FLOAT32}:
            raise IRSchemaError("RotaryEmbedding output_dtype must be BF16/FP32.")
        math_mode = attrs["math_mode"]
        if math_mode not in {"fast", "accurate"}:
            raise IRSchemaError("RotaryEmbedding math_mode must be fast or accurate.")
        try:
            frequencies = tuple(float(value) for value in attrs["inverse_frequencies"])
        except (TypeError, ValueError) as error:
            raise IRSchemaError("RotaryEmbedding inverse_frequencies must be a sequence of numbers.") from error
        if frequencies and (len(frequencies) != head_dim // 2 or
                            any(not isfinite(value) or value <= 0 for value in frequencies)):
            raise IRSchemaError("RotaryEmbedding inverse_frequencies must contain head_dim/2 positive finite values.")
        result = {
            "head_dim": head_dim,
            "theta": theta,
            "attention_scaling": scaling,
            "output_lanes": lanes,
            "output_dtype": output_dtype.value,
        }
        if math_mode != "fast":
            result["math_mode"] = math_mode
        if frequencies:
            result["inverse_frequencies"] = frequencies
        return result

    @classmethod
    def ir_attrs(cls, attrs: Mapping[str, object]) -> Mapping[str, object]:
        # Existing FP32 scalar/vector tables keep their canonical attributes.
        # A narrower table dtype is an explicit final-store rounding boundary.
        return {key: value for key, value in attrs.items()
                if (key != "output_lanes" or value) and (key != "output_dtype" or value != "float32")}

    @classmethod
    def infer_type(cls, inputs: Sequence[Node], attrs: Mapping[str, object]) -> IRType:
        reference = tensor_of(cls.reference.type_of(inputs))
        caches = cls.cache.read(inputs)
        if len(caches) > 1:
            raise IRSchemaError("RotaryEmbedding accepts at most one precomputed cache.")
        if caches:
            cache = tensor_of(caches[0].type)
            cache_lanes = prod(cache.dtype.lanes) if isinstance(cache.dtype, VectorType) else 1
            cache_dtype = cache.dtype.elem_type if isinstance(cache.dtype, VectorType) else cache.dtype
            if (cache.rank != 3 or not cache.shape[0].is_fixed or cache.shape[1].fixed_value != 2
                    or cache.shape[2].fixed_value * cache_lanes != int(attrs["head_dim"])
                    or cache_dtype != DType(attrs.get("output_dtype", "float32"))):
                raise IRSchemaError("RotaryEmbedding cache must have [positions, 2, head_dim] and the output dtype.")
        lanes = tuple(attrs.get("output_lanes", ()))
        dtype = DType(attrs.get("output_dtype", DType.FLOAT32))
        result = tensor_type(
            VectorType(dtype, lanes) if lanes else dtype,
            (reference.shape[0], 1, int(attrs["head_dim"]) // prod(lanes)),
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
        if state.sequence_length + sequence > state.config.max_sequence_length:
            raise EvaluationError("RotaryEmbedding query positions exceed the configured paged cache capacity.")
        caches = cls.cache.read(arguments)
        if caches:
            cache = caches[0]
            if state.sequence_length + sequence > cache.shape[0]:
                raise EvaluationError("RotaryEmbedding query positions exceed its precomputed cache.")
            selected = cache[state.sequence_length:state.sequence_length + sequence].to(reference.device)
            head_dim = int(node.attrs["head_dim"])
            lanes = tuple(node.attrs.get("output_lanes", ()))
            shape = (sequence, 1, head_dim // prod(lanes), *lanes)
            return selected[:, 0].reshape(shape), selected[:, 1].reshape(shape)
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
        if node.attrs.get("inverse_frequencies"):
            inverse_frequency = context.torch.tensor(node.attrs["inverse_frequencies"],
                                                     dtype=context.torch.float32, device=reference.device)
        elif node.attrs.get("math_mode", "fast") == "accurate":
            inverse_frequency = 1.0 / (float(node.attrs["theta"]) ** (indices / head_dim))
        else:
            inverse_frequency = float(node.attrs["theta"]) ** (-indices / head_dim)
        angles = context.torch.outer(positions, inverse_frequency)
        frequency = context.torch.cat((angles, angles), dim=-1).unsqueeze(1)
        scale = float(node.attrs["attention_scaling"])
        lanes = tuple(node.attrs.get("output_lanes", ()))
        shape = (sequence, 1, head_dim // prod(lanes), *lanes)
        dtype = getattr(context.torch, node.attrs.get("output_dtype", "float32"))
        return ((frequency.cos() * scale).to(dtype).reshape(shape),
                (frequency.sin() * scale).to(dtype).reshape(shape))

    @classmethod
    def cost(cls, node: Node) -> OpCost:
        return OpCost(notes=("position-dependent-rotary-cos-sin",))


__all__ = ["RotaryEmbedding"]
