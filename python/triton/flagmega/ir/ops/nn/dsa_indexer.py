# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Reference DSA indexer with packed per-layer state."""

from __future__ import annotations

from typing import Mapping, Sequence

from triton.flagmega.errors import EvaluationError, IRSchemaError
from triton.flagmega.ir.memory_effect import MemoryEffect
from triton.flagmega.ir.distributed_inference import tensor_of
from triton.flagmega.ir.model import DType, DistributedType, Effect, IRType, Node, RefType, TensorType, TupleType, effect
from triton.flagmega.ir.distributed_type import SBPBroadCast
from triton.flagmega.ir.ops.core import OpCost, OpDefinition, ParameterKind, attribute_parameter, input_parameter, op_definition
from triton.flagmega.ir.ops.nn._dsa_indexer_state import DsaIndexerState
from triton.flagmega.ir.ops.nn.qwen3_paged_attention import _scalar_bool, _scalar_int
from triton.flagmega.ir.type_pattern import has_dtype, has_rank, is_ref, is_tensor


def _dsa_broadcast_choices(choices, tensor_indices):
    """Keep tensor operands either plain or one common fully-broadcast mesh."""
    tensor_indices = frozenset(tensor_indices)

    def product(index=0, current=(), mode=None, placement=None):
        if index == len(choices):
            yield current
            return
        for value in choices[index]:
            next_mode, next_placement = mode, placement
            if index in tensor_indices:
                distributed = isinstance(value, DistributedType)
                if mode is not None and distributed != mode:
                    continue
                if distributed:
                    if (value.partial is not None or value.exclusive is not None
                            or any(not isinstance(policy, SBPBroadCast) for policy in value.axis_policies)
                            or (placement is not None and value.placement != placement)):
                        continue
                    next_placement = value.placement
                next_mode = distributed
            yield from product(index + 1, current + (value,), next_mode, next_placement)

    yield from product()


@op_definition(
    "nn.dsa_indexer",
    namespace="nn",
    functional_name="dsa_indexer",
    display_name="NN.DSAIndexer",
)
class DsaIndexer(OpDefinition):
    q = input_parameter(is_tensor() & has_rank(2))
    key = input_parameter(is_tensor() & has_rank(2))
    gate = input_parameter(is_tensor() & has_rank(2))
    weights = input_parameter(is_tensor() & has_rank(2))
    state = input_parameter(is_ref(), parameter_kind=ParameterKind.ATTRIBUTE,
                             memory_effect=MemoryEffect.CHIP_READ_WRITE.partitioned_by_argument(4))
    layer_id = input_parameter(is_tensor() & has_rank(0) & has_dtype(DType.INT32),
                               parameter_kind=ParameterKind.ATTRIBUTE)
    advance_sequence = input_parameter(is_tensor() & has_rank(0) & has_dtype(DType.BOOL),
                                       parameter_kind=ParameterKind.ATTRIBUTE)
    k_norm_weight = input_parameter(is_tensor() & has_rank(1))
    k_norm_bias = input_parameter(is_tensor() & has_rank(1))
    compress_ape = input_parameter(is_tensor() & has_rank(2))
    index_topk = attribute_parameter()
    index_head_dim = attribute_parameter()
    index_n_heads = attribute_parameter()
    index_kpool = attribute_parameter()
    always_select_tail = attribute_parameter()
    epsilon = attribute_parameter()
    state_layer_ids = attribute_parameter(default=())

    @classmethod
    def normalize_attrs(cls, attributes: Mapping[str, object]) -> dict[str, object]:
        attrs = super().normalize_attrs(attributes)
        for name in ("index_topk", "index_head_dim", "index_n_heads", "index_kpool"):
            value = attrs[name]
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise IRSchemaError(f"DSA indexer {name} must be positive.")
        if attrs["index_topk"] % attrs["index_kpool"]:
            raise IRSchemaError("DSA index_topk must be divisible by index_kpool.")
        if not isinstance(attrs["always_select_tail"], bool):
            raise IRSchemaError("DSA always_select_tail must be boolean.")
        epsilon = float(attrs["epsilon"])
        if epsilon <= 0:
            raise IRSchemaError("DSA indexer epsilon must be positive.")
        attrs["epsilon"] = epsilon
        layer_ids = attrs["state_layer_ids"]
        if (not isinstance(layer_ids, (tuple, list))
                or any(isinstance(value, bool) or not isinstance(value, int) or value < 0
                       for value in layer_ids)
                or len(set(layer_ids)) != len(layer_ids)):
            raise IRSchemaError("DSA state_layer_ids must be unique non-negative integers.")
        attrs["state_layer_ids"] = tuple(layer_ids)
        return attrs

    @classmethod
    def infer_type(cls, inputs: Sequence[Node], attrs: Mapping[str, object]) -> IRType:
        q_source = cls.q.type_of(inputs)
        q = tensor_of(q_source)
        state = cls.state.type_of(inputs)
        if not isinstance(q_source, (TensorType, DistributedType)) or not isinstance(state, RefType):
            raise IRSchemaError("DSA indexer requires tensor q input and a state reference.")
        if attrs["state_layer_ids"]:
            packed = dict(state.fields).get("packed_states")
            if (not isinstance(packed, TensorType) or len(packed.shape) != 3
                    or not packed.shape[0].is_fixed
                    or packed.shape[0].fixed_value != len(attrs["state_layer_ids"])):
                raise IRSchemaError("DSA state_layer_ids must describe every packed cache layer.")
        tokens = q.shape[0]
        width = int(attrs["index_topk"]) + (int(attrs["index_kpool"]) - 1 if attrs["always_select_tail"] else 0)
        plain = TensorType(DType.INT32, (tokens, width))
        topk = (DistributedType(plain, q_source.axis_policies, q_source.placement,
                                q_source.partial, q_source.exclusive)
                if isinstance(q_source, DistributedType) else plain)
        # The reference is mutated in place, matching paged/KDA state ops;
        # carrying it as a separate function result would make ref tuple
        # extraction an artificial compiler boundary.
        return topk

    @classmethod
    def distributed_input_type_tuples(cls, choices, attrs):
        yield from _dsa_broadcast_choices(choices, (0, 1, 2, 3, 7, 8, 9))

    @classmethod
    def infer_effect(cls, inputs: Sequence[Node], attrs: Mapping[str, object]) -> Effect:
        return effect("read_write", "dsa_indexer_state")

    @classmethod
    def evaluate(cls, node, arguments, context):
        import torch
        q_flat = cls.q.read(arguments)
        key = cls.key.read(arguments)
        gate = cls.gate.read(arguments)
        weights = cls.weights.read(arguments)
        state = cls.state.read(arguments)
        if not isinstance(state, DsaIndexerState):
            raise EvaluationError("DSA indexer state must evaluate to DsaIndexerState.")
        if node.attrs.get("state_layer_ids") and tuple(node.attrs["state_layer_ids"]) != state.config.layer_ids:
            raise EvaluationError("DSA state_layer_ids disagree with the runtime state contract.")
        layer = _scalar_int(cls.layer_id.read(arguments))
        advance = _scalar_bool(cls.advance_sequence.read(arguments))
        d = int(node.attrs["index_head_dim"])
        heads = int(node.attrs["index_n_heads"])
        pool = int(node.attrs["index_kpool"])
        topk = int(node.attrs["index_topk"])
        eps = float(node.attrs["epsilon"])
        tokens = q_flat.shape[0]
        q = q_flat.float().reshape(tokens, heads, d)
        k = key.float()
        norm_weight = cls.k_norm_weight.read(arguments).float()
        norm_bias = cls.k_norm_bias.read(arguments).float()
        k = (k - k.mean(-1, keepdim=True)) / torch.sqrt(k.var(-1, unbiased=False, keepdim=True) + eps)
        k = (k * norm_weight + norm_bias).to(key.dtype)
        gate = gate.to(key.dtype)
        valid = torch.ones((tokens, 1), dtype=key.dtype, device=key.device)
        packed = torch.cat((k, gate, valid), dim=-1)
        base = state.sequence_length
        state.append(packed, layer_id=layer, advance_sequence=advance)
        length = base + tokens
        packed_states = state.history(layer_id=layer, length=length)
        keys = packed_states[:, :d]
        gate_scores = packed_states[:, d:2 * d]
        valid_keys = packed_states[:, -1].bool()
        first = int(torch.nonzero(valid_keys, as_tuple=False)[0].item()) if bool(valid_keys.any()) else length
        pool_count = (length - first + pool - 1) // pool if first < length else 0
        if pool_count:
            offsets = torch.arange(pool_count * pool, device=key.device).reshape(pool_count, pool)
            pool_indices = first + offsets
            safe = pool_indices.clamp(0, max(length - 1, 0))
            grouped_keys = keys[safe]
            grouped_gate = gate_scores[safe]
            grouped_valid = valid_keys[safe] & (pool_indices < length)
            pool_valid = grouped_valid.all(-1)
            logits = grouped_gate.float() + cls.compress_ape.read(arguments).float()[None, :, :]
            logits = logits.masked_fill(~grouped_valid[:, :, None], float("-inf"))
            probabilities = torch.nan_to_num(torch.softmax(logits, dim=1)).to(keys.dtype)
            pool_keys = (probabilities * grouped_keys).sum(dim=1)
            scores = torch.einsum("thd,pd->thp", q.float(), pool_keys.float())
            scores = torch.relu(scores * (d ** -0.5))
            index_scores = (weights.float()[:, :, None] * scores).sum(dim=1)
        else:
            pool_indices = torch.empty((0, pool), dtype=torch.int64, device=key.device)
            pool_valid = torch.empty((0,), dtype=torch.bool, device=key.device)
            index_scores = torch.empty((tokens, 0), dtype=torch.float32, device=key.device)
        visible = torch.zeros((tokens, length), dtype=torch.bool, device=key.device)
        positions = base + torch.arange(tokens, device=key.device)
        visible = (torch.arange(length, device=key.device)[None, :] <= positions[:, None]) & valid_keys[None, :]
        pool_end = pool_indices[:, -1].clamp(0, max(length - 1, 0)) if pool_count else pool_indices.new_empty((0,))
        candidates = (visible[:, pool_end] if pool_count else visible.new_empty((tokens, 0)))
        candidates &= pool_valid[None, :] if pool_count else candidates
        select = min(topk // pool, index_scores.shape[-1])
        if select:
            masked = index_scores.masked_fill(~candidates, torch.finfo(index_scores.dtype).min)
            selected = torch.topk(masked, select, dim=-1).indices
            selected_valid = candidates.gather(-1, selected)
            selected_indices = pool_indices[selected]
            selected_indices = selected_indices.masked_fill(~selected_valid[..., None], -1).reshape(tokens, -1)
        else:
            selected_indices = torch.empty((tokens, 0), dtype=torch.int64, device=key.device)
        width = topk + (pool - 1 if node.attrs["always_select_tail"] else 0)
        result = torch.full((tokens, width), -1, dtype=torch.int32, device=key.device)
        result[:, :min(selected_indices.shape[1], width)] = selected_indices[:, :width].to(torch.int32)
        if node.attrs["always_select_tail"] and pool > 1:
            visible_count = visible.sum(-1)
            tail_count = visible_count.remainder(pool)
            tail_start = first + visible_count - tail_count
            offsets = torch.arange(pool - 1, device=key.device)
            tail = tail_start[:, None] + offsets[None, :]
            tail_valid = (offsets[None, :] < tail_count[:, None]) & (tail < length)
            tail = tail.masked_fill(~tail_valid, -1).to(torch.int32)
            begin = topk
            result[:, begin:begin + pool - 1] = tail
        return result

    @classmethod
    def cost(cls, node):
        return OpCost(notes=("dsa-indexer-stateful",))


__all__ = ["DsaIndexer"]
