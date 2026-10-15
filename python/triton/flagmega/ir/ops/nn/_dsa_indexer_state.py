# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Persistent cache for GLM DSA indexer keys and gate scores."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from triton.flagmega.errors import EvaluationError, IRSchemaError
from triton.flagmega.ir.model import RefType, TensorType, tensor_type
from triton.flagmega.ir.types import DType


@dataclass(frozen=True)
class DsaIndexerStateConfig:
    """Physical ABI for one packed indexer cache per full DSA layer."""

    layer_ids: tuple[int, ...]
    max_sequence_length: int
    index_head_dim: int
    output_width: int

    def __post_init__(self):
        if (not self.layer_ids or len(set(self.layer_ids)) != len(self.layer_ids)
                or any(isinstance(x, bool) or not isinstance(x, int) or x < 0 for x in self.layer_ids)):
            raise IRSchemaError("DSA indexer state requires unique non-negative layer ids.")
        values = (self.max_sequence_length, self.index_head_dim, self.output_width)
        if any(isinstance(x, bool) or not isinstance(x, int) or x <= 0 for x in values):
            raise IRSchemaError("DSA indexer state dimensions must be positive integers.")

    @property
    def packed_shape(self):
        return (self.max_sequence_length, 2 * self.index_head_dim + 1)

    @property
    def ref_type(self):
        return RefType("dsa_indexer_state", (
            ("packed_states", tensor_type(DType.BFLOAT16,
                                            (len(self.layer_ids), *self.packed_shape))),
            ("seq_lens", tensor_type(DType.INT32, (1,))),
        ))


@dataclass
class DsaIndexerState:
    packed_states: Any
    seq_lens: Any
    config: DsaIndexerStateConfig

    def clone(self):
        return DsaIndexerState(self.packed_states.clone(), self.seq_lens.clone(), self.config)

    @property
    def sequence_length(self):
        return int(self.seq_lens[0].item())

    def _index(self, layer_id):
        try:
            return self.config.layer_ids.index(int(layer_id))
        except (ValueError, TypeError) as error:
            raise EvaluationError(f"DSA layer id {layer_id!r} is not in the state contract.") from error

    def validate(self):
        import torch
        expected = (len(self.config.layer_ids), *self.config.packed_shape)
        if tuple(self.packed_states.shape) != expected or self.packed_states.dtype != torch.bfloat16:
            raise EvaluationError(f"DSA packed state must be BF16{expected}.")
        if tuple(self.seq_lens.shape) != (1,) or self.seq_lens.dtype != torch.int32:
            raise EvaluationError("DSA seq_lens must be int32[1].")
        if not 0 <= self.sequence_length <= self.config.max_sequence_length:
            raise EvaluationError("DSA sequence length is outside cache capacity.")

    def append(self, value, *, layer_id, advance_sequence=False):
        self.validate()
        index = self._index(layer_id)
        if value.ndim != 2 or tuple(value.shape[1:]) != (2 * self.config.index_head_dim + 1,):
            raise EvaluationError("DSA packed updates must be [tokens, 2*index_head_dim+1].")
        tokens = int(value.shape[0])
        begin = self.sequence_length
        if begin + tokens > self.config.max_sequence_length:
            raise EvaluationError("DSA indexer cache capacity is exhausted.")
        self.packed_states[index, begin:begin + tokens].copy_(value.to(dtype=self.packed_states.dtype))
        if advance_sequence:
            self.seq_lens[0] = begin + tokens
        return begin

    def history(self, *, layer_id, length=None):
        self.validate()
        if length is None:
            length = self.sequence_length
        if length < 0 or length > self.config.max_sequence_length:
            raise EvaluationError("DSA history length is outside cache capacity.")
        return self.packed_states[self._index(layer_id), :length]


def create_dsa_indexer_state(config: DsaIndexerStateConfig, *, device="cpu"):
    import torch
    state = DsaIndexerState(
        torch.zeros((len(config.layer_ids), *config.packed_shape), dtype=torch.bfloat16, device=device),
        torch.zeros((1,), dtype=torch.int32, device=device), config)
    state.validate()
    return state


__all__ = ["DsaIndexerState", "DsaIndexerStateConfig", "create_dsa_indexer_state"]
