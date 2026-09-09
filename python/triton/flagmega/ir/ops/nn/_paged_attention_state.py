# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Physical paged-attention cache ABI for single-request token chunks.

The semantic axes and vector-lane representation follow nncase's
``PagedAttentionConfig``: ``[NumBlocks, NumLayers, KV, BlockSize,
NumKVHeads, HeadDim / lanes]<lanes>``.  Scheduler tensors are explicit fields
so Python IR checkpoints retain the complete resumable entry ABI.
``slot_mapping[0]`` is the logical base position of the current query chunk,
not a per-token physical-slot list. ``query_start_loc`` is ``[0, num_tokens]``.
Only the final updating layer advances ``seq_lens`` by the chunk length.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from triton.flagmega.errors import EvaluationError, IRSchemaError
from math import prod

from triton.flagmega.ir.model import IRType, RefType, TensorType, tensor_type
from triton.flagmega.ir.types import DType, VectorType


@dataclass(frozen=True)
class PagedAttentionStateConfig:
    num_layers: int
    num_kv_heads: int
    head_dim: int
    block_size: int = 256
    num_blocks: int = 16
    lanes: int = 8

    def __post_init__(self) -> None:
        values = (self.num_layers, self.num_kv_heads, self.head_dim, self.block_size, self.num_blocks, self.lanes)
        if any(isinstance(value, bool) or not isinstance(value, int) or value <= 0 for value in values):
            raise IRSchemaError("Paged-attention dimensions must be positive integers.")
        if self.head_dim % self.lanes:
            raise IRSchemaError(
                f"Paged-attention head dimension {self.head_dim} is not divisible by {self.lanes} lanes.")

    @property
    def max_sequence_length(self) -> int:
        return self.block_size * self.num_blocks

    @property
    def cache_type(self):
        return tensor_type(
            VectorType(DType.BFLOAT16, (self.lanes,)),
            [self.num_blocks, self.num_layers, 2, self.block_size, self.num_kv_heads, self.head_dim // self.lanes],
        )

    @property
    def storage_shape(self) -> tuple[int, ...]:
        return (
            self.num_blocks,
            self.num_layers,
            2,
            self.block_size,
            self.num_kv_heads,
            self.head_dim // self.lanes,
            self.lanes,
        )

    @property
    def ref_type(self) -> RefType:
        return RefType(
            "paged_attention_kv_cache",
            (
                ("kv_caches", self.cache_type),
                ("query_start_loc", tensor_type(DType.INT32, [2])),
                ("seq_lens", tensor_type(DType.INT32, [1])),
                ("slot_mapping", tensor_type(DType.INT64, [1])),
                ("block_table", tensor_type(DType.INT32, [1, self.num_blocks])),
            ),
        )


def paged_attention_state_config_from_type(
    value_type: IRType,
) -> PagedAttentionStateConfig:
    """Recover the semantic cache contract encoded by a resumable RefType.

    The Python checkpoint must carry enough information to resume compilation
    without an importer-side object.  Consequently the cache configuration is
    reconstructed from the reference fields instead of hidden in compiler
    session state, matching nncase's configured reference element type.
    """

    if not isinstance(value_type, RefType) or value_type.name != "paged_attention_kv_cache":
        raise IRSchemaError(
            "Expected a configured paged-attention cache reference."
        )
    fields = dict(value_type.fields)
    cache = fields.get("kv_caches")
    if not isinstance(cache, TensorType) or not isinstance(cache.dtype, VectorType):
        raise IRSchemaError(
            "Expected a configured paged-attention cache reference with a "
            "vectorized kv_caches field."
        )
    if cache.dtype.elem_type != DType.BFLOAT16:
        raise IRSchemaError("Paged-attention KV cache element type must be bfloat16.")
    if len(cache.dtype.lanes) != 1:
        raise IRSchemaError(
            "Paged-attention state currently requires one cache vector axis."
        )
    if cache.rank != 6 or any(not dimension.is_fixed for dimension in cache.shape):
        raise IRSchemaError(
            "Paged-attention KV cache must have a static rank-six physical type."
        )
    shape = tuple(dimension.fixed_value for dimension in cache.shape)
    if shape[2] != 2:
        raise IRSchemaError("Paged-attention KV cache axis 2 must contain K and V.")
    lanes = prod(cache.dtype.lanes)
    config = PagedAttentionStateConfig(
        num_layers=shape[1],
        num_kv_heads=shape[4],
        head_dim=shape[5] * lanes,
        block_size=shape[3],
        num_blocks=shape[0],
        lanes=lanes,
    )
    expected_fields = dict(config.ref_type.fields)
    if fields != expected_fields:
        raise IRSchemaError(
            "Paged-attention cache reference fields do not match its encoded configuration."
        )
    return config


@dataclass
class PagedAttentionState:
    kv_caches: Any
    query_start_loc: Any
    seq_lens: Any
    slot_mapping: Any
    block_table: Any
    config: PagedAttentionStateConfig

    def clone(self) -> "PagedAttentionState":
        return PagedAttentionState(
            self.kv_caches.clone(),
            self.query_start_loc.clone(),
            self.seq_lens.clone(),
            self.slot_mapping.clone(),
            self.block_table.clone(),
            self.config,
        )

    @property
    def sequence_length(self) -> int:
        return int(self.seq_lens[0].item())

    def validate(self, *, dynamic_values: bool = True) -> None:
        torch = _torch()
        if tuple(self.kv_caches.shape) != self.config.storage_shape or self.kv_caches.dtype != torch.bfloat16:
            raise EvaluationError(
                f"Packed KV cache must be BF16{self.config.storage_shape}, got "
                f"{self.kv_caches.dtype}{tuple(self.kv_caches.shape)}.")
        contracts = (
            ("query_start_loc", self.query_start_loc, (2,), torch.int32),
            ("seq_lens", self.seq_lens, (1,), torch.int32),
            ("slot_mapping", self.slot_mapping, (1,), torch.int64),
            ("block_table", self.block_table, (1, self.config.num_blocks), torch.int32),
        )
        for name, value, shape, dtype in contracts:
            if tuple(value.shape) != shape or value.dtype != dtype:
                raise EvaluationError(f"Paged-attention {name} must be {dtype}{shape}.")
        if not dynamic_values:
            return
        if self.sequence_length < 0 or self.sequence_length > self.config.max_sequence_length:
            raise EvaluationError("Paged-attention sequence length is outside cache capacity.")
        self._validate_page_table(self.sequence_length)

    def _validate_page_table(self, length: int) -> None:
        torch = _torch()
        table = self.block_table.flatten()
        if bool(((table < 0) | (table >= self.config.num_blocks)).any()):
            raise EvaluationError("Paged-attention physical page is outside cache capacity.")
        active = (length + self.config.block_size - 1) // self.config.block_size
        if torch.unique(table[:active]).numel() != active:
            raise EvaluationError("Paged-attention active logical pages must not alias physical storage.")

    def append(
        self,
        key,
        value,
        *,
        layer_id: int,
        advance_sequence: bool = True,
    ) -> int:
        key = self._slot_chunk(key)
        value = self._slot_chunk(value)
        if key.shape != value.shape:
            raise EvaluationError("Paged-attention K/V chunks must have matching shapes.")
        self.update(key, cache_kind="key", layer_id=layer_id)
        return self.update(value, cache_kind="value", layer_id=layer_id, advance_sequence=advance_sequence)

    def update(
        self,
        slots,
        *,
        cache_kind: str,
        layer_id: int,
        advance_sequence: bool = False,
    ) -> int:
        """Update a K or V chunk while preserving reference identity.

        Accept legacy [head, dim] slots or nonempty [seq, head, dim] chunks.
        Validate the entire destination before writing any token.
        """

        self.validate()
        self._validate_layer(layer_id)
        if cache_kind not in {"key", "value"}:
            raise EvaluationError("Paged-attention cache kind must be key or value.")
        slots = self._slot_chunk(slots)
        tokens = slots.shape[0]
        position = self.sequence_length
        if position + tokens > self.config.max_sequence_length:
            raise EvaluationError("Paged-attention cache capacity is exhausted.")
        self._validate_page_table(position + tokens)
        packed_shape = (
            self.config.num_kv_heads,
            self.config.head_dim // self.config.lanes,
            self.config.lanes,
        )
        cache_index = 0 if cache_kind == "key" else 1
        for row in range(tokens):
            block, offset = divmod(position + row, self.config.block_size)
            physical_block = int(self.block_table[0, block].item())
            self.kv_caches[physical_block, layer_id, cache_index, offset].copy_(slots[row].reshape(packed_shape))
        self.slot_mapping[0] = position
        self.query_start_loc[0] = 0
        self.query_start_loc[1] = tokens
        if advance_sequence:
            self.seq_lens[0] = position + tokens
        return position

    def _slot_chunk(self, slots):
        expected = (self.config.num_kv_heads, self.config.head_dim)
        if tuple(slots.shape) == expected:
            return slots.unsqueeze(0)
        if slots.ndim != 3 or slots.shape[0] <= 0 or tuple(slots.shape[1:]) != expected:
            raise EvaluationError(
                f"Paged-attention slots must have shape {expected} or nonempty [seq, {expected[0]}, {expected[1]}].")
        return slots

    def gather(self, *, layer_id: int, length: int | None = None):
        self.validate()
        self._validate_layer(layer_id)
        if length is None:
            length = self.sequence_length
        if isinstance(length, bool) or not isinstance(length, int):
            raise EvaluationError("Paged-attention gather length must be an integer.")
        if length < 0 or length > self.config.max_sequence_length:
            raise EvaluationError("Paged-attention gather length is outside cache capacity.")
        self._validate_page_table(length)
        keys = []
        values = []
        for position in range(length):
            block, offset = divmod(position, self.config.block_size)
            physical_block = int(self.block_table[0, block].item())
            keys.append(self.kv_caches[physical_block, layer_id, 0, offset].reshape(
                self.config.num_kv_heads, self.config.head_dim))
            values.append(self.kv_caches[physical_block, layer_id, 1, offset].reshape(
                self.config.num_kv_heads, self.config.head_dim))
        torch = _torch()
        if not keys:
            empty = torch.empty(
                (0, self.config.num_kv_heads, self.config.head_dim),
                dtype=torch.bfloat16,
                device=self.kv_caches.device,
            )
            return empty, empty.clone()
        return torch.stack(keys), torch.stack(values)

    def _validate_layer(self, layer_id: int) -> None:
        if isinstance(layer_id, bool) or layer_id < 0 or layer_id >= self.config.num_layers:
            raise EvaluationError(f"Paged-attention layer id {layer_id!r} is out of range.")


def create_paged_attention_state(config: PagedAttentionStateConfig, *, device: str = "cpu") -> PagedAttentionState:
    torch = _torch()
    state = PagedAttentionState(
        torch.zeros(config.storage_shape, dtype=torch.bfloat16, device=device),
        torch.tensor([0, 1], dtype=torch.int32, device=device),
        torch.zeros((1,), dtype=torch.int32, device=device),
        torch.zeros((1,), dtype=torch.int64, device=device),
        torch.arange(config.num_blocks, dtype=torch.int32, device=device).reshape(1, -1),
        config,
    )
    state.validate()
    return state


def _torch():
    try:
        import torch
    except ImportError as error:
        raise EvaluationError("Paged-attention state requires PyTorch.") from error
    return torch


__all__ = [
    "PagedAttentionState",
    "PagedAttentionStateConfig",
    "create_paged_attention_state",
    "paged_attention_state_config_from_type",
]
