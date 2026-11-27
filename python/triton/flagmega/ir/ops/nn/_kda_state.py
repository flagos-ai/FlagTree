# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Persistent state object for decode-time KDA layers.

KDA state is deliberately kept as a first-class object, like GDN state and
paged attention state.  The compiler ABI may expose one pair of tensors per
layer, but callers should not have to manage those individual buffers.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from triton.flagmega.errors import EvaluationError, IRSchemaError
from triton.flagmega.ir.model import DistributedType, TensorType
from triton.flagmega.ir.types import DType, VectorType


@dataclass(frozen=True)
class KdaStateConfig:
    """Physical tensor contract for a collection of homogeneous KDA layers."""

    layer_ids: tuple[object, ...]
    convolution_type: TensorType
    recurrent_type: TensorType
    # Physical tensors alone cannot distinguish two equal-sized head shards.
    # Keep their logical ownership and, after device loading, their chip.
    logical_types: tuple[TensorType | DistributedType, TensorType | DistributedType] | None = None
    chip_index: int | None = None

    def __post_init__(self) -> None:
        if not self.layer_ids or len(set(self.layer_ids)) != len(self.layer_ids):
            raise IRSchemaError("KDA state requires a non-empty set of unique layer ids.")
        for name, value in (("convolution", self.convolution_type), ("recurrent", self.recurrent_type)):
            if not isinstance(value, TensorType) or isinstance(value.dtype, VectorType):
                raise IRSchemaError(f"KDA {name} state must be an ordinary tensor type.")
            if any(not dimension.is_fixed or dimension.fixed_value <= 0 for dimension in value.shape):
                raise IRSchemaError(f"KDA {name} state must have a fixed positive shape.")
        if self.logical_types is not None:
            if not isinstance(self.logical_types, tuple) or len(self.logical_types) != 2:
                raise IRSchemaError("KDA state logical_types must contain convolution and recurrent tensor types.")
            for logical, physical in zip(self.logical_types, (self.convolution_type, self.recurrent_type), strict=True):
                tensor = logical.tensor if isinstance(logical, DistributedType) else logical
                if not isinstance(tensor, TensorType) or tensor.dtype != physical.dtype:
                    raise IRSchemaError("KDA logical and physical state dtypes must match.")
                if isinstance(logical, DistributedType) and (logical.partial is not None or logical.exclusive is not None):
                    raise IRSchemaError("KDA persistent state must be materialized and non-exclusive.")
        if self.chip_index is not None:
            if self.logical_types is None or type(self.chip_index) is not int or self.chip_index < 0:
                raise IRSchemaError("KDA state chip_index requires explicit ownership and a non-negative integer.")
            placements = {value.placement for value in self.logical_types if isinstance(value, DistributedType)}
            if len(placements) != 1:
                raise IRSchemaError("KDA chip ownership requires one shared placement.")
            placement, = placements
            chips = placement.physical_level_size("c") * placement.physical_level_size("d")
            if self.chip_index >= chips:
                raise IRSchemaError("KDA state chip_index is outside its placement.")

    @property
    def convolution_shape(self) -> tuple[int, ...]:
        return tuple(dimension.fixed_value for dimension in self.convolution_type.shape)

    @property
    def recurrent_shape(self) -> tuple[int, ...]:
        return tuple(dimension.fixed_value for dimension in self.recurrent_type.shape)


@dataclass
class KdaState:
    """Stacked backing storage with layer-id based accessors.

    ``convolution`` and ``recurrent`` have a leading state-layer axis.  The
    layer ids need not be contiguous model layer numbers; this is important
    for models that interleave KDA and MLA blocks.
    """

    convolution: Any
    recurrent: Any
    config: KdaStateConfig

    def clone(self) -> KdaState:
        return KdaState(self.convolution.clone(), self.recurrent.clone(), self.config)

    def _index(self, layer_id: object) -> int:
        try:
            return self.config.layer_ids.index(layer_id)
        except ValueError as error:
            raise EvaluationError(f"KDA layer id {layer_id!r} is not in the state contract.") from error

    def convolution_layer(self, layer_id: object):
        self.validate()
        return self.convolution[self._index(layer_id)]

    def recurrent_layer(self, layer_id: object):
        self.validate()
        return self.recurrent[self._index(layer_id)]

    def update_convolution_layer(self, value, layer_id: object) -> None:
        self.validate()
        expected = self.config.convolution_shape
        if tuple(value.shape) != expected:
            raise EvaluationError(f"KDA convolution state must have shape {expected}.")
        self.convolution[self._index(layer_id)].copy_(value)

    def update_recurrent_layer(self, value, layer_id: object) -> None:
        self.validate()
        expected = self.config.recurrent_shape
        if tuple(value.shape) != expected:
            raise EvaluationError(f"KDA recurrent state must have shape {expected}.")
        self.recurrent[self._index(layer_id)].copy_(value)

    def validate(self) -> None:
        torch = _torch()
        expected_convolution = (len(self.config.layer_ids), *self.config.convolution_shape)
        expected_recurrent = (len(self.config.layer_ids), *self.config.recurrent_shape)
        if tuple(self.convolution.shape) != expected_convolution:
            raise EvaluationError(
                f"KDA convolution state must have shape {expected_convolution}, got {tuple(self.convolution.shape)}.")
        if tuple(self.recurrent.shape) != expected_recurrent:
            raise EvaluationError(
                f"KDA recurrent state must have shape {expected_recurrent}, got {tuple(self.recurrent.shape)}.")
        expected_dtypes = (
            ("convolution", self.convolution, _torch_dtype(torch, self.config.convolution_type.dtype)),
            ("recurrent", self.recurrent, _torch_dtype(torch, self.config.recurrent_type.dtype)),
        )
        for name, value, dtype in expected_dtypes:
            if value.dtype != dtype:
                raise EvaluationError(f"KDA {name} state must use dtype {dtype}, got {value.dtype}.")


def create_kda_state(
    config: KdaStateConfig,
    *,
    device: str = "cpu",
) -> KdaState:
    torch = _torch()
    return KdaState(
        torch.zeros(
            (len(config.layer_ids), *config.convolution_shape),
            dtype=_torch_dtype(torch, config.convolution_type.dtype),
            device=device,
        ),
        torch.zeros(
            (len(config.layer_ids), *config.recurrent_shape),
            dtype=_torch_dtype(torch, config.recurrent_type.dtype),
            device=device,
        ),
        config,
    )


def _torch_dtype(torch, dtype: DType):
    return {
        DType.BFLOAT16: torch.bfloat16,
        DType.FLOAT16: torch.float16,
        DType.FLOAT32: torch.float32,
    }.get(dtype) or (_raise_dtype(dtype))


def _raise_dtype(dtype):
    raise EvaluationError(f"KDA state dtype {dtype!r} is not supported by the runtime state object.")


def _torch():
    try:
        import torch
    except ImportError as error:
        raise EvaluationError("KDA state creation requires PyTorch.") from error
    return torch


__all__ = ["KdaState", "KdaStateConfig", "create_kda_state"]
