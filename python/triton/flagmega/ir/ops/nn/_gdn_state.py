# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""nncase-compatible physical ABI for persistent Gated DeltaNet state."""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum
from typing import Any

from triton.flagmega.errors import EvaluationError, IRSchemaError
from triton.flagmega.ir.model import RefType, TensorType, tensor_type
from triton.flagmega.ir.ops.tensors.pack import pack_physical
from triton.flagmega.ir.ops.tensors.unpack import unpack_physical
from triton.flagmega.ir.types import DType, VectorType


class GatedDeltaNetStateKind(str, Enum):
    CONVOLUTION = "convolution"
    RECURRENT = "recurrent"


class GatedDeltaNetStateDimKind(str, Enum):
    NUM_LAYERS = "num_layers"
    CONV_CHANNELS = "conv_channels"
    CONV_HISTORY = "conv_history"
    NUM_VALUE_HEADS = "num_value_heads"
    KEY_HEAD_DIM = "key_head_dim"
    VALUE_HEAD_DIM = "value_head_dim"


@dataclass(frozen=True)
class GatedDeltaNetStateConfig:
    """Logical layout and trailing-lane storage contract copied from nncase."""

    num_layers: int
    num_key_heads: int
    num_value_heads: int
    key_head_dim: int
    value_head_dim: int
    conv_kernel_size: int
    hidden_size: int
    activation_dtype: DType = DType.BFLOAT16
    activation_lanes: tuple[int, ...] = (8,)
    convolution_layout: tuple[GatedDeltaNetStateDimKind, ...] = (
        GatedDeltaNetStateDimKind.NUM_LAYERS,
        GatedDeltaNetStateDimKind.CONV_CHANNELS,
        GatedDeltaNetStateDimKind.CONV_HISTORY,
    )
    convolution_vectorized_axes: tuple[GatedDeltaNetStateDimKind, ...] = (
        GatedDeltaNetStateDimKind.CONV_CHANNELS,
    )
    convolution_lanes: tuple[int, ...] = (8,)
    recurrent_layout: tuple[GatedDeltaNetStateDimKind, ...] = (
        GatedDeltaNetStateDimKind.NUM_LAYERS,
        GatedDeltaNetStateDimKind.NUM_VALUE_HEADS,
        GatedDeltaNetStateDimKind.VALUE_HEAD_DIM,
        GatedDeltaNetStateDimKind.KEY_HEAD_DIM,
    )
    recurrent_vectorized_axes: tuple[GatedDeltaNetStateDimKind, ...] = (
        GatedDeltaNetStateDimKind.KEY_HEAD_DIM,
    )
    recurrent_lanes: tuple[int, ...] = (4,)

    def __post_init__(self) -> None:
        integer_fields = (
            self.num_layers,
            self.num_key_heads,
            self.num_value_heads,
            self.key_head_dim,
            self.value_head_dim,
            self.conv_kernel_size,
            self.hidden_size,
        )
        if any(isinstance(value, bool) or not isinstance(value, int) or value <= 0 for value in integer_fields):
            raise IRSchemaError("GDN state dimensions must be positive integers.")
        if self.conv_kernel_size < 2:
            raise IRSchemaError("GDN convolution kernel size must be at least two.")
        if self.num_value_heads % self.num_key_heads:
            raise IRSchemaError("GDN value heads must be divisible by key heads.")
        if self.activation_dtype != DType.BFLOAT16:
            raise IRSchemaError("Qwen3.8 P0 GDN state currently requires BF16 activations.")
        self._validate_layout(
            GatedDeltaNetStateKind.CONVOLUTION,
            self.convolution_layout,
            (
                GatedDeltaNetStateDimKind.NUM_LAYERS,
                GatedDeltaNetStateDimKind.CONV_CHANNELS,
                GatedDeltaNetStateDimKind.CONV_HISTORY,
            ),
            self.convolution_vectorized_axes,
            self.convolution_lanes,
        )
        self._validate_layout(
            GatedDeltaNetStateKind.RECURRENT,
            self.recurrent_layout,
            (
                GatedDeltaNetStateDimKind.NUM_LAYERS,
                GatedDeltaNetStateDimKind.NUM_VALUE_HEADS,
                GatedDeltaNetStateDimKind.KEY_HEAD_DIM,
                GatedDeltaNetStateDimKind.VALUE_HEAD_DIM,
            ),
            self.recurrent_vectorized_axes,
            self.recurrent_lanes,
        )
        if not self.activation_lanes or any(lane <= 1 for lane in self.activation_lanes):
            raise IRSchemaError("GDN activation lanes must be greater than one.")
        if self.hidden_size % _product(self.activation_lanes):
            raise IRSchemaError("GDN hidden size is not divisible by activation lanes.")

    @property
    def conv_dim(self) -> int:
        return (2 * self.num_key_heads * self.key_head_dim) + self.num_value_heads * self.value_head_dim

    def dimension(self, axis: GatedDeltaNetStateDimKind) -> int:
        return {
            GatedDeltaNetStateDimKind.NUM_LAYERS: self.num_layers,
            GatedDeltaNetStateDimKind.CONV_CHANNELS: self.conv_dim,
            GatedDeltaNetStateDimKind.CONV_HISTORY: self.conv_kernel_size - 1,
            GatedDeltaNetStateDimKind.NUM_VALUE_HEADS: self.num_value_heads,
            GatedDeltaNetStateDimKind.KEY_HEAD_DIM: self.key_head_dim,
            GatedDeltaNetStateDimKind.VALUE_HEAD_DIM: self.value_head_dim,
        }[axis]

    def logical_tensor_type(self, kind: GatedDeltaNetStateKind) -> TensorType:
        layout, vectorized_axes, lanes = self._parts(kind)
        shape = [self.dimension(axis) for axis in layout]
        for axis, lane in zip(vectorized_axes, lanes):
            shape[layout.index(axis)] //= lane
        dtype = self.activation_dtype if kind == GatedDeltaNetStateKind.CONVOLUTION else DType.FLOAT32
        return tensor_type(VectorType(dtype, lanes), shape)

    def storage_shape(self, kind: GatedDeltaNetStateKind) -> tuple[int, ...]:
        value_type = self.logical_tensor_type(kind)
        assert isinstance(value_type.dtype, VectorType)
        return (
            *(dimension.fixed_value for dimension in value_type.shape),
            *value_type.dtype.lanes,
        )

    @property
    def ref_type(self) -> RefType:
        return RefType(
            "qwen3_5_gated_delta_net_state",
            (
                ("convolution", self.logical_tensor_type(GatedDeltaNetStateKind.CONVOLUTION)),
                ("recurrent", self.logical_tensor_type(GatedDeltaNetStateKind.RECURRENT)),
            ),
        )

    def _parts(self, kind: GatedDeltaNetStateKind):
        if kind == GatedDeltaNetStateKind.CONVOLUTION:
            return self.convolution_layout, self.convolution_vectorized_axes, self.convolution_lanes
        if kind == GatedDeltaNetStateKind.RECURRENT:
            return self.recurrent_layout, self.recurrent_vectorized_axes, self.recurrent_lanes
        raise ValueError(f"Unknown GDN state kind {kind!r}.")

    def _validate_layout(self, kind, layout, required, vectorized_axes, lanes) -> None:
        if len(layout) != len(required) or set(layout) != set(required):
            raise IRSchemaError(f"GDN {kind.value} layout must contain each semantic axis exactly once.")
        if len(vectorized_axes) != len(lanes) or len(set(vectorized_axes)) != len(vectorized_axes):
            raise IRSchemaError(f"GDN {kind.value} vectorized axes and lanes are inconsistent.")
        if any(axis not in layout for axis in vectorized_axes):
            raise IRSchemaError(f"GDN {kind.value} vectorized axis is not present in its layout.")
        if not lanes or any(isinstance(lane, bool) or lane <= 1 for lane in lanes):
            raise IRSchemaError(f"GDN {kind.value} packed lanes must be greater than one.")
        for axis, lane in zip(vectorized_axes, lanes):
            if self.dimension(axis) % lane:
                raise IRSchemaError(
                    f"GDN {kind.value} axis {axis.value} extent {self.dimension(axis)} is not divisible by {lane}.")


@dataclass
class GatedDeltaNetState:
    """Physical torch backing; vector lanes are explicit trailing dimensions."""

    convolution: Any
    recurrent: Any
    config: GatedDeltaNetStateConfig

    def clone(self) -> GatedDeltaNetState:
        return GatedDeltaNetState(self.convolution.clone(), self.recurrent.clone(), self.config)

    def convolution_layer(self, layer_id: int = 0):
        self._validate_layer(layer_id)
        return unpack_physical(
            self.convolution[layer_id],
            outer_rank=2,
            lanes=self.config.convolution_lanes,
            axes=(0,),
        )

    def update_convolution_layer(self, value, layer_id: int = 0) -> None:
        self._validate_layer(layer_id)
        expected = (self.config.conv_dim, self.config.conv_kernel_size - 1)
        if tuple(value.shape) != expected:
            raise EvaluationError(f"Logical GDN convolution layer must have shape {expected}.")
        packed = pack_physical(
            value,
            outer_rank=2,
            lanes=self.config.convolution_lanes,
            axes=(0,),
        )
        self.convolution[layer_id].copy_(packed)

    def recurrent_layer(self, layer_id: int = 0):
        self._validate_layer(layer_id)
        value_key = unpack_physical(
            self.recurrent[layer_id],
            outer_rank=3,
            lanes=self.config.recurrent_lanes,
            axes=(2,),
        )
        return value_key.permute(0, 2, 1).contiguous()

    def update_recurrent_layer(self, value, layer_id: int = 0) -> None:
        self._validate_layer(layer_id)
        expected = (self.config.num_value_heads, self.config.key_head_dim, self.config.value_head_dim)
        if tuple(value.shape) != expected:
            raise EvaluationError(f"Logical GDN recurrent layer must have shape {expected}.")
        value_key = value.permute(0, 2, 1).contiguous()
        packed = pack_physical(
            value_key,
            outer_rank=3,
            lanes=self.config.recurrent_lanes,
            axes=(2,),
        )
        self.recurrent[layer_id].copy_(packed)

    def validate(self) -> None:
        torch = _torch()
        expected_convolution = self.config.storage_shape(GatedDeltaNetStateKind.CONVOLUTION)
        expected_recurrent = self.config.storage_shape(GatedDeltaNetStateKind.RECURRENT)
        if tuple(self.convolution.shape) != expected_convolution or self.convolution.dtype != torch.bfloat16:
            raise EvaluationError(
                f"Packed GDN convolution state must be BF16{expected_convolution}, got "
                f"{self.convolution.dtype}{tuple(self.convolution.shape)}.")
        if tuple(self.recurrent.shape) != expected_recurrent or self.recurrent.dtype != torch.float32:
            raise EvaluationError(
                f"Packed GDN recurrent state must be F32{expected_recurrent}, got "
                f"{self.recurrent.dtype}{tuple(self.recurrent.shape)}.")

    def _validate_layer(self, layer_id: int) -> None:
        self.validate()
        if isinstance(layer_id, bool) or layer_id < 0 or layer_id >= self.config.num_layers:
            raise EvaluationError(f"GDN layer id {layer_id!r} is outside [0, {self.config.num_layers}).")

    def __flagmega_ref_slice__(self, index: int, length: int = 1):
        self._validate_layer(index)
        if length <= 0 or index + length > self.config.num_layers:
            raise EvaluationError("GDN reference slice is outside the layer extent.")
        if any(layout[0] != GatedDeltaNetStateDimKind.NUM_LAYERS
               for layout in (self.config.convolution_layout, self.config.recurrent_layout)):
            raise EvaluationError("GDN reference slicing requires a leading layer axis.")
        return GatedDeltaNetState(self.convolution[index:index + length], self.recurrent[index:index + length],
                                  replace(self.config, num_layers=length))


def gdn_state_config(config) -> GatedDeltaNetStateConfig:
    return GatedDeltaNetStateConfig(
        num_layers=int(config.num_hidden_layers),
        num_key_heads=int(config.num_key_heads),
        num_value_heads=int(config.num_value_heads),
        key_head_dim=int(config.key_head_dim),
        value_head_dim=int(config.value_head_dim),
        conv_kernel_size=int(config.conv_kernel_size),
        hidden_size=int(config.hidden_size),
    )


def create_gdn_state(config, *, device: str = "cpu", activation_dtype=None) -> GatedDeltaNetState:
    torch = _torch()
    state_config = config if isinstance(config, GatedDeltaNetStateConfig) else gdn_state_config(config)
    activation_dtype = activation_dtype or torch.bfloat16
    if activation_dtype != torch.bfloat16:
        raise EvaluationError("Qwen3.8 packed GDN state requires torch.bfloat16 activation storage.")
    convolution = torch.zeros(
        state_config.storage_shape(GatedDeltaNetStateKind.CONVOLUTION),
        dtype=activation_dtype,
        device=device,
    )
    recurrent = torch.zeros(
        state_config.storage_shape(GatedDeltaNetStateKind.RECURRENT),
        dtype=torch.float32,
        device=device,
    )
    return GatedDeltaNetState(convolution, recurrent, state_config)


def _product(values) -> int:
    result = 1
    for value in values:
        result *= value
    return result


def _torch():
    try:
        import torch
    except ImportError as error:
        raise EvaluationError("Gated DeltaNet state creation requires PyTorch.") from error
    return torch


__all__ = [
    "GatedDeltaNetState",
    "GatedDeltaNetStateConfig",
    "GatedDeltaNetStateDimKind",
    "GatedDeltaNetStateKind",
    "create_gdn_state",
    "gdn_state_config",
]
