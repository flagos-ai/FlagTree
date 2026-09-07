# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Typed ownership contract for microkernel transfer pipelines."""

from __future__ import annotations

from dataclasses import dataclass

from triton.flagmega.errors import IRSchemaError
from triton.flagmega.ir.tir.auxiliary_consumer_contract import (
    TIRAuxiliaryConsumerContract,
)
from triton.flagmega.ir.tir.base import TIRNode, tir_node
from triton.flagmega.ir.tir.transfer_pipeline_channel import (
    TIRTransferPipelineChannel,
)


@tir_node("transfer_pipeline_contract")
@dataclass(frozen=True)
class TIRTransferPipelineContract(TIRNode):
    channels: tuple[TIRTransferPipelineChannel, ...]
    consumer_shared_workspace_indices: tuple[int, ...] = ()
    auxiliary_consumer: TIRAuxiliaryConsumerContract | None = None
    capacity: int | None = None

    def __post_init__(self) -> None:
        channels = tuple(self.channels)
        if not channels or any(
            not isinstance(value, TIRTransferPipelineChannel) for value in channels
        ):
            raise IRSchemaError(
                "A transfer pipeline must contain at least one typed channel."
            )
        names = tuple(value.name for value in channels)
        if len(set(names)) != len(names):
            duplicate = next(name for name in names if names.count(name) > 1)
            raise IRSchemaError(
                f"Transfer pipeline contains duplicate channel {duplicate}."
            )
        channel_workspaces = tuple(
            index
            for channel in channels
            for index in channel.shared_workspace_indices
        )
        if len(set(channel_workspaces)) != len(channel_workspaces):
            duplicate = next(
                index for index in channel_workspaces
                if channel_workspaces.count(index) > 1
            )
            raise IRSchemaError(
                f"Shared workspace {duplicate} is owned by multiple transfer channels."
            )
        consumer = tuple(self.consumer_shared_workspace_indices)
        if (
            any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in consumer)
            or len(set(consumer)) != len(consumer)
        ):
            raise IRSchemaError(
                "Consumer Shared workspace indexes must be non-negative and unique."
            )
        conflicts = set(channel_workspaces) & set(consumer)
        if conflicts:
            index = min(conflicts)
            raise IRSchemaError(
                f"Shared workspace {index} is owned by both a transfer channel and the consumer."
            )
        auxiliary = self.auxiliary_consumer
        if auxiliary is not None:
            if not isinstance(auxiliary, TIRAuxiliaryConsumerContract):
                raise IRSchemaError(
                    "Transfer pipeline auxiliary consumer must be a typed contract."
                )
            invalid_channel = next(
                (index for index in auxiliary.channel_indices if index >= len(channels)),
                None,
            )
            if invalid_channel is not None:
                raise IRSchemaError(
                    f"Auxiliary consumer channel index {invalid_channel} is outside "
                    f"the transfer channel range [0, {len(channels)})."
                )
            invalid_workspace = next(
                (
                    index
                    for index in auxiliary.consumer_shared_workspace_indices
                    if index not in consumer
                ),
                None,
            )
            if invalid_workspace is not None:
                raise IRSchemaError(
                    f"Auxiliary consumer Shared workspace {invalid_workspace} is not "
                    "owned by the transfer pipeline consumer."
                )
        capacity = self.capacity
        if capacity is not None and (
            isinstance(capacity, bool)
            or not isinstance(capacity, int)
            or capacity <= 0
        ):
            raise IRSchemaError(
                "Transfer pipeline capacity must be a positive integer or None."
            )
        object.__setattr__(self, "channels", channels)
        object.__setattr__(self, "consumer_shared_workspace_indices", consumer)

    @property
    def source_argument_indices(self) -> tuple[int, ...]:
        return tuple(dict.fromkeys(
            index for channel in self.channels
            for index in channel.source_argument_indices
        ))

    @property
    def shared_workspace_indices(self) -> tuple[int, ...]:
        return tuple(
            index for channel in self.channels
            for index in channel.shared_workspace_indices
        )


__all__ = ["TIRTransferPipelineContract"]
