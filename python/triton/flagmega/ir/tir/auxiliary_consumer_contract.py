# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Fixed auxiliary consumer partition of a transfer-pipelined helper."""

from __future__ import annotations

from dataclasses import dataclass

from triton.flagmega.errors import IRSchemaError
from triton.flagmega.ir.tir.base import TIRNode, tir_node


@tir_node("auxiliary_consumer_contract")
@dataclass(frozen=True)
class TIRAuxiliaryConsumerContract(TIRNode):
    channel_indices: tuple[int, ...]
    consumer_shared_workspace_indices: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "channel_indices",
            _indices(self.channel_indices, allow_empty=False, owner="channel"),
        )
        object.__setattr__(
            self,
            "consumer_shared_workspace_indices",
            _indices(
                self.consumer_shared_workspace_indices,
                allow_empty=True,
                owner="consumer Shared workspace",
            ),
        )


def _indices(values, *, allow_empty: bool, owner: str) -> tuple[int, ...]:
    result = tuple(values)
    if (
        (not result and not allow_empty)
        or any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in result)
        or len(set(result)) != len(result)
    ):
        qualifier = "" if allow_empty else "non-empty, "
        raise IRSchemaError(
            f"Auxiliary consumer {owner} indexes must be {qualifier}non-negative and unique."
        )
    return result


__all__ = ["TIRAuxiliaryConsumerContract"]
