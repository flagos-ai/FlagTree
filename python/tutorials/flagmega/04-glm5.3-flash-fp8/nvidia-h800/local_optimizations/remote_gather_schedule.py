# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Choose chip-shared peer-gather tasks before temporary storage planning."""

from dataclasses import replace

SUFFIX = ".cooperative_peer_gather"


def implementation_model(model):
    additions = tuple(
        replace(implementation, id=implementation.id + SUFFIX,
                contract={**implementation.contract, "peer_gather": True},
                facts={**implementation.facts, "result_memory_scope": "chip",
                       "require_cooperative_remote_gather": True})
        for implementation in model.implementations
        if implementation.family == "distributed_boxing"
        and implementation.contract["transition"] in {"gather_reduce_scatter", "tensor_store"}
    )
    return replace(model, implementations=(*model.implementations, *additions),
                   preferences={**model.preferences, "distributed_boxing": (
                       *(implementation.id for implementation in additions),
                       *model.preferences["distributed_boxing"])})
