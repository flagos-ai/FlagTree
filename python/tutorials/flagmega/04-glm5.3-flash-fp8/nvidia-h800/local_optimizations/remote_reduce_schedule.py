# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Choose chip-shared peer-sum output tiles before storage and hazard planning."""

from dataclasses import replace

SUFFIX = ".cooperative_peer_reduce"


def implementation_model(model):
    additions = tuple(
        replace(implementation, id=implementation.id + SUFFIX,
                parameters={**implementation.parameters, "tile": 128},
                contract={**implementation.contract, "peer_partial_reduce": True},
                facts={**implementation.facts, "result_memory_scope": "chip",
                       "require_cooperative_remote_reduce": True})
        for implementation in model.implementations
        if implementation.family == "distributed_boxing"
        and implementation.contract["transition"] == "gather_reduce_scatter"
        and not implementation.contract.get("peer_gather")
        and not implementation.contract.get("peer_partial_reduce")
    )
    return replace(model, implementations=(*model.implementations, *additions),
                   preferences={**model.preferences, "distributed_boxing": (
                       *(implementation.id for implementation in additions),
                       *model.preferences["distributed_boxing"])})
