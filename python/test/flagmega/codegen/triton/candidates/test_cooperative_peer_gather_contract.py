# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Peer tile assignment is an explicit implementation/storage alternative."""

from dataclasses import replace

import pytest
from triton.flagmega import ir as fm
from triton.flagmega.targets import NvidiaSm90Target

from python.test.flagmega.codegen.triton.candidates.test_distributed_boxing_provider import _boxing_module


def cooperative_gather_model(model):
    additions = tuple(replace(implementation,
        id=implementation.id + ".cooperative_peer_gather",
        contract={**implementation.contract, "peer_gather": True},
        facts={**implementation.facts, "result_memory_scope": "chip",
               "require_cooperative_remote_gather": True})
        for implementation in model.implementations
        if implementation.family == "distributed_boxing"
        and implementation.contract["transition"] in {"gather_reduce_scatter", "tensor_store"})
    return replace(model, implementations=(*model.implementations, *additions),
                   preferences={**model.preferences, "distributed_boxing": (
                       *(implementation.id for implementation in additions),
                       *model.preferences["distributed_boxing"])})


@pytest.mark.parametrize("case", ("peer", "local", "reduction", "split_target", "tiny", "plain", "packed"))
def test_peer_gather_implementation_obeys_its_semantic_contract(case):
    mesh = fm.Placement((4, 8, 16), "cyx", "cbb")
    kind = fm.vector_type("bfloat16", (8,)) if case == "packed" else "bfloat16"
    width = 1024 if case == "packed" else 512 if case == "tiny" else 8192
    tensor = fm.tensor_type(kind, (1, width))
    broadcast = fm.DistributedType(tensor, (fm.SBP.broadcast(), fm.SBP.broadcast()), mesh)
    source = replace(broadcast, axis_policies=(fm.SBP.broadcast(), fm.SBP.split_contiguous(
        (2,) if case == "local" else (0, 2))))
    if case == "reduction":
        source = replace(broadcast, partial=fm.SBP.partial((0,)))
    target_type = (replace(broadcast, axis_policies=(fm.SBP.broadcast(), fm.SBP.split_contiguous((1,))))
                   if case == "split_target" else tensor if case == "plain" else broadcast)
    target = NvidiaSm90Target()
    target = NvidiaSm90Target(triton_implementation_model=cooperative_gather_model(target.triton_implementation_model))
    proposed = target.propose_tir(_boxing_module(source, target_type))
    point = next(point for point in proposed.selection_points if point.id == "tir.output")
    wanted = case in {"peer", "plain", "packed"}
    assert any(candidate.id.endswith(".cooperative_peer_gather") for candidate in point.candidates) == wanted
    assert point.default_candidate.endswith(".cooperative_peer_gather") == wanted
    if wanted:
        selected = next(candidate for candidate in point.candidates if candidate.id == point.default_candidate)
        assert selected.facts["source_memory_scope"] == "device"
        assert selected.facts["result_memory_scope"] == "chip"
