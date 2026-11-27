# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Peer reductions have a separate legality contract from remote copies."""

from dataclasses import replace

import pytest
from triton.flagmega import ir as fm
from triton.flagmega.targets import NvidiaSm90Target

from python.test.flagmega.codegen.triton.candidates.test_distributed_boxing_provider import _boxing_module

SUFFIX = ".cooperative_peer_reduce"


def cooperative_reduce_model(model):
    original = model.implementation("tir.distributed_boxing.gather_reduce_scatter")
    alternative = replace(original, id=original.id + SUFFIX,
                          parameters={**original.parameters, "tile": 128},
                          contract={**original.contract, "peer_partial_reduce": True},
                          facts={**original.facts, "result_memory_scope": "chip",
                                 "require_cooperative_remote_reduce": True})
    return replace(model, implementations=(*model.implementations, alternative),
                   preferences={**model.preferences, "distributed_boxing": (
                       alternative.id, *model.preferences["distributed_boxing"])})


@pytest.mark.parametrize("case", ["peer", "plain", "packed", "packed_small", "copy", "local_sum", "partial_block",
                                 "split_target", "retained_partial", "tiny", "tuple_identity", "mixed_copy"])
def test_peer_reduction_choice_requires_chip_partial_and_broadcast_output_tiles(case):
    mesh = fm.Placement((4, 8, 16), "cyx", "cbb")
    lanes = 8 if case in {"packed", "packed_small"} else 1
    width = 64 if case == "tiny" else 256 if case == "packed_small" else 4096
    tensor = fm.tensor_type("float32" if lanes == 1 else fm.vector_type("float32", (lanes,)), (1, width // lanes))
    broad = fm.DistributedType(tensor, (fm.SBP.broadcast(),) * 2, mesh)
    axes = (1,) if case in {"tiny", "packed_small"} else (2,) if case in {"local_sum", "partial_block"} else (1, 2)
    owner_count = 1
    for axis in axes:
        owner_count *= mesh.hierarchy[axis]
    source = replace(broad, axis_policies=(fm.SBP.broadcast(), fm.SBP.split_contiguous(axes, width // (lanes * owner_count))),
                     partial=fm.SBP.partial((1,) if case == "local_sum" else (0, 1) if case == "partial_block" else (0,)))
    target = broad
    if case == "copy":
        source = replace(broad, axis_policies=(fm.SBP.broadcast(), fm.SBP.split_contiguous((0, 2), 64)))
    elif case == "split_target":
        target = replace(broad, axis_policies=(fm.SBP.broadcast(), fm.SBP.split_contiguous((1,), 512)))
    elif case == "retained_partial":
        target = replace(broad, partial=source.partial)
    elif case == "plain":
        target = tensor
    elif case in {"tuple_identity", "mixed_copy"}:
        other = broad if case == "tuple_identity" else replace(
            broad, axis_policies=(fm.SBP.broadcast(), fm.SBP.split_contiguous((0, 2), 64)))
        source, target = fm.TupleType((source, other)), fm.TupleType((broad, broad))
    base = NvidiaSm90Target()
    target_model = NvidiaSm90Target(triton_implementation_model=cooperative_reduce_model(base.triton_implementation_model))
    proposed = target_model.propose_tir(_boxing_module(source, target))
    point = next(point for point in proposed.selection_points if point.id == "tir.output")
    wanted = case in {"peer", "plain", "packed", "packed_small", "tuple_identity"}
    assert any(candidate.id.endswith(SUFFIX) for candidate in point.candidates) == wanted
    assert point.default_candidate.endswith(SUFFIX) == wanted
    if wanted:
        candidate = next(candidate for candidate in point.candidates if candidate.id == point.default_candidate)
        assert candidate.facts["source_memory_scope"] == "device"
        assert candidate.facts["result_memory_scope"] == "chip"
