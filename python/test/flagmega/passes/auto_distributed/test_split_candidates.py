# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT

from triton.flagmega import ir as fm
from triton.flagmega.passes.auto_distributed import DistributedCandidateContext
from triton.flagmega.targets.pyntt_split import PyNttDistributedSplitCandidateProvider

from .helpers import packed_matmul_module


def _context(*, use_pyntt: bool) -> DistributedCandidateContext:
    module = packed_matmul_module(size=128)
    node = module.node_map["output"]
    arguments = (
        module,
        node,
        fm.Placement((8, 16), "yx", "bb"),
        tuple((module.node_map[input_id].type,) for input_id in node.inputs),
    )
    if use_pyntt:
        return DistributedCandidateContext(
            *arguments,
            PyNttDistributedSplitCandidateProvider(block_bytes=128),
        )
    return DistributedCandidateContext(*arguments)


def test_pyntt_split_provider_retains_ragged_block_cyclic_full_mesh_policy():
    context = _context(use_pyntt=True)
    tensor = fm.tensor_type("bfloat16", (1, 160))

    candidates = context.split_candidates(
        tensor,
        1,
        (0, 1),
        purpose="output",
    )

    assert candidates == (fm.SBP.split_block_cyclic((0, 1), 1),)


def test_portable_split_provider_rejects_non_divisible_contiguous_policy():
    context = _context(use_pyntt=False)
    tensor = fm.tensor_type("bfloat16", (1, 160))

    assert context.split_candidates(tensor, 1, (0, 1)) == ()
