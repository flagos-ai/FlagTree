# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT

import pytest

from triton.flagmega import ir as fm
from triton.flagmega.targets import NvidiaSm90Target


def _boxing_module(source_type, target_type):
    builder = fm.IRBuilder(dialect="high_level", stage="frozen_constants")
    source = builder.var("source", source_type, id="source")
    output = builder.call(
        "distributed.boxing",
        (source,),
        target_type,
        id="output",
        attrs={"new_type": target_type},
    )
    builder.function("main", (source,), (output,))
    return fm.verify_module(builder.build(entry="main"))


@pytest.mark.parametrize("kind", ("contiguous", "block_cyclic", "partial"))
def test_gather_reduce_scatter_candidate_covers_general_distributed_reshard(kind):
    placement = fm.Placement((2, 4), "yx", "bb")
    tensor = fm.tensor_type("bfloat16", (1, 1024))
    broadcast = fm.DistributedType(
        tensor, (fm.SBP.broadcast(), fm.SBP.broadcast()), placement
    )
    if kind == "contiguous":
        source = fm.DistributedType(
            tensor,
            (fm.SBP.broadcast(), fm.SBP.split_contiguous((0, 1))),
            placement,
        )
    elif kind == "block_cyclic":
        source = fm.DistributedType(
            tensor,
            (fm.SBP.broadcast(), fm.SBP.split_block_cyclic((0, 1), 64)),
            placement,
        )
    else:
        source = fm.DistributedType(
            tensor,
            (fm.SBP.broadcast(), fm.SBP.broadcast()),
            placement,
            partial=fm.SBP.partial((0, 1)),
        )

    proposed = NvidiaSm90Target().propose_tir(_boxing_module(source, broadcast))
    point = next(value for value in proposed.selection_points if value.id == "tir.output")
    candidate = point.candidates[0]

    assert tuple(value.id for value in point.candidates) == (
        "tir.distributed_boxing.gather_reduce_scatter",
    )
    assert candidate.parameters["source_type"] == source
    assert candidate.parameters["target_type"] == broadcast
    assert candidate.facts["collective_semantics"] == "gather-reduce-scatter"


@pytest.mark.parametrize(
    ("source_distributed", "candidate_id", "semantics"),
    (
        (False, "tir.distributed_boxing.tensor_load", "tensor-load"),
        (True, "tir.distributed_boxing.tensor_store", "tensor-store"),
    ),
)
def test_tensor_boundary_boxing_matches_nncase_transfer_forms(
    source_distributed, candidate_id, semantics
):
    placement = fm.Placement((2, 4), "yx", "bb")
    tensor = fm.tensor_type("bfloat16", (1, 8))
    distributed = fm.DistributedType(
        tensor, (fm.SBP.broadcast(), fm.SBP.broadcast()), placement
    )
    source, target = (
        (distributed, tensor) if source_distributed else (tensor, distributed)
    )

    proposed = NvidiaSm90Target().propose_tir(_boxing_module(source, target))
    point = next(value for value in proposed.selection_points if value.id == "tir.output")

    assert tuple(value.id for value in point.candidates) == (candidate_id,)
    assert point.candidates[0].facts["collective_semantics"] == semantics
    # Scalar/vector element size must not restrict the CTA to a half warp.
    assert point.candidates[0].parameters["tile"] == 1024


def test_tuple_partial_all_reduce_is_one_recursive_boxing_candidate():
    placement = fm.Placement((2, 4), "yx", "bb")
    tensors = (
        fm.tensor_type("bfloat16", (1, 16)),
        fm.tensor_type("bfloat16", (1, 8)),
        fm.tensor_type("bfloat16", (1, 8)),
    )
    source = fm.TupleType(tuple(
        fm.DistributedType(
            tensor,
            (fm.SBP.broadcast(), fm.SBP.broadcast()),
            placement,
            partial=fm.SBP.partial((0, 1)),
        )
        for tensor in tensors
    ))
    target = fm.TupleType(tuple(
        fm.DistributedType(
            tensor,
            (fm.SBP.broadcast(), fm.SBP.broadcast()),
            placement,
        )
        for tensor in tensors
    ))

    proposed = NvidiaSm90Target().propose_tir(_boxing_module(source, target))
    point = next(value for value in proposed.selection_points if value.id == "tir.output")
    candidate = point.candidates[0]

    assert candidate.id == "tir.distributed_boxing.gather_reduce_scatter"
    assert candidate.parameters["leaf_transitions"] == (
        "gather_reduce_scatter",
        "gather_reduce_scatter",
        "gather_reduce_scatter",
    )
    assert candidate.facts["collective_semantics"] == (
        "tuple-gather-reduce-scatter"
    )
    assert candidate.facts["tuple_field_count"] == 3


@pytest.mark.parametrize("partial_axes", [(0,), (1,), (0, 1)])
def test_partial_to_plain_tensor_still_requires_collective_reduction(partial_axes):
    placement = fm.Placement((2, 4), "yx", "bb")
    tensor = fm.tensor_type("float32", (1, 8))
    source = fm.DistributedType(
        tensor, (fm.SBP.broadcast(), fm.SBP.broadcast()), placement,
        partial=fm.SBP.partial(partial_axes),
    )
    proposed = NvidiaSm90Target().propose_tir(_boxing_module(source, tensor))
    point = next(value for value in proposed.selection_points if value.id == "tir.output")
    assert point.candidates[0].id == "tir.distributed_boxing.gather_reduce_scatter"
    assert point.candidates[0].parameters["leaf_transitions"] == ("gather_reduce_scatter",)
