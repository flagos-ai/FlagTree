# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT

from dataclasses import replace

from triton.flagmega import ir as fm
from triton.flagmega.passes.packed_qkv_combine import (
    fold_materialized_packed_qkv_parallel_linear_combine,
)
from triton.flagmega.passes.rewriter import DataflowPass
from triton.flagmega.rules.ntt import (
    fold_materialized_packed_qkv_parallel_linear_combine_rule,
)


def _materialized_module():
    placement = fm.Placement((8, 16), "yx", "bb")
    result_type = fm.TupleType(tuple(
        fm.DistributedType(
            fm.tensor_type(fm.vector_type("bfloat16", (8,)), (1, extent)),
            (fm.SBP.broadcast(), fm.SBP.split_block_cyclic((1,), 8)),
            placement,
        )
        for extent in (256, 128, 128)
    ))
    builder = fm.IRBuilder(dialect="ntt", stage="distributed")
    value = builder.var("qkv", result_type, id="qkv")
    combine = builder.call(
        "ntt.packed_qkv_parallel_linear_combine",
        (value,),
        result_type,
        id="combine",
        attrs={"output_type": result_type},
    )
    builder.function("main", (value,), (combine,))
    return builder.build(entry="main")


def test_fold_materialized_combine_redirects_all_uses_to_qkv():
    module = DataflowPass(
        "FoldMaterializedPackedQKVParallelLinearCombine",
        (fold_materialized_packed_qkv_parallel_linear_combine_rule(),),
    ).run(_materialized_module())

    assert "combine" not in module.node_map
    assert module.function_map["main"].outputs == ("qkv",)


def test_fold_drops_combine_selection_before_redirecting_to_projection_owner():
    module = _materialized_module()
    projection = replace(
        module.node_map["qkv"],
        metadata={"distributed_candidate": "projection-layout"},
    )
    combine = replace(
        module.node_map["combine"],
        metadata={"distributed_candidate": "combine-layout"},
    )
    module = replace(
        module,
        nodes=(projection, combine),
        selection_points=(
            fm.SelectionPoint(
                "distribution.qkv",
                "distribution",
                (fm.Candidate("projection-layout", {}, {}),),
                "projection-layout",
                "qkv",
            ),
            fm.SelectionPoint(
                "distribution.combine",
                "distribution",
                (fm.Candidate("combine-layout", {}, {}),),
                "combine-layout",
                "combine",
            ),
        ),
        selections=(
            fm.SelectionRecord(
                "distribution.qkv", "projection-layout", "test", "test/v1"
            ),
            fm.SelectionRecord(
                "distribution.combine", "combine-layout", "test", "test/v1"
            ),
        ),
    )

    result = fold_materialized_packed_qkv_parallel_linear_combine(module)

    assert "combine" not in result.node_map
    assert tuple(point.id for point in result.selection_points) == (
        "distribution.qkv",
    )
    assert result.selection_points[0].owner == "qkv"
    assert result.selections[0].candidate_id == "projection-layout"
