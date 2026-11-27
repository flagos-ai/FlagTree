# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT

from triton.flagmega.codegen.triton.templates import TritonTemplateRegistry
from triton.flagmega.codegen.triton.kernels.distributed_boxing.reduction import partial_reduction_context


def test_call_graph_emits_each_boxing_leaf_and_partial_owner_reduction():
    reduction = {
        **partial_reduction_context("sum", "bfloat16"),
        "reduction": True,
        "capacity": 256,
        "tile": 16,
        "active": "boxing_offsets < 256",
        "source": "partial",
        "result": "materialized",
        "source_offset": "boxing_offsets",
        "result_offset": "shard_y * 256 + boxing_offsets",
        "writer_active": "shard_x == 0",
        "partial_owner_count": 16,
        "partial_owner_tile": 16,
        "placement_owner_count": 128,
        "partial_owner": "shard_y * 16 + boxing_partial_member",
        "owner_stride": 256,
        "output_type": "tl.bfloat16",
    }
    copy = {
        "reduction": False,
        "capacity": 32,
        "tile": 16,
        "active": "boxing_offsets < 32",
        "source": "source",
        "result": "result",
        "source_offset": "boxing_offsets",
        "result_offset": "boxing_offsets",
        "writer_active": "True",
    }
    call = {
        "symbol": "_flagmega_test_boxing",
        "signature": "partial, materialized, source, result",
        "execution_kind": "collective",
        "family": "distributed_boxing",
        "variant": "gather_reduce_scatter",
        "internal_grid_barriers": 0,
        "leaves": (reduction, copy),
    }

    source = TritonTemplateRegistry().render(
        "kernels/distributed_boxing/gather_reduce_scatter.py.jinja",
        {
            "render_calls": (call,),
            "distributed_entry": True,
            "mesh_axis_names": ("y", "x"),
            "mesh_hierarchy": (8, 16),
        },
    )

    compile(source, "boxing_call_graph.py", "exec")
    assert "0, 16, 16," in source
    assert "boxing_partial_member[:, None] < 16" in source
    assert "tl.sum(boxing_partial_values, axis=0)" in source
    assert "tl.static_range" not in source
    assert "+ boxing_partial_owner[:, None] * 256" in source
    assert "boxing_accumulator.to(tl.bfloat16)" in source
    assert "mask=boxing_mask & (shard_x == 0)" in source
    assert "source + (boxing_offsets)" in source
