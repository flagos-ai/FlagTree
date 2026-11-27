# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT

from triton.flagmega import ir as fm


def test_internal_full_mesh_producer_is_promoted_for_canonical_sharded_view():
    tensor = fm.tensor_type("bfloat16", (128, 128))
    placement = fm.Placement((8, 16), "yx", "bb")
    split_type = fm.DistributedType(
        tensor,
        (
            fm.SBP.split_block_cyclic((0,), 1),
            fm.SBP.split_block_cyclic((1,), 1),
        ),
        placement,
    )
    broadcast_type = fm.DistributedType(
        tensor,
        (fm.SBP.broadcast(), fm.SBP.broadcast()),
        placement,
    )
    builder = fm.IRBuilder(dialect="semantic_tir", stage="packaged_tir")
    lhs = builder.var("lhs", split_type, id="lhs")
    rhs = builder.var("rhs", split_type, id="rhs")
    produced = builder.call("math.add", (lhs, rhs), split_type, id="produced")
    view = builder.call(
        "distributed.sharded_view",
        (produced,),
        broadcast_type,
        id="view",
        attrs={"new_type": broadcast_type},
    )
    builder.function("main", (lhs, rhs), (view,))

    plan = fm.make_buffer_plan(builder.build(entry="main"))
    produced_buffer = plan.buffer_map["produced"]
    view_buffer = plan.buffer_map["view"]

    assert produced_buffer.mem_span.must_alias(view_buffer.mem_span)
    assert (
        produced_buffer.distributed_storage_kind
        is fm.DistributedBufferStorageKind.CANONICAL_GLOBAL
    )
    assert (
        view_buffer.distributed_storage_kind
        is fm.DistributedBufferStorageKind.CANONICAL_GLOBAL
    )
    assert produced_buffer.nbytes == 128 * 128 * 2
    assert produced_buffer.strides == (128, 1)
