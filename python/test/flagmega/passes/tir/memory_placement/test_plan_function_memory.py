# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT

from dataclasses import replace

import pytest

from triton.flagmega import ir as fm
from triton.flagmega.errors import IRVerificationError
from triton.flagmega.passes.tir import (
    FUNCTION_MEMORY_PLACEMENT_SCHEMA,
    MEMORY_SPACE_METADATA,
    materialize_kernel_prim_functions,
    plan_function_memory,
)
from triton.flagmega.passes.tir.bufferize import BufferizationOptions


def _options() -> BufferizationOptions:
    generic = BufferizationOptions.generic(alignment=64)
    workspace, rdata, external = generic.memory_spaces
    return BufferizationOptions(
        (
            replace(workspace, sharing_scope=fm.MemorySharingScope.CHIP),
            replace(
                workspace,
                name="block_local_data",
                sharing_scope=fm.MemorySharingScope.BLOCK,
            ),
            rdata,
            external,
        ),
        block_local="block_local_data",
    )


def _distributed_type(*, partial=False):
    placement = fm.Placement((2, 4), "yx", "bb")
    return fm.DistributedType(
        fm.tensor_type("float32", (8, 16)),
        (fm.SBP.split_contiguous((0,), 8), fm.SBP.broadcast()),
        placement,
        fm.SBP.partial((0,)) if partial else None,
    )


def _kernel_attrs(*, collective=False, synchronized=False):
    facts = {"collective_semantics": "test"} if collective else {}
    if synchronized:
        facts["requires"] = ("grid_sync",)
    return {
        "semantic_op": "math.silu",
        "candidate": "tir.silu.local",
        "parameters": {"family": "elementwise", "variant": "silu"},
        "facts": facts,
        "semantic_attrs": {},
    }


def _chain(
    *,
    partial=False,
    collective_producer=False,
    collective_consumer=False,
    through_view=False,
    through_sharded_view=False,
):
    value_type = _distributed_type(partial=partial)
    builder = fm.IRBuilder(dialect="semantic_tir", stage="packaged_tir")
    source = builder.var("source", value_type, id="source")
    producer = builder.call(
        "tir.kernel",
        (source,),
        value_type,
        id="producer",
        attrs=_kernel_attrs(collective=collective_producer),
    )
    consumer_input = producer
    if through_view:
        consumer_input = builder.call(
            "tir.buffer_view",
            (producer,),
            value_type,
            id="view",
            attrs={"alias_kind": "reshape"},
        )
    if through_sharded_view:
        view_type = fm.DistributedType(
            value_type.tensor,
            (fm.SBP.broadcast(), fm.SBP.broadcast()),
            value_type.placement,
        )
        consumer_input = builder.call(
            "distributed.sharded_view",
            (producer,),
            view_type,
            id="sharded_view",
            attrs={"new_type": view_type},
        )
    consumer_type = consumer_input.type
    consumer = builder.call(
        "tir.kernel",
        (consumer_input,),
        consumer_type,
        id="consumer",
        attrs=_kernel_attrs(collective=collective_consumer),
    )
    builder.function("main", (source,), (consumer,))
    return materialize_kernel_prim_functions(builder.build(entry="main"))


def test_owner_local_distributed_edge_is_placed_in_block_pool():
    result = plan_function_memory(_chain(), _options())

    assert result.node_map["producer"].metadata[MEMORY_SPACE_METADATA] == (
        "block_local_data"
    )
    assert MEMORY_SPACE_METADATA not in result.node_map["consumer"].metadata
    assert result.metadata["function_memory_placement"] == {
        "schema": FUNCTION_MEMORY_PLACEMENT_SCHEMA,
        "default": "workspace",
        "block_local": "block_local_data",
        "promoted": ("producer",),
    }


def test_collective_producer_with_owner_local_output_uses_block_pool():
    module = _chain(collective_producer=True)
    producer = fm.kernel_dispatch_for_call(module, module.node_map["producer"])
    assert producer is not None
    assert (
        fm.kernel_execution_kind_for_call(module, module.node_map["producer"])
        is fm.KernelExecutionKind.COLLECTIVE
    )
    output_effect = producer.memory_effect_map[producer.outputs[0]]
    assert output_effect.scope is not fm.MemoryAccessScope.CHIP
    assert output_effect.owner_access is fm.MemoryOwnerAccess.LOCAL

    result = plan_function_memory(module, _options())

    assert result.node_map["producer"].metadata[MEMORY_SPACE_METADATA] == (
        "block_local_data"
    )


@pytest.mark.parametrize(
    "module",
    (
        _chain(partial=True),
        _chain(through_sharded_view=True),
    ),
    ids=("partial", "canonical-sharded-view"),
)
def test_unsafe_or_nonlocal_edges_remain_in_default_pool(module):
    result = plan_function_memory(module, _options())

    assert MEMORY_SPACE_METADATA not in result.node_map["producer"].metadata
    assert result.metadata["function_memory_placement"]["promoted"] == ()


def test_tuple_edge_consumed_by_synchronized_kernel_uses_block_pool():
    """Internal synchronization does not widen owner-local operand access."""

    value_type = _distributed_type()
    pair_type = fm.TupleType((value_type, value_type))
    builder = fm.IRBuilder(dialect="semantic_tir", stage="packaged_tir")
    source = builder.var("source", value_type, id="source")
    pair = builder.call(
        "tir.kernel",
        (source,),
        pair_type,
        id="pair",
        attrs=_kernel_attrs(),
    )
    first = builder.call(
        "builtin.get_item", (pair,), value_type, id="first", attrs={"index": 0}
    )
    second = builder.call(
        "builtin.get_item", (pair,), value_type, id="second", attrs={"index": 1}
    )
    consumed = builder.call(
        "tir.kernel",
        (first,),
        value_type,
        id="consumed",
        attrs=_kernel_attrs(synchronized=True),
    )
    result = builder.call(
        "tir.kernel",
        (second, consumed),
        value_type,
        id="result",
        attrs={**_kernel_attrs(), "semantic_op": "math.mul"},
    )
    builder.function("main", (source,), (result,))
    module = materialize_kernel_prim_functions(builder.build(entry="main"))

    placed = plan_function_memory(module, _options())

    assert placed.node_map["pair"].metadata[MEMORY_SPACE_METADATA] == (
        "block_local_data"
    )
    plan = fm.make_buffer_plan(placed, options=_options())
    assert plan.buffer_map["pair.0"].storage == "block_local_data"
    assert plan.buffer_map["pair.1"].storage == "block_local_data"


def test_tuple_field_with_inplace_path_to_function_result_stays_caller_visible():
    value_type = _distributed_type()
    pair_type = fm.TupleType((value_type, value_type))
    builder = fm.IRBuilder(dialect="semantic_tir", stage="packaged_tir")
    source = builder.var("source", value_type, id="source")
    pair = builder.call(
        "tir.kernel", (source,), pair_type, id="pair", attrs=_kernel_attrs()
    )
    first = builder.call(
        "builtin.get_item", (pair,), value_type, id="first", attrs={"index": 0}
    )
    second = builder.call(
        "builtin.get_item", (pair,), value_type, id="second", attrs={"index": 1}
    )
    result = builder.call(
        "tir.kernel",
        (first, second),
        value_type,
        id="result",
        attrs={**_kernel_attrs(), "semantic_op": "math.add"},
    )
    builder.function("decode", (source,), (result,))
    module = materialize_kernel_prim_functions(builder.build(entry="decode"))

    placed = plan_function_memory(module, _options())

    assert MEMORY_SPACE_METADATA not in placed.node_map[pair.id].metadata
    plan = fm.make_buffer_plan(placed, options=_options())
    assert plan.buffer_map["pair.0"].storage != "block_local_data"



def test_owner_local_edge_through_buffer_view_keeps_one_block_pool_backing():
    options = _options()
    result = plan_function_memory(_chain(through_view=True), options)

    assert result.node_map["producer"].metadata[MEMORY_SPACE_METADATA] == (
        "block_local_data"
    )
    plan = fm.make_buffer_plan(result, options=options)
    producer = plan.buffer_map["producer"]
    view = plan.buffer_map["view"]
    assert producer.storage == "block_local_data"
    assert producer.mem_span.must_alias(view.mem_span)


def test_metadata_only_expanding_view_does_not_force_canonical_backing():
    """A shape-only branch must not widen a separately consumed local value."""

    local_type = _distributed_type()
    broadcast_type = fm.DistributedType(
        local_type.tensor,
        (fm.SBP.broadcast(), fm.SBP.broadcast()),
        local_type.placement,
    )
    rotary_type = fm.DistributedType(
        fm.tensor_type("float32", (8, 1, 8)),
        (fm.SBP.broadcast(), fm.SBP.broadcast(), fm.SBP.broadcast()),
        local_type.placement,
    )
    state_type = fm.RefType(
        "state",
        (("seq_lens", fm.tensor_type("int32", (1,))),),
    )
    builder = fm.IRBuilder(dialect="semantic_tir", stage="packaged_tir")
    source = builder.var("source", local_type, id="source")
    state = builder.var("state", state_type, id="state")
    producer = builder.call(
        "tir.kernel", (source,), local_type, id="producer", attrs=_kernel_attrs()
    )
    metadata_view = builder.call(
        "distributed.sharded_view",
        (producer,),
        broadcast_type,
        id="metadata_view",
        attrs={"new_type": broadcast_type},
    )
    rotary = builder.call(
        "tir.kernel",
        (metadata_view, state),
        fm.TupleType((rotary_type, rotary_type)),
        id="rotary",
        attrs={
            "semantic_op": "nn.rotary_embedding",
            "candidate": "tir.rotary_embedding.decode",
            "parameters": {
                "family": "rotary_embedding",
                "variant": "decode",
                "elements_per_program": 8,
            },
            "facts": {},
            "semantic_attrs": {
                "head_dim": 8,
                "theta": 10000.0,
                "attention_scaling": 1.0,
            },
        },
    )
    consumed = builder.call(
        "tir.kernel", (producer,), local_type, id="consumed", attrs=_kernel_attrs()
    )
    builder.function("main", (source, state), (consumed, rotary))
    module = materialize_kernel_prim_functions(builder.build(entry="main"))

    placed = plan_function_memory(module, _options())

    assert placed.node_map["producer"].metadata[MEMORY_SPACE_METADATA] == (
        "block_local_data"
    )
    plan = fm.make_buffer_plan(placed, options=_options())
    values = dict(plan.function_map["main"].values)
    assert "metadata_view" not in values
    assert plan.buffer_map["producer"].storage == "block_local_data"
    assert plan.buffer_map["producer"].distributed_storage_kind is (
        fm.DistributedBufferStorageKind.COMPACT_LOCAL
    )


def test_broadcast_replica_feeds_narrowing_sharded_views_in_one_block_pool():
    """A complete per-block replica is canonical within that block only."""

    tensor = fm.tensor_type("float32", (8, 16))
    placement = fm.Placement((2, 4), "yx", "bb")
    broadcast = fm.DistributedType(
        tensor,
        (fm.SBP.broadcast(), fm.SBP.broadcast()),
        placement,
    )
    split_y = fm.DistributedType(
        tensor,
        (fm.SBP.split_contiguous((0,), 4), fm.SBP.broadcast()),
        placement,
    )
    split_yx = fm.DistributedType(
        tensor,
        (
            fm.SBP.split_contiguous((0,), 4),
            fm.SBP.split_contiguous((1,), 4),
        ),
        placement,
    )
    builder = fm.IRBuilder(dialect="semantic_tir", stage="packaged_tir")
    source = builder.var("source", broadcast, id="source")
    produced = builder.call(
        "tir.kernel",
        (source,),
        broadcast,
        id="producer",
        attrs=_kernel_attrs(),
    )
    y_view = builder.call(
        "distributed.sharded_view",
        (produced,),
        split_y,
        id="y_view",
        attrs={"new_type": split_y},
    )
    yx_view = builder.call(
        "distributed.sharded_view",
        (produced,),
        split_yx,
        id="yx_view",
        attrs={"new_type": split_yx},
    )
    y_result = builder.call(
        "tir.kernel", (y_view,), split_y, id="y_result", attrs=_kernel_attrs()
    )
    yx_result = builder.call(
        "tir.kernel", (yx_view,), split_yx, id="yx_result", attrs=_kernel_attrs()
    )
    builder.function("main", (source,), (y_result, yx_result))
    module = materialize_kernel_prim_functions(builder.build(entry="main"))

    placed = plan_function_memory(module, _options())

    assert placed.node_map["producer"].metadata[MEMORY_SPACE_METADATA] == (
        "block_local_data"
    )
    plan = fm.make_buffer_plan(placed, options=_options())
    producer = plan.buffer_map["producer"]
    assert producer.storage == "block_local_data"
    assert producer.distributed_storage_kind is (
        fm.DistributedBufferStorageKind.COMPACT_LOCAL
    )
    for name in ("y_view", "yx_view"):
        assert MEMORY_SPACE_METADATA not in placed.node_map[name].metadata
        view = plan.buffer_map[name]
        assert view.storage == "block_local_data"
        assert view.distributed_storage_kind is (
            fm.DistributedBufferStorageKind.REPLICATED_LOCAL
        )
        assert view.mem_span.must_alias(producer.mem_span)
        assert view.nbytes == 8 * 16 * 4


def test_contiguous_split_refinement_keeps_parent_shard_in_one_block_pool():
    """A y shard can back the yx subview owned by the same block."""

    tensor = fm.tensor_type("float32", (128,))
    placement = fm.Placement((8, 16), "yx", "bb")
    split_y = fm.DistributedType(
        tensor,
        (fm.SBP.split_contiguous((0,), 16),),
        placement,
    )
    split_yx = fm.DistributedType(
        tensor,
        (fm.SBP.split_contiguous((0, 1), 1),),
        placement,
    )
    builder = fm.IRBuilder(dialect="semantic_tir", stage="packaged_tir")
    source = builder.var("source", split_y, id="source")
    produced = builder.call(
        "tir.kernel",
        (source,),
        split_y,
        id="producer",
        attrs=_kernel_attrs(),
    )
    refined = builder.call(
        "distributed.sharded_view",
        (produced,),
        split_yx,
        id="refined",
        attrs={"new_type": split_yx},
    )
    result = builder.call(
        "tir.kernel", (refined,), split_yx, id="result", attrs=_kernel_attrs()
    )
    builder.function("main", (source,), (result,))
    module = materialize_kernel_prim_functions(builder.build(entry="main"))

    placed = plan_function_memory(module, _options())

    assert placed.node_map["producer"].metadata[MEMORY_SPACE_METADATA] == (
        "block_local_data"
    )
    plan = fm.make_buffer_plan(placed, options=_options())
    producer = plan.buffer_map["producer"]
    view = plan.buffer_map["refined"]
    assert producer.distributed_storage_kind is (
        fm.DistributedBufferStorageKind.COMPACT_LOCAL
    )
    assert view.distributed_storage_kind is (
        fm.DistributedBufferStorageKind.COMPACT_LOCAL
    )
    assert view.distributed_backing_type == split_y
    assert view.storage_distributed_type == split_y
    assert view.mem_span.must_alias(producer.mem_span)
    assert view.nbytes == 16 * 4


def test_agent_placement_on_an_alias_cannot_silently_relocate_storage():
    module = _chain(through_view=True)
    view = module.node_map["view"]
    edited = replace(
        module,
        nodes=tuple(
            replace(
                node,
                metadata={**node.metadata, MEMORY_SPACE_METADATA: "block_local_data"},
            )
            if node.id == view.id
            else node
            for node in module.nodes
        ),
    )

    with pytest.raises(IRVerificationError, match="Alias .* requests memory space"):
        fm.make_buffer_plan(edited, options=_options())


def test_resume_rejects_a_buffer_plan_that_ignores_edited_placement():
    options = _options()
    placed = plan_function_memory(_chain(), options)
    plan = fm.make_buffer_plan(placed, options=options)
    edited = replace(
        placed,
        stage="bufferized_tir",
        dialect="bufferized_tir",
        nodes=tuple(
            replace(
                node,
                metadata={**node.metadata, MEMORY_SPACE_METADATA: "workspace"},
            )
            if node.id == "producer"
            else node
            for node in placed.nodes
        ),
        metadata={**placed.metadata, "buffer_plan": plan.to_data()},
    )

    with pytest.raises(
        IRVerificationError,
        match="requests memory space 'workspace'.*placed in 'block_local_data'",
    ):
        fm.verify_buffer_plan(edited)


def test_explicit_auto_placement_overrides_optional_inplace_reuse():
    value_type = _distributed_type()
    builder = fm.IRBuilder(dialect="semantic_tir", stage="packaged_tir")
    lhs = builder.var("lhs", value_type, id="lhs")
    rhs = builder.var("rhs", value_type, id="rhs")
    added = builder.call(
        "tir.kernel",
        (lhs, rhs),
        value_type,
        id="added",
        attrs={
            **_kernel_attrs(),
            "semantic_op": "math.add",
        },
    )
    result = builder.call(
        "tir.kernel",
        (added,),
        value_type,
        id="result",
        attrs=_kernel_attrs(),
    )
    builder.function("main", (lhs, rhs), (result,))
    module = materialize_kernel_prim_functions(builder.build(entry="main"))

    placed = plan_function_memory(module, _options())

    assert placed.node_map["added"].metadata[MEMORY_SPACE_METADATA] == (
        "block_local_data"
    )
    plan = fm.make_buffer_plan(placed, options=_options())
    added_buffer = plan.buffer_map["added"]
    assert added_buffer.storage == "block_local_data"
    assert added_buffer.alias is None


def test_same_pool_inplace_candidate_still_reuses_dead_temporary():
    value_type = _distributed_type()
    builder = fm.IRBuilder(dialect="semantic_tir", stage="packaged_tir")
    source = builder.var("source", value_type, id="source")
    rhs = builder.var("rhs", value_type, id="rhs")
    temporary = builder.call(
        "tir.kernel", (source,), value_type, id="temporary", attrs=_kernel_attrs()
    )
    added = builder.call(
        "tir.kernel",
        (temporary, rhs),
        value_type,
        id="added",
        attrs={**_kernel_attrs(), "semantic_op": "math.add"},
    )
    result = builder.call(
        "tir.kernel", (added,), value_type, id="result", attrs=_kernel_attrs()
    )
    builder.function("decode", (source, rhs), (result,))
    module = materialize_kernel_prim_functions(builder.build(entry="decode"))

    plan = fm.make_buffer_plan(module, options=_options())

    assert plan.buffer_map["added"].mem_span.must_alias(
        plan.buffer_map["temporary"].mem_span
    )
