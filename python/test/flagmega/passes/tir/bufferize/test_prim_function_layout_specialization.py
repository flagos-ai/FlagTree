# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT

from triton.flagmega import ir as fm
from triton.flagmega.passes.tir.bufferize import (
    BufferizationOptions,
    NttBufferizationPolicy,
)


def _fully_sharded_type():
    tensor = fm.tensor_type("bfloat16", (4, 4))
    placement = fm.Placement((2, 2), "yx", "bb")
    return fm.DistributedType(
        tensor,
        (
            fm.SBP.split_block_cyclic((0,), 1),
            fm.SBP.split_block_cyclic((1,), 1),
        ),
        placement,
    )


def _shared_unary_module():
    value_type = _fully_sharded_type()
    dispatch = fm.T.kernel_dispatch(
        semantic_op="math.silu",
        semantic_candidate="tir.unary.silu",
        arguments=("input",),
        outputs=("output",),
        reads=("input",),
        writes=("output",),
    )
    kernel = fm.T.prim_function(
        "shared_silu",
        "triton",
        (
            fm.T.prim_parameter(
                "input", value_type, fm.T.PrimParameterRole.INPUT
            ),
            fm.T.prim_parameter(
                "output", value_type, fm.T.PrimParameterRole.OUTPUT
            ),
        ),
        fm.T.sequential((dispatch,)),
        fm.T.return_((
            fm.T.return_binding(
                fm.T.value_ref("output", value_type), "output"
            ),
        )),
    )
    builder = fm.IRBuilder(dialect="semantic_tir", stage="selected_microkernels")
    weight = builder.weight(
        "weight", value_type, source="weights", key="weight", id="weight"
    )
    builder.prim_function(kernel)
    first = builder.call(
        "tir.call", (weight,), value_type,
        id="first", attrs={"callee": kernel.name},
    )
    second = builder.call(
        "tir.call", (first,), value_type,
        id="second", attrs={"callee": kernel.name},
    )
    third = builder.call(
        "tir.call", (second,), value_type,
        id="third", attrs={"callee": kernel.name},
    )
    tail = builder.call("math.silu", (third,), value_type, id="tail")
    builder.function("main", (), (tail,))
    return builder.build(entry="main")


def test_bufferize_specializes_only_distinct_physical_layouts_and_preserves_reuse():
    module = NttBufferizationPolicy(
        BufferizationOptions.generic()
    ).bufferize(_shared_unary_module())
    plan = fm.verify_buffer_plan(module)

    callees = {
        node.id: str(node.attrs["callee"])
        for node in module.nodes
        if node.op == "tir.call"
    }
    assert callees["second"] == callees["third"] == "shared_silu"
    assert callees["first"] == "shared_silu__layout_1"
    assert len(module.prim_functions) == 2

    compact = module.prim_function_map["shared_silu"]
    canonical = module.prim_function_map["shared_silu__layout_1"]
    assert (
        compact.runtime_parameters[0].buffers[0].distributed_storage_kind
        is fm.DistributedBufferStorageKind.COMPACT_LOCAL
    )
    assert (
        canonical.runtime_parameters[0].buffers[0].distributed_storage_kind
        is fm.DistributedBufferStorageKind.CANONICAL_GLOBAL
    )
    assert all(
        function.output_parameters[0].buffers[0].distributed_storage_kind
        is fm.DistributedBufferStorageKind.COMPACT_LOCAL
        for function in (compact, canonical)
    )
    assert plan.kernel_call_map["second"].callee == "shared_silu"
    assert plan.kernel_call_map["third"].callee == "shared_silu"
    records = module.metadata["buffer_layout_specializations"]
    assert records[0]["source"] == "shared_silu"
    assert tuple(
        tuple(variant["calls"])
        for variant in records[0]["variants"]
    ) == (("second", "third"), ("first",))
