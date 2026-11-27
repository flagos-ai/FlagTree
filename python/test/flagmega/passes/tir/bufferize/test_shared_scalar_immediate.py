# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT

from triton.flagmega import ir as fm
from triton.flagmega.passes.tir.bufferize import plan_buffers


def test_scalar_immediate_shared_by_callee_and_caller_has_one_binding():
    tensor = fm.tensor_type("float32", (4,))
    scalar = fm.tensor_type("bool", ())
    placement = fm.Placement((2, 4), "yx", "bb")
    distributed_scalar = fm.DistributedType(scalar, (), placement)
    layer_result_type = fm.TupleType((tensor, distributed_scalar))
    program_result_type = fm.TupleType((layer_result_type, distributed_scalar))
    shared = fm.Node(
        "shared",
        "tir.scalar_const",
        (),
        scalar,
        attrs={"value": False},
    )
    shared_view = fm.Node(
        "shared_view",
        "distributed.sharded_view",
        ("shared",),
        distributed_scalar,
        attrs={"new_type": distributed_scalar},
    )
    layer_arg = fm.Node("layer_arg", "builtin.var", (), tensor)
    layer_result = fm.Node(
        "layer_result",
        "builtin.tuple",
        ("layer_arg", "shared_view"),
        layer_result_type,
    )
    main_arg = fm.Node("main_arg", "builtin.var", (), tensor)
    call = fm.Node(
        "call",
        "tir.call",
        ("main_arg",),
        layer_result_type,
        attrs={"callee": "layer"},
    )
    program_result = fm.Node(
        "program_result",
        "builtin.tuple",
        ("call", "shared_view"),
        program_result_type,
    )
    module = fm.IRModule(
        dialect="semantic_tir",
        stage="packaged_tir",
        nodes=(
            shared,
            shared_view,
            layer_arg,
            layer_result,
            main_arg,
            call,
            program_result,
        ),
        functions=(
            fm.Function("layer", ("layer_arg",), ("layer_result",)),
            fm.Function("main", ("main_arg",), ("program_result",)),
        ),
        entry="main",
    )

    plan = plan_buffers(module)

    immediate = tuple(value for value in plan.buffers if value.id == "shared")
    assert len(immediate) == 1
    assert immediate[0].storage == "scalar"
    assert immediate[0].function is None
    assert immediate[0].mem_span.buffer.id == "scalar:shared"
    views = tuple(value for value in plan.buffers if value.id == "shared_view")
    assert len(views) == 1
    assert views[0].mem_span.must_alias(immediate[0].mem_span)
