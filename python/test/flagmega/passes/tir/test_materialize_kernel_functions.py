# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT

from triton.flagmega import ir as fm
from triton.flagmega.passes.functions import callee_first_functions
from triton.flagmega.passes.tir import materialize_kernel_prim_functions


def _two_add_kernels(*, second_candidate: str = "tir.add.vector") -> fm.IRModule:
    builder = fm.IRBuilder(dialect="semantic_tir", stage="selected_tir")
    value_type = fm.tensor_type("bfloat16", (1, 16))
    lhs = builder.var("lhs", value_type, id="lhs")
    rhs = builder.var("rhs", value_type, id="rhs")
    attrs = {
        "semantic_op": "math.add",
        "candidate": "tir.add.vector",
        "parameters": {"family": "add", "variant": "vector", "lanes": 8},
        "facts": {"bytes_read": 64, "bytes_written": 32},
        "semantic_attrs": {},
    }
    first = builder.call("tir.kernel", (lhs, rhs), value_type, id="first", attrs=attrs)
    second = builder.call(
        "tir.kernel",
        (lhs, rhs),
        value_type,
        id="second",
        attrs={**attrs, "candidate": second_candidate},
    )
    builder.function("main", (lhs, rhs), (first, second))
    return fm.verify_module(builder.build(entry="main"))


def test_equivalent_selected_kernels_share_one_definition_not_a_function():
    result = materialize_kernel_prim_functions(_two_add_kernels())

    assert not [node for node in result.nodes if node.op == "tir.kernel"]
    calls = [node for node in result.nodes if node.op == "tir.call"]
    assert len(calls) == 2
    assert calls[0].attrs == calls[1].attrs
    assert len(result.kernel_definitions) == 1
    assert not result.prim_functions
    assert isinstance(result.kernel_definitions[0], fm.KernelDefinition)
    assert "body" not in result.kernel_definitions[0].to_data()
    dispatch = fm.kernel_dispatch_for_call(result, calls[0])
    assert dispatch.semantic_op == "math.add"
    assert dispatch.candidate == "tir.add.vector"
    assert dispatch.arguments == ("lhs", "rhs")
    assert dispatch.outputs == ("result",)
    assert dispatch.reads == ("lhs", "rhs")
    assert dispatch.writes == ("result",)
    assert dispatch.inplace_alias_candidates == (
        fm.InplaceAliasCandidate(output="result", input="lhs"),
    )


def test_candidate_change_creates_a_distinct_prim_function():
    result = materialize_kernel_prim_functions(
        _two_add_kernels(second_candidate="tir.add.scalar")
    )

    calls = [node for node in result.nodes if node.op == "tir.call"]
    assert len(result.kernel_definitions) == 2
    assert calls[0].attrs["callee"] != calls[1].attrs["callee"]


def test_boxing_sites_do_not_share_a_prim_function_before_buffer_layout_binding():
    placement = fm.Placement((2, 4), "yx", "bb")
    tensor = fm.tensor_type("bfloat16", (1, 16))
    split = fm.DistributedType(
        tensor,
        (fm.SBP.broadcast(), fm.SBP.split_contiguous((0, 1))),
        placement,
    )
    broadcast = fm.DistributedType(
        tensor,
        (fm.SBP.broadcast(), fm.SBP.broadcast()),
        placement,
    )
    builder = fm.IRBuilder(dialect="semantic_tir", stage="selected_tir")
    first_input = builder.var("first_input", split, id="first_input")
    second_input = builder.var("second_input", split, id="second_input")
    attrs = {
        "semantic_op": "distributed.boxing",
        "candidate": "tir.distributed_boxing.gather_reduce_scatter",
        "parameters": {
            "family": "distributed_boxing",
            "variant": "gather_reduce_scatter",
        },
        "facts": {},
        "semantic_attrs": {"new_type": broadcast},
    }
    first = builder.call(
        "tir.kernel", (first_input,), broadcast, id="first", attrs=attrs
    )
    second = builder.call(
        "tir.kernel", (second_input,), broadcast, id="second", attrs=attrs
    )
    builder.function("main", (first_input, second_input), (first, second))

    result = materialize_kernel_prim_functions(builder.build(entry="main"))
    calls = tuple(node for node in result.nodes if node.op == "tir.call")

    assert len(result.kernel_definitions) == 2
    assert calls[0].attrs["callee"] != calls[1].attrs["callee"]


def test_semantic_tir_candidate_materializes_without_a_target_microkernel():
    builder = fm.IRBuilder(dialect="semantic_tir", stage="selected_tir")
    value_type = fm.tensor_type("bfloat16", (1, 16))
    lhs = builder.var("lhs", value_type, id="lhs")
    rhs = builder.var("rhs", value_type, id="rhs")
    result = builder.call(
        "tir.kernel",
        (lhs, rhs),
        value_type,
        id="result",
        attrs={
            "semantic_op": "math.add",
            "semantic_candidate": "semantic.math.add",
            "semantic_parameters": {},
            "semantic_facts": {"elementwise": True},
            "semantic_attrs": {},
        },
    )
    builder.function("main", (lhs, rhs), (result,))

    lowered = materialize_kernel_prim_functions(builder.build(entry="main"))
    dispatch = fm.kernel_dispatch_for_call(lowered, lowered.node_map["result"])

    assert dispatch.semantic_candidate == "semantic.math.add"
    assert dispatch.semantic_facts == {"elementwise": True}
    assert dispatch.microkernel is None


def test_prim_function_calls_are_leaves_for_graph_call_order_and_bufferize():
    result = materialize_kernel_prim_functions(_two_add_kernels())

    assert [value.name for value in callee_first_functions(result)] == ["main"]
    plan = fm.make_buffer_plan(result)
    assert dict(plan.entry_outputs).keys() == {"first", "second"}


def test_materialized_kernel_python_checkpoint_round_trips(tmp_path):
    result = materialize_kernel_prim_functions(_two_add_kernels())

    path = fm.emit_module(result, tmp_path / "selected.py")
    loaded = fm.load_module(path)

    assert loaded.semantic_hash == result.semantic_hash
    assert fm.kernel_dispatch_for_call(loaded, loaded.node_map["first"]) == (
        fm.kernel_dispatch_for_call(result, result.node_map["first"])
    )


def test_tuple_result_alias_contract_uses_named_kernel_abi():
    builder = fm.IRBuilder(dialect="semantic_tir", stage="selected_tir")
    matrix = fm.tensor_type("bfloat16", (1, 16))
    stats = fm.tensor_type("float32", (1, 1, 1))
    lhs = builder.var("lhs", matrix, id="lhs")
    rhs = builder.var("rhs", matrix, id="rhs")
    addend = builder.var("addend", matrix, id="addend")
    result_type = fm.TupleType((matrix, stats))
    result = builder.call(
        "tir.kernel",
        (lhs, rhs, addend),
        result_type,
        id="result",
        attrs={
            "semantic_op": "ntt.matmul_norm_stats",
            "semantic_candidate": "semantic.ntt.matmul_norm_stats",
            "semantic_parameters": {},
            "semantic_facts": {},
            "semantic_attrs": {
                "transpose_a": False,
                "transpose_b": False,
                "axis": 1,
                "use_mean": False,
            },
        },
    )
    builder.function("main", (lhs, rhs, addend), (result,))

    lowered = materialize_kernel_prim_functions(builder.build(entry="main"))
    dispatch = fm.kernel_dispatch_for_call(lowered, lowered.node_map["result"])

    assert dispatch.inplace_alias_candidates == (
        fm.InplaceAliasCandidate(output="result_0", input="addend"),
    )


def test_read_write_parameter_becomes_explicit_inout_abi():
    builder = fm.IRBuilder(dialect="semantic_tir", stage="selected_tir")
    value_type = fm.tensor_type("bfloat16", (1, 16))
    scalar_i32 = fm.tensor_type("int32", ())
    scalar_bool = fm.tensor_type("bool", ())
    state_type = fm.RefType("state", (("cache", value_type),))
    types = (
        value_type,
        state_type,
        scalar_i32,
        scalar_bool,
    )
    arguments = tuple(
        builder.var(f"arg{index}", value, id=f"arg{index}")
        for index, value in enumerate(types)
    )
    result_type = fm.TupleType((value_type, state_type))
    output = builder.call(
        "tir.kernel",
        arguments,
        result_type,
        id="attention",
        effect=fm.effect("read_write", "cache"),
        attrs={
            "semantic_op": "nn.update_paged_attention_kv_cache",
            "candidate": "tir.attention.unit",
            "parameters": {"family": "attention", "variant": "unit"},
            "facts": {},
            "semantic_attrs": {},
        },
    )
    builder.function("main", arguments, (output,))

    result = materialize_kernel_prim_functions(builder.build(entry="main"))
    function = result.kernel_definitions[0]
    dispatch = fm.kernel_dispatch_for_call(result, result.node_map["attention"])

    assert function.parameters[1].name == "state"
    assert function.parameters[1].role is fm.PrimParameterRole.INOUT
    assert "state" in dispatch.reads
    assert "state" in dispatch.writes
