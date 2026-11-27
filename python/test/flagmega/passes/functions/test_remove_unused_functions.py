# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT

from triton.flagmega import ir as fm
from triton.flagmega.compiler import Compiler
from triton.flagmega.passes.functions import remove_unused_functions


def _module_with_dead_function() -> fm.IRModule:
    builder = fm.IRBuilder(dialect="high_level", stage="stats_combined")
    value_type = fm.tensor_type("float32", (16,))

    dead_input = builder.var("dead_input", value_type, id="dead_input")
    dead_result = builder.call(
        "math.silu", (dead_input,), value_type, id="dead_result"
    )
    builder.function("dead", (dead_input,), (dead_result,))

    layer_input = builder.var("layer_input", value_type, id="layer_input")
    layer_result = builder.call(
        "math.silu", (layer_input,), value_type, id="layer_result"
    )
    builder.function("layer", (layer_input,), (layer_result,))

    main_input = builder.var("main_input", value_type, id="main_input")
    main_result = builder.call(
        "builtin.call",
        (main_input,),
        value_type,
        id="main_result",
        attrs={"callee": "layer"},
    )
    builder.function("main", (main_input,), (main_result,))
    return fm.verify_module(builder.build(entry="main"))


def test_remove_unused_functions_keeps_only_entry_call_graph_and_node_closure():
    module = _module_with_dead_function()

    result = fm.verify_module(remove_unused_functions(module))

    assert [function.name for function in result.functions] == ["layer", "main"]
    assert [node.id for node in result.nodes] == [
        "layer_input",
        "layer_result",
        "main_input",
        "main_result",
    ]


def test_remove_unused_functions_is_idempotent():
    once = remove_unused_functions(_module_with_dead_function())

    assert remove_unused_functions(once) is once


def test_pipeline_removes_dead_program_before_auto_distribution_search():
    result = Compiler().compile(
        _module_with_dead_function(), stop_after="auto-distributed"
    ).module

    assert "dead" not in result.function_map
    assert "dead_result" not in result.node_map
