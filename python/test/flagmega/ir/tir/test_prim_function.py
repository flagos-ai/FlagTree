# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT

from dataclasses import replace
from pathlib import Path

from triton.flagmega import ir as fm



def make_prim_module() -> fm.IRModule:
    tensor = fm.tensor_type("float32", (4,))
    input_buffer = fm.T.buffer(
        "input", "float32",
        fm.T.mem_span(fm.T.physical_buffer("input.storage", "global", 16, 16, role="input")),
        (4,), (1,),
    )
    output_buffer = fm.T.buffer(
        "output", "float32",
        fm.T.mem_span(fm.T.physical_buffer("output.storage", "global", 16, 16, role="output")),
        (4,), (1,),
    )
    index = fm.dim("i", minimum=0, maximum=3)
    load = fm.T.buffer_load(input_buffer, (index,))
    store = fm.T.buffer_store(output_buffer, (index,), load)
    block = fm.T.block(
        "copy", fm.T.sequential((store,)),
        reads=(fm.T.buffer_region(input_buffer, (fm.T.range(0, 4),)),),
        writes=(fm.T.buffer_region(output_buffer, (fm.T.range(0, 4),)),),
    )
    loop = fm.T.for_loop(
        index, fm.T.range(0, 4), fm.T.LoopMode.SERIAL, fm.T.sequential((block,))
    )
    prim = fm.T.prim_function(
        "copy_4", "triton",
        (
            fm.T.prim_parameter("input", tensor, fm.T.PrimParameterRole.INPUT),
            fm.T.prim_parameter("output", tensor, fm.T.PrimParameterRole.OUTPUT),
        ),
        fm.T.sequential((loop,)),
        fm.T.return_((fm.T.return_binding(output_buffer, "output"),)),
    )
    builder = fm.IRBuilder(dialect="semantic_tir", stage="tir_selected")
    source = builder.var("source", tensor, id="source")
    builder.prim_function(prim)
    result = builder.call("tir.call", (source,), tensor, id="result", attrs={"callee": "copy_4"})
    builder.function("main", (source,), (result,))
    return fm.verify_module(builder.build(entry="main"))


def test_prim_function_has_explicit_runtime_output_and_workspace_abi():
    module = make_prim_module()
    function = module.prim_function_map["copy_4"]

    assert tuple(value.name for value in function.runtime_parameters) == ("input",)
    assert tuple(value.name for value in function.output_parameters) == ("output",)
    assert function.runtime_parameter_types == (fm.tensor_type("float32", (4,)),)
    assert function.runtime_return_type == fm.tensor_type("float32", (4,))
    assert module.node_map["result"].attrs["callee"] == "copy_4"


def test_prim_function_data_and_editable_python_round_trip(tmp_path: Path):
    module = make_prim_module()
    from_data = fm.IRModule.from_data(module.to_data())
    checkpoint = fm.emit_module(module, tmp_path / "tir.py")
    loaded = fm.load_module(checkpoint)

    assert from_data.semantic_hash == module.semantic_hash
    assert loaded.semantic_hash == module.semantic_hash
    source = checkpoint.read_text(encoding="utf-8")
    assert "T.prim_function(" in source
    assert "T.buffer_store(" in source
    assert "T.mem_span(" in source


def test_python_round_trip_preserves_tuple_and_list_values_in_prim_function_attrs(tmp_path: Path):
    """TIR attrs are semantic data, so Python emission must not merge sequence kinds."""

    module = make_prim_module()
    function = replace(
        module.prim_functions[0],
        attrs={
            "buffer_layout_signature": (
                {
                    "parameter": "input",
                    "tuple_leaves": ("dense", "global"),
                    "list_leaves": ["dense", "global"],
                },
            ),
        },
    )
    module = replace(module, prim_functions=(function,))

    checkpoint = fm.emit_module(module, tmp_path / "tir_attrs.py")
    loaded = fm.load_module(checkpoint)

    assert loaded.semantic_hash == module.semantic_hash
    attrs = loaded.prim_functions[0].attrs
    assert isinstance(attrs["buffer_layout_signature"], tuple)
    assert isinstance(attrs["buffer_layout_signature"][0]["tuple_leaves"], tuple)
    assert isinstance(attrs["buffer_layout_signature"][0]["list_leaves"], list)


def test_script_printer_exposes_loop_block_buffer_and_memspan():
    source = fm.script_source(make_prim_module())

    assert 'T.PrimFunc("copy_4"' in source
    assert "for i in T.Range(0, 4, 1) [serial" in source
    assert "with T.Block('copy'" in source
    assert "T.MemSpan('output.storage'" in source


def test_tir_cost_visits_load_store_and_binary_nodes_once():
    function = make_prim_module().prim_function_map["copy_4"]
    cost = fm.estimate_tir_cost(function)

    assert cost.flops == 0
    assert cost.bytes_read == 4
    assert cost.bytes_written == 4
