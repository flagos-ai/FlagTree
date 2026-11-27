# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT

from __future__ import annotations

import inspect
from dataclasses import replace

import pytest

from triton.flagmega import ir as fm
from triton.flagmega.compiler import Compiler
from triton.flagmega.diagnostics import DumpFlags, DumpManager
from triton.flagmega.egraph import EGraph
from triton.flagmega.errors import IRVerificationError


def make_unary_module(*, stage="imported"):
    builder = fm.IRBuilder(dialect="high_level", stage=stage, metadata={"model": "unit"})
    tensor = fm.tensor_type("bfloat16", [1, 128])
    source = builder.var("source", tensor, id="source")
    result = builder.call("math.silu", [source], tensor, id="result")
    builder.function("main", [source], [result])
    return builder.build(entry="main")


def test_python_ir_round_trip_is_semantically_stable(tmp_path):
    module = make_unary_module()
    path = fm.emit_module(module, tmp_path / "imported.py")

    loaded = fm.load_module(path, expected_stage="imported", expected_dialect="high_level")

    assert loaded == module
    assert loaded.semantic_hash == module.semantic_hash
    assert fm.module_source(loaded) == path.read_text(encoding="utf-8")
    assert path.with_suffix(".il").is_file()
    il = path.with_suffix(".il").read_text(encoding="utf-8")
    assert "%main = fn(" in il
    assert "Math.Silu(" in il


def test_python_ir_is_real_module_builder_source_not_a_serialized_spec():
    source = fm.module_source(make_unary_module())

    assert "class Graph(fm.Module):" in source
    assert "def forward(self) -> None:" in source
    assert "source = self.input(" in source
    assert "result = F.math.silu(" in source
    assert "self.function(" in source
    assert "self.call(" not in source
    assert "SPEC =" not in source
    assert "IRModule.from_data" not in source


def test_op_decorator_generates_functional_visit_and_cost_contracts():
    class Graph(fm.Module):
        def __init__(self):
            super().__init__(dialect="high_level", stage="imported", entry="main")

        def forward(self):
            source = self.input("source", fm.tensor_type("bfloat16", [1, 16]), id="source")
            doubled = fm.F.math.add(lhs=source, rhs=source, name="doubled")
            output = fm.F.math.silu(doubled, name="output")
            self.function("main", [source], [output])

    module = Graph().build()
    output = module.node_map["output"]
    definitions = fm.definitions()

    assert output.op == "math.silu"
    assert output.type == module.node_map["source"].type
    assert module.node_map["doubled"].inputs == ("source", "source")
    assert fm.F.math.silu.__flagmega_op_definition__ is fm.get_definition("math.silu")
    assert tuple(inspect.signature(fm.F.math.silu).parameters) == ("value", "name", "metadata")
    assert tuple(parameter.name for parameter in fm.get_definition("math.silu").parameters) == ("value", )
    assert not hasattr(fm.get_definition("math.silu"), "minimum_inputs")
    assert fm.get_definition("math.silu").__module__.endswith(".ir.ops.math.silu")
    assert len({definition.__module__ for definition in definitions}) == len(definitions)
    assert all(".ir.ops." in definition.__module__ for definition in definitions)
    assert fm.get_cost(output).bytes_written == 32

    class Visitor:
        def visit_math_silu(self, node):
            return f"visited:{node.id}"

    assert fm.visit_node(output, Visitor()) == "visited:output"

    recurrent = fm.get_definition("nn.gdn_recurrent_core")
    assert recurrent.state.memory_effect == fm.MemoryEffect.READ_WRITE
    assert tuple(inspect.signature(fm.F.nn.gated_delta_net_recurrent_core).parameters) == (
        *(parameter.name for parameter in recurrent.parameters),
        "name",
        "metadata",
    )


def test_tir_companion_uses_nncase_style_script():
    module = make_unary_module()
    compiled = Compiler().compile(module).module
    script = fm.script_source(compiled)

    assert 'T.PrimFunc("main"' in script
    assert "T.Kernel(" in script
    assert "T.Return(" in script


def test_pass_dump_before_after_are_directories_with_one_file_per_function(tmp_path):
    builder = fm.IRBuilder(dialect="high_level", stage="imported")
    tensor = fm.tensor_type("bfloat16", [1, 8])
    main_input = builder.var("main_input", tensor, id="main_input")
    main_output = builder.call("math.silu", [main_input], tensor, id="main_output")
    helper_input = builder.var("helper_input", tensor, id="helper_input")
    helper_output = builder.call("math.silu", [helper_input], tensor, id="helper_output")
    builder.function("main", [main_input], [main_output])
    builder.function("helper", [helper_input], [helper_output])
    module = builder.build(entry="main")
    dumper = DumpManager(tmp_path, DumpFlags.PASS_IR).root

    emitted = dumper.dump_module(module, "Before", category=DumpFlags.PASS_IR)

    assert emitted is not None
    assert emitted.directory == tmp_path / "Before"
    assert {item.function for item in emitted.functions} == {"main", "helper"}
    for name in ("main", "helper"):
        assert (tmp_path / "Before" / f"{name}.py").is_file()
        assert (tmp_path / "Before" / f"{name}.il").is_file()
        view = fm.load_module(tmp_path / "Before" / f"{name}.py")
        assert tuple(view.function_map) == (name, )
        assert {node.id for node in view.nodes} == {f"{name}_input", f"{name}_output"}
        source = (tmp_path / "Before" / f"{name}.py").read_text(encoding="utf-8")
        assert "DUMP_INFO = fm.FunctionDumpInfo(" in source

    merged = fm.load_module(tmp_path / "Before")

    assert merged == module
    assert merged.semantic_hash == module.semantic_hash
    resumed = Compiler().compile_checkpoint(tmp_path / "Before")
    # The directory checkpoint faithfully reconstructs both functions; the
    # compiler then mirrors nncase RemoveUnusedFunctions before distribution.
    assert tuple(resumed.module.function_map) == ("main",)
    assert resumed.module.stage == "bufferized_tir"


def test_python_ir_is_editable_and_reverified(tmp_path):
    module = make_unary_module()
    path = fm.emit_module(module, tmp_path / "imported.py")
    source = path.read_text(encoding="utf-8")
    source = source.replace("'model': 'unit'", "'model': 'agent-edited'")
    path.write_text(source, encoding="utf-8")

    edited = fm.load_module(path)

    assert edited.metadata["model"] == "agent-edited"
    assert edited.semantic_hash != module.semantic_hash


def test_verifier_rejects_non_topological_input():
    module = make_unary_module()
    broken = replace(module, nodes=tuple(reversed(module.nodes)))

    with pytest.raises(IRVerificationError, match="non-topological"):
        fm.verify_module(broken)


def test_egraph_keeps_effectful_state_as_opaque_boundary():
    builder = fm.IRBuilder(dialect="high_level", stage="imported")
    tensor = fm.tensor_type("bfloat16", [1, 8])
    state_type = fm.RefType("state", (("data", tensor), ))
    source = builder.var("source", tensor, id="source")
    state = builder.var("state", state_type, id="state")
    stateful_type = fm.TupleType((tensor, state_type))
    stateful = builder.call(
        "tir.kernel",
        [state, source],
        stateful_type,
        id="stateful",
        effect=fm.effect("read_write", "gdn_state"),
        attrs={
            "semantic_op": "test.effectful_boundary",
            "candidate": "reference",
            "parameters": {},
            "facts": {},
            "semantic_attrs": {},
        },
    )
    value = builder.call("builtin.get_item", [stateful], tensor, id="value", attrs={"index": 0})
    output = builder.call("math.silu", [value], tensor, id="output")
    builder.function("main", [source, state], [output])
    module = builder.build(entry="main")
    fm.verify_module(module)

    graph = EGraph()
    graph.add_module(module)

    assert "stateful" in {node.source_node for value in graph.classes() for node in value.nodes}
    assert all(node.op != "tir.kernel" for value in graph.classes() for node in value.nodes)


def test_egraph_refuses_union_of_different_types():
    builder = fm.IRBuilder(dialect="high_level", stage="imported")
    lhs = builder.var("lhs", fm.tensor_type("bfloat16", [1]), id="lhs")
    rhs = builder.var("rhs", fm.tensor_type("float32", [1]), id="rhs")
    builder.function("main", [lhs, rhs], [lhs])
    module = builder.build(entry="main")
    graph = EGraph()
    classes = graph.add_module(module)

    with pytest.raises(IRVerificationError, match="different IR types"):
        graph.union(classes["lhs"], classes["rhs"])
