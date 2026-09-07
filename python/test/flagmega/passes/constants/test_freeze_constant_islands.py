# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT

from dataclasses import replace

import pytest
import torch

from triton.flagmega import ir as fm
from triton.flagmega.compiler import Compiler
from triton.flagmega.diagnostics import DumpFlags, DumpManager
from triton.flagmega.evaluator import DictWeightResolver, materialize_constant_assets
from triton.flagmega.errors import StageError
from triton.flagmega.passes import DataflowPass, EGraphRulesPass, freeze_constant_islands
from triton.flagmega.rules import RewriteRule


def _shared_island_module():
    builder = fm.IRBuilder(dialect="high_level", stage="canonical_constants")
    value_type = fm.tensor_type("float32", (2, 8))
    runtime = builder.var("runtime", value_type, id="runtime")
    weight = builder.weight("weight", value_type, source="memory", key="weight", id="weight")
    weight_silu = builder.call("math.silu", (weight,), value_type, id="weight_silu")
    weight_square = builder.call("math.mul", (weight, weight_silu), value_type, id="weight_square")
    first = builder.call("math.add", (runtime, weight_silu), value_type, id="first")
    output = builder.call("math.add", (first, weight_square), value_type, id="output")
    builder.function("main", (runtime,), (output,))
    return builder.build(entry="main")


def test_freeze_outlines_one_maximal_island_with_multiple_assets():
    frozen = freeze_constant_islands(_shared_island_module())

    assert frozen.metadata["constant_phase"] == "frozen"
    assert {node.id for node in frozen.nodes if node.op == "builtin.const_asset"} == {
        "weight_silu", "weight_square",
    }
    assert "weight" not in frozen.node_map
    assert len(frozen.constant_recipes) == 1
    recipe = frozen.constant_recipes[0]
    assert recipe.outputs == ("weight_silu", "weight_square")
    assert [node.op for node in recipe.nodes] == ["builtin.weight", "math.silu", "math.mul"]
    assert frozen.node_map["first"].inputs == ("runtime", "weight_silu")
    assert frozen.node_map["output"].inputs == ("first", "weight_square")


def test_freeze_absorbs_materialized_constant_decisions_into_the_editable_recipe():
    module = _shared_island_module()
    nodes = tuple(
        replace(node, metadata={**dict(node.metadata), "realized_choice": node.id})
        if node.id in {"weight_silu", "weight_square"}
        else node
        for node in module.nodes
    )
    candidates = (fm.Candidate("scalar"), fm.Candidate("selected"))
    points = (
        fm.SelectionPoint(
            "vectorization.weight_silu",
            "vectorization",
            candidates,
            "scalar",
            owner="weight_silu",
        ),
        fm.SelectionPoint(
            "distribution.weight_square",
            "distribution",
            candidates,
            "scalar",
            owner="weight_square",
        ),
    )
    selections = tuple(
        fm.SelectionRecord(
            point.id,
            "selected",
            "unit-test",
            "unit-test",
            evidence=("choice already materialized in recipe IR",),
        )
        for point in points
    )

    frozen = freeze_constant_islands(replace(
        module,
        nodes=nodes,
        selection_points=points,
        selections=selections,
    ))

    assert frozen.selection_points == ()
    assert frozen.selections == ()
    recipe = frozen.constant_recipes[0]
    assert recipe.node_map["weight_silu"].metadata["realized_choice"] == "weight_silu"
    assert recipe.node_map["weight_square"].metadata["realized_choice"] == "weight_square"


def test_frozen_distributed_constant_pad_materializes_the_logical_tensor():
    builder = fm.IRBuilder(dialect="high_level", stage="canonical_constants")
    placement = fm.Placement((2, 2), "yx", "bb")
    logical = fm.tensor_type("bfloat16", (2, 3))
    distributed = fm.DistributedType(
        logical,
        (fm.SBP.broadcast(), fm.SBP.broadcast()),
        placement,
    )
    padded_type = fm.DistributedType(
        fm.tensor_type("bfloat16", (2, 4)),
        (fm.SBP.broadcast(), fm.SBP.broadcast()),
        placement,
    )
    weight = builder.weight(
        "weight", distributed, source="memory", key="weight", id="weight")
    padded = builder.call(
        "tensors.pad",
        (weight,),
        padded_type,
        attrs={"pad_end": (0, 1), "pad_value": 0.0},
        id="padded",
    )
    builder.function("main", (), (padded,))
    frozen = freeze_constant_islands(builder.build(entry="main"))

    source = torch.arange(6, dtype=torch.bfloat16).reshape(2, 3)
    values = materialize_constant_assets(
        frozen, DictWeightResolver({"weight": source}))

    torch.testing.assert_close(
        values["padded"],
        torch.nn.functional.pad(source, (0, 1)),
        rtol=0,
        atol=0,
    )


def test_dataflow_and_egraph_see_normal_ops_before_freeze_and_reject_after_it():
    module = _shared_island_module()
    commute = RewriteRule(
        "CommuteWeightSquare",
        lambda node, _module: node.id == "weight_square" and "commuted" not in node.metadata,
        lambda node, _module: replace(
            node, inputs=tuple(reversed(node.inputs)), metadata={"commuted": True}),
    )
    dataflow = DataflowPass("Dataflow", (commute,)).run(module)
    assert dataflow.node_map["weight_square"].inputs == ("weight_silu", "weight")
    egraph = EGraphRulesPass(
        "EGraph",
        (commute,),
        selector=lambda _original, alternatives, _module: alternatives[0].node,
    ).run(module)
    assert egraph.node_map["weight_square"].inputs == ("weight_silu", "weight")

    frozen = freeze_constant_islands(module)
    with pytest.raises(StageError, match="constants_open"):
        DataflowPass("TooLate", ()).run(frozen)
    with pytest.raises(StageError, match="constants_open"):
        EGraphRulesPass("TooLate", ()).run(frozen)


def test_frozen_python_ir_is_editable_and_recomputes_recipe_fingerprint(tmp_path):
    frozen = replace(freeze_constant_islands(_shared_island_module()), stage="frozen_constants")
    path = fm.emit_module(frozen, tmp_path / "frozen.py")
    source = path.read_text(encoding="utf-8")

    assert "class ConstantRecipes(fm.ConstantModule):" in source
    assert "weight_silu = F.math.silu(" in source
    assert "F.builtin.const_asset(" in source
    assert "self.call(" not in source
    assert fm.load_module(path) == frozen

    old_fingerprint = frozen.constant_recipes[0].fingerprint
    old_call = "weight_silu = F.math.silu(\n            weight,"
    new_call = "weight_silu = F.math.mul(\n            weight,\n            weight,"
    path.write_text(
        source.replace(old_call, new_call),
        encoding="utf-8",
    )
    edited = fm.load_module(path)
    assert edited.constant_recipes[0].node_map["weight_silu"].op == "math.mul"
    assert edited.constant_recipes[0].fingerprint != old_fingerprint
    assert Compiler().compile(edited).module.stage == "bufferized_tir"


def test_multi_function_frozen_dump_remains_mergeable(tmp_path):
    builder = fm.IRBuilder(dialect="high_level", stage="canonical_constants")
    value_type = fm.tensor_type("float32", (2, 8))
    main_input = builder.var("main_input", value_type, id="main_input")
    main_weight = builder.weight("main_weight", value_type, source="memory", key="main_weight", id="main_weight")
    main_output = builder.call("math.add", (main_input, main_weight), value_type, id="main_output")
    helper_input = builder.var("helper_input", value_type, id="helper_input")
    helper_weight = builder.weight(
        "helper_weight", value_type, source="memory", key="helper_weight", id="helper_weight")
    helper_output = builder.call("math.add", (helper_input, helper_weight), value_type, id="helper_output")
    builder.function("main", (main_input,), (main_output,))
    builder.function("helper", (helper_input,), (helper_output,))
    frozen = freeze_constant_islands(builder.build(entry="main"))
    dumper = DumpManager(tmp_path, DumpFlags.PASS_IR).root

    emitted = dumper.dump_module(frozen, "Frozen", category=DumpFlags.PASS_IR)

    assert emitted is not None
    for name in ("main", "helper"):
        view = fm.load_module(tmp_path / "Frozen" / f"{name}.py")
        assert len(view.constant_recipes) == 2
        assert {node.id for node in view.nodes if node.op == "builtin.const_asset"} == {
            "main_weight", "helper_weight",
        }
    assert fm.load_module(tmp_path / "Frozen") == frozen
