# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT

from dataclasses import replace

import pytest

from triton.flagmega import ir as fm
from triton.flagmega.ir.ops.core import get_definition
from triton.flagmega.ir.printer import il_source, script_source
from triton.flagmega.passes import freeze_constant_islands
from .test_weights_section import weight_chain_module


@pytest.mark.parametrize("render", (il_source, script_source))
@pytest.mark.parametrize("guard", ("const_evaluable", "deterministic", "effect"))
def test_weight_input_alone_is_not_enough_to_move_an_operation(render, guard, monkeypatch):
    module = weight_chain_module()
    if guard == "effect":
        # Exercise diagnostic printing before verification of an edited IR.
        module = replace(
            module, nodes=tuple(
                replace(node, effect=fm.Effect(fm.EffectKind.READ, "state")) if node.id == "weight_silu" else node
                for node in module.nodes))
    else:
        monkeypatch.setattr(get_definition("math.silu"), guard, False)
    weights, compute = render(module).split("  // compute\n")
    assert "name='weight_add'" in weights
    assert "name='weight_silu'" not in weights and "name='weight_silu'" in compute
    if guard == "effect":
        assert "!read<state>" in compute


@pytest.mark.parametrize("render", (il_source, script_source))
def test_frozen_recipes_are_partitioned_by_function_not_repeated_globally(render):
    builder = fm.IRBuilder(dialect="high_level", stage="imported")
    tensor = fm.tensor_type("float32", (4, ))
    for name in ("main", "helper"):
        weight = builder.weight(name, tensor, source="missing", key=name, id=name + "_weight")
        result = builder.call("math.silu", (weight, ), tensor, id=name + "_transform")
        builder.function(name, (), (result, ))
    module = freeze_constant_islands(fm.verify_module(builder.build(entry="main")))
    source = render(module)
    first, second = source.split("  weights {\n")[1:]
    assert "name='main_transform'" in first and "name='helper_transform'" not in first
    assert "name='helper_transform'" in second and "name='main_transform'" not in second
    assert source.count("fingerprint=") == len(module.constant_recipes) == 2


def test_physical_tir_calls_and_mutable_storage_stay_in_execution_body():
    # A small diagnostic TIR fragment, not a complete executable PrimFunction.
    builder = fm.IRBuilder(dialect="semantic_tir", stage="unit")
    tensor = fm.tensor_type("float32", (4, ))
    readonly = builder.node(op="tir.buffer", type=tensor, attrs={"storage": "rdata", "key": "w"}, id="readonly")
    workspace = builder.node(op="tir.buffer", type=tensor, attrs={"storage": "workspace"}, id="workspace")
    call = builder.call("tir.call", (readonly, ), tensor, attrs={"callee": "physical"}, id="dispatch")
    builder.function("main", (), (call, workspace))
    source = script_source(builder.build(entry="main"))
    assert "weights {" not in source
    assert "T.Call(@physical," in source and "name='dispatch'" in source and "Name: 'workspace'" in source
