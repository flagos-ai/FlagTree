# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Pure projections may precede a single explicit state-owning stage."""

from dataclasses import replace

import pytest

from triton.flagmega import ir as fm
from triton.flagmega.errors import IRVerificationError
from triton.flagmega.passes.functions import outline_function_stages
from python.test.flagmega.passes.functions.test_outline_stages import STAGES, reusable_graph


def effect_graph():
    source = reusable_graph(effectful=True)
    value = source.node_map["w_left0"]
    formal = fm.Node("leaf_x", "builtin.var", (), source.node_map["w_prefix"].type, attrs={"name": "leaf_x"})
    body = replace(value, id="leaf_result", inputs=(formal.id, "wa"))
    return fm.verify_module(replace(source, nodes=(*tuple(
        replace(n, op="builtin.call", effect=fm.effect("write", "state"), attrs={"callee": "leaf"})
        if n.id == value.id else n for n in source.nodes), formal, body),
        functions=(*source.functions, fm.Function("leaf", (formal.id, "wa"), (body.id,)))))


def test_explicit_effect_stage_keeps_effects_on_the_state_owner():
    source = effect_graph()
    result = outline_function_stages(source, "worker", STAGES, effect_stage=1)
    calls = [n for n in result.nodes if n.op == "builtin.call" and n.attrs["callee"].startswith("worker.")]
    assert [(n.id, n.effect) for n in calls if not n.effect.is_pure] == [
        (f"call{i}.stage1", fm.effect("read_write", "state")) for i in (0, 1)]
    assert result.node_map["w_left0"].effect == source.node_map["w_left0"].effect
    assert result.metadata["outlined_function_stages"][-1]["effect_stage"] == 1


@pytest.mark.parametrize("effect_stage", [True, -1, 6, "1"])
def test_invalid_effect_stage_is_rejected(effect_stage):
    with pytest.raises(IRVerificationError, match="effect stage"):
        outline_function_stages(effect_graph(), "worker", STAGES, effect_stage=effect_stage)


def test_effect_work_cannot_leak_into_a_pure_stage():
    with pytest.raises(IRVerificationError, match="stateful/reference"):
        outline_function_stages(effect_graph(), "worker", STAGES, effect_stage=2)


def split_effect_graph():
    source = effect_graph()
    return fm.verify_module(replace(source, nodes=tuple(
        replace(n, op="builtin.call", effect=fm.effect("write", "state"), attrs={"callee": "leaf"})
        if n.id == "w_right0" else n for n in source.nodes)))


def test_explicit_sequential_effect_stages_keep_each_body_once():
    source = split_effect_graph()
    result = outline_function_stages(source, "worker", STAGES, effect_stages=(1, 3))
    calls = [n for n in result.nodes if n.op == "builtin.call" and n.attrs["callee"].startswith("worker.")]
    assert [n.id for n in calls if not n.effect.is_pure] == [
        f"call{i}.stage{stage}" for i in (0, 1) for stage in (1, 3)]
    assert sum(n.id == "w_left0" for n in result.nodes) == 1
    assert sum(n.id == "w_right0" for n in result.nodes) == 1
    assert result.metadata["outlined_function_stages"][-1]["effect_stages"] == (1, 3)


def test_sequential_effect_stages_cannot_reorder_mutations():
    stages = (STAGES[0], STAGES[3], STAGES[4], STAGES[1], STAGES[2], STAGES[5])
    with pytest.raises(IRVerificationError, match="stateful.*order"):
        outline_function_stages(split_effect_graph(), "worker", stages, effect_stages=(1, 3))


@pytest.mark.parametrize("stages", [(), (True,), (-1,), (6,), (3, 1), (1, 1), "1"])
def test_invalid_sequential_effect_stages_are_rejected(stages):
    with pytest.raises(IRVerificationError, match="effect stages"):
        outline_function_stages(split_effect_graph(), "worker", STAGES, effect_stages=stages)
