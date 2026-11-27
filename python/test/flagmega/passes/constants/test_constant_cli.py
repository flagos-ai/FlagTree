# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT

import json
from dataclasses import replace

from triton.flagmega import ir as fm
from triton.flagmega.cli import main
from triton.flagmega.passes import freeze_constant_islands


def _frozen_module(op: str):
    builder = fm.IRBuilder(dialect="high_level", stage="canonical_constants")
    value_type = fm.tensor_type("float32", (2, 8))
    runtime = builder.var("runtime", value_type, id="runtime")
    weight = builder.weight("weight", value_type, source="memory", key="weight", id="weight")
    if op == "math.silu":
        constant = builder.call(op, (weight,), value_type, id="constant")
    else:
        constant = builder.call(op, (weight, weight), value_type, id="constant")
    output = builder.call("math.add", (runtime, constant), value_type, id="output")
    builder.function("main", (runtime,), (output,))
    return replace(freeze_constant_islands(builder.build(entry="main")), stage="frozen_constants")


def test_cli_inspect_and_diff_report_constant_recipes(tmp_path, capsys):
    lhs_path = fm.emit_module(_frozen_module("math.silu"), tmp_path / "lhs.py")
    rhs_path = fm.emit_module(_frozen_module("math.mul"), tmp_path / "rhs.py")

    assert main(["inspect", str(lhs_path), "--json"]) == 0
    inspected = json.loads(capsys.readouterr().out)
    assert inspected["module"]["constant_phase"] == "frozen"
    assert inspected["module"]["constant_recipes"] == 1
    assert inspected["constant_recipes"][0]["outputs"] == ["constant"]

    assert main(["diff", str(lhs_path), str(rhs_path), "--json"]) == 0
    difference = json.loads(capsys.readouterr().out)
    assert difference["changed_nodes"] == []
    assert difference["changed_constant_recipes"] == ["constant_recipe_0000"]
