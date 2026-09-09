# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT

from dataclasses import replace
import torch

from triton.flagmega import ir as fm
from triton.flagmega.evaluator import DictWeightResolver, TorchEvaluator
from triton.flagmega.passes.functions import lift_constant_parameter_expressions as lift
from .helpers import module


def test_caller_preceding_callee_moves_only_its_late_constant_dependency():
    original = module()
    literal = fm.Node("late_one", "builtin.splat_const", (), original.node_map["runtime"].type, attrs={"value": 1.0})
    body = {"p", "x", "cast", "transformed", "output"}
    nodes = [n for n in original.nodes if n.id not in body]
    nodes.append(literal)
    nodes.extend(
        replace(n, inputs=("cast", "late_one")) if n.id == "transformed" else n for n in original.nodes if n.id in body)
    source = fm.verify_module(replace(original, nodes=tuple(nodes)))
    before = source.semantic_hash
    lifted = lift(source)
    assert source.semantic_hash == before
    assert "cast" not in lifted.node_map
    order = {n.id: i for i, n in enumerate(lifted.nodes)}
    assert order["late_one"] < order["call0"]
    assert order["call0"] < order["call1"]
    weights = {f"weight{i}": torch.arange(8, dtype=torch.bfloat16) + i for i in range(2)}
    evaluator = TorchEvaluator(DictWeightResolver(weights))
    values = {"runtime": torch.linspace(-1, 1, 8)}
    torch.testing.assert_close(evaluator.run(lifted, values), evaluator.run(source, values), rtol=0, atol=0)
