# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT

from dataclasses import replace

from triton.flagmega import ir as fm
from triton.flagmega.passes.norm_stats import finalize_norm_stats_bindings


def _module():
    class Graph(fm.Module):
        def __init__(self):
            super().__init__(dialect="nn", stage="distributed", entry="main")

        def forward(self):
            value = self.input("value", fm.tensor_type("float32", (2, 8)), id="value")
            stats = self.input("stats", fm.tensor_type("float32", (1, 2, 1)), id="stats")
            scale = self.input("scale", fm.tensor_type("float32", (8,)), id="scale")
            bias = self.input("bias", fm.tensor_type("float32", (8,)), id="bias")
            bound = fm.F.nn.bind_norm_stats(
                value, stats, axis=1, use_mean=False, name="bound")
            output = fm.F.nn.norm_apply(
                value, bound, scale, bias,
                axis=1, epsilon=1e-6, use_mean=False, name="output")
            self.function("main", (value, stats, scale, bias), (output,))

    module = Graph().build()
    point = fm.SelectionPoint(
        "distribution.bound",
        "distribution",
        (fm.Candidate("distribution.bound.broadcast"),),
        "distribution.bound.broadcast",
        owner="bound",
    )
    selection = fm.SelectionRecord(
        point.id,
        point.default_candidate,
        "ortools-cp-sat",
        "unit",
        "unit binding selection",
    )
    return replace(module, selection_points=(point,), selections=(selection,))


def test_finalize_redirects_bind_to_stats_and_removes_its_selection_boundary():
    result = finalize_norm_stats_bindings(_module())

    assert "bound" not in result.node_map
    assert result.node_map["output"].inputs[1] == "stats"
    assert not result.selection_points
    assert not result.selections
