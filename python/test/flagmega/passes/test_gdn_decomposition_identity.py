# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""A semantic decomposition preserves graphs with no applicable operator."""

from triton.flagmega import ir as fm
from triton.flagmega.passes.gated_delta_net import decompose_gated_delta_net


def test_gdn_decomposition_preserves_an_unrelated_weight_graph():
    builder = fm.IRBuilder(dialect="high_level", stage="imported")
    value_type = fm.tensor_type("float32", (16,))
    runtime = builder.var("runtime", value_type, id="runtime")
    weight = builder.weight("weight", value_type, source="memory", key="weight", id="weight")
    transformed = builder.call("math.silu", (weight,), value_type, id="transformed")
    output = builder.call("math.add", (runtime, transformed), value_type, id="output")
    builder.function("main", (runtime,), (output,))
    module = builder.build(entry="main")
    assert decompose_gated_delta_net(module) is module
