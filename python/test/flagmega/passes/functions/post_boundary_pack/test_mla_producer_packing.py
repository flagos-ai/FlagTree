# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Both propagation paths move final-axis demand into one MLA projection."""

from dataclasses import replace

import pytest
import torch
from triton.flagmega import ir as fm
from triton.flagmega.evaluator import DictWeightResolver, TorchEvaluator
from triton.flagmega.passes.functions import post_function_boundary_pack_propagation
from triton.flagmega.passes.functions.post_boundary_pack import _representation_cost
from triton.flagmega.targets import NvidiaSm90Target


@pytest.mark.parametrize("large", [False, True])
@pytest.mark.parametrize("existing_lanes, demand", [((), (2, 4)), ((2,), (4,))])
def test_shared_projection_packing_preserves_scalar_results_and_fixed_point(large, existing_lanes, demand):
    class Graph(fm.Module):
        def forward(self):
            x = self.input("x", fm.tensor_type("bfloat16", (1, 2, 16)), id="x")
            w = self.input("w", fm.tensor_type("bfloat16", (48, 32)), id="w")
            projection = fm.F.nn.mla_projection(
                x,
                w,
                fm.F.builtin.none(),
                direction="absorb_query",
                qk_head_dim=16,
                value_head_dim=8,
                latent_dim=32,
                output_lanes=existing_lanes,
            )
            packed = fm.F.tensors.pack(projection, demand, axes=(-1,) * len(demand))
            outputs = [packed, projection]
            if large:
                padding = x
                for _ in range(20_001):
                    padding = fm.F.math.add(padding, x)
                outputs.append(padding)
            self.function("main", (x, w), outputs)

    source = Graph(dialect="high_level", stage="packed", entry="main").build()
    if large:
        assert len(source.nodes) > 20_000
    result = post_function_boundary_pack_propagation(source, NvidiaSm90Target())
    (producer,) = (n for n in result.nodes if n.op == "nn.mla_projection")
    assert producer.attrs["output_lanes"] == (*demand, *existing_lanes)
    assert not any(n.op == "tensors.pack" for n in result.nodes)
    assert post_function_boundary_pack_propagation(result, NvidiaSm90Target()).semantic_hash == result.semantic_hash
    (scalar,) = (n for n in source.nodes if n.op == "nn.mla_projection")
    if not existing_lanes:
        assert _representation_cost(producer, result) < _representation_cost(scalar, source)
    if large:
        source = replace(source, functions=(replace(source.functions[0], outputs=source.functions[0].outputs[:-1]),))
        result = replace(result, functions=(replace(result.functions[0], outputs=result.functions[0].outputs[:-1]),))
    values = {
        "x": torch.arange(32).bfloat16().reshape(1, 2, 16) / 16,
        "w": (torch.arange(48 * 32) % 17).bfloat16().reshape(48, 32) / 16,
    }
    evaluator = TorchEvaluator(DictWeightResolver({}))
    torch.testing.assert_close(evaluator.run(source, values), evaluator.run(result, values), rtol=0, atol=0)
