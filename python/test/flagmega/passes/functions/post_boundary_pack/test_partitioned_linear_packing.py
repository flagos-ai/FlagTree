# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Packed K inputs and shared output demands survive both graph paths."""

import pytest
import torch
from triton.flagmega import ir as fm
from triton.flagmega.evaluator import DictWeightResolver, TorchEvaluator
from triton.flagmega.passes.functions import post_function_boundary_pack_propagation
from triton.flagmega.targets import NvidiaSm90Target


@pytest.mark.parametrize("large", [False, True])
@pytest.mark.parametrize("accumulation_splits", [1, 2])
def test_shared_projection_uses_packed_input_and_result_without_duplication(large, accumulation_splits):
    class Graph(fm.Module):
        def forward(self):
            x = self.input("x", fm.tensor_type(fm.vector_type("bfloat16", (2, 4)), (1, 16)), id="x")
            w = self.input("w", fm.tensor_type("bfloat16", (80, 128)), id="w")
            value = fm.F.tensors.unpack(x, axes=(-1, -1))
            result = fm.F.nn.partitioned_linear(value, w, partitions=4,
                                                accumulation_splits=accumulation_splits, name="projection")
            outputs = [result, fm.F.tensors.pack(result, (2, 4), axes=(-1, -1)),
                       fm.F.tensors.pack(result, (4,), axes=(-1,))]
            if large:
                padding = w
                for _ in range(20_001):
                    padding = fm.F.math.add(padding, w)
                outputs.append(padding)
            self.function("main", (x, w), outputs)

    original = Graph(dialect="high_level", stage="packed", entry="main").build()
    result = post_function_boundary_pack_propagation(original, NvidiaSm90Target())
    projection, = (node for node in result.nodes if node.op == "nn.partitioned_linear")
    assert projection.attrs.get("accumulation_splits", 1) == accumulation_splits
    assert projection.attrs["output_lanes"] == (2, 4)
    assert isinstance(result.node_map[projection.inputs[0]].type.dtype, fm.VectorType)
    assert not any(node.op in {"tensors.pack", "tensors.unpack"} for node in result.nodes)
    assert post_function_boundary_pack_propagation(result, NvidiaSm90Target()).semantic_hash == result.semantic_hash
    if not large:
        generator = torch.Generator().manual_seed(18831)
        values = {node.id: torch.randn(tuple(d.fixed_value for d in node.type.shape)
                                      + tuple(getattr(node.type.dtype, "lanes", ())), generator=generator).bfloat16()
                  for node in original.nodes if node.op == "builtin.var"}
        evaluator = TorchEvaluator(DictWeightResolver({}))
        torch.testing.assert_close(evaluator.run(original, values), evaluator.run(result, values), rtol=0, atol=0)
