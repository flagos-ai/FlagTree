# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Propagate complete lanes into one shared material-partition producer."""

import pytest
import torch
from triton.flagmega import ir as fm
from triton.flagmega.evaluator import DictWeightResolver, TorchEvaluator
from triton.flagmega.passes.functions import post_function_boundary_pack_propagation
from triton.flagmega.targets import NvidiaSm90Target


@pytest.mark.parametrize("large", (False, True))
def test_shared_partitioned_projection_reconciles_input_and_output_lanes(large):
    class Graph(fm.Module):
        def forward(self):
            x = self.input("x", fm.tensor_type(fm.vector_type("bfloat16", (2, 4)), (1, 128)), id="x")
            w = self.input("weight", fm.tensor_type("float8_e4m3fn", (80, 1024)), id="weight")
            scales = self.input("scales", fm.tensor_type("float32", (1, 8)), id="scales")
            scalar = fm.F.tensors.unpack(x, axes=(-1, -1))
            result = fm.F.nn.partitioned_block_scaled_linear(scalar, w, scales, partitions=4, name="projection")
            outputs = [result, fm.F.tensors.pack(result, (2, 4), axes=(-1, -1)),
                       fm.F.tensors.pack(result, (4,), axes=(-1,))]
            if large:
                padding = scales
                for _ in range(20_001):
                    padding = fm.F.math.add(padding, scales)
                outputs.append(padding)
            self.function("main", (x, w, scales), outputs)

    original = Graph(dialect="high_level", stage="packed", entry="main").build()
    target = NvidiaSm90Target()
    result = post_function_boundary_pack_propagation(original, target)
    producer, = (node for node in result.nodes if node.op == "nn.partitioned_block_scaled_linear")
    assert producer.attrs["output_lanes"] == (2, 4)
    assert isinstance(result.node_map[producer.inputs[0]].type.dtype, fm.VectorType)
    assert not any(node.op in {"tensors.pack", "tensors.unpack"} for node in result.nodes)
    assert post_function_boundary_pack_propagation(result, target).semantic_hash == result.semantic_hash
    if not large:
        generator = torch.Generator().manual_seed(98129)
        values = {"x": torch.randn((1, 128, 2, 4), generator=generator).bfloat16(),
                  "weight": torch.randn((80, 1024), generator=generator).to(torch.float8_e4m3fn),
                  "scales": torch.linspace(.01, .08, 8).reshape(1, 8)}
        evaluator = TorchEvaluator(DictWeightResolver({}))
        torch.testing.assert_close(evaluator.run(original, values), evaluator.run(result, values), rtol=0, atol=0)
