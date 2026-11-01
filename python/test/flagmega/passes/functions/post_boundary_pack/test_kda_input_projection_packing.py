# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""One producer satisfies different low-rank consumers through both graph paths."""

import pytest
import torch
from triton.flagmega import ir as fm
from triton.flagmega.evaluator import DictWeightResolver, TorchEvaluator
from triton.flagmega.passes.functions import post_function_boundary_pack_propagation
from triton.flagmega.targets import NvidiaSm90Target


def consumers(large=False):
    class Graph(fm.Module):
        def forward(self):
            x = self.input("x", fm.tensor_type(fm.vector_type("bfloat16", (2, 4)), (1, 8)), id="x")
            source = fm.F.tensors.unpack(x, axes=(-1, -1))
            weights = tuple(self.input(name, fm.tensor_type("bfloat16", shape), id=name)
                            for name, shape in zip(("q", "k", "v", "b", "f", "g"),
                                                   ((8, 16, 64),) * 3 + ((8, 64), (32, 64), (64, 64))))
            projection = fm.F.nn.kda_input_projection(source, *weights, name="projection")
            fields = fm.F.tensors.get_items(projection, *range(6))
            outputs = [projection, *fields, fm.F.tensors.pack(fields[4], (2, 4), axes=(-1, -1)),
                       fm.F.tensors.pack(fields[5], (4,), axes=(-1,)),
                       fm.F.tensors.pack(fields[4], (4,), axes=(-1,))]
            if large:
                padding = weights[-1]
                for _ in range(20_001):
                    padding = fm.F.math.add(padding, weights[-1])
                outputs.append(padding)
            self.function("main", (x, *weights), outputs)

    return Graph(dialect="high_level", stage="packed", entry="main").build()


@pytest.mark.parametrize("large", [False, True])
def test_per_field_demands_preserve_shared_scalar_and_tuple_users(large):
    original = consumers(large)
    target = NvidiaSm90Target()
    rewritten = post_function_boundary_pack_propagation(original, target)
    projection, = (node for node in rewritten.nodes if node.op == "nn.kda_input_projection")
    assert projection.attrs["output_lanes"] == ((), (), (), (), (2, 4), (4,))
    assert isinstance(rewritten.node_map[projection.inputs[0]].type.dtype, fm.VectorType)
    assert not any(node.op == "tensors.pack" for node in rewritten.nodes)
    assert post_function_boundary_pack_propagation(rewritten, target).semantic_hash == rewritten.semantic_hash
    before = tuple(original.node_map[value].type for value in original.functions[0].outputs)
    after = tuple(rewritten.node_map[value].type for value in rewritten.functions[0].outputs)
    assert before == after
    if not large:
        generator = torch.Generator().manual_seed(1231)
        values = {node.id: torch.randn(tuple(d.fixed_value for d in node.type.shape) +
                                      tuple(getattr(node.type.dtype, "lanes", ())), generator=generator).bfloat16()
                  for node in original.nodes if node.op == "builtin.var"}
        evaluator = TorchEvaluator(DictWeightResolver({}))
        torch.testing.assert_close(evaluator.run(original, values), evaluator.run(rewritten, values), rtol=0, atol=0)
