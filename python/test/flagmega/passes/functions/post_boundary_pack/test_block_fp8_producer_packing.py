# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""FP8 projections must produce the layout demanded by norm and query users."""

from math import ceil, prod

import pytest
import torch

from triton.flagmega import ir as fm
from triton.flagmega.evaluator import DictWeightResolver, TorchEvaluator
from triton.flagmega.passes.functions import post_function_boundary_pack_propagation
from triton.flagmega.targets import NvidiaSm90Target


class Fp8Consumers(fm.Module):
    def __init__(self, *, packed_weights=True, lanes=(8,), output_lanes=(),
                 scalar_users=False, mixed_users=False, large=False, n=40, k=128,
                 dtype="bfloat16"):
        super().__init__(dialect="high_level", stage="packed", entry="main")
        self.packed_weights, self.lanes, self.output_lanes = packed_weights, lanes, output_lanes
        self.scalar_users, self.mixed_users, self.large = scalar_users, mixed_users, large
        self.n, self.k, self.dtype = n, k, dtype

    def forward(self):
        value = self.input("value", fm.tensor_type(self.dtype, (1, self.k)), id="value")
        weight_type = (fm.tensor_type(fm.vector_type("float8_e4m3fn", (2, 16)), (self.n, self.k // 32))
                       if self.packed_weights else fm.tensor_type("float8_e4m3fn", (self.n, self.k)))
        weight = self.weight("weight", weight_type, source="weights", key="weight", id="weight")
        scale = self.weight("weight_scale", fm.tensor_type("float32", (ceil(self.n / 16), ceil(self.k / 128))),
                            source="weights", key="weight_scale", id="weight_scale")
        kwargs = {"weight_block_n": 16, "weight_block_k": 128, "name": "projection"}
        if self.output_lanes:
            kwargs["output_lanes"] = self.output_lanes
        if self.packed_weights:
            projection = fm.F.math.packed_block_scaled_matmul(value, weight, scale, k_pack=2, k_vector=16, **kwargs)
        else:
            projection = fm.F.math.block_scaled_matmul(value, weight, scale, **kwargs)
        outputs = [fm.F.tensors.pack(projection, self.lanes, axes=(-1,) * len(self.lanes), name="norm_input")]
        # A second Pack represents a distinct norm consumer before folding.
        stats_input = fm.F.tensors.pack(projection, self.lanes, axes=(-1,) * len(self.lanes))
        outputs.append(fm.F.nn.norm_stats(stats_input, axis=-1, use_mean=False))
        if self.scalar_users:
            outputs.append(projection)
        if self.mixed_users:
            outputs.append(fm.F.tensors.pack(projection, (4,), axes=(-1,)))
        if self.large:
            padding = value
            for _ in range(20_001):
                padding = fm.F.math.add(padding, value)
            outputs.append(padding)
        self.function("main", (value,), tuple(outputs))


def operands(*, packed_weights=True, n=40, k=128, dtype="bfloat16"):
    generator = torch.Generator().manual_seed(23)
    value = torch.randn(1, k, generator=generator).to(getattr(torch, dtype))
    weight = torch.randn(n, k, generator=generator).to(torch.float8_e4m3fn)
    if packed_weights:
        weight = weight.reshape(n, k // 32, 2, 16)
    scales = torch.rand(ceil(n / 16), ceil(k / 128), generator=generator) + .25
    return {"value": value}, {"weight": weight, "weight_scale": scales}


@pytest.mark.parametrize("packed_weights", [False, True])
@pytest.mark.parametrize("lanes", [(8,), (2, 4)])
@pytest.mark.parametrize("scalar_users", [False, True])
def test_fp8_matmul_produces_shared_packed_output_without_activation_pack(packed_weights, lanes, scalar_users):
    original = Fp8Consumers(packed_weights=packed_weights, lanes=lanes, scalar_users=scalar_users).build()
    rewritten = post_function_boundary_pack_propagation(original, NvidiaSm90Target())
    projection, = (node for node in rewritten.nodes if node.op in
                   {"math.block_scaled_matmul", "math.packed_block_scaled_matmul"})
    assert projection.attrs.get("output_lanes") == lanes
    assert projection.type == fm.tensor_type(fm.vector_type("bfloat16", lanes), (1, 40 // prod(lanes)))
    assert not any(node.op == "tensors.pack" for node in rewritten.nodes)
    values, weights = operands(packed_weights=packed_weights)
    evaluator = TorchEvaluator(DictWeightResolver(weights))
    torch.testing.assert_close(evaluator.run(rewritten, values), evaluator.run(original, values), rtol=0, atol=0)
    assert post_function_boundary_pack_propagation(rewritten, NvidiaSm90Target()).semantic_hash == rewritten.semantic_hash


def test_fp8_packing_composes_with_existing_lanes_and_keeps_mixed_consumers():
    original = Fp8Consumers(lanes=(4,), output_lanes=(2,), scalar_users=True).build()
    rewritten = post_function_boundary_pack_propagation(original, NvidiaSm90Target())
    projection, = (node for node in rewritten.nodes if node.op == "math.packed_block_scaled_matmul")
    assert projection.attrs["output_lanes"] == (4, 2)
    assert not any(node.op == "tensors.pack" for node in rewritten.nodes)
    values, weights = operands()
    evaluator = TorchEvaluator(DictWeightResolver(weights))
    torch.testing.assert_close(evaluator.run(rewritten, values), evaluator.run(original, values), rtol=0, atol=0)


def test_fp8_shared_projection_preserves_different_consumer_widths():
    original = Fp8Consumers(scalar_users=True, mixed_users=True).build()
    rewritten = post_function_boundary_pack_propagation(original, NvidiaSm90Target())
    projection, = (node for node in rewritten.nodes if node.op == "math.packed_block_scaled_matmul")
    assert projection.attrs["output_lanes"] in {(4,), (8,)}
    assert not any(node.op == "tensors.pack" for node in rewritten.nodes)
    values, weights = operands()
    evaluator = TorchEvaluator(DictWeightResolver(weights))
    torch.testing.assert_close(evaluator.run(rewritten, values), evaluator.run(original, values), rtol=0, atol=0)


def test_fp8_output_packing_is_registered_on_the_large_graph_path():
    original = Fp8Consumers(large=True).build()
    assert len(original.nodes) > 20_000
    rewritten = post_function_boundary_pack_propagation(original, NvidiaSm90Target())
    projection, = (node for node in rewritten.nodes if node.op == "math.packed_block_scaled_matmul")
    assert projection.attrs.get("output_lanes") == (8,)
    assert not any(node.op == "tensors.pack" for node in rewritten.nodes)
