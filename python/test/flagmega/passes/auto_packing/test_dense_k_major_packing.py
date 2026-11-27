# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT

from triton.flagmega import ir as fm
import torch

from triton.flagmega.compiler import Compiler
from triton.flagmega.evaluator import (
    DictWeightResolver,
    TorchEvaluator,
    materialize_constant_assets,
)
from triton.flagmega.ir.ops.tensors._k_major import (
    unpack_k_major_n8_k16_weight,
)
from triton.flagmega.passes.constants import FreezeConstantIslandsPass


def _dense_projection_module(*, n: int, k: int = 2048, with_logits: bool = False):
    class DenseProjection(fm.Module):
        def __init__(self):
            super().__init__(dialect="high_level", stage="imported", entry="main")

        def forward(self):
            value = self.input(
                "value", fm.tensor_type("bfloat16", (1, k)), id="value")
            weight = self.weight(
                "weight",
                fm.tensor_type("bfloat16", (n, k)),
                source="memory",
                key="weight",
                id="weight",
            )
            output = fm.F.math.matmul(
                value, weight, transpose_b=True, name="projection")
            if with_logits:
                logits = fm.F.tensors.cast(output, fm.DType.FLOAT32, name="logits")
                token = fm.F.nn.greedy_sample(logits, name="token")
                self.function("main", (value,), (logits, token))
            else:
                self.function("main", (value,), (output,))

    return DenseProjection().build()


def test_large_output_dense_projection_gets_shape_driven_k_major_candidate_after_vectorization():
    proposed = Compiler().compile(
        _dense_projection_module(n=151936), stop_after="propose-packing").module

    point = next(
        point for point in proposed.selection_points
        if point.id == "packing.projection"
    )
    projection = proposed.node_map[point.owner]

    assert projection.op == "math.vectorized_matmul"
    assert projection.metadata["vectorization_root"] == "projection"
    assert point.default_candidate == "packing.k_major_n8_k16"
    assert {candidate.id for candidate in point.candidates} == {
        "packing.logical",
        "packing.k_major_n8_k16",
    }


def test_small_output_dense_projection_keeps_pyntt_k_major_default():
    proposed = Compiler().compile(
        _dense_projection_module(n=2048), stop_after="propose-packing").module
    point = next(
        point for point in proposed.selection_points
        if point.id == "packing.projection"
    )

    assert point.default_candidate == "packing.k_major_n8_k16"
    assert proposed.selection_map[point.id].policy.startswith(
        "pyntt-auto-packing/"
    )


def test_dense_projection_pack_is_an_editable_constant_graph_and_round_trips():
    original = _dense_projection_module(n=8192, k=128)
    packed = Compiler().compile(original, stop_after="apply-packing").module
    point = next(
        point for point in packed.selection_points
        if point.id == "packing.projection"
    )
    projection = packed.node_map[point.owner]

    assert projection.op == "ntt.packed_matmul"
    physical = packed.node_map[projection.inputs[1]]
    assert tuple(dimension.fixed_value for dimension in physical.type.shape) == (8, 1024)
    assert physical.type.dtype == fm.vector_type("bfloat16", (8, 2, 8))
    assert [
        node.op for node in packed.nodes
        if node.id.startswith("projection.weight_pack.")
    ] == []
    assert [
        node.op for node in packed.nodes
        if node.id.startswith("projection.vectorized.compute.rhs_k_major.")
        and node.op != "builtin.none"
    ] == [
        "tensors.unpack",
        "tensors.pack",
        "tensors.pack",
        "tensors.pack",
        "tensors.permute",
    ]
    assert projection.attrs == {
        "fused_reduce": False,
        "output_data_type": "bfloat16",
        "rhs_layout": "k_major",
    }

    generator = torch.Generator().manual_seed(20260901)
    lhs = torch.randn((1, 128), generator=generator).to(torch.bfloat16)
    weight = torch.randn((8192, 128), generator=generator).to(torch.bfloat16)
    evaluator = TorchEvaluator(DictWeightResolver({"weight": weight}))
    torch.testing.assert_close(
        evaluator.run(packed, {"value": lhs})[0],
        evaluator.run(original, {"value": lhs})[0],
        rtol=0,
        atol=0,
    )
    frozen = FreezeConstantIslandsPass().run(packed)
    values = materialize_constant_assets(
        frozen, DictWeightResolver({"weight": weight}))
    frozen_projection = frozen.node_map[point.owner]
    torch.testing.assert_close(
        unpack_k_major_n8_k16_weight(values[frozen_projection.inputs[1]]),
        weight,
        rtol=0,
        atol=0,
    )


def test_dense_projection_and_sampling_keep_vocab_output_sharded():
    distributed = Compiler().compile(
        _dense_projection_module(n=151936, k=2048, with_logits=True),
        stop_after="auto-distributed",
    ).module
    projection = next(
        node for node in distributed.nodes
        if node.op == "ntt.packed_matmul"
        and node.metadata.get("vectorization_root") == "projection"
    )

    assert isinstance(projection.type, fm.DistributedType)
    # The following greedy sampler consumes vocab shards directly.  Keeping N
    # split avoids constructing a full 151936-element logits vector per owner.
    output_policy = projection.type.axis_policies[1]
    assert isinstance(output_policy, fm.SBPSplit)
    assert output_policy.hierarchy_axes == (0, 1)
    assert isinstance(output_policy.stages[0].distribution, fm.BlockCyclicSplit)
    assert projection.type.partial is None
    packed_weight = distributed.node_map[projection.inputs[1]]
    assert packed_weight.op == "distributed.sharded_view"
    assert isinstance(packed_weight.type, fm.DistributedType)
    assert packed_weight.type.tensor.rank == 2
    assert packed_weight.type.tensor.dtype == fm.vector_type(
        "bfloat16", (8, 2, 8)
    )
    assert isinstance(packed_weight.type.axis_policies[1], fm.SBPSplit)
    assert packed_weight.type.axis_policies[1].hierarchy_axes == (0, 1)
    assert isinstance(
        packed_weight.type.axis_policies[1].stages[0].distribution,
        fm.BlockCyclicSplit,
    )
