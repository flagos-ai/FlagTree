# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT

import torch

from triton.flagmega import ir as fm
from triton.flagmega.compiler import Compiler
from triton.flagmega.evaluator import (
    DictWeightResolver,
    TorchEvaluator,
    materialize_constant_assets,
)
from triton.flagmega.ir.ops.nn.packed_dense_matmul_glu import (
    unpack_k_major_n8_k16_weight,
)
from triton.flagmega.passes.constants import FreezeConstantIslandsPass


def _dense_glu_module():
    class DenseGlu(fm.Module):
        def __init__(self):
            super().__init__(dialect="high_level", stage="imported", entry="main")

        def forward(self):
            value = self.input("value", fm.tensor_type("bfloat16", (1, 128)), id="value")
            gate = self.weight(
                "gate", fm.tensor_type("bfloat16", (64, 128)),
                source="memory", key="gate", id="gate",
            )
            up = self.weight(
                "up", fm.tensor_type("bfloat16", (64, 128)),
                source="memory", key="up", id="up",
            )
            output = fm.F.nn.dense_matmul_glu(
                value, gate, up, activation="silu", name="glu")
            self.function("main", (value,), (output,))

    return DenseGlu().build()


def test_auto_packing_builds_editable_k_major_constant_graphs_for_dense_glu():
    packed = Compiler().compile(
        _dense_glu_module(), stop_after="apply-packing").module
    glu = packed.node_map["glu"]

    assert glu.op == "nn.packed_dense_matmul_glu"
    assert glu.attrs["packed_layout"] == "k_major_n8_k16"
    assert packed.selection_map["packing.glu"].candidate_id == "packing.k_major_n8_k16"
    for packed_id, source_id in zip(glu.inputs[1:], ("gate", "up")):
        physical = packed.node_map[packed_id]
        assert physical.op == "tensors.reshape"
        assert physical.inputs[0].endswith(".lane_major")
        assert tuple(dim.fixed_value for dim in physical.type.shape) == (8, 8, 2, 64)
        assert physical.metadata["packed_from"] == source_id
        assert physical.metadata["packed_layout"] == "k_major_n8_k16"
        assert [
            node.op for node in packed.nodes
            if node.id.startswith(f"glu.{source_id}_pack.")
        ] == ["tensors.reshape", "tensors.permute", "tensors.reshape"]


def test_k_major_dense_glu_pack_round_trips_and_preserves_evaluation():
    original = _dense_glu_module()
    packed = Compiler().compile(original, stop_after="apply-packing").module
    generator = torch.Generator().manual_seed(20260901)
    value = torch.randn((1, 128), generator=generator).to(torch.bfloat16)
    gate = torch.randn((64, 128), generator=generator).to(torch.bfloat16)
    up = torch.randn((64, 128), generator=generator).to(torch.bfloat16)
    evaluator = TorchEvaluator(DictWeightResolver({"gate": gate, "up": up}))

    torch.testing.assert_close(
        evaluator.run(packed, {"value": value})[0],
        evaluator.run(original, {"value": value})[0],
        rtol=0,
        atol=0,
    )
    frozen = FreezeConstantIslandsPass().run(packed)
    assets = materialize_constant_assets(
        frozen, DictWeightResolver({"gate": gate, "up": up}))
    glu = frozen.node_map["glu"]
    for packed_id, logical in zip(glu.inputs[1:], (gate, up)):
        torch.testing.assert_close(
            unpack_k_major_n8_k16_weight(assets[packed_id]), logical,
            rtol=0, atol=0,
        )


def test_dense_glu_k_major_pack_is_shape_derived_without_model_special_cases():
    packed = Compiler().compile(
        _dense_glu_module(), stop_after="apply-packing").module
    source = fm.module_source(packed)

    assert "qwen" not in source.lower()
    assert "2048" not in source
    assert "6144" not in source
    glu = packed.node_map["glu"]
    for packed_id, source_id in zip(glu.inputs[1:], ("gate", "up")):
        logical = packed.node_map[source_id].type
        physical = packed.node_map[packed_id].type
        n, k = (dimension.fixed_value for dimension in logical.shape)
        assert tuple(dimension.fixed_value for dimension in physical.shape) == (
            k // 16, n // 8, 2, 64,
        )


def test_auto_distribution_treats_offline_pack_graph_as_constant_storage():
    distributed = Compiler().compile(
        _dense_glu_module(), stop_after="auto-distributed").module
    selection = distributed.selection_map["distribution.glu"]
    point = next(
        point
        for point in distributed.selection_points
        if point.id == selection.point_id
    )
    candidate = next(
        candidate
        for candidate in point.candidates
        if candidate.id == selection.candidate_id
    )
    glu = distributed.node_map["glu"]

    assert candidate.parameters["reason"] == "matmul-glu-output-sbp"
    assert isinstance(glu.type, fm.DistributedType)
    assert isinstance(glu.type.axis_policies[1], fm.SBPSplit)
    for input_id in glu.inputs[1:]:
        sharded = distributed.node_map[input_id]
        assert sharded.op == "distributed.sharded_view"
        assert sharded.metadata["realization"] == "sharded_view"
        assert distributed.node_map[sharded.inputs[0]].metadata["packed_layout"] == (
            "k_major_n8_k16"
        )
