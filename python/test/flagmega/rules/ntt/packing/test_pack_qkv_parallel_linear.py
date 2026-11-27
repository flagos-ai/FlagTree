# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT

import torch

from triton.flagmega import ir as fm
from triton.flagmega.evaluator import DictWeightResolver, TorchEvaluator
from triton.flagmega.rules.ntt.packing import NttPackingPolicy
from triton.flagmega.targets import NvidiaSm90Target


class QKVModule(fm.Module):
    def __init__(self):
        super().__init__(dialect="high_level", stage="imported", entry="main")

    def forward(self):
        value = self.input("value", fm.tensor_type("bfloat16", (2, 32)))
        q_weight = self.weight(
            "q_weight", fm.tensor_type("bfloat16", (32, 64)),
            source="memory", key="q_weight", id="q_weight")
        k_weight = self.weight(
            "k_weight", fm.tensor_type("bfloat16", (32, 32)),
            source="memory", key="k_weight", id="k_weight")
        v_weight = self.weight(
            "v_weight", fm.tensor_type("bfloat16", (32, 32)),
            source="memory", key="v_weight", id="v_weight")
        none = fm.F.builtin.none(name="none")
        qkv = fm.F.nn.qkv_parallel_linear(
            value,
            q_weight,
            k_weight,
            v_weight,
            none,
            none,
            none,
            none,
            none,
            none,
            none,
            none,
            none,
            num_heads=8,
            num_kv_heads=4,
            output_data_type="bfloat16",
            name="qkv",
        )
        self.function("main", (value,), (qkv,))


def _pack(module):
    policy = NttPackingPolicy(vector_bytes=16, k_pack=2)
    target = NvidiaSm90Target()
    return policy.apply(policy.propose(module, target), target)


def test_qkv_k_major_rule_keeps_three_vector_weights_and_no_machine_tile():
    packed = _pack(QKVModule().build())
    root = packed.node_map["qkv"]
    projection = next(
        node for node in packed.nodes if node.op == "ntt.packed_qkv_parallel_linear"
    )
    combine = next(
        node
        for node in packed.nodes
        if node.op == "ntt.packed_qkv_parallel_linear_combine"
    )

    assert root.op == "builtin.tuple"
    assert combine.inputs == (projection.id,)
    packed_items = tuple(
        packed.node_map[packed.node_map[value].inputs[0]] for value in root.inputs
    )
    assert tuple(item.inputs[0] for item in packed_items) == (combine.id,) * 3
    assert projection.attrs == {
        "num_heads": 8,
        "num_kv_heads": 4,
        "output_data_type": "bfloat16",
        "rhs_layout": "k_major",
    }
    assert len(projection.inputs) == 13
    for weight_id in projection.inputs[1:4]:
        weight = packed.node_map[weight_id]
        assert isinstance(weight.type, fm.TensorType)
        assert weight.type.dtype == fm.VectorType(fm.DType.BFLOAT16, (8, 2, 8))
    assert not {
        "block_k", "mesh", "mesh_x", "mesh_y", "context_mesh_size",
        "head_mesh_size", "num_stages", "num_warps",
    }.intersection(projection.attrs)
    selection = packed.selection_map["packing.qkv"]
    assert selection.policy.startswith("pyntt-auto-packing/")


def test_qkv_k_major_rule_is_evaluator_equivalent_and_python_round_trips(tmp_path):
    logical = QKVModule().build()
    packed = _pack(logical)
    generator = torch.Generator().manual_seed(20260902)
    inputs = {"value": torch.randn((2, 32), generator=generator).to(torch.bfloat16)}
    weights = {
        "q_weight": torch.randn((32, 64), generator=generator).to(torch.bfloat16),
        "k_weight": torch.randn((32, 32), generator=generator).to(torch.bfloat16),
        "v_weight": torch.randn((32, 32), generator=generator).to(torch.bfloat16),
    }
    evaluator = TorchEvaluator(DictWeightResolver(weights))
    expected = evaluator.run(logical, inputs)[0]
    actual = evaluator.run(packed, inputs)[0]
    for packed_value, logical_value in zip(actual, expected):
        torch.testing.assert_close(packed_value, logical_value, rtol=0, atol=0)

    path = tmp_path / "qkv.py"
    fm.emit_module(packed, path)
    resumed = fm.load_module(path)
    assert resumed.semantic_hash == packed.semantic_hash
