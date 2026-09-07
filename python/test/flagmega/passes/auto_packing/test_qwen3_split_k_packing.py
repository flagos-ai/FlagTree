# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT

import torch

from ..auto_distributed.helpers import qwen3_packed_module
from triton.flagmega import ir as fm
from triton.flagmega.evaluator import DictWeightResolver, TorchEvaluator
from triton.flagmega.ir.ops.nn.packed_qwen3_paged_attention import unpack_split_k_qkv_weight


def test_imported_qkv_uses_portable_pyntt_pack_without_target_asset_abi():
    module = qwen3_packed_module()
    projection = module.node_map[
        "self_attention_qkv_projection.packed_projection"
    ]

    assert projection.op == "ntt.packed_qkv_parallel_linear"
    assert module.selection_map[
        "packing.self_attention_qkv_projection"
    ].candidate_id == "packing.k_major_n8_k16"
    assert not any("target_asset_abi" in node.metadata for node in module.nodes)
    assert module.node_map["self_attention_paged_attention"].op == (
        "nn.paged_attention"
    )


def test_constant_qkv_pack_does_not_require_a_reusable_function_abi():
    module = qwen3_packed_module(reusable=False)

    assert module.node_map[
        "self_attention_qkv_projection.packed_projection"
    ].op == "ntt.packed_qkv_parallel_linear"
    assert "packing.self_attention_qkv_projection" in module.selection_map


def test_split_k_k_major_constant_graph_round_trips_three_logical_weights():
    class PackGraph(fm.Module):
        def __init__(self):
            super().__init__(dialect="high_level", stage="imported", entry="main")

        def forward(self):
            q = self.input("q", fm.tensor_type("bfloat16", (128, 128)), id="q")
            k = self.input("k", fm.tensor_type("bfloat16", (64, 128)), id="k")
            v = self.input("v", fm.tensor_type("bfloat16", (64, 128)), id="v")
            q_mesh = fm.F.tensors.permute(
                fm.F.tensors.reshape(q, (2, 64, 2, 64)), (2, 0, 1, 3))
            kv_mesh = fm.F.tensors.permute(
                fm.F.tensors.reshape(
                    fm.F.tensors.concat(k, v, axis=0), (2, 64, 2, 64)),
                (2, 0, 1, 3),
            )
            local = fm.F.tensors.concat(q_mesh, kv_mesh, axis=2)
            tiled = fm.F.tensors.permute(
                fm.F.tensors.reshape(local, (4, 128, 1, 64)), (0, 2, 1, 3))
            lanes = fm.F.tensors.reshape(tiled, (4, 1, 16, 8, 4, 16))
            physical = fm.F.tensors.reshape(
                fm.F.tensors.permute(lanes, (0, 1, 2, 4, 3, 5)),
                (64, 4, 2, 64),
                name="physical",
            )
            self.function("main", (q, k, v), (physical,))

    generator = torch.Generator().manual_seed(20260901)
    q = torch.randn((128, 128), generator=generator).to(torch.bfloat16)
    k = torch.randn((64, 128), generator=generator).to(torch.bfloat16)
    v = torch.randn((64, 128), generator=generator).to(torch.bfloat16)
    packed = TorchEvaluator(DictWeightResolver({})).run(
        PackGraph().build(), {"q": q, "k": k, "v": v})[0]

    actual = unpack_split_k_qkv_weight(
        packed,
        hidden_size=128,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=64,
        context_mesh_size=2,
        head_mesh_size=2,
        block_k=64,
    )
    for result, expected in zip(actual, (q, k, v)):
        torch.testing.assert_close(result, expected, rtol=0, atol=0)
