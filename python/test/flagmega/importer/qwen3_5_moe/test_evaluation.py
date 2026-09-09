# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT

import torch

from triton.flagmega import ir as fm
from triton.flagmega.evaluator import CheckpointWeightResolver, DictWeightResolver, TorchEvaluator
from triton.flagmega.importer import Qwen35MoeImporter
from triton.flagmega.importer.qwen3_5_moe.decoder import rms_norm
from triton.flagmega.ir.ops.nn._gdn_state import create_gdn_state
from triton.flagmega.ir.ops.nn._paged_attention_state import create_paged_attention_state
from python.test.flagmega.importer.qwen3_5_moe.helpers import checkpoint


def test_mixed_decoder_multistep_updates_every_independent_layer():
    source = checkpoint(with_values=True)
    importer = Qwen35MoeImporter(source, block_size=2, num_blocks=3)
    module = importer.import_module()
    evaluator = TorchEvaluator(CheckpointWeightResolver(source))
    gdn, paged = create_gdn_state(importer.gdn_config), create_paged_attention_state(importer.paged_config)
    for step, token in enumerate((1, 4, 2, 5)):
        logits, sampled, updated_gdn, updated_paged = evaluator.run(
            module, {
                "input_ids": torch.tensor([token], dtype=torch.int32),
                "gated_delta_net_state": gdn,
                "paged_attention_state": paged,
            })
        assert updated_gdn is gdn and updated_paged is paged
        assert paged.sequence_length == step + 1
        assert logits.shape == (1, 32) and logits.dtype == torch.float32 and torch.isfinite(logits).all()
        assert sampled.item() == logits.argmax(-1).item()
        for layer in range(2):
            assert torch.count_nonzero(gdn.convolution_layer(layer))
            assert torch.count_nonzero(gdn.recurrent_layer(layer))
        assert not torch.equal(gdn.convolution_layer(0), gdn.convolution_layer(1))


def test_attention_query_and_gate_are_split_within_each_head():
    source = checkpoint(with_values=True)
    importer = Qwen35MoeImporter(source, layer=1, block_size=2, num_blocks=2)
    module = importer.import_module()
    _, trace = TorchEvaluator(CheckpointWeightResolver(source)).run_with_trace(
        module, {
            "input_ids": torch.tensor([7], dtype=torch.int32),
            "gated_delta_net_state": create_gdn_state(importer.gdn_config),
            "paged_attention_state": create_paged_attention_state(importer.paged_config),
        })
    packed = trace["decode_attention_q_gate"].reshape(1, 2, 16)
    torch.testing.assert_close(trace["decode_attention_query_slice"], packed[:, :, :8], rtol=0, atol=0)
    torch.testing.assert_close(trace["decode_attention_gate_slice"], packed[:, :, 8:], rtol=0, atol=0)


def test_qwen35_norm_does_not_round_before_one_plus_weight_scaling():

    class Graph(fm.Module):

        def forward(self):
            value = self.input("value", fm.tensor_type("bfloat16", (2, 8)))
            weight = self.input("weight", fm.tensor_type("bfloat16", (8, )))
            self.function("main", (value, weight), (rms_norm(value, weight, 1e-6, name="normalized"), ))

    module = Graph(dialect="high_level", stage="imported", entry="main").build()
    value = torch.linspace(-3, 5, 16).reshape(2, 8).bfloat16()
    weight = torch.linspace(-.5, .5, 8).bfloat16()
    actual = TorchEvaluator(DictWeightResolver({})).run(module, {"value": value, "weight": weight})[0]
    expected = (value.float() * torch.rsqrt(value.float().square().mean(-1, keepdim=True) + 1e-6) *
                (1 + weight.float())).bfloat16()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
