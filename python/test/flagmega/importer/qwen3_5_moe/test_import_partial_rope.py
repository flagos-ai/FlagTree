# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT

import pytest

from triton.flagmega import ir as fm
from triton.flagmega.importer import import_model
from triton.flagmega.importer.numerics import VLLM_AE10_INDUCTOR_LEVEL3
from triton.flagmega.ir.print_weights import WeightPrintAnalysis
from python.test.flagmega.importer.qwen3_5_moe.helpers import checkpoint


@pytest.mark.parametrize("profile", ["nncase", VLLM_AE10_INDUCTOR_LEVEL3])
def test_import_uses_native_partial_rope_and_keeps_full_head_norm(profile, tmp_path):
    module = import_model(checkpoint(), numerical_profile=profile)
    ropes = [node for node in module.nodes if node.op == "nn.rope"]
    assert len(ropes) == 2
    for rope in ropes:
        assert rope.attrs == {"rotary_dim": 4}
        assert rope.type.shape[-1].fixed_value == 8
        assert module.node_map[rope.inputs[1]].type.shape[-1].fixed_value == 4
        norm = module.node_map[rope.inputs[0]]
        while norm.op == "tensors.cast":
            norm = module.node_map[norm.inputs[0]]
        assert norm.op == "nn.norm_apply"
        assert norm.type.shape[-1].fixed_value == 8
    weights = WeightPrintAnalysis.analyze(module)
    slices = [node for node in module.nodes if node.op == "tensors.slice" and node.id not in weights.values]
    assert {node.id for node in slices} == {"decode_attention_query_slice", "decode_attention_gate_slice"}
    assert not any(node.op == "tensors.concat" for node in module.nodes)
    assert fm.load_module(fm.emit_module(module, tmp_path / "imported.py")).semantic_hash == module.semantic_hash
