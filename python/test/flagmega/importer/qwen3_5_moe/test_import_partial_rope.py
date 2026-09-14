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
    from triton.flagmega.importer import Qwen35MoeImporter
    from triton.flagmega.importer.model import apply_numerical_profile, importer_registry

    source = checkpoint()
    spec = importer_registry.resolve(source.config, full_model=False)
    importer = Qwen35MoeImporter(source, layer=1, block_size=2, num_blocks=2,
                                 fused_qkvg_projection=True)
    module = apply_numerical_profile(importer.import_module(), profile)
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
    assert {node.id for node in slices} == {
        "decode_attention_query_slice", "decode_attention_gate_slice",
        "decode_attention_key_slice", "decode_attention_value_slice",
    }
    # The fused q/k/v/gate projection regroups checkpoint rows with weight-only
    # concats; no concat may sit on a runtime value path.
    for node in (item for item in module.nodes if item.op == "tensors.concat"):
        stack, seen = list(node.inputs), set()
        while stack:
            current = module.node_map[stack.pop()]
            if current.id in seen:
                continue
            seen.add(current.id)
            assert current.op in {"builtin.var", "builtin.weight", "tensors.slice", "tensors.concat"}, current
            stack.extend(current.inputs)
    assert fm.load_module(fm.emit_module(module, tmp_path / "imported.py")).semantic_hash == module.semantic_hash
