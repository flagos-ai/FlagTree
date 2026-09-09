# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT

from collections import Counter
from dataclasses import replace

import pytest

from triton.flagmega import ir as fm
from triton.flagmega.errors import ImporterError
from triton.flagmega.importer import Qwen35MoeImporter, import_model, import_model_layer
from python.test.flagmega.importer.qwen3_5_moe.helpers import checkpoint, configuration


def test_importer_uses_metadata_only_and_reuses_each_decoder_kind():
    source = checkpoint()
    source.load_tensor = lambda *args, **kwargs: pytest.fail("Import must not load weights")
    module = import_model(source)
    assert set(module.function_map) == {"main", "decode_linear", "decode_attention"}
    assert Counter(node.attrs["callee"]
                   for node in module.nodes
                   if node.op == "builtin.call") == {"decode_linear": 2, "decode_attention": 2}
    assert sum(node.op == "nn.gdn_recurrent_core" for node in module.nodes) == 1
    assert sum(node.op == "nn.paged_attention" for node in module.nodes) == 1
    assert sum(node.op == "nn.sparse_experts" for node in module.nodes) == 2
    assert not any("fp8" in node.op or "block_scaled" in node.op for node in module.nodes)
    assert len(module.function_map[module.entry].outputs) == 4
    for name in ("decode_linear", "decode_attention"):
        assert module.function_map[name].attrs["reusable"] is True


def test_state_layer_ids_are_dense_within_kind_and_sequence_advances_once():
    module = import_model(checkpoint())
    assert [module.node_map[f"layer_{index}_state_id"].attrs["value"] for index in range(4)] == [0, 0, 1, 1]
    assert module.node_map["layer_1_advance_sequence"].attrs["value"] is False
    assert module.node_map["layer_3_advance_sequence"].attrs["value"] is True
    gdn = dict(module.node_map["gdn_state"].type.fields)
    assert gdn["recurrent"].shape[0].fixed_value == 2
    cache = dict(module.node_map["paged_state"].type.fields)["kv_caches"]
    assert cache.shape[1].fixed_value == 2


@pytest.mark.parametrize("layer,callee", [(0, "decode_linear"), (1, "decode_attention"), (3, "decode_attention")])
def test_single_layer_import_retains_embedding_moe_lm_head_and_sampler(layer, callee):
    module = import_model_layer(checkpoint(), layer=layer)
    assert set(module.function_map) == {"main", callee}
    assert module.metadata["imported_layer_indices"] == (layer, )
    assert all(op in {node.op
                      for node in module.nodes}
               for op in ("nn.embedding", "nn.sparse_experts", "nn.greedy_sample"))
    assert module.node_map[f"layer_{layer}_state_id"].attrs["value"] == 0


def test_stacked_weight_slices_are_outside_reusable_decoder():
    module = import_model(checkpoint())
    for layer in range(4):
        gate, up = (module.node_map[f"layer_{layer}_expert_{part}"] for part in ("gate", "up"))
        assert gate.inputs == up.inputs
        assert gate.type == up.type == fm.tensor_type("bfloat16", (4, 8, 16))
        assert (gate.attrs["starts"], up.attrs["starts"]) == ((0, ), (8, ))
        call = module.node_map[f"layer_{layer}_call"]
        assert gate.id in call.inputs and up.id in call.inputs


def test_importer_python_checkpoint_rebuilds_identical_module():
    module = import_model(checkpoint(), revision="test-revision")
    source = fm.module_source(module)
    assert "F.tensors.top_k(" in source and "F.nn.gated_delta_net_state_slice(" in source
    assert "self.call(" not in source
    namespace = {}
    exec(source, namespace)
    assert namespace["MODULE"].semantic_hash == module.semantic_hash


def test_importer_validates_checkpoint_weights_and_does_not_claim_vllm_profile():
    source = checkpoint()
    key = "model.language_model.layers.1.self_attn.q_proj.weight"
    source._tensors[key] = replace(source._tensors[key], shape=(16, 16))
    with pytest.raises(ImporterError, match="q_proj.weight.*requires"):
        import_model(source)
    with pytest.raises(ImporterError, match="does not support numerical profile"):
        import_model(checkpoint(), numerical_profile="vllm-inductor-level3")


@pytest.mark.parametrize("layer", [-1, 4, True])
def test_importer_rejects_invalid_selected_layer(layer):
    with pytest.raises(ImporterError, match="out of range"):
        Qwen35MoeImporter(checkpoint(), layer=layer)


def test_full_layer_count_scales_calls_but_not_decoder_bodies():
    module = import_model(checkpoint(configuration(("linear_attention", ) * 3 + ("full_attention", ))))
    assert Counter(node.attrs["callee"]
                   for node in module.nodes
                   if node.op == "builtin.call") == {"decode_linear": 3, "decode_attention": 1}
