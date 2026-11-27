# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""GLM-5.3-Flash importer: import, compile, execute, and compare against TorchEvaluator."""

from __future__ import annotations

import copy

import pytest
import torch
from triton.flagmega.artifacts import write_artifact
from triton.flagmega.compiler import Compiler
from triton.flagmega.errors import ImporterError
from triton.flagmega.evaluator import CheckpointWeightResolver, TorchEvaluator
from triton.flagmega.importer import MemoryCheckpoint, TensorInfo
from triton.flagmega.importer import import_model
from triton.flagmega.importer.glm5_moe import Glm5MoeImporter, Glm5MoeModelImporter
from triton.flagmega.importer.glm5_moe.config import Glm5MoeConfig
from triton.flagmega.ir import DType
from triton.flagmega.ir.ops.nn._dsa_indexer_state import create_dsa_indexer_state
from triton.flagmega.ir.ops.nn._paged_attention_state import create_paged_attention_state
from triton.flagmega.runtime import load

from python.test.flagmega.importer.glm5_moe import helpers as glm_helpers
from python.test.flagmega.importer.glm5_moe.helpers import PUBLISHED, REDUCED

ARTIFACT_TOLERANCE = {"atol": 0.05, "rtol": 0.05}


def assert_greedy_token(logits, reference_logits, token):
    """Argmax must match unless the reference top-2 gap is inside the noise.

    Random checkpoints routinely place the top two logits within 1e-3 of each
    other over a 154880-word vocabulary, so an exact-token assert flips on
    last-ulp scheduling differences rather than measure anything.
    """
    gap = torch.topk(reference_logits.flatten(), 2).values
    if (gap[0] - gap[1]).abs() <= ARTIFACT_TOLERANCE["atol"]:
        pytest.skip("reference top-2 logits tie inside tolerance; token is not decisive")
    assert int(token.item()) == int(torch.argmax(reference_logits.flatten()).item())


def test_import_validates_and_records_layer_contract():
    checkpoint = glm_helpers.checkpoint(REDUCED, with_values=False)
    module = Glm5MoeImporter(checkpoint).import_module()
    assert module.metadata["architecture"] == "Glm5NextForConditionalGeneration"
    assert module.metadata["layer_types"] == ("deepseek_sparse_attention", )
    assert module.metadata["num_experts_per_tok"] == 2
    names = {node.op for node in module.nodes}
    assert "nn.sparse_experts" in names
    assert "nn.staged_routing" in names
    assert "nn.paged_attention" in names
    # One staged-routing stage replaces the sigmoid/add/top_k/gather/sum/div/mul chain.


def test_glm_mla_advances_paged_cache_once_per_decode_token():
    checkpoint = glm_helpers.checkpoint(REDUCED, with_values=False)
    module = Glm5MoeImporter(checkpoint).import_module()
    updates = [node for node in module.nodes if node.op == "nn.update_paged_attention_kv_cache"]
    assert [node.attrs["cache_kind"] for node in updates] == ["key", "value"]
    advance_nodes = [module.node_map[node.inputs[3]] for node in updates]
    assert advance_nodes[0].attrs["value"] is False
    assert advance_nodes[1].attrs["name"] == "advance_sequence"


def test_full_importer_builds_general_mhc_stream_contract():
    scale = dict(REDUCED, mhc=True, hc_mult=4, hc_sinkhorn_iters=2)
    checkpoint = glm_helpers.checkpoint(scale, with_values=False, full_model=True)
    module = Glm5MoeModelImporter(checkpoint).import_module()
    assert module.metadata["mhc"] is True
    assert module.metadata["hc_mult"] == 4
    assert any(node.op == "math.matmul" for node in module.nodes)
    assert any(node.id == "collapse_head_mean" for node in module.nodes)


def test_full_importer_stacks_published_per_expert_weight_layout():
    checkpoint = glm_helpers.checkpoint(REDUCED, with_values=False, full_model=True,
                                       expert_layout="separate")
    module = Glm5MoeModelImporter(checkpoint).import_module()
    assert module.metadata["num_hidden_layers"] == len(REDUCED["layer_types"])


def test_importer_imports_kda_layer_with_new_operators():
    # REDUCED layer 4 is the first linear-attention (KDA) layer with sparse MoE.
    checkpoint = glm_helpers.checkpoint(REDUCED, with_values=False, kind="kda")
    module = Glm5MoeImporter(checkpoint, layer=4).import_module()
    assert module.metadata["layer_types"] == ("linear_attention", )
    names = {node.op for node in module.nodes}
    assert "nn.kda_convolution" in names
    # The gate-fused variant: the forget gate and the beta sigmoid live inside
    # the recurrence, so the plain kda_recurrent_core is no longer emitted here.
    assert "nn.kda_gated_recurrent_core" in names
    assert "nn.kda_recurrent_core" not in names
    kda_types = Glm5MoeImporter(checkpoint, layer=4).config.weight_types("linear_attention")
    assert tuple(dim.fixed_value for dim in kda_types["self_attn.q_conv1d.weight"].shape[-2:]) == (1, 4)
    assert "self_attn.k_conv1d.weight" in kda_types
    assert "self_attn.v_conv1d.weight" in kda_types
    assert kda_types["self_attn.A_log"].dtype == DType.FLOAT32


def test_registry_resolves_glm_checkpoint():
    from triton.flagmega.importer import import_model_layer

    checkpoint = glm_helpers.checkpoint(REDUCED, with_values=False)
    module = import_model_layer(checkpoint, layer=glm_helpers.checkpoint_layer(REDUCED))
    assert module.metadata["architecture"] == "Glm5NextForConditionalGeneration"


def test_synthetic_glm_kda_evaluator_runs_with_published_weight_names():
    checkpoint = glm_helpers.checkpoint(REDUCED, kind="kda")
    importer = Glm5MoeImporter(checkpoint, layer=4)
    module = importer.import_module()
    text = glm_helpers.configuration(REDUCED)["text_config"]
    inputs = {
        "input_ids": torch.tensor([11], dtype=torch.int32),
        "kda_conv_state": torch.zeros(3, text["kda_num_heads"], text["kda_head_dim"],
                                       text["kda_conv_kernel_size"] - 1, dtype=torch.bfloat16),
        "kda_recurrent_state": torch.zeros(1, text["kda_num_heads"], text["kda_head_dim"],
                                            text["kda_head_dim"], dtype=torch.float32),
    }
    outputs = TorchEvaluator(CheckpointWeightResolver(checkpoint)).run(module, inputs)
    assert outputs[0].shape == (1, text["vocab_size"])
    assert outputs[1].shape == (1, )
    assert outputs[2].shape == inputs["kda_conv_state"].shape
    assert outputs[3].shape == inputs["kda_recurrent_state"].shape


def test_glm_importer_rejects_native_fp8_weight_without_scale_contract():
    source = glm_helpers.checkpoint(REDUCED, with_values=False)
    infos = {
        key: TensorInfo(
            info.key,
            DType.FLOAT8_E4M3FN if key.endswith("mlp.gate.weight") else info.dtype,
            info.shape,
            info.source,
        )
        for key in source.keys
        for info in (source.tensor_info(key),)
    }
    checkpoint = MemoryCheckpoint(source.config, infos)
    with pytest.raises(ImporterError, match="requires bfloat16"):
        Glm5MoeImporter(checkpoint).import_module()


def test_full_importer_compiles_native_fp8_block_scaled_separate_experts():
    scale = dict(REDUCED, quantized=True, fp8_block_size=(4, 4))
    checkpoint = glm_helpers.checkpoint(scale, with_values=False, full_model=True,
                                        expert_layout="separate")
    module = Glm5MoeModelImporter(checkpoint).import_module()
    assert any(node.op == "math.block_scaled_matmul" for node in module.nodes)
    assert any(node.op == "nn.sparse_experts" for node in module.nodes)
    compiled = Compiler().compile(module).module
    assert compiled.stage == "bufferized_tir"


def test_full_importer_evaluates_native_fp8_block_scaled_paths():
    scale = dict(REDUCED, quantized=True, fp8_block_size=(4, 4))
    checkpoint = glm_helpers.checkpoint(scale, full_model=True, expert_layout="separate")
    importer = Glm5MoeModelImporter(checkpoint)
    module = importer.import_module()
    text = glm_helpers.configuration(scale)["text_config"]
    inputs = {
        "input_ids": torch.tensor([11], dtype=torch.int32),
        "paged_attention_state": create_paged_attention_state(importer.paged_config),
    }
    for layer in importer.kda_layers:
        qkv = text["kda_num_heads"] * text["kda_head_dim"]
        inputs[f"kda_conv_state_{layer}"] = torch.zeros(
            (3, text["kda_num_heads"], text["kda_head_dim"], text["kda_conv_kernel_size"] - 1),
            dtype=torch.bfloat16)
        inputs[f"kda_recurrent_state_{layer}"] = torch.zeros(
            (1, text["kda_num_heads"], text["kda_head_dim"], text["kda_head_dim"]), dtype=torch.float32)
    outputs = TorchEvaluator(CheckpointWeightResolver(checkpoint)).run(module, inputs)
    assert outputs[0].shape == (1, text["vocab_size"])
    assert outputs[1].shape == (1,)


def test_full_importer_evaluates_stateful_dsa_indexer_and_sparse_attention():
    scale = dict(
        REDUCED,
        indexer_types=("full",) * len(REDUCED["layer_types"]),
        index_topk=4,
        index_head_dim=4,
        index_n_heads=2,
        index_kpool=2,
    )
    checkpoint = glm_helpers.checkpoint(scale, full_model=True)
    importer = Glm5MoeModelImporter(checkpoint, block_size=8, num_blocks=2)
    module = importer.import_module()
    text = checkpoint.config["text_config"]
    dsa_state = create_dsa_indexer_state(importer.dsa_config)
    paged_state = create_paged_attention_state(importer.paged_config)
    inputs = {
        "input_ids": torch.tensor([11], dtype=torch.int32),
        "paged_attention_state": paged_state,
        "dsa_indexer_state": dsa_state,
    }
    for layer in importer.kda_layers:
        qkv = text["kda_num_heads"] * text["kda_head_dim"]
        inputs[f"kda_conv_state_{layer}"] = torch.zeros(
            (3, text["kda_num_heads"], text["kda_head_dim"], text["kda_conv_kernel_size"] - 1),
            dtype=torch.bfloat16)
        inputs[f"kda_recurrent_state_{layer}"] = torch.zeros(
            (1, text["kda_num_heads"], text["kda_head_dim"], text["kda_head_dim"]),
            dtype=torch.float32)
    outputs = TorchEvaluator(CheckpointWeightResolver(checkpoint)).run(module, inputs)
    assert outputs[0].shape == (1, text["vocab_size"])
    assert dsa_state.sequence_length == 1
    assert paged_state.sequence_length == 1
    assert any(node.op == "nn.dsa_indexer" for node in module.nodes)
    assert any(node.op == "nn.dsa_paged_attention" for node in module.nodes)


def test_glm_importer_rejects_mla_cache_dimension_mismatch():
    source = glm_helpers.checkpoint(REDUCED, with_values=False)
    config = copy.deepcopy(source.config)
    config["text_config"]["v_head_dim"] += 1
    checkpoint = MemoryCheckpoint(
        config,
        {key: source.tensor_info(key) for key in source.keys},
    )
    with pytest.raises(ImporterError, match="equal MLA key and value head dimensions"):
        Glm5MoeImporter(checkpoint)


def test_glm_config_reads_nested_gate_bound_and_rejects_unsupported_real_features():
    config = glm_helpers.configuration(REDUCED)
    parsed = Glm5MoeConfig.parse(config)
    assert parsed.kda_gate_lower_bound == -5.0

    unsupported = copy.deepcopy(config)
    unsupported["text_config"]["mhc"] = True
    with pytest.raises(ImporterError, match="hyper-connections"):
        Glm5MoeConfig.parse(unsupported)

    unsupported = copy.deepcopy(config)
    unsupported["text_config"]["indexer_types"] = ["full"] * unsupported["text_config"]["num_hidden_layers"]
    with pytest.raises(ImporterError, match="sparse-indexer"):
        Glm5MoeConfig.parse(unsupported)


def test_glm_config_rejects_conflicting_gate_bound_locations():
    config = glm_helpers.configuration(REDUCED)
    config["text_config"]["gate_lower_bound"] = -4.0
    with pytest.raises(ImporterError, match="conflicting"):
        Glm5MoeConfig.parse(config)


@pytest.mark.cuda
def test_synthetic_glm_executable_matches_reference(tmp_path):
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0):
        pytest.skip("SM90 CUDA is required for the GLM executable integration test")
    checkpoint = glm_helpers.checkpoint(REDUCED)
    importer = Glm5MoeImporter(checkpoint)
    imported = importer.import_module()
    compiled = Compiler().compile(imported).module
    artifact = write_artifact(compiled, tmp_path / "glm-artifact", target="nvidia-sm90", checkpoint=checkpoint,
                              emit_executable=True)

    state = create_paged_attention_state(importer.paged_config)
    input_ids = torch.tensor([11], dtype=torch.int32)
    reference_logits, _, _ = TorchEvaluator(CheckpointWeightResolver(checkpoint)).run(
        imported, {"input_ids": input_ids, "paged_attention_state": state})

    runtime = load(artifact, device="cuda:0")
    runtime_state = runtime.create_state()
    device_ids = input_ids.to("cuda:0")
    logits = torch.empty((1, glm_helpers.configuration(REDUCED)["text_config"]["vocab_size"]), dtype=torch.float32,
                         device="cuda:0")
    token = torch.empty((1, ), dtype=torch.int32, device="cuda:0")
    runtime.prepare(device_ids, runtime_state, logits=logits, next_token=token)
    runtime.run_into(logits, token, device_ids, runtime_state)
    torch.cuda.synchronize()

    assert runtime.prepare_count == 1
    torch.testing.assert_close(logits.cpu(), reference_logits, **ARTIFACT_TOLERANCE)
    assert_greedy_token(logits.cpu(), reference_logits, token)
    torch.testing.assert_close(runtime_state.kv_caches.cpu().float(), state.kv_caches.float(),
                               **ARTIFACT_TOLERANCE)


@pytest.mark.cuda
def test_synthetic_glm_kda_layer_matches_reference(tmp_path):
    """The KDA (linear-attention) layer through the new kda operators."""
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0):
        pytest.skip("SM90 CUDA is required for the GLM executable integration test")
    checkpoint = glm_helpers.checkpoint(REDUCED, kind="kda")
    imported = Glm5MoeImporter(checkpoint, layer=4).import_module()
    compiled = Compiler().compile(imported).module
    artifact = write_artifact(compiled, tmp_path / "glm-kda-artifact", target="nvidia-sm90",
                              checkpoint=checkpoint, emit_executable=True)

    text = glm_helpers.configuration(REDUCED)["text_config"]
    input_ids = torch.tensor([11], dtype=torch.int32)
    conv = torch.zeros(3, text["kda_num_heads"], text["kda_head_dim"], text["kda_conv_kernel_size"] - 1,
                       dtype=torch.bfloat16)
    recurrent = torch.zeros(1, text["kda_num_heads"], text["kda_head_dim"], text["kda_head_dim"],
                            dtype=torch.float32)
    reference_logits, _, reference_conv, reference_recurrent = TorchEvaluator(
        CheckpointWeightResolver(checkpoint)).run(imported, {"input_ids": input_ids, "kda_conv_state": conv,
                                                             "kda_recurrent_state": recurrent})

    runtime = load(artifact, device="cuda:0")
    device_ids = input_ids.to("cuda:0")
    device_conv = conv.to("cuda:0")
    device_rec = recurrent.to("cuda:0")
    out_logits = torch.empty((1, text["vocab_size"]), dtype=torch.float32, device="cuda:0")
    out_token = torch.empty((1, ), dtype=torch.int32, device="cuda:0")
    out_conv = torch.empty_like(device_conv)
    out_rec = torch.empty_like(device_rec)
    runtime.prepare(device_ids, device_conv, device_rec, out_logits, out_token, out_conv, out_rec)
    runtime.run_into(device_ids, device_conv, device_rec, out_logits, out_token, out_conv, out_rec)
    torch.cuda.synchronize()

    torch.testing.assert_close(out_logits.cpu(), reference_logits, **ARTIFACT_TOLERANCE)
    torch.testing.assert_close(out_conv.cpu(), reference_conv, **ARTIFACT_TOLERANCE)
    torch.testing.assert_close(out_rec.cpu(), reference_recurrent, **ARTIFACT_TOLERANCE)


def test_full_glm_import_covers_the_declared_layer_schedule():
    checkpoint = glm_helpers.checkpoint(REDUCED, full_model=True)
    module = import_model(checkpoint)
    importer = Glm5MoeModelImporter(checkpoint)

    assert module.metadata["imported_layer_indices"] == tuple(range(8))
    assert module.metadata["mla_layer_indices"] == (3, 7)
    assert module.metadata["kda_layer_indices"] == (0, 1, 2, 4, 5, 6)
    assert len(module.function_map["main"].outputs) == 2 + 2 * len(importer.kda_layers) + 1

    text = glm_helpers.configuration(REDUCED)["text_config"]
    inputs = {
        "input_ids": torch.tensor([11], dtype=torch.int32),
        "paged_attention_state": create_paged_attention_state(importer.paged_config),
    }
    for layer in importer.kda_layers:
        qkv = text["kda_num_heads"] * text["kda_head_dim"]
        inputs[f"kda_conv_state_{layer}"] = torch.zeros(
            (3, text["kda_num_heads"], text["kda_head_dim"], text["kda_conv_kernel_size"] - 1),
            dtype=torch.bfloat16)
        inputs[f"kda_recurrent_state_{layer}"] = torch.zeros(
            (1, text["kda_num_heads"], text["kda_head_dim"], text["kda_head_dim"]), dtype=torch.float32)
    outputs = TorchEvaluator(CheckpointWeightResolver(checkpoint)).run(module, inputs)
    assert outputs[0].shape == (1, text["vocab_size"])
    assert outputs[1].shape == (1,)


def test_published_glm_contract_uses_real_dimensions():
    """The metadata contract records the real vocab and published layer sizes.

    Executable numeric tests intentionally use ``REDUCED``.  A synthetic
    random checkpoint at the full GLM vocabulary is a multi-gigabyte fixture,
    and the four-GPU compiler path is not implemented yet.
    """
    checkpoint = glm_helpers.checkpoint(PUBLISHED, with_values=False, kind="kda", layer=4)
    importer = Glm5MoeImporter(checkpoint, layer=4)
    module = importer.import_module()
    assert module.metadata["vocab_size"] == 154880
    assert module.metadata["hidden_size"] == 4096
    assert module.metadata["storage_dtype"] == "bfloat16"
