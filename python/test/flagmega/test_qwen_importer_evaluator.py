# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from triton.flagmega.errors import ArtifactError, CodegenError, ImporterError
from triton.flagmega.compiler import Compiler
from triton.flagmega.artifacts import load_artifact, verify_rdata, write_artifact
from triton.flagmega.evaluator import (
    CheckpointWeightResolver,
    TorchEvaluator,
    block_scaled_linear,
    create_gdn_state,
    rms_norm,
)
from triton.flagmega.importer import DirectoryCheckpoint, MemoryCheckpoint, Qwen35Layer0Importer, TensorInfo
from triton.flagmega.ir import DType, verify_buffer_plan
from triton.flagmega.passes import decompose_gated_delta_net
from triton.flagmega.runtime import (
    GeneratedTirGatedDeltaNetModule,
    create_tir_runtime,
    load as load_runtime,
)
from triton.flagmega.codegen.triton import describe_tir_package
from triton.flagmega.codegen.triton.distribution import requires_distributed_grid


torch = pytest.importorskip("torch")


_PREFIX = "model.language_model.layers.0."
_EMBEDDING_KEY = "model.language_model.embed_tokens.weight"


def _config() -> dict[str, object]:
    return {
        "model_type": "qwen3_5",
        "architectures": ["Qwen3_5ForConditionalGeneration"],
        "layer_types": ["linear"],
        "vocab_size": 32,
        "pad_token_id": 0,
        "num_hidden_layers": 2,
        "hidden_size": 8,
        "intermediate_size": 16,
        "linear_num_key_heads": 1,
        "linear_num_value_heads": 2,
        "linear_key_head_dim": 4,
        "linear_value_head_dim": 4,
        "linear_conv_kernel_dim": 3,
        "rms_norm_eps": 1e-6,
        "quantization_config": {
            "quant_method": "fp8",
            "activation_scheme": "dynamic",
            "fmt": "e4m3",
            "weight_block_size": [4, 4],
        },
    }


def _tensor_specs() -> dict[str, tuple[tuple[int, ...], DType]]:
    relative = {
        "input_layernorm.weight": ((8,), DType.BFLOAT16),
        "linear_attn.in_proj_qkv.weight": ((16, 8), DType.FLOAT8_E4M3FN),
        "linear_attn.in_proj_qkv.weight_scale_inv": ((4, 2), DType.FLOAT32),
        "linear_attn.in_proj_z.weight": ((8, 8), DType.FLOAT8_E4M3FN),
        "linear_attn.in_proj_z.weight_scale_inv": ((2, 2), DType.FLOAT32),
        "linear_attn.in_proj_b.weight": ((2, 8), DType.BFLOAT16),
        "linear_attn.in_proj_a.weight": ((2, 8), DType.BFLOAT16),
        "linear_attn.conv1d.weight": ((16, 1, 3), DType.BFLOAT16),
        "linear_attn.A_log": ((2,), DType.FLOAT32),
        "linear_attn.dt_bias": ((2,), DType.FLOAT32),
        "linear_attn.norm.weight": ((4,), DType.BFLOAT16),
        "linear_attn.out_proj.weight": ((8, 8), DType.FLOAT8_E4M3FN),
        "linear_attn.out_proj.weight_scale_inv": ((2, 2), DType.FLOAT32),
        "post_attention_layernorm.weight": ((8,), DType.BFLOAT16),
        "mlp.gate_proj.weight": ((16, 8), DType.FLOAT8_E4M3FN),
        "mlp.gate_proj.weight_scale_inv": ((4, 2), DType.FLOAT32),
        "mlp.up_proj.weight": ((16, 8), DType.FLOAT8_E4M3FN),
        "mlp.up_proj.weight_scale_inv": ((4, 2), DType.FLOAT32),
        "mlp.down_proj.weight": ((8, 16), DType.FLOAT8_E4M3FN),
        "mlp.down_proj.weight_scale_inv": ((2, 4), DType.FLOAT32),
    }
    return {
        _EMBEDDING_KEY: ((32, 8), DType.BFLOAT16),
        **{_PREFIX + name: spec for name, spec in relative.items()},
    }


def _torch_dtype(dtype: DType):
    return {
        DType.BFLOAT16: torch.bfloat16,
        DType.FLOAT32: torch.float32,
        DType.FLOAT8_E4M3FN: torch.float8_e4m3fn,
    }[dtype]


def _memory_checkpoint(*, wrong_qkv_scale: bool = False) -> MemoryCheckpoint:
    generator = torch.Generator().manual_seed(20250831)
    infos = {}
    values = {}
    for key, (shape, dtype) in _tensor_specs().items():
        if wrong_qkv_scale and key.endswith("in_proj_qkv.weight_scale_inv"):
            shape = (1, 1)
        infos[key] = TensorInfo(key, dtype, shape, "layer-0.safetensors")
        if dtype == DType.FLOAT8_E4M3FN:
            value = (torch.randn(shape, generator=generator) * 0.125).to(torch.float8_e4m3fn)
        elif key.endswith("weight_scale_inv"):
            value = torch.full(shape, 0.125, dtype=torch.float32)
        elif key.endswith("layernorm.weight") or key.endswith("linear_attn.norm.weight"):
            value = torch.zeros(shape, dtype=_torch_dtype(dtype))
        elif key.endswith("A_log"):
            value = torch.zeros(shape, dtype=torch.float32)
        else:
            value = (torch.randn(shape, generator=generator) * 0.05).to(_torch_dtype(dtype))
        values[key] = value
    return MemoryCheckpoint(_config(), infos, values)


def test_qwen_layer0_import_builds_stateful_fp8_module():
    checkpoint = _memory_checkpoint()
    importer = Qwen35Layer0Importer(checkpoint, revision="unit-revision")
    module = importer.import_module()

    assert module.stage == "imported"
    assert module.dialect == "high_level"
    assert module.metadata["architecture"] == "Qwen3_5ForConditionalGeneration"
    assert module.metadata["revision"] == "unit-revision"
    assert module.function_map["main"].parameters == ("input_ids", "gdn_state")
    assert module.function_map["main"].outputs == ("output", "updated_state")
    assert len([node for node in module.nodes if node.op == "builtin.weight"]) == 21
    assert module.node_map["token_embedding"].op == "nn.embedding"
    assert module.node_map["token_embedding"].attrs["padding_idx"] == 0
    gdn = module.node_map["gated_delta_net"]
    assert gdn.effect.resource == "gated_delta_net_state"
    assert gdn.attrs["num_value_heads"] == 2


def test_qwen_importer_accepts_huggingface_text_config_nesting():
    checkpoint = _memory_checkpoint()
    nested_config = {
        "model_type": "qwen3_5",
        "architectures": ["Qwen3_5ForConditionalGeneration"],
        "text_config": {
            key: value
            for key, value in checkpoint.config.items()
            if key not in {"model_type", "architectures", "quantization_config"}
        },
        "quantization_config": checkpoint.config["quantization_config"],
    }
    nested_checkpoint = MemoryCheckpoint(
        nested_config,
        {key: checkpoint.tensor_info(key) for key in checkpoint.keys},
        {key: checkpoint.load_tensor(key) for key in checkpoint.keys},
    )

    importer = Qwen35Layer0Importer(nested_checkpoint)

    assert importer.config.hidden_size == 8
    assert importer.config.num_value_heads == 2
    assert importer.import_module().metadata["hidden_size"] == 8


def test_qwen_importer_distinguishes_language_layer_from_mtp_layer_namespace():
    checkpoint = _memory_checkpoint()
    infos = {key: checkpoint.tensor_info(key) for key in checkpoint.keys}
    decoy = "mtp.layers.0.input_layernorm.weight"
    infos[decoy] = TensorInfo(decoy, DType.BFLOAT16, (8,), "mtp.safetensors")
    with_mtp = MemoryCheckpoint(
        checkpoint.config,
        infos,
        {key: checkpoint.load_tensor(key) for key in checkpoint.keys},
    )

    importer = Qwen35Layer0Importer(with_mtp)

    assert importer.layer_prefix == _PREFIX


def test_qwen_layer0_evaluator_updates_explicit_state_and_is_finite():
    checkpoint = _memory_checkpoint()
    importer = Qwen35Layer0Importer(checkpoint)
    module = importer.import_module()
    evaluator = TorchEvaluator(CheckpointWeightResolver(checkpoint))
    state = create_gdn_state(importer.config)
    input_ids = torch.tensor([7], dtype=torch.int32)

    output0, returned_state0 = evaluator.run(module, {"input_ids": input_ids, "gated_delta_net_state": state})
    recurrent0 = state.recurrent.clone()
    convolution0 = state.convolution.clone()
    output1, returned_state1 = evaluator.run(module, {"input_ids": input_ids, "gated_delta_net_state": state})

    assert output0.shape == (1, 8)
    assert output0.dtype == torch.bfloat16
    assert torch.isfinite(output0.float()).all()
    assert torch.isfinite(output1.float()).all()
    assert returned_state0 is state
    assert returned_state1 is state
    assert not torch.equal(convolution0, state.convolution)
    assert not torch.equal(recurrent0, state.recurrent)


def test_gdn_decomposition_preserves_two_state_effect_boundaries_and_values():
    checkpoint = _memory_checkpoint()
    importer = Qwen35Layer0Importer(checkpoint)
    imported = importer.import_module()
    decomposed = decompose_gated_delta_net(replace(imported, stage="extracted"))
    evaluator = TorchEvaluator(CheckpointWeightResolver(checkpoint))
    input_ids = torch.tensor([17], dtype=torch.int32)
    imported_state = create_gdn_state(importer.config)
    decomposed_state = imported_state.clone()

    imported_output, _ = evaluator.run(
        imported, {"input_ids": input_ids, "gated_delta_net_state": imported_state})
    decomposed_output, _ = evaluator.run(
        decomposed, {"input_ids": input_ids, "gated_delta_net_state": decomposed_state})

    assert len([node for node in decomposed.nodes if node.op == "nn.gdn_convolution"]) == 1
    assert len([node for node in decomposed.nodes if node.op == "nn.gdn_recurrent_core"]) == 1
    torch.testing.assert_close(decomposed_output, imported_output)
    torch.testing.assert_close(decomposed_state.convolution, imported_state.convolution)
    torch.testing.assert_close(decomposed_state.recurrent, imported_state.recurrent)


def test_qwen_layer_compiles_to_single_entry_bufferized_tir_without_agent():
    module = Qwen35Layer0Importer(_memory_checkpoint()).import_module()

    result = Compiler().compile(module)
    plan = verify_buffer_plan(result.module)

    assert result.module.stage == "bufferized_tir"
    assert result.module.dialect == "bufferized_tir"
    assert result.module.metadata["launch_contract"]["kind"] == "single_prepared_entry"
    assert result.module.metadata["launch_contract"]["ordinary_launch_allowed"] is False
    assert result.module.metadata["launch_contract"]["cooperative_grid"] is False
    distribution_records = [
        record for record in result.module.selections
        if record.point_id.startswith("distribution.")
    ]
    assert distribution_records
    # Provider ids are descriptive and need not end in ``.replicated``.
    # The materialized SBP types, rather than an id naming convention,
    # define whether the program actually needs a distributed grid.
    assert not requires_distributed_grid(result.module)
    assert result.module.metadata["auto_distribution"]["solver"] == "ortools-cp-sat"
    package = result.module.metadata["codegen_package_plan"]
    assert package["kind"] == "tir_call_graph"
    assert any(
        call["family"] == "block_fp8" and call["variant"] == "mma"
        for call in package["calls"]
    )
    descriptor = describe_tir_package(result.module)
    assert descriptor["grid"] == [1, 1, 1]
    assert descriptor["num_warps"] == result.module.metadata["launch_contract"]["num_warps"]
    runtime = create_tir_runtime(
        Path("."),
        {"codegen": {**descriptor, "kind": "tir_call_graph/v1"}},
        result.module,
        object(),
    )
    assert isinstance(runtime, GeneratedTirGatedDeltaNetModule)
    external = [
        argument["buffer"] for argument in runtime.external_arguments
    ]
    assert external[:3] == [
        "input_ids",
        "gdn_state.convolution",
        "gdn_state.recurrent",
    ]
    output_id = result.module.function_map[result.module.entry].outputs[0]
    assert external[-1] == dict(plan.entry_outputs)[output_id][0]
    assert runtime.external_arguments[-1]["role"] == "result"
    assert all(
        record.candidate_id == "packing.logical"
        for point_id, record in result.module.selection_map.items()
        if point_id.startswith("packing.")
    )
    assert all(
        node.op in {
            "builtin.var", "builtin.get_item", "tir.buffer", "tir.buffer_view", "tir.call"
        }
        for node in result.module.nodes
    )
    sharded_views = [
        node for node in result.module.nodes if node.op == "tir.buffer_view"
    ]
    assert sharded_views
    # This small graph is fully replicated, so only logical/vector ABI views
    # remain; no artificial sharded-storage view is introduced.
    assert {
        node.attrs["alias_kind"] for node in sharded_views
    } == {"vector_reinterpret"}
    # The explicit NormApply retains its verified-zero bias as a semantic TIR
    # operand.  Unlike the old re-fused RMSNorm path, it must therefore remain
    # in rdata beside the 21 checkpoint-derived buffers.
    assert len([node for node in result.module.nodes if node.op == "tir.buffer"]) == 22
    assert plan.workspace_bytes > 0
    input_state_buffers = dict(plan.entry_inputs)["gdn_state"]
    output_state_buffers = dict(plan.entry_outputs)["updated_state"]
    assert output_state_buffers == input_state_buffers


def test_qwen_artifact_packs_and_verifies_deterministic_rdata(tmp_path):
    checkpoint = _memory_checkpoint()
    module = Compiler().compile(Qwen35Layer0Importer(checkpoint).import_module()).module

    artifact = write_artifact(module, tmp_path / "artifact", target="nvidia-sm90", checkpoint=checkpoint)
    index = verify_rdata(artifact / "assets")
    manifest, loaded = load_artifact(artifact)

    assert len(index["entries"]) == 22
    assert index["nbytes"] == module.metadata["buffer_plan"]["rdata_bytes"]
    assert manifest["rdata"]["sha256"] == index["sha256"]
    assert loaded.semantic_hash == module.semantic_hash


def test_qwen_executable_requires_explicit_checkpoint_for_rdata(tmp_path):
    module = Compiler().compile(Qwen35Layer0Importer(_memory_checkpoint()).import_module()).module

    with pytest.raises(ArtifactError, match="--checkpoint"):
        write_artifact(
            module,
            tmp_path / "missing-rdata",
            target="nvidia-sm90",
            emit_executable=True,
        )


def test_tir_renderer_rejects_an_unmaterialized_agent_override():
    module = Compiler().compile(Qwen35Layer0Importer(_memory_checkpoint()).import_module()).module
    overridden = replace(
        module,
        selections=tuple(
            replace(record, candidate_id="tir.block_fp8.simt", origin="agent", policy="agent")
            if record.point_id == "tir.mlp_down"
            else record
            for record in module.selections
        ),
    )

    with pytest.raises(CodegenError, match="does not match materialized KernelDispatch"):
        describe_tir_package(overridden)


def test_synthetic_qwen_executable_matches_reference_on_h800(tmp_path):
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0):
        pytest.skip("SM90 CUDA is required for the Qwen executable integration test")
    checkpoint = _memory_checkpoint()
    importer = Qwen35Layer0Importer(checkpoint)
    imported = importer.import_module()
    compiled = Compiler().compile(imported).module
    artifact = write_artifact(
        compiled,
        tmp_path / "qwen-artifact",
        target="nvidia-sm90",
        checkpoint=checkpoint,
        emit_executable=True,
    )
    manifest, _ = load_artifact(artifact)
    assert manifest["codegen"]["kind"] == "tir_call_graph/v1"
    assert manifest["codegen"]["renderer_spec"] == "bufferized-tir"
    assert "source_provenance" not in manifest["codegen"]
    assert {
        value["kernel"]
        for value in manifest["codegen"]["kernel_template_specs"]
    } == {
        "block_fp8",
        "distributed_boxing",
        "elementwise",
        "embedding",
        "gdn_convolution",
        "gdn_recurrent",
        "matmul_glu",
        "norm_apply",
        "norm_stats",
    }
    runtime = load_runtime(artifact, device="cuda:0")
    runtime_state = runtime.create_state()
    input_ids_cpu = torch.tensor([27], dtype=torch.int32)
    reference_state = create_gdn_state(importer.config)
    reference_output, _ = TorchEvaluator(CheckpointWeightResolver(checkpoint)).run(
        imported,
        {"input_ids": input_ids_cpu, "gated_delta_net_state": reference_state},
    )
    input_ids_cuda = input_ids_cpu.to("cuda:0")
    output_cuda = torch.empty((1, 8), dtype=torch.bfloat16, device="cuda:0")

    runtime.prepare(input_ids_cuda, runtime_state, output=output_cuda)
    runtime.run_into(output_cuda, input_ids_cuda, runtime_state)
    torch.cuda.synchronize()

    assert runtime.prepare_count == 1
    torch.testing.assert_close(output_cuda.cpu().float(), reference_output.float(), atol=0.05, rtol=0.05)
    torch.testing.assert_close(
        runtime_state.convolution.cpu().float(), reference_state.convolution.float(), atol=0.02, rtol=0.02)
    torch.testing.assert_close(
        runtime_state.recurrent.cpu(), reference_state.recurrent, atol=0.02, rtol=0.02)

    reference_output_step2, _ = TorchEvaluator(CheckpointWeightResolver(checkpoint)).run(
        imported,
        {"input_ids": input_ids_cpu, "gated_delta_net_state": reference_state},
    )
    runtime.run_into(output_cuda, input_ids_cuda, runtime_state)
    torch.cuda.synchronize()

    assert runtime.prepare_count == 1
    torch.testing.assert_close(output_cuda.cpu().float(), reference_output_step2.float(), atol=0.05, rtol=0.05)
    torch.testing.assert_close(
        runtime_state.convolution.cpu().float(), reference_state.convolution.float(), atol=0.02, rtol=0.02)
    torch.testing.assert_close(
        runtime_state.recurrent.cpu(), reference_state.recurrent, atol=0.02, rtol=0.02)


def test_rms_norm_matches_explicit_formula():
    hidden = torch.tensor([[0.25, -0.5, 1.0, -2.0]], dtype=torch.bfloat16)
    weight = torch.tensor([0.1, -0.2, 0.3, -0.4], dtype=torch.bfloat16)
    actual = rms_norm(hidden, weight, epsilon=1e-6, weight_bias=1.0)
    variance = hidden.float().pow(2).mean(dim=-1, keepdim=True)
    expected = (hidden.float() * torch.rsqrt(variance + 1e-6)).to(torch.bfloat16)
    expected = expected * (weight + 1.0)
    torch.testing.assert_close(actual, expected)


def test_block_scaled_linear_zero_activation_has_no_nan():
    value = torch.zeros((1, 8), dtype=torch.bfloat16)
    weight = torch.ones((8, 8), dtype=torch.float8_e4m3fn)
    scale = torch.ones((2, 2), dtype=torch.float32)
    output = block_scaled_linear(value, weight, scale, block_n=4, block_k=4)
    assert torch.equal(output, torch.zeros_like(output))
    assert torch.isfinite(output.float()).all()


def test_importer_rejects_checkpoint_scale_shape_at_source():
    with pytest.raises(ImporterError, match="weight_scale_inv.*must have shape"):
        Qwen35Layer0Importer(_memory_checkpoint(wrong_qkv_scale=True)).import_module()


def test_directory_checkpoint_reads_safetensors_metadata_and_values(tmp_path):
    safetensors_torch = pytest.importorskip("safetensors.torch")
    (tmp_path / "config.json").write_text(json.dumps(_config()), encoding="utf-8")
    tensor = torch.arange(8, dtype=torch.bfloat16)
    key = _PREFIX + "input_layernorm.weight"
    safetensors_torch.save_file({key: tensor}, tmp_path / "model.safetensors")

    checkpoint = DirectoryCheckpoint(tmp_path)
    info = checkpoint.tensor_info(key)

    assert checkpoint.keys == (key,)
    assert info.dtype == DType.BFLOAT16
    assert info.shape == (8,)
    assert info.source == "model.safetensors"
    torch.testing.assert_close(checkpoint.load_tensor(key), tensor)
