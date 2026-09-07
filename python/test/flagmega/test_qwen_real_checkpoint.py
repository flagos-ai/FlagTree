# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Opt-in real Qwen3.8-27B-FP8 layer-0 compile/run regression."""

from __future__ import annotations

import os
import time
from dataclasses import replace

import pytest

from triton.flagmega.artifacts import write_artifact
from triton.flagmega.compiler import Compiler
from triton.flagmega.evaluator import CheckpointWeightResolver, TorchEvaluator, create_gdn_state
from triton.flagmega.importer import DirectoryCheckpoint, Qwen35Layer0Importer
from triton.flagmega.runtime import load as load_runtime
from triton.flagmega.passes import decompose_gated_delta_net


torch = pytest.importorskip("torch")


@pytest.mark.skipif(
    not os.environ.get("FLAGMEGA_QWEN3_8_FP8"),
    reason="set FLAGMEGA_QWEN3_8_FP8 to the official layer-0 checkpoint directory",
)
def test_real_qwen3_8_27b_fp8_layer0_two_step_compile_run_and_reference(tmp_path):
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0):
        pytest.skip("SM90 CUDA is required")
    checkpoint = DirectoryCheckpoint(os.environ["FLAGMEGA_QWEN3_8_FP8"])
    importer = Qwen35Layer0Importer(
        checkpoint,
        revision="017b9c7af6b5689d5dd426a76e0bc077eb5ca20a",
    )
    imported = importer.import_module()
    compile_started = time.perf_counter()
    compiled = Compiler().compile(imported).module
    artifact = write_artifact(
        compiled,
        tmp_path / "qwen3.8-27b-fp8-layer0",
        target="nvidia-sm90",
        checkpoint=checkpoint,
        emit_executable=True,
    )
    compile_seconds = time.perf_counter() - compile_started

    runtime = load_runtime(artifact, device="cuda:0")
    runtime_state = runtime.create_state()
    input_ids = torch.tensor([1234], dtype=torch.int32, device="cuda:0")
    output = torch.empty((1, 5120), dtype=torch.bfloat16, device="cuda:0")
    prepare_started = time.perf_counter()
    runtime.prepare(input_ids, runtime_state, output=output)
    torch.cuda.synchronize()
    prepare_seconds = time.perf_counter() - prepare_started

    evaluator = TorchEvaluator(CheckpointWeightResolver(checkpoint, device="cuda:0"))
    decomposed = decompose_gated_delta_net(replace(imported, stage="extracted"))
    reference_state = create_gdn_state(importer.config, device="cuda:0")
    measurements = []
    for step in range(2):
        (reference_output, _), trace = evaluator.run_with_trace(
            decomposed,
            {"input_ids": input_ids, "gated_delta_net_state": reference_state},
        )
        torch.cuda.synchronize()
        started = time.perf_counter()
        runtime.run_into(output, input_ids, runtime_state)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        observed_output = output.clone()
        observed_convolution = runtime_state.convolution.clone()
        observed_recurrent = runtime_state.recurrent.clone()
        expected_output = reference_output.clone()
        expected_convolution = reference_state.convolution.clone()
        expected_recurrent = reference_state.recurrent.clone()
        torch.cuda.synchronize()
        hidden_error = (observed_output.float() - expected_output.float()).abs()
        recurrent_error = (observed_recurrent - expected_recurrent).abs()
        cosine = torch.nn.functional.cosine_similarity(
            observed_output.float().reshape(1, -1), expected_output.float().reshape(1, -1)).item()
        measurements.append({
            "step": step,
            "seconds": elapsed,
            "hidden_max_abs": hidden_error.max().item(),
            "hidden_cosine": cosine,
            "recurrent_max_abs": recurrent_error.max().item(),
        })
        if step == 1:
            workspace = runtime.diagnostic_workspace()
            convolution_id = next(
                node.id for node in decomposed.nodes if node.op == "nn.gdn_convolution")
            prefix = convolution_id.rsplit('.', 1)[0]
            measurements[-1]["recurrent_input_max_abs"] = {
                "query": (workspace[f"{convolution_id}.0"].float() - trace[f"{prefix}.query"].float()).abs().max().item(),
                "key": (workspace[f"{convolution_id}.1"].float() - trace[f"{prefix}.key"].float()).abs().max().item(),
                "gate": (workspace[f"{convolution_id}.3"].float() - trace[f"{prefix}.gate"].float()).abs().max().item(),
                "beta": (
                    torch.sigmoid(workspace[f"{convolution_id}.4"].to(torch.bfloat16)).float()
                    - trace[f"{prefix}.beta"].float()
                ).abs().max().item(),
                "decay_input": (
                    workspace[f"{convolution_id}.5"].float() - trace[f"{prefix}.decay"].float()
                ).abs().max().item(),
            }
            print(measurements[-1])
        torch.testing.assert_close(observed_output.float(), expected_output.float(), atol=4e-2, rtol=2e-2)
        torch.testing.assert_close(
            observed_convolution.float(), expected_convolution.float(), atol=1e-1, rtol=2e-2)
        torch.testing.assert_close(observed_recurrent, expected_recurrent, atol=1.2e-2, rtol=5e-3)
        assert cosine >= 0.999
        if step == 0:
            workspace = runtime.diagnostic_workspace()
            convolution_id = next(
                node.id for node in decomposed.nodes if node.op == "nn.gdn_convolution")
            recurrent_id = next(
                node.id for node in decomposed.nodes if node.op == "nn.gdn_recurrent_core")
            comparisons = {
                "token_embedding": (workspace["token_embedding"], trace["token_embedding"]),
                "input_norm": (workspace["input_norm"], trace["input_norm"]),
                "query": (workspace[f"{convolution_id}.0"], trace[f"{convolution_id.rsplit('.', 1)[0]}.query"]),
                "key": (workspace[f"{convolution_id}.1"], trace[f"{convolution_id.rsplit('.', 1)[0]}.key"]),
                "gate": (workspace[f"{convolution_id}.3"], trace[f"{convolution_id.rsplit('.', 1)[0]}.gate"]),
                "beta": (
                    torch.sigmoid(workspace[f"{convolution_id}.4"].to(torch.bfloat16)),
                    trace[f"{convolution_id.rsplit('.', 1)[0]}.beta"],
                ),
                "decay": (workspace[f"{convolution_id}.5"], trace[f"{convolution_id.rsplit('.', 1)[0]}.decay"]),
                "gdn_output": (workspace[f"{recurrent_id}.0"], trace["gdn_output"]),
                "after_gdn": (workspace["after_gdn"], trace["after_gdn"]),
                "post_norm": (workspace["post_attention_norm"], trace["post_attention_norm"]),
                "mlp_glu": (workspace["mlp_gate_up"], trace["mlp_gate_up"]),
                "mlp_down": (workspace["mlp_down"], trace["mlp_down"]),
            }
            measurements[-1]["intermediate_max_abs"] = {
                name: (actual.float() - expected.float()).abs().max().item()
                for name, (actual, expected) in comparisons.items()
            }
            tolerance = 2e-2 + 2e-2 * reference_output.float().abs()
            mismatch_indices = torch.nonzero(hidden_error > tolerance, as_tuple=False)[:, 1]
            measurements[-1]["mismatches"] = [
                {
                    "index": int(index),
                    "actual": output[0, index].float().item(),
                    "expected": reference_output[0, index].float().item(),
                    "actual_after_gdn": workspace["after_gdn"][0, index].float().item(),
                    "expected_after_gdn": trace["after_gdn"][0, index].float().item(),
                    "actual_mlp_down": workspace["mlp_down"][0, index].float().item(),
                    "expected_mlp_down": trace["mlp_down"][0, index].float().item(),
                }
                for index in mismatch_indices
            ]
            print(measurements[-1])

    assert runtime.prepare_count == 1
    assert runtime.resource_report["spill_bytes"] == 0
    print({
        "compile_seconds": compile_seconds,
        "prepare_seconds": prepare_seconds,
        "resources": runtime.resource_report,
        "measurements": measurements,
    })
