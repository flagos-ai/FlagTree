# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT

import os
from pathlib import Path

import numpy as np
import pytest

from triton.flagmega.evaluator import CheckpointWeightResolver, TorchEvaluator, create_paged_attention_state
from triton.flagmega.importer import DirectoryCheckpoint, Qwen3LayerImporter
from triton.flagmega.artifacts import write_artifact
from triton.flagmega.compiler import Compiler
from triton.flagmega.runtime import load as load_runtime


torch = pytest.importorskip("torch")


MODEL = Path("/root/repos/vllm/build/models/Qwen3-1.7B-1layer")
NNCASE_RESULT = Path("/root/repos/nncase/tests_output/test_qwen3_1_7b_no_lm_head/cpu_result_0_0.npy")
NNCASE_RESULTS = tuple(
    NNCASE_RESULT.with_name(f"cpu_result_{step}_0.npy") for step in range(3)
)
NNCASE_GENERATED_RESULTS = tuple(
    NNCASE_RESULT.parent
    / "infer"
    / "pyntt"
    / "noptq"
    / f"nncase_result_{step}_0.npy"
    for step in range(3)
)


@pytest.mark.skipif(not MODEL.is_dir() or not NNCASE_RESULT.is_file(), reason="local Qwen3/nncase fixtures unavailable")
def test_real_qwen3_1_7b_layer_matches_nncase_reference():
    checkpoint = DirectoryCheckpoint(MODEL)
    importer = Qwen3LayerImporter(checkpoint)
    module = importer.import_module()
    state = create_paged_attention_state(importer.state_config)

    actual, _ = TorchEvaluator(CheckpointWeightResolver(checkpoint)).run(
        module,
        {"input_ids": torch.tensor([151644], dtype=torch.int32), "paged_attention_kv_cache": state},
    )
    expected = torch.from_numpy(np.load(NNCASE_RESULT))

    cosine = torch.nn.functional.cosine_similarity(actual.flatten(), expected.flatten(), dim=0)
    assert cosine.item() > 0.99999
    torch.testing.assert_close(actual, expected, atol=0.07, rtol=0.02)
    assert state.sequence_length == 1


@pytest.mark.skipif(
    not os.environ.get("FLAGMEGA_QWEN3_1_7B_RUNTIME"),
    reason="set FLAGMEGA_QWEN3_1_7B_RUNTIME to the Qwen3-1.7B one-layer checkpoint",
)
def test_real_qwen3_1_7b_generated_kernel_is_zero_spill_and_matches_nncase(tmp_path):
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0):
        pytest.skip("SM90 CUDA is required")
    model = Path(os.environ["FLAGMEGA_QWEN3_1_7B_RUNTIME"])
    checkpoint = DirectoryCheckpoint(model)
    imported = Qwen3LayerImporter(checkpoint).import_module()
    compiled = Compiler().compile(imported).module
    artifact = write_artifact(
        compiled,
        tmp_path / "qwen3-1.7b-layer0",
        target="nvidia-sm90",
        checkpoint=checkpoint,
        emit_executable=True,
    )
    runtime = load_runtime(artifact, device="cuda:0")
    state = runtime.create_state()
    input_ids = torch.tensor([151644], dtype=torch.int32, device="cuda:0")
    output = torch.empty((1, 2048), dtype=torch.float32, device="cuda:0")

    runtime.prepare(input_ids, state, output=output)
    measurements = []
    baselines = []
    for token, reference, nncase_generated in zip(
        (151644, 2176, 2176),
        NNCASE_RESULTS,
        NNCASE_GENERATED_RESULTS,
        strict=True,
    ):
        step_input = torch.tensor([token], dtype=torch.int32, device="cuda:0")
        runtime.run_into(output, step_input, state)
        torch.cuda.synchronize()
        expected = torch.from_numpy(np.load(reference)).to(
            device="cuda:0", dtype=torch.float32).reshape_as(output)
        nncase_output = torch.from_numpy(np.load(nncase_generated)).to(
            device="cuda:0", dtype=torch.float32).reshape_as(output)
        measurements.append((
            torch.nn.functional.cosine_similarity(output.flatten(), expected.flatten(), dim=0).item(),
            (output - expected).abs().max().item(),
        ))
        baselines.append((
            torch.nn.functional.cosine_similarity(
                nncase_output.flatten(), expected.flatten(), dim=0
            ).item(),
            (nncase_output - expected).abs().max().item(),
        ))

    assert runtime.resource_report["spill_bytes"] == 0
    # Different legal FP32 reduction trees need not reproduce the CPU oracle
    # bit-for-bit.  The executable contract is explicit and evidence-based:
    # every step must be at least as accurate as nncase/PyNTT's checked-in GPU
    # result against the same CPU oracle, by both cosine and maximum error.
    assert all(
        cosine >= baseline_cosine
        for (cosine, _), (baseline_cosine, _) in zip(
            measurements, baselines, strict=True
        )
    )
    assert all(
        max_abs <= baseline_max_abs
        for (_, max_abs), (_, baseline_max_abs) in zip(
            measurements, baselines, strict=True
        )
    )
    assert state.sequence_length == 3
