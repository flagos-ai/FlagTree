# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""FlagMega-on-Radeon end-to-end: a real Qwen3 bf16 model compiles through the
amd-gfx1201 target, renders an RDNA megakernel (tile core / pipe / warp
specialize), runs on an AMD RDNA GPU, and matches the torch evaluator."""

import pytest

from triton.flagmega.artifacts import write_artifact
from triton.flagmega.compiler import Compiler
from triton.flagmega.options import CompileOptions
from triton.flagmega.evaluator import (
    CheckpointWeightResolver,
    TorchEvaluator,
    create_paged_attention_state,
)
from triton.flagmega.importer import import_qwen3_model
from triton.flagmega.runtime import load as load_runtime

from .helpers import full_checkpoint

torch = pytest.importorskip("torch")


def _amd_rdna_target():
    if not torch.cuda.is_available() or getattr(torch.version, "hip", None) is None:
        return None
    # Read the real gfx arch; gcnArchName may carry feature suffixes, e.g.
    # "gfx1100:sramecc+:xnack-". Map only known RDNA parts -- an unknown arch
    # returns None so the test skips rather than silently compiling for the
    # wrong target.
    arch = str(getattr(torch.cuda.get_device_properties(0), "gcnArchName", "") or "").split(":")[0]
    return {"gfx1100": "amd-gfx1100", "gfx1201": "amd-gfx1201"}.get(arch)


def test_two_layer_qwen3_runs_on_amd_rdna(tmp_path):
    target = _amd_rdna_target()
    if target is None:
        pytest.skip("AMD RDNA (ROCm/HIP) GPU is required")

    checkpoint = full_checkpoint(num_hidden_layers=2)
    imported = import_qwen3_model(checkpoint)
    compiled = Compiler(CompileOptions(target=target)).compile(imported).module
    assert compiled.stage == "bufferized_tir"

    artifact = write_artifact(
        compiled,
        tmp_path / "qwen3-amd",
        target=target,
        checkpoint=checkpoint,
        emit_executable=True,
    )
    runtime = load_runtime(artifact, device="cuda:0")
    runtime._pool_values["workspace"].fill_(0xA5)

    state = runtime.create_state()
    reference_state = create_paged_attention_state(runtime.state_config, device="cuda:0")
    input_ids = torch.tensor([3], dtype=torch.int32, device="cuda:0")

    evaluator = TorchEvaluator(CheckpointWeightResolver(checkpoint, device="cuda:0"))
    expected_logits, expected_token, _ = evaluator.run(
        imported,
        {"input_ids": input_ids, "paged_attention_kv_cache": reference_state},
    )

    logits, next_token = runtime.create_outputs()
    runtime.prepare(input_ids, state, logits=logits, next_token=next_token)
    runtime.run_into(logits, next_token, input_ids, state)
    torch.cuda.synchronize()

    torch.testing.assert_close(logits, expected_logits, atol=0.15, rtol=0.04)
    assert next_token.item() == expected_token.item()
