# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Small-shape H800 integration for the model-neutral TIR call-graph runtime."""

import pytest

from triton.flagmega.artifacts import write_artifact
from triton.flagmega.compiler import Compiler
from triton.flagmega.evaluator import (
    CheckpointWeightResolver,
    TorchEvaluator,
    create_paged_attention_state,
)
from triton.flagmega.importer import import_qwen3_model
from triton.flagmega.runtime import load as load_runtime

from .helpers import full_checkpoint


torch = pytest.importorskip("torch")


def test_two_layer_call_graph_runtime_matches_evaluator_on_h800(tmp_path):
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0):
        pytest.skip("SM90 CUDA is required")
    checkpoint = full_checkpoint(num_hidden_layers=2)
    imported = import_qwen3_model(checkpoint)
    compiled = Compiler().compile(imported).module
    artifact = write_artifact(
        compiled,
        tmp_path / "two-layer-call-graph",
        target="nvidia-sm90",
        checkpoint=checkpoint,
        emit_executable=True,
    )
    runtime = load_runtime(artifact, device="cuda:0")
    # SAT workspaces have no initialization contract. Poison every byte so a
    # kernel that reads padding or a supposedly defined range before its
    # producer writes it fails deterministically instead of depending on the
    # CUDA caching allocator's previous occupant.
    runtime._pool_values["workspace"].fill_(0xA5)
    state = runtime.create_state()
    reference_state = create_paged_attention_state(
        runtime.state_config, device="cuda:0"
    )
    input_ids = torch.tensor([3], dtype=torch.int32, device="cuda:0")
    evaluator = TorchEvaluator(
        CheckpointWeightResolver(checkpoint, device="cuda:0")
    )
    expected_logits, expected_token, _ = evaluator.run(
        imported,
        {
            "input_ids": input_ids,
            "paged_attention_kv_cache": reference_state,
        },
    )
    logits, next_token = runtime.create_outputs()

    runtime.prepare(
        input_ids, state, logits=logits, next_token=next_token
    )
    runtime.run_into(logits, next_token, input_ids, state)
    torch.cuda.synchronize()

    assert runtime.resource_report["spill_bytes"] == 0
    torch.testing.assert_close(logits, expected_logits, atol=0.15, rtol=0.04)
    assert next_token.item() == expected_token.item()
    assert torch.isfinite(state.kv_caches.float()).all()
    # The generated kernels and torch evaluator reduce FP32 values in
    # different legal orders before rounding to BF16. Compare the state with
    # an explicit BF16 numerical contract rather than float32 defaults.  One
    # BF16 ULP at unit magnitude is 2^-7; the deterministic difference is
    # bounded by that representational step.
    torch.testing.assert_close(
        state.kv_caches,
        reference_state.kv_caches,
        atol=1 / 128,
        rtol=0.04,
    )
    diagnostic_views = runtime.diagnostic_workspace()
    assert diagnostic_views["token_embedding"].shape == (1, 16)
    assert any(
        name.startswith("layer_0_decode_layer_call::decode_layer_")
        for name in diagnostic_views
    )
    assert any(
        name.startswith("layer_1_decode_layer_call::decode_layer_")
        for name in diagnostic_views
    )
    assert all(value.device == logits.device for value in diagnostic_views.values())
    assert state.sequence_length == reference_state.sequence_length == 1
