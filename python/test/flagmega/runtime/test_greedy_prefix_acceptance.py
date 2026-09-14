# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT

import importlib.util
import json
from pathlib import Path

import pytest


@pytest.fixture
def runner():
    path = Path(__file__).parents[3] / "tutorials/flagmega/02-qwen3.5-35b-a3b-bf16/nvidia-h800/accuracy.py"
    spec = importlib.util.spec_from_file_location("greedy_prefix_runner", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_acceptance_uses_the_requested_prefix_without_mutating_reference(runner):
    record = {"token_ids": [1, 2, 3, 4, 5]}
    assert runner.prefix_token_ids(record, 3) == [1, 2, 3]
    assert record["token_ids"] == [1, 2, 3, 4, 5]


@pytest.mark.parametrize("count", [0, -1, True, 6])
def test_invalid_or_unavailable_prefix_is_rejected(runner, count):
    with pytest.raises(ValueError):
        runner.prefix_token_ids({"token_ids": [1, 2, 3]}, count)


def test_logits_and_rounding_diagnostics_are_not_acceptance_conditions(runner):
    row = {"initial_state_roundtrip_exact": True, "token_ids": [1, 2, 3], "reference_token_ids": [1, 2, 3],
           "diagnostics": {"logits_exact": False}, "first_decode_state_errors": {"rounding_exact": False}}
    result = runner.acceptance_summary([row], 1, 3)
    assert result["complete"] and result["all_prefixes_match"]
    row["token_ids"][-1] = 9
    assert not runner.acceptance_summary([row], 1, 3)["all_prefixes_match"]


def test_missing_scenarios_or_changed_initial_state_cannot_pass(runner):
    row = {"initial_state_roundtrip_exact": True, "token_ids": [1], "reference_token_ids": [1]}
    assert not runner.acceptance_summary([row], 4, 1)["complete"]
    row["initial_state_roundtrip_exact"] = False
    assert not runner.acceptance_summary([row], 1, 1)["all_prefixes_match"]


def test_empty_results_cannot_pass(runner):
    result = runner.acceptance_summary([], 4, 3)
    assert not result["complete"] and not result["all_prefixes_match"]
    with pytest.raises(ValueError):
        runner.acceptance_summary([], 0, 3)


@pytest.mark.parametrize("prefix,steps", [([], 32), ([1, 2, 3], 2), ([1], True), ([1], 0)])
def test_timing_length_cannot_weaken_the_validated_prefix(runner, prefix, steps):
    path = Path(runner.__file__).with_name("prepared_gpu_timing.py")
    spec = importlib.util.spec_from_file_location("prepared_prefix_timing", path)
    timing = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(timing)
    with pytest.raises(ValueError, match="validated prefix"):
        timing.benchmark(None, (), {}, {}, prefix, "token", repeats=1, steps=steps)


@pytest.mark.parametrize("version,tokens,accepted", [(1, 32, True), (2, 32, True), (2, 3, False)])
def test_performance_comparison_accepts_both_schemas_but_not_a_shorter_prefix(
        runner, tmp_path, version, tokens, accepted):
    path = Path(runner.__file__).with_name("render_results.py")
    spec = importlib.util.spec_from_file_location("prefix_comparison_renderer", path)
    renderer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(renderer)
    expected = {1: {"state_sha256": "state", "token_ids": list(range(32))}}
    row = {"prefix_length": 1, "initial_state_sha256": "state", "initial_state_roundtrip_exact": True,
           "token_ids": list(range(tokens)), "reference_token_ids": list(range(tokens)), "exact_match": True}
    accuracy = {"schema": f"flagmega.prepared-decode-candidate/v{version}", "complete": True,
                "reference_sha256": "reference", "semantic_hash": "ir", "source_sha256": "source",
                "records": [row]}
    if version == 1:
        accuracy["all_sequences_match"] = True
    else:
        accuracy.update(runner.acceptance_summary([row], 1, tokens))
    timing = {"schema": "flagmega.prepared-decode-gpu-timing/v1", "performance_valid": True,
              "semantic_hash": "ir", "source_sha256": "source", "device": "H800", "torch": "test",
              "scenarios": [{"prefix_length": 1, "serving_latency": False, "all_sequences_match": True,
                             "cache_policy": "natural sequential decode; no synthetic flush", "warmup_rounds": 2,
                             "rounds": [{"token_ids": list(range(tokens)), "samples_ms": [1.0] * tokens}] * 5}]}
    if version == 2:
        timing["acceptance"] = accuracy["acceptance"]
    (tmp_path / "report.json").write_text(json.dumps(accuracy))
    (tmp_path / "gpu-timing.json").write_text(json.dumps(timing))
    if accepted:
        samples, _, _ = renderer.read_trial(tmp_path, "baseline", "reference", expected)
        assert len(samples[1]) == 160
    else:
        with pytest.raises(ValueError, match="Short-prefix"):
            renderer.read_trial(tmp_path, "baseline", "reference", expected)
