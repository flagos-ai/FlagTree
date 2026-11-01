import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]


def renderer():
    spec = importlib.util.spec_from_file_location("qwen35_request_renderer", ROOT / "render_results.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def trials(tmp_path):
    paths = []
    for round_id in range(3):
        for label in ("native", "baseline", "agent", "chat_cli"):
            scenarios = []
            for length in (32, 128, 1024, 2048):
                # Request rate must be aggregated per request, not inverted pooled latency.
                runs = [{"prompt_tokens": [1] * length, "token_ids": [2] * 64,
                         "decode_step_ms": [1.] * 62 + [20.], "ttft_ms": 10.,
                         "e2e_ms": 92., "total_ms": 92., "cached_tokens": 0} for _ in range(5)]
                scenarios.append({"prompt_length": length, "decode_tokens": 64, "runs": runs,
                                  "independent_eager_token_ids": [2] * 64})
            report = {"schema": "flagmega.request-comparison-trial/v1", "label": label, "round": round_id,
                      "runtime": "chat_cli" if label == "chat_cli" else "vllm", "batch": 1, "concurrency": 1,
                      "prefix_caching": False, "performance_valid": True,
                      "checkpoint_revision": "checkpoint", "checkpoint_config_sha256": "config",
                      "device": "H800", "visible_devices": "6", "torch": "test", "triton": "test",
                      "warmups_per_scenario": 2, "vllm_imported": label != "chat_cli",
                      "eager_reference_kind": "ir_torch" if label != "native" else None,
                      "prefill_mode": "compiled_batched" if label != "native" else "native_batched",
                      "backend": {"artifact_source_sha256": label if label != "chat_cli" else "agent",
                                  "artifact_semantic_hash": label if label != "chat_cli" else "agent"},
                      "scenarios": scenarios}
            path = tmp_path / f"round{round_id:02d}.{label}.json"
            path.write_text(json.dumps(report))
            paths.append(path)
    return paths


def test_request_contract_and_per_request_throughput(tmp_path):
    result = renderer().aggregate_requests(trials(tmp_path))
    assert [s["prompt_length"] for s in result["scenarios"]] == [32, 128, 1024, 2048]
    metrics = result["scenarios"][0]["variants"]["agent"]
    assert metrics["runs"] == 15 and metrics["decode_samples"] == 945
    assert metrics["decode_median_ms"] == 1.
    assert metrics["decode_tokens_per_second"] == pytest.approx(63000 / 82)
    assert metrics["request_tokens_per_second"] == pytest.approx(64000 / 92)


def test_native_sequence_differences_are_reported(tmp_path):
    paths = trials(tmp_path)
    # Numerical parity with the independent eager run remains required;
    # native output differences are visible without blocking performance plots.
    for path in paths:
        report = json.loads(path.read_text())
        if report["label"] == "native":
            continue
        scenario = report["scenarios"][1]
        scenario["independent_eager_token_ids"][14] = 220
        for run in scenario["runs"]:
            run["token_ids"][14] = 220
        path.write_text(json.dumps(report))
    result = renderer().aggregate_requests(paths)
    variant = result["scenarios"][1]["variants"]["agent"]
    assert variant["native_sequence_match"] is False
    assert variant["first_native_difference"] == 14
    assert variant["native_token_agreement"] == pytest.approx(63 / 64)
    assert result["acceptance"]["native_sequence_match_required"] is False


@pytest.mark.parametrize("change", ["prepared", "scenario", "tokens", "samples", "artifact", "round"])
def test_incompatible_or_incomplete_measurements_are_rejected(tmp_path, change):
    paths = trials(tmp_path)
    report = json.loads(paths[-1].read_text())
    if change == "prepared":
        report["schema"] = "flagmega.prepared-decode-gpu-timing/v2"
    elif change == "scenario":
        report["scenarios"][0]["prompt_length"] = 1
    elif change == "tokens":
        report["scenarios"][0]["runs"][0]["token_ids"][-1] = 9
    elif change == "samples":
        report["scenarios"][0]["runs"][0]["decode_step_ms"].pop()
    elif change == "artifact":
        report["backend"]["artifact_source_sha256"] = "different"
    else:
        report["round"] = 0
    paths[-1].write_text(json.dumps(report))
    with pytest.raises(ValueError):
        renderer().aggregate_requests(paths)


@pytest.mark.parametrize("kind", [None, "uncaptured_compiled"])
def test_compiled_replay_is_not_an_independent_eager_reference(tmp_path, kind):
    paths = trials(tmp_path)
    report = json.loads(paths[-1].read_text())
    report["eager_reference_kind"] = kind
    paths[-1].write_text(json.dumps(report))
    with pytest.raises(ValueError, match="IR/PyTorch"):
        renderer().aggregate_requests(paths)


def native_prefill_trials(tmp_path):
    paths = trials(tmp_path)
    for path in paths:
        value = json.loads(path.read_text())
        value.update(schema="flagmega.native-prefill-decode-trial/v1", prefill="native_vllm",
                     prefill_timed=False, decode_intervals=63, cuda_graph=True,
                     native_reference_sha256="native", benchmark_sha256="benchmark", vllm="test",
                     thermal_monitor={"kind": "nvml-clock-event-counters/v1", "gpu_uuid": "GPU-test",
                                      "max_start_temperature_c": 65, "minimum_idle_seconds": .5})
        if value["label"] != "native":
            value["backend"].update(num_layers=40, prefill="native_vllm_snapshot", resources={"spill_bytes": 0})
        else:
            value["backend"] = None
        for scenario in value["scenarios"]:
            scenario.update(native_token_ids=[2] * 64, initial_state_roundtrip_exact=True)
            for run in scenario["runs"]:
                run.update(thermal_before={"temperature_c": 60, "thermal_active": False, "thermal_slowdown_us": 0},
                           thermal_after={"temperature_c": 62, "thermal_active": False, "thermal_slowdown_us": 0})
            if value["label"] != "native":
                for run in scenario["runs"]:
                    run["token_ids"][14] = 9
        path.write_text(json.dumps(value))
    return paths


def test_native_prefill_decode_scope_and_later_token_differences(tmp_path):
    module = renderer()
    result = module.aggregate_native_prefill_decode(native_prefill_trials(tmp_path))
    metrics = result["scenarios"][0]["variants"]["agent"]
    assert metrics["requests"] == 15 and metrics["timed_intervals"] == 945
    assert metrics["decode_median_ms"] == 1.
    assert metrics["decode_tokens_per_second"] == pytest.approx(63000 / 82)
    assert metrics["native_token_agreement"] == 63 and metrics["first_native_difference"] == 14
    assert "request_tokens_per_second" not in metrics
    svg = module.request_chart(result["scenarios"], "decode_median_ms", "Decode", "ms/token", native_prefill=True)
    assert "Native vLLM prefill; state preparation excluded" in svg
    assert "Compiled batched prefill" not in svg


@pytest.mark.parametrize("change", ("prefill", "state", "prefix", "spills", "intervals", "thermal", "schema"))
def test_native_prefill_scope_rejects_incompatible_data(tmp_path, change):
    paths = native_prefill_trials(tmp_path)
    value = json.loads(paths[-1].read_text())
    if change == "prefill":
        value["prefill_timed"] = True
    elif change == "state":
        value["scenarios"][0]["initial_state_roundtrip_exact"] = False
    elif change == "prefix":
        value["scenarios"][0]["runs"][0]["token_ids"][1] = 9
    elif change == "spills":
        value["backend"]["resources"]["spill_bytes"] = 8
    elif change == "intervals":
        value["scenarios"][0]["runs"][0]["decode_step_ms"].pop()
    elif change == "thermal":
        value["scenarios"][0]["runs"][0]["thermal_after"]["thermal_slowdown_us"] = 1
    else:
        value["schema"] = "flagmega.request-comparison-trial/v1"
    paths[-1].write_text(json.dumps(value))
    with pytest.raises(ValueError):
        renderer().aggregate_native_prefill_decode(paths)
