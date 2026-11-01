"""Validate full requests and render 01-aligned performance figures.

The prepared-state reader remains available for auditing historical results.
"""

import argparse
from collections import defaultdict
import hashlib
from html import escape
import json
import math
from pathlib import Path
import shutil
import statistics


LABELS = {"native": "vLLM", "baseline": "FlagMega Initial", "agent": "FlagMega Agent"}
COLORS = {"native": "#556579", "baseline": "#c47725", "agent": "#198269"}
REQUEST_LABELS = {**LABELS, "chat_cli": "FlagMega chat CLI"}
REQUEST_COLORS = {**COLORS, "chat_cli": "#8252b3"}
REQUEST_CHARTS = (
    ("decode_latency.svg", "decode_median_ms", "Decode step latency (lower is better)", "ms/token"),
    ("decode_throughput.svg", "decode_tokens_per_second", "Decode throughput (higher is better)", "tokens/s"),
    ("request_throughput.svg", "request_tokens_per_second", "Request throughput incl. prefill (higher is better)", "tokens/s"),
)


def aggregate_requests(paths):
    groups, reports, identities, seen = defaultdict(list), [], {}, set()
    contract = None
    for path in sorted(paths):
        report = read_json(path)
        require(report.get("schema") == "flagmega.request-comparison-trial/v1",
                "Expected full requests, not prepared-state GPU events")
        label, round_id = report["label"], report["round"]
        require(label in REQUEST_LABELS and type(round_id) is int and round_id >= 0, "Invalid label or round")
        require((label, round_id) not in seen, "Duplicate variant round")
        seen.add((label, round_id))
        require(report.get("performance_valid") is True and report.get("prefix_caching") is False
                and report.get("warmups_per_scenario", 0) >= 2, "Invalid or unwarmed performance run")
        require(report["batch"] == report["concurrency"] == 1, "Expected batch/concurrency one")
        require(report.get("vllm_imported") is (label != "chat_cli"), "Runtime identity differs")
        current = tuple(report[k] for k in ("checkpoint_revision", "checkpoint_config_sha256", "device", "visible_devices"))
        require(contract is None or current == contract, "Different checkpoint or GPU")
        contract = current
        backend = report.get("backend", {})
        identity = (backend.get("artifact_source_sha256"),
                    backend.get("artifact_semantic_hashes", backend.get("artifact_semantic_hash")),
                    report["torch"], report["triton"], report.get("vllm"), backend.get("backend_source_sha256"))
        require(label not in identities or identity == identities[label], "Different artifact or environment for one variant")
        identities[label] = identity
        if label != "native":
            require(report.get("eager_reference_kind") == "ir_torch",
                    "Full requests require an independent IR/PyTorch eager reference")
            require(report.get("prefill_mode") == "compiled_batched",
                    "Full requests require compiled batched prefill")
        require([s["prompt_length"] for s in report["scenarios"]] == [32, 128, 1024, 2048],
                "Scenario lengths must match tutorial 01")
        for scenario in report["scenarios"]:
            require(scenario["decode_tokens"] == 64 and len(scenario["runs"]) == 5, "Expected 64 tokens and five requests")
            for run in scenario["runs"]:
                if label != "native":
                    require(scenario.get("independent_eager_token_ids") == run["token_ids"],
                            "Measured sequence differs from independent eager execution")
            groups[scenario["prompt_length"], label].extend(scenario["runs"])
        reports.append({"file": path.name, "sha256": sha256(path), "label": label, "round": round_id})
    require(seen == {(label, round_id) for label in REQUEST_LABELS for round_id in range(3)},
            "Three complete rounds of every variant are required")
    require(identities["agent"][:2] == identities["chat_cli"][:2], "Chat CLI must use the same Agent artifact")
    rows = []
    for length in (32, 128, 1024, 2048):
        native = groups[length, "native"][0]
        row = {"prompt_length": length, "decode_tokens": 64, "variants": {}}
        for label in REQUEST_LABELS:
            runs = groups[length, label]
            for run in runs:
                require(len(run["prompt_tokens"]) == length and run["prompt_tokens"] == native["prompt_tokens"], "Different prompts")
                require(len(run["token_ids"]) == 64, "Expected a complete 64-token output sequence")
                require(run["token_ids"] == runs[0]["token_ids"], "Repeated independent request differs")
                require(run.get("cached_tokens", 0) == 0, "Reused request prefix")
                require(len(run["decode_step_ms"]) == 63, "Decode samples must exclude exactly the first output token")
                duration = run["total_ms"] if label == "chat_cli" else run["e2e_ms"]
                require(all(type(v) in (int, float) and math.isfinite(v) and v > 0
                            for v in [*run["decode_step_ms"], run["ttft_ms"], duration]), "Invalid request timing")
            samples = sorted(v for run in runs for v in run["decode_step_ms"])
            row["variants"][label] = {
                "decode_median_ms": statistics.median(samples), "decode_p95_ms": samples[int(.95 * len(samples))],
                "decode_tokens_per_second": statistics.median(63000 / sum(run["decode_step_ms"]) for run in runs),
                "request_tokens_per_second": statistics.median(64000 / (run["total_ms"] if label == "chat_cli" else run["e2e_ms"]) for run in runs),
                "ttft_median_ms": statistics.median(run["ttft_ms"] for run in runs),
                "runs": len(runs), "decode_samples": len(samples),
                "native_token_agreement": sum(a == b for run in runs for a, b in zip(run["token_ids"], native["token_ids"])) / (64 * len(runs)),
                "native_sequence_match": runs[0]["token_ids"] == native["token_ids"],
                "first_native_difference": next((i for i, (a, b) in
                    enumerate(zip(runs[0]["token_ids"], native["token_ids"], strict=True)) if a != b), None),
                "independent_eager_agreement": True,
            }
        rows.append(row)
    return {"schema": "flagmega.request-comparison/v1", "labels": REQUEST_LABELS,
            "batch": 1, "concurrency": 1, "prompt_lengths": [32, 128, 1024, 2048], "output_tokens": 64,
            "rounds": 3, "requests_per_scenario_per_round": 5, "prefix_caching": False,
            "acceptance": {"native_sequence_match_required": False,
                           "native_comparison_tokens": 64, "complete_independent_eager_sequence": True},
            "checkpoint_revision": contract[0], "checkpoint_config_sha256": contract[1],
            "device": contract[2], "visible_devices": contract[3], "sources": reports,
            "identities": {label: dict(zip(("source_sha256", "semantic_hash", "torch", "triton", "vllm", "backend_source_sha256"), identity))
                           for label, identity in identities.items()}, "scenarios": rows}


def aggregate_native_prefill_decode(paths):
    groups, identities, provenance, backends = defaultdict(list), {}, [], {}
    rounds = defaultdict(set)
    common_identity = None
    for path in paths:
        report = read_json(path)
        require(report.get("schema") == "flagmega.native-prefill-decode-trial/v1", "Different decode timing scope")
        label = report["label"]
        require(label in REQUEST_LABELS and report.get("performance_valid") is True, "Unaccepted performance trial")
        require(report.get("prefill") == "native_vllm" and report.get("prefill_timed") is False, "Prefill must be excluded")
        require(report.get("batch") == report.get("concurrency") == 1 and report.get("prefix_caching") is False,
                "Expected batch/concurrency one without prefix caching")
        require(report.get("decode_intervals") == 63 and report.get("warmups_per_scenario") == 2, "Different sampling plan")
        require(report.get("cuda_graph") is True, "Uncaptured performance trial")
        require(report.get("vllm_imported") == (label != "chat_cli"), "Standalone CLI must not import vLLM")
        conditioning = report.get("thermal_monitor", {})
        require(conditioning.get("kind") == "nvml-clock-event-counters/v1"
                and conditioning.get("gpu_uuid") and conditioning.get("max_start_temperature_c") == 65
                and conditioning.get("minimum_idle_seconds", 0) >= .5, "Missing thermal monitoring")
        identity = tuple(report.get(k) for k in ("checkpoint_config_sha256", "checkpoint_revision",
            "native_reference_sha256", "device", "visible_devices", "torch")) + (json.dumps(conditioning, sort_keys=True),)
        require(common_identity is None or common_identity == identity, "Different checkpoint, prefix, GPU or Torch")
        common_identity = identity
        backend = report.get("backend")
        if label != "native":
            require(backend and backend.get("num_layers") == 40 and backend.get("prefill") == "native_vllm_snapshot",
                    "Missing complete native-prefilled decoder")
            require(backend["resources"]["spill_bytes"] == 0, "Spilling artifact")
        per_label = (report["benchmark_sha256"], report["triton"], report["vllm"],
            backend.get("artifact_source_sha256") if backend else None,
            backend.get("artifact_semantic_hash") if backend else None,
            backend.get("backend_source_sha256") if backend else None,
            backend.get("native_prefill_backend_sha256") if backend else None,
            backend.get("compiler_lib_sha256") if backend else None)
        require(label not in identities or identities[label] == per_label, "Variant changed across rounds")
        identities[label] = per_label
        backends[label] = backend
        require(report["round"] in (0, 1, 2) and report["round"] not in rounds[label], "Duplicate or invalid round")
        rounds[label].add(report["round"])
        require({s["prompt_length"] for s in report["scenarios"]} == {32, 128, 1024, 2048}, "Different prompt lengths")
        for scenario in report["scenarios"]:
            n = scenario["prompt_length"]
            require(scenario["decode_tokens"] == 64 and len(scenario["runs"]) == 5, "Expected five 64-token outputs")
            require(scenario.get("initial_state_roundtrip_exact") is True, "Prefill state conversion changed values")
            native = scenario["native_token_ids"]
            require(len(native) == 64, "Incomplete native comparison")
            for run in scenario["runs"]:
                before, after = run.get("thermal_before", {}), run.get("thermal_after", {})
                require(before.get("thermal_active") is False and after.get("thermal_active") is False
                        and before.get("temperature_c", 100) <= 65
                        and type(before.get("thermal_slowdown_us")) is int
                        and before["thermal_slowdown_us"] == after.get("thermal_slowdown_us"),
                        "Thermal throttling invalidated the request")
                require(len(run["prompt_tokens"]) == n and len(run["token_ids"]) == 64, "Wrong input/output counts")
                require(run["token_ids"][:4] == native[:4], "Independent three-token decode check failed")
                require(len(run["decode_step_ms"]) == 63 and all(math.isfinite(x) and x > 0 for x in run["decode_step_ms"]),
                        "Invalid host decode intervals")
                groups[n, label].append((run, native))
        provenance.append({"file": path.name, "sha256": sha256(path)})
    require(set(rounds) == set(REQUEST_LABELS) and all(v == {0, 1, 2} for v in rounds.values()), "Missing variant or round")
    require(identities["agent"][3:] == identities["chat_cli"][3:], "CLI must execute the same Agent artifact")
    rows = []
    for n in (32, 128, 1024, 2048):
        variants = {}
        for label in REQUEST_LABELS:
            samples = groups[n, label]
            first, native = samples[0]
            require(all(r["token_ids"] == first["token_ids"] and r["prompt_tokens"] == first["prompt_tokens"]
                        and ref == native for r, ref in samples), "Request sequence changed across rounds")
            intervals = [x for r, _ in samples for x in r["decode_step_ms"]]
            variants[label] = {"decode_median_ms": statistics.median(intervals),
                "decode_tokens_per_second": statistics.median(63000 / sum(r["decode_step_ms"]) for r, _ in samples),
                "requests": len(samples), "timed_intervals": len(intervals),
                "native_token_agreement": sum(a == b for a, b in zip(first["token_ids"], native)),
                "first_native_difference": next((i for i, (a, b) in enumerate(zip(first["token_ids"], native)) if a != b), None)}
        rows.append({"prompt_length": n, "decode_tokens": 64, "variants": variants})
    return {"schema": "flagmega.native-prefill-decode-comparison/v1", "labels": REQUEST_LABELS,
        "prefill": "native_vllm", "prefill_timed": False, "rounds": 3, "requests_per_scenario_per_round": 5,
        "acceptance": {"independent_native_decode_prefix": 3, "complete_native_sequence_required": False,
                       "initial_state_roundtrip_exact": True},
        "checkpoint_config_sha256": common_identity[0], "checkpoint_revision": common_identity[1],
        "native_reference_sha256": common_identity[2], "device": common_identity[3],
        "visible_devices": common_identity[4], "torch": common_identity[5],
        "thermal_monitor": json.loads(common_identity[6]),
        "identities": {label: dict(zip(("benchmark_sha256", "triton", "vllm", "source_sha256",
            "semantic_hash", "backend_source_sha256", "native_prefill_backend_sha256", "compiler_lib_sha256"), value))
            for label, value in identities.items()}, "backends": backends,
        "sources": provenance, "scenarios": rows}


def request_chart(rows, metric, title, unit, *, native_prefill=False):
    width, height, left, top, plot_width, plot_height = 1200, 690, 85, 145, 1085, 395
    maximum = max(row["variants"][label][metric] for row in rows for label in REQUEST_LABELS) * 1.18
    boundary = ("Decode intervals exclude prefill and the first output token."
                if metric == "decode_median_ms" else
                "Median per-request decode tokens / decode time; excludes prefill and the first output token."
                if metric == "decode_tokens_per_second" else
                "Output tokens / complete request time, including prefill; not inverse decode latency.")
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img">',
             f'<title>{escape(title)}</title>', '<desc>Qwen3.5-35B-A3B BF16, full 40-layer model, one H800.</desc>',
             '<rect width="100%" height="100%" fill="white"/>', '<g font-family="sans-serif" fill="#243343">',
             f'<text x="{left}" y="37" font-size="24">{escape(title)}</text>',
             f'<text x="{left}" y="64" font-size="14">Qwen3.5-35B-A3B BF16 / H800 / batch 1 / 64 output tokens / host-visible timing</text>',
             f'<text x="{left}" y="89" font-size="14">{escape(boundary)}</text>',
             f'<text x="{left}" y="113" font-size="14">' +
             ('Native vLLM prefill; state preparation excluded. CLI terminal I/O excluded.' if native_prefill else
              'Compiled batched prefill in every series. CLI terminal I/O excluded.') + '</text>']
    for tick in range(6):
        y = top + plot_height * (1 - tick / 5)
        parts += [f'<path d="M {left} {y:.1f} H {left + plot_width}" stroke="#dfe5eb"/>',
                  f'<text x="{left - 12}" y="{y + 5:.1f}" text-anchor="end" font-size="13">{maximum * tick / 5:.1f}</text>']
    group_width = plot_width / len(rows)
    bar_width = group_width * .82 / (len(REQUEST_LABELS) + (len(REQUEST_LABELS) - 1) * .15)
    for index, row in enumerate(rows):
        for offset, label in enumerate(REQUEST_LABELS):
            value = row["variants"][label][metric]
            h = plot_height * value / maximum
            x, y = left + group_width * (index + .09) + offset * bar_width * 1.15, top + plot_height - h
            number = f"{value:.3f}" if metric == "decode_median_ms" else f"{value:.2f}"
            parts += [f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_width:.1f}" height="{h:.1f}" fill="{REQUEST_COLORS[label]}"/>',
                      f'<text x="{x + bar_width / 2:.1f}" y="{y - 8:.1f}" text-anchor="middle" font-size="12">{number}</text>']
        parts.append(f'<text x="{left + group_width * (index + .5):.1f}" y="{top + plot_height + 28}" text-anchor="middle" font-size="14">Prompt {row["prompt_length"]}</text>')
    parts.append(f'<text x="16" y="{top - 16}" font-size="13">{unit}</text>')
    for index, (label, name) in enumerate(REQUEST_LABELS.items()):
        x, y = left + (index % 2) * 460, height - 90 + (index // 2) * 25
        parts += [f'<rect x="{x}" y="{y - 12}" width="14" height="14" fill="{REQUEST_COLORS[label]}"/>',
                  f'<text x="{x + 22}" y="{y}" font-size="13">{escape(name)}</text>']
    footer = ('3 rounds × 5 continuations. Native state exact; first 3 decode tokens checked; later differences reported.'
              if native_prefill else
              '3 rounds × 5 requests. Artifact outputs checked against independent eager; native token differences are reported.')
    parts.append(f'<text x="{left}" y="660" font-size="13">{footer}</text>')
    return "\n".join(parts + ['</g>', '</svg>', ''])


def read_json(path):
    return json.loads(path.read_text())


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def keyed(rows):
    require(all(type(row.get("prefix_length")) is int and row["prefix_length"] > 0 for row in rows),
            "Context lengths must be positive integers")
    result = {row["prefix_length"]: row for row in rows}
    require(len(result) == len(rows), "Duplicate context lengths")
    return result


def read_trial(directory, variant, reference_hash, expected, *, include_contract=False):
    accuracy = read_json(directory / "report.json")
    native = variant == "native"
    timing = accuracy if native else read_json(directory / "gpu-timing.json")
    schemas = ("flagmega.native-prepared-decode/v1",) if native else (
        "flagmega.prepared-decode-gpu-timing/v1", "flagmega.prepared-decode-gpu-timing/v2")
    require(timing.get("schema") in schemas, "Unexpected timing schema")
    extended = timing.get("schema") == "flagmega.prepared-decode-gpu-timing/v2"
    require(timing.get("performance_valid") is True and accuracy.get("complete") is True,
            "Incomplete or invalid performance run")
    if extended:
        require(timing.get("complete") is True, "Incomplete timing report")
    require(accuracy.get("reference_sha256") == reference_hash, "Different reference inputs")
    validated = None
    if not native:
        accuracy_schema = accuracy.get("schema")
        if accuracy_schema == "flagmega.prepared-decode-candidate/v2":
            acceptance = accuracy.get("acceptance", {})
            require(accuracy.get("all_prefixes_match") is True, "Independent accuracy failed")
            validated = acceptance.get("tokens_per_scenario")
            require(acceptance.get("kind") == "independent-greedy-prefix" and type(validated) is int
                    and all(0 < validated <= len(row["token_ids"]) for row in expected.values()),
                    "Invalid independent prefix acceptance")
            if not extended:
                require(all(validated == len(row["token_ids"]) for row in expected.values()),
                        "Short-prefix acceptance cannot establish a full-reference performance comparison")
            require(timing.get("acceptance") == acceptance, "Timing and accuracy use different acceptance criteria")
        else:
            require(accuracy_schema == "flagmega.prepared-decode-candidate/v1", "Unexpected accuracy schema")
            require(not extended, "Extended timing requires explicit prefix acceptance")
            require(accuracy.get("all_sequences_match") is True, "Independent accuracy failed")
        require(all(accuracy.get(key) == timing.get(key) for key in ("semantic_hash", "source_sha256")),
                "Timing and accuracy name different artifacts")
        if extended:
            require(accuracy.get("toolchain") == timing.get("toolchain") and bool(timing.get("toolchain")),
                    "Timing and accuracy use different or unrecorded toolchains")
        records = keyed(accuracy["records"])
        require(records.keys() == expected.keys(), "Incomplete accuracy scenarios")
        for context, row in records.items():
            require(row.get("initial_state_roundtrip_exact") is True and row.get("exact_match") is True,
                    "Initial state or independent sequence mismatch")
            require(row["initial_state_sha256"] == expected[context]["state_sha256"], "Different initial state")
            require(row["token_ids"] == expected[context]["token_ids"][:validated], "Independent sequence differs")
    else:
        require(timing.get("serving_latency") is False and timing.get("logprobs_requested") is False,
                "Native timing boundary differs")
    scenarios = keyed(timing["scenarios"])
    require(scenarios.keys() == expected.keys(), "Incomplete timing scenarios")
    samples, steps = {}, {}
    for context, scenario in scenarios.items():
        require(scenario.get("serving_latency") is False and scenario.get("all_sequences_match") is True,
                "Invalid timing boundary or sequence")
        require(scenario.get("cache_policy") == "natural sequential decode; no synthetic flush",
                "Different cache policy")
        require(scenario.get("warmup_rounds", 0) >= 2, "Missing warmup rounds")
        if native:
            require(scenario.get("native_full_graph_reused") is True and scenario.get("initial_state_exact") is True,
                    "Native graph or state contract changed")
            require(scenario["initial_state_sha256"] == expected[context]["state_sha256"], "Different native state")
        require(len(scenario["rounds"]) >= 5, "At least five formal rounds required")
        sequence = expected[context]["token_ids"]
        if extended:
            prefix = expected[context]["token_ids"][:validated]
            sequence = scenario["rounds"][0]["token_ids"]
            require(scenario.get("validated_prefix") == prefix and sequence[:validated] == prefix,
                    "Timed prefix differs from independent acceptance")
            require(type(scenario.get("timed_steps")) is int and scenario["timed_steps"] == len(sequence)
                    and len(sequence) > validated, "Inconsistent extended decode length")
            require(scenario.get("timed_sequence_reference") == "candidate independent eager decode",
                    "Extended timing requires an independent candidate trajectory")
            if "independent_eager_token_ids" in scenario:
                require(scenario["independent_eager_token_ids"] == sequence, "Timed and eager trajectories differ")
        steps[context] = len(sequence)
        samples[context] = []
        for run in scenario["rounds"]:
            require(run["token_ids"] == sequence, "Timed independent sequence differs")
            values = run["samples_ms"]
            require(len(values) == len(run["token_ids"]) and bool(values), "Incomplete per-token samples")
            require(all(not isinstance(v, bool) and isinstance(v, (float, int)) and math.isfinite(v) and v > 0
                        for v in values), "Non-positive or non-finite GPU event time")
            samples[context].extend(values)
    device = (timing["device"], timing.get("visible_devices"))
    identity = (json.dumps(timing["source_sha256"], sort_keys=True), timing.get("semantic_hash"),
                timing.get("benchmark_sha256"), timing["torch"], timing.get("vllm"),
                timing.get("engine_config"), timing.get("toolchain"), accuracy.get("runner_sha256"))
    result = (samples, device, identity)
    return (*result, steps, validated) if include_contract else result


def aggregate(reference, trials):
    reference_hash = sha256(reference)
    source = read_json(reference)
    require(source.get("schema") == "flagmega.prepared-decode-reference/v1", "Unexpected reference schema")
    expected = keyed(source["records"])
    require(bool(expected) and set(trials) == set(LABELS),
            "vLLM, FlagMega Initial, FlagMega Agent and reference scenarios are required")
    require(len({len(paths) for paths in trials.values()}) == 1 and all(trials.values()),
            "Use equally many trials per variant")
    groups = defaultdict(list)
    provenance, identities, seen = [], {}, set()
    device, timed_steps = None, None
    validated_counts, assemblers = set(), set()
    for variant in LABELS:
        paths = trials[variant]
        for path in paths:
            path = path.resolve()
            require(path not in seen, "A trial cannot be counted twice")
            seen.add(path)
            samples, current_device, identity, steps, validated = read_trial(
                path, variant, reference_hash, expected, include_contract=True)
            require(timed_steps is None or timed_steps == steps, "Variants time different decode lengths")
            timed_steps = steps
            if variant != "native":
                validated_counts.add(validated)
                assemblers.add(json.dumps(identity[-2], sort_keys=True))
            require(device is None or device == current_device, "Use the same GPU for all variants")
            device = current_device
            require(variant not in identities or identities[variant] == identity,
                    "Different kernel, harness or environment for one variant")
            identities[variant] = identity
            for context, values in samples.items():
                groups[context, variant].extend(values)
            files = [path / "report.json"] + ([] if variant == "native" else [path / "gpu-timing.json"])
            provenance.extend({"variant": variant, "file": file.name, "sha256": sha256(file)} for file in files)
    rows = []
    for context in sorted(expected):
        require(len({len(groups[context, variant]) for variant in trials}) == 1, "Unequal sample counts")
        row = {"prefix_length": context, "decode_tokens": timed_steps[context], "variants": {}}
        for variant in trials:
            values = sorted(groups[context, variant])
            row["variants"][variant] = {
                "median_ms": statistics.median(values), "p95_ms": values[int(.95 * len(values))],
                "tokens_per_second": 1000 / statistics.mean(values), "samples": len(values),
            }
        rows.append(row)
    require(len(validated_counts) == 1, "Variants use different numerical acceptance lengths")
    require(len(assemblers) == 1, "FlagMega variants use different toolchains")
    # Validate full identities above, but omit machine paths from published metadata.
    public_identities = {}
    for variant, identity in identities.items():
        source_hash, semantic_hash, benchmark_hash, torch_version, vllm_version, engine, toolchain, runner = identity
        public_identities[variant] = {
            "source_sha256": json.loads(source_hash), "semantic_hash": semantic_hash,
            "benchmark_sha256": benchmark_hash, "runner_sha256": runner,
            "torch": torch_version, "vllm": vllm_version,
            "engine_config_sha256": hashlib.sha256(json.dumps(engine, sort_keys=True).encode()).hexdigest()
                if engine is not None else None,
            "toolchain": {key: value for key, value in toolchain.items() if key != "ptxas_path"}
                if toolchain else toolchain,
        }
    validated = next(iter(validated_counts))
    return {"schema": "flagmega.decode-comparison/v2", "reference_sha256": reference_hash,
            "validated_tokens": validated, "labels": LABELS,
            "serving_latency": False, "device": device, "sources": provenance, "identities": public_identities,
            "scenarios": rows}


def chart(rows, metric, *, labels=None, validated_tokens=None):
    labels = labels or {key: LABELS[key] for key in rows[0]["variants"]}
    latency = metric == "median_ms"
    title = "Decode latency (lower is better)" if latency else "Decode throughput (higher is better)"
    unit = "ms/token" if latency else "tokens/s"
    width, height, left, top, plot_width, plot_height = 1120, 650, 85, 135, 1000, 355
    maximum = max(row["variants"][variant][metric] for row in rows for variant in labels) * 1.2
    tokens = "/".join(str(v) for v in sorted({row["decode_tokens"] for row in rows}))
    description = ("Full 40-layer BF16 decoder, H800, batch 1. Prepared GPU events include model, logits, "
                   "greedy sampling, metadata and token feedback. Not serving latency.")
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
             f'viewBox="0 0 {width} {height}" role="img">', f'<title>{escape(title)}</title>',
             f'<desc>{escape(description)}</desc>', '<rect width="100%" height="100%" fill="white"/>',
             '<g font-family="sans-serif" fill="#243343">',
             f'<text x="{left}" y="36" font-size="24">{escape(title)}</text>',
             f'<text x="{left}" y="63" font-size="14">Qwen3.5-35B-A3B BF16 / H800 / batch 1 / {tokens} decode tokens</text>',
             f'<text x="{left}" y="86" font-size="14">Prepared GPU events: full decoder + greedy sampler + metadata/token feedback</text>',
             f'<text x="{left}" y="108" font-size="14">' +
             ('Native: 3 graphs. FlagMega: 1 graph. ' if "native" in labels else 'FlagMega: one graph per token. ') +
             'No prefix work or CPU scheduling.</text>']
    for tick in range(6):
        y = top + plot_height * (1 - tick / 5)
        parts.extend([f'<path d="M {left} {y:.1f} H {left + plot_width}" stroke="#e2e8ef"/>',
                      f'<text x="{left - 12}" y="{y + 5:.1f}" text-anchor="end" font-size="13">{maximum * tick / 5:.1f}</text>'])
    group_width = plot_width / len(rows)
    bar_width = group_width * .66 / len(labels)
    for index, row in enumerate(rows):
        for offset, variant in enumerate(labels):
            value = row["variants"][variant][metric]
            bar_height = plot_height * value / maximum
            x = left + group_width * (index + .12) + offset * bar_width * 1.15
            y = top + plot_height - bar_height
            parts.extend([f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_width:.1f}" height="{bar_height:.1f}" fill="{COLORS[variant]}"/>',
                          f'<text x="{x + bar_width / 2:.1f}" y="{y - 8:.1f}" text-anchor="middle" font-size="13">{value:.3f}</text>'])
        parts.append(f'<text x="{left + group_width * (index + .5):.1f}" y="{top + plot_height + 29}" text-anchor="middle" font-size="14">Context {row["prefix_length"]}</text>')
    parts.append(f'<text x="15" y="{top - 12}" font-size="13">{unit}</text>')
    for index, (variant, label) in enumerate(labels.items()):
        x = left + index * 310
        parts.extend([f'<rect x="{x}" y="555" width="15" height="15" fill="{COLORS[variant]}"/>',
                      f'<text x="{x + 23}" y="568" font-size="14">{escape(label)}</text>'])
    statistic = "Median per-token latency." if latency else "1000 / mean per-token latency."
    acceptance = (f"First {validated_tokens} tokens match the reference; timed rounds reproduce each candidate's eager trajectory."
                  if validated_tokens is not None else "Complete independent greedy sequences match in every measured round.")
    parts.append(f'<text x="{left}" y="602" font-size="13">{escape(statistic)}</text>')
    parts.append(f'<text x="{left}" y="626" font-size="13">{escape(acceptance)}</text>')
    return "\n".join(parts + ["</g>", "</svg>", ""])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, help="Three full-request rounds of vLLM/Initial/Agent/chat CLI")
    parser.add_argument("--native-prefill-results", type=Path, help="Three decode rounds after native vLLM prefill")
    parser.add_argument("--reference", type=Path, help="Historical prepared-state reference")
    for variant in LABELS:
        parser.add_argument(f"--{variant}", type=Path, nargs="+",
                            help=f"Immutable {LABELS[variant]} trial directories")
    parser.add_argument("--output", type=Path, required=True, help="New output directory")
    args = parser.parse_args()
    if args.native_prefill_results:
        paths = sorted(p for p in args.native_prefill_results.iterdir() if p.name.startswith("round") and p.suffix == ".json")
        report = aggregate_native_prefill_decode(paths)
        args.output.mkdir(parents=True, exist_ok=False)
        raw_output = args.output / "raw"
        raw_output.mkdir()
        for path in paths:
            shutil.copy2(path, raw_output / path.name)
        for source in report["sources"]:
            source["file"] = "raw/" + source["file"]
        (args.output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
        for filename, metric, title, unit in REQUEST_CHARTS[:2]:
            (args.output / filename).write_text(request_chart(report["scenarios"], metric, title, unit, native_prefill=True))
        print(json.dumps(report["scenarios"]))
        return
    if args.results:
        report = aggregate_requests(sorted(args.results.glob("round*.json")))
        args.output.mkdir(parents=True, exist_ok=False)
        (args.output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
        for filename, metric, title, unit in REQUEST_CHARTS:
            (args.output / filename).write_text(request_chart(report["scenarios"], metric, title, unit))
        for row in report["scenarios"]:
            print(json.dumps(row))
        return
    if not args.reference or any(not getattr(args, variant) for variant in LABELS):
        parser.error("Use --results, or supply the historical reference and all three prepared-state variants")
    report = aggregate(args.reference, {variant: getattr(args, variant) for variant in LABELS if getattr(args, variant)})
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    for name, metric in (("decode_latency", "median_ms"), ("decode_throughput", "tokens_per_second")):
        (args.output / f"{name}.svg").write_text(chart(report["scenarios"], metric,
                                                   labels=report["labels"], validated_tokens=report["validated_tokens"]))
    print("| Context | " + " | ".join(report["labels"].values()) + " |")
    print("| --- | " + " | ".join("---:" for _ in report["labels"]) + " |")
    for row in report["scenarios"]:
        print(f"| {row['prefix_length']} | " + " | ".join(
            f"{row['variants'][variant]['median_ms']:.3f}" for variant in report["labels"]) + " |")


if __name__ == "__main__":
    main()
