"""Validate independent decode trials and derive two SVGs from raw GPU events."""

import argparse
from collections import defaultdict
import hashlib
from html import escape
import json
import math
from pathlib import Path
import statistics


LABELS = {"native": "vLLM", "baseline": "FlagMega Initial", "agent": "FlagMega Agent"}
COLORS = {"native": "#556579", "baseline": "#c47725", "agent": "#198269"}


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
    parser.add_argument("--reference", type=Path, required=True)
    for variant in LABELS:
        parser.add_argument(f"--{variant}", type=Path, nargs="+", required=True,
                            help=f"Immutable {LABELS[variant]} trial directories")
    parser.add_argument("--output", type=Path, required=True, help="New output directory")
    args = parser.parse_args()
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
