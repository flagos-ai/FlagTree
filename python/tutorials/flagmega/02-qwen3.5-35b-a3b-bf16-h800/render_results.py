"""Validate independent decode trials and derive two SVGs from raw GPU events."""

import argparse
from collections import defaultdict
import hashlib
from html import escape
import json
import math
from pathlib import Path
import statistics


LABELS = {"native": "Native vLLM", "baseline": "FlagMega default", "agent": "FlagMega + agent tiles"}
COLORS = {"native": "#536579", "baseline": "#c47725", "agent": "#16836c"}


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


def read_trial(directory, variant, reference_hash, expected):
    accuracy = read_json(directory / "report.json")
    native = variant == "native"
    timing = accuracy if native else read_json(directory / "gpu-timing.json")
    schema = "flagmega.native-prepared-decode/v1" if native else "flagmega.prepared-decode-gpu-timing/v1"
    require(timing.get("schema") == schema, "Unexpected timing schema")
    require(timing.get("performance_valid") is True and accuracy.get("complete") is True,
            "Incomplete or invalid performance run")
    require(accuracy.get("reference_sha256") == reference_hash, "Different reference inputs")
    if not native:
        require(accuracy.get("all_sequences_match") is True, "Independent accuracy failed")
        require(all(accuracy.get(key) == timing.get(key) for key in ("semantic_hash", "source_sha256")),
                "Timing and accuracy name different artifacts")
        require(accuracy.get("schema") == "flagmega.prepared-decode-candidate/v1", "Unexpected accuracy schema")
        records = keyed(accuracy["records"])
        require(records.keys() == expected.keys(), "Incomplete accuracy scenarios")
        for context, row in records.items():
            require(row.get("initial_state_roundtrip_exact") is True and row.get("exact_match") is True,
                    "Initial state or independent sequence mismatch")
            require(row["initial_state_sha256"] == expected[context]["state_sha256"], "Different initial state")
            require(row["token_ids"] == expected[context]["token_ids"], "Independent sequence differs")
    else:
        require(timing.get("serving_latency") is False and timing.get("logprobs_requested") is False,
                "Native timing boundary differs")
    scenarios = keyed(timing["scenarios"])
    require(scenarios.keys() == expected.keys(), "Incomplete timing scenarios")
    samples = {}
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
        samples[context] = []
        for run in scenario["rounds"]:
            require(run["token_ids"] == expected[context]["token_ids"], "Timed independent sequence differs")
            values = run["samples_ms"]
            require(len(values) == len(run["token_ids"]) and bool(values), "Incomplete per-token samples")
            require(all(not isinstance(v, bool) and isinstance(v, (float, int)) and math.isfinite(v) and v > 0
                        for v in values), "Non-positive or non-finite GPU event time")
            samples[context].extend(values)
    device = (timing["device"], timing.get("visible_devices"))
    identity = (json.dumps(timing["source_sha256"], sort_keys=True), timing.get("semantic_hash"),
                timing.get("benchmark_sha256"), timing["torch"], timing.get("vllm"),
                timing.get("engine_config"))
    return samples, device, identity


def aggregate(reference, trials):
    reference_hash = sha256(reference)
    source = read_json(reference)
    require(source.get("schema") == "flagmega.prepared-decode-reference/v1", "Unexpected reference schema")
    expected = keyed(source["records"])
    require(bool(expected) and set(trials) == set(LABELS), "All variants and reference scenarios are required")
    require(len({len(paths) for paths in trials.values()}) == 1 and all(trials.values()),
            "Use equally many trials per variant")
    groups = defaultdict(list)
    provenance, identities, seen = [], {}, set()
    device = None
    for variant, paths in trials.items():
        for path in paths:
            path = path.resolve()
            require(path not in seen, "A trial cannot be counted twice")
            seen.add(path)
            samples, current_device, identity = read_trial(path, variant, reference_hash, expected)
            require(device is None or device == current_device, "Use the same GPU for all variants")
            device = current_device
            require(variant not in identities or identities[variant] == identity,
                    "Different kernel, harness or environment for one variant")
            identities[variant] = identity
            for context, values in samples.items():
                groups[context, variant].extend(values)
            files = [path / "report.json"] + ([] if variant == "native" else [path / "gpu-timing.json"])
            provenance.extend({"variant": variant, "path": str(file), "sha256": sha256(file)} for file in files)
    rows = []
    for context in sorted(expected):
        require(len({len(groups[context, variant]) for variant in LABELS}) == 1, "Unequal sample counts")
        row = {"prefix_length": context, "decode_tokens": len(expected[context]["token_ids"]), "variants": {}}
        for variant in LABELS:
            values = sorted(groups[context, variant])
            row["variants"][variant] = {
                "median_ms": statistics.median(values), "p95_ms": values[int(.95 * len(values))],
                "tokens_per_second": 1000 / statistics.mean(values), "samples": len(values),
            }
        rows.append(row)
    return {"schema": "flagmega.decode-comparison/v1", "reference_sha256": reference_hash,
            "serving_latency": False, "device": device, "sources": provenance, "identities": identities,
            "scenarios": rows}


def chart(rows, metric):
    latency = metric == "median_ms"
    title = "Decode latency (lower is better)" if latency else "Decode throughput (higher is better)"
    unit = "ms/token" if latency else "tokens/s"
    width, height, left, top, plot_width, plot_height = 1120, 650, 85, 135, 1000, 355
    maximum = max(row["variants"][variant][metric] for row in rows for variant in LABELS) * 1.2
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
             f'<text x="{left}" y="108" font-size="14">Native: 3 graphs. FlagMega: 1 graph. No prefix work or CPU scheduling.</text>']
    for tick in range(6):
        y = top + plot_height * (1 - tick / 5)
        parts.extend([f'<path d="M {left} {y:.1f} H {left + plot_width}" stroke="#e2e8ef"/>',
                      f'<text x="{left - 12}" y="{y + 5:.1f}" text-anchor="end" font-size="13">{maximum * tick / 5:.1f}</text>'])
    group_width = plot_width / len(rows)
    bar_width = group_width * .22
    for index, row in enumerate(rows):
        for offset, variant in enumerate(LABELS):
            value = row["variants"][variant][metric]
            bar_height = plot_height * value / maximum
            x = left + group_width * (index + .12) + offset * bar_width * 1.15
            y = top + plot_height - bar_height
            parts.extend([f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_width:.1f}" height="{bar_height:.1f}" fill="{COLORS[variant]}"/>',
                          f'<text x="{x + bar_width / 2:.1f}" y="{y - 8:.1f}" text-anchor="middle" font-size="13">{value:.3f}</text>'])
        parts.append(f'<text x="{left + group_width * (index + .5):.1f}" y="{top + plot_height + 29}" text-anchor="middle" font-size="14">Context {row["prefix_length"]}</text>')
    parts.append(f'<text x="15" y="{top - 12}" font-size="13">{unit}</text>')
    for index, (variant, label) in enumerate(LABELS.items()):
        x = left + index * 310
        parts.extend([f'<rect x="{x}" y="555" width="15" height="15" fill="{COLORS[variant]}"/>',
                      f'<text x="{x + 23}" y="568" font-size="14">{escape(label)}</text>'])
    statistic = "Median per-token latency." if latency else "1000 / mean per-token latency."
    parts.append(f'<text x="{left}" y="610" font-size="13">{statistic} Complete independent greedy sequences match in every measured round.</text>')
    return "\n".join(parts + ["</g>", "</svg>", ""])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    for variant in LABELS:
        parser.add_argument(f"--{variant}", type=Path, nargs="+", required=True, help="Immutable trial directories")
    parser.add_argument("--output", type=Path, required=True, help="New output directory, normally under .local")
    args = parser.parse_args()
    report = aggregate(args.reference, {variant: getattr(args, variant) for variant in LABELS})
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    for name, metric in (("decode_latency", "median_ms"), ("decode_throughput", "tokens_per_second")):
        (args.output / f"{name}.svg").write_text(chart(report["scenarios"], metric))
    print("| Context | Native ms | Default ms | Agent ms | Agent vs default | Agent vs native |")
    print("| --- | ---: | ---: | ---: | ---: | ---: |")
    for row in report["scenarios"]:
        native, baseline, agent = (row["variants"][variant]["median_ms"] for variant in LABELS)
        print(f"| {row['prefix_length']} | {native:.3f} | {baseline:.3f} | {agent:.3f} | {baseline / agent:.2f}x | {native / agent:.2f}x |")


if __name__ == "__main__":
    main()
