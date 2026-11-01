"""Deterministic tutorial SVGs derived from both complete measured request orders."""

import argparse
from collections import defaultdict
import hashlib
from html import escape
import json
import math
from pathlib import Path
import statistics

from local_optimizations.scenarios import report_scenarios, reference_scenarios

LABELS = {"native_vllm": "Native vLLM", "scheduled": "Agent: layout + kernel choices"}
COLORS = {"native_vllm": "#556579", "scheduled": "#247da0"}
CHARTS = (
    ("decode_latency.svg", "decode_median_ms", "Decode step latency (lower is better)", "ms/token"),
    ("decode_throughput.svg", "decode_tokens_per_second", "Decode throughput (higher is better)", "tokens/s"),
    ("request_throughput.svg", "request_tokens_per_second", "Request throughput incl. prefill (higher is better)", "tokens/s"),
)
ORDERS = {"NFFN": ("N0", "F0", "F1", "N1"), "FNNF": ("F0", "N0", "N1", "F1")}
TOKENS = [3847, 284, 1373, 8282, 4218, 3755, 947, 906]


def aggregate(directory):
    groups = defaultdict(list)
    scenario_contracts = None
    sources, cards, scope, source_hash, native_versions = {}, None, None, None, None

    def read(path):
        data = path.read_bytes()
        sources[str(path.relative_to(directory))] = hashlib.sha256(data).hexdigest()
        return json.loads(data)

    for order, phases in ORDERS.items():
        root = directory / order
        if not (root / "DONE").is_file():
            raise ValueError(f"Incomplete measurement order: {order}")
        comparison = read(root / "comparison.json")
        if (tuple(comparison["phase_order"]) != phases
                or not all(comparison[key] for key in ("hardware_acceptance", "all_official8_passed",
                                                       "compute_body_five_op_gate_passed"))):
            raise ValueError(f"Unverified measurement order: {order}")
        identities = {card["uuid"] for card in comparison["cards"]}
        if len(identities) != 4 or (cards is not None and identities != cards):
            raise ValueError("Both orders must use the same four H800s")
        cards = identities
        if scope is not None and scope != comparison["scope"]:
            raise ValueError("Cannot combine different workload contracts")
        scope = comparison["scope"]
        rows = comparison["rows"]
        lengths = comparison.get("scenarios", [{"prompt_length": 1}])
        lengths = [r["prompt_length"] for r in lengths]
        if tuple(dict.fromkeys(row["label"] for row in rows)) != phases:
            raise ValueError("Every order requires all four measured legs")
        native_report = read(root / "N0" / "report.json")
        contracts = reference_scenarios(native_report, lengths)
        current_contracts = {length: (r["prompt"], r["output_tokens"], r["requests"][0]["tokens"]) for length, r in contracts.items()}
        if scenario_contracts is not None and current_contracts != scenario_contracts:
            raise ValueError("Independent native requests or prompts differ between orders")
        scenario_contracts = current_contracts
        if (len(rows) != len(phases)*len(lengths)
                or {(r["label"], r.get("prompt_length", 1)) for r in rows}
                != {(phase, length) for phase in phases for length in lengths}):
            raise ValueError("Every prompt length requires four complete engine legs")
        for row in rows:
            length = row.get("prompt_length", 1)
            native = row["label"].startswith("N")
            if (row["runtime"] != ("vllm" if native else "flagmega")
                    or not row["correctness"] or not row["cuda_graph"] or not row["guards"]["passed"]):
                raise ValueError("Leg correctness, timing or hardware checks failed")
            files = ["report.json"] if native else [f"rank-{rank}.json" for rank in range(4)]
            reports = [read(root / row["label"] / name) for name in files]
            scenario_reports = []
            for rank, report in enumerate(reports):
                scenario = {s["prompt_length"]: s for s in report_scenarios(report)}[length]
                scenario_reports.append(scenario)
                if (not report["passed"] or scenario["prompt"] != contracts[length]["prompt"]
                        or scenario["output_tokens"] != 8 or len(scenario["requests"]) != 12):
                    raise ValueError("Every leg requires the full two-warmup, ten-formal workload")
                if native:
                    if (report["versions"]["vllm"] != "0.31.0" or report["enforce_eager"]
                            or report["prefix_caching"]
                            or "NONE" in report["configuration"]["cudagraph_mode"].upper()):
                        raise ValueError("Native vLLM must retain the measured graph configuration")
                    versions = (report["checkpoint_revision"], report["versions"])
                    if native_versions is not None and versions != native_versions:
                        raise ValueError("Native checkpoint or runtime changed between legs")
                    native_versions = versions
                else:
                    if (report["rank"] != rank or not report["cuda_graph"]
                            or len(report["resources"]) != 228
                            or any(kernel["spill_bytes"] for kernel in report["resources"])):
                        raise ValueError("FlagMega requires four correct, zero-spill graph ranks")
                    if source_hash is not None and report["source_sha256"] != source_hash:
                        raise ValueError("Cannot combine different generated kernels")
                    source_hash = report["source_sha256"]
                for index, request in enumerate(scenario["requests"]):
                    if (request["request"] != index or request["tokens"] != contracts[length]["requests"][0]["tokens"]
                            or request["warmup"] != (index < 2)):
                        raise ValueError("Request histories or warmup labels differ")
            samples = []
            for index in range(2, 12):
                # Keep a full TP4 request as one sample, using the slowest rank.
                decode = max(report["requests"][index]["decode_tpot_ms"] for report in scenario_reports)
                wall = max(report["requests"][index]["wall_request_ms"] for report in scenario_reports)
                if any(not math.isfinite(value) or value <= 0 for value in (decode, wall)):
                    raise ValueError("Timing samples must be positive finite measurements")
                samples.append(decode)
                groups[(length, "native_vllm" if native else "scheduled")].append({
                    "order": order, "leg": row["label"], "request": index,
                    "decode_tpot_ms": decode, "wall_request_ms": wall,
                })
            if samples != row["samples_ms"] or statistics.median(samples) != row["median_decode_tpot_ms"]:
                raise ValueError("Raw requests differ from the accepted leg samples")
    scenarios = []
    for length in sorted(scenario_contracts):
        variants = {}
        for label in LABELS:
            requests = groups[(length, label)]
            if len(requests) != 40:
                raise ValueError("All 40 formal requests per scenario and runtime must be retained")
            variants[label] = {
                "decode_median_ms": statistics.median(r["decode_tpot_ms"] for r in requests),
                "decode_tokens_per_second": statistics.median(1000 / r["decode_tpot_ms"] for r in requests),
                "request_tokens_per_second": statistics.median(8000 / r["wall_request_ms"] for r in requests),
                "request_e2e_median_ms": statistics.median(r["wall_request_ms"] for r in requests),
                "runs": len(requests), "decode_samples": len(requests),
            }
        scenarios.append(dict(prompt_length=length, decode_tokens=8, variants=variants))
    return {"sources": sources, "source_sha256": source_hash, "scope": scope,
            "all_formal_requests": {str(length): {label: groups[(length, label)] for label in LABELS}
                                    for length in sorted(scenario_contracts)},
            "no_formal_samples_removed": True, "scenarios": scenarios}


def chart(rows, metric, title, unit):
    width, height = 1200, 690
    left, right, top, bottom = 85, 30, 145, 150
    plot_width, plot_height = width-left-right, height-top-bottom
    maximum = max(row["variants"][label][metric] for row in rows for label in LABELS) * 1.18
    output_tokens = "/".join(str(value) for value in sorted({row["decode_tokens"] for row in rows}))
    description = "GLM-5.3-Flash FP8, four H800s, TP4, batch one. Bars are medians of all 40 formal requests per runtime."
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img">',
             f'<title>{escape(title)}</title>',
             f'<desc>{escape(description)}</desc>',
             '<rect width="100%" height="100%" fill="#ffffff"/>',
             '<g font-family="sans-serif" fill="#243343">',
             f'<text x="{left}" y="37" font-size="24">{escape(title)}</text>',
             f'<text x="{left}" y="64" font-size="14">FP8 / 4 H800 / TP4 / batch 1 / {output_tokens} output tokens / host-visible timing</text>']
    boundary = ("Median per-request decode TPOT; excludes prefill and the first output token."
                if metric == "decode_median_ms" else
                "Median per-request decode tokens / decode time; excludes prefill and the first output token."
                if metric == "decode_tokens_per_second" else
                "Output tokens / complete request time, including prefill; not inverse decode latency.")
    parts.append(f'<text x="{left}" y="89" font-size="14">{escape(boundary)}</text>')
    parts.append(f'<text x="{left}" y="113" font-size="14">CUDA Graph / N/F/F/N + F/N/N/F / 40 requests per prompt/runtime / FlagMega: slowest rank</text>')
    for tick in range(6):
        value = maximum*tick/5
        y = top+plot_height*(1-tick/5)
        parts += [f'<path d="M {left} {y:.1f} H {width-right}" stroke="#dfe5eb"/>',
                  f'<text x="{left-12}" y="{y+5:.1f}" text-anchor="end" font-size="13">{value:.1f}</text>']
    group_width = plot_width/len(rows)
    bar_width = group_width*.82/(len(LABELS)+(len(LABELS)-1)*.15)
    for group, row in enumerate(rows):
        for variant, label in enumerate(LABELS):
            value = row["variants"][label][metric]
            h = plot_height*value/maximum
            x = left+group_width*(group+.09)+variant*bar_width*1.15
            y = top+plot_height-h
            number = f"{value:.3f}" if metric == "decode_median_ms" else f"{value:.2f}"
            parts += [f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_width:.1f}" height="{h:.1f}" fill="{COLORS[label]}"/>',
                      f'<text x="{x+bar_width/2:.1f}" y="{y-8:.1f}" text-anchor="middle" font-size="12">{number}</text>']
        parts.append(f'<text x="{left+group_width*(group+.5):.1f}" y="{top+plot_height+28}" text-anchor="middle" font-size="14">Prompt {row["prompt_length"]}</text>')
    parts.append(f'<text x="{left}" y="{height-36}" font-size="13">FlagMega: compiled token-scan prefill. vLLM: native batched prefill. Prefix caching disabled.</text>')
    parts.append(f'<text x="16" y="{top-16}" font-size="13">{escape(unit)}</text>')
    for i, (label, text_label) in enumerate(LABELS.items()):
        x, y = left + (i % 2)*460, height-90+(i//2)*25
        parts += [f'<rect x="{x}" y="{y-12}" width="14" height="14" fill="{COLORS[label]}"/>',
                  f'<text x="{x+22}" y="{y}" font-size="13">{escape(text_label)}</text>']
    return "\n".join(parts + ['</g>', '</svg>', ''])


def write_charts(rows, directory):
    directory.mkdir(parents=True, exist_ok=True)
    for filename, metric, title, unit in CHARTS:
        (directory / filename).write_text(chart(rows, metric, title, unit))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True,
                        help="Completed reproduce.py output with NFFN/ and FNNF/ raw requests")
    parser.add_argument("--output", type=Path, help="SVG directory; defaults to the results directory")
    args = parser.parse_args()
    summary = aggregate(args.results)
    (args.results / "summary.json").write_text(json.dumps(summary, indent=2)+"\n")
    write_charts(summary["scenarios"], args.output or args.results)
    print(json.dumps(summary["scenarios"], indent=2))


if __name__ == "__main__":
    main()
