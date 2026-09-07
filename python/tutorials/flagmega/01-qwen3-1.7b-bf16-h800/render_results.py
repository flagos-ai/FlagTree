"""Deterministic SVGs derived exclusively from successful measured requests."""

import argparse
from collections import defaultdict
from html import escape
import json
from pathlib import Path
import statistics

from accuracy import require_same_tokens

LABELS = {"native_vllm": "Native vLLM", "baseline": "FlagMega initial (vLLM numerics)",
          "scheduled": "Agent: layout + kernel choices", "serving": "Agent: + serving ABI"}
CHART_LABELS = {**LABELS, "chat_cli": "FlagMega chat CLI (no vLLM)"}
COLORS = {"native_vllm": "#556579", "baseline": "#c47725", "scheduled": "#247da0", "serving": "#198269",
          "chat_cli": "#8252b3"}
CHARTS = (
    ("decode_latency.svg", "decode_median_ms", "Decode step latency (lower is better)", "ms/token"),
    ("request_throughput.svg", "request_tokens_per_second", "Request throughput incl. prefill (higher is better)", "tokens/s"),
)


def aggregate(paths):
    groups = defaultdict(list)
    metadata = None
    artifact_hashes = {}
    for path in sorted(paths):
        report = json.loads(path.read_text())
        if not report.get("performance_valid"):
            raise ValueError(f"Not a performance run: {path}")
        if report["label"] not in LABELS:
            raise ValueError(f"Unknown variant {report['label']}")
        label = report["label"]
        source_hash = report.get("artifact_source_sha256")
        if label in artifact_hashes and artifact_hashes[label] != source_hash:
            raise ValueError("Cannot combine different generated kernels for one variant")
        artifact_hashes[label] = source_hash
        contract = tuple(report[key] for key in ("batch", "concurrency", "cuda_graph", "checkpoint_revision",
                                                 "visible_devices", "torch", "vllm"))
        if metadata is not None and metadata != contract:
            raise ValueError("Cannot combine different environments/workload contracts")
        metadata = contract
        for scenario in report["scenarios"]:
            groups[(scenario["prompt_length"], report["label"])].extend(scenario["runs"])
    if not groups:
        raise ValueError("No measured results")
    rows = []
    for length in sorted({key[0] for key in groups}):
        row = {"prompt_length": length, "variants": {}}
        native = groups[(length, "native_vllm")]
        if not native:
            raise ValueError("Every scenario needs native reference runs")
        row["decode_tokens"] = len(native[0]["token_ids"])
        for label in LABELS:
            runs = groups[(length, label)]
            if not runs or not native or len(runs) != len(native):
                raise ValueError("Every scenario needs equally many runs of every variant")
            if any(run["prompt_tokens"] != native[0]["prompt_tokens"] for run in runs):
                raise ValueError("Mismatched prompt inputs")
            if any(len(run["token_ids"]) != len(native[0]["token_ids"]) for run in runs):
                raise ValueError("Mismatched decode lengths")
            for run in runs:
                require_same_tokens(run["token_ids"], native[0]["token_ids"],
                                    context=f"{label}, prompt length {length}")
            decode = sorted(value for run in runs for value in run["decode_step_ms"])
            row["variants"][label] = {
                "decode_median_ms": statistics.median(decode),
                "decode_p95_ms": decode[min(len(decode)-1, int(.95 * len(decode)))],
                "ttft_median_ms": statistics.median(run["ttft_ms"] for run in runs),
                "request_tokens_per_second": statistics.median(run["output_tokens_per_second"] for run in runs),
                "request_e2e_median_ms": statistics.median(run["e2e_ms"] for run in runs),
                "native_token_agreement": sum(a == b for run, ref in zip(runs, native, strict=True)
                                              for a, b in zip(run["token_ids"], ref["token_ids"], strict=True))
                                          / sum(len(run["token_ids"]) for run in runs),
                "runs": len(runs), "decode_samples": len(decode),
            }
        final = row["variants"]["serving"]["decode_median_ms"]
        row["speedup_vs_native"] = row["variants"]["native_vllm"]["decode_median_ms"] / final
        row["speedup_vs_initial"] = row["variants"]["baseline"]["decode_median_ms"] / final
        rows.append(row)
    return rows


def chart(rows, metric, title, unit):
    if not rows or any(set(row["variants"]) != set(rows[0]["variants"]) for row in rows):
        raise ValueError("Every chart scenario must contain the same variants")
    labels = {key: value for key, value in CHART_LABELS.items() if key in rows[0]["variants"]}
    if not labels or set(labels) != set(rows[0]["variants"]):
        raise ValueError("Unknown chart variant")
    has_cli = "chat_cli" in labels
    width, height = 1200, 690
    left, right, top, bottom = 85, 30, 145, 150
    plot_width, plot_height = width-left-right, height-top-bottom
    maximum = max(row["variants"][label][metric] for row in rows for label in labels) * 1.18
    output_tokens = "/".join(str(value) for value in sorted({row["decode_tokens"] for row in rows}))
    description = "Qwen3-1.7B BF16, one H800, batch one. Bars are measured medians."
    if has_cli:
        description += " CLI and vLLM runtime/prefill paths differ; decode latency is not an end-to-end speedup."
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img">',
             f'<title>{escape(title)}</title>',
             f'<desc>{escape(description)}</desc>',
             '<rect width="100%" height="100%" fill="#ffffff"/>',
             '<g font-family="sans-serif" fill="#243343">',
             f'<text x="{left}" y="37" font-size="24">{escape(title)}</text>',
             f'<text x="{left}" y="64" font-size="14">BF16 / H800 / batch 1 / {output_tokens} output tokens / host-visible timing</text>']
    boundary = ("Decode intervals exclude the first output token; not end-to-end request latency."
                if metric == "decode_median_ms" else
                "Output tokens / complete request time, including prefill; not inverse decode latency.")
    parts.append(f'<text x="{left}" y="89" font-size="14">{escape(boundary)}</text>')
    if has_cli:
        note = ("CLI: compiled token-scan prefill. vLLM: batched prefill. Data collected in separate runs."
                if "native_vllm" in labels else "CLI: compiled token-scan prefill.")
        parts.append(f'<text x="{left}" y="113" font-size="14">{escape(note)} CLI terminal I/O excluded.</text>')
    for tick in range(6):
        value = maximum*tick/5
        y = top+plot_height*(1-tick/5)
        parts += [f'<path d="M {left} {y:.1f} H {width-right}" stroke="#dfe5eb"/>',
                  f'<text x="{left-12}" y="{y+5:.1f}" text-anchor="end" font-size="13">{value:.1f}</text>']
    group_width = plot_width/len(rows)
    bar_width = group_width*.82/(len(labels)+(len(labels)-1)*.15)
    for group, row in enumerate(rows):
        for variant, label in enumerate(labels):
            value = row["variants"][label][metric]
            h = plot_height*value/maximum
            x = left+group_width*(group+.09)+variant*bar_width*1.15
            y = top+plot_height-h
            number = f"{value:.3f}" if metric == "decode_median_ms" else f"{value:.2f}"
            parts += [f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_width:.1f}" height="{h:.1f}" fill="{COLORS[label]}"/>',
                      f'<text x="{x+bar_width/2:.1f}" y="{y-8:.1f}" text-anchor="middle" font-size="12">{number}</text>']
        parts.append(f'<text x="{left+group_width*(group+.5):.1f}" y="{top+plot_height+28}" text-anchor="middle" font-size="14">Prompt {row["prompt_length"]}</text>')
    parts.append(f'<text x="16" y="{top-16}" font-size="13">{escape(unit)}</text>')
    for i, (label, text_label) in enumerate(labels.items()):
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
    parser.add_argument("--results", type=Path, required=True)
    args = parser.parse_args()
    paths = sorted(args.results.glob("round*.json"))
    rows = aggregate(paths)
    (args.results / "summary.json").write_text(json.dumps({"sources": [p.name for p in paths], "scenarios": rows}, indent=2)+"\n")
    write_charts(rows, args.results)
    print("| Prompt | Native ms | Initial ms | Scheduled ms | Serving ms | vs native | vs initial |")
    print("| --- | --- | --- | --- | --- | --- | --- |")
    for row in rows:
        numbers = " | ".join(f"{row['variants'][label]['decode_median_ms']:.3f}" for label in LABELS)
        print(f"| {row['prompt_length']} | {numbers} | {row['speedup_vs_native']:.2f}x | {row['speedup_vs_initial']:.2f}x |")


if __name__ == "__main__":
    main()
