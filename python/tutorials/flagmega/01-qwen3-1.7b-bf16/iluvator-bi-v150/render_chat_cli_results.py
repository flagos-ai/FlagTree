"""Add standalone measurements to the decode latency and throughput SVGs.

Runtime/prefill paths differ: decode latency alone implies no request speedup.
Raw sequences are rechecked before the comparison charts can be emitted.
"""

import argparse
from collections import defaultdict
import json
from pathlib import Path
import statistics

from accuracy import require_same_tokens
from render_results import aggregate, write_charts
from triton.flagmega.serving.metrics import summarize


def aggregate_chat(paths):
    groups = defaultdict(list)
    contract = None
    reports = []
    for path in sorted(paths):
        report = json.loads(path.read_text())
        if (not report.get("performance_valid") or report.get("vllm_imported") is not False
                or report.get("prefill_mode") != "compiled_token_scan" or report.get("prefix_caching") is not False):
            raise ValueError("Expected independent standalone token-scan measurements")
        identity = (report["checkpoint_revision"], report["visible_devices"], report["torch"],
                    report["backend"]["artifact_source_sha256"], report["backend"]["cuda_graph"],
                    report["backend"]["serving_source_sha256"],
                    report["batch"], report["concurrency"], json.dumps(report["sampling"], sort_keys=True))
        if contract is not None and contract != identity:
            raise ValueError("Cannot merge different artifacts, environments, or sampling contracts")
        contract = identity
        reports.append(report)
        for scenario in report["scenarios"]:
            groups[scenario["prompt_length"]].extend(scenario["runs"])
    if not groups:
        raise ValueError("No standalone measurements")
    rows = []
    for length, runs in sorted(groups.items()):
        expected = runs[0]
        for run in runs:
            if run["prompt_tokens"] != expected["prompt_tokens"] or run["cached_tokens"] != 0:
                raise ValueError("Different prompts or reused prefixes")
            require_same_tokens(run["token_ids"], expected["token_ids"], context="standalone repeated request")
            if len(run["decode_step_ms"]) != len(run["token_ids"]) - 1:
                raise ValueError("Decode samples must exclude exactly the first output token")
        decode = summarize([value for run in runs for value in run["decode_step_ms"]])
        if decode["count"] == 0:
            raise ValueError("Need multi-token decode measurements")
        if any(run["total_ms"] <= 0 for run in runs):
            raise ValueError("Request throughput needs positive complete request times")
        metrics = {"runs": len(runs), "decode_median_ms": decode["median"], "decode_p95_ms": decode["p95"],
                   "decode_tokens_per_second": statistics.median(run["decode_tokens_per_second"] for run in runs),
                   "prefill_median_ms": statistics.median(run["prefill_ms"] for run in runs),
                   "ttft_median_ms": statistics.median(run["ttft_ms"] for run in runs),
                   "request_e2e_median_ms": statistics.median(run["total_ms"] for run in runs),
                   "request_tokens_per_second": statistics.median(
                       len(run["token_ids"]) * 1000 / run["total_ms"] for run in runs)}
        rows.append({"prompt_length": length, "output_tokens": len(expected["token_ids"]),
                     "decode_tokens": len(expected["token_ids"]), **metrics,
                     "variants": {"chat_cli": metrics}})
    return rows, groups, reports


def add_vllm_comparison(rows, groups, reports, directory):
    if (directory / "failure.json").exists():
        raise ValueError("Cannot publish a failed vLLM experiment")
    paths = sorted(directory.glob("round*.json"))
    native = [json.loads(p.read_text()) for p in paths if p.name.endswith(".native_vllm.json")]
    serving = [json.loads(p.read_text()) for p in paths if p.name.endswith(".serving.json")]
    if not native or not serving:
        raise ValueError("Need native and FlagMega-vLLM measurements")
    for report in [*native, *serving]:
        for key in ("checkpoint_revision", "visible_devices", "torch", "batch", "concurrency"):
            if report[key] != reports[0][key]:
                raise ValueError(f"Cannot compare mismatched {key}")
    if any(r["artifact_source_sha256"] != reports[0]["backend"]["artifact_source_sha256"] for r in serving):
        raise ValueError("Standalone and vLLM must execute the same serving artifact")
    native_runs = {s["prompt_length"]: s["runs"][0] for s in native[0]["scenarios"]}
    if set(native_runs) != set(groups):
        raise ValueError("Standalone and vLLM scenario sets differ")
    for length, runs in groups.items():
        reference = native_runs[length]
        for run in runs:
            if run["prompt_tokens"] != reference["prompt_tokens"]:
                raise ValueError("Standalone and vLLM prompts differ")
            require_same_tokens(run["token_ids"], reference["token_ids"], context="standalone vs native vLLM")
    comparison = {row["prompt_length"]: row for row in aggregate(paths)}
    for row in rows:
        variants = comparison[row["prompt_length"]]["variants"]
        row["variants"] = {**variants, **row["variants"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--vllm-results", type=Path)
    parser.add_argument("--figure-dir", type=Path, help="Write the three SVGs here (default: results directory)")
    args = parser.parse_args()
    if (args.results / "failure.json").exists():
        raise ValueError("Cannot publish a failed experiment")
    paths = sorted(args.results.glob("round*.chat_cli.json"))
    rows, groups, reports = aggregate_chat(paths)
    if args.vllm_results:
        add_vllm_comparison(rows, groups, reports, args.vllm_results)
    (args.results / "summary.json").write_text(json.dumps({"sources": [p.name for p in paths], "scenarios": rows}, indent=2) + "\n")
    write_charts(rows, args.figure_dir or args.results)
    for row in rows:
        print(json.dumps(row))


if __name__ == "__main__":
    main()
