"""Add standalone measurements to the decode latency and throughput SVGs.

Runtime/prefill paths differ: decode latency alone implies no request speedup.
Raw sequences are rechecked before the comparison charts can be emitted.
"""

import argparse
from collections import defaultdict
import gzip
import hashlib
import json
import math
from pathlib import Path
import statistics

from accuracy import require_same_tokens
from render_results import aggregate, write_charts
from triton.flagmega.serving.metrics import summarize


def read_report(path):
    return json.loads(gzip.decompress(path.read_bytes()) if path.suffix == ".gz" else path.read_text())


def require_environment(actual, expected, keys):
    for key in keys:
        value, reference = actual.get(key), expected[key]
        # Older CLI acquisition recorded torch.__version__ (2.10.0), whereas
        # FlagOS records distribution metadata (2.10.0+corex...). Preserve both
        # raw values; COREX device and compiler identity are checked separately.
        if key == "torch" and isinstance(value, str) and isinstance(reference, str):
            if "+" not in value or "+" not in reference:
                value, reference = value.split("+", 1)[0], reference.split("+", 1)[0]
        if value != reference:
            raise ValueError(f"Cannot compare mismatched {key}")


def aggregate_chat(paths):
    groups = defaultdict(list)
    contract = None
    reports = []
    rounds = set()
    scenario_set = None
    for path in sorted(paths):
        report = read_report(path)
        if (not report.get("performance_valid") or report.get("vllm_imported") is not False
                or report.get("prefill_mode") != "compiled_token_scan" or report.get("prefix_caching") is not False):
            raise ValueError("Expected independent standalone token-scan measurements")
        identity = (report["checkpoint_revision"], report["visible_devices"], report["torch"],
                    report["backend"]["artifact_source_sha256"], report["backend"]["cuda_graph"],
                    report["backend"]["serving_source_sha256"],
                    report["batch"], report["concurrency"], json.dumps(report["sampling"], sort_keys=True),
                    report.get("triton"), report.get("compiler_sha256"), report.get("benchmark_sha256"),
                    json.dumps(report.get("checkpoint_files"), sort_keys=True),
                    json.dumps(report.get("artifact_codegen"), sort_keys=True),
                    report.get("native_reference_prefill"), report.get("native_reference_sha256"),
                    report.get("process_rounds"), report.get("max_start_temperature_c"),
                    report.get("conditions_sha256"))
        if contract is not None and contract != identity:
            raise ValueError("Cannot merge different artifacts, environments, or sampling contracts")
        contract = identity
        if report["round"] in rounds:
            raise ValueError("Duplicate standalone round")
        rounds.add(report["round"])
        lengths = {s["prompt_length"] for s in report["scenarios"]}
        if len(lengths) != len(report["scenarios"]) or (scenario_set is not None and lengths != scenario_set):
            raise ValueError("Standalone scenario sets differ across rounds")
        scenario_set = lengths
        reports.append(report)
        for scenario in report["scenarios"]:
            if not scenario["runs"] or ("repeats" in report and len(scenario["runs"]) != report["repeats"]):
                raise ValueError("Incomplete standalone requests")
            groups[scenario["prompt_length"]].extend(scenario["runs"])
    if not groups:
        raise ValueError("No standalone measurements")
    rows = []
    for length, runs in sorted(groups.items()):
        expected = runs[0]
        warmups = [run for report in reports for scenario in report.get("warmup_scenarios", [])
                   if scenario["prompt_length"] == length for run in scenario["runs"]]
        for run in runs + warmups:
            if run["prompt_tokens"] != expected["prompt_tokens"] or run["cached_tokens"] != 0:
                raise ValueError("Different prompts or reused prefixes")
            require_same_tokens(run["token_ids"], expected["token_ids"], context="standalone repeated request")
            if len(run["decode_step_ms"]) != len(run["token_ids"]) - 1:
                raise ValueError("Decode samples must exclude exactly the first output token")
            if len(run["prompt_tokens"]) != length or any(not math.isfinite(v) or v <= 0
                    for v in [*run["decode_step_ms"], run["total_ms"], run["prefill_ms"], run["ttft_ms"]]):
                raise ValueError("Invalid standalone prompt or timing boundary")
        decode = summarize([value for run in runs for value in run["decode_step_ms"]])
        if decode["count"] == 0:
            raise ValueError("Need multi-token decode measurements")
        if any(run["total_ms"] <= 0 for run in runs):
            raise ValueError("Request throughput needs positive complete request times")
        metrics = {"runs": len(runs), "decode_samples": decode["count"],
                   "decode_median_ms": decode["median"], "decode_p95_ms": decode["p95"],
                   "decode_tokens_per_second": statistics.median(
                       len(run["decode_step_ms"]) * 1000 / sum(run["decode_step_ms"]) for run in runs),
                   "prefill_median_ms": statistics.median(run["prefill_ms"] for run in runs),
                   "ttft_median_ms": statistics.median(run["ttft_ms"] for run in runs),
                   "request_e2e_median_ms": statistics.median(run["total_ms"] for run in runs),
                   "request_tokens_per_second": statistics.median(
                       len(run["token_ids"]) * 1000 / run["total_ms"] for run in runs)}
        rows.append({"prompt_length": length, "output_tokens": len(expected["token_ids"]),
                     "decode_tokens": len(expected["token_ids"]), **metrics,
                     "variants": {"chat_cli": metrics}})
    return rows, groups, reports


def add_flagos_comparison(rows, groups, reports, directory, comparison=None, native_reference=None):
    from render_flagos_results import aggregate
    if (directory / "failure.json").exists():
        raise ValueError("Cannot publish a failed FlagOS experiment")
    comparison = aggregate(directory) if comparison is None else comparison
    environment = comparison["environment"]
    for report in reports:
        if report.get("process_rounds") != 1:
            raise ValueError("FlagOS comparison requires one standalone round per fresh process")
        for key, value in comparison.get("conditions", {}).items():
            if report.get(key) != value:
                raise ValueError(f"Standalone and FlagOS scenario conditions differ at {key}")
        for scenario in report["scenarios"]:
            if report.get("max_start_temperature_c") is not None:
                conditions = scenario["start_conditions"]
                if (conditions["max_start_temperature_c"] != report["max_start_temperature_c"] or
                        conditions["observations"][-1]["temperature_c"] > report["max_start_temperature_c"]):
                    raise ValueError("Standalone scenario start temperature exceeds its measurement contract")
        require_environment(report, environment,
                            ("checkpoint_files", "visible_devices", "torch", "triton", "compiler_sha256"))
        if report["backend"]["gpu"] != environment["device"]:
            raise ValueError("Standalone and FlagOS GPUs differ")
        if report.get("artifact_codegen") != comparison["codegen"]:
            raise ValueError("Standalone and FlagOS must execute the same source and library")
        if report["backend"]["artifact_source_sha256"] != comparison["source_sha256"]:
            raise ValueError("Standalone backend source differs from its codegen identity")
        if not report["backend"]["cuda_graph"] or report["batch"] != 1 or report["concurrency"] != 1:
            raise ValueError("Expected single-request standalone CUDA Graph")
        if report["sampling"] != {"temperature": 0.0, "top_k": 0, "top_p": 1.0, "seed": 0}:
            raise ValueError("Standalone and FlagOS sampling contracts differ")
        if not report.get("native_token_agreement"):
            raise ValueError("Standalone needs independent native token verification")
    native_path = next(path for path in comparison["sources"] if ".native.json" in path)
    native = read_report(directory / native_path)
    reference = native
    if native_reference:
        reference = read_report(native_reference)
        if (reference.get("schema") != "flagmega.native-token-scan-reference/v1"
                or not reference.get("accepted") or reference.get("max_num_batched_tokens") != 1
                or reference.get("prefill_mode") != "native_token_scan"):
            raise ValueError("Need accepted independent native token-scan reference")
        require_environment(reference, native,
                            ("checkpoint_files", "visible_devices", "torch", "triton", "compiler_sha256", "vllm"))
        for key in ("vllm_source", "flaggems_source", "plugin_source"):
            if reference[key] != native[key]:
                raise ValueError(f"Native token-scan reference source differs at {key}")
        for report in reports:
            if report.get("native_reference_prefill") != "native_token_scan":
                raise ValueError("Standalone acquisition did not use native token-scan verification")
            reference_bytes = native_reference.read_bytes()
            if native_reference.suffix == ".gz":
                reference_bytes = gzip.decompress(reference_bytes)
            if report.get("native_reference_sha256") != hashlib.sha256(reference_bytes).hexdigest():
                raise ValueError("Standalone native reference identity differs")
    elif any(report.get("native_reference_prefill") == "native_token_scan" for report in reports):
        raise ValueError("Provide the standalone native token-scan reference")
    expected = {s["prompt_length"]: s["runs"][0] for s in reference["scenarios"]}
    batched = {s["prompt_length"]: s["runs"][0] for s in native["scenarios"]}
    if set(expected) != set(groups):
        raise ValueError("Standalone and FlagOS scenario sets differ")
    for length, runs in groups.items():
        for run in runs:
            if run["prompt_tokens"] != expected[length]["prompt_tokens"]:
                raise ValueError("Standalone and FlagOS prompts differ")
            require_same_tokens(run["token_ids"], expected[length]["token_ids"],
                                context="standalone vs native FlagOS prefill reference")
    for report in reports:
        for scenario in report.get("warmup_scenarios", []):
            for run in scenario["runs"]:
                require_same_tokens(run["token_ids"], expected[scenario["prompt_length"]]["token_ids"],
                                    context="standalone warmup vs native FlagOS")
    by_length = {row["prompt_length"]: row for row in rows}
    for row in comparison["scenarios"]:
        cli = by_length[row["prompt_length"]]
        if cli["runs"] != row["variants"]["native"]["runs"]:
            raise ValueError("Standalone and FlagOS need equally many measured requests")
        row["variants"]["chat_cli"] = cli["variants"]["chat_cli"]
    comparison["labels"]["chat_cli"] = "FlagMega chat CLI (CUDA Graph, no vLLM)"
    comparison["chat_cli"] = {"rounds": len(reports), "native_token_agreement": True,
                               "native_reference_prefill": "native_token_scan" if native_reference else "native_batched",
                               "native_batched_tokens_exact": all(run["token_ids"] == batched[length]["token_ids"]
                                                                   for length, runs in groups.items() for run in runs),
                               "prefill": "compiled_token_scan", "terminal_io": False,
                               "boundary": reports[0]["boundary"],
                               "serving_source_sha256": reports[0]["backend"]["serving_source_sha256"]}
    return comparison


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
    comparisons = parser.add_mutually_exclusive_group()
    comparisons.add_argument("--vllm-results", type=Path)
    comparisons.add_argument("--flagos-results", type=Path)
    parser.add_argument("--native-reference", type=Path, help="Independent native token-scan correctness report")
    parser.add_argument("--figure-dir", type=Path, help="Write the three SVGs here (default: results directory)")
    args = parser.parse_args()
    if (args.results / "failure.json").exists():
        raise ValueError("Cannot publish a failed experiment")
    paths = sorted([*args.results.glob("round*.chat_cli.json"), *args.results.glob("round*.chat_cli.json.gz")])
    rows, groups, reports = aggregate_chat(paths)
    if args.vllm_results:
        add_vllm_comparison(rows, groups, reports, args.vllm_results)
    summary = {"sources": [p.name for p in paths], "scenarios": rows}
    if args.flagos_results:
        from render_flagos_results import chart, CHARTS
        summary = add_flagos_comparison(rows, groups, reports, args.flagos_results,
                                        native_reference=args.native_reference)
        summary["chat_cli"]["sources"] = [p.name for p in paths]
        directory = args.figure_dir or args.results
        directory.mkdir(parents=True, exist_ok=True)
        for filename, metric, title, unit in CHARTS:
            (directory / filename).write_text(chart(summary["scenarios"], metric, title, unit, summary["labels"],
                                                     summary.get("conditions")))
    else:
        write_charts(rows, args.figure_dir or args.results)
    (args.results / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    for row in rows:
        print(json.dumps(row))


if __name__ == "__main__":
    main()
