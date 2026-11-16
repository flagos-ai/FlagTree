#!/usr/bin/env python3
"""Capture actual full decode CUDA activities; preserve unobserved inner work.

This uses Torch/CUPTI tracing, independently from ordinary performance trials.
Generated-kernel inner operator attribution stays unavailable when the trace has
only a monolithic kernel. No synthetic partition of that kernel is produced.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import traceback

from benchmark_model import prompts, sha, timed_backend, validate_checkpoint, write_new


MARKER = "flagmega_whole_decode_step"


def audit_trace(path):
    trace = json.loads(Path(path).read_text())
    events = trace["traceEvents"]
    markers = [e for e in events if e.get("name") == MARKER and e.get("ph") == "X"]
    if len(markers) != 1:
        raise ValueError(f"Expected exactly one declared full decode window, found{len(markers)}")
    marker = markers[0]
    start, end = marker["ts"], marker["ts"] + marker["dur"]
    activities = [e for e in events if e.get("ph") == "X" and
                  any(word in e.get("cat", "").lower() for word in ["kernel", "gpu_memcpy", "gpu_memset"]) and
                  start <= e["ts"] and e["ts"] + e.get("dur", 0) <= end]
    if not activities:
        raise ValueError("Trace has no actual CUDA activities in the declared full decode window")
    return marker, activities


def figures(trace_path, capture_id, device_ms):
    marker, activities = audit_trace(trace_path)
    start = marker["ts"]
    groups = defaultdict(list)
    for index, event in enumerate(activities):
        groups[event["name"]].append({"id": f"cuda{index}",
            "start_ns": round((event["ts"]-start)*1000),
            "end_ns": round((event["ts"]+event["dur"]-start)*1000),
            "kind": "transfer" if "memcpy" in event.get("cat", "").lower() else "simt",
            "evidence": "measured", "raw_event_index": index})
    provenance = [{"path": str(trace_path.resolve()), "sha256": sha(trace_path)}]
    common = {"schema_version": 1, "synthetic": False, "cohort": capture_id,
              "provenance": provenance, "uncertainty_method": "single instrumented invocation; no confidence claim"}
    graph = dict(common, title="Full Qwen3-1.7B decode CUDA activity timeline", kind="timeline", unit="ns",
                 metric="Actual recorded CUDA activities", invocation=capture_id,
                 scope="whole_model", chart_role="full_model_timeline",
                 window_ns=[0, round(marker["dur"]*1000)],
                 timebase={"source": "Torch/CUPTI Chrome trace timestamp in microseconds converted to ns",
                           "resolution_ns": 1000, "alignment": "CPU and CUDA share Chrome trace clock"},
                 rows=[{"label": name, "status": "captured", "intervals": spans} for name, spans in groups.items()])
    operators = dict(common, title="Full decode recorded CUDA activities, longest first", kind="bar", unit="us",
                     metric="Summed CUDA activity duration", chart_role="graph_operator_latency",
                     sort_by="value_desc",
                     aggregation="sum actual CUDA activity durations by kernel name; overlap is not elapsed time; fused inner operators unavailable",
                     rows=[{"label": name, "value": sum(s["end_ns"]-s["start_ns"] for s in spans)/1000,
                            "status": "captured"} for name, spans in groups.items()] +
                     [{"label": "inner operators of fused kernels", "value": None,
                       "reason": "unavailable: require native TLE instrumentation; cannot derive from kernel duration"}])
    audit = {"capture_id": capture_id, "trace_sha256": sha(trace_path),
             "full_host_window_us": marker["dur"], "device_event_window_ms": device_ms,
             "recorded_cuda_activities": len(activities),
             "activity_total_us": sum(e["dur"] for e in activities),
             "unattributed_inner_operators": True, "figure_acceptance": "pending independent raw-to-chart review"}
    return graph, operators, audit


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--capture-dir", type=Path, required=True)
    parser.add_argument("--capture-id", required=True)
    parser.add_argument("--prompt-length", type=int, default=128)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args(argv)
    if args.capture_dir.exists():
        raise ValueError("Capture directories are immutable; choose a fresh ID")
    validate_checkpoint(args.checkpoint)
    args.capture_dir.mkdir(parents=True)
    backend = None
    try:
        import torch
        from triton.flagmega.serving.engine import TextGenerationEngine
        torch.set_num_threads(1)
        backend = timed_backend(args.artifact, args.device, True)
        engine = TextGenerationEngine(backend)
        prompt = prompts(args.checkpoint, [args.prompt_length])[0]

        def prepare():
            engine.reset()
            result = engine.generate(prompt, max_new_tokens=1, ignore_eos=True)
            return result.token_ids[0]

        controls = []
        for _ in range(3):
            token = prepare()
            backend.gpu_steps_ms.clear()
            backend.append(token)
            sampled = backend.sample()
            controls.append({"token": sampled, "device_ms": backend.gpu_steps_ms[-1]})
        token = prepare()
        backend.gpu_steps_ms.clear()
        trace = args.capture_dir / "trace.json"
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                               torch.profiler.ProfilerActivity.CUDA]) as prof:
            with torch.profiler.record_function(MARKER):
                backend.append(token)
                sampled = backend.sample()
        prof.export_chrome_trace(str(trace))
        instrumented_ms = backend.gpu_steps_ms[-1]
        graph, operators, audit = figures(trace, args.capture_id, instrumented_ms)
        write_new(args.capture_dir / "figure-graph.json", graph)
        write_new(args.capture_dir / "figure-operators.json", operators)
        write_new(args.capture_dir / "audit.json", audit)
        write_new(args.capture_dir / "controls.json", {"ordinary": controls,
            "instrumented": {"token": sampled, "device_ms": instrumented_ms},
            "overhead_ratio_vs_ordinary_mean": instrumented_ms/(sum(c["device_ms"] for c in controls)/len(controls)),
            "scope": "identical independently computed next decode token after reset and real prompt prefill",
            "status": "diagnostic controls, not performance acceptance"})
        write_new(args.capture_dir / "manifest.json", {"schema": "flagmega.capture/v1",
            "capture_id": args.capture_id, "status": "captured", "provider": "torch.profiler/CUPTI",
            "artifact_manifest_sha256": sha(args.artifact / "artifact.json"), "script_sha256": sha(__file__),
            "scope": "whole28layer decode + embedding/LMhead/state/greedy/delivery",
            "prompt_tokens": prompt, "input_decode_token": token, "generated_token": sampled,
            "raw_files": [{"path": trace.name, "sha256": sha(trace)}],
            "timebase": graph["timebase"], "coverage": audit,
            "open_gates": ["native fused-inner operator timing", "independent raw-to-chart review"]})
    except BaseException as error:
        write_new(args.capture_dir / "failed.json", {"status": "failed", "error": str(error), "traceback": traceback.format_exc()})
        raise
    finally:
        if backend is not None:
            backend.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
