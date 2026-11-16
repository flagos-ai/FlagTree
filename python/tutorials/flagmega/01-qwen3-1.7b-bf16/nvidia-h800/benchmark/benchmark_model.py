#!/usr/bin/env python3
"""Measure independently generated full-model FlagMega decode, retaining raw trials.

The event cohort encloses input transfer, embedding, all layers, final norm, LM head,
greedy sampling, state update and device-to-host token copy. Host intervals also
include Python dispatch and synchronous delivery. The ordinary host cohort uses
the unchanged serving backend without CUDA events. Prefill/first output are
excluded from both decode metrics. Cache resets and warmup are outside the interval.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import time
import traceback


PROMPT_TEXT = "Explain why the sky is blue, and how sunlight interacts with air. "
CHECKPOINT_REVISION = "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_new(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(data, stream, indent=2, allow_nan=False)
        stream.write("\n")


def summary(values):
    values = sorted(values)
    if not values or any(not math.isfinite(x) or x <= 0 for x in values):
        raise ValueError("Timing samples must be finite and positive")
    return {"count": len(values), "median_ms": statistics.median(values),
            "mean_ms": statistics.mean(values), "min_ms": values[0],
            "p95_ms": values[min(len(values)-1, math.ceil(.95 * len(values))-1)]}


def prompts(checkpoint, lengths):
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(checkpoint), local_files_only=True)
    source = tokenizer.encode(PROMPT_TEXT, add_special_tokens=False)
    return [(source * math.ceil(length / len(source)))[:length] for length in lengths]


def validate_checkpoint(checkpoint):
    checkpoint = Path(checkpoint).resolve()
    config = json.loads((checkpoint / "config.json").read_text())
    required = {"model_type": "qwen3", "num_hidden_layers": 28,
                "hidden_size": 2048, "num_attention_heads": 16,
                "num_key_value_heads": 8, "vocab_size": 151936}
    for key, expected in required.items():
        if config.get(key) != expected:
            raise ValueError(f"Expected Qwen3-1.7B {key}={expected}, got {config.get(key)}")
    if config.get("torch_dtype", config.get("dtype")) != "bfloat16":
        raise ValueError("Published checkpoint must be BF16")
    weights = sorted(checkpoint.glob("model*.safetensors"))
    if not weights:
        raise ValueError("No actual safetensors checkpoint payload")
    return {"path": str(checkpoint), "revision": checkpoint.name,
            "config_sha256": sha(checkpoint / "config.json"),
            "weights": [{"path": str(p), "bytes": p.stat().st_size,
                         "hf_blob": p.resolve().name} for p in weights]}


def gpu_inventory():
    return subprocess.check_output([
        "nvidia-smi", "--query-gpu=index,uuid,name,driver_version,memory.total,"
        "temperature.gpu,power.draw,clocks.sm,clocks.mem,utilization.gpu,memory.used",
        "--format=csv,noheader,nounits"], text=True).strip()


def timed_backend(artifact, device, cuda_graph, collect_events=True):
    import torch
    from triton.flagmega.serving.backend import ArtifactBackend
    from triton.flagmega.serving.sampling import SamplingConfig

    if not collect_events:
        return ArtifactBackend.load(artifact, device=device,
                                    sampling=SamplingConfig(temperature=0, seed=0),
                                    cuda_graph=cuda_graph, warmups=3)

    class TimedBackend(ArtifactBackend):
        def append(self, token):
            self._measure_start.record(torch.cuda.current_stream(self.device))
            super().append(token)

        def sample(self):
            self._require_open()
            if not self._has_logits:
                raise ValueError("Sampling requires current logits")
            # temperature=0 is part of the frozen contract. The artifact/graph
            # already computes greedy argmax, exactly as ArtifactBackend.sample.
            self._cpu_token.copy_(self.token, non_blocking=True)
            self._measure_end.record(torch.cuda.current_stream(self.device))
            self.synchronize()
            self.gpu_steps_ms.append(self._measure_start.elapsed_time(self._measure_end))
            return int(self._cpu_token.item())

    result = TimedBackend.load(artifact, device=device,
                              sampling=SamplingConfig(temperature=0, seed=0),
                              cuda_graph=cuda_graph, warmups=3)
    result.gpu_steps_ms = []
    # CUDA event handles are allocated and initialized before any request.
    # Reuse is safe because sample synchronizes before the next append.
    result._measure_start = torch.cuda.Event(enable_timing=True)
    result._measure_end = torch.cuda.Event(enable_timing=True)
    result._measure_start.record(torch.cuda.current_stream(result.device))
    result._measure_end.record(torch.cuda.current_stream(result.device))
    result._measure_end.synchronize()
    return result


def one_request(engine, prompt, decode_tokens, collect_events=True):
    engine.reset()
    engine.backend.synchronize()
    if collect_events:
        engine.backend.gpu_steps_ms.clear()
    result = engine.generate(prompt, max_new_tokens=decode_tokens, ignore_eos=True)
    record = result.to_dict()
    if collect_events:
        record["device_decode_step_ms"] = engine.backend.gpu_steps_ms[1:]
    if collect_events and len(record["device_decode_step_ms"]) != decode_tokens - 1:
        raise RuntimeError("Expected exactly one actual device interval per decoded output after first")
    if len(record["token_ids"]) != decode_tokens or len(record["decode_step_ms"]) != decode_tokens - 1:
        raise RuntimeError("Incomplete independently generated trajectory")
    if len(engine.cached_tokens) != len(prompt) + decode_tokens - 1:
        raise RuntimeError("Consumed history did not advance once per decoded token")
    record["final_consumed_tokens"] = list(engine.cached_tokens)
    record["final_state_length"] = engine.backend.state.sequence_length
    if record["final_state_length"] != len(engine.cached_tokens):
        raise RuntimeError("GPU KV length differs from actual independently generated history")
    return record


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--label", default="flagmega-baseline")
    parser.add_argument("--measurement-id", required=True)
    parser.add_argument("--prompt-lengths", nargs="+", type=int, default=[32, 128, 250, 1024, 2048])
    parser.add_argument("--decode-tokens", type=int, default=64)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--round-index", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--no-cuda-graph", action="store_true")
    parser.add_argument("--cohort", choices=["event_instrumented", "ordinary_no_events"], default="event_instrumented")
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args(argv)
    collect_events = args.cohort == "event_instrumented"
    if args.output.exists() or args.output.with_suffix(".failed.json").exists():
        raise ValueError("Raw trials are immutable; choose a fresh output")
    if args.decode_tokens < 2 or args.repeats < 1 or args.warmups < 0 or min(args.prompt_lengths) < 2:
        raise ValueError("Require ≥2 output/prompt tokens, ≥1 repeat and ≥0 warmups")
    checkpoint = validate_checkpoint(args.checkpoint)
    # The loadable FlagMega runtime manifest is artifact.json. The workspace
    # publication manifest.json describes exported source files, not a runtime.
    manifest = args.artifact / "artifact.json"
    if not manifest.is_file():
        raise ValueError("Require a complete artifact manifest")
    source = Path(__file__).resolve()
    report = {"schema": "flagmega.model-measurement/v1", "measurement_id": args.measurement_id,
              "label": args.label, "backend": "flagmega", "status": "preflight",
              "timestamp_utc": datetime.now(timezone.utc).isoformat(),
              "script": {"path": str(source), "sha256": sha(source)},
              "checkpoint": checkpoint, "artifact": {"path": str(args.artifact.resolve()),
                  "manifest_sha256": sha(manifest)}, "round_index": args.round_index,
              "visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
              "cohort": args.cohort,
              "workload": {"batch": 1, "concurrency": 1, "dtype": "bfloat16", "layers": 28,
                  "prompt_lengths": args.prompt_lengths, "decode_tokens": args.decode_tokens,
                  "prefix_caching": False, "speculative_decoding": False,
                  "sampling": "greedy", "prompt_text": PROMPT_TEXT},
              "controls": {"warmups_per_scenario": args.warmups, "repeats": args.repeats,
                  "reset": "zero KV and all token/history state before every request",
                  "cuda_graph": not args.no_cuda_graph, "cache_flush": "none; resident decode",
                  "event_resources": {"pairs": 1, "allocation": "outside all requests", "reuse": "after stream synchronization"} if collect_events else None,
                  "comparison_order": "external process coordinator rotates per round"},
              "boundary": {"device": "input fill+fullmodel+state+argmax+GPU-to-host token copy, CUDA events" if collect_events else None,
                  "host": "synchronous generation token delivery with event record/query" if collect_events else "ordinary synchronous generation token delivery; no event record/query",
                  "prefill": "actual compiled token scan, excluded from decode"}, "scenarios": []}
    if args.preflight_only:
        write_new(args.output, report)
        print(json.dumps({"status": "preflight", "artifact": report["artifact"]}))
        return 0
    backend = None
    try:
        import torch
        from triton.flagmega.serving.engine import TextGenerationEngine
        torch.set_num_threads(1)
        if not torch.cuda.is_available():
            raise RuntimeError("Actual CUDA device required")
        report["gpu_before"] = gpu_inventory()
        backend = timed_backend(args.artifact, args.device, not args.no_cuda_graph, collect_events)
        if backend.model.state_config.num_layers != 28 or backend.vocab_size != 151936:
            raise ValueError("Require all28layers and full LMhead, not isolated layer or hidden output")
        if max(args.prompt_lengths) + args.decode_tokens > backend.capacity:
            raise ValueError("Full scenario exceeds artifact KV capacity")
        engine = TextGenerationEngine(backend)
        for length, prompt in zip(args.prompt_lengths, prompts(args.checkpoint, args.prompt_lengths), strict=True):
            for _ in range(args.warmups):
                one_request(engine, prompt, args.decode_tokens, collect_events)
            runs = []
            scenario = {"prompt_length": length, "runs": runs}
            report["scenarios"].append(scenario)
            for repeat in range(args.repeats):
                run = one_request(engine, prompt, args.decode_tokens, collect_events)
                run["repeat_index"] = repeat
                runs.append(run)
            wall_samples = [value for run in runs for value in run["decode_step_ms"]]
            scenario.update(host=summary(wall_samples), total_decode_tokens=len(wall_samples),
                            host_tokens_per_second=len(wall_samples)*1000/sum(wall_samples))
            if collect_events:
                device_samples = [value for run in runs for value in run["device_decode_step_ms"]]
                scenario.update(device=summary(device_samples), total_device_ms=sum(device_samples),
                                device_tokens_per_second=len(device_samples)*1000/sum(device_samples))
            print(json.dumps({k:v for k,v in scenario.items() if k != "runs"}), flush=True)
        report["runtime"] = backend.info()
        report["gpu_after"] = gpu_inventory()
        report["status"] = "measured"
        report["acceptance"] = "pending independent reviewer/reference comparison"
        write_new(args.output, report)
    except BaseException as error:
        report.update(status="failed", error=str(error), traceback=traceback.format_exc())
        write_new(args.output.with_suffix(".failed.json"), report)
        raise
    finally:
        if backend is not None:
            backend.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
