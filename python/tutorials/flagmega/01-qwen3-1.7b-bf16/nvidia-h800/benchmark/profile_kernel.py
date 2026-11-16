#!/usr/bin/env python3
"""Small real Triton/TLE NCU access smoke; timing is diagnostic, not Qwen evidence.

Run this process under NCU with --profile-from-start off. Only the one launch
between cudaProfilerStart/Stop is included. Ordinary warm/control launches are
kept separate. Full-model capture bindings will be supplied by its build manifest.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import statistics
import ctypes
from datetime import datetime, timezone
import os
import subprocess
import sys
import time
import traceback

import torch
import triton
import triton.language as tl
import triton.experimental.tle.language as tle


@triton.jit
def tle_stream_smoke(source, output, size: tl.constexpr, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    slots = tl.arange(0, BLOCK)
    shared = tle.gpu.alloc((BLOCK,), tl.bfloat16, nv_mma_shared_layout=False)
    shared_ptrs = tle.gpu.local_ptr(shared, (slots,))
    values = tl.load(source + offsets, offsets < size, other=0)
    tl.store(shared_ptrs, values)
    tl.debug_barrier()
    recovered = tl.load(shared_ptrs).to(tl.float32) * 2.0
    tl.store(output + offsets, recovered, offsets < size)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--elements", type=int, default=32 * 1024 * 1024)
    parser.add_argument("--controls", type=int, default=20)
    parser.add_argument("--profile", action="store_true")
    args = parser.parse_args()
    if args.elements <= 0 or args.controls <= 0:
        parser.error("elements and controls must be positive")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.cuda.set_device(0)
    properties = torch.cuda.get_device_properties(0)
    source = torch.ones(args.elements, device="cuda", dtype=torch.bfloat16)
    output = torch.empty_like(source)
    grid = (triton.cdiv(args.elements, 1024),)

    def launch():
        return tle_stream_smoke[grid](source, output, args.elements, 1024, num_warps=4)

    compiled = launch()
    for _ in range(5):
        launch()
    torch.cuda.synchronize()
    samples = []
    for _ in range(args.controls):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        launch()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) * 1_000_000)
    if args.profile:
        cudart = torch.cuda.cudart()
        cudart.cudaProfilerStart()
        launch()
        torch.cuda.synchronize()
        cudart.cudaProfilerStop()
    torch.testing.assert_close(output, torch.full_like(output, 2), rtol=0, atol=0)
    source_path = Path(__file__).resolve()
    document = {
        "schema": "flagmega.kernel-smoke/v1",
        "scope": "Triton/TLE shared-pointer streaming smoke, not full Qwen decode",
        "source": str(source_path),
        "source_sha256": hashlib.sha256(source_path.read_bytes()).hexdigest(),
        "device": {"name": properties.name, "visible_index": 0,
                   "uuid": str(getattr(properties, "uuid", "unavailable")),
                   "total_memory": properties.total_memory,
                   "sm_count": properties.multi_processor_count},
        "torch": torch.__version__,
        "triton": triton.__version__,
        "elements": args.elements,
        "dtype": "bfloat16",
        "logical_global_bytes_per_launch": args.elements * 4,
        "launch": {"grid": list(grid), "block_elements": 1024, "num_warps": 4},
        "profiler_start_stop": args.profile,
        "ordinary_control_duration_ns": samples,
        "ordinary_control_median_ns": statistics.median(samples),
        "correctness": "all output elements equal BF16 2, exact",
        "compiled_resources": {"registers": compiled.n_regs,
                               "spills": compiled.n_spills,
                               "shared_bytes": compiled.metadata.shared},
        "ptx_sha256": hashlib.sha256(compiled.asm["ptx"].encode()).hexdigest(),
        "cubin_sha256": hashlib.sha256(compiled.asm["cubin"]).hexdigest(),
        "recorders": {"disabled": "not integrated; ordinary controls above",
                      "minimal": "unavailable", "full": "unavailable"},
        "wg_hardware_activity": "unavailable; counters describe whole kernel",
    }
    args.output.write_text(json.dumps(document, indent=2) + "\n")
    args.output.with_suffix(".ptx").write_text(compiled.asm["ptx"])
    args.output.with_suffix(".cubin").write_bytes(compiled.asm["cubin"])
    print(json.dumps({"output": str(args.output), "correctness": document["correctness"],
                      "ordinary_control_median_ns": document["ordinary_control_median_ns"]}))


def _sha(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def _write_new(path, document):
    with Path(path).open("x") as stream:
        json.dump(document, stream, indent=2, allow_nan=False)
        stream.write("\n")


def _inventory():
    return subprocess.check_output([
        "nvidia-smi", "--query-gpu=index,uuid,name,driver_version,memory.total,"
        "temperature.gpu,power.draw,clocks.sm,clocks.mem,utilization.gpu,memory.used",
        "--format=csv,noheader,nounits"], text=True)


def _native_packet(backend, directory):
    packet = {"runtime": backend.info(), "kernels": []}
    for index, prepared in enumerate(backend.model._prepared):
        compiled = prepared.compiled_kernel
        metadata = compiled.metadata._asdict()
        item = {"name": compiled.name, "resources": prepared.resource_report,
                "metadata": {k: v for k, v in metadata.items()
                             if v is None or isinstance(v, (str, int, float, bool))},
                "assembly": {}}
        # Preserve the real specialization inputs for provider-control builders.
        # Tuple attribute/constexpr keys are arrays, not stringified guesses.
        ast_source = compiled.src
        item["ast_source"] = {
            "hash": ast_source.hash(), "signature": ast_source.signature,
            "attrs": [{"key": list(k) if isinstance(k, tuple) else k, "value": v}
                      for k, v in ast_source.attrs.items()],
            "constexprs": [{"key": list(k), "value": v}
                           for k, v in ast_source.constants.items()],
            "function_argument_names": list(ast_source.fn.arg_names),
        }
        for extension in ("ttir", "ttgir", "llir", "ptx", "cubin"):
            value = compiled.asm.get(extension)
            if value is None:
                continue
            payload = value.encode() if isinstance(value, str) else value
            path = directory / f"kernel-{index}.{extension}"
            path.write_bytes(payload)
            item["assembly"][extension] = {"path": str(path), "sha256": _sha(path),
                                            "bytes": len(payload)}
        cubin = directory / f"kernel-{index}.cubin"
        if cubin.exists():
            sass = directory / f"kernel-{index}.sass"
            with sass.open("wb") as stream:
                subprocess.run(["/usr/local/cuda/bin/cuobjdump", "--dump-sass", str(cubin)],
                               stdout=stream, stderr=subprocess.STDOUT, check=True)
            item["assembly"]["sass"] = {"path": str(sass), "sha256": _sha(sass),
                                          "bytes": sass.stat().st_size}
        packet["kernels"].append(item)
    return packet


def _profiler_api():
    # Driver calls are the markers intercepted by NCU. The CUDA context has
    # already been made current by the real backend preparation/launch.
    library = ctypes.CDLL("libcuda.so.1")
    library.cuProfilerStart.argtypes = []
    library.cuProfilerStop.argtypes = []
    library.cuProfilerStart.restype = ctypes.c_int
    library.cuProfilerStop.restype = ctypes.c_int
    return library


def model_main(argv):
    """Capture complete real own-history decode without changing its graph policy.

    NCU app-range supports the serving CUDA graph and aggregates the full marked
    interval. Kernel replay is an additional main-kernel diagnostic. Every NCU
    application replay retains a unique PID-bound process packet instead of
    overwriting earlier pass evidence. Prefill/reset/checks are outside markers.
    """
    parser = argparse.ArgumentParser(description=model_main.__doc__)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--artifact-manifest-sha256", required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--reference-sha256", required=True)
    parser.add_argument("--capture-dir", type=Path, required=True)
    parser.add_argument("--prompt-lengths", nargs="+", type=int, default=[128])
    parser.add_argument("--decode-tokens", type=int, default=64)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--cohort", choices=["ordinary_no_events", "event_instrumented"],
                        default="ordinary_no_events")
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args(argv)
    if args.decode_tokens != 64 or not set(args.prompt_lengths).issubset({32, 128, 250, 1024, 2048}):
        parser.error("Retain original64-output scenarios; selection is capture allocation only")
    if args.repeats < 1 or args.warmups < 0:
        parser.error("repeats≥1 and warmups≥0 required")
    manifest = args.artifact / "artifact.json"
    if _sha(manifest) != args.artifact_manifest_sha256 or _sha(args.reference) != args.reference_sha256:
        raise ValueError("Frozen artifact/reference identity changed")
    artifact_data = json.loads(manifest.read_text())
    if (artifact_data["codegen_backend"] != "triton" or not artifact_data["codegen"]["use_tle"]
            or artifact_data["target"] != "nvidia-sm90"):
        raise ValueError("Require actual NVIDIA SM90 Triton+TLE complete artifact")
    reference = json.loads(args.reference.read_text())
    cases = {}
    for scenario in reference["scenarios"]:
        first = scenario["runs"][0]
        if any(run["prompt_tokens"] != first["prompt_tokens"] or run["token_ids"] != first["token_ids"]
               for run in scenario["runs"]):
            raise ValueError("Reference must contain deterministic pinned native greedy trajectories")
        if len(first["token_ids"]) != 64 or len(first["prompt_tokens"]) != scenario["prompt_length"]:
            raise ValueError("Reference trajectory incomplete")
        cases[scenario["prompt_length"]] = first
    for length in args.prompt_lengths:
        if length not in cases:
            raise ValueError("Selected original scenario missing from reference")
    process = args.capture_dir / "harness" / f"process-{os.getpid()}"
    process.mkdir(parents=True, exist_ok=False)
    source = Path(__file__).resolve()
    report = {"schema": "flagmega.native-fullmodel-process/v1", "pid": os.getpid(),
              "timestamp_utc": datetime.now(timezone.utc).isoformat(), "status": "preflight",
              "script": {"path": str(source), "sha256": _sha(source)},
              "artifact": {"path": str(args.artifact.resolve()), "manifest_sha256": _sha(manifest),
                           "semantic_hash": artifact_data["semantic_hash"]},
              "reference": {"path": str(args.reference.resolve()), "sha256": _sha(args.reference),
                            "backend": reference["backend"], "version": reference["version"]},
              "workload": {"batch": 1, "layers": 28, "dtype": "bfloat16",
                           "vocabulary": 151936, "prompt_lengths": args.prompt_lengths,
                           "output_tokens": 64, "timed_decode_steps": 63, "greedy": True,
                           "own_history": True, "prefix_reuse": False},
              "provider": {"ncu_markers": "driver cuProfilerStart/Stop", "enabled": args.profile,
                           "cohort": args.cohort, "cuda_graph": True,
                           "replay_mode": "externally selected and pinned job argument"},
              "boundary": {"marked_range": "63steps input fill+original CUDAgraph full128CTA/all28model+LMhead/argmax/state+GPU-to-host token copy and synchronization",
                           "excluded": "prefill firstoutput, cache/state reset, preparation, native export, correctness/finiteness checks",
                           "host_interval": "actual63 independently generated synchronous append+sample calls; profiler perturbation retained",
                           "main_kernel_counters": "separate diagnostic excludes auxiliary kernels/transfer/host gaps",
                           "gpu_events": "preallocated per-step same stream inputfill through D2H, only event cohort"},
              "controls": {"repeats": args.repeats, "warmups": args.warmups,
                           "reset": "all KV/cache/history zero before each scenario/repeat",
                           "cache_flush": "none; ordinary residentdecode policy",
                           "state_progression": "candidate-own greedy output is next input"},
              "scenarios": []}
    if args.preflight_only:
        _write_new(process / "preflight.json", report)
        print(json.dumps({"preflight": str(process), "status": "CPU_validated_no_CUDA"}), flush=True)
        return 0
    backend = None
    try:
        from benchmark_model import timed_backend
        torch.set_num_threads(1)
        torch.cuda.set_device(0)
        report["gpu_before"] = _inventory()
        backend = timed_backend(args.artifact, "cuda:0", True,
                                args.cohort == "event_instrumented")
        if backend.model.state_config.num_layers != 28 or backend.vocab_size != 151936:
            raise ValueError("Full model/lmhead scope changed")
        if backend.capacity < 2112:
            raise ValueError("Artifact cannot cover all original contexts")
        report["native"] = _native_packet(backend, process)
        driver = _profiler_api()

        def run_request(length, profile):
            prompt = cases[length]["prompt_tokens"]
            backend.reset()
            backend.synchronize()
            for token in prompt:
                backend.append(token)
            first = backend.sample()
            generated = [first]
            intervals = []
            if args.cohort == "event_instrumented":
                backend.gpu_steps_ms.clear()
            if profile and driver.cuProfilerStart() != 0:
                raise RuntimeError("Driver cuProfilerStart failed")
            started = time.perf_counter_ns()
            try:
                for _ in range(63):
                    step = time.perf_counter_ns()
                    backend.append(generated[-1])
                    generated.append(backend.sample())
                    intervals.append(time.perf_counter_ns() - step)
            finally:
                ended = time.perf_counter_ns()
                if profile and driver.cuProfilerStop() != 0:
                    raise RuntimeError("Driver cuProfilerStop failed")
            consumed = prompt + generated[:-1]
            if backend.length != len(consumed) or backend.state.sequence_length != len(consumed):
                raise RuntimeError("Own history/KV length did not advance once per actual input")
            if generated != cases[length]["token_ids"]:
                raise RuntimeError("Full64 actual greedy trajectory differs from pinned native reference")
            if not bool(torch.isfinite(backend.logits).all().item()):
                raise RuntimeError("Actual complete vocabulary logits nonfinite")
            run = {"prompt_tokens": prompt, "token_ids": generated,
                   "final_consumed_history": consumed, "final_state_length": len(consumed),
                   "full64_native_reference_exact": True, "whole_decode_host_ns": ended - started,
                   "decode_step_host_ns": intervals, "marked_ncu": profile,
                   "state_slot_mapping": backend.state.slot_mapping.cpu().tolist()}
            if args.cohort == "event_instrumented":
                if len(backend.gpu_steps_ms) != 63:
                    raise RuntimeError("GPUevent cohort does not cover63actual decode steps")
                run["decode_step_gpu_event_ns"] = [v * 1_000_000 for v in backend.gpu_steps_ms]
            return run

        for length in args.prompt_lengths:
            for _ in range(args.warmups):
                run_request(length, False)
            scenario = {"prompt_length": length, "runs": []}
            report["scenarios"].append(scenario)
            for repeat in range(args.repeats):
                run = run_request(length, args.profile)
                run["repeat_index"] = repeat
                scenario["runs"].append(run)
                _write_new(process / f"scenario-{length}-repeat-{repeat}.json", run)
                print(json.dumps({"prompt_length": length, "repeat": repeat,
                                  "whole_decode_host_ns": run["whole_decode_host_ns"],
                                  "full64_native_reference_exact": True}), flush=True)
        report["gpu_after"] = _inventory()
        report["status"] = "captured_native_process_trajectory_valid_counters_external"
        report["ended_utc"] = datetime.now(timezone.utc).isoformat()
        _write_new(process / "process.json", report)
    except BaseException as error:
        report.update(status="failed", error=str(error), traceback=traceback.format_exc())
        _write_new(process / "failed.json", report)
        raise
    finally:
        if backend is not None:
            backend.close()
    return 0


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "model":
        raise SystemExit(model_main(sys.argv[2:]))
    main()
