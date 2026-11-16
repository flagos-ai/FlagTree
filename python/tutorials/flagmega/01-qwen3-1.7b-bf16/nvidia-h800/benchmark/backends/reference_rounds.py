#!/usr/bin/env python3
"""CPU planning and per-step measurement contracts for native reference rounds.

No backend or CUDA import occurs at module import or during --plan-only. GPU
commands are emitted as argv arrays; this planner never launches them or leases
devices. The orchestrator/reviewer must authorize their later execution.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import sys
import time
import traceback
from device_identity import (selector_policy, validate_visibility, prepare_identity,
                             verify_torch_identity)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from benchmark_model import sha, summary, validate_checkpoint, write_new, gpu_inventory

LENGTHS = [32, 128, 250, 1024, 2048]
COHORTS = ["ordinary_no_events", "event_instrumented"]
WORK = Path(__file__).resolve().parents[2]
REFERENCE = WORK / ".local/cache/comparison/vllm-baseline-round001.json"


def common_arguments(parser):
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--measurement-id", required=True)
    parser.add_argument("--prompt-lengths", nargs="+", type=int, default=LENGTHS)
    parser.add_argument("--decode-tokens", type=int, default=64)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--round-index", type=int, default=0)
    parser.add_argument("--cohort", choices=COHORTS, default="ordinary_no_events")
    parser.add_argument("--reference", type=Path, default=REFERENCE)
    parser.add_argument("--physical-gpu-uuid")
    parser.add_argument("--gpu-selector-kind", choices=["uuid", "nvml-index"], default="uuid")
    parser.add_argument("--cuda-visible-device")
    parser.add_argument("--device-identity-proof", type=Path)
    parser.add_argument("--preflight-only", action="store_true")


def check_arguments(args, backend="flagmega"):
    if args.output.exists() or args.output.with_suffix(".failed.json").exists():
        raise ValueError("Use new immutable output paths for every attempt")
    if (sorted(args.prompt_lengths) != LENGTHS or args.decode_tokens != 64 or
            args.repeats != 3 or args.warmups != 2):
        raise ValueError("Formal reference rounds require all five contexts, own64, warmup2/repeat3")
    selector_policy(backend, args.physical_gpu_uuid, args.gpu_selector_kind,
                    args.cuda_visible_device or (args.physical_gpu_uuid if args.gpu_selector_kind == "uuid" else None))


def reference_prompts(args):
    ref = json.loads(args.reference.read_text())
    checkpoint = validate_checkpoint(args.checkpoint)
    if ref.get("status") != "measured" or ref.get("checkpoint") != checkpoint:
        raise ValueError("Reference must be the measured pinned checkpoint, never a new token source")
    scenarios = {s["prompt_length"]: s for s in ref["scenarios"]}
    if sorted(scenarios) != LENGTHS:
        raise ValueError("Reference omits original contexts")
    selected = {}
    for length in args.prompt_lengths:
        runs = scenarios[length]["runs"]
        prompt, tokens = runs[0]["prompt_tokens"], runs[0]["token_ids"]
        if len(prompt) != length or len(tokens) != 64 or any(r["prompt_tokens"] != prompt or r["token_ids"] != tokens for r in runs):
            raise ValueError("Reference trajectories are incomplete or internally inconsistent")
        selected[length] = {"prompt": prompt, "expected": tokens}
    return checkpoint, selected, {"path": str(args.reference.resolve()), "sha256": sha(args.reference),
        "usage": "prompt IDs at reset and post-run NUM comparison only; never teacher-forced timed inputs"}


def controls(args):
    return {"warmups_per_scenario": 2, "repeats": 3, "round_index": args.round_index,
        "shared_adapter_source": {"path": str(Path(__file__).resolve()), "sha256": sha(__file__)},
        "physical_gpu_uuid": args.physical_gpu_uuid,
        "selector_policy": selector_policy("vllm" if args.gpu_selector_kind == "nvml-index" else "flagmega",
            args.physical_gpu_uuid, args.gpu_selector_kind,
            args.cuda_visible_device or (args.physical_gpu_uuid if args.gpu_selector_kind == "uuid" else None)),
        "actual_device_identity": "pending actual launcher NVML, context-free CUDA and native/Torch checks",
        "NUM_gate": "independent first10 decode IDs [1:11]; token0 and all64/firstdiff retained separately; reference only after timing",
        "synchronization": "complete token and GPU work before each next step; exactly63 per-step intervals",
        "event_allocation": "one pair allocated/initialized before requests and reused after synchronization" if args.cohort == "event_instrumented" else "no harness CUDA events created/recorded/queried",
        "host_clock": "perf_counter_ns; start at step entry, stop after sync and host token delivery; event elapsed query excluded",
        "prefill": "original prompt and first own output untimed", "cache_flush": "none",
        "lifecycle": "reset -> untimed prefill/first output ->63 own-history steps -> outside-timer checks",
        "statistics": "TPS = total timed tokens / total timed seconds; median/p95 diagnostic only",
        "acceptance": "independent classification, NUM, same UUID, rotated-round uncertainty pending"}


def validate_run(run, prompt, expected):
    if (run["prompt_tokens"] != prompt or len(run["token_ids"]) != 64 or
            run["consumed_token_ids"] != prompt + run["token_ids"][:-1] or len(run["decode_step_ms"]) != 63):
        raise RuntimeError("Full64/own-history/63 interval contract failed")
    mismatch = next((i for i, (a, b) in enumerate(zip(run["token_ids"], expected)) if a != b), None)
    run["NUM"] = {"first10_decode_indices": [1, 11], "first10_exact": run["token_ids"][1:11] == expected[1:11],
                  "first_output_exact": run["token_ids"][0] == expected[0], "all64_exact": mismatch is None,
                  "first_mismatch": mismatch, "reference_used_inside_timing": False}


def aggregate(scenario):
    host = [v for run in scenario["runs"] for v in run["decode_step_ms"]]
    scenario.update(host=summary(host), total_decode_tokens=len(host), total_host_ms=sum(host),
                    host_tokens_per_second=len(host) * 1000 / sum(host))
    device = [v for run in scenario["runs"] for v in run.get("device_decode_step_ms", [])]
    if device:
        if len(device) != len(host):
            raise RuntimeError("Incomplete device coverage")
        scenario.update(device=summary(device), total_device_ms=sum(device),
                        device_tokens_per_second=len(device) * 1000 / sum(device))


def bind_gpu(torch, uuid, identity=None):
    if identity is None:
        # UUID-native existing adapters keep their public call; the new monitored
        # launcher supplies their independently queried NVML proof via the env.
        args = argparse.Namespace(physical_gpu_uuid=uuid, gpu_selector_kind="uuid",
            cuda_visible_device=uuid, device_identity_proof=None)
        proof_path = os.environ.get("FLAGMEGA_DEVICE_IDENTITY_PROOF")
        if not proof_path: raise ValueError("UUID-native runtime requires launcher identity proof")
        backend_name = json.loads(Path(proof_path).read_text())["backend"]
        if backend_name == "vllm": raise ValueError("vLLM numeric binding needs explicit prepared identity")
        identity = prepare_identity(args, backend_name)
    validate_visibility(identity)
    if identity["physical_gpu_uuid"] != uuid:
        raise ValueError("Identity receipt differs from requested leased GPU")
    torch.cuda.set_device(0)
    return verify_torch_identity(torch, identity)


class StepTimer:
    """Shared per-step policy. GPU event queries and raw bookkeeping are untimed."""
    def __init__(self, torch, cohort, synchronize=None):
        self.torch, self.cohort = torch, cohort
        self.synchronize = synchronize or torch.cuda.synchronize
        self.start = self.end = None
        if cohort == "event_instrumented":
            self.start, self.end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            self.start.record(torch.cuda.current_stream())
            self.end.record(torch.cuda.current_stream())
            self.synchronize()

    def measure(self, operation, deliver=lambda value: value, probe=None):
        started = time.perf_counter_ns()
        if probe:
            probe.begin()
        elif self.start is not None:
            self.start.record(self.torch.cuda.current_stream())
        try:
            value = operation()
            if self.end is not None and probe is None:
                self.end.record(self.torch.cuda.current_stream())
            self.synchronize()
            value = deliver(value)
            ended = time.perf_counter_ns()
            if probe:
                probe.finish()
            raw = {"step_start_ns": started, "step_end_ns": ended,
                   "host_ms": (ended-started)/1e6}
            if self.start is not None:
                raw["device_ms"] = self.start.elapsed_time(self.end)
            return value, raw
        finally:
            if probe:
                probe.active = False


def step_arrays(steps):
    result = {"step_start_ns": [s["step_start_ns"] for s in steps],
              "step_end_ns": [s["step_end_ns"] for s in steps],
              "decode_step_ms": [s["host_ms"] for s in steps]}
    if steps and "device_ms" in steps[0]:
        result["device_decode_step_ms"] = [s["device_ms"] for s in steps]
    return result


def plan(work, checkpoint, uuid=None, nvml_index=None):
    benchmark = work / "benchmark/backends"
    checkpoint = Path(checkpoint).resolve()
    engines = ["flagmega", "vllm", "sglang", "tilefoundry", "llama.cpp"]
    interpreters = {"flagmega": "/root/miniconda3/envs/flagtree/bin/python",
        "vllm": "/root/miniconda3/envs/flagtree/bin/python",
        "sglang": "/root/miniconda3/envs/sglang_bench/bin/python",
        "tilefoundry": "/root/miniconda3/envs/tilefoundry-qwen/bin/python",
        "llama.cpp": "/root/miniconda3/envs/flagtree/bin/python"}
    jobs, gaps = [], ["FlashRT native H800/1.7B BF16 full decode binding unavailable",
        "llama.cpp DEVICE needs separately approved actual native CUDA-stream/event driver binding"]
    for round_index in range(3):
        order = engines[2*round_index:] + engines[:2*round_index]
        contexts = LENGTHS[round_index:] + LENGTHS[:round_index]
        for cohort_index, cohort in enumerate(COHORTS if round_index % 2 == 0 else list(reversed(COHORTS))):
            for position, backend in enumerate(order):
                if backend == "llama.cpp" and cohort == "event_instrumented":
                    continue
                mid = f"native-round{round_index}-{cohort}-{backend.replace('.', '')}"
                script = "reference_rounds.py" if backend == "flagmega" else {"vllm":"vllm_native.py", "sglang":"sglang_native.py", "tilefoundry":"tilefoundry_native.py", "llama.cpp":"llama_native.py"}[backend]
                argv = [interpreters[backend], str(benchmark/script)]
                if backend == "flagmega":
                    argv += ["--backend", "flagmega", "--artifact", str(work/".local/baseline/build/b001/runtime-artifact")]
                argv += ["--checkpoint", str(checkpoint), "--output", str(work/".local/cache/comparison/native-rounds"/(mid+".json")), "--measurement-id", mid,
                    "--reference", str(work/".local/cache/comparison/vllm-baseline-round001.json"), "--cohort", cohort,
                    "--prompt-lengths", *map(str, contexts), "--round-index", str(round_index), "--repeats", "3", "--warmups", "2", "--decode-tokens", "64"]
                if uuid:
                    argv += ["--physical-gpu-uuid", uuid]
                if backend == "tilefoundry":
                    argv += ["--inventory", str(work/".local/agents/dispatch/analyze_tile_native/inventory.json")]
                if backend == "llama.cpp":
                    argv += ["--build-receipt", str(work/".local/cache/comparison/llama-reference-build001.json"), "--driver", str(work/".local/cache/comparison/llama-native-driver001"), "--driver-binding", str(work/".local/cache/comparison/llama-native-driver001-binding.json")]
                kind = "nvml-index" if backend == "vllm" else "uuid"
                selector = str(nvml_index) if kind == "nvml-index" and nvml_index is not None else (uuid if kind == "uuid" else None)
                policy = selector_policy(backend, uuid, kind, selector)
                identity_path = work/".local/jobs"/mid/"device-identity.json"
                argv += ["--gpu-selector-kind", kind, "--device-identity-proof", str(identity_path)]
                if selector is not None: argv += ["--cuda-visible-device", selector]
                environment = {"CUDA_VISIBLE_DEVICES": selector, "CUDA_DEVICE_ORDER": "PCI_BUS_ID", "OMP_NUM_THREADS": "1"}
                if backend in ("flagmega", "vllm"):
                    environment["PYTHONPATH"] = str(next(p for p in work.parents if (p/"python/triton").is_dir())/"python")
                if backend == "sglang":
                    base = Path(interpreters[backend]).parents[1]/"lib/python3.10/site-packages"
                    environment["LD_LIBRARY_PATH"] = ":".join(str(base/p) for p in ("nvidia/cu13/lib", "torch/lib", "tvm_ffi/lib"))
                jobs.append({"round": round_index, "cohort_order": cohort_index, "backend_position": position,
                    "backend": backend, "cohort": cohort, "argv": argv, "environment": environment,
                    "selector_policy": policy, "GPU_access": False,
                    "runnable_after_UUID_binding": False, "requires": ["reviewer implemented contract/classification", "root dashboard idle UUID lease", "fresh explicit selector and actual NVML/CUDA/native UUID+PCI proof", "0.25s NVML pre/in/post isolation with owned process identity", "native build and NUM admission"],
                    "allowed_now": "CPU preflight only; append --preflight-only and CUDA_VISIBLE_DEVICES=''"})
    return {"schema": "flagmega.native-reference-round-plan/v1", "generated_utc": datetime.now(timezone.utc).isoformat(),
        "physical_gpu_uuid": uuid, "declared_nvml_index": nvml_index, "original_contexts": LENGTHS, "output_tokens": 64, "decode_intervals": 63,
        "warmups": 2, "repeats_per_scenario_per_round": 3, "rounds": 3, "jobs": jobs, "gaps": gaps,
        "execution": "sequential separate native backend processes; no GPU command run by planner",
        "uncertainty": "paired per-round TPS ratios; all3 ratios>1 and 95% Student-t log-ratio lower bound>0 (df2,t4.302653), plus strict independent gates. No IID treatment of token samples.",
        "cohort_matching": "HOST and DEVICE analyzed separately; DEVICE worker/input+fullmodel+sampler+D2H event boundary includes in-window launch idle; no aggregate-vs-per-step comparison"}


def flagmega(args):
    check_arguments(args)
    checkpoint, inputs, reference = reference_prompts(args)
    manifest = args.artifact/"artifact.json"
    model = json.loads(manifest.read_text())
    source = args.artifact/model["codegen"]["source"]
    report = {"schema":"flagmega.model-measurement/v1", "backend":"flagmega", "status":"preflight", "measurement_id":args.measurement_id,
        "cohort":args.cohort, "round_index":args.round_index, "checkpoint":checkpoint, "reference":reference,
        "script":{"path":str(Path(__file__).resolve()),"sha256":sha(__file__)}, "controls":controls(args),
        "artifact":{"path":str(args.artifact.resolve()),"manifest_sha256":sha(manifest)},
        "runtime":{"artifact_source_sha256":sha(source)}, "workload":{"batch":1,"dtype":"bfloat16","layers":28,"prompt_lengths":args.prompt_lengths,"decode_tokens":64},
        "boundary":{"host":"append/fullmodel/nativegreedy/D2H/sync/deliver; exactly63 intervals", "device":"input fill+all28+LMhead+greedy+state+pinned D2H; event end before sync/host delivery"}, "scenarios":[]}
    if args.preflight_only:
        write_new(args.output,report);return 0
    backend=None
    try:
        identity = prepare_identity(args, "flagmega")
        import torch
        from triton.flagmega.serving.backend import ArtifactBackend
        from triton.flagmega.serving.sampling import SamplingConfig
        bind_gpu(torch,args.physical_gpu_uuid,identity=identity)
        report.update(device_identity=identity,gpu_before=gpu_inventory(),visible_devices=os.environ["CUDA_VISIBLE_DEVICES"])
        backend=ArtifactBackend.load(args.artifact,device="cuda:0",sampling=SamplingConfig(temperature=0,seed=0),cuda_graph=True,warmups=3)
        if not backend._builtin_token or backend.vocab_size!=151936 or backend.model.state_config.num_layers!=28:
            raise RuntimeError("Full model/native greedy artifact required")
        timer=StepTimer(torch,args.cohort,synchronize=backend.synchronize)
        def request(prompt):
            backend.reset();backend.synchronize()
            for token in prompt: backend.append(token)
            generated=[backend.sample()];steps=[]
            for _ in range(63):
                def operation():
                    backend.append(generated[-1])
                    backend._cpu_token.copy_(backend.token,non_blocking=True)
                    return backend._cpu_token
                token,sample=timer.measure(operation,lambda tensor:int(tensor.item()))
                generated.append(token);steps.append(sample)
            if backend.state.sequence_length!=len(prompt)+63: raise RuntimeError("Actual FlagMega KV length differs")
            return {"prompt_tokens":prompt,"token_ids":generated,"consumed_token_ids":prompt+generated[:-1],"final_state_length":backend.state.sequence_length,**step_arrays(steps)}
        for length in args.prompt_lengths:
            prompt=inputs[length]["prompt"]
            for _ in range(2): request(prompt)
            scenario={"prompt_length":length,"runs":[]};report["scenarios"].append(scenario)
            for repeat in range(3):
                run=request(prompt);run["repeat_index"]=repeat;validate_run(run,prompt,inputs[length]["expected"]);scenario["runs"].append(run)
            aggregate(scenario)
        if sha(manifest)!=report["artifact"]["manifest_sha256"] or sha(source)!=report["runtime"]["artifact_source_sha256"]:
            raise RuntimeError("FlagMega artifact/source changed during measurement")
        report.update(status="measured",NUM_passed=all(r["NUM"]["first10_exact"] for s in report["scenarios"] for r in s["runs"]),
            all64_exact=all(r["NUM"]["all64_exact"] for s in report["scenarios"] for r in s["runs"]),
            gpu_after=gpu_inventory(),runtime={**backend.info(),**report["runtime"]},acceptance="pending independent review; no gain claim")
        write_new(args.output,report);return 0
    except BaseException as error:
        report.update(status="failed",error=str(error),traceback=traceback.format_exc());write_new(args.output.with_suffix(".failed.json"),report);raise
    finally:
        if backend: backend.close()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan-only",action="store_true")
    parser.add_argument("--backend",choices=["flagmega"])
    parser.add_argument("--artifact",type=Path)
    parser.add_argument("--work-dir",type=Path,default=WORK)
    parser.add_argument("--declared-nvml-index",type=int)
    common_arguments(parser)
    args=parser.parse_args()
    if args.plan_only:
        write_new(args.output,plan(args.work_dir,args.checkpoint,args.physical_gpu_uuid,args.declared_nvml_index));return 0
    if args.backend!="flagmega" or not args.artifact: parser.error("--backend flagmega --artifact required")
    return flagmega(args)


if __name__=="__main__": raise SystemExit(main())
