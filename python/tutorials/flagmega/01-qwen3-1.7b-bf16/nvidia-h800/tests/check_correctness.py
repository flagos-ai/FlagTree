#!/usr/bin/env python3
"""Independent full Qwen3 BF16 generation, state and native-admission checks.

All 28 layers, full vocabulary and all five frozen scenarios execute on CUDA.
The candidate consumes its own generated outputs. Reference tokens are used
only to compare results and record mismatch diagnostics, never as decode input.
These instrumented NUM runs are not performance samples.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
import traceback


SCENARIOS = (32, 128, 250, 1024, 2048)
CHECKPOINT_REVISION = "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_new(path, report):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def reference_inputs(path, expected_sha):
    require(sha(path) == expected_sha, "Native reference hash changed")
    reference = json.loads(path.read_text())
    require(reference.get("backend") == "vllm" and reference.get("status") == "measured",
            "Need real native-vLLM trajectories, not a preflight or failed report")
    require("g493bd8323" in reference.get("version", ""), "Pinned numerical reference revision differs")
    workload = reference["workload"]
    require(workload["batch"] == workload["concurrency"] == 1 and workload["layers"] == 28,
            "Reference must execute the full single-sequence model")
    require(workload["dtype"] == "bfloat16" and workload["sampling"] == "greedy",
            "Reference arithmetic/sampling differs")
    require(workload["decode_tokens"] == 64 and tuple(workload["prompt_lengths"]) == SCENARIOS,
            "Require all five original scenarios and complete 64-output trajectories")
    require(not workload["prefix_caching"] and not workload["speculative_decoding"],
            "Reference prefix/speculative cache contract differs")
    require(reference["checkpoint"]["revision"] == CHECKPOINT_REVISION, "Checkpoint revision differs")
    cases = {}
    for scenario in reference["scenarios"]:
        length = scenario["prompt_length"]
        runs = scenario["runs"]
        require(runs, f"Missing actual reference trajectory for {length}")
        first = runs[0]
        require(len(first["prompt_tokens"]) == length and len(first["token_ids"]) == 64,
                "Reference trajectory is incomplete")
        for run in runs:
            require(run["prompt_tokens"] == first["prompt_tokens"] and run["token_ids"] == first["token_ids"],
                    "Pinned native greedy reference is not deterministic across requests")
        require(length not in cases, "Duplicate reference scenario")
        cases[length] = first
    require(set(cases) == set(SCENARIOS), "Reference scenario coverage incomplete")
    return reference, cases


def inventory():
    return subprocess.check_output([
        "nvidia-smi", "--query-gpu=index,uuid,name,driver_version,memory.total,pstate,"
        "clocks.sm,clocks.mem,utilization.gpu,memory.used", "--format=csv,noheader,nounits",
    ], text=True).strip()


def checking_backend(artifact, device):
    import torch
    from triton.flagmega.serving.backend import ArtifactBackend
    from triton.flagmega.serving.sampling import SamplingConfig

    class CheckingBackend(ArtifactBackend):
        def reset(self):
            super().reset()
            self.consumed = []
            self.logit_checks = 0
            self.sample_index = 0
            self.first_difference = None
            self.expected_tokens = None
            self.generated = []
            self.last_input_attempt = None

        def append(self, token):
            previous = self.length
            self.last_input_attempt = token
            super().append(token)
            self.synchronize()
            require(self.state.sequence_length == previous + 1 == self.length,
                    f"State must advance exactly once per input token at {previous}")
            self.consumed.append(token)

        def sample(self):
            # Normal greedy sampling from this candidate's actual logits/token.
            token = super().sample()
            require(bool(torch.isfinite(self.logits).all().item()), "Nonfinite full-vocabulary logits")
            self.logit_checks += 1
            index = self.sample_index
            if (self.expected_tokens is not None and self.first_difference is None
                    and token != self.expected_tokens[index]):
                values, indices = torch.topk(self.logits[0].float(), 16)
                expected = self.expected_tokens[index]
                self.first_difference = {
                    "output_index": index, "decode_index": index - 1 if index else None,
                    "prefill_output": index == 0, "actual_token": token, "reference_token": expected,
                    "candidate_own_history": list(self.consumed),
                    "top16_token_ids": indices.cpu().tolist(), "top16_logits": values.cpu().tolist(),
                    "greedy_margin": float((values[0] - values[1]).item()),
                    "reference_token_logit_on_candidate_history": float(self.logits[0, expected].item()),
                    "scope": "Read-only mismatch evidence, no reference forward or teacher-forced input",
                }
            self.sample_index += 1
            self.generated.append(token)
            return token

    return CheckingBackend.load(artifact, device=device, cuda_graph=True,
                                sampling=SamplingConfig(temperature=0, seed=0), warmups=1)


def native_admission(backend):
    result = {"runtime": backend.info(), "kernels": []}
    prepared = backend.model._prepared
    require(prepared is not None, "Actual kernel specialization/admission did not run")
    for item in prepared if isinstance(prepared, tuple) else (prepared,):
        compiled = item.compiled_kernel
        metadata = compiled.metadata
        fields = vars(metadata) if hasattr(metadata, "__dict__") else metadata._asdict()
        scalar_fields = {str(k): v for k, v in fields.items()
                         if v is None or isinstance(v, (str, int, float, bool))}
        assembly = {}
        for kind in ("ptx", "ttir", "ttgir"):
            source = compiled.asm.get(kind)
            if isinstance(source, str):
                assembly[kind] = {"sha256": hashlib.sha256(source.encode()).hexdigest(),
                                  "bytes": len(source.encode())}
                if kind == "ptx":
                    assembly[kind]["BF16_conversion_mentions"] = source.count("bf16")
                    assembly[kind]["scope"] = "Actual native source identity/count; not a per-boundary correctness proof"
        result["kernels"].append({"name": compiled.name, "metadata": scalar_fields,
                                  "resources": item.resource_report, "assembly": assembly})
    return result


def state_evidence(backend, expected_length):
    import torch
    backend.synchronize()
    state, config = backend.state, backend.state.config
    state.validate()
    require(state.sequence_length == backend.length == expected_length,
            "Physical KV and actual consumed history length differ")
    require(bool(torch.isfinite(state.kv_caches).all().item()), "Nonfinite KV state")
    require(state.query_start_loc.cpu().tolist() == [0, 1], "One-token query-start ABI differs")
    require(state.slot_mapping.item() == expected_length - 1, "Cache write-position ABI differs")
    table = state.block_table[0].cpu().tolist()
    probes = []
    for logical_page in range((expected_length + config.block_size - 1) // config.block_size):
        page_start = logical_page * config.block_size
        logical_positions = (page_start, min(expected_length - 1, page_start + config.block_size - 1))
        for position in sorted(set(logical_positions)):
            physical_page = table[logical_page]
            slot = state.kv_caches[physical_page, :, :, position % config.block_size]
            counts = slot.reshape(28, 2, -1).count_nonzero(dim=-1)
            require(bool((counts > 0).all().item()), f"A layer K/V slot was not written at position {position}")
            probes.append({"logical_position": position, "physical_page": physical_page,
                           "per_layer_KV_nonzero_counts": counts.cpu().tolist()})
    cache_pages = state.kv_caches.view(config.num_blocks, config.num_layers, 2, config.block_size, -1)
    for logical_page, physical_page in enumerate(table):
        first_unwritten = max(0, min(config.block_size, expected_length - logical_page * config.block_size))
        tail = cache_pages[physical_page, :, :, first_unwritten:]
        require(tail.count_nonzero().item() == 0, "Cache bytes beyond consumed history were written")
    return {"length": expected_length, "slot_mapping": int(state.slot_mapping.item()),
            "query_start_loc": state.query_start_loc.cpu().tolist(), "block_table": table,
            "nonzero_all28layers_KV_probes": probes, "finite_KV": True, "unwritten_tail_zero": True,
            "cache_dtype": str(state.kv_caches.dtype), "cache_shape": list(state.kv_caches.shape)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--artifact-manifest-sha256", required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--reference-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--preflight-only", action="store_true",
                        help="Actually compile/warm/admit the complete kernel; no NUM qualification")
    args = parser.parse_args()
    require(not args.output.exists() and not args.output.with_suffix(".failed.json").exists(),
            "Use fresh immutable result paths")
    report = {"schema": "flagmega.independent-full-NUM/v1", "agent_name": "reviewer",
              "agent_id": "/root/reviewer", "started_utc": datetime.now(timezone.utc).isoformat(),
              "script": {"path": str(Path(__file__).resolve()), "sha256": sha(__file__)},
              "visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"), "device": args.device,
              "mode": "native_admission_only" if args.preflight_only else "own_history_full_NUM",
              "acceptance_boundary": "First10 independently decoded outputs indices1:11; prefill output excluded",
              "full_trajectory": "All64 outputs and comparisons retained; later differences reported separately",
              "instrumentation": "State/finiteness/read-only mismatch checks; not performance measurements",
              "scenarios": [], "status": "pending", "NUM_qualified": False}
    backend = None
    try:
        manifest = args.artifact / "artifact.json"
        require(sha(manifest) == args.artifact_manifest_sha256, "Artifact manifest hash changed")
        artifact_data = json.loads(manifest.read_text())
        require(artifact_data["codegen_backend"] == "triton" and artifact_data["codegen"]["use_tle"],
                "Require Triton+TLE artifact")
        reference, cases = reference_inputs(args.reference, args.reference_sha256)
        report["reference"] = {"path": str(args.reference.resolve()), "sha256": args.reference_sha256,
                               "version": reference["version"], "checkpoint": reference["checkpoint"]}
        report["artifact"] = {"path": str(args.artifact.resolve()), "manifest_sha256": sha(manifest),
                              "semantic_hash": artifact_data["semantic_hash"]}
        import torch
        from triton.flagmega.serving.engine import TextGenerationEngine
        torch.set_num_threads(1)
        require(torch.cuda.is_available(), "Actual CUDA execution required")
        report["gpu_before"] = inventory()
        print("Preparing actual full28layer kernel and CUDA graph", flush=True)
        began = time.monotonic()
        backend = checking_backend(args.artifact, args.device)
        require(backend.model.state_config.num_layers == 28 and backend.vocab_size == 151936,
                "Require all28 layers and the full151936-vocabulary LMhead")
        require(backend.capacity >= max(SCENARIOS) + 64, "Artifact cannot cover original scenarios")
        report["native_admission"] = native_admission(backend)
        report["actual_prepare_seconds"] = time.monotonic() - began
        print("Actual kernel prepared/admitted", flush=True)
        if args.preflight_only:
            report["status"] = "native_admitted_NUM_not_run"
        else:
            engine = TextGenerationEngine(backend)
            for length in SCENARIOS:
                expected = cases[length]["token_ids"]
                prompt = cases[length]["prompt_tokens"]
                engine.reset()
                backend.expected_tokens = list(expected)
                print(f"Independent own-history generation prompt{length} output64", flush=True)
                result = engine.generate(prompt, max_new_tokens=64, ignore_eos=True)
                actual = list(result.token_ids)
                consumed = list(prompt) + actual[:-1]
                require(list(engine.cached_tokens) == consumed == backend.consumed,
                        "Candidate did not feed its own outputs into actual consumed history")
                require(len(actual) == 64 and backend.logit_checks == 64, "Incomplete generated/logit-checked trajectory")
                first10 = actual[1:11] == expected[1:11]
                scenario = {"prompt_length": length, "prompt_tokens": prompt, "token_ids": actual,
                            "reference_token_ids": expected, "first10_decode_exact": first10,
                            "full64_exact": actual == expected, "first_difference": backend.first_difference,
                            "final_consumed_history": consumed,
                            "all64_logits_finite": True, "decode_page_crossings": [
                                n for n in range(length, len(consumed)) if n % backend.state.config.block_size == 0]}
                report["scenarios"].append(scenario)
                scenario["state"] = state_evidence(backend, len(consumed))
                print(json.dumps({k: scenario[k] for k in ["prompt_length", "first10_decode_exact", "full64_exact"]}), flush=True)
            report["NUM_qualified"] = all(s["first10_decode_exact"] for s in report["scenarios"])
            report["status"] = "passed" if report["NUM_qualified"] else "NUM_failed"
        report["gpu_after"] = inventory()
        report["ended_utc"] = datetime.now(timezone.utc).isoformat()
        write_new(args.output, report)
        return 0 if args.preflight_only or report["NUM_qualified"] else 1
    except BaseException as error:
        report.update(status="execution_failed", error=str(error), traceback=traceback.format_exc(),
                      ended_utc=datetime.now(timezone.utc).isoformat())
        if backend is not None:
            report["partial_own_history"] = list(backend.consumed)
            report["partial_generated_tokens"] = list(backend.generated)
            report["last_input_attempt"] = backend.last_input_attempt
            report["first_difference"] = backend.first_difference
        write_new(args.output.with_suffix(".failed.json"), report)
        raise
    finally:
        if backend is not None:
            backend.close()


if __name__ == "__main__":
    raise SystemExit(main())
