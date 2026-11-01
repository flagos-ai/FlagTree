from pathlib import Path as _TrialPath
import sys as _trial_sys
_trial_sys.path.insert(0,str(_TrialPath(__file__).resolve().parent))
from local_optimizations.kernel_recipe import activate as _trial_activate
_trial_activate()

"""Full-model CUDA Graph replay, independent official prefixes, state carry included."""

import argparse
from types import MethodType
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import traceback

root = Path(__file__).resolve().parent
sys.path.insert(0,str(root))

import torch
import torch.distributed as dist
from local_optimizations.reservation_client import load_reserved as load
from local_optimizations.run_tp4 import reset_request, check_state
from local_optimizations.scenarios import (OUTPUT_TOKENS, OFFICIAL_ONE_TOKEN,
                                         reference_scenarios, validate_lengths)
from triton.flagmega.runtime.prepared import _validate_resources


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--requests", type=int, default=12)
    parser.add_argument("--trial", type=Path, required=True)
    parser.add_argument("--prompt-lengths", type=int, nargs="+", default=[1])
    parser.add_argument("--native-reference", type=Path)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    validate_lengths(args.prompt_lengths)
    if args.validate_only:
        args.requests = 2
    elif args.requests != 12:
        parser.error("Formal measurements require two warmups and ten requests")
    if args.native_reference:
        reference_data = args.native_reference.read_bytes()
        scenarios = reference_scenarios(json.loads(reference_data), args.prompt_lengths)
    elif args.prompt_lengths == [1]:
        reference_data = None
        scenarios = {1: dict(prompt=[11], requests=[dict(tokens=OFFICIAL_ONE_TOKEN)])}
    else:
        parser.error("Multiple prompt lengths require an independent native reference")
    trial = args.trial.resolve()
    warmup = json.loads((trial / "warmup.json").read_text())
    rank = int(os.environ["RANK"])
    device = f"cuda:{os.environ['LOCAL_RANK']}"
    torch.cuda.set_device(device)
    report = {"rank": rank, "passed": False, "diagnostic_only": True,
              "performance_acceptance": False, "shared_phase_programs": True, "requests": [],
              "artifact": str(trial / "artifact"), "warmups_per_leg": 2, "measurements_per_leg": 10,
              "scope": "Full45/E288/top8/TP4/cache4096; graph-on full runtime including compiled token-scan prefill, state carry, greedy sample and token handoff.",
              "scenarios": [], "validation_only": args.validate_only,
              "prefill_mode": "compiled_token_scan",
              "native_reference_sha256": hashlib.sha256(reference_data).hexdigest() if reference_data else None}
    runtime, host = None, None
    graphs = {}
    begin = time.perf_counter()
    try:
        print(json.dumps({"event": "load_begin", "rank": rank}), flush=True)
        runtime = load(trial / "artifact", device=device)
        host = dist.new_group(backend="gloo")
        assert runtime.ir_module.metadata["num_hidden_layers"] == 45
        assert runtime.ir_module.metadata["num_experts"] == 288
        assert runtime._paged_config.max_sequence_length == 4096
        assert runtime.codegen["source_sha256"] == warmup["source_sha256"]
        report["source_sha256"] = runtime.codegen["source_sha256"]
        assert runtime.ir_module.metadata["coarse_native_row_projection"]["accumulation_splits"] == 2
        state = runtime.create_state()
        ids = torch.full((1,), 11, dtype=torch.int32, device=device)
        logits, token = runtime.create_outputs()
        runtime.prepare(ids, state, logits=logits, next_token=token, prefill=True)
        expected_entries = {entry["symbol"]: entry["hash"] for entry in warmup["entries"]}
        assert len(runtime._prepared) == 228 and warmup["joint_resources_ok"]
        for spec, prepared in zip(runtime.codegen["kernels"], runtime._prepared, strict=True):
            assert prepared.compiled_kernel.hash == expected_entries[spec["entry_symbol"]], spec["symbol"]
        assert len({p.compiled_kernel.hash for p in runtime._prepared}) == len(expected_entries)
        assert not runtime._dynamic_descriptor_indices
        report["resources"] = runtime.resource_report
        assert all(r["spill_bytes"] == 0 for r in report["resources"])
        report["compiled_entries"] = expected_entries
        report["cubin_sha256"] = {s["entry_symbol"]: hashlib.sha256(p.compiled_kernel.asm["cubin"]).hexdigest()
            for s, p in zip(runtime.codegen["kernels"], runtime._prepared, strict=True)}
        report["joint_resources"] = runtime._program_dag.resource_report
        materializations = {}
        active_label = "outside_requests"
        original_materialize = runtime._tensor_descriptors.materialize_many

        def measured_materialize(kernel_name, specs, sources):
            counts = materializations.setdefault(active_label, {"calls": 0, "descriptors": 0})
            counts["calls"] += 1
            counts["descriptors"] += len(specs)
            return original_materialize(kernel_name, specs, sources)

        runtime._tensor_descriptors.materialize_many = measured_materialize
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        graphs = {}
        for phase in (True, False):
            dist.barrier(group=host)
            captured = torch.cuda.CUDAGraph()
            with torch.cuda.graph(captured, stream=stream):
                runtime.run_into(logits, token, ids, state, prefill=phase)
            graphs[phase] = captured
        torch.cuda.current_stream().wait_stream(stream)
        report['cuda_graph'] = True
        report['output_tokens'] = 8
        collected = torch.empty(8, dtype=torch.int32, device=device)
        starts = [torch.cuda.Event(enable_timing=True) for _ in range(8)]
        ends = [torch.cuda.Event(enable_timing=True) for _ in range(8)]
        torch.cuda.synchronize()
        report["initialization_seconds"] = time.perf_counter() - begin
        print(json.dumps({"event": "prepared", "rank": rank, "resources": report["resources"]}), flush=True)
        for leg, length in enumerate(args.prompt_lengths):
            active_label = f"prompt{length}"
            prompt = scenarios[length]['prompt']
            expected = scenarios[length]['requests'][0]['tokens']
            prompt_device = torch.tensor(prompt, dtype=torch.int32, device=device)
            prompt_views = [prompt_device[i:i+1] for i in range(length)]
            scenario = dict(prompt_length=length, prompt=prompt, output_tokens=8,
                            requests=[], validation_requests=[])
            report['scenarios'].append(scenario)
            for request in range(args.requests):
                dist.barrier(group=host)
                torch.cuda.synchronize()
                start_utc, wall = time.time(), time.perf_counter()
                reset_request(state)
                # Every input prompt token uses the explicit chunk arithmetic.
                # Its sampled output is discarded until the final prompt token.
                for prompt_input in prompt_views[:-1]:
                    ids.copy_(prompt_input)
                    graphs[True].replay()
                ids.copy_(prompt_views[-1])
                host_launch_ms = []
                for step in range(8):
                    starts[step].record()
                    host_begin = time.perf_counter()
                    graphs[step == 0].replay()
                    host_launch_ms.append((time.perf_counter() - host_begin) * 1000)
                    ends[step].record()
                    collected[step].copy_(token[0])
                    ids.copy_(token)
                    if step == 0:
                        torch.cuda.synchronize()
                        first = time.perf_counter()
                tokens = collected.cpu().tolist()
                finish = time.perf_counter()
                row = {"leg": leg, "label": f"prompt{length}", "prompt_length": length,
                       "request": request, "warmup": request < 2,
                       "wall_start_utc": start_utc, "wall_end_utc": time.time(), "tokens": tokens,
                       "wall_request_ms": (finish - wall) * 1000, "ttft_ms": (first - wall) * 1000,
                       "decode_tpot_ms": (finish - first) * 1000 / 7,
                       "device_step_ms_with_state_carry": [a.elapsed_time(b) for a, b in zip(starts, ends)],
                       "host_launch_ms": host_launch_ms}
                scenario["requests"].append(row)
                if length == 1 and len(args.prompt_lengths) == 1:
                    report["prompt"] = prompt
                    report["requests"].append(row)
                print(json.dumps({"event": "observed_prefix", "rank": rank, "tokens": tokens,
                                  "finite": bool(torch.isfinite(logits).all())}), flush=True)
                assert tokens == expected, (rank, length, tokens, expected)
                assert torch.isfinite(logits).all() and int(state.paged.seq_lens.item()) == length + 7
                if request in (1, args.requests - 1):
                    check_state(state, None, length + 6)
                    scenario['validation_requests'].append(dict(request=request,
                        all_active_state_finite=True, cache_tail_untouched=True,
                        sequence_length=length+7, independent_native_tokens_equal=True))
                if rank == 0:
                    print(json.dumps({"event": "request", **row}), flush=True)
        report["descriptor_materializations"] = materializations
        assert all(materializations.get(f"prompt{length}", {"calls": 0})["calls"] == 0
                   for length in args.prompt_lengths)
        report["passed"] = True
    except BaseException:
        report["traceback"] = traceback.format_exc()
        raise
    finally:
        report["seconds"] = time.perf_counter() - begin
        (args.output / f"rank-{rank}.json").write_text(json.dumps(report, indent=2) + "\n")
        graphs.clear()
        if host is not None:
            dist.destroy_process_group(host)
        if runtime is not None:
            runtime.close()


if __name__ == "__main__":
    main()
