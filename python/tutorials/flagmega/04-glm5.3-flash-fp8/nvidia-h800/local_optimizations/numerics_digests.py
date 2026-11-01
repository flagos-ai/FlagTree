from pathlib import Path as _TrialPath
import sys as _trial_sys
_trial_sys.path.insert(0,str(_TrialPath(__file__).resolve().parent.parent))
from local_optimizations.kernel_recipe import activate as _trial_activate
_trial_activate()

"""Full-model state diagnostics, separate from all performance measurements."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import subprocess
import time
import traceback

root = Path(__file__).resolve().parent
sys.path.insert(0,str(root))

import torch
import torch.distributed as dist
from local_optimizations.reservation_client import load_reserved
from local_optimizations.run_tp4 import check_state, reset_request
from local_optimizations.owned_state_diagnostics_v1 import collect_kda


def other_state(state, logits):
    paged = state.paged
    length = int(paged.seq_lens.item())
    values = {"logits": logits.cpu(), "paged.seq_lens": paged.seq_lens.cpu(),
              "paged.block_table": paged.block_table.cpu(), "paged.slot_mapping": paged.slot_mapping.cpu(),
              "paged.query_start_loc": paged.query_start_loc.cpu(),
              "paged.active_kv": torch.stack([paged.kv_caches[
                  int(paged.block_table[0, pos // paged.config.block_size]), :, :, pos % paged.config.block_size]
                  for pos in range(length)]).cpu()}
    if state.dsa is not None:
        values["dsa.packed"] = state.dsa.packed_states[:, :length].cpu()
        values["dsa.seq_lens"] = state.dsa.seq_lens.cpu()
    return values


def exact(got, want, label):
    assert list(got.shape) == want["shape"] and str(got.dtype) == want["dtype"], label
    data = got.contiguous().view(torch.uint8).numpy()
    assert data.nbytes == want["nbytes"], label
    digest = hashlib.sha256(data).hexdigest()
    assert digest == want["sha256"], f"{label} byte digest differs: {digest}"


def assert_exclusive(output, *, group):
    launch = json.loads((output / "launch.json").read_text())
    rows = subprocess.check_output(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader"], text=True)
    owners = {}
    for row in rows.splitlines():
        uuid, pid = row.split(",")
        owners.setdefault(uuid.strip(), set()).add(int(pid))
    acceptable = all(launch["keeper"] in owners.get(card["uuid"], set())
                     and len(owners.get(card["uuid"], set())) == 2 for card in launch["cards"])
    verdict = torch.tensor(int(acceptable), dtype=torch.int32)
    dist.all_reduce(verdict, op=dist.ReduceOp.MIN, group=group)
    assert bool(verdict), "foreign CUDA context detected; no megakernel submitted"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trial", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--mode", choices=("check-owned",), required=True)
    parser.add_argument("--requests", type=int, default=2)
    args = parser.parse_args()
    rank = int(os.environ["RANK"])
    device = f"cuda:{os.environ['LOCAL_RANK']}"
    torch.cuda.set_device(device)
    torch.empty(1, device=device)
    runtime, host, graphs = None, None, {}
    report = {"rank": rank, "mode": args.mode, "passed": False, "requests": [],
              "diagnostic_only": True, "performance_acceptance": False,
              "reference_sha256": hashlib.sha256(args.reference.read_bytes()).hexdigest(),
              "scope": "Full45 official8 independent Graph requests; every KDA head/element and non-KDA state compared outside timing."}
    started = time.perf_counter()
    guards = []
    try:
        runtime = load_reserved(args.trial / "artifact", device=device)
        host = dist.new_group(backend="gloo")
        assert runtime.ir_module.metadata["num_hidden_layers"] == 45
        assert runtime.ir_module.metadata["num_experts"] == 288
        assert runtime._paged_config.max_sequence_length == 4096
        state = runtime.create_state()
        if args.mode == "check-owned":
            assert state.kda.config.logical_types is not None and state.kda.config.chip_index == rank
            for collection in (state.kda, state.next_kda):
                for field in ("convolution", "recurrent"):
                    old = getattr(collection, field)
                    count = 256 // old.element_size()
                    whole = torch.full((old.numel() + 2 * count,), -123., dtype=old.dtype, device=device)
                    setattr(collection, field, whole[count:-count].reshape(old.shape))
                    guards.extend((whole[:count], whole[-count:]))
        else:
            assert state.kda.config.logical_types is None
        ids = torch.full((1,), 11, dtype=torch.int32, device=device)
        logits, token = runtime.create_outputs()
        runtime.prepare(ids, state, logits=logits, next_token=token, prefill=True)
        resource = runtime.resource_report
        assert len(resource) == 228 and all(r["spill_bytes"] == 0 for r in resource)
        warmup = json.loads((args.trial / "warmup.json").read_text())
        expected_entries = {entry["symbol"]: entry["hash"] for entry in warmup["entries"]}
        assert warmup["jit_ok"] and warmup["joint_resources_ok"]
        for spec, prepared in zip(runtime.codegen["kernels"], runtime._prepared, strict=True):
            assert prepared.compiled_kernel.hash == expected_entries[spec["entry_symbol"]], spec["symbol"]
        assert len({p.compiled_kernel.hash for p in runtime._prepared}) == len(expected_entries)
        assert runtime.codegen["source_sha256"] == warmup["source_sha256"]
        report.update(diagnostic_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                      source_sha256=runtime.codegen["source_sha256"], semantic_hash=runtime.ir_module.semantic_hash,
                      compiled_entries=expected_entries, resources=resource,
                      joint_resources=runtime._program_dag.resource_report,
                      cubin_sha256={s["entry_symbol"]: hashlib.sha256(p.compiled_kernel.asm["cubin"]).hexdigest()
                          for s,p in zip(runtime.codegen["kernels"],runtime._prepared,strict=True)})
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        for phase in (True, False):
            assert_exclusive(args.output, group=host)
            dist.barrier(group=host)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                runtime.run_into(logits, token, ids, state, prefill=phase)
            graphs[phase] = graph
        torch.cuda.current_stream().wait_stream(stream)
        assert args.mode == "check-owned", "Digest reference is immutable; baseline generation is a separate step"
        assert args.reference.is_file()
        dist.barrier(group=host)
        for request in range(args.requests):
            reset_request(state)
            ids.fill_(11)
            tokens = []
            for step, expected_token in enumerate((3847, 284, 1373, 8282, 4218, 3755, 947, 906)):
                assert_exclusive(args.output, group=host)
                dist.barrier(group=host)
                graphs[step == 0].replay()
                torch.cuda.synchronize()
                selected = int(token.item())
                assert selected == expected_token, (rank, request, step, selected, expected_token)
                tokens.append(selected)
                ids.copy_(token)
                check_state(state, None, step)
                assert all(bool((guard == -123).all()) for guard in guards)
                if args.mode == "check-owned":
                    global_states = collect_kda(runtime, state, group=host)
                else:
                    local = {layer: (conv.cpu(), rec.cpu()) for layer, (conv, rec) in state.tensor_states.items()}
                    digest = hashlib.sha256()
                    for layer in sorted(local):
                        for value in local[layer]:
                            digest.update(value.contiguous().view(torch.uint8).numpy())
                    checksums = [None] * dist.get_world_size(group=host)
                    dist.all_gather_object(checksums, digest.hexdigest(), group=host)
                    assert len(set(checksums)) == 1, "full baseline state replicas disagree"
                    global_states = local if rank == 0 else None
                other = other_state(state, logits)
                reference = json.loads(args.reference.read_text())
                assert reference["schema"] == "flagmega.glm53-full-state-byte-digests/v1"
                assert reference["official_tokens"] == [3847, 284, 1373, 8282, 4218, 3755, 947, 906]
                assert reference["scope"] == dict(layers=45, experts=288, top_k=8, hidden=4096,
                    cache_capacity=4096, tp=4, prompt=[11], output=8)
                expected = reference["steps"][step]
                assert expected["step"] == step
                expected = {**expected, "tensor_states": {int(k): v for k, v in expected["tensor_states"].items()}}
                assert set(other) == set(expected["other"]), "Other-state field set differs"
                dist.barrier(group=host)
                for name, value in other.items():
                    exact(value, expected["other"][name], f"rank{rank} request{request} step{step} {name}")
                status = [None]
                if rank == 0:
                    try:
                        assert set(global_states) == set(expected["tensor_states"])
                        for layer in global_states:
                            for field, (got, want) in enumerate(zip(global_states[layer], expected["tensor_states"][layer], strict=True)):
                                exact(got, want, f"request{request} step{step} layer{layer} field{field}")
                        status[0] = {"passed": True}
                    except BaseException:
                        status[0] = {"passed": False, "traceback": traceback.format_exc()}
                dist.broadcast_object_list(status, src=0, group=host)
                assert status[0]["passed"], status[0]
                print(json.dumps({"event": "state_step_passed", "rank": rank, "request": request,
                                  "step": step, "token": selected, "kda_layers": 34}), flush=True)
            report["requests"].append({"request": request, "tokens": tokens, "all_state_bytes_equal": True})
        report["passed"] = True
    except BaseException:
        report["traceback"] = traceback.format_exc()
        raise
    finally:
        report["seconds"] = time.perf_counter() - started
        (args.output / f"rank-{rank}.json").write_text(json.dumps(report, indent=2) + "\n")
        graphs.clear()
        if host is not None:
            dist.destroy_process_group(host)
        if runtime is not None:
            runtime.close()


if __name__ == "__main__":
    main()
