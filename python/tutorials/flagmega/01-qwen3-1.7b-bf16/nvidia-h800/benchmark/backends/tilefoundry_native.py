#!/usr/bin/env python3
"""Bind the unchanged official TileFoundry/TileLang runtime twin, never evaluator.

CPU preflight reads pinned source/package/checkpoint metadata only. A later leased
GPU invocation builds every official kernel and retains native sources/libraries.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import sys
import traceback

from reference_rounds import (common_arguments, check_arguments, reference_prompts, controls,
    bind_gpu, StepTimer, step_arrays, validate_run, aggregate, sha, write_new, gpu_inventory)

MODE = "official-tilefoundry-tilelang-runtime-twin"


def validate_sources(path):
    inventory = json.loads(path.read_text())
    for pin in inventory["source_pins"]:
        if sha(pin["path"]) != pin["sha256"]:
            raise ValueError("Official/dependency source changed: " + pin["path"])
    versions = {name: importlib.metadata.version(name) for name in ("tilefoundry", "tilelang", "torch", "transformers")}
    if any(versions[key] != inventory["environment"][key] for key in versions):
        raise ValueError("Use the matching pinned installed official package versions")
    root = Path(inventory["repo"]["path"])/"examples/qwen3_1_7b-tilelang"
    return inventory, root, versions


def load_engine(root):
    fast = root/"fast"
    sys.path.insert(0, str(fast))
    path = fast/"engine.py"
    spec = importlib.util.spec_from_file_location("official_tile_engine", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    if Path(module.K.__file__).resolve() != (fast/"kernels.py").resolve():
        raise RuntimeError("An unrelated kernels module displaced the official native binding")
    return module.Engine


def native_artifacts(engine, directory):
    directory.mkdir(parents=True, exist_ok=False)
    result = []
    for name, kernel in vars(engine).items():
        if not name.startswith("k_"):
            continue
        cuda = directory/(name+".cu")
        host = directory/(name+".host.txt")
        library = directory/(name+".so")
        cuda.write_text(kernel.get_kernel_source())
        host.write_text(kernel.get_host_source())
        kernel.export_library(str(library))
        if not cuda.stat().st_size or not library.stat().st_size:
            raise RuntimeError("Missing full native source/library evidence")
        result.append({"name": name, "target": str(kernel.target), "execution_backend": kernel.execution_backend,
            "adapter": type(kernel.adapter).__module__+"."+type(kernel.adapter).__name__,
            "files": [{"path":str(p),"sha256":sha(p)} for p in (cuda,host,library)]})
    if len(result) != 12:
        raise RuntimeError("All12 official kernel call bindings must be retained")
    return result


def request(engine, prompt, timer, torch, cpu_token):
    # Same setup and state owners as official generate; only the timing split differs.
    engine._reset()
    engine.inp.zero_()
    engine.inp[:len(prompt)] = torch.tensor(prompt, device=engine.device, dtype=torch.int32)
    engine.plen.fill_(len(prompt))
    engine.ids.fill_(prompt[0])
    timer.synchronize()
    for _ in range(len(prompt)):
        engine.graph.replay()
    timer.synchronize()
    generated = [int(engine.ids.item())]
    steps = []
    for _ in range(63):
        def operation():
            engine.graph.replay()  # Engine sample_step updates its own Ids and Pos.
            cpu_token.copy_(engine.ids, non_blocking=True)  # matched one-token D2H
            return cpu_token
        token, sample = timer.measure(operation, lambda value: int(value.item()))
        generated.append(token)
        steps.append(sample)
    # Bulk64/state inspection happens outside all timers.
    recorded = engine.sam[len(prompt)-1:len(prompt)+63].tolist()
    position = int(engine.pos.item())
    if position != len(prompt)+63 or recorded != generated:
        raise RuntimeError("Official sampled history/position does not match all64 delivered outputs")
    return {"prompt_tokens":prompt, "token_ids":generated, "consumed_token_ids":prompt+generated[:-1],
        "final_state_length":position, "official_samples":recorded, **step_arrays(steps)}


def inspect_state(engine, torch):
    position = int(engine.pos.item())
    if tuple(engine.kc.shape) != (28,engine.cap,1024) or engine.kc.dtype != torch.bfloat16 or engine.vc.dtype != torch.bfloat16:
        raise RuntimeError("Official full28 BF16 KV geometry differs")
    prefixes = [tensor[:,:position] for tensor in (engine.kc,engine.vc)]
    finite = all(bool(torch.isfinite(value).all().item()) for value in prefixes)
    per_layer = [bool(torch.count_nonzero(engine.kc[layer,:position]).item()) and bool(torch.count_nonzero(engine.vc[layer,:position]).item()) for layer in range(28)]
    tails_zero = all(int(torch.count_nonzero(tensor[:,position:]).item()) == 0 for tensor in (engine.kc,engine.vc))
    if not finite or not all(per_layer) or not tails_zero:
        raise RuntimeError("Native full28 KV finite/write/tail checks failed")
    return {"actual_position":position,"cap":engine.cap,"shape":list(engine.kc.shape),"dtype":"BF16",
        "finite_prefix":finite,"per_layer_KV_written":per_layer,"unwritten_tails_zero":tails_zero,
        "logits_dtype":str(engine.logits.dtype),"logits_shape":list(engine.logits.shape)}


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    common_arguments(parser)
    parser.add_argument("--inventory",type=Path,required=True)
    parser.add_argument("--native-artifact-dir",type=Path)
    parser.add_argument("--save-kv",action="store_true",help="Untimed full state export for independent NUM admission")
    args=parser.parse_args(argv)
    check_arguments(args)
    report={"schema":"flagmega.model-measurement/v1","backend":"tilefoundry","label":MODE,"execution_mode":MODE,
        "status":"preflight","measurement_id":args.measurement_id,"cohort":args.cohort,"round_index":args.round_index,
        "timestamp_utc":datetime.now(timezone.utc).isoformat(),"script":{"path":str(Path(__file__).resolve()),"sha256":sha(__file__)},
        "controls":controls(args),"scenarios":[],"GPU_initialized":False,
        "boundary":{"host":"63 synchronous official graph replay+one-token pinned D2H+completion+host delivery intervals",
            "device":"preallocated events around each full official graph replay and one-token pinned D2H; excludes bulk64/logits/KV export; includes in-window launch idle"},
        "workload":{"batch":1,"concurrency":1,"layers":28,"vocab":151936,"dtype":"bfloat16","kv_dtype":"bfloat16","logits_dtype":"float32","prompt_lengths":args.prompt_lengths,"decode_tokens":64,"sampling":"official FP32 greedy/tie","prefix_caching":False,"speculative_decoding":False}}
    try:
        inventory,root,versions=validate_sources(args.inventory)
        checkpoint,inputs,reference=reference_prompts(args)
        report.update(checkpoint=checkpoint,reference=reference,version=versions["tilefoundry"],versions=versions,
            inventory={"path":str(args.inventory.resolve()),"sha256":sha(args.inventory)},source_pins=inventory["source_pins"],
            official_repo=inventory["repo"],sizing={str(p):{"max_new":64,"prompt_room":p+8,"nsteps":p+72,"cap":((p+72+255)//256)*256} for p in args.prompt_lengths})
        if args.preflight_only:
            write_new(args.output,report);return 0
        import torch
        bind_gpu(torch,args.physical_gpu_uuid)
        torch.set_num_threads(1)
        Engine=load_engine(root)
        report.update(GPU_initialized=True,gpu_before=gpu_inventory(),visible_devices=os.environ["CUDA_VISIBLE_DEVICES"])
        directory=args.native_artifact_dir or args.output.with_suffix(".native")
        timer=StepTimer(torch,args.cohort)
        cpu_token=torch.empty(1,dtype=torch.int64,device="cpu",pin_memory=True)
        for length in args.prompt_lengths:
            engine=Engine(args.checkpoint,root/"ref_src",device="cuda:0",max_new=64,prompt_room=length+8)
            if (engine.L,engine.H,engine.I,engine.V)!=(28,2048,6144,151936) or engine.logits.dtype!=torch.float32:
                raise RuntimeError("Unchanged official full model/FP32 logits contract required")
            native=native_artifacts(engine,directory/f"prompt-{length}")
            prompt=inputs[length]["prompt"]
            for _ in range(2): request(engine,prompt,timer,torch,cpu_token)
            scenario={"prompt_length":length,"runs":[],"native_kernels":native};report["scenarios"].append(scenario)
            for repeat in range(3):
                run=request(engine,prompt,timer,torch,cpu_token);run["repeat_index"]=repeat
                run["state_checks"]=inspect_state(engine,torch)
                validate_run(run,prompt,inputs[length]["expected"])
                scenario["runs"].append(run)
                if args.save_kv and repeat==0:
                    state=directory/f"prompt-{length}"/"actual-own-history-state.pt"
                    torch.save({"K":engine.kc.cpu(),"V":engine.vc.cpu(),"Pos":engine.pos.cpu(),"Ids":engine.ids.cpu(),"Sam":engine.sam.cpu(),"Inp":engine.inp.cpu(),"PromptLen":engine.plen.cpu()},state)
                    run["state_snapshot"]={"path":str(state),"sha256":sha(state)}
            aggregate(scenario)
            del engine
            torch.cuda.synchronize()
        validate_sources(args.inventory)
        report.update(status="measured",NUM_passed=all(r["NUM"]["first10_exact"] for s in report["scenarios"] for r in s["runs"]),
            all64_exact=all(r["NUM"]["all64_exact"] for s in report["scenarios"] for r in s["runs"]),gpu_after=gpu_inventory(),acceptance="pending independent native build, NUM, boundary/isolation and matched-round review; no target claim")
        write_new(args.output,report);return 0
    except BaseException as error:
        report.update(status="failed",error=str(error),traceback=traceback.format_exc())
        write_new(args.output.with_suffix(".failed.json"),report);raise


if __name__=="__main__":raise SystemExit(main())
