"""Validate an independent greedy-token prefix from a native state snapshot.

Only initial state and the boundary input come from the reference. Every later
input is this artifact's own output. --benchmark-repeats additionally measures
GPU decode graphs; diagnostic CPU reads and snapshots are outside those timings.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
from time import perf_counter

import torch

from triton.flagmega import ir as fm
from triton.flagmega.ir.ops.nn._gdn_state import GatedDeltaNetState, GatedDeltaNetStateConfig
from triton.flagmega.ir.ops.nn._paged_attention_state import (
    PagedAttentionState, paged_attention_state_config_from_type,
)
from triton.flagmega.runtime import load


def prefix_token_ids(record, count):
    if type(count) is not int or count <= 0:
        raise ValueError("Acceptance token count must be a positive integer")
    tokens = record["token_ids"]
    if len(tokens) < count:
        raise ValueError("Reference does not contain the requested token prefix")
    return list(tokens[:count])


def acceptance_summary(records, scenario_count, token_count):
    if type(scenario_count) is not int or scenario_count <= 0 or type(token_count) is not int or token_count <= 0:
        raise ValueError("Acceptance requires positive scenario and token counts")
    return {
        "acceptance": {"kind": "independent-greedy-prefix", "tokens_per_scenario": token_count,
                       "logits_comparison_required": False, "intermediate_rounding_match_required": False},
        "all_prefixes_match": bool(records) and all(row["initial_state_roundtrip_exact"]
                                                    and len(row["token_ids"]) == token_count
                                                    and row["token_ids"] == row["reference_token_ids"] for row in records),
        "complete": len(records) == scenario_count,
    }


def logical_states(gdn, paged, layer_types, length):
    result = {}
    linear_index = attention_index = 0
    for index, kind in enumerate(layer_types):
        if kind == "linear_attention":
            result[index] = {"kind": kind,
                             "convolution": gdn.convolution_layer(linear_index).cpu(),
                             "recurrent": gdn.recurrent_layer(linear_index).cpu()}
            linear_index += 1
        elif kind == "full_attention":
            key, value = paged.gather(layer_id=attention_index, length=length)
            result[index] = {"kind": kind, "key": key.cpu(), "value": value.cpu()}
            attention_index += 1
        else:
            raise ValueError(f"Unknown layer kind: {kind}")
    return result


def state_errors(actual, expected):
    if actual.keys() != expected.keys():
        raise ValueError("State must contain the same complete set of decoder layers")
    errors = []
    for index, state in actual.items():
        reference = expected[index]
        if state.keys() != reference.keys() or state["kind"] != reference["kind"]:
            raise ValueError(f"layer {index} state fields differ")
        fields = {}
        for key, value in state.items():
            if key == "kind":
                continue
            target = reference[key]
            if value.shape != target.shape or value.dtype != target.dtype:
                raise ValueError(f"layer {index} {key} shape or dtype differs")
            delta = (value.float() - target.float()).abs()
            fields[key] = {"exact": torch.equal(value, target), "max_abs": delta.max().item(),
                           "rms": delta.square().mean().sqrt().item(),
                           "unequal": int((value != target).sum().item()), "numel": value.numel()}
        errors.append({"layer": index, "kind": state["kind"], "fields": fields})
    return errors


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--trial", type=Path, required=True)
    parser.add_argument("--label", required=True, help="Explicit variant label for provenance and timing reports")
    parser.add_argument("--benchmark-repeats", type=int, default=0)
    parser.add_argument("--benchmark-steps", type=int, help="Timed independent sequence length; acceptance still uses --tokens")
    parser.add_argument("--benchmark-host", action="store_true", help="Also time per-token host observations")
    parser.add_argument("--tokens", type=int, default=3, help="Independent greedy-prefix length required for acceptance")
    parser.add_argument("--profile-first", action="store_true",
                        help="Only the first actual decode is inside cudaProfilerStart/Stop; timings are invalid")
    args = parser.parse_args()
    if args.benchmark_repeats < 0:
        raise ValueError("benchmark repeats cannot be negative")
    if args.benchmark_steps is not None and args.benchmark_steps < args.tokens:
        raise ValueError("benchmark steps must cover the validated prefix")
    if (args.benchmark_steps is not None or args.benchmark_host) and not args.benchmark_repeats:
        raise ValueError("benchmark options require positive --benchmark-repeats")
    if args.profile_first and args.benchmark_repeats:
        raise ValueError("Profiling and formal performance measurements must run separately")
    args.trial.mkdir(parents=True, exist_ok=False)
    (args.trial / "runner.py").write_bytes(Path(__file__).read_bytes())
    if args.benchmark_repeats:
        helper_path = Path(__file__).with_name("prepared_gpu_timing.py")
        (args.trial / helper_path.name).write_bytes(helper_path.read_bytes())
    reference = json.loads(args.reference.read_text())
    if reference["schema"] != "flagmega.prepared-decode-reference/v1":
        raise ValueError("Only prepared-state decode-only references are supported")
    if not reference["records"]:
        raise ValueError("Acceptance requires at least one reference scenario")
    for record in reference["records"]:
        prefix_token_ids(record, args.tokens)
    config = json.loads((args.checkpoint / "config.json").read_text())["text_config"]
    layer_types = config["layer_types"]
    if len(layer_types) != 40:
        raise ValueError("Acceptance requires the complete 40-layer model")
    torch.set_num_threads(1)
    import triton
    from triton.backends.nvidia.compiler import get_ptxas, get_ptxas_version
    capability = torch.cuda.get_device_capability()
    arch = capability[0] * 10 + capability[1]
    assembler = Path(get_ptxas(arch).path)
    toolchain = {"triton": triton.__version__, "ptxas_path": str(assembler),
                 "ptxas_version": get_ptxas_version(arch),
                 "ptxas_sha256": hashlib.sha256(assembler.read_bytes()).hexdigest()}
    started = perf_counter()
    from agent_optimizations import shared_router_collective  # Register public local-op serialization.
    runtime = load(args.artifact)
    print("Manifest/rdata verification seconds", perf_counter() - started, flush=True)
    runtime.load("cuda:0")
    print("Device loading cumulative seconds", perf_counter() - started, flush=True)
    values = {}
    for argument in runtime.external_arguments:
        buffer = runtime.buffer_plan.buffer_map[argument["buffer"]]
        lanes = buffer.dtype.lanes if isinstance(buffer.dtype, fm.VectorType) else ()
        dtype = buffer.dtype.elem_type if lanes else buffer.dtype
        values[buffer.id] = torch.zeros((*buffer.shape, *lanes), dtype=getattr(torch, dtype.value), device="cuda:0")
    arguments = tuple(values[argument["buffer"]] for argument in runtime.external_arguments)
    outputs = runtime.buffer_plan.function_map[runtime.ir_module.entry].outputs
    assert len(outputs[0][1]) == len(outputs[1][1]) == 1
    logits_buffer, token_buffer = outputs[0][1][0], outputs[1][1][0]
    gdn_config = GatedDeltaNetStateConfig(
        num_layers=layer_types.count("linear_attention"), num_key_heads=config["linear_num_key_heads"],
        num_value_heads=config["linear_num_value_heads"], key_head_dim=config["linear_key_head_dim"],
        value_head_dim=config["linear_value_head_dim"], conv_kernel_size=config["linear_conv_kernel_dim"],
        hidden_size=config["hidden_size"],
    )
    gdn = GatedDeltaNetState(values["gdn_state.convolution"], values["gdn_state.recurrent"], gdn_config)
    gdn.validate()
    paged_config = paged_attention_state_config_from_type(runtime.ir_module.node_map["paged_state"].type)
    paged = PagedAttentionState(*(values[f"paged_state.{name}"] for name in
                                 ("kv_caches", "query_start_loc", "seq_lens", "slot_mapping", "block_table")),
                                config=paged_config)
    if paged_config.num_layers != layer_types.count("full_attention"):
        raise ValueError("KV ABI does not contain all full-attention layers")
    print("Candidate ABI prepared", flush=True)
    records, timings = [], []
    for target in reference["records"]:
        expected = prefix_token_ids(target, args.tokens)
        # The previous scenario may have captured a graph on another stream.
        # Bind a new prepared instance for this scenario's eager diagnostics.
        runtime.prepare(*arguments)
        length = target["prefix_length"]
        state_path = args.reference.parent / target["state"]
        if hashlib.sha256(state_path.read_bytes()).hexdigest() != target["state_sha256"]:
            raise ValueError("Initial snapshot hash mismatch")
        initial = torch.load(state_path, weights_only=True, map_location="cpu")
        if (initial["schema"] != "flagmega.qwen35-prepared-decode-state/v1"
                or initial["prefix_length"] != length or initial["initial_token"] != target["initial_token"]):
            raise ValueError("Initial snapshot boundary mismatch")
        for value in values.values():
            value.zero_()
        paged.block_table.copy_(torch.arange(paged.block_table.numel(), dtype=torch.int32, device="cuda:0")
                                .reshape_as(paged.block_table))
        linear_index = attention_index = 0
        for index, kind in enumerate(layer_types):
            state = initial["layers"][index]
            if state["kind"] != kind:
                raise ValueError("Model layer and reference state types differ")
            if kind == "linear_attention":
                gdn.update_convolution_layer(state["convolution"], linear_index)
                gdn.update_recurrent_layer(state["recurrent"], linear_index)
                linear_index += 1
            else:
                for field in ("key", "value"):
                    paged.update(state[field], cache_kind=field, layer_id=attention_index, advance_sequence=False)
                attention_index += 1
        paged.seq_lens.fill_(length)
        paged.query_start_loc.copy_(torch.tensor([0, 1], dtype=torch.int32, device="cuda:0"))
        errors = state_errors(logical_states(gdn, paged, layer_types, length), initial["layers"])
        if any(not field["exact"] for row in errors for field in row["fields"].values()):
            raise RuntimeError("Initial physical-layout conversion must preserve every value")
        token = initial["initial_token"]
        values["input_ids"].fill_(token)
        benchmark_initial = {
            name: value.clone() for name, value in values.items()
            if name.startswith(("gdn_state.", "paged_state.")) or name == "input_ids"
        } if args.benchmark_repeats else None
        generated, diagnostics, first_state_errors = [], [], None
        # Reference outputs determine only the requested length, never inputs.
        for step in range(len(expected)):
            position = length + step
            if paged.sequence_length != position:
                raise RuntimeError("The model must advance its own state on every step")
            values["input_ids"].fill_(token)
            paged.slot_mapping.fill_(position)
            logits = values[logits_buffer]
            logits.fill_(float("nan"))
            values[token_buffer].fill_(-1)
            profile_this_step = args.profile_first and not records and step == 0
            if profile_this_step:
                torch.cuda.synchronize()
                torch.cuda.cudart().cudaProfilerStart()
            runtime.run_into(*arguments)
            if profile_this_step:
                torch.cuda.synchronize()
                torch.cuda.cudart().cudaProfilerStop()
            token = int(values[token_buffer].item())
            if not bool(torch.isfinite(logits).all().item()) or not 0 <= token < logits.shape[-1]:
                raise RuntimeError(f"context {length}, step {step}: Non-finite output or invalid token")
            if paged.sequence_length != position + 1:
                raise RuntimeError("The model did not advance the past-token count exactly once")
            generated.append(token)
            top = torch.topk(logits.flatten(), 20)
            diagnostics.append({"position": position, "ids": top.indices.tolist(), "logits": top.values.tolist()})
            torch.save(logits.cpu(), args.trial / f"context-{length}-logits-{step}.pt")
            if step == 0:
                state = logical_states(gdn, paged, layer_types, length + 1)
                torch.save(state, args.trial / f"context-{length}-after-first.pt")
                first_reference = torch.load(args.reference.parent / f"context-{length}-after-first.pt",
                                             map_location="cpu", weights_only=True)
                first_state_errors = state_errors(state, first_reference)
            print("context", length, "Independent tokens", generated, flush=True)
        first_mismatch = next((i for i, pair in enumerate(zip(generated, expected)) if pair[0] != pair[1]), None)
        records.append({"prefix_length": length, "initial_token": initial["initial_token"],
                        "initial_state_sha256": target["state_sha256"], "initial_state_roundtrip_exact": True,
                        "token_ids": generated, "reference_token_ids": expected,
                        "full_reference_token_count": len(target["token_ids"]),
                        "exact_match": generated == expected, "first_mismatch": first_mismatch,
                        "diagnostics": diagnostics, "first_decode_state_errors": first_state_errors})
        report = {"schema": "flagmega.prepared-decode-candidate/v2", "performance_valid": False,
                  "variant": args.label, "artifact": str(args.artifact.resolve()),
                  **acceptance_summary(records, len(reference["records"]), args.tokens),
                  "numerical_contract": runtime.ir_module.metadata.get("numerical_contract", "unspecified"),
                  "semantic_hash": runtime.ir_module.semantic_hash, "source_sha256": runtime.codegen["source_sha256"],
                  "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  "reference_sha256": hashlib.sha256(args.reference.read_bytes()).hexdigest(),
                  "reference": str(args.reference.resolve()), "device": torch.cuda.get_device_name(),
                  "toolchain": toolchain,
                  "visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"), "records": records}
        (args.trial / "report.json").write_text(json.dumps(report, indent=2) + "\n")
        print("context", length, "greedy prefix", args.tokens, "match", generated == expected,
              "first mismatch", first_mismatch, flush=True)
        if args.benchmark_repeats:
            if generated != expected:
                raise RuntimeError("Artifacts that fail numerical acceptance cannot enter performance measurement")
            from prepared_gpu_timing import benchmark
            timing = benchmark(runtime, arguments, values, benchmark_initial, expected, token_buffer,
                               repeats=args.benchmark_repeats, steps=args.benchmark_steps,
                               host_feedback=args.benchmark_host, logits_buffer=logits_buffer)
            timings.append({"prefix_length": length, **timing})
            extended = args.benchmark_steps is not None and args.benchmark_steps > args.tokens
            benchmark_report = {"schema": "flagmega.prepared-decode-gpu-timing/v2" if extended else "flagmega.prepared-decode-gpu-timing/v1", "performance_valid": True,
                                "comparison_to_vllm": False, "variant": args.label,
                                "artifact": str(args.artifact.resolve()),
                                "semantic_hash": runtime.ir_module.semantic_hash,
                                "source_sha256": runtime.codegen["source_sha256"], "resources": runtime.resource_report,
                                "benchmark_sha256": hashlib.sha256(helper_path.read_bytes()).hexdigest(),
                                "torch": torch.__version__, "visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                                "device": torch.cuda.get_device_name(), "scenarios": timings,
                                "toolchain": toolchain,
                                "acceptance": report["acceptance"],
                                "complete": len(timings) == len(reference["records"])}
            (args.trial / "gpu-timing.json").write_text(json.dumps(benchmark_report, indent=2) + "\n")
            print("GPU", args.label, length, timing["median_ms"], "ms", flush=True)
    if not report["complete"] or not report["all_prefixes_match"]:
        raise RuntimeError("Independent greedy-prefix acceptance failed; see report.json")


if __name__ == "__main__":
    main()
