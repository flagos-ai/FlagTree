"""Validate independent decode from a pinned native prefix snapshot.

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
            raise ValueError(f"未知 layer kind: {kind}")
    return result


def state_errors(actual, expected):
    if actual.keys() != expected.keys():
        raise ValueError("状态必须包含同一组完整 decoder layers")
    errors = []
    for index, state in actual.items():
        reference = expected[index]
        if state.keys() != reference.keys() or state["kind"] != reference["kind"]:
            raise ValueError(f"layer {index} 状态字段不一致")
        fields = {}
        for key, value in state.items():
            if key == "kind":
                continue
            target = reference[key]
            if value.shape != target.shape or value.dtype != target.dtype:
                raise ValueError(f"layer {index} {key} 的 shape/dtype 不一致")
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
    parser.add_argument("--profile-first", action="store_true",
                        help="Only the first actual decode is inside cudaProfilerStart/Stop; timings are invalid")
    args = parser.parse_args()
    if args.benchmark_repeats < 0:
        raise ValueError("benchmark repeats 不能是负数")
    if args.profile_first and args.benchmark_repeats:
        raise ValueError("profiler 与正式性能测量必须分开")
    args.trial.mkdir(parents=True, exist_ok=False)
    (args.trial / "runner.py").write_bytes(Path(__file__).read_bytes())
    if args.benchmark_repeats:
        helper_path = Path(__file__).with_name("prepared_gpu_timing.py")
        (args.trial / helper_path.name).write_bytes(helper_path.read_bytes())
    reference = json.loads(args.reference.read_text())
    if reference["schema"] != "flagmega.prepared-decode-reference/v1":
        raise ValueError("只接受 decode-only 的已准备状态参考")
    config = json.loads((args.checkpoint / "config.json").read_text())["text_config"]
    layer_types = config["layer_types"]
    if len(layer_types) != 40:
        raise ValueError("验收要求完整 40 层模型")
    torch.set_num_threads(1)
    started = perf_counter()
    runtime = load(args.artifact)
    print("manifest/rdata 校验秒", perf_counter() - started, flush=True)
    if runtime.ir_module.metadata.get("numerical_contract") != "vllm-ae10e855a-inductor-level3":
        raise ValueError("候选必须是固定 vLLM 数值合同，不能使用 nncase 默认 profile")
    runtime.load("cuda:0")
    print("设备加载累计秒", perf_counter() - started, flush=True)
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
        raise ValueError("KV ABI 不包含全部 full-attention layers")
    print("候选 ABI 准备完成", flush=True)
    records, timings = [], []
    for target in reference["records"]:
        # The previous scenario may have captured a graph on another stream.
        # Bind a new prepared instance for this scenario's eager diagnostics.
        runtime.prepare(*arguments)
        length = target["prefix_length"]
        state_path = args.reference.parent / target["state"]
        if hashlib.sha256(state_path.read_bytes()).hexdigest() != target["state_sha256"]:
            raise ValueError("初始快照 hash 不匹配")
        initial = torch.load(state_path, weights_only=True, map_location="cpu")
        if (initial["schema"] != "flagmega.qwen35-prepared-decode-state/v1"
                or initial["prefix_length"] != length or initial["initial_token"] != target["initial_token"]):
            raise ValueError("初始快照边界不匹配")
        for value in values.values():
            value.zero_()
        paged.block_table.copy_(torch.arange(paged.block_table.numel(), dtype=torch.int32, device="cuda:0")
                                .reshape_as(paged.block_table))
        linear_index = attention_index = 0
        for index, kind in enumerate(layer_types):
            state = initial["layers"][index]
            if state["kind"] != kind:
                raise ValueError("模型层与参考状态类型不一致")
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
            raise RuntimeError("初始物理布局转换不能改变任何数值")
        token = initial["initial_token"]
        values["input_ids"].fill_(token)
        benchmark_initial = {
            name: value.clone() for name, value in values.items()
            if name.startswith(("gdn_state.", "paged_state.")) or name == "input_ids"
        } if args.benchmark_repeats else None
        generated, diagnostics, first_state_errors = [], [], None
        # Reference outputs determine only the requested length, never inputs.
        for step in range(len(target["token_ids"])):
            position = length + step
            if paged.sequence_length != position:
                raise RuntimeError("模型必须独立逐步推进状态")
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
                raise RuntimeError(f"context {length}, step {step}: 非有限输出或非法 token")
            if paged.sequence_length != position + 1:
                raise RuntimeError("模型没有把 past-token 长度推进恰好一次")
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
            print("context", length, "独立 tokens", generated, flush=True)
        expected = target["token_ids"]
        first_mismatch = next((i for i, pair in enumerate(zip(generated, expected)) if pair[0] != pair[1]), None)
        records.append({"prefix_length": length, "initial_token": initial["initial_token"],
                        "initial_state_sha256": target["state_sha256"], "initial_state_roundtrip_exact": True,
                        "token_ids": generated, "reference_token_ids": expected,
                        "exact_match": generated == expected, "first_mismatch": first_mismatch,
                        "diagnostics": diagnostics, "first_decode_state_errors": first_state_errors})
        report = {"schema": "flagmega.prepared-decode-candidate/v1", "performance_valid": False,
                  "variant": args.label, "artifact": str(args.artifact.resolve()),
                  "all_sequences_match": all(row["exact_match"] for row in records),
                  "complete": len(records) == len(reference["records"]),
                  "semantic_hash": runtime.ir_module.semantic_hash, "source_sha256": runtime.codegen["source_sha256"],
                  "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  "reference_sha256": hashlib.sha256(args.reference.read_bytes()).hexdigest(),
                  "reference": str(args.reference.resolve()), "device": torch.cuda.get_device_name(),
                  "visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"), "records": records}
        (args.trial / "report.json").write_text(json.dumps(report, indent=2) + "\n")
        print("context", length, "完整序列一致", generated == expected, "首个分歧", first_mismatch, flush=True)
        if args.benchmark_repeats:
            if generated != expected:
                raise RuntimeError("数值失败的 artifact 不进入性能测量")
            from prepared_gpu_timing import benchmark
            timing = benchmark(runtime, arguments, values, benchmark_initial, expected, token_buffer,
                               repeats=args.benchmark_repeats)
            timings.append({"prefix_length": length, **timing})
            benchmark_report = {"schema": "flagmega.prepared-decode-gpu-timing/v1", "performance_valid": True,
                                "comparison_to_vllm": False, "variant": args.label,
                                "artifact": str(args.artifact.resolve()),
                                "semantic_hash": runtime.ir_module.semantic_hash,
                                "source_sha256": runtime.codegen["source_sha256"], "resources": runtime.resource_report,
                                "benchmark_sha256": hashlib.sha256(helper_path.read_bytes()).hexdigest(),
                                "torch": torch.__version__, "visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                                "device": torch.cuda.get_device_name(), "scenarios": timings,
                                "complete": len(timings) == len(reference["records"])}
            (args.trial / "gpu-timing.json").write_text(json.dumps(benchmark_report, indent=2) + "\n")
            print("GPU", args.label, length, timing["median_ms"], "ms", flush=True)


if __name__ == "__main__":
    main()
