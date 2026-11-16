#!/usr/bin/env python3
"""Pinned native llama.cpp BF16 requests, retaining independent greedy histories.

Driver compilation and preflight are CPU-only. A measured invocation requires a
device lease and reports host decode+backend-sampling+token-delivery intervals.
Native GPU graph timing remains a separate capture requirement.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import traceback

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from benchmark_model import gpu_inventory, sha, summary, validate_checkpoint, write_new
from build_llama_reference import COMMIT
from reference_rounds import common_arguments, check_arguments, reference_prompts, controls, validate_run, aggregate


def validate_build(receipt):
    data = json.loads(receipt.read_text())
    if data.get("status") != "built_cpu_only" or data["source"]["commit"] != COMMIT or data["source"]["dirty"]:
        raise ValueError("Require the completed clean pinned native build receipt")
    source = Path(data["source"]["path"])
    if subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip() != COMMIT:
        raise ValueError("Native source commit changed")
    if subprocess.check_output(["git", "-C", str(source), "status", "--porcelain"], text=True).strip():
        raise ValueError("Native source must remain unmodified")
    for binary in data["binaries"]:
        if sha(binary["path"]) != binary["sha256"]:
            raise ValueError("Native binary changed after its pinned build")
    gguf = Path(data["gguf"]["path"])
    if sha(gguf) != data["gguf"]["sha256"]:
        raise ValueError("BF16 GGUF changed after its pinned conversion")
    sys.path.insert(0, str(source / "gguf-py"))
    from gguf import GGUFReader, GGMLQuantizationType
    reader = GGUFReader(gguf)
    fields = {key: reader.fields[key].contents() for key in
              ("general.architecture", "qwen3.block_count", "qwen3.embedding_length", "qwen3.attention.head_count",
               "qwen3.attention.head_count_kv", "qwen3.feed_forward_length")}
    expected = {"general.architecture": "qwen3", "qwen3.block_count": 28, "qwen3.embedding_length": 2048,
                "qwen3.attention.head_count": 16, "qwen3.attention.head_count_kv": 8,
                "qwen3.feed_forward_length": 6144}
    if fields != expected or len(reader.fields["tokenizer.ggml.tokens"].data) != 151936:
        raise ValueError("Converted GGUF does not match the full Qwen3-1.7B dimensions and vocabulary")
    if any(t.tensor_type not in (GGMLQuantizationType.BF16, GGMLQuantizationType.F32) for t in reader.tensors):
        raise ValueError("Converted reference contains a different dtype or quantization")
    matrices = [t for t in reader.tensors if len(t.shape) > 1]
    if len(matrices) < 169 or any(t.tensor_type != GGMLQuantizationType.BF16 for t in matrices):
        raise ValueError("Require BF16 embedding/projection/LM-head matrices for the full model")
    lib = next(Path(b["path"]).parent for b in data["binaries"] if Path(b["path"]).name.startswith("libllama.so"))
    return data, source, gguf, lib, {"fields": fields,
        "tensor_types": dict(Counter(t.tensor_type.name for t in reader.tensors)), "bf16_matrices": len(matrices)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    common_arguments(parser)
    parser.add_argument("--build-receipt", type=Path, required=True)
    parser.add_argument("--driver", type=Path, required=True)
    parser.add_argument("--driver-binding", type=Path, required=True)
    parser.add_argument("--build-driver-only", action="store_true")
    args = parser.parse_args(argv)
    check_arguments(args)
    report = {"schema": "flagmega.model-measurement/v1", "backend": "llama.cpp", "label": "native-llama.cpp",
              "measurement_id": args.measurement_id, "round_index": args.round_index, "cohort": args.cohort,
              "status": "preflight", "GPU_initialized": False,
              "timestamp_utc": datetime.now(timezone.utc).isoformat(), "scenarios": []}
    try:
        if args.cohort != "ordinary_no_events":
            raise ValueError("llama.cpp DEVICE unsupported: Python events cannot observe the separate native GGML CUDA stream; requires approved C++ driver binding")
        native, source, gguf, lib, geometry = validate_build(args.build_receipt)
        checkpoint, inputs, reference = reference_prompts(args)
        if checkpoint != native["checkpoint"]:
            raise ValueError("HF checkpoint identity differs from the native BF16 conversion")
        cpp = Path(__file__).with_suffix(".cpp")
        report.update(checkpoint=checkpoint, reference=reference, version=COMMIT,
            native_build={"path": str(args.build_receipt.resolve()), "sha256": sha(args.build_receipt)},
            gguf={**native["gguf"], **geometry},
            script={"path": str(Path(__file__).resolve()), "sha256": sha(__file__)},
            visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
            workload={"batch": 1, "concurrency": 1, "dtype": "bfloat16", "kv_dtype": "bfloat16", "layers": 28,
                "prompt_lengths": args.prompt_lengths, "decode_tokens": args.decode_tokens, "sampling": "greedy",
                "prefix_caching": False, "speculative_decoding": False},
            controls={**controls(args), "host_clock": "native CLOCK_MONOTONIC nanoseconds; input staging through per-step delivery/completion",
                "cuda_graph": "native GGML_CUDA_GRAPHS=ON; actual graph use requires runtime/capture evidence",
                "flash_attention": "requested enabled; unsupported native runtime raises",
                "gpu_sampler": "native context sampler chain; CUDA greedy capability probe required",
                "n_ctx": 3072, "n_batch": 3072, "n_ubatch": 3072, "n_gpu_layers": -1,
                "threads": 1, "cache_flush": "none", "reset": "clear KV data and sampler before each independent request"},
            boundary={"host": "native input staging+llama_decode+backend greedy+accept+token delivery synchronization; prefill/first output excluded",
                "device": None, "device_gap": "Validated CUDA graph/stream timing and sampler placement capture pending"})
        if args.build_driver_only:
            if args.driver.exists() or args.driver_binding.exists():
                raise ValueError("Use fresh driver and binding paths")
            args.driver.parent.mkdir(parents=True, exist_ok=True)
            command = ["g++", "-std=c++17", "-O3", str(cpp), "-I", str(source / "include"), "-I",
                str(source / "ggml/include"), "-L", str(lib), "-Wl,-rpath," + str(lib), "-lllama", "-lggml",
                "-lggml-base", "-o", str(args.driver)]
            env = dict(os.environ, CUDA_VISIBLE_DEVICES="")
            result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env)
            log = args.output.with_suffix(".build.log")
            log.write_text(result.stdout)
            report["compile"] = {"argv": command, "exit_code": result.returncode, "log": str(log), "log_sha256": sha(log)}
            if result.returncode:
                raise RuntimeError("Native benchmark driver compilation failed; see retained log")
            binding = {"schema": "flagmega.llama-native-driver/v1", "status": "built_cpu_only",
                "build_receipt_sha256": sha(args.build_receipt), "driver": {"path": str(args.driver.resolve()),
                    "sha256": sha(args.driver)}, "cpp": {"path": str(cpp.resolve()), "sha256": sha(cpp)},
                "compile": report["compile"]}
            write_new(args.driver_binding, binding)
            report.update(status="built_cpu_only", driver_binding=binding)
            write_new(args.output, report)
            return 0
        binding = json.loads(args.driver_binding.read_text())
        if (binding["status"] != "built_cpu_only" or binding["build_receipt_sha256"] != sha(args.build_receipt)
                or binding["cpp"]["sha256"] != sha(cpp) or binding["driver"]["sha256"] != sha(args.driver)
                or Path(binding["driver"]["path"]).resolve() != args.driver.resolve()):
            raise ValueError("Native driver/build/source binding changed")
        report["driver_binding"] = {"path": str(args.driver_binding.resolve()), "sha256": sha(args.driver_binding)}
        prompt_ids = [inputs[length]["prompt"] for length in args.prompt_lengths]
        if args.preflight_only:
            report["prompts"] = prompt_ids
            write_new(args.output, report)
            return 0
        if not args.physical_gpu_uuid or os.environ.get("CUDA_VISIBLE_DEVICES") != args.physical_gpu_uuid:
            raise ValueError("Native invocation requires exact leased UUID matching CUDA_VISIBLE_DEVICES")
        raw, prompt_file = args.output.with_suffix(".raw.json"), args.output.with_suffix(".prompts.txt")
        stdout, stderr = args.output.with_suffix(".stdout.log"), args.output.with_suffix(".stderr.log")
        if any(p.exists() for p in (raw, prompt_file, stdout, stderr)):
            raise ValueError("Use fresh native raw and log paths")
        with prompt_file.open("x") as stream:
            stream.write(str(len(prompt_ids)) + "\n")
            for prompt in prompt_ids:
                stream.write(str(len(prompt)) + " " + " ".join(map(str, prompt)) + "\n")
        report.update(GPU_initialized=True, gpu_before=gpu_inventory())
        command = [str(args.driver.resolve()), str(gguf), str(prompt_file), str(raw), str(args.decode_tokens),
            str(args.repeats), str(args.warmups)]
        with stdout.open("x") as out, stderr.open("x") as err:
            result = subprocess.run(command, stdout=out, stderr=err)
        report["native_command"] = {"argv": command, "exit_code": result.returncode,
            "stdout": {"path": str(stdout), "sha256": sha(stdout)},
            "stderr": {"path": str(stderr), "sha256": sha(stderr)}}
        if result.returncode:
            raise RuntimeError("Native llama.cpp reference failed; retain logs and unsupported-mode gap")
        data = json.loads(raw.read_text())
        if data["layers"] != 28 or data["vocab"] != 151936 or not data["capabilities"]["cuda_greedy_backend_probe"]:
            raise RuntimeError("Native runtime did not satisfy the full-model/sampler contract")
        if len(data["scenarios"]) != len(prompt_ids):
            raise RuntimeError("Native runtime omitted a requested scenario")
        for prompt, scenario in zip(prompt_ids, data["scenarios"], strict=True):
            if scenario["prompt_length"] != len(prompt) or len(scenario["runs"]) != args.repeats:
                raise RuntimeError("Native runtime omitted a requested trajectory")
            for run in scenario["runs"]:
                if (run["prompt_tokens"] != prompt or len(run["token_ids"]) != args.decode_tokens
                        or run["consumed_token_ids"] != prompt + run["token_ids"][:-1]
                        or run["final_kv_position"] != len(prompt) + args.decode_tokens - 2
                        or len(run["decode_step_ms"]) != args.decode_tokens - 1):
                    raise RuntimeError("Native output/history/KV progression differs from the matched workload")
                validate_run(run, prompt, inputs[len(prompt)]["expected"])
            aggregate(scenario)
        report.update(status="measured", scenarios=data["scenarios"], native_capabilities=data["capabilities"],
            NUM_passed=all(r["NUM"]["first10_exact"] for s in data["scenarios"] for r in s["runs"]),
            all64_exact=all(r["NUM"]["all64_exact"] for s in data["scenarios"] for r in s["runs"]),
            raw={"path": str(raw), "sha256": sha(raw)}, gpu_after=gpu_inventory(),
            acceptance="pending independent review; native host baseline only")
        write_new(args.output, report)
        return 0
    except BaseException as error:
        report.update(status="failed", error=str(error), traceback=traceback.format_exc())
        write_new(args.output.with_suffix(".failed.json"), report)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
