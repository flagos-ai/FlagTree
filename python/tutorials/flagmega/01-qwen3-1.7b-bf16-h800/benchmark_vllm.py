"""Real vLLM requests: native prefill + native/FlagMega decode + native sampler.

Run each variant in a separate process, on the same otherwise idle GPU.
Measurements exclude engine construction, compilation and warmup requests.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import time
from datetime import datetime, timezone


def summarize(values):
    ordered = sorted(values)
    return {"median": statistics.median(ordered), "mean": statistics.mean(ordered),
            "p95": ordered[min(len(ordered) - 1, int(len(ordered) * .95))], "count": len(ordered)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--artifact", type=Path)
    parser.add_argument("--label", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prompt-lengths", nargs="+", type=int, default=[32, 128, 1024, 2048])
    parser.add_argument("--decode-tokens", type=int, default=64)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--validate-native", action="store_true", help="Require native greedy tokens; diagnose logits, no performance reporting")
    parser.add_argument("--compilation-level", type=int, choices=(0, 3), default=3)
    args = parser.parse_args()
    if args.output.exists() or args.output.with_suffix(".failed.json").exists():
        raise ValueError("Choose a new result file; keep raw trials immutable")
    if args.decode_tokens < 2 or args.repeats < 1 or args.warmups < 0:
        raise ValueError("Need >=2 output tokens, >=1 repeat and >=0 warmups")
    if min(args.prompt_lengths) < 2 or max(args.prompt_lengths) + args.decode_tokens > 3072:
        raise ValueError("Prompt lengths >=2 and prompt+decode <=3072 required")
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
    os.environ["VLLM_USE_V1"] = "1"
    os.environ["VLLM_ATTENTION_BACKEND"] = "FLASH_ATTN"
    if args.artifact:
        os.environ["FLAGMEGA_TUTORIAL_ARTIFACT"] = str(args.artifact.resolve())
    else:
        os.environ.pop("FLAGMEGA_TUTORIAL_ARTIFACT", None)
    if args.validate_native and not args.artifact:
        raise ValueError("--validate-native requires a FlagMega artifact")
    os.environ["FLAGMEGA_TUTORIAL_VALIDATE_NATIVE"] = "1" if args.validate_native else "0"
    os.environ["FLAGMEGA_TUTORIAL_FAILURE_REPORT"] = str(args.output.with_suffix(".failed.json"))

    import torch
    import triton
    import vllm
    from vllm import EngineArgs, SamplingParams
    from vllm.v1.engine.llm_engine import LLMEngine

    torch.set_num_threads(1)
    engine_args = EngineArgs(
        model=str(args.checkpoint), dtype="bfloat16", tensor_parallel_size=1,
        max_model_len=3072, max_num_seqs=1, max_num_batched_tokens=3072,
        block_size=256, num_gpu_blocks_override=16, gpu_memory_utilization=.3,
        enable_prefix_caching=False, enable_chunked_prefill=False,
        disable_log_stats=True, worker_cls="vllm_adapter.TutorialWorker",
        compilation_config={"level": args.compilation_level, "cudagraph_mode": "FULL", "cudagraph_capture_sizes": [1]},
        seed=0,
    )
    engine = LLMEngine.from_engine_args(engine_args)
    tokenizer = engine.tokenizer
    source = tokenizer.encode("Explain why the sky is blue, and how sunlight interacts with air. ",
                              add_special_tokens=False)
    sampling = SamplingParams(temperature=0, max_tokens=args.decode_tokens, ignore_eos=True)
    counter = 0

    def request(length):
        nonlocal counter
        counter += 1
        prompt = (source * ((length + len(source) - 1) // len(source)))[:length]
        torch.cuda.synchronize()
        started = time.perf_counter()
        engine.add_request(str(counter), {"prompt_token_ids": prompt}, sampling)
        steps, output = [], None
        previous = started
        while engine.has_unfinished_requests():
            result = engine.step()
            now = time.perf_counter()
            if result:
                if len(result) != 1:
                    raise RuntimeError("Unexpected concurrent request output")
                output = result[0]
                steps.append((now - previous) * 1000)
                previous = now
        elapsed = (time.perf_counter() - started) * 1000
        if output is None or len(output.outputs[0].token_ids) != args.decode_tokens or len(steps) != args.decode_tokens:
            raise RuntimeError("Expected exactly one output token per step and no early EOS")
        return {"prompt_tokens": prompt, "token_ids": list(output.outputs[0].token_ids),
                "ttft_ms": steps[0], "decode_step_ms": steps[1:], "e2e_ms": elapsed,
                "output_tokens_per_second": args.decode_tokens * 1000 / elapsed}

    scenarios = []
    for length in args.prompt_lengths:
        for _ in range(args.warmups):
            request(length)
        runs = [request(length) for _ in range(args.repeats)]
        scenarios.append({"prompt_length": length, "decode_tokens": args.decode_tokens, "runs": runs,
                          "decode_ms": summarize([x for run in runs for x in run["decode_step_ms"]]),
                          "ttft_ms": summarize([run["ttft_ms"] for run in runs]),
                          "e2e_ms": summarize([run["e2e_ms"] for run in runs])})
        print(json.dumps({"label": args.label, "prompt_length": length,
                          "decode_ms": scenarios[-1]["decode_ms"]}), flush=True)
    driver = subprocess.check_output(["nvidia-smi", "--query-gpu=name,uuid,driver_version", "--format=csv,noheader"], text=True)
    report = {"label": args.label, "boundary": "synchronous vLLM engine.step, scheduler+forward+sampler+output processing",
              "batch": 1, "concurrency": 1,
              "cuda_graph": f"FULL, capture_sizes=[1], level={args.compilation_level}",
              "prefix_caching": False, "chunked_prefill": engine.vllm_config.scheduler_config.enable_chunked_prefill,
              "native_prefill": True, "performance_valid": not args.validate_native,
              "torch": torch.__version__, "triton": triton.__version__, "vllm": vllm.__version__,
              "vllm_commit": subprocess.check_output(
                  ["git", "-C", str(Path(vllm.__file__).resolve().parents[1]), "rev-parse", "HEAD"], text=True).strip(),
              "gpu_inventory": driver, "visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
              "checkpoint_revision": args.checkpoint.name, "warmups_per_scenario": args.warmups,
              "timestamp_utc": datetime.now(timezone.utc).isoformat(),
              "flagtree_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
              "artifact_source_sha256": (hashlib.sha256((args.artifact / "generated_kernels.py").read_bytes()).hexdigest()
                                         if args.artifact else None),
              "executor_stats": engine.collective_rpc("tutorial_stats"), "scenarios": scenarios}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    engine.engine_core.shutdown()
    from vllm.distributed.parallel_state import destroy_model_parallel, destroy_distributed_environment
    destroy_model_parallel()
    destroy_distributed_environment()


if __name__ == "__main__":
    main()
