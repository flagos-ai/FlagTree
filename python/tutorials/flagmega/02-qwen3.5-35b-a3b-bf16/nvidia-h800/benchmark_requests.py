"""01-aligned raw-token requests, with complete prefill and 63 decode intervals."""

import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import statistics
import sys
from time import perf_counter


PROMPT_LENGTHS = [32, 128, 1024, 2048]
PROMPT = "Explain why the sky is blue, and how sunlight interacts with air. "


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--artifact", type=Path)
    parser.add_argument("--runtime", choices=("vllm", "chat_cli"), required=True)
    parser.add_argument("--label", choices=("native", "baseline", "agent", "chat_cli"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--round", type=int, default=0)
    parser.add_argument("--rounds", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--prompt-lengths", type=int, nargs="+", default=PROMPT_LENGTHS)
    parser.add_argument("--decode-tokens", type=int, default=64)
    parser.add_argument("--eager", action="store_true", help="Uncaptured compiled diagnostic; not an independent eager reference")
    parser.add_argument("--eager-reference", type=Path, help="Independent IR/PyTorch eager request report")
    parser.add_argument("--native-reference", type=Path, help="Independent native requests for recording output differences")
    parser.add_argument("--native-prefill-reference", type=Path,
                        help="Native prefill snapshots; compare only the following 63 decode intervals")
    parser.add_argument("--inter-request-pause", type=float, default=.5,
                        help="Untimed minimum idle seconds between native-prefilled requests")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Use a new immutable result file")
    if args.repeats < 1 or args.rounds < 1 or args.warmups < 0 or args.decode_tokens < 2:
        parser.error("Invalid request counts")
    if min(args.prompt_lengths) < 2 or max(args.prompt_lengths) + args.decode_tokens > 3072:
        parser.error("Prompt plus output must fit 3072 tokens")
    if (args.label == "native") != (args.artifact is None):
        parser.error("Native has no artifact; all FlagMega variants require one")
    if args.native_prefill_reference:
        from native_prefill_benchmark import run
        return run(args)
    import torch
    from transformers import AutoTokenizer
    torch.set_num_threads(1)
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint, local_files_only=True, trust_remote_code=False)
    source = tokenizer.encode(PROMPT, add_special_tokens=False)
    if args.runtime == "chat_cli":
        from phase_backend import PhaseBundleBackend
        from triton.flagmega.serving import SamplingConfig, TextGenerationEngine
        from triton.flagmega.serving.chat import TextStream
        sampling = SamplingConfig()
        backend = PhaseBundleBackend(args.artifact, sampling=sampling, cuda_graph=not args.eager)
        engine = TextGenerationEngine(backend)

        def request(length):
            engine.reset()
            backend.synchronize()
            prompt = (source * ((length + len(source) - 1) // len(source)))[:length]
            stream = TextStream(tokenizer, lambda text: None)
            result = engine.generate(prompt, max_new_tokens=args.decode_tokens, ignore_eos=True, on_token=stream.put)
            stream.finish()
            return {**result.to_dict(), "text": stream.text}

        metadata = {"backend": backend.info(), "sampling": asdict(sampling), "vllm": None,
                    "boundary": "synchronous standalone generation + sampling + host delivery + text decoding; terminal I/O excluded",
                    "prefill_mode": "compiled_batched", "cuda_graph": not args.eager,
                    "eager_reference_kind": "uncaptured_compiled" if args.eager else None}
        close = backend.close
    else:
        os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
        import vllm
        from vllm import EngineArgs, SamplingParams
        from vllm.v1.engine.llm_engine import LLMEngine
        options = {}
        if args.artifact:
            # Load the tutorial's TLE-enabled Triton only for artifact execution.
            # The reference process uses the pinned vLLM wheel's Triton.
            from vllm import ModelRegistry
            from vllm_adapter import FlagMegaModel
            ModelRegistry.register_model("FlagMegaQwen35ForCausalLM", FlagMegaModel)
            os.environ["FLAGMEGA_REQUEST_ARTIFACT"] = str(args.artifact.resolve())
            def overrides(config):
                config.architectures = ["FlagMegaQwen35ForCausalLM"]
                if hasattr(config, "text_config"):
                    config.text_config.architectures = config.architectures
                return config
            options.update(hf_overrides=overrides, enforce_eager=True)
        engine_args = EngineArgs(model=str(args.checkpoint.resolve()), dtype="bfloat16", tensor_parallel_size=1,
                                max_model_len=3072, max_num_seqs=1, max_num_batched_tokens=3072,
                                block_size=256, num_gpu_blocks_override=16, gpu_memory_utilization=.95,
                                enable_prefix_caching=False, enable_chunked_prefill=False, language_model_only=True,
                                mamba_ssm_cache_dtype="float32", disable_log_stats=True, seed=0,
                                compilation_config={"mode": 3, "custom_ops": ["none"], "cudagraph_capture_sizes": [1]},
                                **options)
        engine = LLMEngine.from_engine_args(engine_args)
        sampling = SamplingParams(temperature=0, max_tokens=args.decode_tokens, ignore_eos=True)
        counter = 0

        def request(length):
            nonlocal counter
            counter += 1
            prompt = (source * ((length + len(source) - 1) // len(source)))[:length]
            torch.cuda.synchronize()
            started = previous = perf_counter()
            engine.add_request(str(counter), {"prompt_token_ids": prompt}, sampling)
            output, steps = None, []
            while engine.has_unfinished_requests():
                results = engine.step()
                now = perf_counter()
                if results:
                    if len(results) != 1:
                        raise RuntimeError("Expected batch/concurrency one")
                    output = results[0]
                    steps.append((now - previous) * 1000)
                    previous = now
            elapsed = (perf_counter() - started) * 1000
            if output is None or len(output.outputs[0].token_ids) != args.decode_tokens or len(steps) != args.decode_tokens:
                raise RuntimeError("Expected exactly one output token per step")
            return {"prompt_tokens": prompt, "token_ids": list(output.outputs[0].token_ids),
                    "ttft_ms": steps[0], "decode_step_ms": steps[1:], "e2e_ms": elapsed,
                    "output_tokens_per_second": args.decode_tokens * 1000 / elapsed}

        metadata = {"vllm": vllm.__version__, "cuda_graph": str(engine.vllm_config.compilation_config.cudagraph_mode),
                    "boundary": "synchronous vLLM engine.step, scheduler + forward + sampler + output processing",
                    "prefill_mode": "compiled_batched" if args.artifact else "native_batched",
                    "engine_config": str(engine.vllm_config)}
        if args.artifact:
            runner = engine.engine_core.engine_core.model_executor.driver_worker.worker.model_runner
            metadata["backend"] = runner.model.backend.info()
        close = engine.engine_core.shutdown
    args.output.parent.mkdir(parents=True, exist_ok=True)
    try:
        references = {}
        for kind, path in (("eager", args.eager_reference), ("native", args.native_reference)):
            if path:
                reference = json.loads(path.read_text())
                if kind == "eager" and (reference.get("eager_reference_kind") != "ir_torch"
                                        or reference.get("performance_valid") is not False
                                        or reference.get("schema") != "flagmega.ir-eager-request-reference/v1"):
                    raise ValueError("Eager acceptance requires an untimed independent IR/PyTorch reference")
                if kind == "eager":
                    imported = metadata["backend"].get("imported_semantic_hashes")
                    if not imported or imported != reference.get("imported_semantic_hashes"):
                        raise ValueError("Eager reference and compiled phases have different imported IR")
                    if reference.get("checkpoint_config_sha256") != hashlib.sha256((args.checkpoint / "config.json").read_bytes()).hexdigest():
                        raise ValueError("Eager reference checkpoint differs")
                references[kind] = {s["prompt_length"]: s["runs"][0] for s in reference["scenarios"]}
        import triton
        if args.runtime == "chat_cli" and any(n == "vllm" or n.startswith("vllm.") for n in sys.modules):
            raise RuntimeError("Standalone execution imported vLLM")
        common = {"schema": "flagmega.request-comparison-trial/v1", "label": args.label, "runtime": args.runtime,
                  "round": args.round, "batch": 1, "concurrency": 1, "prefix_caching": False,
                  "checkpoint_revision": "59d61f3ce65a6d9863b86d2e96597125219dc754",
                  "checkpoint_config_sha256": hashlib.sha256((args.checkpoint / "config.json").read_bytes()).hexdigest(),
                  "visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"), "device": torch.cuda.get_device_name(),
                  "torch": torch.__version__, "triton": triton.__version__, "warmups_per_scenario": args.warmups,
                  "performance_valid": not args.eager, "vllm_imported": args.runtime == "vllm",
                  "benchmark_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  **metadata}
        if args.eager_reference:
            common["eager_reference_kind"] = "ir_torch"
        for round_offset in range(args.rounds):
            round_id = args.round + round_offset
            scenarios = []
            lengths = args.prompt_lengths[round_offset % len(args.prompt_lengths):] + args.prompt_lengths[:round_offset % len(args.prompt_lengths)]
            for length in lengths:
                for _ in range(args.warmups):
                    request(length)
                runs = [request(length) for _ in range(args.repeats)]
                if any(run["token_ids"] != runs[0]["token_ids"] for run in runs):
                    raise RuntimeError("Independent repeated requests changed token sequences")
                for kind, expected in references.items():
                    target = expected[length]
                    for run in runs:
                        if run["prompt_tokens"] != target["prompt_tokens"]:
                            raise RuntimeError("Reference prompt differs")
                        actual = run["token_ids"]
                        wanted = target["token_ids"]
                        if kind == "eager" and actual != wanted:
                            raise RuntimeError(f"Independent {kind} acceptance failed at prompt {length}")
                scenarios.append({"prompt_length": length, "decode_tokens": args.decode_tokens, "runs": runs,
                                  "independent_eager_token_ids": references.get("eager", {}).get(length, {}).get("token_ids"),
                                  "native_sequence_match": (runs[0]["token_ids"] == references["native"][length]["token_ids"])
                                  if "native" in references else None})
                print(json.dumps({"label": args.label, "round": round_id, "prompt_length": length,
                                  "median_ms": statistics.median(x for r in runs for x in r["decode_step_ms"]),
                                  "token_ids": runs[0]["token_ids"]}), flush=True)
            output = args.output if args.rounds == 1 else args.output.with_name(f"round{round_id:02d}.{args.label}.json")
            if output.exists():
                raise ValueError("Result file already exists")
            report = {**common, "round": round_id, "scenarios": sorted(scenarios, key=lambda s: s["prompt_length"])}
            output.write_text(json.dumps(report, indent=2) + "\n")
    except BaseException as error:
        args.output.with_suffix(".failed.json").write_text(json.dumps({"accepted": False, "error": str(error)}, indent=2) + "\n")
        raise
    finally:
        close()


if __name__ == "__main__":
    main()
