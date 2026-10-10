"""Measure the standalone chat CLI engine with real artifact-only prefill.

Uses the same raw token prompts as benchmark_vllm.py, but imports no vLLM.
Text decoding is timed; terminal writes are replaced with a sink. No cache
flush is inserted into request timing, and no prefix is reused between runs.
"""

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys

from accuracy import require_same_tokens
from triton.flagmega.serving import SamplingConfig, TextGenerationEngine
from triton.flagmega.serving.backend import ArtifactBackend
from triton.flagmega.serving.chat import TextStream, load_tokenizer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--native-reference", type=Path, help="Optional native raw report for exact sequence verification; no vLLM execution")
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--prompt-lengths", type=int, nargs="+", default=[32, 128, 1024, 2048])
    parser.add_argument("--decode-tokens", type=int, default=64)
    parser.add_argument("--cuda-graph", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    if args.rounds < 1 or args.repeats < 1 or args.warmups < 0 or args.decode_tokens < 2:
        parser.error("Need positive rounds/repeats, nonnegative warmups, and at least two output tokens")
    if len(set(args.prompt_lengths)) != len(args.prompt_lengths) or min(args.prompt_lengths) < 1:
        parser.error("Prompt lengths must be distinct positive integers")
    args.results.mkdir(parents=True, exist_ok=False)

    import torch
    import triton
    torch.set_num_threads(1)
    sampling = SamplingConfig()
    backend = ArtifactBackend.load(args.artifact, sampling=sampling, cuda_graph=args.cuda_graph)
    try:
        tokenizer, eos = load_tokenizer(args.checkpoint, backend)
        engine = TextGenerationEngine(backend, eos_token_ids=eos)
        source = tokenizer.encode("Explain why the sky is blue, and how sunlight interacts with air. ",
                                  add_special_tokens=False)
        reference = (json.loads(args.native_reference.read_text()) if args.native_reference else None)
        expected = ({s["prompt_length"]: s["runs"][0] for s in reference["scenarios"]}
                    if reference else None)
        if reference and reference["checkpoint_revision"] != args.checkpoint.name:
            raise ValueError("Reference checkpoint revision differs")
        reference_ids = {}

        def request(length):
            engine.reset()
            backend.synchronize()
            prompt = (source * ((length + len(source) - 1) // len(source)))[:length]
            stream = TextStream(tokenizer, lambda text: None)
            result = engine.generate(prompt, max_new_tokens=args.decode_tokens, ignore_eos=True,
                                     on_token=stream.put)
            stream.finish()
            if expected is not None:
                native = expected[length]
                if prompt != native["prompt_tokens"]:
                    raise ValueError("Native and standalone prompts differ")
                require_same_tokens(result.token_ids, native["token_ids"],
                                    context=f"standalone compiled prefill+decode, prompt {length}")
            if length in reference_ids:
                require_same_tokens(result.token_ids, reference_ids[length], context="repeated independent request")
            reference_ids[length] = result.token_ids
            return {**result.to_dict(), "text": stream.text}

        for length in args.prompt_lengths:
            for _ in range(args.warmups):
                request(length)
        for round_id in range(args.rounds):
            order = args.prompt_lengths[round_id % len(args.prompt_lengths):] + args.prompt_lengths[:round_id % len(args.prompt_lengths)]
            scenarios = []
            for length in order:
                runs = [request(length) for _ in range(args.repeats)]
                scenarios.append({"prompt_length": length, "decode_tokens": args.decode_tokens, "runs": runs})
                print(json.dumps({"round": round_id, "prompt_length": length,
                                  "median_ms": runs[-1]["decode_latency_ms"]["median"],
                                  "ttft_ms": runs[-1]["ttft_ms"]}), flush=True)
            if any(name == "vllm" or name.startswith("vllm.") for name in sys.modules):
                raise RuntimeError("Standalone benchmark unexpectedly imported vLLM")
            report = {"schema": "flagmega.chat-benchmark/v1", "label": "chat_cli",
                      "boundary": "synchronous standalone token generation + sampling + host delivery + text decoding; no terminal I/O",
                      "batch": 1, "concurrency": 1, "prefill_mode": "compiled_token_scan", "prefix_caching": False,
                      "sampling": asdict(sampling), "checkpoint_revision": args.checkpoint.name,
                      "visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"), "torch": torch.__version__,
                      "triton": triton.__version__, "round": round_id,
                      "native_token_agreement": True if reference else None,
                      "performance_valid": True, "vllm_imported": False,
                      "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                      "warmups_per_scenario": args.warmups, "backend": backend.info(), "scenarios": scenarios}
            (args.results / f"round{round_id:02d}.chat_cli.json").write_text(json.dumps(report, indent=2) + "\n")
    except BaseException as error:
        (args.results / "failure.json").write_text(json.dumps({"accepted": False, "error": str(error)}, indent=2) + "\n")
        raise
    finally:
        backend.close()


if __name__ == "__main__":
    main()
