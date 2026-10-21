"""Independent native FlagOS token-scan reference for standalone chat CLI.

This is a correctness run. Single-token prefill and batched prefill use
different matrix/attention kernels and need not generate identical BF16 tokens.
Published native FULL-graph performance still uses batched prefill.
"""

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path

from benchmark_flagos import sha256, source_identity


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prompt-lengths", type=int, nargs="+", default=[32, 128, 1024, 2048])
    parser.add_argument("--decode-tokens", type=int, default=64)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Use a fresh output file")
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
    os.environ["CUDA_MODULE_LOADING"] = "0"
    import torch
    import triton
    import vllm
    import flag_gems
    import vllm_fl
    from triton._C import libtriton
    from vllm import LLM, SamplingParams

    torch.set_num_threads(1)
    checkpoint = args.checkpoint.resolve()
    llm = LLM(model=str(checkpoint), dtype="bfloat16", max_model_len=3072, max_num_seqs=1,
              max_num_batched_tokens=1, gpu_memory_utilization=.35, enforce_eager=True,
              compilation_config={"mode": 0, "cudagraph_mode": "NONE"},
              block_size=256, num_gpu_blocks_override=16, enable_prefix_caching=False,
              enable_chunked_prefill=True, disable_log_stats=True, seed=0)
    report = {"schema": "flagmega.native-token-scan-reference/v1", "accepted": False,
              "performance_valid": False, "prefill_mode": "native_token_scan",
              "max_num_batched_tokens": 1, "dtype": "bfloat16", "batch": 1, "concurrency": 1,
              "block_size": 256, "num_blocks": 16, "prefix_caching": False,
              "decode_tokens": args.decode_tokens, "checkpoint_revision": checkpoint.name,
              "checkpoint_files": {p.name: sha256(p) for p in sorted(checkpoint.glob("*.safetensors"))},
              "torch": torch.__version__, "triton": triton.__version__, "vllm": vllm.__version__,
              "device": torch.cuda.get_device_name(), "visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
              "compiler_sha256": sha256(libtriton.__file__), "script_sha256": sha256(__file__),
              "vllm_source": source_identity(vllm.__file__), "flaggems_source": source_identity(flag_gems.__file__),
              "plugin_source": source_identity(vllm_fl.__file__),
              "timestamp_utc": datetime.now(timezone.utc).isoformat(), "scenarios": []}
    report["checkpoint_files"]["config.json"] = sha256(checkpoint / "config.json")
    try:
        tokenizer = llm.get_tokenizer()
        source = tokenizer.encode("Explain why the sky is blue, and how sunlight interacts with air. ",
                                  add_special_tokens=False)
        params = SamplingParams(temperature=0, max_tokens=args.decode_tokens, ignore_eos=True)
        for length in args.prompt_lengths:
            prompt = (source * ((length + len(source) - 1) // len(source)))[:length]
            output = llm.generate([{"prompt_token_ids": prompt}], params, use_tqdm=False)[0].outputs[0]
            if len(output.token_ids) != args.decode_tokens:
                raise ValueError("Incomplete native token-scan generation")
            report["scenarios"].append({"prompt_length": length, "decode_tokens": args.decode_tokens,
                                        "runs": [{"prompt_tokens": prompt, "token_ids": output.token_ids,
                                                  "text": output.text}]})
            print(json.dumps({"prompt_length": length, "token_ids": output.token_ids}), flush=True)
        report["accepted"] = True
    finally:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        llm.llm_engine.engine_core.shutdown()


if __name__ == "__main__":
    main()
