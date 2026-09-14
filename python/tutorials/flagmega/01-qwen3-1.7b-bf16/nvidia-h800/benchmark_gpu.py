"""Strict-cold full-model CUDA graph timing, separate from vLLM request latency.

Zero-cache/fixed-token inputs isolate performance, not numerical acceptance.
For a logits-only artifact --include-argmax adds torch argmax + int32 copy inside
the graph. This extra launch overhead is reported; it cannot make the candidate
artificially faster than a legacy artifact with sampling already included.
"""

import argparse
import hashlib
import json
from pathlib import Path
import statistics

import torch

from triton.flagmega.runtime import load


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifact", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--contexts", nargs="+", type=int, default=[1, 128, 1024])
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--flush-mib", type=int, default=256)
    parser.add_argument("--include-argmax", action="store_true")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if args.output.exists() or args.samples < 1 or args.flush_mib < 1:
        raise ValueError("Use a fresh output, positive sample count and cache flush size")
    torch.set_num_threads(1)
    runtime = load(args.artifact, device=args.device)
    device = torch.device(args.device)
    tokens = torch.tensor([151644], dtype=torch.int32, device=device)
    outputs = runtime.create_outputs()
    if runtime.result_kind == "logits_token_state":
        logits, next_token = outputs
        sampling = "artifact"

        def prepare(state):
            runtime.prepare(tokens, state, logits=logits, next_token=next_token)

        def launch(state):
            runtime.run_into(logits, next_token, tokens, state)
    elif runtime.result_kind == "tensor_state":
        logits = outputs
        next_token = torch.empty((1,), dtype=torch.int32, device=device)
        sampling = "torch_argmax_and_int32_copy" if args.include_argmax else "none"

        def prepare(state):
            runtime.prepare(tokens, state, output=logits)

        def launch(state):
            runtime.run_into(logits, tokens, state)
            if args.include_argmax:
                next_token.copy_(logits.argmax(dim=-1))
    else:
        raise ValueError(f"Unsupported public ABI: {runtime.result_kind}")
    flush = torch.randn((args.flush_mib * 1024 * 1024 // 4,), device=device)
    flush_out = torch.empty((), device=device)
    scenarios = []
    with torch.no_grad():
        for context in args.contexts:
            state = runtime.create_state()
            state.seq_lens.fill_(context - 1)
            prepare(state)
            launch(state)
            torch.cuda.synchronize(device)
            # Prepared scratch belongs to one stream. Allocate a fresh binding
            # for the capture stream after the eager compilation launch.
            prepare(state)
            state.seq_lens.fill_(context - 1)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                launch(state)
            for _ in range(5):
                state.seq_lens.fill_(context - 1)
                torch.sum(flush, dim=0, out=flush_out)
                graph.replay()
            torch.cuda.synchronize(device)
            starts = [torch.cuda.Event(enable_timing=True) for _ in range(args.samples)]
            ends = [torch.cuda.Event(enable_timing=True) for _ in starts]
            for start, end in zip(starts, ends, strict=True):
                state.seq_lens.fill_(context - 1)
                torch.sum(flush, dim=0, out=flush_out)
                start.record()
                graph.replay()
                end.record()
            torch.cuda.synchronize(device)
            samples = [a.elapsed_time(b) for a, b in zip(starts, ends, strict=True)]
            scenarios.append({"context": context, "samples_ms": samples,
                              "median_ms": statistics.median(samples),
                              "p95_ms": sorted(samples)[min(len(samples)-1, int(.95*len(samples)))]})
            print(json.dumps({key: value for key, value in scenarios[-1].items() if key != "samples_ms"}), flush=True)
    report = {"artifact": str(args.artifact.resolve()), "boundary": "full model GPU CUDA graph replay",
              "sampling": sampling, "numerical_contract": runtime.ir_module.metadata.get("numerical_contract", "nncase"),
              "num_layers": runtime.state_config.num_layers, "batch": 1, "flush_mib": args.flush_mib,
              "gpu": torch.cuda.get_device_name(device), "device": str(device), "torch": torch.__version__,
              "resource": runtime.resource_report, "scenarios": scenarios,
              "source_sha256": hashlib.sha256((args.artifact / "generated_kernels.py").read_bytes()).hexdigest()}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
