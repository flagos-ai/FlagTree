"""Reproduce the isolated long-context attention layout bottleneck on H800.

CUDA-event diagnostics are separate from the complete-request figure data.
"""

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import statistics
import sys

import torch

from agent_optimizations.shared_router import Registry
from native_prefill_benchmark import _ThermalMonitor
from triton.flagmega.codegen.triton.templates import KernelTemplateSpec, TritonTemplateRegistry


HEADER = "import triton\nimport triton.language as tl\nimport triton.experimental.tle.language as tle\n"
ENTRY = '''
@triton.jit
def entry(query, cache, slot, table, maximum, summation, accumulator):
    owner = tl.program_id(0)
    head, shard = owner % 16, owner // 16
    _flagmega_paged_attention_partial(query + head * 256, cache, slot, table, 0,
        maximum + owner, summation + owner, accumulator + owner * 256,
        shard, head, 0, 10, 16, 2, 256, 256, .0625, 32, 8, 8, 8, 1, 16)
'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trial", type=Path, required=True)
    parser.add_argument("--contexts", nargs="+", type=int, default=[128, 1024, 2048, 3072])
    args = parser.parse_args()
    if not args.contexts or any(not 0 < n <= 4096 for n in args.contexts):
        raise ValueError("Context length must be within the 16-page cache")
    args.trial.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    torch.manual_seed(831)
    monitor = _ThermalMonitor(.5)
    query = torch.randn((16, 256), device="cuda", dtype=torch.bfloat16) * .5
    cache = torch.randn((16, 10, 2, 256, 2, 256), device="cuda", dtype=torch.bfloat16) * .5
    table = torch.randperm(16, device="cuda", dtype=torch.int32)
    slot = torch.zeros((), device="cuda", dtype=torch.int32)
    maximum, summation = (torch.empty(128, device="cuda") for _ in range(2))
    accumulator = torch.empty((128, 256), device="cuda")
    arguments = query, cache, slot, table, maximum, summation, accumulator
    original = KernelTemplateSpec("paged_attention_partial", "decode_t32", "nvidia", "sm90")
    optimized = KernelTemplateSpec("paged_attention_partial", "decode_t64", "nvidia", "sm90")
    sources = {"original": TritonTemplateRegistry().render_kernel(original, {}).source,
               "agent": Registry().render_kernel(optimized, {}).source}
    report = dict(schema="flagmega.attention-layout-diagnostic/v1", performance_valid=False,
                  scope="isolated CUDA-event kernel timing; excluded from request figures",
                  torch=torch.__version__, thermal_monitor=monitor.info(), records=[])
    for label, source in sources.items():
        path = args.trial / (label + ".py")
        entry = ENTRY if label == "original" else ENTRY.replace(
            "_flagmega_paged_attention_partial(", "_flagmega_paged_attention_decode_dispatch(").replace(
            ".0625, 32, 8", ".0625, 64, 8")
        path.write_text(HEADER + source + entry)
        module_spec = importlib.util.spec_from_file_location("attention_diagnostic_" + label, path)
        module = importlib.util.module_from_spec(module_spec)
        sys.modules[module_spec.name] = module
        module_spec.loader.exec_module(module)
        kernel = module.entry[(128,)](*arguments, num_warps=8)
        assert kernel.n_spills == 0
        for _ in range(5):
            module.entry[(128,)](*arguments, num_warps=8)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            for _ in range(10):
                module.entry[(128,)](*arguments, num_warps=8)
        for length in args.contexts:
            slot.fill_(length - 1)
            graph.replay()
            factors = torch.exp(maximum.reshape(8, 16) - maximum.reshape(8, 16).max(0).values)
            output = (accumulator.reshape(8, 16, 256) * factors[:, :, None]).sum(0)
            output /= (summation.reshape(8, 16) * factors).sum(0)[:, None]
            positions = torch.arange(length, device="cuda")
            key = cache[table[positions // 256].long(), 0, 0, positions % 256].repeat_interleave(8, 1)
            value = cache[table[positions // 256].long(), 0, 1, positions % 256].repeat_interleave(8, 1)
            scores = torch.einsum("hd,thd->ht", query.float(), key.float()) * .0625
            expected = torch.einsum("ht,thd->hd", scores.softmax(1), value.float())
            torch.testing.assert_close(output, expected, rtol=.005, atol=.001)
            timings = []
            for _ in range(5):
                before = monitor.before()
                start, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
                start.record()
                graph.replay()
                end.record()
                end.synchronize()
                monitor.after(before)
                timings.append(start.elapsed_time(end) * 100)
            row = dict(label=label, context_length=length, median_us=statistics.median(timings),
                       samples_us=timings, registers=kernel.n_regs, spills=kernel.n_spills,
                       shared_bytes=kernel.metadata.shared, max_abs=float((output - expected).abs().max()),
                       source_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
            report["records"].append(row)
            print(json.dumps(row), flush=True)
    (args.trial / "report.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
