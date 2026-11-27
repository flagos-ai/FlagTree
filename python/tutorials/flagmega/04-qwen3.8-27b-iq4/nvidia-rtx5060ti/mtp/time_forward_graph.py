"""Measure one forward pass with CUDA graph replay.

The signed r031 baseline (35.3155 ms) was measured with CUDA graph replay; llama.cpp is a
C++ program with no host dispatch cost.  Timing this engine with per-program Python launches
therefore charges it 7-12 ms of host overhead that neither comparison point pays.  This script
captures the whole prepared call into a graph and replays it, which is the fair protocol.

Usage: time_forward_graph.py ARTIFACT TOKENS [ITERATIONS]
"""
import json
import statistics
import sys
from pathlib import Path

import torch

from triton.flagmega.ir import VectorType
from triton.flagmega.runtime import load


def build(artifact, device='cuda:0'):
    runtime = load(Path(artifact).resolve(), device=device)
    from r054_runtime import wrap
    runtime = wrap(runtime)
    values = {}
    for argument in runtime.external_arguments:
        b = runtime.buffer_plan.buffer_map[argument['buffer']]
        lanes = b.dtype.lanes if isinstance(b.dtype, VectorType) else ()
        scalar = b.dtype.elem_type if lanes else b.dtype
        values[b.id] = torch.zeros((*b.shape, *lanes), dtype=getattr(torch, scalar.value), device=device)
    values['paged_state.block_table'].copy_(torch.arange(
        values['paged_state.block_table'].numel(),
        dtype=values['paged_state.block_table'].dtype,
        device=device).reshape_as(values['paged_state.block_table']))
    logits_key = next(name for name in values if name.startswith('logits'))
    return runtime, values, logits_key


def main():
    artifact = sys.argv[1]
    tokens = int(sys.argv[2])
    iterations = int(sys.argv[3]) if len(sys.argv) > 3 else 25

    torch.cuda.init()
    runtime, values, logits_key = build(artifact)
    values['input_ids'].copy_(torch.tensor(
        [1000 + i for i in range(tokens)], dtype=values['input_ids'].dtype, device='cuda'))
    args = tuple(values[x['buffer']] for x in runtime.external_arguments)

    def reset():
        values['paged_state.seq_lens'].fill_(0)
        values['paged_state.slot_mapping'].fill_(0)
        values['paged_state.query_start_loc'].copy_(
            torch.tensor([0, tokens], dtype=torch.int32, device='cuda'))

    # A prepared kernel with compiler global scratch binds to the stream of its
    # first launch, so prepare, warmup and capture must all use one stream.
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        runtime.prepare(*args)
    stream.synchronize()

    for _ in range(3):
        reset()
        runtime.run_into(*args, stream=stream.cuda_stream)
    stream.synchronize()

    graph = torch.cuda.CUDAGraph()
    reset()
    try:
        with torch.cuda.graph(graph, stream=stream):
            runtime.run_into(*args, stream=stream.cuda_stream)
    except Exception as error:  # noqa: BLE001 - report the capture failure verbatim
        print(json.dumps({'artifact': artifact, 'tokens': tokens,
                          'capture_failed': True, 'error': str(error)[:600]}, indent=1))
        return
    stream.synchronize()

    samples = []
    for _ in range(iterations):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        graph.replay()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end))

    logits = values[logits_key]
    print(json.dumps({
        'artifact': artifact,
        'tokens': tokens,
        'protocol': 'cuda-graph-replay',
        'iterations': iterations,
        'ms_median': round(statistics.median(samples), 4),
        'ms_mean': round(statistics.fmean(samples), 4),
        'ms_min': round(min(samples), 4),
        'ms_max': round(max(samples), 4),
        'logits_rows_finite': [bool(torch.isfinite(logits[i]).all()) for i in range(logits.shape[0])],
        'samples': [round(v, 3) for v in samples],
    }, indent=1))


if __name__ == '__main__':
    main()
