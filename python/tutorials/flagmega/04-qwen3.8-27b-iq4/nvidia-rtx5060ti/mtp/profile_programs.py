"""Per-program GPU time profile of one forward pass.

Times each prepared program with CUDA events on one stream and aggregates by
entry symbol, so the report answers "which program kinds dominate the call".

Usage: profile_programs.py ARTIFACT TOKENS
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

import torch

from triton.flagmega.ir import VectorType
from triton.flagmega.runtime import load


def main():
    artifact, tokens = sys.argv[1], int(sys.argv[2])
    runtime = load(Path(artifact).resolve(), device='cuda:0')
    stream = torch.cuda.Stream()
    torch.cuda.set_stream(stream)
    values = {}
    for argument in runtime.external_arguments:
        b = runtime.buffer_plan.buffer_map[argument['buffer']]
        lanes = b.dtype.lanes if isinstance(b.dtype, VectorType) else ()
        scalar = b.dtype.elem_type if lanes else b.dtype
        values[b.id] = torch.zeros((*b.shape, *lanes), dtype=getattr(torch, scalar.value), device='cuda')
    values['paged_state.block_table'].copy_(torch.arange(
        values['paged_state.block_table'].numel(),
        dtype=values['paged_state.block_table'].dtype, device='cuda'
    ).reshape_as(values['paged_state.block_table']))
    values['input_ids'].copy_(torch.tensor(
        [1000 + i for i in range(tokens)], dtype=values['input_ids'].dtype, device='cuda'))
    args = tuple(values[x['buffer']] for x in runtime.external_arguments)
    runtime.prepare(*args)
    stream.synchronize()

    specs = runtime._kernel_launch_specs()

    def reset():
        values['paged_state.seq_lens'].fill_(0)
        values['paged_state.slot_mapping'].fill_(0)
        values['paged_state.query_start_loc'].copy_(
            torch.tensor([0, tokens], dtype=torch.int32, device='cuda'))

    reset()
    runtime.run_into(*args, stream=stream.cuda_stream)
    stream.synchronize()

    totals = defaultdict(float)
    counts = defaultdict(int)
    starts, ends = [], []
    for program in runtime._prepared:
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        starts.append(s)
        ends.append(e)
    reset()
    for index, program in enumerate(runtime._prepared):
        roots = [args[i] for i in getattr(program, '_root_dynamic', ())]
        starts[index].record(stream)
        try:
            program.launch(*roots, stream=stream.cuda_stream)
        except TypeError:
            program.launch(*roots, stream=stream)
        ends[index].record(stream)
    stream.synchronize()

    for index in range(len(starts)):
        ms = starts[index].elapsed_time(ends[index])
        spec = specs[index] if index < len(specs) else {}
        calls = [str(c) for c in spec.get('calls', ())] or ['<none>']
        key = ' + '.join(calls)[:90]
        totals[key] += ms
        counts[key] += 1

    grand = sum(totals.values())
    rows = sorted(totals.items(), key=lambda kv: -kv[1])
    print(json.dumps({
        'artifact': artifact,
        'tokens': tokens,
        'programs': len(starts),
        'sum_ms': round(grand, 3),
        'top': [dict(key=k, ms=round(v, 3), n=counts[k], pct=round(100 * v / grand, 1))
                for k, v in rows[:22]],
    }, indent=1))


if __name__ == '__main__':
    main()
