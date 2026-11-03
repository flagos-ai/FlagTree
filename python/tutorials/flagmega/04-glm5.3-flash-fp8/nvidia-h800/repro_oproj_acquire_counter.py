"""Finite counter election/broadcast regression, without GPU polling loops."""
import argparse
import importlib.machinery
import importlib.util
import json
from pathlib import Path

import torch
import triton
import triton.language as tl
import triton.experimental.tle.language as tle
from triton.flagmega.runtime.nvidia_launch import compilation_options
from triton.flagmega.runtime.prepared import ResourceContract, prepare_jit_kernel


poll = None


@triton.jit
def snapshot(Flags, Out):
    indices = tl.arange(0, 4)
    values = poll(Flags + indices)
    tl.store(Out + tl.program_id(0) * 4 + indices, values)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--poll-template', type=Path, default=Path(__file__).parent / 'local_optimizations/oproj-observe.py.jinja')
    parser.add_argument('--num-warps', type=int, choices=(4, 8), default=8)
    args = parser.parse_args()
    loader = importlib.machinery.SourceFileLoader('_oproj_counter_regression', str(args.poll_template.resolve()))
    spec = importlib.util.spec_from_loader(loader.name, loader)
    module = importlib.util.module_from_spec(spec)
    module.__dict__.update(triton=triton, tl=tl, tle=tle)
    loader.exec_module(module)
    global poll
    poll = module._flagmega_oproj_acquire_poll
    flags = torch.empty(4, dtype=torch.int64, device='cuda')
    out = torch.empty((128, 4), dtype=torch.int64, device='cuda')
    contract = ResourceContract(compute_num_warps=args.num_warps, resident_blocks_per_sm=1)
    program = prepare_jit_kernel(snapshot, (flags, out), (), grid=(128, 1, 1), contract=contract,
                                 allocation_device='cuda:0', **compilation_options(contract, floating_point_contract='strict'))
    assert program.resource_report['spill_bytes'] == 0
    graphs = []
    for _ in range(2):
        with torch.cuda.graph(graph := torch.cuda.CUDAGraph()):
            program.launch()
        graphs.append(graph)
    for case in range(32):
        flags.copy_(torch.tensor([case * 111 + 1, case * 777 + 3, 2**62 - case - 1, -case - 1], dtype=torch.int64, device='cuda'))
        for graph in graphs:
            graph.replay()
            torch.cuda.synchronize()
            assert torch.equal(out, flags[None, :].expand_as(out)), ('counter broadcast differs', case)
        program.launch()
        torch.cuda.synchronize()
        assert torch.equal(out, flags[None, :].expand_as(out)), ('counter broadcast differs', case)
    print(json.dumps(dict(passed=True, cases=32, grid=[128, 1, 1], num_warps=args.num_warps,
                          launch_modes=['graph1', 'graph2', 'eager'], resource=program.resource_report,
                          full_model_acceptance=False)), flush=True)


if __name__ == '__main__':
    main()
