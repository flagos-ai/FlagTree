"""Exercise complete DSA ranking with destructive publication and a CPU oracle.

Run with the FlagTree environment on an idle GPU:
  python repro_parallel_dsa_pool_sort.py --num-warps 8
  python repro_parallel_dsa_pool_sort.py --num-warps 4
Only synthetic FP32 scores and int32 selections are used.

Regression: the original destructive small-sort helper had no CTA barriers
between capturing scores, initializing the aliased result, and publishing
selections. With eight warps, count=16/missing=2/signed-zero ties could leave
valid fourth-token slots at -1 (for example output35 should be66). Repeated
eager launches after the two graph replays expose this warp scheduling race.
The actual template is tested here, together with full-pool ranking neighbors.

"""
import argparse
import importlib.util
import json
from pathlib import Path
import tempfile

import numpy as np
import torch

from local_optimizations.parallel_dsa_pool_sort import HELPER, SHORT_SORT_DISPATCH
from local_optimizations.native_dsa_pool_scores import KEY, _template
from local_optimizations.native_dsa_pool_sort import template
from triton.flagmega.codegen.triton.templates import TritonTemplateRegistry
from triton.flagmega.runtime.nvidia_launch import compilation_options
from triton.flagmega.runtime.prepared import ResourceContract, prepare_jit_kernel


def expected_selection(values, first, visible, base):
    scores = values.copy()
    scores[np.isnan(scores)] = -np.inf
    scores[scores == 0] = np.float32(0)
    bits = scores.view(np.uint32)
    order = np.where((bits & 0x80000000) != 0, ~bits, bits ^ 0x80000000)
    ids = sorted(range(len(scores)), key=lambda i: (-int(order[i]), i))[:512]
    result = np.full(2051, -1, dtype=np.int32)
    for rank, pool in enumerate(ids):
        if order[pool] > 0x007fffff:
            result[rank*4:rank*4+4] = first + pool*4 + np.arange(4)
    count = visible % 4
    for offset in range(count):
        value = first + visible - count + offset
        if value < base + 1:
            result[2048+offset] = value
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--num-warps', type=int, choices=(4, 8), default=8)
    parser.add_argument('--replays', type=int, default=9, help='Two graph replays followed by eager launches')
    args = parser.parse_args()
    if args.replays < 3:
        parser.error('--replays must include two graph replays and at least one eager launch')
    registry = TritonTemplateRegistry()
    raw = registry.environment.loader.get_source(registry.environment, KEY)[0]
    rendered_template = template(_template(raw))
    begin = rendered_template.index('@triton.jit(noinline=True)\ndef {{ call.symbol }}__sort_tile')
    end = rendered_template.index('@triton.jit(noinline=True)\ndef {{ call.symbol }}__sort_merge', begin)
    call = dict(symbol='repro', result='Result', pool=4, pool_tile=1024,
                output_tile=4096, output_width=2051, index_topk=2048)
    small_sort = registry.environment.from_string(rendered_template[begin:end]).render(call=call)
    small_select = registry.environment.from_string(
        '@triton.jit(noinline=True)\ndef repro__small_select(Result, Meta):\n'
        '    first = tl.load(Meta)\n    visible_count = tl.load(Meta + 1)\n'
        '    base = tl.load(Meta + 2)\n'
        '    pool_count = tl.maximum((base + 1 - first + 3) // 4, 0)\n'
        + SHORT_SORT_DISPATCH).render(call=call)
    # The reproducer's explicit launch choice must agree with its encoding.
    # Full generated GLM programs use eight warps; this also exercises four
    # warps with the same complete pool population and publication lifetime.
    rank_layout = 'tle.gpu.BlockEncoding([1, 1], [1, 32], [8, 1], [1, 0])'
    assert HELPER.count(rank_layout) == 2
    parallel_helper = HELPER.replace(
        rank_layout, f'tle.gpu.BlockEncoding([1, 1], [1, 32], [{args.num_warps}, 1], [1, 0])')
    source = ('import triton\nimport triton.language as tl\n'
              'import triton.experimental.tle.language as tle\n'
              'MESH=tl.constexpr(tle.device_mesh({"block":[("block_y",8),("block_x",16)]}))\n'
              + parallel_helper.replace('{{ call.symbol }}', 'repro').replace(
                  '{{ entry.stage_barrier() }}', '    tle.distributed_barrier(MESH)')
              + '\n' + small_sort + '\n' + small_select
              + '\n@triton.jit\ndef entry(Result, Meta, Tile:tl.constexpr):\n'
              '    count = tl.maximum((tl.load(Meta+2) + 1 - tl.load(Meta) + 3) // 4, 0)\n'
              '    if count < 257:\n'
              '        if tl.program_id(0) == 0:\n'
              '            repro__small_select(Result, Meta)\n'
              '    else:\n'
              '        repro__parallel_select(Result, tl.load(Meta), tl.load(Meta+1), tl.load(Meta+2), 0, Tile)\n')
    with tempfile.TemporaryDirectory(prefix='dsa-rank-repro-') as directory:
        path = Path(directory) / 'kernels.py'
        path.write_text(source)
        spec = importlib.util.spec_from_file_location('dsa_rank_repro_kernels', path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        torch.cuda.set_device(0)
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        torch.cuda.set_stream(stream)
        buffer = torch.full((2099,), -991271, dtype=torch.int32, device='cuda')
        result = buffer[16:2067]
        meta = torch.zeros(16, dtype=torch.int32, device='cuda')
        contract = ResourceContract(compute_num_warps=args.num_warps,
                                    resident_blocks_per_sm=1, requires_grid_wide_sync=True)
        options = compilation_options(contract)
        options['enable_fp_fusion'] = False
        options['launch_cooperative_grid'] = True
        programs, graphs = {}, {}
        for tile in (512, 528, 1024):
            program = prepare_jit_kernel(module.entry, (result, meta, tile), (),
                                         grid=(128, 1, 1), contract=contract,
                                         allocation_device='cuda:0', **options)
            assert program.resource_report['spill_bytes'] == 0
            programs[tile] = program
            with torch.cuda.graph(graph := torch.cuda.CUDAGraph(), stream=stream):
                program.launch()
            graphs[tile] = graph
        cases = 0
        for count in (0, 1, 4, 8, 16, 17, 32, 64, 65, 128, 256, 257, 511, 512, 513, 514, 528, 529, 768, 1024):
            tile = 512 if count <= 512 else 528 if count <= 528 else 1024
            for missing in range(1 if count == 0 else 4):
                for pattern in range(6):
                    values = np.random.default_rng(98211+count*7+pattern).standard_normal(1024).astype(np.float32)
                    if pattern == 1:
                        values.fill(1)
                    elif pattern == 2:
                        values[:] = np.arange(1024) % 7 - 3
                    elif pattern == 3:
                        values[::5], values[1::5], values[2::5] = np.nan, np.inf, -np.inf
                    elif pattern == 4:
                        values.fill(0)
                        values[1::2] = -0.0
                    elif pattern == 5:
                        values *= np.float32(1e30)
                    values[count:] = -np.inf
                    if missing:
                        values[count-1] = -np.inf
                    first, visible = 31, count*4-missing
                    base = first+visible-1
                    meta[:3].copy_(torch.tensor([first, visible, base], dtype=torch.int32, device='cuda'))
                    expected = expected_selection(values, first, visible, base)
                    for replay in range(args.replays):
                        buffer.fill_(-991271)
                        result.view(torch.float32)[:1024].copy_(torch.from_numpy(values))
                        if replay < 2:
                            graphs[tile].replay()
                        else:
                            programs[tile].launch()
                        actual = result.cpu().numpy()
                        bad = np.flatnonzero(actual != expected)[:32]
                        assert not len(bad), (count, missing, pattern, replay,
                                             bad.tolist(), actual[bad].tolist(), expected[bad].tolist())
                        assert bool(buffer[:16].eq(-991271).all()) and bool(buffer[2067:].eq(-991271).all())
                    cases += 1
        print(json.dumps(dict(passed=True, cases=cases, num_warps=args.num_warps,
                              all_2051_int32_exact=True, two_graph_and_eager=True,
                              full_model_acceptance=False)))


if __name__ == '__main__':
    main()
