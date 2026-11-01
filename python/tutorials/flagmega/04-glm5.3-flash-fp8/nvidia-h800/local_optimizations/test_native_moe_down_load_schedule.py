"""Actual E288 MoE template, all TP coordinates, byte and memory-policy regression."""
from pathlib import Path
import importlib.util
import sys

from jinja2 import ChoiceLoader, DictLoader, Environment, FileSystemLoader
import pytest
import torch


def render_fixture(path):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from local_optimizations.native_moe_down_load_schedule import KEY, _template
    root = Path(__file__).resolve().parents[6] / 'python/triton/flagmega/codegen/triton'
    loader = FileSystemLoader(root)
    environment = Environment(loader=loader, trim_blocks=True, lstrip_blocks=True)
    raw = loader.get_source(environment, KEY)[0]
    raw = raw.replace('{% import "entrypoints/_common.py.jinja" as entry with context %}', '')
    mesh = '''{% macro kernel_decorator(call) %}@triton.jit(noinline=True){% endmacro %}
{% macro mesh_coordinates() %}
    shard_coord0 = CHIP
    shard_coord1 = tl.program_id(0).to(tl.int64) // 16
    shard_coord2 = tl.program_id(0).to(tl.int64) % 16
{% endmacro %}'''
    fragments = ['import triton', 'import triton.language as tl', 'import triton.experimental.tle.language as tle']
    for name, template in (('baseline', raw), ('scheduled', _template(raw))):
        call = dict(family='partitioned_sparse_experts_down', variant='wgmma', symbol=name,
            signature='x, ids, w, s, out, CHIP: tl.constexpr', partitions=1, tokens=1, routes=1,
            scalar_n=256, scalar_k=512, output_tile_n=256, tasks_per_owner=1,
            native_moe_down_loads=name == 'scheduled', pointers=dict(activations='x',
                router_expert_ids='ids', down_weight='w'), result='out',
            expert_offset='shard_coord1', slot_active='True', n_active='True', k_active='True',
            activation_offset='shard_coord1 * 2048 + shard_coord0 * 512 + _fm_k',
            weight_offset='shard_coord2 * 37748736 + _fm_expert * 131072 + _fm_n[:, None] * 512 + _fm_k[None, :]',
            result_offset='shard_coord0 * 32768 + shard_coord1 * 4096 + shard_coord2 * 256 + _fm_n',
            scale_loads=dict(down_input_scale='tl.full((), 1., tl.float32)'),
            block_scale_loads=dict(down_proj_scale='tl.load(s + _fm_expert * 512 + shard_coord0 * 128 + (_fm_n + shard_coord2 * 256) // 128 * 4 + _fm_k_start // 128, mask=_fm_n_mask, other=1.)[:, None]'),
            attrs=dict(round_projection=False), route_active='True', writer_active='True')
        environment.loader = ChoiceLoader([DictLoader({KEY: template, 'fixture_mesh': mesh}), loader])
        fragments.append(environment.from_string('{% import "fixture_mesh" as entry with context %}{% include "' + KEY + '" %}').render(render_calls=[call], num_warps=8))
        fragments.append(f'''@triton.jit
def {name}_entry(x, ids, w, s, out, CHIP: tl.constexpr):
    {name}(x, ids, w, s, out, CHIP)
''')
    path.write_text('\n\n'.join(fragments))
    compile(path.read_text(), str(path), 'exec')


@pytest.mark.skipif(not torch.cuda.is_available(), reason='SM90 CUDA required')
def test_full_expert_bank_all_chip_coordinates_and_load_boundaries(tmp_path):
    if torch.cuda.get_device_capability()[0] < 9:
        pytest.skip('Native Hopper FP8 MMA required')
    from triton.flagmega.runtime.prepared import ResourceContract, prepare_jit_kernel
    from triton.flagmega.runtime.nvidia_launch import compilation_options
    path = tmp_path / 'native_moe_down_fixture.py'
    render_fixture(path)
    spec = importlib.util.spec_from_file_location('native_moe_down_fixture', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    x = torch.ones(16384, dtype=torch.bfloat16, device='cuda')
    weights = torch.ones((16, 288, 256, 512), dtype=torch.float8_e4m3fn, device='cuda')
    scales = torch.full((288, 512), .25, dtype=torch.float32, device='cuda')
    ids = torch.arange(8, dtype=torch.int32, device='cuda')
    outputs = [torch.full((131072 + 32,), -101., dtype=torch.float32, device='cuda') for _ in range(2)]
    programs, graphs = {}, {}
    for chip in range(4):
        for index, name in enumerate(('baseline', 'scheduled')):
            args = (x, ids, weights, scales, outputs[index][16:-16], chip)
            contract = ResourceContract(compute_num_warps=8, resident_blocks_per_sm=1)
            program = prepare_jit_kernel(getattr(module, name + '_entry'), args, (), grid=(128, 1, 1),
                contract=contract, allocation_device='cuda:0', **compilation_options(contract))
            assert program.resource_report['spill_bytes'] == 0
            if name == 'scheduled':
                copies = [line for line in program.compiled_kernel.asm['ttgir'].splitlines()
                          if 'async_copy_global_to_local' in line and 'tensor<256x128x' in line]
                assert copies and all('contiguity = 16 : i32' in line for line in copies)
            with torch.cuda.graph(graph := torch.cuda.CUDAGraph()):
                program.launch()
            programs[chip, index], graphs[chip, index] = program, graph
    for case in ('ones', 'zero', 'random840', 'random20726'):
        if case == 'zero':
            x.zero_()
        elif case.startswith('random'):
            generator = torch.Generator(device='cuda').manual_seed(int(case[6:]))
            x.copy_(torch.randn(x.shape, generator=generator, device='cuda').mul_(.125).bfloat16())
            weights.copy_(torch.randn(weights.shape, generator=generator, device='cuda').mul_(.125).to(weights.dtype))
        immutable_x, immutable_w, immutable_s = x.clone(), weights.clone(), scales.clone()
        for expert_start in (0, 280):
            ids.copy_(torch.arange(expert_start, expert_start + 8, dtype=torch.int32, device='cuda'))
            immutable_ids = ids.clone()
            for chip in range(4):
                for request in range(3):
                    for index, output in enumerate(outputs):
                        output.fill_(-101.)
                        if request < 2:
                            graphs[chip, index].replay()
                        else:
                            programs[chip, index].launch()
                    torch.cuda.synchronize()
                    assert torch.equal(outputs[0].view(torch.uint8), outputs[1].view(torch.uint8))
                    for output in outputs:
                        active = output[16 + chip * 32768:16 + (chip + 1) * 32768]
                        assert bool(torch.isfinite(active).all())
                        assert bool(output[:16 + chip * 32768].eq(-101.).all())
                        assert bool(output[16 + (chip + 1) * 32768:].eq(-101.).all())
                        if case in ('ones', 'zero'):
                            assert bool(active.eq(128. if case == 'ones' else 0.).all())
                assert torch.equal(ids, immutable_ids)
        assert torch.equal(x, immutable_x)
        assert torch.equal(weights.view(torch.uint8), immutable_w.view(torch.uint8))
        assert torch.equal(scales, immutable_s)
