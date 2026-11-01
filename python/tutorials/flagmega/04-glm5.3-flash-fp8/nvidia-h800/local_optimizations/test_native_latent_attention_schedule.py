"""Portable numerical regression for the actual Native latent template overlay."""
from pathlib import Path
import importlib.util
import sys

import pytest
import torch
from jinja2 import ChoiceLoader, DictLoader, Environment, FileSystemLoader


def render_fixture(path):
    tutorial = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(tutorial))
    from local_optimizations.native_latent_attention_schedule import KEY, _template
    root = Path(__file__).resolve().parents[6] / 'python/triton/flagmega/codegen/triton'
    loader = FileSystemLoader(root)
    environment = Environment(loader=loader, trim_blocks=True, lstrip_blocks=True)
    raw = loader.get_source(environment, KEY)[0]
    mesh = '''{% macro mesh_coordinates() %}
    shard_coord0 = tl.full((), 0, tl.int64)
    shard_coord1 = tl.program_id(0).to(tl.int64) // 16
    shard_coord2 = tl.program_id(0).to(tl.int64) % 16
    block_index = tl.program_id(0).to(tl.int64)
{% endmacro %}'''
    fragments = ['import triton', 'import triton.language as tl', 'import triton.experimental.tle.language as tle']
    for name, source, writer in (('baseline', raw, 'block_index == 0'),
                                 ('columns', _template(raw), 'block_index < 8'),
                                 ('compact', _template(raw, compact_pipeline=True), 'block_index < 8')):
        call = dict(symbol=name, signature='q, indices, cache, starts, lengths, position, blocks, layer, result',
            query='q', topk='indices', kv_cache='cache', query_start_loc='starts', sequence_lengths='lengths',
            slot_mapping='position', block_table='blocks', layer_id='layer', result='result',
            writer_active=writer, index_traversal='compact', compact_topk_capacity=4096,
            topk_width=2051, topk_stride=1, num_query_heads=16, head_dim=512, padded_head_dim=512,
            head_sharded=True, head_active='16', block_size=256,
            cache_strides=(2883584, 262144, 131072, 512, 512, 8), scale_log2=repr(.0625 * 1.4426950408889634),
            mma_query_offset='_fm_dsa_head[:, None] * 512 + _fm_dsa_reduction[None, :]',
            mma_result_offset='_fm_dsa_head[:, None] * 512 + _fm_dsa_dim[None, :]',
            native_latent_schedule=dict(group=dict(index='block_index'), column_tile=64))
        environment.loader = ChoiceLoader([DictLoader({'fixture_mesh': mesh, KEY: source}), loader])
        template = environment.from_string('{% import "fixture_mesh" as entry with context %}{% include "' + KEY + '" %}')
        fragments.append(template.render(call=call, num_warps=8))
        fragments.append(f'''@triton.jit
def {name}_entry(q, indices, cache, starts, lengths, position, blocks, layer, result):
    {name}(q, indices, cache, starts, lengths, position, blocks, tl.load(layer).to(tl.int32), result)
''')
    source = '\n\n'.join(fragments)
    compile(source, str(path), 'exec')
    path.write_text(source)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='SM90 CUDA required')
def test_actual_template_complete_column_outputs_and_online_rounding(tmp_path):
    if torch.cuda.get_device_capability()[0] < 9:
        pytest.skip('Native Hopper MMA layout required')
    from triton.flagmega.runtime.prepared import ResourceContract, prepare_jit_kernel
    from triton.flagmega.runtime.nvidia_launch import compilation_options
    path = tmp_path / 'native_latent_fixture.py'
    render_fixture(path)
    spec = importlib.util.spec_from_file_location('native_latent_fixture', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    generator = torch.Generator(device='cuda').manual_seed(840)
    q = torch.randn(16 * 512, generator=generator, device='cuda').mul_(.1).bfloat16()
    cache = torch.randn(16 * 2883584, generator=generator, device='cuda').mul_(.1).bfloat16()
    indices = torch.full((2083,), -1, dtype=torch.int32, device='cuda')
    starts = torch.tensor([0, 1], dtype=torch.int32, device='cuda')
    lengths = torch.ones(1, dtype=torch.int32, device='cuda')
    position = torch.zeros(1, dtype=torch.int32, device='cuda')
    blocks = torch.arange(16, dtype=torch.int32, device='cuda')
    layer = torch.tensor(4, dtype=torch.int32, device='cuda')
    outputs = [torch.full((16 * 512 + 64,), -100, dtype=torch.bfloat16, device='cuda') for _ in range(3)]
    prepared, graphs = [], []
    for name, output in zip(('baseline', 'columns', 'compact'), outputs, strict=True):
        args = (q, indices, cache, starts, lengths, position, blocks, layer, output[32:-32])
        contract = ResourceContract(compute_num_warps=8, resident_blocks_per_sm=1)
        program = prepare_jit_kernel(getattr(module, name + '_entry'), args, (), grid=(128, 1, 1),
            contract=contract, allocation_device='cuda:0', **compilation_options(contract))
        assert program.resource_report['spill_bytes'] == 0
        with torch.cuda.graph(graph := torch.cuda.CUDAGraph()):
            program.launch()
        prepared.append(program)
        graphs.append(graph)
    for constant in (None, .25):
        if constant is not None:
            q.zero_()
            cache.fill_(constant)
        cache_before, q_before = cache.clone(), q.clone()
        for count in (0, 1, 63, 64, 65, 1024, 2051):
            indices.fill_(-1)
            indices[:count].copy_(torch.arange(count, dtype=torch.int32, device='cuda'))
            position.fill_(max(count - 1, 0))
            lengths.fill_(max(count, 1))
            for request in range(3):
                for output, program, graph in zip(outputs, prepared, graphs, strict=True):
                    output.fill_(-100)
                    if request < 2:
                        graph.replay()
                    else:
                        program.launch()
                torch.cuda.synchronize()
                assert all(torch.equal(outputs[0].view(torch.uint8), output.view(torch.uint8))
                           for output in outputs[1:])
                for output in outputs:
                    assert bool(output[:32].eq(-100).all()) and bool(output[-32:].eq(-100).all())
                    assert bool(torch.isfinite(output[32:-32]).all())
                    if count == 0 or constant is not None:
                        assert bool(output[32:-32].eq(0 if count == 0 else constant).all())
            assert bool(indices[2051:].eq(-1).all())
        assert torch.equal(cache, cache_before) and torch.equal(q, q_before)
