# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Direct paired writes preserve the full cache, source strides, and epochs."""

import pytest
from triton.flagmega import ir as fm
from triton.flagmega.artifacts import write_artifact
from triton.flagmega.compiler import Compiler
from triton.flagmega.evaluator import PagedAttentionStateConfig, create_paged_attention_state
from triton.flagmega.runtime import load

from python.test.flagmega.ir.ops.nn.test_update_paged_attention_kv_cache_pair import (
    PairModule,
    physical_input,
    state_arguments,
)


@pytest.mark.parametrize('lanes,layout,split_axis,block_cyclic,heads,advance,reusable', [
    ((), ('seq', 'head', 'dim'), None, False, 4, False, False),
    ((1,), ('seq', 'head', 'dim'), None, False, 4, True, False),
    ((4,), ('seq', 'head', 'dim'), 'head', False, 4, True, False),
    ((2, 4), ('head', 'dim', 'seq'), 'head', True, 7, False, False),
    ((16,), ('dim', 'seq', 'head'), 'dim', False, 4, True, False),
    ((8,), ('head', 'seq', 'dim'), 'seq', True, 4, True, False),
    ((8,), ('seq', 'head', 'dim'), 'head', False, 4, True, True),
])
def test_pair_writes_all_scalar_lanes_without_touching_neighbors(
        tmp_path, lanes, layout, split_axis, block_cyclic, heads, advance, reusable):
    torch = pytest.importorskip('torch')
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0):
        pytest.skip('SM90 CUDA required')
    config = PagedAttentionStateConfig(3, heads, 32, block_size=16, num_blocks=4)
    source = PairModule(config, lanes=lanes, layout=layout, placement=fm.Placement((2, 2), 'yx', 'bb'),
                        split_axis=split_axis, block_cyclic=block_cyclic, advance=advance, reusable=reusable).build()
    module = Compiler().compile(source).module
    artifact = write_artifact(module, tmp_path/'pair', target='nvidia-sm90', emit_executable=True)
    generated = (artifact/'generated_kernels.py').read_text()
    assert 'cache_dimension' in generated and 'cache_destination' in generated
    assert 'flagmega-kernel: slice/' not in generated
    runtime = load(artifact, device='cuda:0')
    try:
        state = create_paged_attention_state(config, device='cuda')
        state.kv_caches.fill_(-9)
        state.block_table.copy_(torch.tensor([[2, 0, 3, 1]], dtype=torch.int32, device='cuda'))
        semantic, slots = physical_input(config, 3, lanes, layout, device='cuda')
        bindings = runtime.buffer_plan.function_map[module.entry]
        buffers = {}
        for value, names in bindings.parameters:
            if isinstance(module.node_map[value].type, fm.RefType):
                buffers.update(zip(names, state_arguments(state), strict=True))
            else:
                buffers[names[0]] = slots
        for _, names in bindings.outputs:
            buffers.update(zip(names, state_arguments(state), strict=True))
        arguments = [buffers[str(argument['buffer'])] for argument in runtime.external_arguments]
        runtime.prepare(*arguments)
        expected = state.kv_caches.clone()
        for position in (13, 15, 16, 31, 34):
            state.seq_lens.fill_(position)
            runtime.run_into(*arguments)
            for row in range(3):
                page = (2, 0, 3, 1)[(position + row)//16]
                for kind in range(2):
                    chunk = semantic[row, :, kind*32:(kind+1)*32].reshape(heads, 4, 8)
                    expected[page, 1, kind, (position + row)%16].copy_(chunk)
            torch.testing.assert_close(state.kv_caches, expected, rtol=0, atol=0)
            assert state.slot_mapping.item() == position
            assert state.seq_lens.item() == position + 3*advance
            assert state.query_start_loc.tolist() == [0, 3]
        assert runtime.prepare_count == 1
    finally:
        runtime.close()


@pytest.mark.parametrize('variant', ['tir.block_fp8.simt', 'tir.block_fp8.mma'])
def test_projection_reshape_pair_pipeline_keeps_packed_output_on_device(tmp_path, variant):
    torch = pytest.importorskip('torch')
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0):
        pytest.skip('SM90 CUDA required')
    from python.test.flagmega.codegen.triton.kernels.block_fp8.test_packed_output_runtime import (
        artifact,
        linear_module,
        run,
    )

    config = PagedAttentionStateConfig(1, 2, 32, block_size=4, num_blocks=4)

    class Graph(fm.Module):
        def forward(self):
            value = self.input('value', fm.tensor_type('bfloat16', (1, 128)), id='value')
            weight = self.input('weight', fm.tensor_type(fm.vector_type('float8_e4m3fn', (2, 16)), (128, 4)), id='weight')
            scale = self.input('scale', fm.tensor_type('float32', (8, 1)), id='scale')
            state = self.input('state', config.ref_type, id='state')
            kv = fm.F.math.packed_block_scaled_matmul(value, weight, scale, k_pack=2, k_vector=16,
                    weight_block_n=16, weight_block_k=128, name='projection')
            slots = fm.F.tensors.pack(fm.F.tensors.reshape(kv, (1, 2, 64)), 8, axis=-1)
            layer = fm.F.builtin.scalar_const(fm.tensor_type('int32', ()), 0)
            advance = fm.F.builtin.scalar_const(fm.tensor_type('bool', ()), True)
            updated = fm.F.nn.update_paged_attention_kv_cache_pair(slots, state, layer, advance,
                                                                  layout=('seq', 'head', 'dim'))
            self.function('main', (value, weight, scale, state), (fm.F.builtin.tuple(updated),))

    path = artifact(tmp_path, 'pipeline', Graph(dialect='high_level', stage='imported', entry='main').build(),
                    variant, placement=fm.Placement((2, 2), 'yx', 'bb'))
    generated = (path/'generated_kernels.py').read_text()
    assert '# flagmega-kernel: pack/' not in generated and '# flagmega-kernel: slice/' not in generated
    runtime = load(path, device='cuda:0')
    try:
        state = create_paged_attention_state(config, device='cuda')
        state.kv_caches.fill_(-9)
        state.block_table.copy_(torch.tensor([[2, 0, 3, 1]], dtype=torch.int32, device='cuda'))
        value = (torch.arange(128, device='cuda') % 7 - 3).reshape(1, 128).bfloat16()
        weight = (torch.arange(128*128, device='cuda') % 15 - 7).reshape(128, 128).to(torch.float8_e4m3fn)
        scale = torch.ones((8, 1), device='cuda', dtype=torch.float32)
        values = {'value': value, 'weight': weight.reshape(128, 4, 2, 16), 'scale': scale}
        # Preserve the current compiler's split-K rounding contract: compare
        # against its scalar-output projection, not an unsplit torch GEMM.
        reference = artifact(tmp_path, 'scalar_projection', linear_module(n=128, k=128), variant,
                             placement=fm.Placement((2, 2), 'yx', 'bb'))
        semantic = run(reference, {'value': value, 'weight': values['weight'], 'weight_scale': scale},
                       (1, 128), torch).cuda().reshape(1, 2, 64)
        bindings = runtime.buffer_plan.function_map[runtime.ir_module.entry]
        buffers = {}
        for name, names in bindings.parameters:
            buffers.update(zip(names, state_arguments(state), strict=True) if name == 'state' else [(names[0], values[name])])
        for _, names in bindings.outputs:
            buffers.update(zip(names, state_arguments(state), strict=True))
        args = [buffers[str(arg['buffer'])] for arg in runtime.external_arguments]
        runtime.prepare(*args)
        expected = state.kv_caches.clone()
        for position in (3, 4, 7):
            state.seq_lens.fill_(position)
            runtime.run_into(*args)
            page = (2, 0, 3, 1)[position//4]
            for kind in range(2):
                expected[page, 0, kind, position%4].copy_(semantic[0, :, kind*32:(kind+1)*32].reshape(2, 4, 8))
            torch.testing.assert_close(state.kv_caches, expected, rtol=0, atol=0)
            assert state.seq_lens.item() == position+1 and state.slot_mapping.item() == position
    finally:
        runtime.close()
