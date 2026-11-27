# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""A paired cache consumer keeps one projection's packed output through lowering."""

import torch
from triton.flagmega import ir as fm
from triton.flagmega.evaluator import (
    DictWeightResolver,
    PagedAttentionStateConfig,
    TorchEvaluator,
    create_paged_attention_state,
)
from triton.flagmega.passes.auto_distributed.policy import lower_vectorization_contracts
from triton.flagmega.passes.functions import post_function_boundary_pack_propagation
from triton.flagmega.targets import NvidiaSm90Target


def test_pair_consumes_native_packed_projection_without_slice_or_pack_restoration():
    config = PagedAttentionStateConfig(1, 2, 32, block_size=4, num_blocks=2)

    class Graph(fm.Module):
        def forward(self):
            value = self.input('value', fm.tensor_type('bfloat16', (1, 128)), id='value')
            state = self.input('state', config.ref_type, id='state')
            weight = self.weight('weight', fm.tensor_type(fm.vector_type('float8_e4m3fn', (2, 16)), (128, 4)),
                                 source='weights', key='weight')
            scale = self.weight('scale', fm.tensor_type('float32', (8, 1)), source='weights', key='scale')
            kv = fm.F.math.packed_block_scaled_matmul(value, weight, scale, k_pack=2, k_vector=16,
                                                       weight_block_n=16, weight_block_k=128, name='projection')
            kv = fm.F.tensors.reshape(kv, (1, 2, 64))
            kv = fm.F.tensors.pack(kv, (8,), axis=-1)
            layer = fm.F.builtin.scalar_const(fm.tensor_type('int32', ()), 0)
            advance = fm.F.builtin.scalar_const(fm.tensor_type('bool', ()), True)
            result = fm.F.nn.update_paged_attention_kv_cache_pair(kv, state, layer, advance,
                                                                 layout=('seq', 'head', 'dim'))
            self.function('main', (value, state), (result,))

    before = Graph(dialect='high_level', stage='packed', entry='main').build()
    after = post_function_boundary_pack_propagation(before, NvidiaSm90Target())
    assert len([node for node in after.nodes if node.op == 'math.packed_block_scaled_matmul']) == 1
    assert not [node for node in after.nodes if node.op == 'tensors.pack']
    projection, = [node for node in after.nodes if node.op == 'math.packed_block_scaled_matmul']
    assert projection.attrs['output_lanes'] == (8,)
    lowered = lower_vectorization_contracts(after)
    for module in (after, lowered):
        assert not [node for node in module.nodes if node.op in {'tensors.pack', 'tensors.slice'}]
        pair, = [node for node in module.nodes if node.op == 'nn.update_paged_attention_kv_cache_pair']
        assert isinstance(module.node_map[pair.inputs[0]].type.dtype, fm.VectorType)
    generator = torch.Generator().manual_seed(17)
    value = torch.randn(1, 128, generator=generator).bfloat16()
    weights = {'weight': torch.randn(128, 4, 2, 16, generator=generator).to(torch.float8_e4m3fn),
               'scale': torch.rand(8, 1, generator=generator)+.25}
    states = [create_paged_attention_state(config) for _ in range(3)]
    for module, state in zip((before, after, lowered), states):
        state.kv_caches.fill_(-9)
        state.seq_lens.fill_(3)
        TorchEvaluator(DictWeightResolver(weights)).run(module, {'value': value, 'state': state})
    for state in states[1:]:
        torch.testing.assert_close(state.kv_caches, states[0].kv_caches, rtol=0, atol=0)
        assert state.seq_lens.item() == 4 and state.slot_mapping.item() == 3
