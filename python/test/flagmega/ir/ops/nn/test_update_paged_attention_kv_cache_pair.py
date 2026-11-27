# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Pair publication is one state epoch with a complete, typed source."""

from dataclasses import replace
from itertools import permutations
from math import prod

import pytest
import torch
from triton.flagmega import ir as fm
from triton.flagmega import pattern_match as pm
from triton.flagmega.errors import IRSchemaError
from triton.flagmega.evaluator import DictWeightResolver, TorchEvaluator
from triton.flagmega.ir.ops.nn._paged_attention_state import PagedAttentionStateConfig, create_paged_attention_state
from triton.flagmega.ir.ops.tensors.pack import pack_physical
from triton.flagmega.passes.rewriter import DataflowPass
from triton.flagmega.rules.neutral.form_paged_attention_kv_cache_pair import form_paged_attention_kv_cache_pair_rule

LAYOUTS = tuple(permutations(('seq', 'head', 'dim')))


class PairModule(fm.Module):
    def __init__(self, config, *, lanes=(), tokens=3, layout=('seq', 'head', 'dim'),
                 legacy=False, branch_pack=False, advance=True, negative=None,
                 placement=None, split_axis=None, block_cyclic=False, reusable=False):
        super().__init__(dialect='high_level', stage='distributed' if placement else 'imported', entry='main',
                         metadata={'auto_distribution': {'placement': placement.to_data()}} if placement else {})
        self.config, self.lanes, self.tokens, self.layout = config, lanes, tokens, layout
        self.legacy, self.branch_pack, self.advance, self.negative = legacy, branch_pack, advance, negative
        self.placement, self.split_axis, self.block_cyclic = placement, split_axis, block_cyclic
        self.reusable = reusable

    def forward(self):
        config, lanes, layout = self.config, self.lanes, self.layout
        extents = {'seq': self.tokens, 'head': config.num_kv_heads, 'dim': 2 * config.head_dim // prod(lanes)}
        dtype = fm.vector_type('bfloat16', lanes) if lanes else fm.DType.BFLOAT16
        source = self.input('slots', fm.tensor_type(dtype, tuple(extents[axis] for axis in layout)))
        slots = source
        if self.placement:
            policies = [fm.SBP.broadcast()] * 3
            if self.split_axis:
                baxis = self.placement.hierarchy_levels.index('b')
                policies[layout.index(self.split_axis)] = (
                    fm.SBP.split_block_cyclic((baxis,), 2) if self.block_cyclic
                    else fm.SBP.split_contiguous((baxis,)))
            slots = fm.F.distributed.boxing(source, fm.DistributedType(source.type, tuple(policies), self.placement))
        state = self.input('state', config.ref_type)
        layer = fm.F.builtin.scalar_const(fm.tensor_type('int32', ()), 1, name='layer')
        advance = fm.F.builtin.scalar_const(fm.tensor_type('bool', ()), self.advance, name='advance')
        extras = ()
        if self.legacy:
            width = extents['dim'] // 2
            key = fm.F.tensors.slice(slots, starts=(0,), ends=(width,), axes=(layout.index('dim'),), name='key')
            value_source = slots if self.negative != 'different_source' else self.input('other', source.type)
            value = fm.F.tensors.slice(value_source, starts=(width,), ends=(2*width,), axes=(layout.index('dim'),), name='value')
            if self.branch_pack:
                key = fm.F.tensors.pack(key, config.lanes, axis=layout.index('dim'), name='key_pack')
                value = fm.F.tensors.pack(value, config.lanes, axis=layout.index('dim'), name='value_pack')
            no_advance = fm.F.builtin.scalar_const(fm.tensor_type('bool', ()), self.negative == 'key_advances')
            first = fm.F.nn.update_paged_attention_kv_cache(key, state, layer, no_advance,
                    cache_kind='key', layout=layout, name='key_write')
            if self.negative == 'intervening_effect':
                fm.F.nn.update_paged_attention_kv_cache(key, state, layer, no_advance,
                    cache_kind='key', layout=layout, name='intervening')
            if self.negative == 'state_observer':
                extras = (first,)
            elif self.negative == 'shared_key':
                extras = (key,)
            result = fm.F.nn.update_paged_attention_kv_cache(value, first, layer, advance,
                    cache_kind='value', layout=layout, name='update')
        else:
            result = fm.F.nn.update_paged_attention_kv_cache_pair(slots, state, layer, advance, layout=layout, name='update')
        parameters = (source, state)
        if self.negative == 'different_source':
            parameters += (value_source,)
        if self.reusable:
            self.function('publish', parameters, (result,), attrs={'reusable': True, 'noinline': True})
            outer_slots = self.input('outer_slots', source.type)
            outer_state = self.input('outer_state', state.type)
            result = fm.F.builtin.call(outer_slots, outer_state, callee='publish', result_type=config.ref_type,
                                       effect=fm.effect('read_write', config.ref_type.name))
            parameters = (outer_slots, outer_state)
        self.function('main', parameters, (fm.F.builtin.tuple(result), *extras))


def fuse(module):
    return DataflowPass('FormPagedAttentionKVCachePair', (form_paged_attention_kv_cache_pair_rule(),)).run(module)


def physical_input(config, tokens, lanes, layout, device='cpu'):
    semantic = torch.arange(tokens * config.num_kv_heads * 2 * config.head_dim,
                            device=device).reshape(tokens, config.num_kv_heads, 2 * config.head_dim).bfloat16()
    value = semantic.permute(tuple(('seq', 'head', 'dim').index(axis) for axis in layout)).contiguous()
    if lanes:
        value = pack_physical(value, 3, lanes, (layout.index('dim'),) * len(lanes))
    return semantic, value


def state_arguments(state):
    return (state.kv_caches, state.query_start_loc, state.seq_lens, state.slot_mapping, state.block_table)


@pytest.mark.parametrize('layout', LAYOUTS)
@pytest.mark.parametrize('lanes', [(), (1,), (4,), (2, 4), (16,)])
@pytest.mark.parametrize('advance', [False, True])
def test_pair_evaluator_preserves_complete_state_identity_and_chunk_epoch(layout, lanes, advance):
    config = PagedAttentionStateConfig(3, 4, 32, block_size=4, num_blocks=4)
    module = PairModule(config, layout=layout, lanes=lanes, advance=advance).build()
    state = create_paged_attention_state(config)
    state.kv_caches.fill_(-9)
    state.seq_lens.fill_(3)
    state.block_table.copy_(torch.tensor([[2, 0, 3, 1]], dtype=torch.int32))
    semantic, physical = physical_input(config, 3, lanes, layout)
    expected = state.kv_caches.clone()
    for row in range(3):
        page = (2, 0, 3, 1)[(3 + row) // 4]
        for kind in range(2):
            expected[page, 1, kind, (3 + row) % 4].copy_(semantic[row, :, kind*32:(kind+1)*32].reshape(4, 4, 8))
    result = TorchEvaluator(DictWeightResolver({})).run(module, {'slots': physical, 'state': state})[0][0]
    assert result is state
    torch.testing.assert_close(state.kv_caches, expected, rtol=0, atol=0)
    assert state.seq_lens.item() == 3 + 3*advance
    assert state.slot_mapping.item() == 3
    assert state.query_start_loc.tolist() == [0, 3]


@pytest.mark.parametrize('branch_pack', [False, True])
def test_legacy_slice_chain_forms_pair_before_packing_without_copying_or_duplicating_source(branch_pack):
    config = PagedAttentionStateConfig(3, 4, 32, block_size=4, num_blocks=4)
    before = PairModule(config, legacy=True, branch_pack=branch_pack).build()
    after = fuse(before)
    assert sum(n.op == 'tensors.slice' for n in before.nodes) == 2
    assert not [n for n in after.nodes if n.op in {'tensors.slice', 'nn.update_paged_attention_kv_cache'}]
    assert after.node_map['update'].op == 'nn.update_paged_attention_kv_cache_pair'
    _, physical = physical_input(config, 3, (), ('seq', 'head', 'dim'))
    states = [create_paged_attention_state(config) for _ in range(2)]
    for module, state in zip((before, after), states):
        state.seq_lens.fill_(3)
        TorchEvaluator(DictWeightResolver({})).run(module, {'slots': physical, 'state': state})
    for a, b in zip(state_arguments(states[0]), state_arguments(states[1])):
        torch.testing.assert_close(a, b, rtol=0, atol=0)


@pytest.mark.parametrize('negative', ['key_advances', 'state_observer', 'intervening_effect', 'different_source'])
def test_pair_formation_preserves_non_equivalent_state_regions(negative):
    config = PagedAttentionStateConfig(3, 4, 32)
    after = fuse(PairModule(config, legacy=True, branch_pack=True, negative=negative).build())
    assert after.node_map['update'].op == 'nn.update_paged_attention_kv_cache'
    assert 'key_write' in after.node_map


def test_shared_key_selection_is_preserved_for_its_other_consumer():
    config = PagedAttentionStateConfig(3, 4, 32)
    after = fuse(PairModule(config, legacy=True, negative='shared_key').build())
    assert after.node_map['update'].op == 'nn.update_paged_attention_kv_cache_pair'
    assert after.node_map['key'].op == 'tensors.slice'
    assert 'value' not in after.node_map


def test_pair_roundtrip_schema_patterns_effects_and_cost(tmp_path):
    config = PagedAttentionStateConfig(3, 4, 32)
    module = PairModule(config, lanes=(2, 4)).build()
    fm.emit_module(module, tmp_path/'pair.py')
    restored = fm.load_module(tmp_path/'pair.py')
    assert restored.semantic_hash == module.semantic_hash
    node = restored.node_map['update']
    definition = fm.get_definition(node.op)
    assert not node.effect.is_pure and node.type == config.ref_type
    assert definition.state.memory_effect == fm.get_definition('nn.update_paged_attention_kv_cache').state.memory_effect
    assert pm.try_match_root(node, pm.F.nn.is_update_paged_attention_kv_cache_pair(layout=('seq', 'head', 'dim')), restored)
    factors = definition.cost_factors([restored.node_map[x] for x in node.inputs], node.attrs, node.type)
    assert factors.block_local_memory_load_bytes == 3*4*64*2
    assert factors.block_local_memory_store_bytes == 3*4*64*2


@pytest.mark.parametrize('mode', ['partial', 'chip_split', 'head_count', 'dim_count', 'dtype'])
def test_pair_rejects_incomplete_or_mismatched_slots(mode):
    config = PagedAttentionStateConfig(3, 4, 32)
    module = PairModule(config).build()
    node = module.node_map['update']
    inputs = [module.node_map[x] for x in node.inputs]
    tensor = inputs[0].type
    if mode in ('partial', 'chip_split'):
        policies = [fm.SBP.broadcast()] * 3
        if mode == 'chip_split':
            policies[1] = fm.SBP.split_contiguous((0,))
        changed = fm.DistributedType(tensor, tuple(policies), fm.Placement((2, 2), 'cb', 'cb'),
                  partial=fm.SBP.partial((1,)) if mode == 'partial' else None)
    else:
        shape = [3, 4, 64]
        if mode == 'head_count': shape[1] = 3
        if mode == 'dim_count': shape[2] = 32
        changed = fm.tensor_type('float32' if mode == 'dtype' else 'bfloat16', shape)
    inputs[0] = replace(inputs[0], type=changed)
    with pytest.raises(IRSchemaError):
        fm.get_definition(node.op).infer_type(inputs, node.attrs)


def test_distribution_offers_block_shards_and_materialized_chip_publication():
    from triton.flagmega.passes.auto_distributed import DistributedCandidateContext
    from triton.flagmega.passes.auto_distributed.inference_providers import TypeInferenceCandidateProvider
    config = PagedAttentionStateConfig(3, 4, 32)
    module = PairModule(config, lanes=(8,)).build()
    node = module.node_map['update']
    inputs = [module.node_map[x] for x in node.inputs]
    placement = fm.Placement((2, 2), 'cb', 'cb')
    tensor = inputs[0].type
    broad = fm.DistributedType(tensor, (fm.SBP.broadcast(),)*3, placement)
    block_head = replace(broad, axis_policies=(fm.SBP.broadcast(), fm.SBP.split_contiguous((1,)), fm.SBP.broadcast()))
    block_dim = replace(broad, axis_policies=(fm.SBP.broadcast(), fm.SBP.broadcast(), fm.SBP.split_contiguous((1,))))
    chip_head = replace(broad, axis_policies=(fm.SBP.broadcast(), fm.SBP.split_contiguous((0,)), fm.SBP.broadcast()))
    context = DistributedCandidateContext(module, node, placement,
                 ((broad, block_head, block_dim, chip_head), *((value.type,) for value in inputs[1:])))
    candidates = TypeInferenceCandidateProvider(frozenset({node.op})).get_candidates(context)
    assert {candidate.input_types[0] for candidate in candidates} >= {broad, block_head, block_dim}
    assert chip_head not in {candidate.input_types[0] for candidate in candidates}
    assert all(candidate.return_type == config.ref_type for candidate in candidates)
    assert all(candidate.input_types[1:] == tuple(value.type for value in inputs[1:]) for candidate in candidates)


def test_pair_formation_keeps_required_partial_boxing():
    config = PagedAttentionStateConfig(3, 4, 32)
    before = PairModule(config, legacy=True).build()
    source = before.node_map[before.function_map['main'].parameters[0]]
    placement = fm.Placement((2,), 'b', 'b')
    partial = fm.DistributedType(source.type, (fm.SBP.broadcast(),)*3, placement, partial=fm.SBP.partial((0,)))
    complete = replace(partial, partial=None)
    var = replace(source, type=partial)
    definition = fm.get_definition('distributed.boxing')
    prepared = definition.prepare((var,), {'new_type': complete})
    boxing = fm.Node('collect', definition.op_name, (source.id,), prepared.result_type, prepared.effect, prepared.attrs)
    nodes = (var, boxing, *(replace(node, inputs=tuple('collect' if edge == source.id else edge for edge in node.inputs))
                            for node in before.nodes if node.id != source.id))
    typed = {}
    for node in nodes:
        if node.inputs:
            prepared = fm.get_definition(node.op).prepare([typed[x] for x in node.inputs], node.attrs)
            node = replace(node, type=prepared.result_type, effect=prepared.effect, attrs=prepared.attrs)
        typed[node.id] = node
    before = replace(before, nodes=tuple(typed.values()))
    before = fm.verify_module(before)
    after = fuse(before)
    assert after.node_map['update'].op == 'nn.update_paged_attention_kv_cache_pair'
    assert after.node_map['update'].inputs[0] == 'collect'
    assert after.node_map['collect'].inputs == (source.id,)
    assert after.node_map[source.id].type.partial is not None
