# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Packed representations must win before producer packing is applied."""

from dataclasses import replace

import pytest
from triton.flagmega import ir as fm
from triton.flagmega.egraph.rewriter import EGraphSession
from triton.flagmega.evaluator import DictWeightResolver, TorchEvaluator
from triton.flagmega.passes.functions.post_boundary_pack import _representation_cost
from triton.flagmega.rules.neutral.fold_pack_reshape import fold_pack_reshape_rule


@pytest.mark.parametrize('operation', [
    'builtin.var', 'builtin.weight', 'builtin.scalar_const', 'builtin.tuple', 'builtin.get_item',
    'tensors.bitcast', 'tensors.pack', 'tensors.unpack', 'tensors.permute', 'tensors.reshape',
    'tensors.slice', 'math.add', 'nn.norm_apply', 'ntt.packed_matmul',
])
@pytest.mark.parametrize('distributed', [False, True])
@pytest.mark.parametrize('lanes', [(1,), (8,), (2, 4)])
def test_packed_type_cost_is_strictly_smaller_in_every_operation_category(operation, distributed, lanes):
    from math import prod
    scalar = fm.tensor_type('bfloat16', (1, 64))
    packed = fm.tensor_type(fm.vector_type('bfloat16', lanes), (1, 64//prod(lanes)))
    if distributed:
        placement = fm.Placement((2, 2), 'yx', 'bb')
        scalar = fm.DistributedType(scalar, (fm.SBP.broadcast(),)*2, placement)
        packed = fm.DistributedType(packed, (fm.SBP.broadcast(),)*2, placement)
    scalar_node = fm.Node('value', operation, (), scalar)
    packed_node = replace(scalar_node, type=packed)
    assert 0 <= _representation_cost(packed_node, None) < _representation_cost(scalar_node, None)


def test_each_packed_tuple_field_reduces_cost_without_reclassifying_cache_handles():
    scalar = fm.tensor_type('bfloat16', (1, 64))
    packed = fm.tensor_type(fm.vector_type('bfloat16', (8,)), (1, 8))
    cache = fm.RefType('state', (('packed_storage', packed),))
    fields = [(scalar, scalar, cache), (packed, scalar, cache), (packed, packed, cache)]
    costs = [_representation_cost(fm.Node('tuple', 'builtin.tuple', (), fm.TupleType(value)), None) for value in fields]
    assert costs[2] < costs[1] < costs[0]
    assert _representation_cost(fm.Node('scalar', 'builtin.get_item', (), scalar), None) == _representation_cost(
        fm.Node('tuple', 'builtin.get_item', (), fm.TupleType((scalar, cache))), None)


def test_existing_reshape_propagation_wins_at_extraction_without_an_extra_normalization_pass():
    import torch

    class Graph(fm.Module):
        def forward(self):
            value = self.input('value', fm.tensor_type('bfloat16', (1, 128)), id='value')
            heads = fm.F.tensors.reshape(value, (1, 2, 64))
            output = fm.F.tensors.pack(heads, 8, axis=-1)
            self.function('main', (value,), (output,))

    before = Graph(dialect='high_level', stage='packed', entry='main').build()
    session = EGraphSession()
    session.construct(before)
    session.apply_rules('FoldPackReshape', (fold_pack_reshape_rule(),), cost=_representation_cost,
                        cost_model='representation-boundary-preference/v3')
    after = session.extract()
    output = after.node_map[after.function_map['main'].outputs[0]]
    assert output.op == 'tensors.reshape'
    producer_pack = after.node_map[output.inputs[0]]
    assert producer_pack.op == 'tensors.pack' and producer_pack.inputs == ('value',)
    values = {'value': torch.arange(128).reshape(1, 128).bfloat16()}
    evaluator = TorchEvaluator(DictWeightResolver({}))
    torch.testing.assert_close(evaluator.run(after, values), evaluator.run(before, values), rtol=0, atol=0)
