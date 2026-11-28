# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
from dataclasses import replace
import pytest
import torch

from triton.flagmega import ir as fm
from triton.flagmega.errors import IRSchemaError
from triton.flagmega.evaluator import DictWeightResolver, TorchEvaluator, PagedAttentionStateConfig, create_paged_attention_state
from triton.flagmega.ir.fusion import Fusion
from triton.flagmega.ir.ops.core import get_definition
from triton.flagmega.codegen.triton.fusion import fusion_rules
from triton.flagmega.passes.rewriter import DataflowPass
from triton.flagmega.passes.shape_references import rebind_shape_only_view_users as rebind


def fixture(*, shared_value=False, exported=False):
    mesh = fm.Placement((2, 2), 'yx', 'bb')
    b, s = fm.SBP.broadcast(), fm.SBP.split_contiguous((0, 1), granularity=2)
    weight_type = fm.DistributedType(fm.tensor_type(fm.vector_type('bfloat16', (4, )), (16, 8)), (b, s), mesh)
    config = PagedAttentionStateConfig(1, 2, 8, block_size=4, num_blocks=2)
    builder = fm.IRBuilder(dialect='high_level', stage='frozen_constants')
    ids = builder.var('ids', fm.DistributedType(fm.tensor_type('int32', (1, )), (b, ), mesh), id='ids')
    weight = builder.var('weight', weight_type, id='weight')
    state = builder.var('state', config.ref_type, id='state')

    def op(name, inputs, attrs, id):
        p = get_definition(name).prepare(inputs, attrs)
        return builder.call(name, inputs, p.result_type, id=id, attrs=p.attrs, effect=p.effect)

    embedding = op('nn.embedding', (ids, weight), {}, 'embedding')
    narrow = op('distributed.sharded_view', (embedding, ), {'new_type': replace(embedding.type, axis_policies=(b, b))},
                'narrow')
    rotary = op('nn.rotary_embedding', (narrow, state), {'head_dim': 8, 'theta': 10000., 'output_dtype': 'bfloat16'},
                'rotary')
    cosine = op('builtin.get_item', (rotary, ), {'index': 0}, 'cosine')
    wide = op('ntt.vectorized_cast', (embedding, ),
              {'new_type': fm.vector_type('float32', (4, )), 'vectorize_axes': (1, )}, 'wide')
    broadcast = op('distributed.sharded_view', (wide, ), {'new_type': replace(wide.type, axis_policies=(b, b))},
                   'broadcast')
    stats = op('nn.norm_stats', (wide, ), {'axis': 1, 'use_mean': False}, 'stats')
    outputs = [broadcast, stats, cosine]
    if shared_value:
        outputs.append(op('math.add', (narrow, narrow), {}, 'read_value'))
    if exported: outputs.append(narrow)
    builder.function('main', (ids, weight, state), tuple(outputs))
    return fm.verify_module(builder.build(entry='main')), config


def apply(module):
    return rebind(module, view_id='narrow', replacement_id='broadcast')


def fuse(module):
    return DataflowPass('Fuse', fusion_rules(), rewrite_constants=False).run(module)


def test_rebind_exposes_existing_producer_fusion_and_keeps_rotary_values(tmp_path):
    before, config = fixture()
    assert any(n.op == 'ntt.vectorized_cast' for n in fuse(before).nodes)
    after = apply(before)
    assert after.node_map['wide'] == before.node_map['wide']
    assert 'narrow' not in after.node_map
    assert after.node_map['rotary'].inputs == ('broadcast', 'state')
    assert after.node_map['rotary'].type == before.node_map['rotary'].type
    assert after.node_map['rotary'].effect == before.node_map['rotary'].effect
    order = {n.id: i for i, n in enumerate(after.nodes)}
    assert order['embedding'] < order['wide'] < order['broadcast'] < order['rotary']
    result = fuse(after)
    assert not any(n.op in ('ntt.vectorized_cast', 'tensors.cast') for n in result.nodes)
    assert any(n.op == 'nn.embedding' and n.attrs.get('post_ops') for n in result.nodes)
    weights = torch.linspace(-3, 3, 512).bfloat16().reshape(16, 8, 4)
    state = create_paged_attention_state(config)
    state.seq_lens[0] = 3
    values = {'ids': torch.tensor([5], dtype=torch.int32), 'weight': weights, 'state': state}
    evaluator = TorchEvaluator(DictWeightResolver({}))
    for x, y in zip(evaluator.run(before, values), evaluator.run(result, values)):
        torch.testing.assert_close(x, y, rtol=0, atol=0)
    fm.emit_module(result, tmp_path / 'result.py')
    assert fm.load_module(tmp_path / 'result.py') == result


@pytest.mark.parametrize('kwargs,match', [({'shared_value': True}, 'real-value'), ({'exported': True}, 'exported')])
def test_real_value_sharing_and_public_alias_rejected(kwargs, match):
    m, _ = fixture(**kwargs)
    with pytest.raises(IRSchemaError, match=match):
        apply(m)


def test_numeric_view_preop_is_not_treated_as_a_shape_alias():
    m, _ = fixture()
    n = m.node_map['narrow']
    t = m.node_map[n.inputs[0]].type
    arg = fm.Node('arg', 'builtin.var', (), t, attrs={'name': 'arg'})
    value = fm.Node('double', 'math.add', (arg.id, arg.id), t)
    body = Fusion('double', (arg, value), value.id)
    changed = replace(n, attrs={**n.attrs, 'pre_ops': {'value': body}})
    m = fm.verify_module(replace(m, nodes=tuple(changed if x.id == n.id else x for x in m.nodes)))
    with pytest.raises(IRSchemaError, match='pure unfused'):
        apply(m)


def test_late_arithmetic_dependency_is_not_hoisted_as_a_view():
    m, _ = fixture()
    wide = m.node_map['wide']
    add = fm.Node('late_add', 'math.add', (wide.inputs[0], wide.inputs[0]), m.node_map[wide.inputs[0]].type)
    nodes = []
    for n in m.nodes:
        if n.id == 'wide':
            nodes.append(add)
            n = replace(n, inputs=(add.id, ))
        nodes.append(n)
    m = fm.verify_module(replace(m, nodes=tuple(nodes)))
    with pytest.raises(IRSchemaError, match='Only pure cast/view'):
        apply(m)


def test_no_replacement_geometry_change():
    m, _ = fixture()
    with pytest.raises(IRSchemaError, match='equal complete tensor coordinates'):
        rebind(m, view_id='narrow', replacement_id='wide')
