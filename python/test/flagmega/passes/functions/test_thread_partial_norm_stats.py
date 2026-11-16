# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
from dataclasses import replace
import pytest
from triton.flagmega import ir as fm
from triton.flagmega.errors import IRSchemaError
from triton.flagmega.ir.ops.core import get_definition
from triton.flagmega.passes.functions.thread_partial_norm_stats import thread_partial_norm_stats_across_function_boundaries as transform


def fixture(layers=2):
    p, b = fm.Placement((2, 2), 'yx', 'bb'), fm.SBP.broadcast()
    s = fm.SBP.split_contiguous((0, 1), granularity=4)
    ty = fm.DistributedType(fm.tensor_type('float32', (1, 16)), (b, s), p)
    broad = replace(ty, axis_policies=(b, b))
    stats_ty = fm.DistributedType(fm.tensor_type('float32', (1, 1, 1)), (b, b, b), p)
    scale_ty = fm.DistributedType(fm.tensor_type('float32', (16,)), (s,), p)
    builder = fm.IRBuilder(dialect='high_level', stage='attention_gate_fused')
    def op(name, inputs, attrs, id):
        q = get_definition(name).prepare(inputs, attrs)
        return builder.call(name, inputs, q.result_type, id=id, attrs=q.attrs, effect=q.effect)
    def view(x, t, id):
        return op('distributed.sharded_view', (x,), {'new_type': t}, id)
    params = tuple(builder.var(n, t, id=n) for n, t in (
        ('h', broad), ('p', stats_ty), ('rhs', ty), ('gamma', scale_ty), ('bias', scale_ty)))
    local = view(params[0], ty, 'h_local')
    norm = op('nn.norm_apply', (local, params[1], params[3], params[4]),
              {'axis': 1, 'epsilon': 1e-6, 'use_mean': False}, 'norm')
    raw = op('math.add', (local, params[2]), {}, 'raw')
    out = view(raw, broad, 'raw_return')
    stats = op('nn.norm_stats', (out,), {'axis': 1, 'use_mean': False}, 'stats_out')
    fields = (out.type, stats.type, norm.type)
    builder.function('worker', params, (out, stats, norm), attrs={'reusable': True, 'noinline': True})
    seed = builder.var('seed', ty, id='seed')
    rhs = builder.var('addend', ty, id='addend')
    gamma = builder.var('scale', scale_ty, id='scale')
    bias = builder.var('zero', scale_ty, id='zero')
    h = view(seed, broad, 'seed_b')
    st = op('nn.norm_stats', (h,), {'axis': 1, 'use_mean': False}, 'seed_stats')
    norms = []
    for i in range(layers):
        c = builder.call('builtin.call', (h, st, rhs, gamma, bias), fm.TupleType(fields), id=f'call{i}', attrs={'callee': 'worker'})
        h = op('builtin.get_item', (c,), {'index': 0}, f'h{i}')
        st = op('builtin.get_item', (c,), {'index': 1}, f'p{i}')
        norms.append(op('builtin.get_item', (c,), {'index': 2}, f'n{i}'))
    end = view(h, ty, 'last_local')
    norms.append(op('nn.norm_apply', (end, st, gamma, bias), {'axis': 1, 'epsilon': 1e-6, 'use_mean': False}, 'final'))
    outputs = builder.call('builtin.tuple', norms, fm.TupleType(tuple(n.type for n in norms)), id='norms')
    builder.function('main', (seed, rhs, gamma, bias), (outputs,))
    return fm.verify_module(builder.build(entry='main'))


def run(m):
    return transform(m, function_name='worker', stats_parameter_index=1, stats_output_index=1)


def test_complete_formal_actual_tuple_seed_final_channel_and_original_raw():
    before = fixture(); after = run(before)
    assert after.node_map['raw'] == before.node_map['raw']
    assert after.node_map['norms'].type == before.node_map['norms'].type
    for name in ('p', 'stats_out', 'seed_stats', 'p0', 'p1'):
        assert after.node_map[name].type.partial.axes == (0, 1)
    assert after.node_map['stats_out'].inputs == ('raw',)
    assert after.node_map['seed_stats'].inputs == ('seed',)
    for name in ('norm', 'final'):
        n = after.node_map[name]
        assert n.op == 'ntt.gather_reduce_norm_apply'
        assert n.attrs['has_bias'] is True
        assert n.attrs['materialized_stats_type'] == before.node_map['p'].type
    assert run(after) == after


def test_emit_reload_and_default_threading_unaffected(tmp_path):
    from triton.flagmega.passes.functions.thread_norm_stats import thread_norm_stats_across_function_boundaries
    before = fixture()
    assert thread_norm_stats_across_function_boundaries(before) == before
    after = run(before)
    fm.emit_module(after, tmp_path/'module.py')
    assert fm.load_module(tmp_path/'module.py') == after


@pytest.mark.parametrize('order', ['tree', 'tile_fma', 'group_fma'])
def test_ordered_original_sum_is_not_silently_reassociated(order):
    m = fixture()
    m = replace(m, nodes=tuple(replace(n, attrs={**n.attrs, 'reduction_order': order,
                                               **({'reduction_block_size': 16} if order == 'tile_fma' else {})})
                               if n.op == 'nn.norm_stats' else n for n in m.nodes))
    with pytest.raises(IRSchemaError, match='Ordered'):
        run(m)


@pytest.mark.parametrize('output', ['p1', 'call0'])
def test_public_scalar_stats_or_whole_tuple_rejected(output):
    m = fixture()
    m = replace(m, functions=tuple(replace(f, outputs=(*f.outputs, output)) if f.name == 'main' else f for f in m.functions))
    with pytest.raises(IRSchemaError, match='Public|exported'):
        run(m)


def test_unrelated_stats_consumer_rejected():
    m = fixture()
    node = m.node_map['p1']
    user = fm.Node('unrelated', 'math.add', (node.id, node.id), node.type)
    m = replace(m, nodes=(*m.nodes, user))
    with pytest.raises(IRSchemaError, match='shared consumer'):
        run(m)


def test_external_b_scalar_actual_rejected():
    m = fixture(); ty = m.node_map['seed_stats'].type
    external = fm.Node('external_stats', 'builtin.var', (), ty, attrs={'name': 'external_stats'})
    m = replace(m, nodes=(external, *(replace(n, inputs=(n.inputs[0], external.id, *n.inputs[2:])) if n.id == 'call0' else n for n in m.nodes)),
        functions=tuple(replace(f, parameters=(*f.parameters, external.id)) if f.name == 'main' else f for f in m.functions))
    with pytest.raises(IRSchemaError, match='closed'):
        run(m)


def test_no_owner_local_source_rejected():
    m = fixture()
    broad = m.node_map['h'].type
    m = replace(m, nodes=tuple(replace(n, inputs=('h',)) if n.id == 'stats_out' else n for n in m.nodes))
    with pytest.raises(IRSchemaError, match='owner-local'):
        run(m)


def test_unselected_tuple_fields_unchanged_with_28_calls():
    before = fixture(28); after = run(before)
    for i in range(28):
        assert after.node_map[f'call{i}'].type.fields[0] == before.node_map[f'call{i}'].type.fields[0]
        assert after.node_map[f'call{i}'].type.fields[2] == before.node_map[f'call{i}'].type.fields[2]
        assert after.node_map[f'n{i}'] == before.node_map[f'n{i}']
    assert len(after.functions) == len(before.functions) == 2


@pytest.mark.parametrize('name', ['raw_return', 'stats_out', 'p1', 'norm'])
def test_fused_numeric_views_and_stats_channel_are_not_storage_aliases(name):
    from triton.flagmega.ir.fusion import Fusion
    m = fixture()
    n = m.node_map[name]
    arg = fm.Node('value', 'builtin.var', (), n.type, attrs={'name': 'value'})
    doubled = fm.Node('double', 'math.add', (arg.id, arg.id), n.type)
    fusion = Fusion('numeric_double', (arg, doubled), doubled.id)
    n = replace(n, attrs={**n.attrs, 'post_ops': (fusion,)})
    m = fm.verify_module(replace(m, nodes=tuple(n if old.id == name else old for old in m.nodes)))
    with pytest.raises(IRSchemaError, match='owner-local|Fused'):
        run(m)


def test_preops_on_broadcast_view_is_real_arithmetic():
    from triton.flagmega.ir.fusion import Fusion
    m = fixture(); view = m.node_map['raw_return']; source = m.node_map[view.inputs[0]]
    arg = fm.Node('value', 'builtin.var', (), source.type, attrs={'name': 'value'})
    doubled = fm.Node('double', 'math.add', (arg.id, arg.id), source.type)
    view = replace(view, attrs={**view.attrs, 'pre_ops': {'value': Fusion('double', (arg, doubled), doubled.id)}})
    m = fm.verify_module(replace(m, nodes=tuple(view if n.id == view.id else n for n in m.nodes)))
    with pytest.raises(IRSchemaError, match='owner-local'):
        run(m)


@pytest.mark.parametrize('numeric', [False, True])
def test_identity_statistics_view_requires_no_numeric_fusion(numeric):
    from triton.flagmega.ir.fusion import Fusion
    m = fixture(); source = m.node_map['p1']
    arg = fm.Node('value', 'builtin.var', (), source.type, attrs={'name': 'value'})
    square = fm.Node('square', 'math.mul', (arg.id, arg.id), source.type)
    identity = fm.Node('stats_identity', 'tensors.reshape', (source.id,), source.type,
                       attrs={'shape': (1, 1, 1),
                              **({'post_ops': (Fusion('square', (arg, square), square.id),)} if numeric else {})})
    nodes = []
    for n in m.nodes:
        if n.id == 'final':
            nodes.append(identity)
            n = replace(n, inputs=(n.inputs[0], identity.id, *n.inputs[2:]))
        nodes.append(n)
    m = fm.verify_module(replace(m, nodes=tuple(nodes)))
    if numeric:
        with pytest.raises(IRSchemaError, match='Fused'):
            run(m)
    else:
        after = run(m)
        assert after.node_map['stats_identity'].type.partial.axes == (0, 1)
        assert after.node_map['final'].inputs[0] == 'stats_identity'
