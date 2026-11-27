# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
from dataclasses import replace
from types import SimpleNamespace
import pytest
import torch

from triton.flagmega import ir as fm
from triton.flagmega.errors import IRSchemaError
from triton.flagmega.ir.fusion import Fusion
from triton.flagmega.ir.ops.core import get_definition
from triton.flagmega.passes.vector_cast_distribution import commute_vectorized_widening_before_broadcast as commute


def fixture(*, input_dtype='bfloat16', output_dtype='float32', output_lanes=(4,)):
    mesh = fm.Placement((2, 2), 'yx', 'bb')
    b, s = fm.SBP.broadcast(), fm.SBP.split_contiguous((0, 1), granularity=2)
    ty = fm.DistributedType(fm.tensor_type(fm.vector_type(input_dtype, (4,)), (1, 8)), (b, s), mesh)
    builder = fm.IRBuilder(dialect='high_level', stage='attention_gate_fused')
    x = builder.var('x', ty, id='x')
    def op(name, inputs, attrs, id):
        prepared = get_definition(name).prepare(inputs, attrs)
        return builder.call(name, inputs, prepared.result_type, id=id, attrs=prepared.attrs, effect=prepared.effect)
    view = op('distributed.sharded_view', (x,), {'new_type': replace(ty, axis_policies=(b, b))}, 'view')
    cast = op('ntt.vectorized_cast', (view,), {'new_type': fm.vector_type(output_dtype, output_lanes),
              'vectorize_axes': (1,), 'rounding_contract': 'preserve'}, 'cast')
    # A second view consumer must not become F32 or disappear.
    other = op('math.add', (view, view), {}, 'other')
    builder.function('main', (x,), (cast, other))
    return fm.verify_module(builder.build(entry='main'))


def test_true_cast_before_view_preserves_all_coordinates_consumers_and_rounding(tmp_path):
    before = fixture(); after = commute(before, cast_id='cast')
    owner = after.node_map['cast.owner_widening']
    assert owner.op == 'ntt.vectorized_cast' and owner.inputs == ('x',)
    assert owner.attrs == before.node_map['cast'].attrs
    assert owner.type.axis_policies == before.node_map['x'].type.axis_policies
    assert after.node_map['cast'].type == before.node_map['cast'].type
    assert after.node_map['cast'].op == 'distributed.sharded_view'
    assert after.node_map['view'] == before.node_map['view']
    assert after.node_map['other'] == before.node_map['other']
    assert after.functions == before.functions
    values = torch.arange(-16, 16, dtype=torch.bfloat16).reshape(1, 8, 4)
    values[0, 4, 0] = -0.0
    context = SimpleNamespace(types={n.id: n.type for n in (*before.nodes, *after.nodes)},
                              torch_dtype=lambda dtype: getattr(torch, dtype.value))
    definition = get_definition('ntt.vectorized_cast')
    expected = definition.evaluate(before.node_map['cast'], (values,), context)
    actual = torch.cat([definition.evaluate(owner, (part,), context)
                        for part in values.split(2, dim=1)], dim=1)
    assert torch.equal(expected.view(torch.int32), actual.view(torch.int32))
    fm.emit_module(after, tmp_path/'commuted.py')
    assert fm.load_module(tmp_path/'commuted.py') == after


@pytest.mark.parametrize('target', ['view', 'cast'])
@pytest.mark.parametrize('kind', ['pre_ops', 'post_ops'])
def test_numeric_fusion_is_not_an_alias(target, kind):
    m = fixture(); n = m.node_map[target]
    t = m.node_map[n.inputs[0]].type if kind == 'pre_ops' else n.type
    arg = fm.Node('arg', 'builtin.var', (), t, attrs={'name': 'arg'})
    doubled = fm.Node('double', 'math.add', (arg.id, arg.id), t)
    body = Fusion('double', (arg, doubled), doubled.id)
    attrs = {'value': body} if kind == 'pre_ops' else (body,)
    m = fm.verify_module(replace(m, nodes=tuple(replace(q, attrs={**q.attrs, kind: attrs})
                                                if q.id == target else q for q in m.nodes)))
    with pytest.raises(IRSchemaError, match='unfused pure'):
        commute(m, cast_id='cast')


@pytest.mark.parametrize('kwargs', [{'input_dtype': 'float32', 'output_dtype': 'bfloat16'},
                                   {'output_lanes': (2,)}])
def test_narrowing_and_lane_repacking_are_out_of_scope(kwargs):
    with pytest.raises(IRSchemaError, match='same-lane'):
        commute(fixture(**kwargs), cast_id='cast')


def test_missing_cast_is_explicit_and_nonselected_graph_is_unchanged():
    m = fixture()
    with pytest.raises(IRSchemaError, match='vectorized cast'):
        commute(m, cast_id='other')
    assert m == fixture()


def test_nonbroadcast_target_is_not_implicitly_redistributed():
    m = fixture()
    # Use a different legal sharding, preserving all logical elements.
    ty = replace(m.node_map['view'].type, axis_policies=(fm.SBP.broadcast(),
                 fm.SBP.split_contiguous((0,), granularity=4)))
    nodes = []
    for n in m.nodes:
        if n.id == 'view': n = replace(n, type=ty, attrs={'new_type': ty})
        elif n.id == 'cast': n = replace(n, type=replace(n.type, axis_policies=ty.axis_policies))
        elif n.id == 'other': n = replace(n, type=ty)
        nodes.append(n)
    m = fm.verify_module(replace(m, nodes=tuple(nodes)))
    with pytest.raises(IRSchemaError, match='before broadcast'):
        commute(m, cast_id='cast')
