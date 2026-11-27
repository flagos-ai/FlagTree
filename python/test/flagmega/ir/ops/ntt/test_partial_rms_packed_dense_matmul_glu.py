# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Explicit Psum input is an operation contract, never a hidden workspace."""
from triton.flagmega.ir.ops.core import get_definition

def test_partial_rms_glu_is_registered_with_six_explicit_operands():
    definition = get_definition('ntt.partial_rms_packed_dense_matmul_glu')
    assert tuple(p.name for p in definition.input_parameters) == (
        'raw','scale','bias','gate_weight','up_weight','partial_stats')


from dataclasses import replace
import pytest
from triton.flagmega import ir as fm
from triton.flagmega.errors import IRSchemaError


def arguments():
    p, b = fm.Placement((8, 16), 'yx', 'bb'), fm.SBP.broadcast()
    specs = [('raw', fm.tensor_type('float32', (1, 2048))),
             ('scale', fm.tensor_type('bfloat16', (2048,))),
             ('bias', fm.tensor_type('bfloat16', (2048,))),
             ('gate_weight', fm.tensor_type('bfloat16', (128, 768, 2, 64))),
             ('up_weight', fm.tensor_type('bfloat16', (128, 768, 2, 64)))]
    nodes = [fm.Node(name, 'builtin.var', (), fm.DistributedType(ty, (b,) * ty.rank, p), attrs={'name': name}) for name, ty in specs]
    from triton.flagmega.ir.distributed_type import SBPPartial, ReduceOp
    ty = fm.DistributedType(fm.tensor_type('float32', (1, 1, 1)), (b, b, b), p,
                            partial=SBPPartial((0, 1), ReduceOp.SUM))
    nodes.append(fm.Node('partial_stats', 'builtin.var', (), ty, attrs={'name': 'partial_stats'}))
    return tuple(nodes)


def test_explicit_owner_access_and_exact_input_types():
    definition = get_definition('ntt.partial_rms_packed_dense_matmul_glu')
    values = arguments()
    prepared = definition.prepare(values, {'epsilon': 1e-6, 'norm_tree_group_size': 16, 'norm_partial_group_size': 32})
    assert prepared.result_type.tensor == fm.tensor_type('bfloat16', (1, 6144))
    assert definition.input_parameters[5].memory_effect == fm.MemoryEffect.READ.across_partial_owners()
    assert all(p.memory_effect == fm.MemoryEffect.CHIP_READ for p in definition.input_parameters[:3])


@pytest.mark.parametrize('kind', ('broadcast', 'wrong_owners', 'wrong_shape', 'wrong_dtype'))
def test_partial_is_not_a_broadcast_or_unrelated_statistics_argument(kind):
    values = list(arguments())
    ty = values[5].type
    if kind == 'broadcast': ty = replace(ty, partial=None)
    elif kind == 'wrong_owners': ty = replace(ty, partial=replace(ty.partial, axes=(1,)))
    elif kind == 'wrong_shape': ty = replace(ty, tensor=fm.tensor_type('float32', (1, 1, 128)))
    elif kind == 'wrong_dtype': ty = replace(ty, tensor=fm.tensor_type('bfloat16', (1, 1, 1)))
    values[5] = replace(values[5], type=ty)
    with pytest.raises(IRSchemaError, match='PartialRmsPackedDenseMatMulGlu'):
        get_definition('ntt.partial_rms_packed_dense_matmul_glu').prepare(values,
            {'epsilon': 1e-6, 'norm_tree_group_size': 16, 'norm_partial_group_size': 32})


@pytest.mark.parametrize('compute_blocks', (8, 16))
def test_evaluator_consumes_explicit_additive_stats_instead_of_recomputing_raw(tmp_path, compute_blocks):
    import torch
    from triton.flagmega.evaluator import DictWeightResolver, TorchEvaluator
    from triton.flagmega.ir.distributed_type import SBPPartial, ReduceOp
    from triton.flagmega.ir.ops.tensors._k_major import unpack_k_major_weight
    placement, b = fm.Placement((8,), 'x', 'b'), fm.SBP.broadcast()
    shapes = ((1,128),(128,),(128,),(8,8,2,64),(8,8,2,64),(1,1,1))
    dtypes = ('float32','bfloat16','bfloat16','bfloat16','bfloat16','float32')
    names = ('raw','scale','bias','gate_weight','up_weight','partial_stats')
    types = [fm.DistributedType(fm.tensor_type(dtype,shape), (b,)*len(shape), placement)
             for dtype,shape in zip(dtypes,shapes)]
    types[-1] = replace(types[-1], partial=SBPPartial((0,),ReduceOp.SUM))
    types[:5] = [replace(ty, placement=fm.Placement((compute_blocks,), 'x', 'b')) for ty in types[:5]]
    class Graph(fm.Module):
        def __init__(self):
            super().__init__(dialect='high_level',stage='imported',entry='main')
        def forward(self):
            values = [self.input(name,ty,id=name) for name,ty in zip(names,types)]
            out = fm.F.ntt.partial_rms_packed_dense_matmul_glu(*values, epsilon=1e-6,
                norm_tree_group_size=16,norm_partial_group_size=4,round_activation=False,name='out')
            self.function('main',values,(out,))
    module = Graph().build()
    assert fm.load_module(fm.emit_module(module,tmp_path/'partial.py')).semantic_hash == module.semantic_hash
    torch.manual_seed(7204)
    values = {name:torch.randn(shape,dtype=getattr(torch,dtype)) for name,shape,dtype in zip(names,shapes,dtypes)}
    # Deliberately different from raw's square sum: operand six must be used.
    values['partial_stats'] = (2 * values['raw'].square().sum()).reshape(1,1,1)
    norm = (values['raw'] * torch.rsqrt(values['partial_stats'].reshape(1,1)/128 + 1e-6)
            * values['scale'].float() + values['bias'].float()).to(torch.bfloat16)
    gate = torch.nn.functional.linear(norm,unpack_k_major_weight(values['gate_weight'],'k_major_n8_k16'))
    up = torch.nn.functional.linear(norm,unpack_k_major_weight(values['up_weight'],'k_major_n8_k16'))
    expected = (torch.nn.functional.silu(gate.float())*up.float()).to(torch.bfloat16)
    actual, = TorchEvaluator(DictWeightResolver({})).run(module,values)
    torch.testing.assert_close(actual,expected,rtol=0,atol=0)


def mixed_owner_arguments():
    """Complete chip row at256 compute owners, original128 statistic owners."""
    values = list(arguments())
    compute = fm.Placement((16, 16), 'yx', 'bb')
    for index in range(5):
        ty = replace(values[index].type, placement=compute)
        if index in (3, 4):
            policies = list(ty.axis_policies)
            policies[1] = fm.SBP.split(fm.SplitStage((0, 1), fm.ContiguousSplit(3)))
            ty = replace(ty, axis_policies=tuple(policies))
        values[index] = replace(values[index], type=ty)
    return tuple(values)


def test_partial_owner_domain_is_independent_of_compute_block_mesh(tmp_path):
    values = mixed_owner_arguments()
    definition = get_definition('ntt.partial_rms_packed_dense_matmul_glu')
    attrs = {'epsilon': 1e-6, 'norm_tree_group_size': 16,
             'norm_partial_group_size': 32, 'round_activation': False}
    prepared = definition.prepare(values, attrs)
    from triton.flagmega.ir.distributed_type import local_tensor_type
    assert local_tensor_type(prepared.result_type).shape == (fm.DimConst(1), fm.DimConst(24))
    assert values[5].type.placement.hierarchy == (8, 16)
    assert values[5].type.partial.axes == (0, 1)
    assert prepared.result_type.placement.hierarchy == (16, 16)
    builder = fm.IRBuilder(dialect='semantic_tir', stage='attention_gate_fused')
    args = tuple(builder.var(value.id, value.type, id=value.id) for value in values)
    output = builder.call(definition.op_name, args, prepared.result_type, attrs=prepared.attrs,
                          effect=prepared.effect, id='result')
    builder.function('main', args, (output,))
    module = fm.verify_module(builder.build(entry='main'))
    restored = fm.load_module(fm.emit_module(module, tmp_path / 'foreign_partial.py'))
    assert restored.semantic_hash == module.semantic_hash
    assert restored.node_map['partial_stats'].type == values[5].type
    assert definition.input_parameters[5].memory_effect == fm.MemoryEffect.READ.across_partial_owners()
    assert definition.cost_factors(values, prepared.attrs, prepared.result_type) is not None


@pytest.mark.parametrize('kind', ('chip_domain', 'wrong_shape', 'wrong_dtype', 'broadcast',
                                 'wrong_reduction', '256_owners'))
def test_mixed_owner_read_rejects_unrelated_or_materialized_statistics(kind):
    values = list(mixed_owner_arguments())
    partial = values[5].type
    if kind == 'chip_domain':
        partial = replace(partial, placement=fm.Placement((8, 16), 'yx', 'db'))
    elif kind == 'wrong_shape':
        partial = replace(partial, tensor=fm.tensor_type('float32', (1, 1, 128)))
    elif kind == 'wrong_dtype':
        partial = replace(partial, tensor=fm.tensor_type('bfloat16', (1, 1, 1)))
    elif kind == 'broadcast':
        partial = replace(partial, partial=None)
    elif kind == 'wrong_reduction':
        partial = replace(partial, partial=replace(partial.partial, reduce_op=fm.ReduceOp.MAX))
    elif kind == '256_owners':
        partial = replace(partial, placement=fm.Placement((16, 16), 'yx', 'bb'))
    values[5] = replace(values[5], type=partial)
    with pytest.raises(IRSchemaError, match='PartialRmsPackedDenseMatMulGlu'):
        get_definition('ntt.partial_rms_packed_dense_matmul_glu').prepare(values,
            {'epsilon': 1e-6, 'norm_tree_group_size': 16, 'norm_partial_group_size': 32})
