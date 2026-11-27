# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Statistics consume rounded BF16 data and own exact FP32 widening."""
from dataclasses import replace
import pytest
import torch
from triton.flagmega import ir as fm
from triton.flagmega.evaluator import TorchEvaluator,DictWeightResolver
from triton.flagmega.passes.rewriter import DataflowPass
from triton.flagmega.rules.neutral.fuse_norm_stats_input_cast import fuse_norm_stats_input_cast_rule


def module(*, packed=False, packet=False, order='unspecified', shared=False, source='bfloat16', nonfinal=False):
    class Graph(fm.Module):
        def forward(self):
            dtype=fm.vector_type(source,(8,)) if packed else source
            x=self.input('x',fm.tensor_type(dtype,(2,3,4 if packed else 32)),id='x')
            add=self.input('add',x.type,id='add')
            rounded=fm.F.math.add(x,add,name='rounded')
            if packet:
                cast=fm.F.ntt.vectorized_cast(rounded,fm.vector_type('float32',(2,4)),vectorize_axes=(0 if nonfinal else -1,),name='cast')
                observed=fm.F.tensors.bitcast(cast,fm.vector_type('float32',(4,)),name='packet')
            else:
                cast=fm.F.tensors.cast(rounded,fm.vector_type('float32',(8,)) if packed else 'float32',name='cast')
                observed=cast
            attrs={'reduction_order':order}
            if order=='tile_fma':attrs.update(reduction_group_size=2,reduction_block_size=16,reduction_block_traversal='reverse')
            elif order=='group_fma':attrs.update(reduction_group_size=2,reduction_group_seed_index=1)
            elif order=='tree':attrs.update(reduction_group_size=2,reduction_tree_group_size=4)
            stats=fm.F.nn.norm_stats(observed,axis=-1,use_mean=True,name='stats',**attrs)
            self.function('main',(x,add),(stats,cast) if shared else (stats,))
    return Graph(dialect='high_level',stage='imported',entry='main').build()


@pytest.mark.parametrize('packed,packet',[(False,False),(True,False),(True,True)])
@pytest.mark.parametrize('order',['unspecified','tree','group_fma','tile_fma'])
def test_widening_owner_preserves_exact_statistics_reduction_flags_and_bf16_residual(tmp_path,packed,packet,order):
    old=module(packed=packed,packet=packet,order=order)
    new=DataflowPass('Test',(fuse_norm_stats_input_cast_rule(),)).run(old)
    assert not any(n.op in {'tensors.cast','ntt.vectorized_cast'} for n in new.nodes)
    assert new.node_map['stats'].attrs==old.node_map['stats'].attrs
    assert new.node_map['stats'].type==old.node_map['stats'].type
    shape=(2,3,4,8) if packed else (2,3,32)
    x=torch.randn(shape,generator=torch.Generator().manual_seed(113)).bfloat16()
    add=(-x.float()+0.00391).bfloat16()
    values={'x':x,'add':add}
    eval=TorchEvaluator(DictWeightResolver({}))
    assert torch.equal(eval.run(old,values)[0],eval.run(new,values)[0])
    assert fm.load_module(fm.emit_module(new,tmp_path/'stats.py')).semantic_hash==new.semantic_hash
    assert DataflowPass('Test',(fuse_norm_stats_input_cast_rule(),)).run(new).semantic_hash==new.semantic_hash


def test_shared_exported_conversion_remains_for_its_other_owner():
    old=module(shared=True)
    new=DataflowPass('Test',(fuse_norm_stats_input_cast_rule(),)).run(old)
    assert new.node_map['cast'].op=='tensors.cast'
    assert new.node_map['stats'].inputs==('rounded',)


def test_non_bf16_widening_and_nonfinal_vector_cast_are_rejected():
    old=module(source='float32')
    assert DataflowPass('Test',(fuse_norm_stats_input_cast_rule(),)).run(old).semantic_hash==old.semantic_hash
    old=module(packed=True,packet=True,nonfinal=True)
    assert DataflowPass('Test',(fuse_norm_stats_input_cast_rule(),)).run(old).semantic_hash==old.semantic_hash
