# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Native packed attention consumers retain their physical query expression."""
from dataclasses import replace
from math import prod
import pytest
import torch
from triton.flagmega import ir as fm
from triton.flagmega.evaluator import DictWeightResolver,TorchEvaluator
from triton.flagmega.ir.ops.nn._paged_attention_state import PagedAttentionStateConfig,create_paged_attention_state
from triton.flagmega.passes.auto_distributed.policy import lower_vectorization_contracts


def graph(mode,lanes,sharded,large=False,*,external_plain=False,scalar_reference=False,levels='bc'):
    mesh=fm.Placement((2,2),'yx',levels);b=fm.SBP.broadcast();heads=8 if sharded else 2
    width=32 if mode=='expand_value' else 16
    packets=heads*width//prod(lanes)
    fp=fm.SBP.split_contiguous((1,),packets//2) if sharded else b
    hp=fm.SBP.split_contiguous((1,),heads//2) if sharded else b
    flat_type=fm.DistributedType(fm.tensor_type(fm.vector_type('bfloat16',lanes),(1,packets)),(b,fp),mesh)
    config=PagedAttentionStateConfig(1,1,16,num_blocks=2)
    class Graph(fm.Module):
        def forward(self):
            x=self.input('x',flat_type.tensor if external_plain else flat_type,id='x');value=x
            if external_plain:value=fm.F.distributed.force_boxing(x,flat_type)
            compute=fm.F.math.vectorized_unary(value,unary_op='sigmoid',name='query.compute',metadata={
                'vectorized_from':'math.sigmoid','vectorization_root':'query','vectorization_semantic_id':'query',
                'vectorization_internal':True,'vectorization_candidate':'vectorization.last_axis',
                'vector_axes':(1,)*len(lanes),'vector_lanes':lanes})
            if scalar_reference:
                scalar=fm.F.tensors.unpack(compute,axes=(1,)*len(lanes))
                query=fm.F.tensors.reshape(scalar,(1,heads,width),name='heads.scalar')
            else:
                query=fm.F.tensors.reshape(compute,(1,heads,width//prod(lanes)),name='heads.physical',metadata={
                    'vectorized_from':'tensors.reshape','vectorization_root':'heads','vectorization_semantic_id':'heads',
                    'vectorization_internal':True,'vectorization_candidate':'vectorization.propagated',
                    'vector_axes':(2,)*len(lanes),'vector_lanes':lanes,'vectorization_inputs':('query',),
                    'vectorization_attrs':{'shape':(1,heads,width)}})
            assert query.type.axis_policies==(b,hp,b)
            parameters=[x]
            if mode=='attention':
                kt=fm.DistributedType(fm.tensor_type('int32',(1,3)),(b,b),mesh)
                topk=self.input('topk',kt.tensor if external_plain else kt,id='topk');indices=topk
                if external_plain:indices=fm.F.distributed.force_boxing(topk,kt)
                state=self.input('state',config.ref_type,id='state');parameters.extend((topk,state))
                layer=fm.F.builtin.scalar_const(fm.tensor_type('int32',()),0)
                result=fm.F.nn.dsa_paged_attention(query,indices,state,layer,scale=.25,layout=('seq','head','dim'),
                    hidden_size=heads*16,probability_rounding='online_bf16')
            else:
                wp=fm.SBP.split_contiguous((1,),heads//2*24) if sharded else b
                wt=fm.DistributedType(fm.tensor_type('bfloat16',(heads*24,32)),(wp,b),mesh)
                w=self.input('w',wt.tensor if external_plain else wt,id='w');weight=w
                if external_plain:weight=fm.F.distributed.force_boxing(w,wt)
                parameters.append(w)
                result=fm.F.nn.mla_projection(query,weight,fm.F.builtin.none(),direction=mode,qk_head_dim=16,
                    value_head_dim=8,latent_dim=32,output_lanes=(8,))
            if external_plain:result=fm.F.distributed.force_boxing(result,result.type.tensor)
            outputs=[result]
            if large:
                unused=x
                for _ in range(20_001):unused=fm.F.math.add(unused,x)
                outputs.append(unused)
            self.function('main',parameters,outputs)
    return Graph(dialect='ntt',stage='distributed',entry='main',metadata={'auto_distribution':{'placement':mesh.to_data()}}).build(),config


@pytest.mark.parametrize('mode',['absorb_query','expand_value','attention'])
@pytest.mark.parametrize('lanes',[(8,),(2,4)])
@pytest.mark.parametrize('sharded',[False])
def test_native_attention_keeps_packed_query_graph(tmp_path,mode,lanes,sharded):
    source,config=graph(mode,lanes,sharded)
    result=fm.verify_module(lower_vectorization_contracts(source))
    assert not any(n.op in {'tensors.pack','tensors.unpack'} for n in result.nodes)
    assert result.node_map['heads.physical']==source.node_map['heads.physical']
    consumer=next(n for n in result.nodes if n.op==('nn.dsa_paged_attention' if mode=='attention' else 'nn.mla_projection'))
    assert consumer.inputs[0]=='heads.physical'
    restored=fm.load_module(fm.emit_module(result,tmp_path/'restored.py'));assert restored.semantic_hash==result.semantic_hash
    # The evaluator executes the complete logical tensor independently of its
    # distributed physical owners. Exact values/order must survive lowering.
    flat=source.node_map['x'].type.tensor;size=flat.shape[1].fixed_value
    x=(torch.arange(size*prod(lanes)).float().remainder(17)-8).div(8).bfloat16().reshape(1,size,*lanes)
    values={'x':x}
    if mode=='attention':
        state=create_paged_attention_state(config,device='cpu');state.seq_lens.fill_(3);state.slot_mapping.fill_(2)
        state.kv_caches.copy_((torch.arange(state.kv_caches.numel()).remainder(23).float()-11).div(8).bfloat16().reshape_as(state.kv_caches))
        values.update(topk=torch.tensor([[0,1,2]],dtype=torch.int32),state=state)
    else:
        wt=source.node_map['w'].type.tensor
        values['w']=(torch.arange(prod(d.fixed_value for d in wt.shape)).remainder(19).float()-9).div(16).bfloat16().reshape(tuple(d.fixed_value for d in wt.shape))
    evaluator=TorchEvaluator(DictWeightResolver({}))
    expected=evaluator.run(source,values);actual=evaluator.run(restored,values)
    torch.testing.assert_close(actual,expected,rtol=0,atol=0)


@pytest.mark.parametrize('mode',['absorb_query','attention'])
def test_large_native_attention_uses_the_same_representation_rule(mode):
    source,_=graph(mode,(2,4),False,large=True);assert len(source.nodes)>20_000
    result=fm.verify_module(lower_vectorization_contracts(source))
    assert result.node_map['heads.physical']==source.node_map['heads.physical']
    assert not any(n.op in {'tensors.pack','tensors.unpack'} for n in result.nodes)
