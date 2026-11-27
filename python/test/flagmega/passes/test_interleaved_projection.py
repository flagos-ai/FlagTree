# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Interleaved projection is a single producer with independent tuple fields."""
from dataclasses import replace
import pytest
import torch
from triton.flagmega import ir as fm
from triton.flagmega.errors import IRSchemaError
from triton.flagmega.evaluator import TorchEvaluator,DictWeightResolver
from triton.flagmega.passes.interleaved_projection import form_interleaved_projection
from triton.flagmega.passes.functions.post_boundary_pack import post_function_boundary_pack_propagation
from triton.flagmega.targets import NvidiaSm120Target
from python.test.flagmega.codegen.cpp.test_native_quantized_kernels import packed_rows


def graph(*, heads=24, width=256, lanes=(), shared=False, bad=False, large=False):
    class Graph(fm.Module):
        def forward(self):
            x=self.input('x',fm.tensor_type('bfloat16',(1,256)),id='x')
            w=self.input('w',fm.tensor_type('uint8',(heads*2*width,136)),id='w')
            s=self.input('s',fm.tensor_type('float32',((heads*2*width+127)//128,2)),id='s')
            p=fm.F.math.block_scaled_matmul(x,w,s,weight_block_n=128,weight_block_k=128,weight_encoding='IQ4_XS',name='p')
            view=fm.F.tensors.reshape(p,shape=(1,heads,2,width),name='view')
            fields=[]
            for f in range(2):
                slice=fm.F.tensors.slice(view,starts=(f,),ends=(f+1,),axes=(2,),steps=(-1,) if bad else (1,),name=f'slice{f}')
                if bad: # Some python slice ranges become empty: retain the unsupported slice itself.
                    fields.append(slice);continue
                field=fm.F.tensors.reshape(slice,shape=(1,heads,width),name=f'field{f}')
                if lanes:field=fm.F.tensors.pack(field,axes=(-1,)*len(lanes),lanes=lanes,name=f'pack{f}')
                fields.append(field)
            self.function('main',(x,w,s),(*fields,p) if shared else tuple(fields))
    m=Graph(dialect='high_level',stage='imported',entry='main').build()
    if large:
        # Reach the separate >20k propagation route while leaving the tested
        # expensive producer/output graph intact. Private scalar constants
        # carry no effects and are removable, not duplicated projections.
        constants=tuple(fm.Node(f'padding{i}','builtin.splat_const',(),fm.tensor_type('float32',(1,)),attrs={'value':0.}) for i in range(20001))
        m=replace(m,nodes=(*constants,*m.nodes))
    return m


@pytest.mark.parametrize('lanes',[(8,),(2,4)])
@pytest.mark.parametrize('large',[False,True])
def test_tuple_packing_single_producer_all_head_scalar_order_and_fixed_point(tmp_path,lanes,large):
    old=graph(lanes=lanes,large=large)
    formed=form_interleaved_projection(old)
    packed=post_function_boundary_pack_propagation(formed,NvidiaSm120Target())
    assert sum(n.op=='nn.interleaved_projection' for n in packed.nodes)==1
    assert not any(n.op in {'tensors.pack','tensors.unpack','tensors.slice'} for n in packed.nodes)
    p=next(n for n in packed.nodes if n.op=='nn.interleaved_projection')
    assert p.attrs['output_lanes']==(lanes,lanes)
    assert post_function_boundary_pack_propagation(packed,NvidiaSm120Target()).semantic_hash==packed.semantic_hash
    # Each original row is a unique projection sentinel. Native code8=1,
    # scale1; row d sets its independent head/field identity before BF16 store.
    raw=torch.zeros((24*2*256,136),dtype=torch.uint8)
    raw[:,2:4]=torch.tensor([0xaa,0xaa],dtype=torch.uint8);raw[:,4:8]=0x11;raw[:,8:]=0x88
    d=(torch.arange(raw.shape[0]).float()+1).div(2048).half()
    raw[:,:2]=d.view(torch.uint8).reshape(-1,2)
    x=torch.zeros(1,256,dtype=torch.bfloat16);x[0,0]=1
    out=TorchEvaluator(DictWeightResolver({})).run(packed,{'x':x,'w':raw,'s':torch.ones(96,2)})
    expected=d.float().bfloat16().reshape(1,24,2,256)
    for f in range(2):assert torch.equal(out[f].reshape(1,24,256),expected[:,:,f,:])
    assert fm.load_module(fm.emit_module(packed,tmp_path/'qg.py')).semantic_hash==packed.semantic_hash


@pytest.mark.parametrize('shared,bad',[(True,False),(False,True)])
def test_unproven_or_observed_full_producer_cannot_duplicate_projection(shared,bad):
    old=graph(shared=shared,bad=bad)
    assert form_interleaved_projection(old).semantic_hash==old.semantic_hash


def test_whole_head_split_scales_original_rows_and_rejects_inside_head():
    m=form_interleaved_projection(graph())
    p=m.node_map['p'];inputs=tuple(m.node_map[n] for n in p.inputs)
    placement=fm.Placement((8,),'x','b');b=fm.SBP.broadcast()
    kinds=(fm.DistributedType(inputs[0].type,(b,b),placement),
           fm.DistributedType(inputs[1].type,(fm.SBP.split_block_cyclic((0,),512),b),placement),
           fm.DistributedType(inputs[2].type,(b,b),placement))
    nodes=tuple(replace(n,type=t) for n,t in zip(inputs,kinds))
    result=fm.get_definition(p.op).infer_type(nodes,p.attrs)
    for field in result.fields:assert field.axis_policies==(b,fm.SBP.split_block_cyclic((0,),1),b)
    bad=replace(nodes[1],type=fm.DistributedType(inputs[1].type,(fm.SBP.split_block_cyclic((0,),256),b),placement))
    with pytest.raises(IRSchemaError,match='whole heads'):fm.get_definition(p.op).infer_type((nodes[0],bad,nodes[2]),p.attrs)


def test_invalid_field_packet_width_and_native_encoding_reject_early():
    m=form_interleaved_projection(graph());p=m.node_map['p'];inputs=tuple(m.node_map[n] for n in p.inputs)
    with pytest.raises(IRSchemaError,match='width.*output_lanes'):
        fm.get_definition(p.op).prepare(inputs,{**p.attrs,'output_lanes':((3,),())})
    with pytest.raises(IRSchemaError,match='native IQ4'):
        fm.get_definition(p.op).prepare(inputs,{**p.attrs,'weight_encoding':'float8'})
