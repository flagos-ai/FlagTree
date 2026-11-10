# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Native compressed tables remain bytes while decoded activation lanes propagate."""
from dataclasses import replace
import pytest
import torch
from triton.flagmega import ir as fm
from triton.flagmega.errors import IRSchemaError
from triton.flagmega.evaluator import TorchEvaluator,DictWeightResolver
from triton.flagmega.passes.functions import post_function_boundary_pack_propagation
from triton.flagmega.targets import NvidiaSm120Target
from python.test.flagmega.codegen.cpp.test_native_quantized_kernels import packed_rows


def graph(encoding,lanes,*,shared=False,mixed=False,large=False):
    size=136 if encoding=='IQ4_XS' else 144
    class Graph(fm.Module):
        def forward(self):
            ids=self.input('ids',fm.tensor_type('int32',(3,)),id='ids')
            weight=self.input('weight',fm.tensor_type('uint8',(5,2*size)),id='weight')
            embedding=fm.F.nn.embedding(ids,weight,weight_encoding=encoding,padding_idx=0,name='embedding')
            result=fm.F.tensors.pack(embedding,lanes,axes=(-1,)*len(lanes),name='packed')
            outputs=[result]
            if mixed:outputs.append(fm.F.tensors.pack(embedding,(4,),axes=(-1,)))
            if shared:outputs.append(embedding)
            self.function('main',(ids,weight),tuple(outputs))
    m=Graph(dialect='high_level',stage='packed',entry='main').build()
    if large:
        m=replace(m,nodes=(*tuple(fm.Node(f'constant{i}','builtin.splat_const',(),fm.tensor_type('float32',(1,)),attrs={'value':0.}) for i in range(20001)),*m.nodes))
    return m


@pytest.mark.parametrize('encoding',['IQ4_XS','Q4_K'])
@pytest.mark.parametrize('lanes',[(8,),(2,4)])
@pytest.mark.parametrize('large',[False,True])
def test_native_embedding_layout_preserves_raw_table_all_users_and_fixed_point(tmp_path,encoding,lanes,large):
    old=graph(encoding,lanes,shared=True,mixed=True,large=large)
    if large:assert len(old.nodes)>20000
    new=post_function_boundary_pack_propagation(old,NvidiaSm120Target())
    embedding,=(n for n in new.nodes if n.op=='nn.embedding')
    assert new.node_map[embedding.inputs[1]].type==old.node_map['weight'].type
    assert new.node_map[embedding.inputs[1]].type.dtype==fm.DType.UINT8
    assert not any(n.op in {'tensors.pack','tensors.unpack'} for n in new.nodes)
    raw,decoded=packed_rows(encoding,rows=5)
    inputs={'ids':torch.tensor([4,0,2],dtype=torch.int32),'weight':torch.from_numpy(raw)}
    oldout=TorchEvaluator(DictWeightResolver({})).run(old,inputs)
    newout=TorchEvaluator(DictWeightResolver({})).run(new,inputs)
    for a,b in zip(oldout,newout):assert torch.equal(a,b)
    assert post_function_boundary_pack_propagation(new,NvidiaSm120Target()).semantic_hash==new.semantic_hash
    assert fm.load_module(fm.emit_module(new,tmp_path/'embedding.py')).semantic_hash==new.semantic_hash


def test_native_embedding_output_lanes_reject_nonintegral_feature_packet():
    m=graph('IQ4_XS',(8,));inputs=(m.node_map['ids'],m.node_map['weight'])
    with pytest.raises(IRSchemaError,match='divisible by output_lanes'):
        fm.get_definition('nn.embedding').prepare(inputs,{'weight_encoding':'IQ4_XS','output_lanes':(3,)})
