# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Normal GU rewrite retains genuine readonly roots and reusable stage order."""
from dataclasses import replace

import pytest
from triton.flagmega import ir as fm
from triton.flagmega.errors import IRSchemaError
from triton.flagmega.passes.tir.split_k_dense_glu import rewrite_split_k_dense_glu, split_k_dense_glu_policy


def packed_graph(calls=2, *, mutable_weight=False):
    class Graph(fm.Module):
        def forward(self):
            a_type = fm.tensor_type('bfloat16',(1,2048))
            packed_type = fm.tensor_type('bfloat16',(128,768,2,64))
            a = self.input('formal_a',a_type)
            g, u = [self.input(name,packed_type) for name in ('formal_g','formal_u')]
            before = fm.F.math.add(a,a,name='before')
            gu = fm.F.nn.packed_dense_matmul_glu(before,g,u,round_activation=False,name='decode_layer_mlp_gate_up')
            after = fm.F.math.add(gu,gu,name='after')
            self.function('decode_layer',(a,g,u),(after,),attrs={'reusable':True,'noinline':True,'calling_convention':'device'})
            actual = self.input('actual',a_type)
            outputs = []
            for index in range(calls):
                packed = []
                for side in ('gate','up'):
                    name = f'{side}{index}'
                    kind = fm.tensor_type('bfloat16',(6144,2048))
                    root = self.input(name,kind) if mutable_weight else self.weight(name,kind,source='memory',key=name)
                    r = fm.F.tensors.reshape(root,shape=(768,8,128,2,8),name=name+'.reshape')
                    p = fm.F.tensors.permute(r,axes=(2,0,3,1,4),name=name+'.permute')
                    packed.append(fm.F.tensors.reshape(p,shape=(128,768,2,64),name=name+'.packed',metadata={'packed_from':root.id}))
                outputs.append(fm.F.builtin.call(actual,*packed,callee='decode_layer',result_type=after.type,name=f'call{index}'))
            params = (actual,) + tuple(n for n in self.builder.nodes if mutable_weight and n.op=='builtin.var' and n.id.startswith(('gate','up')))
            self.function('main',params,tuple(outputs))
    return Graph(dialect='nn',stage='unused_functions_removed',entry='main').build()


@pytest.mark.parametrize('calls', [2,28])
def test_normal_rewrite_prunes_obsolete_packing_and_publishes_all_stage_results(tmp_path, calls):
    rewritten = rewrite_split_k_dense_glu(packed_graph(calls))
    assert sum(n.op=='builtin.call' for n in rewritten.nodes)==calls+4
    assert {f.name for f in rewritten.functions}=={'main','decode_layer',*(f'decode_layer.{name}' for name in ('before_gu','gu_projection','gu_epilogue','after_gu'))}
    assert not any(n.id.endswith(('.packed','.permute','.reshape')) for n in rewritten.nodes)
    assert sum(n.op=='ntt.split_k_dense_glu_projection' for n in rewritten.nodes)==1
    assert sum(n.op=='ntt.split_k_dense_glu_epilogue' for n in rewritten.nodes)==1
    recipes = [n for r in rewritten.constant_recipes for n in r.nodes]
    assert sum(n.op=='tensors.concat' for n in recipes)==calls
    assert sum(n.op=='builtin.weight' for n in recipes)==2*calls
    assert fm.load_module(fm.emit_module(rewritten,tmp_path/'rewritten.py')).semantic_hash==rewritten.semantic_hash


def test_runtime_weights_cannot_be_promoted_to_offline_constants():
    with pytest.raises(IRSchemaError,match='immutable logical checkpoint provenance'):
        rewrite_split_k_dense_glu(packed_graph(mutable_weight=True))


def test_unrelated_target_cannot_enable_the_sm90_island():
    from types import SimpleNamespace
    with pytest.raises(IRSchemaError,match='SM90'):
        split_k_dense_glu_policy(SimpleNamespace(codegen_platform='amd',codegen_architecture='gfx942'))
