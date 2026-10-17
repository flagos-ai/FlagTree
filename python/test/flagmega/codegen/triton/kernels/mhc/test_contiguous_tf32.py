# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Explicit SM90 projection geometry and independent native fixed gold."""
from dataclasses import replace

import pytest
import torch
from triton.flagmega import ir as fm
from triton.flagmega.artifacts import write_artifact
from triton.flagmega.codegen.triton.candidates.core import TritonCandidateContext
from triton.flagmega.codegen.triton.candidates.routing import MhcCandidateProvider
from triton.flagmega.compiler import Compiler
from triton.flagmega.errors import CodegenError, IRSchemaError
from triton.flagmega.runtime import load
from triton.flagmega.targets import NvidiaSm90Target
from triton.flagmega.targets.iluvatar_bi_v150 import IluvatarBiV150Target

# Unmodified vLLM 0.31.0 SM90 tf32_hc_prenorm_gemm + mhc_pre_tilelang,
# H64/HC4/split1/20 Sinkhorn sweeps; deterministic integer/power-of-two inputs.
GOLD = {'post': [1.1434571743011475, 0.9135536551475525, 0.7540357112884521, 0.8904325366020203], 'comb': [0.23949341475963593, 0.2306642085313797, 0.24881376326084137, 0.28102758526802063, 0.2669326364994049, 0.2531236708164215, 0.24162782728672028, 0.23831483721733093, 0.26648664474487305, 0.2555241286754608, 0.24940037727355957, 0.22858789563179016, 0.22708626091480255, 0.2606870234012604, 0.2601570188999176, 0.25206872820854187], 'collapsed': [0.2275390625, -0.04345703125, -0.56640625, 0.40234375, 0.62890625, -0.51171875, -0.16796875, 0.8046875, -0.21484375, -0.73828125, 0.2314453125, 0.455078125, -0.68359375, -0.33984375, 0.6328125, -0.38671875, -0.28515625, 0.060302734375, 0.28515625, 0.0135498046875, -0.51171875, 0.4609375, 0.68359375, -0.45703125, -0.11083984375, 0.859375, -0.158203125, -0.68359375, 0.2890625, 0.51171875, -0.62890625, -0.283203125, 0.6875, -0.330078125, -0.228515625, 0.1171875, 0.341796875, 0.07080078125, -0.453125, 0.515625, 0.7421875, -0.3984375, -0.053955078125, 0.91796875, -0.1005859375, -0.625, 0.345703125, 0.5703125, -0.5703125, -0.2255859375, 0.74609375, -0.271484375, -0.796875, 0.1748046875, 0.3984375, -0.7421875, -0.396484375, 0.57421875, -0.443359375, -0.341796875, 0.003204345703125, 0.2275390625, -0.04345703125, -0.56640625]}


def graph(*, hc=4, hidden=64, splits=1, lanes=()):
    class Graph(fm.Module):
        def forward(self):
            kinds = [('streams', 'bfloat16', (1,hc,hidden)),
                     ('projection', 'float32', (hc*(hc+2),hc*hidden)),
                     ('scale','float32',(3,)), ('base','float32',(hc*(hc+2),))]
            v = {name:self.input(name,fm.tensor_type(dtype,shape),id=name) for name,dtype,shape in kinds}
            result = fm.F.nn.mhc_pre(v['streams'],v['projection'],v['base'],v['scale'],epsilon=1e-5,
                sinkhorn_iterations=20,sinkhorn_epsilon=1e-6,projection_reduction='contiguous_tf32',
                projection_splits=splits,collapse_reduction='sequential_fma',output_lanes=lanes)
            post,comb,collapsed = fm.F.tensors.get_items(result,0,1,2)
            if lanes:
                collapsed = fm.F.tensors.unpack(collapsed,axis=-1)
            self.function('main',tuple(v.values()),(post,comb,collapsed))
    return Graph(dialect='high_level',stage='imported',entry='main').build()


def test_contract_edit_resume_and_packed_output(tmp_path):
    source = graph(lanes=(8,))
    restored = fm.load_module(fm.emit_module(source,tmp_path/'projection.py'))
    assert restored.semantic_hash == source.semantic_hash
    node = next(n for n in restored.nodes if n.op=='nn.mhc_pre')
    assert node.type.fields[2].dtype == fm.VectorType(fm.DType.BFLOAT16,(8,))
    assert node.attrs['projection_reduction']=='contiguous_tf32'


@pytest.mark.parametrize('kwargs',[{'hc':5},{'hidden':17},{'splits':8}])
def test_incomplete_geometry_is_rejected(kwargs):
    with pytest.raises(IRSchemaError,match='contiguous_tf32'):
        graph(**kwargs)


@pytest.mark.parametrize('target_type',[NvidiaSm90Target,IluvatarBiV150Target])
def test_selection_requires_sm90(target_type):
    source,target=graph(),target_type()
    node=next(n for n in source.nodes if n.op=='nn.mhc_pre')
    context=TritonCandidateContext(source,target,{}, {}, {},frozenset(),True,target.triton_implementation_model)
    if target_type is IluvatarBiV150Target:
        with pytest.raises(CodegenError,match='SM90 WGMMA'):
            MhcCandidateProvider().propose(node,context)
    else:
        assert MhcCandidateProvider().propose(node,context).default_candidate=='tir.mhc_pre.contiguous_tf32'


@pytest.mark.cuda
@pytest.mark.parametrize('warps,lanes',[(4,()),(8,(8,))])
def test_device_projection_matches_independent_native_gold(tmp_path,warps,lanes):
    compiler=Compiler();target=NvidiaSm90Target();base=target.triton_implementation_model
    catalog=replace(base,implementations=tuple(replace(i,parameters={**i.parameters,'compute_num_warps':warps})
        if i.id=='tir.mhc_pre.contiguous_tf32' else i for i in base.implementations))
    compiler.target=NvidiaSm90Target(triton_implementation_model=catalog)
    compiled=compiler.compile(graph(lanes=lanes)).module
    artifact=write_artifact(compiled,tmp_path/'artifact',target='nvidia-sm90',emit_executable=True)
    runtime=load(artifact,device='cuda:0')
    try:
        values={'streams':(((torch.arange(256)*17)%61)-30).float().mul_(2**-5).bfloat16().reshape(1,4,64),
                'projection':(((torch.arange(24*256)*29)%67)-33).float().mul_(2**-9).reshape(24,256),
                'scale':torch.tensor([1.75,-.75,.3]),'base':torch.arange(24).float()/64-.125}
        values={k:v.cuda() for k,v in values.items()}
        expected=tuple(torch.tensor(GOLD[key],dtype=dtype).reshape(shape) for key,dtype,shape in
                       [('post',torch.float32,(1,4)),('comb',torch.float32,(1,4,4)),
                        ('collapsed',torch.bfloat16,(1,64))])
        outputs=tuple(torch.empty_like(v,device='cuda') for v in expected)
        specs=[s for s in runtime.external_arguments if s['role']=='result']
        values.update({s['value']:v for s,v in zip(specs,outputs,strict=True)})
        args=tuple(values[s['value']] for s in runtime.external_arguments)
        runtime.prepare(*args);runtime.run_into(*args)
        for actual,wanted in zip(outputs,expected,strict=True):
            torch.testing.assert_close(actual.cpu(),wanted,rtol=0,atol=0)
        assert runtime.resource_report['spill_bytes']==0
        assert runtime.resource_report['num_warps']==warps
    finally:
        runtime.close()
