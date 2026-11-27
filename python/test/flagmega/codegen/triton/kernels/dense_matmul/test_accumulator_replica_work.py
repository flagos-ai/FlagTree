# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Every independent accumulator N tile executes once per physical owner."""
from dataclasses import replace
from collections import Counter
from itertools import product
from math import prod
from pathlib import Path
import ast,importlib.util
import pytest
from triton.flagmega import ir as fm
from triton.flagmega.artifacts import write_artifact
from triton.flagmega.codegen.triton.templates import TritonTemplateRegistry
from triton.flagmega.compiler import Compiler
from triton.flagmega.options import CompileOptions
from triton.flagmega.passes.tir.fuse_packed_accumulator_cast import fuse_packed_accumulator_casts
from triton.flagmega.selection import apply_plan, override_plan
from triton.flagmega.targets import NvidiaSm90Target
from triton.flagmega.targets.nvidia.dense_matmul import packed_mma_accumulator_implementations


@pytest.fixture(scope='module')
def profile():
    previous=TritonTemplateRegistry.render
    p=(Path(__file__).resolve().parents[6]/'tutorials/flagmega/04-glm5.3-flash-fp8/nvidia-h800/local_optimizations/replica_work.py')
    spec=importlib.util.spec_from_file_location('_accumulator_replica_work_test',p)
    module=importlib.util.module_from_spec(spec)
    try:spec.loader.exec_module(module)
    finally:TritonTemplateRegistry.render=previous
    return module


def compile_module(mesh,*,n=512,k=256,m=1,splits=2,rounded=True):
    b=fm.SBP.broadcast();axis=mesh.rank-1;groups=(n//8+mesh.hierarchy[axis]-1)//mesh.hierarchy[axis]
    n_policy=(fm.SBP.split_contiguous((axis,),groups) if n//8%mesh.hierarchy[axis]==0
              else fm.SBP.split_block_cyclic((axis,),1))
    block_axes=tuple(a for a in range(mesh.rank) if mesh.is_physical_block_axis(a))
    sentinel_groups=(128+prod(mesh.hierarchy[a] for a in block_axes)-1)//prod(mesh.hierarchy[a] for a in block_axes)
    class Graph(fm.Module):
        def forward(self):
            x=self.input('x',fm.DistributedType(fm.tensor_type('bfloat16',(m,k)),(b,b),mesh),id='x')
            w=self.input('w',fm.DistributedType(fm.tensor_type(fm.vector_type('bfloat16',(8,2,8)),(k//16,n//8)),
                (b,n_policy),mesh),id='w')
            no=fm.F.builtin.none()
            value=fm.F.ntt.packed_matmul(x,w,no,no,output_data_type='float32',name='projection',metadata={
                'selected_vectorization':'vectorization.matmul.n','selected_vector_axes':(1,1),
                'selected_vector_lanes':(2,4)})
            if rounded:value=fm.F.ntt.vectorized_cast(value,fm.vector_type('bfloat16',(8,)),vectorize_axes=(1,1),name='rounded')
            sentinel=self.input('sentinel',fm.tensor_type('float32',(128,)),id='sentinel')
            split=fm.DistributedType(sentinel.type,(fm.SBP.split_contiguous(block_axes,sentinel_groups),),mesh)
            local=fm.F.distributed.force_boxing(sentinel,split)
            copied=fm.F.distributed.force_boxing(fm.F.math.add(local,local),sentinel.type)
            self.function('main',(x,w,sentinel),(value,copied))
    source=Graph(dialect='distributed',stage='frozen_constants',entry='main',
        metadata={'auto_distribution':{'placement':mesh.to_data()}}).build()
    identity='tir.dense_matmul.packed_k_major_mma_accumulator'
    class Target(NvidiaSm90Target):
        def propose_tir(self,module):
            result=super().propose_tir(fuse_packed_accumulator_casts(module))
            points=[p for p in result.selection_points if p.kind=='tir' and p.owner in result.node_map
                    and result.node_map[p.owner].op=='ntt.packed_matmul']
            assert len(points)==1,[(p.id,p.owner,result.node_map[p.owner].op) for p in result.selection_points if p.owner in result.node_map]
            return apply_plan(result,override_plan(result,[(points[0].id,identity)],
                policy='test.accumulator-replica-work',rationale='Same ordered K partitions; only independent N tasks may vary.'))
    base=NvidiaSm90Target().triton_implementation_model
    impl,=packed_mma_accumulator_implementations()
    impl=replace(impl,parameters={**impl.parameters,'accumulation_splits':splits})
    catalog=replace(base,implementations=(*base.implementations,impl))
    c=Compiler(CompileOptions(placement=mesh));c.target=Target(triton_implementation_model=catalog).with_placement(mesh)
    return c.compile(source).module


def render(module,path,profile,enabled):
    previous=TritonTemplateRegistry.render
    try:
        TritonTemplateRegistry.render=profile.override if enabled else profile.original_render
        return write_artifact(module,path,target='nvidia-sm90',emit_executable=True)
    finally:TritonTemplateRegistry.render=previous


def n_loop(source):
    matches=[node for node in ast.walk(ast.parse(source)) if isinstance(node,ast.For)
        and isinstance(node.target,ast.Name) and node.target.id=='dense_local_n_start']
    assert len(matches)==1
    return matches[0]


@pytest.mark.parametrize('hierarchy,levels', [((8,16),'bb'),((2,8,16),'cbb'),((8,2,16),'bcb')])
@pytest.mark.parametrize('n',[512,520])
def test_generated_loop_covers_each_owner_tile_exactly_once(tmp_path,profile,hierarchy,levels,n):
    mesh=fm.Placement(hierarchy,'yx' if len(hierarchy)==2 else 'cyx' if levels=='cbb' else 'ycx',levels)
    module=compile_module(mesh,n=n)
    source=(render(module,tmp_path/'artifact',profile,True)/'generated_kernels.py').read_text()
    loop=n_loop(source)
    bound=[compile(ast.Expression(arg),'<N tile loop>','eval') for arg in loop.iter.args]
    counts={}
    owners=tuple(a for a,level in enumerate(levels) if level=='c')+(mesh.rank-1,)
    for coords in product(*(range(size) for size in hierarchy)):
        variables={f'shard_coord{a}':value for a,value in enumerate(coords)}
        if len(coords)==2:variables.update(shard_y=coords[0],shard_x=coords[1])
        start,stop,step=[eval(code,{'__builtins__':{}},variables) for code in bound]
        owner=tuple(coords[a] for a in owners)
        counts.setdefault(owner,Counter()).update(range(start,stop,step))
    expected=Counter(range(0,((n//8+15)//16)*8,16))
    assert all(actual==expected for actual in counts.values()),counts
    # Task sharing must not introduce a cross-CTA partial reduction or change
    # the two ascending K partitions and their FP32 merge inside each task.
    assert 'for dense_partition in tl.range(0, 2)' in source
    assert 'dense_total = libdevice.add_rn(dense_total, dense_partial)' in source


@pytest.mark.parametrize('m,k,n,splits,rounded',[(1,4096,512,2,True),(2,256,520,1,False),
    (1,512,512,2,False),(1,4096,520,2,True)])
def test_device_matches_original_accumulation_bits(tmp_path,profile,m,k,n,splits,rounded):
    import torch
    from triton.flagmega.runtime import load
    from triton.flagmega.runtime.module import GeneratedTirCallGraphModule as Run
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()!=(9,0):pytest.skip('SM90 required')
    mesh=fm.Placement((8,16),'yx','bb');module=compile_module(mesh,m=m,k=k,n=n,splits=splits,rounded=rounded)
    runtimes=[];outputs=[];arguments=[];guards=[];shared={}
    def allocate(kind):
        kind=fm.logical_type(kind);shape=tuple(d.fixed_value for d in kind.shape)+tuple(getattr(kind.dtype,'lanes',()))
        dtype=getattr(torch,getattr(kind.dtype,'elem_type',kind.dtype).value)
        raw=torch.full((prod(shape)+32,),123,dtype=dtype,device='cuda');guards.append(raw)
        return raw[16:-16].reshape(shape)
    try:
        for enabled in (False,True):
            path=render(module,tmp_path/str(enabled),profile,enabled);runtime=load(path,device='cuda:0');runtimes.append(runtime)
            binding=runtime.buffer_plan.function_map[module.entry];values={};results=[]
            for value,names in binding.parameters:
                if value not in shared:shared[value]=allocate(module.node_map[value].type)
                values[names[0]]=shared[value]
            for value,names in binding.outputs:
                output=allocate(module.node_map[value].type);values[names[0]]=output;results.append(output)
            outputs.append(results);args=tuple(values[s['buffer']] for s in runtime.external_arguments);arguments.append(args)
            Run.prepare(runtime,*args);assert runtime.resource_report['spill_bytes']==0
        g=torch.Generator().manual_seed(8282)
        for mutation in range(6):
            if mutation<3:
                x=(torch.randint(-16,17,(m,k),generator=g).float()/128).bfloat16()
                w=(torch.randint(-16,17,(k,n),generator=g).float()/128).bfloat16()
            else:
                x=(torch.randn((m,k),generator=g)/2).bfloat16();w=(torch.randn((k,n),generator=g)/8).bfloat16()
            shared['x'].copy_(x);shared['w'].copy_(w.reshape(k//16,2,8,n//8,8).permute(0,3,4,1,2).contiguous())
            shared['sentinel'].copy_(torch.arange(128,dtype=torch.float32))
            for runtime,args,results in zip(runtimes,arguments,outputs,strict=True):
                for result in results:result.fill_(-17)
                Run.run_into(runtime,*args)
            torch.cuda.synchronize()
            for before,after in zip(outputs[0],outputs[1],strict=True):
                assert torch.equal(before.view(torch.uint8).flatten(),after.view(torch.uint8).flatten())
            if mutation<3:
                expected=(x.double()@w.double()).float()
                if rounded:expected=expected.bfloat16()
                torch.testing.assert_close(outputs[1][0].cpu().reshape(m,n),expected,rtol=0,atol=0)
            torch.testing.assert_close(outputs[1][1].cpu(),2*torch.arange(128,dtype=torch.float32),rtol=0,atol=0)
            for raw in guards:assert bool((raw[:16]==123).all()) and bool((raw[-16:]==123).all())
    finally:
        for runtime in runtimes:runtime.close()
