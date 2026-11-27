# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Normal compiler regression cases for ordered register-vector BF16 FMA loads."""
from dataclasses import replace
from pathlib import Path
import importlib.util,sys
from math import prod
import pytest
from triton.flagmega import ir as fm
from triton.flagmega.artifacts import write_artifact
from triton.flagmega.codegen.triton.templates import TritonTemplateRegistry
from triton.flagmega.compiler import Compiler
from triton.flagmega.options import CompileOptions
from triton.flagmega.selection import apply_plan,override_plan
from triton.flagmega.targets import NvidiaSm90Target
from triton.flagmega.errors import CodegenError
@pytest.fixture(scope='module')
def profile():
 before=TritonTemplateRegistry.render
 path=(Path(__file__).resolve().parents[6]/'tutorials/flagmega/04-glm5.3-flash-fp8/nvidia-h800/local_optimizations/replica_work.py')
 spec=importlib.util.spec_from_file_location('_ordered_fma_replica',path);module=importlib.util.module_from_spec(spec)
 try:spec.loader.exec_module(module)
 finally:TritonTemplateRegistry.render=before
 return module

def compile_case(*,k=4096,n=288,m=1,tn=2,layout='register8',lanes=8):
 mesh=fm.Placement((8,16),'yx','bb');b=fm.SBP.broadcast()
 class Graph(fm.Module):
  def forward(self):
   x=self.input('x',fm.DistributedType(fm.tensor_type('bfloat16',(m,k)),(b,b),mesh),id='x')
   w=self.input('w',fm.DistributedType(fm.tensor_type(fm.vector_type('bfloat16',(8,2,8)),(k//16,n//8)),(b,fm.SBP.split_block_cyclic((1,),1)),mesh),id='w')
   none=fm.F.builtin.none()
   y=fm.F.ntt.packed_matmul(x,w,none,none,output_data_type='float32',name='projection',metadata={'selected_vectorization':'vectorization.matmul.n','selected_vector_axes':(1,),'selected_vector_lanes':(8,)})
   sentinel=self.input('sentinel',fm.tensor_type('float32',(128,)),id='sentinel')
   split=fm.DistributedType(sentinel.type,(fm.SBP.split_contiguous((0,1),1),),mesh)
   z=fm.F.distributed.force_boxing(sentinel,split)
   z=fm.F.distributed.force_boxing(fm.F.math.add(z,z),sentinel.type)
   self.function('main',(x,w,sentinel),(y,z))
 source=Graph(dialect='distributed',stage='frozen_constants',entry='main',metadata={'auto_distribution':{'placement':mesh.to_data()}}).build()
 oldid='tir.dense_matmul.packed_k_major_threaded_fma';newid=oldid+'.register-load-test'
 catalog=NvidiaSm90Target().triton_implementation_model
 implementation=replace(catalog.implementation(oldid),id=newid,parameters={**catalog.implementation(oldid).parameters,'tile_n':8,'scalar_store_subtile_n':tn,'compute_num_warps':8,'values_per_lane':lanes,'block_k':128*lanes,'weight_load_layout':layout})
 catalog=replace(catalog,implementations=(*catalog.implementations,implementation))
 class Target(NvidiaSm90Target):
  def propose_tir(self,module):
   module=super().propose_tir(module)
   points=[p for p in module.selection_points if p.kind=='tir' and p.owner in module.node_map and module.node_map[p.owner].op=='ntt.packed_matmul']
   assert len(points)==1
   return apply_plan(module,override_plan(module,[(points[0].id,newid)],policy='test.ordered-fma-register-load',rationale='Retain ordered BF16 FMA and the same packed scalar coordinates.'))
 compiler=Compiler(CompileOptions(placement=mesh));compiler.target=Target(triton_implementation_model=catalog).with_placement(mesh)
 return compiler.compile(source).module

def render(module,path,profile):
 before=TritonTemplateRegistry.render
 try:
  TritonTemplateRegistry.render=profile.override
  return write_artifact(module,path,target='nvidia-sm90',emit_executable=True)
 finally:TritonTemplateRegistry.render=before

@pytest.mark.parametrize('layout,lanes',[('wrong',8),('register8',4)])
def test_rejects_unknown_or_incompatible_load_contract(tmp_path,profile,layout,lanes):
 with pytest.raises(CodegenError,match='(weight_load_layout|Register8)'):
  render(compile_case(layout=layout,lanes=lanes,tn=8),tmp_path/'bad',profile)

def test_selected_codegen_contains_register_load_and_fixed_tree(tmp_path,profile):
 module=compile_case();path=render(module,tmp_path/'new',profile)
 source=(path/'generated_kernels.py').read_text()
 assert '_flagmega_threaded_fma_register8' in source
 assert 'dense_register_weights' in source
 assert 'add.rn.f32 $0, $1, $2;' in source
 assert 'dense_local_n_group_start in tl.range(' in source
 assert ' % 4) * 2' in source
 assert fm.verify_buffer_plan(module)

def test_subvector_store_requires_a_verified_replica_group(tmp_path):
 with pytest.raises(CodegenError,match='Subvector FMA tasks require'):
  write_artifact(compile_case(),tmp_path/'no-proof',target='nvidia-sm90',emit_executable=True)

@pytest.mark.parametrize('n',[264,280,288])
def test_every_packed_scalar_lane_has_exactly_one_writer(tmp_path,profile,n):
 import ast
 from collections import Counter
 module=compile_case(n=n)
 path=render(module,tmp_path/'coverage',profile)
 tree=ast.parse((path/'generated_kernels.py').read_text())
 loops=[v for v in ast.walk(tree) if isinstance(v,ast.For) and isinstance(v.target,ast.Name) and v.target.id=='dense_local_n_group_start']
 assert len(loops)==1
 loop=loops[0]
 bounds=[compile(ast.Expression(v),'<group bounds>','eval') for v in loop.iter.args]
 start,=[v.value for v in loop.body if isinstance(v,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='dense_local_n_start' for t in v.targets)]
 expression=compile(ast.Expression(start),'<subvector task>','eval')
 counts=Counter()
 for x in range(16):
  for y in range(8):
   env={'shard_coord0':y,'shard_coord1':x,'shard_y':y,'shard_x':x}
   for group in range(*(eval(v,{'__builtins__':{}},env) for v in bounds)):
    env['dense_local_n_group_start']=group;local=eval(expression,{'__builtins__':{}},env)
    for lane in range(local,local+2):
     scalar=(x+lane//8*16)*8+lane%8
     if scalar<n:counts[scalar]+=1
 assert counts==Counter(range(n)),counts

@pytest.mark.parametrize('m,k,n',[(1,4096,288),(1,1024,264),(2,1024,280)])
def test_device_preserves_ordered_bits_and_guards(tmp_path,profile,m,k,n):
 import torch
 from triton.flagmega.runtime import load
 from triton.flagmega.runtime.module import GeneratedTirCallGraphModule as Run
 if not torch.cuda.is_available() or torch.cuda.get_device_capability()!=(9,0):pytest.skip('SM90 required')
 runtimes=[];arguments=[];results=[];guards=[];shared={}
 def allocate(kind):
  kind=fm.logical_type(kind);shape=tuple(d.fixed_value for d in kind.shape)+tuple(getattr(kind.dtype,'lanes',()))
  dtype=getattr(torch,getattr(kind.dtype,'elem_type',kind.dtype).value)
  raw=torch.full((prod(shape)+32,),123,dtype=dtype,device='cuda');guards.append(raw)
  return raw[16:-16].reshape(shape)
 try:
  for tn,layout in [(8,'scalar'),(8,'register8'),(2,'register8')]:
   module=compile_case(m=m,k=k,n=n,tn=tn,layout=layout);path=render(module,tmp_path/f'{tn}-{layout}',profile);runtime=load(path,device='cuda:0');runtimes.append(runtime)
   binding=runtime.buffer_plan.function_map[module.entry];values={};ys=[]
   for value,names in binding.parameters:
    if value not in shared:shared[value]=allocate(module.node_map[value].type)
    values[names[0]]=shared[value]
   for value,names in binding.outputs:
    y=allocate(module.node_map[value].type);values[names[0]]=y;ys.append(y)
   results.append(ys);args=tuple(values[s['buffer']] for s in runtime.external_arguments);arguments.append(args)
   Run.prepare(runtime,*args);assert runtime.resource_report['spill_bytes']==0
  generator=torch.Generator().manual_seed(947)
  for mutation in range(6):
   if mutation<3:
    x=(torch.randint(-16,17,(m,k),generator=generator).float()/128).bfloat16();w=(torch.randint(-16,17,(k,n),generator=generator).float()/128).bfloat16()
   else:x=torch.randn((m,k),generator=generator).bfloat16();w=(torch.randn((k,n),generator=generator)/8).bfloat16()
   shared['x'].copy_(x);shared['w'].copy_(w.reshape(k//16,2,8,n//8,8).permute(0,3,4,1,2).contiguous());shared['sentinel'].copy_(torch.arange(128,dtype=torch.float32))
   for runtime,args,ys in zip(runtimes,arguments,results):
    for y in ys:y.fill_(-17)
    Run.run_into(runtime,*args)
   torch.cuda.synchronize()
   for ys in results:
    assert torch.equal(ys[0].view(torch.uint8),results[0][0].view(torch.uint8))
    torch.testing.assert_close(ys[1].cpu(),2*torch.arange(128,dtype=torch.float32),rtol=0,atol=0)
    if mutation<3:torch.testing.assert_close(ys[0].cpu().reshape(m,n),x.double()@w.double(),rtol=0,atol=0,check_dtype=False)
   for raw in guards:assert (raw[:16]==123.).all() and (raw[-16:]==123.).all()
 finally:
  for r in runtimes:r.close()
