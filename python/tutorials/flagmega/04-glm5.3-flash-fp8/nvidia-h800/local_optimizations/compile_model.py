"""Rebuild measured TIR from the original pre-microkernel input, with all planners active."""
from pathlib import Path
from dataclasses import replace
from triton.flagmega import ir as fm
import argparse,gzip,hashlib,importlib.util,json,sys,time
BASE=Path(__file__).resolve().parent
sys.path.insert(0,str(BASE.parent))
def receipt(path,data):
 path.write_text(json.dumps(data,indent=2,default=str)+'\n')
def catalog(model):
 """Scoped physical variants; preserve selected semantic candidates and rounding."""
 changes={
 'tir.kda_input_projection.wgmma_local':dict(block_k=256),
 'tir.dense_matmul.packed_k_major_threaded_fma_register8_n8_s2':dict(compute_num_warps=4),
 'tir.sparse_experts_gate_up.complete_k_wgmma_n32':dict(compute_num_warps=4,num_stages=4),
 'tir.partitioned_sparse_experts_down.wgmma':dict(compute_num_warps=4,output_tile_n=64,tasks_per_owner=4),
 'tir.matmul_glu.cooperative_wgmma':dict(compute_num_warps=4),
 'tir.partitioned_block_scaled_linear.local':dict(compute_num_warps=4),
 }
 found=set();values=[]
 for v in model.implementations:
  values.append(v)
  if v.id in changes:
   found.add(v.id);v=replace(v,id=v.id+'.decode_measured',parameters={**v.parameters,**changes[v.id]})
   if v.family=='kda_input_projection':
    v=replace(v,shared_workspaces=tuple(replace(w,type=replace(w.type,shape=(*w.type.shape[:-1],fm.dim(256)))) for w in v.shared_workspaces))
   values.append(v)
 if found!=changes.keys():raise ValueError(f'Absent measured implementation contracts: {changes.keys()-found}')
 from local_optimizations.joint_dsa_absorbed_pool_scores import implementation_model as joint_dsa_model
 from local_optimizations.native_latent_probability_model import declared_model as probability_model
 return probability_model(joint_dsa_model(replace(model,implementations=tuple(values))))
def measured_physical_tir(m,target):
 from triton.flagmega.ir.tir.kernel_definition import replace_kernel_dispatch
 from triton.flagmega.codegen.triton.microkernels.materialization import materialize_shared_workspace_buffers
 definitions=[];changed=[]
 for definition in m.kernel_definitions:
  dispatch=definition.dispatch;old=dispatch.microkernel
  chosen=target.triton_implementation_model.implementation(old.implementation+'.decode_measured')
  if chosen is None or definition.name in {'kernel_matmul_glu_cooperative_wgmma_97494b8867efabf6','kernel_partitioned_block_scaled_linear_local_953bd9fccf25b9fc'}:
   definitions.append(definition);continue
  selected=replace(old,implementation=chosen.id,parameters={**old.parameters,**chosen.parameters},shared_workspaces=chosen.shared_workspaces)
  revised=replace(dispatch,microkernel=selected,shared_workspace_buffers=materialize_shared_workspace_buffers(definition,selected))
  definitions.append(replace_kernel_dispatch(definition,revised));changed.append(dict(kernel=definition.name,old=old.implementation,new=chosen.id))
 if len(changed)!=6:raise ValueError(('measured physical contracts',changed))
 result=replace(m,kernel_definitions=tuple(definitions))
 target.verify(result)
 print(json.dumps(dict(physical_overrides=changed)),flush=True)
 return result

def partition(m):
 from local_optimizations.concurrent_call_lifetimes import declare
 m=declare(m)
 from triton.flagmega.passes.functions import function_nodes
 front,last_effect={},None
 programs=[]
 for n in function_nodes(m,m.function_map[m.entry]):
  deps=set().union(*(front[v] for v in n.inputs))
  if n.op=='tir.call':
   if not n.effect.is_pure and last_effect:deps.add(last_effect)
   symbol=f'program_{len(programs):03d}'
   row=dict(symbol=symbol,calls=(n.id,),num_warps=8,depends_on=tuple(p['symbol'] for p in programs if p['symbol'] in deps))
   if n.attrs.get('callee','').endswith(('.routed_up','.routed_down','.shared_up','.shared_down')):
    row['num_warps']=4
   if n.attrs.get('callee','').endswith('.routed_down'):row.update(grid=(512,1,1),tasks_per_owner=4)
   programs.append(row);front[n.id]={symbol}
   if not n.effect.is_pure:last_effect=symbol
  else:front[n.id]=deps
 def actors(names):
  return tuple(p['symbol'] for p in programs if str(m.node_map[p['calls'][0]].attrs.get('callee','')) in names)
 prefixes={'decode_linear_attention_dense','decode_linear_attention_sparse.prefix','decode_mla_attention_sparse.prefix'}
 mla={'decode_mla_attention_sparse.prefix'}
 specs=[
 ('ordered_join',8448,{'decode_linear_attention_sparse.join','decode_mla_attention_sparse.join'},42),
 ('ordered_oproj',9216,prefixes,45),('mhc_projection_partials',1024,prefixes,45),
 ('qkv_concurrent_partials',4096,mla,11),('dsa_deferred_results',768,mla,11),
 ('router_private_logits',1280,{'decode_linear_attention_sparse.routed_up','decode_mla_attention_sparse.routed_up'},42),
 ('qkv_concurrent_normalized',1024,mla,11)]
 resources=[]
 for name,size,functions,count in specs:
  calls=actors(functions)
  if len(calls)!=count:raise ValueError((name,len(calls),count))
  resources.append(dict(name=name,nbytes=size,alignment=256,initialization='zero',programs=calls))
 if len(programs)!=270:raise ValueError(('full model program count',len(programs)))
 return replace(m,metadata={**m.metadata,'kernel_program_partition':tuple(programs),'persistent_window_resources':dict(storage='workspace',resources=tuple(resources))})
def compose_shared_ffn(m):
 from triton.flagmega.passes.functions import function_nodes
 old_nodes=m.node_map;root=function_nodes(m,m.function_map[m.entry])
 assert m.stage=='selected_microkernels' and not m.execution_functions and 'buffer_plan' not in m.metadata
 users={n.id:[] for n in m.nodes}
 for n in m.nodes:
     for v in n.inputs:users[v].append(n.id)
 pairs=[]
 for i,n in enumerate(root):
     if n.op=='tir.call' and str(n.attrs.get('callee','')).endswith('.shared_up'):
         d=next(v for v in root[i+1:] if v.op=='tir.call')
         assert str(d.attrs.get('callee','')).removesuffix('.shared_down')==str(n.attrs['callee']).removesuffix('.shared_up')
         assert n.effect.is_pure and d.effect.is_pure and users[n.id]==[d.id]
         pairs.append((n,d))
 assert len(pairs)==42
 added=[];functions=[];replacements={};removed=set();wrappers={};receipts=[]
 for up,down in pairs:
     family=up.attrs['callee'].removesuffix('.shared_up')
     actuals=tuple(dict.fromkeys((*up.inputs,*(v for v in down.inputs if v!=up.id))))
     assert len(actuals)==7
     name=family+'.shared_ffn'
     if name not in wrappers:
         formals=tuple(name+f'.arg{i}' for i in range(len(actuals)))
         added.extend(fm.Node(p,'builtin.var',(),old_nodes[v].type,attrs={'name':p}) for p,v in zip(formals,actuals,strict=True))
         binding=dict(zip(actuals,formals,strict=True))
         inner_up=replace(up,id=name+'.up',inputs=tuple(binding[v] for v in up.inputs),metadata={})
         inner_down=replace(down,id=name+'.down',inputs=tuple(inner_up.id if v==up.id else binding[v] for v in down.inputs),metadata={})
         added.extend((inner_up,inner_down))
         functions.append(fm.Function(name,formals,(inner_down.id,),dict(calling_convention='device',noinline=True,reusable=True)))
         wrappers[name]=(tuple(old_nodes[v].type for v in actuals),inner_up,inner_down,
                         tuple(-1 if v==up.id else actuals.index(v) for v in down.inputs))
     types,inner_up,inner_down,down_bindings=wrappers[name]
     assert types==tuple(old_nodes[v].type for v in actuals) and up.type==inner_up.type and down.type==inner_down.type
     binding=dict(zip(actuals,range(len(actuals)),strict=True))
     assert tuple(up.inputs.index(v) for v in up.inputs)==tuple(range(5))
     assert tuple(binding[v] for v in up.inputs)==tuple(range(5))
     assert tuple(-1 if v==up.id else binding[v] for v in down.inputs)==down_bindings
     replacements[down.id]=replace(down,inputs=actuals,attrs={'callee':name})
     removed.add(up.id)
     receipts.append(dict(up=up.id,down=down.id,wrapper=name,actuals=actuals,original_call_types_and_effects_preserved=True,original_complete_inner_callees=(up.attrs['callee'],down.attrs['callee'])))
 old_programs=m.metadata['kernel_program_partition'];bycall={c:p for p in old_programs for c in p['calls']}
 redirect={bycall[d.id]['symbol']:bycall[u.id]['symbol'] for u,d in pairs};up_to_down={u.id:d.id for u,d in pairs}
 programs=[]
 for p in old_programs:
     if p['symbol'] in redirect:continue
     calls=tuple(up_to_down.get(c,c) for c in p['calls'])
     dependencies=tuple(dict.fromkeys(redirect.get(d,d) for d in p['depends_on']))
     assert p['symbol'] not in dependencies
     programs.append({**p,'calls':calls,'depends_on':dependencies})
 assert len(programs)==228
 updated=replace(m,nodes=tuple(n for old in m.nodes if old.id not in removed for n in (replacements.get(old.id,old),))+tuple(added),functions=tuple(m.functions)+tuple(functions),metadata={**m.metadata,'kernel_program_partition':tuple(programs)})
 updated=fm.verify_module(updated)
 return updated

def main():
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);args=p.parse_args()
 out=args.output;out.mkdir(exist_ok=False)
 from local_optimizations.base_recipe import fm,activate
 activate()
 from local_optimizations.native_tp4 import native_implementation_model
 from local_optimizations.target import InputTarget
 from triton.flagmega.targets import NvidiaSm90Target
 from triton.flagmega.options import CompileOptions,parse_placement
 from triton.flagmega.compiler import Compiler
 from triton.flagmega.passes import FunctionalPass,PassManager
 ident=json.loads((BASE/'original-input.json').read_text())
 data=gzip.decompress((BASE/'original-input.py.gz').read_bytes())
 if hashlib.sha256(data).hexdigest()!=ident['source_sha256']:raise ValueError('Original input digest differs')
 f=out/'original.py';f.write_bytes(data)
 start=time.monotonic();m=fm.load_module(f)
 if m.stage!='selected_tir_variants' or m.execution_functions or 'buffer_plan' in m.metadata:raise ValueError('Resume must precede physical candidates and allocation')
 receipt(out/'input.json',dict(source=ident,semantic_hash=m.semantic_hash,stage=m.stage,seconds=time.monotonic()-start))
 placement=parse_placement('cyx=4x8x16/cbb')
 from local_optimizations.native_latent_probability_model import ProbabilityTargetMixin
 class Target(ProbabilityTargetMixin,InputTarget):pass
 target=Target(options=replace(NvidiaSm90Target().options,placements=(placement,)),evidence_dir=out,gemv_compute_warps=8,triton_implementation_model=catalog(native_implementation_model(offline_projection_weights=True)))
 from triton.flagmega.selection import apply_plan,override_plan
 from local_optimizations.native_latent_probability_model import IDENTITY
 points={point.id:point for point in m.selection_points if point.kind in ('tir','semantic_tir')}
 choices={point:m.selection_map[point].candidate_id for point in points if point in m.selection_map}
 opened=replace(m,selection_points=tuple(point for point in m.selection_points if point.id not in points),selections=tuple(choice for choice in m.selections if choice.point_id not in points))
 proposed=target.tir_selection_policy.propose(opened,target)
 attention=[point for point,value in points.items() if m.node_map[value.owner].op=='nn.dsa_paged_attention']
 if len(attention)!=1:raise ValueError('Expected one complete shared MLA attention selection')
 choices[attention[0]]=IDENTITY
 m=apply_plan(proposed,override_plan(proposed,tuple(choices.items()),policy='glm53.parallel-probability-long32/v1',rationale='Declare invocation-owned score and probability metadata before normal lowering.'))
 c=Compiler(CompileOptions(target='nvidia-sm90',placement=placement));c.target=target
 original=FunctionalPass.run
 def measured(self,value):
  begin=time.monotonic();result=original(self,value)
  print(json.dumps(dict(pass_name=self.name,stage=result.stage,seconds=time.monotonic()-begin)),flush=True);return result
 FunctionalPass.run=measured
 try:
  m=c.compile(m,stop_after='selected_microkernels').module
  m=measured_physical_tir(m,target)
  manager=PassManager('decode_measured_programs');manager.add(FunctionalPass('DeclareMeasuredProgramsAndChannels',partition));m=manager.run(m).module
  m=compose_shared_ffn(m)
  fm.emit_module(m,out/'requested.py')
  m=c.compile(m).module
 finally:FunctionalPass.run=original
 from local_optimizations.native_tp4 import rewrite_mla_schedules,rewrite_qkv_norm_schedules
 from local_optimizations.head_schedule import rewrite as rewrite_head_schedules
 for rewrite in (rewrite_mla_schedules,rewrite_qkv_norm_schedules,rewrite_head_schedules):
  m=fm.verify_module(replace(m,kernel_definitions=rewrite(m)))
 fm.emit_module(m,out/'dag-final.py')
 plan=fm.verify_buffer_plan(m)
 receipt(out/'compile-result.json',dict(semantic_hash=m.semantic_hash,stage=m.stage,workspace_bytes=plan.workspace_bytes,functions=len(m.execution_functions),nodes=len(m.nodes),seconds=time.monotonic()-start,real_normal_compiler=True,allocation_regenerated=True))
 print(json.dumps(dict(semantic_hash=m.semantic_hash,stage=m.stage,seconds=time.monotonic()-start)),flush=True)
if __name__=='__main__':main()

