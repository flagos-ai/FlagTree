"""Registered, isolated r007 ready-source region and strict native provider.

The model and frozen Q8/GDN providers are not configured by importing this file.
"""
from dataclasses import replace
from pathlib import Path
import sys
from triton.flagmega import ir as fm
from triton.flagmega.errors import IRSchemaError,CodegenError
from triton.flagmega.ir.ops.core import OpDefinition,op_definition,input_parameter,attribute_parameter
from triton.flagmega.ir.type_pattern import is_tensor
from triton.flagmega.ir.memory_effect import MemoryEffect
from triton.flagmega.ir.distributed_inference import tensor_of,broadcast_ir_type,all_broadcast
from triton.flagmega.ir.model import TupleType,DistributedType,DType,tensor_type
from triton.flagmega.ir.types import VectorType
PLAN='fb69f8d85774d8f1dcf5050695ba185863168492441ef79865d78feec77b851d'
OP='nn.r007_first_region'; FAMILY='r007_first_region'; SCHEMA='flagmega.r007-first-region/v1'
@op_definition(OP,namespace='nn',functional_name='r007_first_region',display_name='NN.R007FirstRegion')
class FirstRegion(OpDefinition):
 value=input_parameter(is_tensor(),memory_effect=MemoryEffect.CHIP_READ)
 weight=input_parameter(is_tensor(),memory_effect=MemoryEffect.CHIP_READ)
 weight_scale=input_parameter(is_tensor(),memory_effect=MemoryEffect.NONE)
 arena=input_parameter(is_tensor(),memory_effect=MemoryEffect.CHIP_READ_WRITE)
 arm=attribute_parameter()
 plan_hash=attribute_parameter(default=PLAN)
 source_version_key=attribute_parameter(default='decode_attention_1_input_norm.reshard85:ready')
 bank_key=attribute_parameter(default='decode_attention_1_q:IQ4_XS[12288,2720]')
 envelope=attribute_parameter(default=16384)
 result_memory_effects=(MemoryEffect.CHIP_WRITE,MemoryEffect.CHIP_WRITE)
 const_evaluable=False
 @classmethod
 def normalize_attrs(cls,attributes):
  a=super().normalize_attrs(attributes)
  if a['arm'] not in ('s','p') or a['plan_hash']!=PLAN or a['envelope']!=16384 or a['source_version_key']!='decode_attention_1_input_norm.reshard85:ready' or a['bank_key']!='decode_attention_1_q:IQ4_XS[12288,2720]':raise IRSchemaError('Invalid r007 source/bank/envelope/options; no fallback.')
  return a
 @classmethod
 def infer_type(cls,inputs,a):
  shapes=(tensor_type('bfloat16',(1,5120)),tensor_type('uint8',(12288,2720)),tensor_type('float32',(96,40)),tensor_type('uint8',(6144,)))
  if tuple(tensor_of(n.type) for n in inputs)!=shapes:raise IRSchemaError('r007 complete source/bank/scale/arena ABI mismatch.')
  out=TupleType(tuple(tensor_type(VectorType(DType.BFLOAT16,(8,)),(1,24,32)) for _ in range(2)))
  kinds=tuple(n.type for n in inputs)
  if not any(isinstance(t,DistributedType) for t in kinds):return out
  if not all(isinstance(t,DistributedType) and all_broadcast(t) for t in kinds) or len({t.placement for t in kinds})!=1:raise IRSchemaError('r007 requires one full chip-local owner.')
  return broadcast_ir_type(out,kinds[0].placement)
 @classmethod
 def evaluate(cls,node,args,context):
  raise CodegenError('r007 numerical acceptance is integration gated; no substitute CPU reference.')

def verify_region(m):
 for n in m.nodes:
  if n.op==OP:
   FirstRegion.verify(n,m)
   if n.metadata.get('r007_plan_hash')!=PLAN or n.metadata.get('r007_source_version')!=n.attrs['source_version_key'] or n.metadata.get('r007_bank')!=n.attrs['bank_key']:raise CodegenError('Stale/mixed r007 per-node contract.')
 return m

def _parameters(arm):
 return dict(plan_hash=PLAN,arm=arm,num_warps=4,num_ctas=1,program_kind='persistent',threads=128,usable_static_bytes=16384,dynamic_shared_bytes=0,logical_shards=4,producer_warp=0,consumer_warps=(1,2,3))

def validate_raw(raw):
 p=raw['parameters']
 if p.get('plan_hash')!=PLAN or p.get('arm') not in ('s','p') or any((tuple(p.get(k,())) if isinstance(v,tuple) else p.get(k))!=v for k,v in _parameters(p.get('arm')).items()):raise CodegenError('r007 selected provider contract mismatch.')
 if raw['semantic_attrs']['arm']!=p['arm'] or raw['semantic_attrs']['envelope']!=16384:raise CodegenError('r007 semantic/provider envelope mismatch.')
 if tuple(x['formal'] for x in raw['outputs'])!=('result_0','result_1'):raise CodegenError('r007 requires both complete Q/gate fields.')

def encode(raw):
 validate_raw(raw)
 names=['value','weight','arena','result_0','result_1']; ptrs=[]
 for name in names:
  side='outputs' if name.startswith('result_') else 'inputs'
  matches=[x for x in raw[side] if x['formal']==name]
  if len(matches)!=1 or len(matches[0]['buffers'])!=1:raise CodegenError('Missing r007 native formal '+name)
  b=matches[0]['buffers'][0]; abi=b['abi']
  if abi.get('pool_scope_stride_bytes',0) or b.get('address_arguments'):raise CodegenError('r007 unsafe/dynamic native span.')
  ptrs.append('('+b['runtime_argument']+' + '+str(abi.get('pool_byte_offset',0)//abi['scalar_itemsize'])+')')
 return dict(schema=SCHEMA,pointers=ptrs,arm=raw['parameters']['arm'])

def configure(target):
 from triton.backends.nvidia.compiler import CUDAOptions,sm_arch_from_capability
 if 'enable_arch_accel' not in CUDAOptions.__dataclass_fields__ or sm_arch_from_capability(120,False)!='sm_120':
  raise CodegenError('r007 requires explicit native plain-SM120 compiler support; no fallback.')
 # Public target extension; existing base policies and frozen providers stay intact.
 from triton.flagmega.codegen.triton.implementation import TritonImplementation,TritonImplementationModel
 from triton.flagmega.codegen.triton.candidates import TritonCandidateProviderRegistry,TritonCandidateProposal
 from triton.flagmega.codegen.triton.selection import TritonTirSelectionPolicy
 from triton.flagmega.codegen.triton.kernel_call_renderers import register_family_encoder
 from triton.flagmega.ir.tir.shared_workspace_descriptor import TIRSharedWorkspaceDescriptor
 from triton.flagmega.ir.launch_recipe import TaskLaunchRecipe
 from triton.flagmega.targets.nvidia_sm120 import NvidiaSm120Target
 from triton.flagmega.passes.auto_distributed.candidates import DistributedCandidate,DistributedCandidateProviderBase
 W=Path(__file__).resolve().parents[4]/'tutorials/flagmega/03-qwen3.8-27b-iq4/nvidia-rtx5060ti'
 if str(W) not in sys.path:sys.path.insert(1,str(W))
 from agent_optimizations.iq4_streaming.graph import StreamingDistributionPolicy,StreamingBufferizationPolicy
 register_family_encoder(FAMILY,encode,schema=SCHEMA,validate_raw=validate_raw)
 class Rows(DistributedCandidateProviderBase):
  op_names=frozenset((OP,));is_exhaustive=True;allows_partial_inputs=False
  def _enumerate_candidates(self,context):
   node=context.source_call
   return (DistributedCandidate('r007-full-owner',broadcast_ir_type(node.type,context.placement),tuple(broadcast_ir_type(context.module.node_map[i].type,context.placement) for i in node.inputs),0,'r007-complete-ready-source',OP,'agent',SCHEMA,(PLAN,)),)
 class Distribution(StreamingDistributionPolicy):
  @property
  def identity(self):return self.base.identity+'+r007:'+PLAN
  def with_placement(self,placement):return Distribution(self.base.with_placement(placement),None)
  def register_candidate_providers(self,registry):self.base.register_candidate_providers(registry);registry.add(Rows())
 class Candidate:
  op_names=frozenset((OP,))
  def propose(self,node,context):
   verify_region(context.module)
   impl=next(i for i in context.implementations(FAMILY) if i.variant=='first_'+node.attrs['arm'])
   recipe=TaskLaunchRecipe(12288,8,'persistent',max_ctas=144,ctas_per_sm=4)
   c=context.configure_implementation(impl,semantic_parameters={'task_mapping':recipe.to_data()},facts={'source_memory_scope':'chip','result_memory_scope':'chip','required_codegen_backend':'triton'})
   return TritonCandidateProposal((c,),c.id)
 class Target(NvidiaSm120Target):
  def install(self,m):
   nodes=tuple(replace(n,metadata={**n.metadata,'r007_plan_hash':PLAN,'r007_source_version':n.attrs['source_version_key'],'r007_bank':n.attrs['bank_key'],'execution_mesh':fm.ExecutionMesh.dynamic().to_data(),'kernel_program_kind':'persistent','kernel_num_warps':4,'kernel_enable_static_shared':True}) if n.op==OP and 'r007_plan_hash' not in n.metadata else n for n in m.nodes)
   return verify_region(replace(m,nodes=nodes,metadata={**m.metadata,'launch_contract':{'cooperative_grid':True}}))
  def auto_distribute(self,m):return super().auto_distribute(self.install(m))
  def propose_distribution(self,m):return super().propose_distribution(self.install(m))
  def with_placement(self,p):return configure(NvidiaSm120Target.with_placement(self,p))
 old=target.triton_implementation_model
 impls=tuple(TritonImplementation('tir.r007_first_region.first_'+arm,FAMILY,'first_'+arm,parameters=_parameters(arm),facts={'internal_grid_barriers':0,'required_codegen_backend':'triton'},shared_workspaces=(TIRSharedWorkspaceDescriptor('r007_envelope',tensor_type('uint8',(16384,)),128),)) for arm in ('s','p'))
 model=TritonImplementationModel((*old.implementations,*impls),{**old.preferences,FAMILY:(impls[0].id,impls[1].id)},name=old.name+'+r007')
 registry=TritonCandidateProviderRegistry()
 for provider in target.tir_selection_policy.registry.providers:registry.add(provider)
 registry.add(Candidate())
 result=Target(target.machine.capability,options=target.options,vectorization_policy=target.vectorization_policy,packing_policy=target.packing_policy,distribution_policy=Distribution(target.distribution_policy,None),selection_policy=target.selection_policy,bufferization_policy=StreamingBufferizationPolicy(target.bufferization_policy.options,synchronization_protocol_provider=target.bufferization_policy.synchronization_protocol_provider),tir_selection_policy=TritonTirSelectionPolicy(target.tir_selection_policy.workspace_annotator,registry),tir_lowering_policy=target.tir_lowering_policy,microkernel_selection_policy=target.microkernel_selection_policy,triton_implementation_model=model)
 result.codegen_backend='triton';return result

def expected_program_options():
 return dict(num_warps=4,num_ctas=1,num_stages=3,maxnreg=None,enable_fp_fusion=False,enable_static_shared=True,ptx_options='--minnctapersm=1',enable_arch_accel=False)
def verify_program_options(options):
 expected=expected_program_options()
 if set(options)!=set(expected) or any(type(options[k]) is not type(v) or options[k]!=v for k,v in expected.items()):raise CodegenError('Invalid checked r007 plain-SM120 program options.')
def program_options(selected,launch):
 if len(selected)!=1 or selected[0]['family']!=FAMILY or launch['num_warps']!=4:raise CodegenError('r007 requires one complete native region per entry.')
 validate_raw(selected[0])
 launch['compile_options']=expected_program_options();launch['requires_grid_wide_sync']=True
 launch['argument_bindings']=[{'kind':'argument','source':i} for i in range(6)]

def register():
 from triton.flagmega.codegen.triton.kernel_call_renderers import register_family_encoder
 register_family_encoder(FAMILY,encode,schema=SCHEMA,validate_raw=validate_raw)
register()
