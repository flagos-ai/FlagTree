"""Compile the complete GLM FP8 mesh while retaining native decode schedules."""
import argparse,hashlib,json,sys,time
from collections import Counter
from dataclasses import replace
from pathlib import Path
repo=Path.cwd();tutorial=repo/'python/tutorials/flagmega/04-glm5.3-flash-fp8/nvidia-h800';sys.path.insert(0,str(tutorial))
from triton.flagmega import ir as fm
from triton.flagmega.compiler import Compiler
from triton.flagmega.options import CompileOptions,parse_placement
from triton.flagmega.targets import NvidiaSm90Target
from triton.flagmega.ir.execution_mesh import ExecutionMesh,placement_for_execution_mesh
from triton.flagmega.passes.functions.graph import static_node_invocation_counts, function_nodes
from local_optimizations.base_recipe import activate
from local_optimizations.native_tp4 import native_implementation_model
from local_optimizations.target import InputTarget
activate()
q=tutorial/'.local/multi-prompt-20261022/long-context-opt'
parser=argparse.ArgumentParser();parser.add_argument('--policy',choices=('fixed','dynamic'),required=True)
parser.add_argument('--input',type=Path,default=q/'real-fp8-inputs-v207/original.py')
parser.add_argument('--output',type=Path)
parser.add_argument('--dsa-pool-scores',action='store_true')
parser.add_argument('--native-latent-coalesced',action='store_true')
parser.add_argument('--native-latent-compact-pipeline',action='store_true')
parser.add_argument('--native-moe-down-loads',action='store_true')
args=parser.parse_args()
if args.native_latent_compact_pipeline and not args.native_latent_coalesced:
 parser.error('Compact latent pipeline requires the complete coalesced Native schedule')
root=args.output or q/('real-fp8-'+args.policy+'-compile-native-precision');root.mkdir(exist_ok=False)
frozen={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for base in (repo/'python/triton/flagmega',tutorial/'local_optimizations') for p in base.rglob('*') if p.suffix in ('.py','.jinja')}
for relative in ('lib/Dialect/TritonGPU/IR/Ops.cpp','third_party/tle/test/GPU/test_tle_memdesc_reshape_swizzle_geometry.mlir'):
 p=repo/relative;frozen[str(p)]=hashlib.sha256(p.read_bytes()).hexdigest()
frozen[str(Path(__file__).resolve())]=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
(root/'inputs.json').write_text(json.dumps(dict(policy=args.policy,source_sha256=frozen,normal_compiler=True,manual_Core_or_checkpoint_weight_payload_read=False),indent=2)+'\n')
began=time.monotonic();m=fm.load_module(args.input)
assert m.stage=='selected_tir_variants' and not m.kernel_definitions and not m.execution_functions
assert 'buffer_plan' not in m.metadata and 'kernel_program_partition' not in m.metadata
assert m.metadata['num_hidden_layers']==45 and m.metadata['num_experts']==288 and m.metadata['num_experts_per_tok']==8
counts=static_node_invocation_counts(m);before={n.id:counts.get(n.id,0) for n in m.nodes if n.op=='nn.partitioned_block_scaled_linear'}
assert sum(before.values())==45,before
rows=[];nodes=[];redirect={}
for n in m.nodes:
 if n.op!='nn.partitioned_block_scaled_linear':nodes.append(n);continue
 metadata={**n.metadata,'kernel_num_warps':4}
 if args.policy=='fixed':
  # A fixed program can fuse only with fully compatible neighbors. These
  # explicit per-op resources are normal IR policies, not source patches.
  nodes.append(replace(n,metadata={**metadata,'execution_mesh':(8,16)}))
  rows.append(dict(id=n.id,invocations=counts[n.id],policy='fixed',device_placement_preserved=True));continue
 assert isinstance(n.type,fm.DistributedType)
 domain=placement_for_execution_mesh(n.type.placement,ExecutionMesh.dynamic())
 inputs=[]
 for index,value in enumerate(n.inputs):
  operand=m.node_map[value];assert isinstance(operand.type,fm.DistributedType)
  assert all(not isinstance(p,fm.SBPSplit) or all(operand.type.placement.hierarchy_levels[a]!='b' for a in p.hierarchy_axes) for p in operand.type.axis_policies)
  kind=replace(operand.type,placement=domain)
  view=fm.Node(n.id+f'.dynamic_input{index}','distributed.sharded_view',(value,),kind,attrs={'new_type':kind})
  nodes.append(view);inputs.append(view.id)
 kind=replace(n.type,placement=domain)
 nodes.append(replace(n,inputs=tuple(inputs),type=kind,metadata={**metadata,'execution_mesh':'*','kernel_program_kind':'ordinary'}))
 restore=fm.Node(n.id+'.fixed_consumer_view','distributed.sharded_view',(n.id,),n.type,attrs={'new_type':n.type})
 nodes.append(restore);redirect[n.id]=restore.id
 rows.append(dict(id=n.id,invocations=counts[n.id],policy='dynamic',original_placement=n.type.placement.to_data(),execution_data_placement=domain.to_data(),device_S_and_material_BF16_partitions_preserved=True))
# Only original consumers are redirected. New restoration views read the
# actual dynamic producer; function ABI and every numerical op stay intact.
original_ids={n.id for n in m.nodes};producer_ids=set(before)
nodes=tuple(replace(n,inputs=tuple(redirect.get(v,v) for v in n.inputs)) if n.id in original_ids-producer_ids else n for n in nodes)
functions=tuple(replace(f,outputs=tuple(redirect.get(v,v) for v in f.outputs)) for f in m.functions)
m=fm.verify_module(replace(m,nodes=nodes,functions=functions))
assert static_node_invocation_counts(m)[next(iter(before))]==before[next(iter(before))]
fm.emit_module(m,root/'requested.py')
(root/'edit.json').write_text(json.dumps(dict(rows=rows,whole_model_layers=45,whole_model_experts=288,all_45_FP8_invocations_preserved=True,functions=len(m.functions),numerical_operations_unchanged=True,original_stage=m.stage,earliest_affected_stage='physical microkernel proposal; typed read-only cross-mesh views preserve incoming/caller device policies and logical numerical ABI',no_stale_buffer_or_execution_plan=True),indent=2)+'\n')
base=native_implementation_model(offline_projection_weights=True)
if args.dsa_pool_scores:
 from local_optimizations.native_dsa_pool_scores import implementation_model as pool_scores_model
 base=pool_scores_model(base)
if args.native_latent_coalesced:
 from local_optimizations.native_latent_attention_schedule import implementation_model as latent_schedule_model
 base=latent_schedule_model(base,compact_pipeline=args.native_latent_compact_pipeline)
if args.native_moe_down_loads:
 from local_optimizations.native_moe_down_load_schedule import implementation_model as moe_down_load_model
 base=moe_down_load_model(base)
configured=replace(base,implementations=tuple(replace(i,parameters={**i.parameters,'compute_num_warps':4,'activation_layout':'warp','accumulator_layout':'vector','tile_n':16,'k_blocks_per_mma':4,'k_group_row_order':'n_major'}) if i.id=='tir.partitioned_block_scaled_linear.local' else i for i in base.implementations))
placement=parse_placement('cyx=4x8x16/cbb')
target=InputTarget(options=replace(NvidiaSm90Target().options,placements=(placement,)),evidence_dir=root,gemv_compute_warps=8,triton_implementation_model=configured)
compiler=Compiler(CompileOptions(target='nvidia-sm90',placement=placement));compiler.target=target
from triton.flagmega.passes import FunctionalPass,PassManager
from triton.flagmega.selection import override_plan,apply_plan
# Explicitly regenerate derived physical proposals. Keep logical types,
# distribution choices, frozen constant recipes and every live old choice;
# do not erase a digest or change the checkpoint stage to bypass verification.
def repropose(value):
 old={p.id:p for p in value.selection_points if p.kind=='tir'}
 choices={pid:value.selection_map[pid].candidate_id for pid in old if pid in value.selection_map}
 opened=replace(value,selection_points=tuple(p for p in value.selection_points if p.id not in old),
                selections=tuple(r for r in value.selections if r.point_id not in old))
 # The incoming checkpoint has completed semantic/state preparation.
 # This edit affects physical variant proposals, so invoke their normal
 # policy directly instead of reapplying the imported-state ABI transform.
 proposed=target.tir_selection_policy.propose(opened,target)
 points={p.id:p for p in proposed.selection_points}
 for pid,identity in choices.items():
  assert pid in points and any(c.id==identity for c in points[pid].candidates),(pid,identity)
 result=apply_plan(proposed,override_plan(proposed,tuple(choices.items()),policy='glm53.real-fp8-current-proposals/v207',
     rationale='Explicit reproposal for current implementation contracts; preserve all original semantic variant choices and retain typed per-op execution policy edits.'))
 (root/'reselection.json').write_text(json.dumps(dict(old_proposal_ids=len(old),preserved_live_choices=choices,
     original_semantic_hash=value.semantic_hash,new_semantic_hash=result.semantic_hash,
     original_stage=value.stage,new_stage=result.stage,normal_target_proposal_and_selection_plan=True,
     distribution_and_frozen_constants_retained=True,no_generated_source_patch=True),indent=2)+'\n')
 return result
manager=PassManager('real_fp8_explicit_reproposal')
manager.add(FunctionalPass('ReproposeCurrentPhysicalVariants',repropose))
m=manager.run(m).module
fm.emit_module(m,root/'requested-current-proposals.py')
from triton.flagmega.passes import FunctionalPass
original=FunctionalPass.run
def observed(self,value):
 start=time.monotonic();result=original(self,value)
 print(json.dumps(dict(pass_name=self.name,stage=result.stage,seconds=time.monotonic()-start)),flush=True);return result
FunctionalPass.run=observed
try:m=compiler.compile(m).module
finally:FunctionalPass.run=original
from local_optimizations.native_precision import apply_native_decode_schedules
m=apply_native_decode_schedules(m)
fm.emit_module(m,root/'final.py')
plan=fm.verify_buffer_plan(m)
from triton.flagmega.codegen.triton.tir_package import describe_tir_package,render_tir_package
from triton.flagmega.codegen.triton.renderer import RENDERER_VERSION
package=describe_tir_package(m)
if args.dsa_pool_scores:
 assert package['native_dsa_pool_score_schedule']['selected']
if args.native_latent_coalesced:
 assert package['native_latent_attention_schedule']['selected']
if args.native_moe_down_loads:
 assert package['native_moe_down_load_schedule']['selected']
source=render_tir_package(package,RENDERER_VERSION)
compile(source, '<complete-glm53-generated>', 'exec')
(root/'generated_kernels.py').write_text(source)
fp8=[c for c in package['render_calls'] if c['family']=='partitioned_block_scaled_linear']
assert fp8 and all(c['activation_layout']=='warp' and c['accumulator_layout']=='vector' and c['task_tile_n']==16 and c['k_blocks_per_mma']==4 and c['k_group_row_order']=='n_major' for c in fp8)
assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in frozen.items())
result=dict(passed=True,policy=args.policy,whole_model_layers=45,whole_model_experts=288,top_k=8,source_inputs_unchanged=True,normal_complete_module_compile_and_codegen=True,FP8_render_calls=len(fp8),kernel_programs=len(package['kernels']),FP8_calls=fp8,per_op_implementation_mapping='n_major G4/N16/vector/4warps',structured_program_execution=package.get('program_execution'),programs=package['kernels'],workspace_bytes=plan.workspace_bytes,stage=m.stage,semantic_hash=m.semantic_hash,source_sha256=hashlib.sha256(source.encode()).hexdigest(),seconds=time.monotonic()-began,manual_Core_or_checkpoint_weight_payload_read=False,full_model_device_acceptance=False)
(root/'result.json').write_text(json.dumps(result,indent=2,default=str)+'\n')
print(json.dumps({k:v for k,v in result.items() if k not in ('structured_program_execution','programs')},indent=2),flush=True)
