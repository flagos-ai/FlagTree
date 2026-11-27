# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""CPU checks of frozen graph/native phase bindings (not device qualification)."""
from copy import deepcopy
from dataclasses import replace
import hashlib
from types import SimpleNamespace

import pytest
import torch

from triton.flagmega import ir as fm
from triton.flagmega.errors import ArtifactError, CodegenError, IRSchemaError, IRVerificationError, RuntimeContractError
from triton.flagmega.rules.ntt.r004_gdn_phase_bindings import (
    reference_phase_plan, validate_phase_plan, phase_writer_proof, reference_writer_census,
    phase_runtime_binding, verify_phase_runtime_binding, verify_phase_external_arguments,
    configure_standalone_phases, REQUEST_SCHEMA, MODULE, DYNAMIC_ROOTS, SCALAR_ROOTS, phase_variant, project_phase_roots,
)
from triton.flagmega.codegen.triton.kernel_programs import _task_owner_partition, target_kernel_programs
from triton.flagmega.targets import NvidiaSm120Target


def aligned(shape,dtype):
    count=1
    for d in shape:count*=d
    storage=torch.empty(count*dtype.itemsize+255,dtype=torch.uint8)
    offset=(-storage.data_ptr())%256
    return storage[offset:offset+count*dtype.itemsize].view(dtype).reshape(shape)


@pytest.fixture(scope='module')
def roots():
    values=tuple((0.0 if i in (14,15) else (0 if i in (4,13) else aligned(r['shape'],getattr(torch,r['dtype']))))
                 for i,r in enumerate(reference_phase_plan()['roots']))
    values[12].zero_()
    return values


def binding_for(roots):
    provenance={r['name']:dict(gdn_ordinal=0,sha256=hashlib.sha256(
        roots[i].view(torch.uint8).reshape(-1).numpy().tobytes()).hexdigest())
        for i,r in enumerate(reference_phase_plan()['roots']) if 6<=i<=10}
    return phase_runtime_binding(0,provenance)


def boundary(*, requested=True):
    b=fm.IRBuilder(dialect='nn',stage='unused_functions_removed')
    p=reference_phase_plan()
    inputs={}
    state_type=fm.RefType('qwen3_5_gated_delta_net_state',(('recurrent',fm.tensor_type(fm.vector_type('float32',(4,)),(48,48,128,32))),))
    for i,r in enumerate(p['roots'][:11]):
        if i==5:continue
        inputs[r['name']]=b.var(r['name'],state_type if i==3 else fm.tensor_type(r['dtype'],r['shape']),id=r['name'])
    # The coarse evaluator sees one selected state layer; native ABI receives
    # complete parent and dynamic layer independently, never a rebased pointer.
    sliced_type=fm.RefType(state_type.name,(('recurrent',fm.tensor_type(fm.vector_type('float32',(4,)),(1,48,128,32))),))
    state=b.call('tir.ref_slice',(inputs['state_parent'],inputs['gdn_layer']),sliced_type,attrs={'length':1},id='state_view')
    args=(state,inputs['qkv'],inputs['z'],inputs['projection_input'],inputs['b_weight'],inputs['a_weight'],
          inputs['a_log'],inputs['dt_bias'],inputs['norm_weight'])
    attrs=dict(num_key_heads=16,num_value_heads=48,key_head_dim=128,value_head_dim=128,epsilon=1e-6,
               value_head_order='tiled',b_weight_encoding='IQ4_XS',a_weight_encoding='IQ4_XS')
    from triton.flagmega.ir.ops.nn.gdn_recurrent_core import GatedDeltaNetRecurrentCore
    attrs=GatedDeltaNetRecurrentCore.normalize_attrs(attrs)
    core=b.call('nn.gdn_recurrent_core',args,fm.TupleType((inputs['z'].type,sliced_type)),attrs=attrs,id='recurrent',effect=fm.effect('read_write','gated_delta_net_state'))
    y=b.call('builtin.get_item',(core,),inputs['z'].type,attrs={'index':0},id='y')
    b.function('main',tuple(inputs.values()),(y,))
    module=b.build(entry='main')
    if requested:
        module=replace(module,metadata={**module.metadata,'r004_standalone_request':{
            'schema':REQUEST_SCHEMA,'module':MODULE,'readonly_layer':0}})
    return module


def selected(phase):
    p=reference_phase_plan();e=next(v for v in p['entries'] if v['phase']==phase)
    request={**e,'data_placement':fm.Placement(tuple(e['execution_mesh']['shape']),'xyz'[:len(e['execution_mesh']['shape'])],'b'*len(e['execution_mesh']['shape'])).to_data()}
    call=dict(parameters={'phase_plan':p,'r004_phase':phase,'program_grid':e['grid'],'program_owner_mapping':'x-fast'},
              scalar_writer_proof=phase_writer_proof(phase),internal_grid_barriers=0,inputs=[],outputs=[])
    return request,call


@pytest.mark.parametrize('phase',['pb','q','core','gather','clear'])
def test_rank_one_to_three_native_owner_partition(phase):
    request,call=selected(phase)
    assert _task_owner_partition(request,{'grid':[32,1,1]},[call]) is None


@pytest.mark.parametrize('cause',['product','grid','mapping','tasks','barrier','fact_barrier','pool_stride','proof','counts','empty'])
def test_rank_three_rejects_incomplete_owner_contract(cause):
    request,call=selected('core')
    calls=[call]
    if cause=='product':request['data_placement']=fm.Placement((32,),'x','b').to_data()
    if cause=='grid':request['grid']=[48,8,1]
    if cause=='mapping':call['parameters']['program_owner_mapping']='row_major'
    if cause=='tasks':request['tasks_per_owner']=2
    if cause=='barrier':call['internal_grid_barriers']=1
    if cause=='fact_barrier':call['facts']={'internal_grid_barriers':1}
    if cause=='pool_stride':call['outputs']=[{'buffers':[{'abi':{'pool_scope_stride_bytes':256}}]}]
    if cause=='proof':call.pop('scalar_writer_proof')
    if cause=='counts':call['scalar_writer_proof']['scalar_writes']['state']-=1
    if cause=='empty':calls=[]
    with pytest.raises(CodegenError):_task_owner_partition(request,{'grid':[32,1,1]},calls)


def test_writer_census_and_x_fast_distribution():
    from triton.flagmega.ir.execution_mesh import placement_for_execution_mesh
    mesh=fm.ExecutionMesh.fixed((48,1,8),axes=('head','batch','column_tile'),mapping='x-fast')
    domain=placement_for_execution_mesh(fm.Placement((32,),'b','b'),mesh)
    assert domain.physical_level_size('b')==384
    assert reference_writer_census()['state']==786432
    assert [h+48*batch+48*z for z in range(8) for batch in range(1) for h in range(48)]==list(range(384))


@pytest.mark.parametrize('value',[True,False])
def test_per_entry_static_option_schema(value):
    request=dict(symbol='phase',calls=['recurrent'],num_warps=4,enable_static_shared=value,maxnreg=80)
    module=replace(boundary(requested=False),metadata={'kernel_program_partition':[request]})
    assert target_kernel_programs(module,NvidiaSm120Target().capability)[0]['enable_static_shared'] is value


@pytest.mark.parametrize('value',[0,1,'true',None,[],{}])
def test_static_option_strict_boolean_rejection(value):
    request=dict(symbol='phase',calls=['recurrent'],num_warps=4,enable_static_shared=value)
    module=replace(boundary(requested=False),metadata={'kernel_program_partition':[request]})
    with pytest.raises(IRVerificationError,match='strict bool'):
        target_kernel_programs(module,NvidiaSm120Target().capability)


def test_static_option_capability_rejection():
    request=dict(symbol='phase',calls=['recurrent'],num_warps=4,enable_static_shared=True)
    module=replace(boundary(requested=False),metadata={'kernel_program_partition':[request]})
    with pytest.raises(IRVerificationError):
        target_kernel_programs(module,SimpleNamespace(warp_size=32,max_threads_per_block=1024,supported_program_compile_options=()))


@pytest.mark.parametrize('cause',['missing_root','static_layer','dtype','row','pool','option','dependency'])
def test_root_schema_failures(roots,cause):
    binding=binding_for(roots)
    if cause=='missing_root':binding['arguments'].pop(4)
    if cause=='static_layer':binding['root_dynamic_indices'].remove(4)
    if cause=='dtype':binding['arguments'][8]['dtype']='bfloat16'
    if cause=='row':binding['readonly_provenance']['b_weight']['gdn_ordinal']=1
    if cause=='pool':binding['pools'][0]['pool_scope_stride_bytes']=256
    if cause=='option':binding['phase_plan']['entries'][0].pop('enable_static_shared')
    if cause=='dependency':binding['phase_plan']['entries'][-2]['depends_on']=[]
    with pytest.raises(IRSchemaError):verify_phase_runtime_binding(binding)


@pytest.mark.parametrize('cause',['oob','shape','alias','lifetime','offset'])
def test_typed_view_verifier_rejects_invalid_span(cause):
    p=reference_phase_plan()
    if cause=='oob':p['pool']['views'][-1]['nbytes']+=1024
    if cause=='shape':p['pool']['views'][0]['shape']=[49]
    if cause=='alias':p['pool']['views'][1]['byte_offset']=0
    if cause=='lifetime':p['pool']['views'][0]['last_reader']='pb'
    if cause=='offset':p['pool']['views'][0]['byte_offset']=4
    with pytest.raises(IRSchemaError):validate_phase_plan(p)


@pytest.mark.parametrize('cause',['layer','row','state_view','alias','readonly_payload','missing_root'])
def test_external_abi_rejections(roots,cause):
    binding=binding_for(roots);values=list(roots)
    if cause=='layer':values[4]=48
    if cause=='row':values[4]=1
    if cause=='state_view':values[3]=values[3][0]
    if cause=='alias':values[5]=values[1]
    if cause=='readonly_payload':binding['readonly_provenance']['b_weight']['sha256']='0'*64
    if cause=='missing_root':values.pop()
    with pytest.raises(RuntimeContractError):verify_phase_external_arguments(binding,values,check_payload=True)


def test_external_abi_valid(roots):
    assert verify_phase_external_arguments(binding_for(roots),roots,check_payload=True)['schema']==reference_phase_plan()['schema']


def test_policy_keeps_coarse_semantics_and_default_unchanged():
    default=boundary(requested=False)
    assert configure_standalone_phases(default) is default
    configured=configure_standalone_phases(boundary())
    assert [n.op for n in configured.nodes].count('nn.gdn_recurrent_core')==1
    assert configured.node_map['recurrent'].effect==default.node_map['recurrent'].effect
    assert len(configured.node_map['recurrent'].attrs['phase_plan']['entries'])==12
    with pytest.raises(IRVerificationError):configure_standalone_phases(replace(boundary(),stage='packed'))
    mixed=replace(configured,metadata={})
    with pytest.raises(IRVerificationError):configure_standalone_phases(mixed)


def test_serialization_roundtrip():
    configured=configure_standalone_phases(boundary())
    reloaded=fm.IRModule.from_data(configured.to_data())
    assert reloaded.semantic_hash==configured.semantic_hash
    assert configure_standalone_phases(reloaded).semantic_hash==configured.semantic_hash
    assert len(NvidiaSm120Target().distributed_placements(reloaded))==1


@pytest.mark.parametrize('field,value',[('enable_static_shared',1),('requires_grid_wide_sync',0),('maxnreg',80.0),('num_warps',4.0)])
def test_frozen_phase_options_reject_python_equal_wrong_types(field,value):
    p=reference_phase_plan();p['entries'][0][field]=value
    with pytest.raises(IRSchemaError):validate_phase_plan(p)


def test_unknown_program_option_rejected():
    request=dict(symbol='phase',calls=['recurrent'],num_warps=4,enable_static_shard=True)
    module=replace(boundary(requested=False),metadata={'kernel_program_partition':[request]})
    with pytest.raises(IRVerificationError,match='unknown fields'):
        target_kernel_programs(module,NvidiaSm120Target().capability)


def test_default_persistent_selection_is_original():
    from triton.flagmega.codegen.triton.candidates.simple import GdnCandidateProvider
    from triton.flagmega.codegen.triton.candidates.core import TritonCandidateContext
    from triton.flagmega.passes.auto_distributed.auto_distributed import AutoDistributedPass
    module=boundary(requested=False)
    # No layer view is needed to check the original candidate identity.
    nodes=tuple(replace(n,inputs=('state_parent',*n.inputs[1:])) if n.id=='recurrent' else n for n in module.nodes if n.id!='state_view')
    core=next(n for n in nodes if n.id=='recurrent')
    module=replace(module,nodes=nodes)
    target=NvidiaSm120Target()
    context=TritonCandidateContext(module,target,{},{},{},frozenset(),True,target.triton_implementation_model)
    proposal=GdnCandidateProvider().propose(core,context)
    assert proposal.default_candidate=='tir.gdn_recurrent.persistent'
    assert all(c.id!='tir.gdn_recurrent.r004_reference_v4' for c in proposal.candidates)


@pytest.fixture(scope='module')
def compiled_phase(roots):
    from triton.flagmega.compiler import Compiler
    compiler=Compiler();compiler.target=NvidiaSm120Target()
    module=compiler.compile(boundary()).module
    return replace(module,metadata={**module.metadata,'r004_runtime_binding':binding_for(roots)})


def test_normal_package_serialization_export_reload(tmp_path,compiled_phase):
    from triton.flagmega.codegen.triton import render_triton_package
    from triton.flagmega.artifacts import write_artifact
    from triton.flagmega.runtime import load
    reloaded=fm.IRModule.from_data(compiled_phase.to_data())
    assert reloaded.semantic_hash==compiled_phase.semantic_hash
    artifact=write_artifact(reloaded,tmp_path,target="nvidia-sm120",emit_executable=True)
    module=load(artifact)
    specs=module._kernel_launch_specs()
    assert len(specs)==12
    assert tuple(module.codegen['dynamic_argument_indices'])==tuple(DYNAMIC_ROOTS)
    assert specs[9]['dynamic_argument_indices']==(2,6,7)
    assert specs[9]['grid']==(48,1,8)
    assert all(s['stream_index']==0 and s['requires_grid_wide_sync'] is False for s in specs)
    assert [s['enable_static_shared'] for s in specs]==[False]*4+[True]+[False]*2+[True]+[False]*4
    assert module.codegen['phase_cache_key']


def test_real_prepare_call_forwards_entry_options_without_monkeypatch(tmp_path,compiled_phase,roots):
    """Protocol test double at compiler boundary; real module/prepare/BoundProgram."""
    from triton.flagmega.codegen.triton import render_triton_package
    from triton.flagmega.artifacts import write_artifact
    from triton.flagmega.runtime import load
    from triton.flagmega.runtime.module import GeneratedTirCallGraphModule
    from triton.backends.nvidia.compiler import CUDAOptions
    from triton.flagmega.runtime.program_arguments import BoundProgram
    artifact=write_artifact(compiled_phase,tmp_path,target="nvidia-sm120",emit_executable=True)
    loaded=load(artifact)
    calls=[];launches=[]
    class JitProtocolProbe:
        def __init__(self,jit):self.fn=jit.fn
        def run(self,*args,grid,warmup,**options):
            assert warmup is True
            # Use actual CUDAOptions, so a silently dropped option is visible.
            parsed=CUDAOptions(**options)
            calls.append((self.fn.__name__,args,grid,options,parsed.enable_static_shared))
            metadata=SimpleNamespace(num_warps=options['num_warps'],shared=0,static_shared=16 if parsed.enable_static_shared else 0,
                ptxas_stack_frame_bytes=0,ptxas_spill_store_bytes=0,ptxas_spill_load_bytes=0,
                global_scratch_size=0,profile_scratch_size=0)
            return SimpleNamespace(metadata=metadata,n_regs=32,n_spills=0,name=self.fn.__name__,function=1,
                packed_metadata=(),src=SimpleNamespace(attrs={}),
                launch_metadata=lambda *args:None,run=lambda *args:launches.append((self.fn.__name__,args)))
    runtime=GeneratedTirCallGraphModule(tmp_path,loaded.manifest,compiled_phase,
                                       tuple(JitProtocolProbe(jit) for jit in loaded.kernels))
    # CPU lifecycle harness supplies real aligned tensors without requesting a GPU.
    runtime._mark_loaded('cpu');runtime._pool_values={'phase_pool':roots[11]}
    runtime.prepare(*(roots[:11]+roots[12:]))
    assert len(calls)==12 and len(runtime._prepared)==12
    assert all(isinstance(p,BoundProgram) for p in runtime._prepared)
    assert [c[4] for c in calls]==[False]*4+[True]+[False]*2+[True]+[False]*4
    assert all(c[3]['maxnreg']==80 and c[3]['ptx_options']=='--minnctapersm=1' for c in calls)
    assert calls[9][2]==(48,1,8)
    assert isinstance(calls[7][1][1],torch.Tensor)  # readonly remains a pointer
    with pytest.raises(RuntimeContractError,match='serial dependency DAG'):
        runtime._prepared[8].launch(*tuple(roots[i] for i in DYNAMIC_ROOTS),stream=17)
    bad=list(tuple(roots[i] for i in DYNAMIC_ROOTS));bad[3]=roots[3][0]
    with pytest.raises(RuntimeContractError):runtime._prepared[0].launch(*bad,stream=17)
    assert len(launches)==0
    runtime.run_into(*tuple(roots[i] for i in DYNAMIC_ROOTS),stream=17,scenario='synthetic32',token_position=1)
    assert len(launches)==11
    assert [v[0] for v in launches]==[e['symbol'] for e in reference_phase_plan()['entries'][1:]]
    assert all(v[1][3]==17 for v in launches)
    initial=list(roots[i] for i in DYNAMIC_ROOTS);initial[7]=1
    runtime.run_into(*initial,stream=17,scenario='synthetic32',token_position=0)
    assert len(launches)==23
    assert [v[0] for v in launches[11:]]==[e['symbol'] for e in reference_phase_plan()['entries']]
    with pytest.raises(RuntimeContractError):
        runtime.run_into(*initial,stream=17,scenario='synthetic32',token_position=1)
    with pytest.raises(RuntimeContractError,match='one caller stream'):
        runtime.run_into(*tuple(roots[i] for i in DYNAMIC_ROOTS),stream=18,scenario='synthetic32',token_position=1)
    wrong=list(tuple(roots[i] for i in DYNAMIC_ROOTS));wrong[4]=1
    with pytest.raises(RuntimeContractError,match='readonly row'):
        runtime.run_into(*wrong,stream=17,scenario='synthetic32',token_position=1)
    assert len(launches)==23
    bound=runtime._prepared[0]
    runtime.close(synchronize=False)
    with pytest.raises(RuntimeContractError,match="prepared"):
        bound.launch(*tuple(roots[i] for i in DYNAMIC_ROOTS),stream=17)


def test_rank_two_padded_owner_partition():
    request,call=selected('pb')
    request['execution_mesh']=fm.ExecutionMesh.fixed((48,1),mapping='x-fast').to_data()
    request['data_placement']=fm.Placement((48,1),'xy','bb').to_data()
    assert _task_owner_partition(request,{'grid':[32,1,1]},[call]) is None


def test_tir_dispatch_roundtrip_and_unknown_kind_rejected(compiled_phase):
    from triton.flagmega.ir.tir.serialization import tir_to_data,tir_from_data
    from triton.flagmega.ir.tir.kernel_definition import replace_kernel_dispatch
    core=compiled_phase.kernel_callable_map['r004_C_core'].dispatch
    assert tir_to_data(tir_from_data(tir_to_data(core)))==tir_to_data(core)
    bad=reference_phase_plan();bad['pool']['views'][-1]['nbytes']+=1024
    invalid=replace(core,semantic_attrs={**core.semantic_attrs,'phase_plan':bad})
    with pytest.raises(IRSchemaError):tir_from_data(tir_to_data(invalid))
    with pytest.raises(ValueError,match='Unknown TIR'):
        tir_from_data({'kind':'invented_entry'})


@pytest.mark.parametrize('cause',['missing_root','static_layer','pool_scope'])
def test_normal_constructor_rejects_invalid_root_projection(tmp_path,compiled_phase,cause):
    from triton.flagmega.codegen.triton import render_triton_package
    from triton.flagmega.runtime.module import GeneratedTirCallGraphModule
    codegen=render_triton_package(compiled_phase,tmp_path)
    if cause=='missing_root':codegen['runtime_binding']['arguments'].pop(4)
    if cause=='static_layer':codegen['dynamic_argument_indices'].remove(4)
    if cause=='pool_scope':codegen['runtime_binding']['pools'][0]['pool_scope_stride_bytes']=256
    with pytest.raises(ArtifactError):
        GeneratedTirCallGraphModule(tmp_path,{'codegen':codegen,'target':'nvidia-sm120'},compiled_phase,SimpleNamespace())


def test_coarse_typed_workspaces_effects_and_cost():
    from triton.flagmega.ir.ops.nn.gdn_recurrent_core import GatedDeltaNetRecurrentCore as Op
    node=configure_standalone_phases(boundary()).node_map['recurrent']
    assert len(Op.phase_workspaces(node))==10
    effects=Op.phase_effects(node)
    assert effects[9]['accesses'][6]['mode']=='write'
    assert effects[9]['accesses'][6]['layer_source']==4
    assert [e['publishes_region_ready'] for e in effects]==[False]*11+[True]
    assert '12-registry-11-subsequent-12-initial' in Op.cost(node).notes
    assert effects[1]['accesses'][0]['mode']=='read'
    assert effects[1]['accesses'][2]['mode']=='write'
    assert effects[9]['accesses'][5]['mode']=='read'
    assert effects[9]['completion_cta_count']==384
    assert effects[0]['state_effect']=='initial_clear'


@pytest.mark.parametrize('cause',['selector','selector_stride','scale_negzero','bias_integer','flag','selector_alias'])
def test_new_external_root_failures(roots,cause):
    values=list(roots)
    if cause=='selector':
        values[12]=aligned([1,1,1,1],torch.int32).fill_(1)
    if cause=='selector_stride':values[12]=values[12].as_strided((1,1,1,1),(0,0,0,0))
    if cause=='scale_negzero':values[14]=-0.0
    if cause=='bias_integer':values[15]=0
    if cause=='flag':values[13]=2
    if cause=='selector_alias':values[12]=values[11][:4].view(torch.int32).reshape(1,1,1,1)
    with pytest.raises(RuntimeContractError):
        verify_phase_external_arguments(binding_for(roots),values,check_payload=True)


@pytest.mark.parametrize('scenario,position,flag',[('synthetic32',0,0),('chinese64',1,1),
                                                 ('synthetic32',32,0),('other',0,1)])
def test_variant_rejects_wrong_clear_boundary(scenario,position,flag):
    with pytest.raises(RuntimeContractError):phase_variant(scenario,position,flag)


def test_numbered_roots_and_scratch_completion(roots):
    binding=binding_for(roots)
    assert project_phase_roots(binding,roots[:11]+roots[12:],roots[11])==roots
    plan=reference_phase_plan()
    scratch=plan['pool']['views'][-1]
    assert scratch['byte_offset']+scratch['nbytes']==3212800
    assert scratch['writer']=='gather' and scratch['last_reader']=='core'
    assert scratch['release_after']=={'symbol':'r004_C_core','completion':'all_ctas','cta_count':384}
    assert plan['variants']['C_INITIAL']['entry_indices']==list(range(12))
    assert plan['variants']['C_SUBSEQUENT']['entry_indices']==list(range(1,12))
    assert plan['scenario_weights']['synthetic32']['executed_launches']==16944
    assert plan['scenario_weights']['chinese64']['executed_launches']==33840


@pytest.mark.parametrize('cause',['missing_option','unknown_option','maxnreg','grid','dependency','native_binding','cache'])
def test_normal_reload_rejects_spec_forgery(tmp_path,compiled_phase,cause):
    from triton.flagmega.artifacts import write_artifact
    from triton.flagmega.runtime import load
    artifact=write_artifact(compiled_phase,tmp_path,target='nvidia-sm120',emit_executable=True)
    runtime=load(artifact)
    spec=runtime.codegen['kernels'][0]
    if cause=='missing_option':spec.pop('enable_static_shared')
    if cause=='unknown_option':spec['enable_static_shard']=True
    if cause=='maxnreg':spec['maxnreg']=88
    if cause=='grid':spec['grid']=[32,1,1]
    if cause=='dependency':runtime.codegen['kernels'][8]['depends_on']=[]
    if cause=='native_binding':spec['argument_bindings'][1]['source']=7
    if cause=='cache':runtime.codegen['phase_cache_key']='0'*64
    with pytest.raises(ArtifactError):runtime._kernel_launch_specs()


def test_fresh_bootstrap_and_reload_process(tmp_path):
    import subprocess
    import os
    script='''import runpy
from dataclasses import replace
from triton.flagmega.compiler import Compiler
from triton.flagmega.targets import NvidiaSm120Target
from triton.flagmega.artifacts import write_artifact
from triton.flagmega.runtime import load
ns=runpy.run_path("python/test/flagmega/codegen/triton/test_r004_phase_bindings.py")
roots=ns["roots"].__wrapped__()
compiler=Compiler();compiler.target=NvidiaSm120Target()
module=compiler.compile(ns["boundary"]()).module
module=replace(module,metadata={**module.metadata,"r004_runtime_binding":ns["binding_for"](roots)})
artifact=write_artifact(module,OUTPUT,target="nvidia-sm120",emit_executable=True)
assert len(load(artifact)._kernel_launch_specs())==12
print("fresh-bootstrap:12-registry-entries")
'''
    script=script.replace('OUTPUT',repr(str(tmp_path)))
    environment={**os.environ,'PYTHONPATH':'/mnt/home-nas/work/repo/flagtree/python'}
    result=subprocess.run(['/home/sunnycase/miniconda3/envs/flagtree/bin/python','-c',script],
                          env=environment,capture_output=True,text=True)
    assert result.returncode==0,result.stdout+result.stderr
    assert 'fresh-bootstrap:12-registry-entries' in result.stdout


def test_phase_contract_preserves_eager_evaluation():
    from triton.flagmega.ir.ops.nn._gdn_state import GatedDeltaNetStateConfig,create_gdn_state
    from triton.flagmega.ir.ops.nn.gdn_recurrent_core import gated_delta_net_recurrent_core
    config=GatedDeltaNetStateConfig(1,16,48,128,128,4,5120)
    state1=create_gdn_state(config);state2=create_gdn_state(config)
    generator=torch.Generator().manual_seed(17)
    initial=torch.randn((48,128,128),generator=generator)*.001
    state1.update_recurrent_layer(initial);state2.update_recurrent_layer(initial)
    arguments=dict(qkv=torch.randn((1,10240),generator=generator).bfloat16(),
                   z=torch.randn((1,6144),generator=generator).bfloat16(),
                   projection_input=torch.zeros((1,5120),dtype=torch.bfloat16),
                   b_weight=torch.zeros((48,2720),dtype=torch.uint8),
                   a_weight=torch.zeros((48,2720),dtype=torch.uint8),
                   a_log=torch.zeros(48),dt_bias=torch.zeros(48),norm_weight=torch.ones(128))
    attrs=dict(boundary().node_map['recurrent'].attrs)
    y1,_=gated_delta_net_recurrent_core(state=state1,attrs=attrs,**arguments)
    y2,_=gated_delta_net_recurrent_core(state=state2,attrs={**attrs,'phase_plan':reference_phase_plan()},**arguments)
    assert torch.equal(y1,y2)
    assert torch.equal(state1.recurrent,state2.recurrent)
