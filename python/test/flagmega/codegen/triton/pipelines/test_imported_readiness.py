# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""A mutable matrix follows a readonly matrix in one reusable pipeline call."""
from dataclasses import replace
import pytest
from triton.flagmega import ir as fm
from triton.flagmega.compiler import Compiler
from triton.flagmega.selection import override_plan
from triton.flagmega.codegen.triton import describe_tir_package, render_tir_package
from triton.flagmega.passes.tir.route_pipeline_readiness import route_pipeline_source_readiness
from triton.flagmega.errors import IRVerificationError, IRSchemaError


def compile_readiness_module(*, readonly_weight=True, nested=False):
    b=fm.IRBuilder(dialect='high_level',stage='vector_contracts_lowered')
    mesh=fm.Placement((8,16),'yx','bb');bc=fm.SBP.broadcast()
    row=fm.DistributedType(fm.tensor_type('bfloat16',(1,128)),(bc,bc),mesh)
    matrix=fm.DistributedType(fm.tensor_type('bfloat16',(128,128)),(bc,bc),mesh)
    owned=replace(matrix,axis_policies=(fm.SBP.split_contiguous((0,1)),bc))
    x=b.var('worker_x',row,id='worker_x')
    weight=b.var('worker_weight',matrix,id='worker_weight')
    mutable=b.var('worker_mutable',matrix,id='worker_mutable')
    def matmul(lhs,rhs,id):
        return b.call('math.matmul',(lhs,rhs),row,id=id,
            attrs={'transpose_a':False,'transpose_b':True},metadata={
                'selected_vectorization':'vectorization.matmul.n',
                'selected_vector_axes':(1,),'selected_vector_lanes':(8,)})
    first=matmul(x,weight,'readonly_projection')
    last=matmul(first,mutable,'mutable_projection')
    b.function('worker',(x,weight,mutable),(last,),attrs={
        'calling_convention':'device','noinline':True,'reusable':True})
    x=b.var('x',row,id='x');a=b.var('a',owned,id='a');v=b.var('v',owned,id='v')
    wr=b.weight('weight_root',matrix.tensor,source='memory',key='weight',id='weight_root')
    w=b.call('distributed.sharded_view',(wr,),matrix,id='weight',attrs={'new_type':matrix})
    if not readonly_weight: w=b.var('unproved_weight',matrix,id='unproved_weight')
    updated=b.call('math.add',(a,v),owned,id='write_mutable')
    updated=b.call('distributed.sharded_view',(updated,),matrix,id='mutable_view',attrs={'new_type':matrix})
    callee='worker'
    if nested:
        args=tuple(b.var('outer_'+str(i),ty,id='outer_'+str(i)) for i,ty in enumerate((row,matrix,matrix)))
        out=b.call('builtin.call',args,row,id='nested_worker',attrs={'callee':'worker'})
        b.function('outer',args,(out,),attrs={'calling_convention':'device','noinline':True,'reusable':True})
        callee='outer'
    call=b.call('builtin.call',(x,w,updated),row,id='call_worker',attrs={'callee':callee})
    b.function('main',(x,a,v,*(() if readonly_weight else (w,))),(call,))
    module=b.build(entry='main')
    module=replace(module,metadata={**module.metadata,'auto_distribution':{'placement':mesh.to_data()},
        'launch_contract':{'cooperative_grid':True,'grid_mesh':mesh.to_data(),'num_warps':4}})
    c=Compiler();m=c.compile(module,stop_after='propose-tir').module
    m=c.run_stage(m,'lower-tir',plan=override_plan(m,tuple(
        ('tir.'+name,'tir.dense_matmul.tensor_descriptor_smem_pipeline_gemv')
        for name in ('readonly_projection','mutable_projection')))).module
    return c.compile(m).module


@pytest.fixture(scope='module')
def pair():
    before=compile_readiness_module()
    after=route_pipeline_source_readiness(before,callees=('worker',))
    return before,after


def stage(module):
    r=module.execution_function_map['main'].body.fields[0]
    return next(x for x in r.produce_body.fields if isinstance(x,fm.PipelineStage))


def test_existing_gate_is_routed_without_deleting_publication_or_drains(pair):
    before,after=pair
    assert stage(before).source_readiness is None
    assert stage(after).source_readiness.before_call_id=='mutable_projection'
    a=before.execution_function_map['main'].body.fields[0]
    b=after.execution_function_map['main'].body.fields[0]
    assert [x for x in a.consume_body.fields if not isinstance(x,fm.PipelineStage)] == [
        x for x in b.consume_body.fields if not isinstance(x,fm.PipelineStage)]
    assert [x for x in a.produce_body.fields if isinstance(x,(fm.Barrier,fm.PipelineDrain))] == [
        x for x in b.produce_body.fields if isinstance(x,(fm.Barrier,fm.PipelineDrain))]
    assert before.metadata['buffer_plan']==after.metadata['buffer_plan']
    assert before.kernel_definitions==after.kernel_definitions
    assert fm.verify_module(after) is after


def test_normal_source_passes_one_existing_reader_and_waits_after_readonly_stage(pair):
    _,after=pair;p=describe_tir_package(after)
    fn=next(f for f in p['device_functions'] if f['function']=='worker')
    e=fn['schedule']['producer_events']
    order=[x.get('call',x.get('endpoint')) for x in e if x['kind'] in ('pipeline_kernel_call','handoff')]
    assert order==['readonly_projection','_flagmega_incoming_source_reader','mutable_projection']
    assert not fn['schedule']['handoffs']
    assert '_flagmega_incoming_source_reader' in fn['schedule']['producer_endpoints']
    parent=p['pipeline_schedule']
    assert len(parent['handoffs'])==1
    producer=next(x for x in parent['producer_events'] if x['kind']=='function_call')
    assert producer['endpoint_bindings']['_flagmega_incoming_source_reader']==parent['handoffs'][0]['reader_name']
    assert not any(x['kind']=='handoff' for x in parent['producer_events'])
    compile(render_tir_package(p,'readiness-test'),'readiness-test','exec')


def test_none_serialization_and_empty_policy_are_identity(pair):
    before,_=pair
    assert route_pipeline_source_readiness(before) is before
    assert 'source_readiness' not in stage(before).to_data()


def test_tir_contract_roundtrip(pair):
    _,after=pair;s=stage(after)
    assert fm.tir_from_data(s.to_data())==s
    assert route_pipeline_source_readiness(after,callees=('worker',)) is after


@pytest.mark.parametrize('field,value',[
    ('before_call_id','readonly_projection'),('handoff_id','wrong_epoch_reader'),
    ('readonly_parameters',('worker_mutable',)),
])
def test_wrong_control_or_readonly_identity_rejected(pair,field,value):
    _,m=pair;main=m.execution_function_map['main'];r=main.body.fields[0]
    original=stage(m);changed=replace(original,source_readiness=replace(original.source_readiness,**{field:value}))
    # Wrong endpoint IDs are rejected already by the local role-balance schema.
    with pytest.raises((IRVerificationError,IRSchemaError)):
        update=lambda body: fm.Sequential(tuple(changed if x==original else x for x in body.fields))
        r=fm.ProducerConsumerRegion(update(r.produce_body),update(r.consume_body))
        n=replace(m,execution_functions=tuple(replace(f,body=fm.Sequential((r,)))
                  if f.name=='main' else f for f in m.execution_functions))
        fm.verify_module(n)


def test_repeated_epoch_rejected(pair):
    _,m=pair
    n=replace(m,execution_functions=tuple(replace(f,attrs={**f.attrs,'region_kind':'for'})
              if f.name=='main' else f for f in m.execution_functions))
    with pytest.raises(IRVerificationError,match='non-repeated'):
        fm.verify_module(n)


def test_actual_external_weight_cannot_be_claimed_readonly():
    m=compile_readiness_module(readonly_weight=False)
    with pytest.raises(IRVerificationError,match='readonly prefix'):
        route_pipeline_source_readiness(m,callees=('worker',))


def test_nested_mutable_transfer_requires_explicit_supported_mapping():
    m=compile_readiness_module(nested=True)
    with pytest.raises(IRVerificationError,match='direct, non-nested'):
        route_pipeline_source_readiness(m,callees=('outer',))


def test_original_consumer_publication_is_still_required(pair):
    _,m=pair;f=m.execution_function_map['main'];r=f.body.fields[0]
    consumer=fm.Sequential(tuple(x for x in r.consume_body.fields if not isinstance(x,fm.Barrier)))
    r=replace(r,consume_body=consumer)
    m=replace(m,execution_functions=tuple(replace(fn,body=fm.Sequential((r,))) if fn.name=='main' else fn
                                        for fn in m.execution_functions))
    with pytest.raises(IRVerificationError,match='publication barrier'):
        fm.verify_module(m)
