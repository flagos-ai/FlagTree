# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""CPU-only admission, physical resource and ordered numerical C2 regressions."""
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
import ctypes
from math import prod
import re
import shutil
import subprocess

import numpy as np
import pytest

from triton.flagmega import ir as fm
from triton.flagmega.compiler import Compiler
from triton.flagmega.options import CompileOptions
from triton.flagmega.codegen.triton.candidates.core import TritonCandidateContext
from triton.flagmega.codegen.triton.candidates.simple import GdnCandidateProvider
from triton.flagmega.codegen.triton.candidates.gdn_head_gates import IMPLEMENTATION, dense_head_geometry
from triton.flagmega.codegen.triton.gdn_head_gates import prepare_fields, validate_call
from triton.flagmega.codegen.cpp.templates import CppTemplateRegistry
from triton.flagmega.errors import CodegenError
from triton.flagmega.ir.ops.nn._gdn_state import GatedDeltaNetStateConfig
from triton.flagmega.targets import get_target
from python.test.flagmega.codegen.cpp.test_native_quantized_kernels import FTT_INCLUDE, bf16, packed_rows, run


def graph(hierarchy=(4, 8)):
    config = GatedDeltaNetStateConfig(1, 16, 48, 128, 128, 4, 5120)
    placement = fm.Placement(hierarchy, "yx", "b" * len(hierarchy))
    b = fm.SBP.broadcast(); split = fm.SBP.split_contiguous((0, 1))
    builder = fm.IRBuilder(dialect="distributed", stage="frozen_constants")
    state = builder.var("state", config.ref_type, id="state")
    inputs = [state]
    for name, shape, dtype in (
            ("qkv", (1, 10240), "bfloat16"), ("z", (1, 6144), "bfloat16"),
            ("projection_input", (1, 5120), "bfloat16"),
            ("b_weight", (48, 2720), "uint8"), ("a_weight", (48, 2720), "uint8"),
            ("a_log", (48,), "float32"), ("dt_bias", (48,), "float32"), ("norm_weight", (128,), "float32")):
        policies = (b, split) if name == "z" else (b,) * len(shape)
        inputs.append(builder.var(name, fm.DistributedType(fm.tensor_type(dtype, shape), policies, placement), id=name))
    output = fm.TupleType((inputs[2].type, state.type))
    result = builder.call("nn.gdn_recurrent_core", tuple(inputs), output,
                          attrs={"key_head_dim": 128, "value_head_dim": 128, "num_key_heads": 16,
                                 "num_value_heads": 48, "epsilon": 1e-6, "b_weight_encoding": "IQ4_XS",
                                 "a_weight_encoding": "IQ4_XS", "value_head_order": "tiled"}, id="recurrent")
    builder.function("main", tuple(inputs), (result,))
    return builder.build(entry="main")


def context(module, backend="cpp", target="nvidia-sm120"):
    target = get_target(target)
    return TritonCandidateContext(module, target, {}, {}, {}, frozenset(), True,
                                  target.triton_implementation_model, backend)


@pytest.mark.parametrize("backend", [None, "triton", "cpp"])
def test_backend_admission_is_before_selection_and_default_stays_ordinary(backend):
    m=graph(); p=GdnCandidateProvider().propose(m.node_map["recurrent"], context(m,backend))
    assert p.default_candidate == "tir.gdn_recurrent.persistent"
    assert (IMPLEMENTATION in [c.id for c in p.candidates]) == (backend == "cpp")


@pytest.mark.parametrize("change", [
    {"round_beta":False}, {"round_core":True}, {"round_normalized_qk":True},
    {"qk_norm_mode":"add"}, {"qk_norm_epsilon":1e-6}, {"gate_math_mode":"fast"},
    {"norm_math_mode":"accurate"}, {"value_head_order":"grouped"}, {"a_weight_encoding":"Q4_K"},
])
def test_uncertified_numeric_modes_do_not_advertise_cache(change):
    m=graph(); nodes=tuple(replace(n,attrs={**n.attrs,**change}) if n.id=="recurrent" else n for n in m.nodes)
    m=replace(m,nodes=nodes); p=GdnCandidateProvider().propose(m.node_map["recurrent"],context(m))
    assert IMPLEMENTATION not in [c.id for c in p.candidates]


def test_backend_context_survives_target_reconfiguration_without_global_mutation():
    target=get_target("nvidia-sm120")
    compiler=Compiler(CompileOptions(target="nvidia-sm120", codegen_backend="cpp",
                                     bufferize_opt_level="fast"))
    assert compiler.target.codegen_backend=="cpp"
    placed=Compiler(CompileOptions(target="nvidia-sm90",codegen_backend="cpp",
                                   placement=fm.Placement((4,8),"yx","bb")))
    assert placed.target.codegen_backend=="cpp"
    assert getattr(target,"codegen_backend",None) is None
    assert getattr(Compiler(CompileOptions(target="nvidia-sm120")).target,"codegen_backend",None) is None


def test_other_target_and_overridden_threads_reject_before_selection(monkeypatch):
    m=graph();node=m.node_map['recurrent']
    assert IMPLEMENTATION not in [c.id for c in GdnCandidateProvider().propose(node,context(m,target='nvidia-sm90')).candidates]
    monkeypatch.setenv('FLAGMEGA_CPP_BLOCK_THREADS','256')
    assert IMPLEMENTATION not in [c.id for c in GdnCandidateProvider().propose(node,context(m)).candidates]


def test_replication_or_packed_projection_does_not_advertise_cache():
    m=graph();node=m.node_map['recurrent'];out=node.type.fields[0]
    replicated=replace(out,axis_policies=(fm.SBP.broadcast(),fm.SBP.broadcast()))
    bad=replace(node,type=fm.TupleType((replicated,node.type.fields[1])))
    assert IMPLEMENTATION not in [c.id for c in GdnCandidateProvider().propose(bad,context(m)).candidates]
    source=m.node_map['projection_input'];packed=replace(source.type,tensor=fm.tensor_type(fm.VectorType(fm.DType.BFLOAT16,(8,)),(1,640)))
    m=replace(m,nodes=tuple(replace(n,type=packed) if n.id==source.id else n for n in m.nodes))
    assert IMPLEMENTATION not in [c.id for c in GdnCandidateProvider().propose(m.node_map['recurrent'],context(m)).candidates]


TYPES={name:typ for name,typ in (
    ('qkv','ftt::bfloat16_t'),('z','ftt::bfloat16_t'),('projection_input','ftt::bfloat16_t'),
    ('b_weight','unsigned char'),('a_weight','unsigned char'),('a_log','float'),('dt_bias','float'),
    ('norm_weight','float'),('state','float'),('scratch','float'),('result','ftt::bfloat16_t'))}


def call(cache=True, offset=0, hierarchy=(4, 8)):
    capacity=6144//prod(hierarchy)
    start=f"shard_y * {hierarchy[1]*capacity} + shard_x * {capacity}"
    return dict(family='gdn_recurrent',variant='persistent',symbol='kernel',arguments=list(TYPES),
        **{name:name for name in TYPES if name!='scratch'},core_scratch='scratch',
        z_offset=start+' + local_values',
        result_offset=start+' + local_values',local_capacity=capacity,active_values=str(capacity),
        global_value='local_values + '+start,value_heads=48,key_heads=16,
        key_dim=128,value_dim=128,projection_dim=5120,repeats=3,epsilon='1e-6',qk_norm_epsilon='1e-12',
        query_scale=repr(128**-0.5),qk_norm_add=False,round_normalized_qk=False,round_beta=True,
        round_core=False,gate_math_mode='accurate',accurate_norm_silu=False,writer_active='True',
        participation_active='True',b_weight_encoding='IQ4_XS',a_weight_encoding='IQ4_XS',tiled_value_heads=True,
        head_gate_cache=cache,head_gate_offset_bytes=offset,head_gate_arena_bytes=offset+16,
        head_gate_physical_buffer='shared:head_gates',head_gate_first_value=start,
        implementation=IMPLEMENTATION,internal_grid_barriers=1,
        parameters=dict(head_gate_cache=cache,head_cache_slots=2,projection_order='ascending_scalar_C1',
                        cache_contents=('beta','decay'),cache_epoch='per_GDN_invocation'),
        facts=dict(internal_cta_barriers=1,required_codegen_backend='cpp'))


def render(c, hierarchy=(4,8)):
    return CppTemplateRegistry().render('kernels/gdn_recurrent/persistent.cpp.jinja',dict(
        render_calls=[c],argument_types={n:dict(type=t,pointer=True) for n,t in TYPES.items()},
        mesh_hierarchy=hierarchy,distributed_entry=True,block_threads=128,
        grid_barrier_axis_groups=[],grid_total_blocks=prod(hierarchy)))


def abi(shape,dtype='float32',lanes=1):
    strides=[]; size=lanes
    for n in reversed(shape):strides.insert(0,size);size*=n
    return dict(local_capacity_shape=shape,scalar_storage_strides=strides,scalar_itemsize=4,
                scalar_dtype=dtype,scalar_lane_count=lanes,scalar_lane_shape=(4,) if lanes==4 else (),
                physical_buffer='private',pool_byte_offset=0)


def raw():
    shapes={'qkv':((1,10240),'bfloat16',1),'z':((1,192),'bfloat16',1),
            'projection_input':((1,5120),'bfloat16',1),'b_weight':((48,2720),'uint8',1),
            'a_weight':((48,2720),'uint8',1),'a_log':((48,),'float32',1),
            'dt_bias':((48,),'float32',1),'norm_weight':((128,),'float32',1)}
    def binding(name,shape,dtype='float32',lanes=1):
        a=abi(shape,dtype,lanes);a['physical_buffer']=name
        return dict(abi=a,runtime_argument=name)
    inputs=[dict(formal=n,buffers=[binding(n,*s)]) for n,s in shapes.items()]
    inputs.append(dict(formal='state',buffers=[binding('conv',(1,1280,3),'bfloat16',8),binding('state',(1,48,128,32),'float32',4)]))
    return dict(implementation=IMPLEMENTATION,parameters=dict(head_gate_cache=True),inputs=inputs,
        outputs=[dict(formal='result_0',buffers=[binding('result',(1,192),'bfloat16')])],
        workspaces=[dict(formal='core_scratch',buffers=[binding('scratch',(48,128))])],
        shared_workspaces=[dict(name='head_gates',dtype='float32',shape=(2,2),strides=(2,1),
                                nbytes=16,alignment_bytes=16,offset_bytes=0,physical_buffer='shared:head_gates')])


@pytest.mark.parametrize('offset',[0,16,32,256])
def test_planned_offset_is_honored_and_typed_span_is_exact(offset):
    r=raw();r['shared_workspaces'][0]['offset_bytes']=offset
    fields=prepare_fields(r,call()); assert fields['head_gate_arena_bytes']==offset+16
    source=render(call(offset=offset))
    assert f'ftt_head_gate_storage + {offset}' in source
    assert source.count('ftt::arch::cta_sync();')==1


@pytest.mark.parametrize('field,value',[('offset_bytes',1),('offset_bytes',-16),('nbytes',32),('dtype','bfloat16'),('strides',(1,2)),('alignment_bytes',8),('matrix_compatible',True)])
def test_invalid_shared_contract_rejects(field,value):
    r=raw();r['shared_workspaces'][0][field]=value
    with pytest.raises(CodegenError):prepare_fields(r,call())


def test_aliasing_state_or_scratch_with_readonly_projection_rejects():
    r=raw();projection=next(p for p in r['inputs'] if p['formal']=='projection_input')['buffers'][0]
    r['workspaces'][0]['buffers'][0]=deepcopy(projection)
    with pytest.raises(CodegenError,match='disjoint'):prepare_fields(r,call())


@pytest.mark.parametrize('field,value',[('block_threads',256),('warp_size',64),('platform','iluvatar'),('target_architecture','sm90')])
def test_final_physical_launch_rechecks_candidate_contract(field,value):
    ctx=dict(platform='nvidia',target_architecture='sm120',block_threads=128,warp_size=32,
             mesh_hierarchy=(4,8),launch_resource_limits=dict(max_shared_memory_bytes=101376))
    validate_call(call(),ctx);ctx[field]=value
    with pytest.raises(CodegenError):validate_call(call(),ctx)


@pytest.mark.parametrize('field,value',[('writer_active','shard_x == 0'),('participation_active','shard_x == 0'),
                                      ('active_values','191'),('local_capacity',193),
                                      ('global_value','local_values + shard_x * 193 + shard_y * 1536')])
def test_partial_nonwriter_or_altered_mapping_cannot_reach_publication(field,value):
    ctx=dict(platform='nvidia',target_architecture='sm120',block_threads=128,warp_size=32,
             mesh_hierarchy=(4,8),launch_resource_limits=dict(max_shared_memory_bytes=101376))
    c=call();c[field]=value
    with pytest.raises(CodegenError):validate_call(c,ctx)


def host_library(tmp_path, hierarchy=(4,8)):
    mesh=prod(hierarchy); xsize=hierarchy[1]
    if shutil.which('c++') is None:pytest.skip('C++ host compiler required')
    cached=render(call(offset=32,hierarchy=hierarchy),hierarchy); ordinary=render(dict(call(False,hierarchy=hierarchy),symbol='ordinary'),hierarchy)
    body=cached+'\n'+ordinary
    body=re.sub(r'ftt::arch::mesh_shard_id<(\d+)>\([^)]*\)',r'test_shard[\1]',body)
    body=body.replace('ftt::arch::thread_id()','test_thread')
    body=body.replace('__shared__ __align__(16)','alignas(16) static')
    body=body.replace('ftt::arch::cta_sync();','test_barrier.arrive_and_wait();')
    body=body.replace('ftt_head_gates[2 * ftt_thread] = beta;',
        f'test_pair_count.fetch_add(1); test_gates[(test_shard[0]*{xsize}+test_shard[1])*4+2*ftt_thread]=beta; '
        f'test_gates[(test_shard[0]*{xsize}+test_shard[1])*4+2*ftt_thread+1]=decay; ftt_head_gates[2 * ftt_thread] = beta;')
    params=', '.join(f'{t} *{n}' for n,t in TYPES.items());args=', '.join(TYPES)
    text=f'''#include <barrier>
#include <atomic>
#include <thread>
#include <vector>
#include <cmath>
struct float4 {{float x,y,z,w;}};
#include <ftt/ftt.h>
static thread_local ftt::dim_t test_shard[2],test_thread;
static std::barrier test_barrier(128);
static std::atomic<int> test_pair_count(0);
static float test_gates[{mesh*4}];
static unsigned int _ftt_grid_barrier_counters[2];
{body}
extern "C" void run({params}) {{
test_pair_count=0;
for(int i=0;i<{mesh*4};++i)test_gates[i]=NAN;
std::vector<std::thread> threads;
for(int t=0;t<128;++t) threads.emplace_back([&,t]{{test_thread=t;
for(int s=0;s<{mesh};++s) {{test_shard[0]=s/{xsize};test_shard[1]=s%{xsize};kernel__core({args});test_barrier.arrive_and_wait();}}
for(int s=0;s<{mesh};++s) {{test_shard[0]=s/{xsize};test_shard[1]=s%{xsize};kernel__norm({args});test_barrier.arrive_and_wait();}}
}});
for(auto &t:threads)t.join();}}
extern "C" int pair_count(){{return test_pair_count.load();}}
extern "C" void observed_gates(float *out){{for(int i=0;i<{mesh*4};++i)out[i]=test_gates[i];}}
extern "C" void reference_gates(ftt::bfloat16_t*x,unsigned char*bw,unsigned char*aw,float*alog,float*bias,float*out){{
for(int h=0;h<48;++h){{float b=0,a=0;
for(int k=0;k<5120;++k){{
b += ftt::quantized::decode<ftt::quantized::encoding::iq4_xs>(bw+h*2720,k)*ftt::to_float(x[k]);
a += ftt::quantized::decode<ftt::quantized::encoding::iq4_xs>(aw+h*2720,k)*ftt::to_float(x[k]);}}
b=ftt::to_float(ftt::to_bfloat16(b));a=ftt::to_float(ftt::to_bfloat16(a));
out[2*h]=ftt::to_float(ftt::to_bfloat16(ftt::sigmoid(b)));
const float d=a+bias[h];const float sp=d>20?d:log1pf(expf(d));
out[2*h+1]=expf(-expf(alog[h])*sp);}}}}
extern "C" void ordinary_run({params}) {{
for(int s=0;s<{mesh};++s) {{test_shard[0]=s/{xsize};test_shard[1]=s%{xsize};for(test_thread=0;test_thread<128;++test_thread)ordinary__core({args});}}
for(int s=0;s<{mesh};++s) {{test_shard[0]=s/{xsize};test_shard[1]=s%{xsize};for(test_thread=0;test_thread<128;++test_thread)ordinary__norm({args});}}
}}
'''
    source=tmp_path/'c2.cpp';source.write_text(text);so=tmp_path/'c2.so'
    subprocess.run(['c++','-std=c++20','-O2','-shared','-fPIC','-pthread','-DFTT_ARCH_ILUVATAR',f'-I{FTT_INCLUDE}',str(source),'-o',str(so)],check=True,capture_output=True,text=True)
    return ctypes.CDLL(str(so))


@pytest.mark.parametrize("hierarchy",[(4,8),(4,24)])
def test_real_128_thread_publication_matches_c1_bits_across_epochs(tmp_path,hierarchy):
    lib=host_library(tmp_path,hierarchy)
    mesh=prod(hierarchy);capacity,bounds=dense_head_geometry(hierarchy,48,128)
    rng=np.random.default_rng(126)
    bw,_=packed_rows('IQ4_XS',rows=48,blocks=20);aw=bw[::-1].copy()
    # Keep projections moderate; threshold +/-20 witnesses come from dt_bias.
    bw[:,:2]=np.frombuffer(np.float16(1/32768).tobytes(),np.uint8)
    aw[:,:2]=np.frombuffer(np.float16(1/32768).tobytes(),np.uint8)
    alog=np.linspace(-3,1,48,dtype=np.float32)
    bias=np.resize(np.array([-100,-20,0,19.999,20,20.001,100],np.float32),48)
    norm=rng.normal(1,.1,128).astype(np.float32)
    state=rng.normal(0,.01,(6144,128)).astype(np.float32);reference=state.copy()
    for epoch in range(3):
        qkv=bf16(rng.normal(0,.2,10240));z=bf16(rng.normal(0,.3,6144));projection=bf16(rng.normal(0,.01,5120))
        if epoch==0:qkv[:128]=0
        scratch=np.full(6144,np.nan,np.float32);expected_scratch=scratch.copy()
        output=np.full(6144,0x7fc0,np.uint16);expected=output.copy()
        arrays=(qkv,z,projection,bw,aw,alog,bias,norm)
        run(lib,*arrays,state,scratch,output)
        assert lib.pair_count()==sum(needed for _,needed in bounds)
        gates=np.empty(mesh*4,np.float32);expected_gates=np.empty(96,np.float32)
        lib.observed_gates(gates.ctypes.data_as(ctypes.c_void_p))
        lib.reference_gates(*(a.ctypes.data_as(ctypes.c_void_p) for a in (projection,bw,aw,alog,bias,expected_gates)))
        expected_gates=expected_gates.reshape(48,2)
        for s,(h,needed) in enumerate(bounds):
            np.testing.assert_array_equal(gates[s*4:s*4+2*needed].view(np.uint32),expected_gates[h:h+needed].reshape(-1).view(np.uint32))
            assert np.isnan(gates[s*4+2*needed:s*4+4]).all()
        lib.ordinary_run(*(a.ctypes.data_as(ctypes.c_void_p) for a in (*arrays,reference,expected_scratch,expected)))
        np.testing.assert_array_equal(state.view(np.uint32),reference.view(np.uint32))
        np.testing.assert_array_equal(scratch.view(np.uint32),expected_scratch.view(np.uint32))
        np.testing.assert_array_equal(output,expected)
        assert np.isfinite(state).all() and np.isfinite(scratch).all()


@pytest.mark.parametrize('hierarchy',[(4,8),(4,24),(2,16),(8,12),(4,32)])
def test_general_dense_geometry_is_proved_before_selection_and_lowering(hierarchy):
    m=graph(hierarchy);node=m.node_map['recurrent']
    proposal=GdnCandidateProvider().propose(node,context(m))
    assert proposal.default_candidate=='tir.gdn_recurrent.persistent'
    assert IMPLEMENTATION in [c.id for c in proposal.candidates]
    capacity,bounds=dense_head_geometry(hierarchy,48,128)
    assert capacity*prod(hierarchy)==6144 and all(needed in (1,2) and h+needed<=48 for h,needed in bounds)
    if hierarchy==(4,24):assert capacity==64 and bounds[-1]==(47,1)
    c=call(hierarchy=hierarchy);ctx=dict(platform='nvidia',target_architecture='sm120',block_threads=128,
        warp_size=32,mesh_hierarchy=hierarchy,launch_resource_limits=dict(max_shared_memory_bytes=101376))
    validate_call(c,ctx)
    c['outputs']=[dict(formal='result_0',buffers=[dict(abi={**abi((1,capacity),'bfloat16'),
        'distributed_type':node.type.fields[0].to_data()})])]
    validate_call(c,ctx)
    c['outputs'][0]['buffers'][0]['abi']['distributed_type']['placement']['hierarchy']=[4,8]
    if hierarchy!=(4,8):
        with pytest.raises(CodegenError):validate_call(c,ctx)


@pytest.mark.parametrize('hierarchy',[(4,25),(2,2),(4,4),(0,8),(),(4,True)])
def test_invalid_tail_or_more_than_two_head_geometry_rejects(hierarchy):
    assert dense_head_geometry(hierarchy,48,128) is None
    c=call();ctx=dict(platform='nvidia',target_architecture='sm120',block_threads=128,
        warp_size=32,mesh_hierarchy=hierarchy,launch_resource_limits=dict(max_shared_memory_bytes=101376))
    with pytest.raises(CodegenError):validate_call(c,ctx)


@pytest.mark.parametrize('change',['partial','exclusive','noncontiguous','subset_axes','block_levels','padded','dynamic_mesh','wrong_mesh','wrong_map'])
def test_geometry_proposal_rejects_unproved_ownership(change):
    from triton.flagmega.codegen.triton.candidates.gdn_head_gates import applicable
    m=graph((4,24));n=m.node_map['recurrent'];out=n.type.fields[0]
    if change=='partial':out=replace(out,partial=fm.SBPPartial((0,)))
    elif change=='exclusive':out=replace(out,axis_policies=(fm.SBP.broadcast(),fm.SBP.broadcast()),exclusive=fm.SBPExclusive((0,)))
    elif change=='noncontiguous':out=replace(out,axis_policies=(fm.SBP.broadcast(),fm.SBP.split_block_cyclic((0,1),1)))
    elif change=='subset_axes':out=replace(out,axis_policies=(fm.SBP.broadcast(),fm.SBP.split_contiguous((1,))))
    elif change=='block_levels':out=replace(out,placement=fm.Placement((4,24),'yx','cb'))
    elif change=='padded':out=replace(out,axis_policies=(fm.SBP.broadcast(),fm.SBP.split_contiguous((0,1),granularity=128)))
    else:
        from triton.flagmega.ir.execution_mesh import ExecutionMesh
        mesh=ExecutionMesh.dynamic() if change=='dynamic_mesh' else ExecutionMesh.fixed((4,8)) if change=='wrong_mesh' else ExecutionMesh.fixed((4,24),mapping='column_major')
        m=replace(m,metadata={**m.metadata,'execution_mesh':mesh.to_data()})
    n=replace(n,type=fm.TupleType((out,n.type.fields[1])))
    assert not applicable(n,context(m))


@pytest.mark.parametrize('field,value',[('head_gate_first_value','0'),('head_gate_first_value','local_values + shard_index * 64'),
    ('head_gate_first_value','shard_index * 65'),('global_value','local_values + shard_index * 65'),
    ('global_value','local_values % 64 + shard_index * 64'),('active_values','63'),('local_capacity',65)])
def test_h96_first_value_and_all_active_scalar_maps_rechecked(field,value):
    c=call(hierarchy=(4,24));c[field]=value
    ctx=dict(platform='nvidia',target_architecture='sm120',block_threads=128,warp_size=32,
        mesh_hierarchy=(4,24),launch_resource_limits=dict(max_shared_memory_bytes=101376))
    with pytest.raises(CodegenError):validate_call(c,ctx)


def test_last_h96_cta_elects_one_bounded_leader_and_keeps_unconditional_barrier():
    capacity,bounds=dense_head_geometry((4,24),48,128);first,needed=bounds[-1]
    assert 95*capacity==6080 and first==47 and needed==1
    assert first+1==48  # The old unconditional second leader was out of bounds.
    source=render(call(hierarchy=(4,24)),(4,24))
    assert 'ftt_thread < needed_heads && first_head + ftt_thread < 48LL' in source
    assert 'gate_slot < 0 || gate_slot >= needed_heads' in source
    assert source.count('ftt::arch::cta_sync();')==1
    assert source.index('ftt_head_gates[2 * ftt_thread] = beta;')<source.index('ftt::arch::cta_sync();')<source.index('beta = ftt_head_gates[2 * gate_slot];')


@pytest.mark.parametrize('hierarchy',[(4,8),(4,24)])
def test_public_recipe_survives_explicit_sm120_placement(tmp_path,hierarchy):
    # Use the same installation-before-CLI bootstrap as public compile.py,
    # in a fresh process so the intentionally immutable recipe is isolated.
    import sys
    delivery=Path(__file__).resolve().parents[4]/'tutorials/flagmega/03-qwen3.8-27b-iq4/nvidia-rtx5060ti-cuda'
    code=f'''
import sys,json
sys.path.insert(0,{str(delivery)!r})
from local_optimizations import install_scalar_head_gates
from local_optimizations.scalar_head_gates import ScalarHeadGateProvider
from triton.flagmega.targets import get_target
from triton.flagmega.compiler import Compiler
from triton.flagmega.options import CompileOptions
from triton.flagmega import ir as fm
from triton.flagmega.codegen.triton.candidates.core import TritonCandidateContext
from python.test.flagmega.codegen.cpp.test_scalar_head_gates import graph,IMPLEMENTATION
install_scalar_head_gates(**json.load(open({str(delivery/'local_optimizations/recipes/scalar-head-gates-h2-pair-v1.json')!r})))
original=get_target('nvidia-sm120'); placement=fm.Placement({hierarchy!r},'yx','bb')
compiler=Compiler(CompileOptions(target='nvidia-sm120',placement=placement,codegen_backend='cpp'))
target=compiler.target;m=graph({hierarchy!r})
assert target is not original and target.codegen_backend=='cpp' and getattr(original,'codegen_backend',None) is None
assert target.machine.capability==original.machine.capability
assert target.triton_implementation_model is original.triton_implementation_model
assert target.distributed_placements(m)==(placement,) and original.distributed_placements(m)==(fm.Placement((4,8),'yx','bb'),)
assert {{k:v for k,v in target.options.to_data().items() if k!='placements'}}=={{k:v for k,v in original.options.to_data().items() if k!='placements'}}
for name in ('tir_selection_policy','packing_policy','vectorization_policy','selection_policy','bufferization_policy','tir_lowering_policy','microkernel_selection_policy'):
 assert getattr(target,name) is getattr(original,name),name
provider=target.tir_selection_policy.registry.provider_for('nn.gdn_recurrent_core')
assert isinstance(provider,ScalarHeadGateProvider)
ctx=TritonCandidateContext(m,target,{{}},{{}},{{}},frozenset(),True,target.triton_implementation_model,'cpp')
proposal=provider.propose(m.node_map['recurrent'],ctx)
assert proposal.default_candidate==IMPLEMENTATION
print(json.dumps({{'hierarchy':{hierarchy!r},'backend':target.codegen_backend,'provider':type(provider).__name__,'selected_C2_default':proposal.default_candidate,'new_distribution':target.options.placements[0].to_data(),'local_registry_preserved':True}}))
'''
    result=subprocess.run([sys.executable,'-c',code],capture_output=True,text=True)
    (tmp_path/'public-recipe-compiler.log').write_text(result.stdout+result.stderr)
    assert result.returncode==0,result.stderr
