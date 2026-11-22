"""CPU ABI and normal lowering tests; no driver initialization or timings."""
import ast
from copy import deepcopy
import hashlib
import json
from math import prod
from pathlib import Path
import subprocess
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from agent_optimizations.iq4_streaming.native import q8_implementations
from triton.flagmega.codegen.triton import iq4_q8_renderer as q8
from triton.flagmega.codegen.triton.kernel_call_renderers import prepare_kernel_calls
from triton.flagmega.codegen.triton.templates import KernelTemplateSpec, TritonTemplateRegistry
from triton.flagmega.errors import CodegenError


def binding(name, dtype, shape, *, lanes=(), offset=None, storage='external'):
    size = {'uint8': 1, 'bfloat16': 2, 'float32': 4}[dtype]
    alignment = 4 if dtype != 'bfloat16' else 2
    return dict(actual=name, runtime_argument=name, abi=dict(
        scalar_dtype=dtype, scalar_itemsize=size, scalar_lane_shape=lanes,
        scalar_lane_count=prod(lanes), logical_shape=shape, local_capacity_shape=shape,
        scalar_storage_strides=tuple(prod(shape[i+1:])*prod(lanes) for i in range(len(shape))),
        active_shape_expressions=tuple(map(str, shape)), alignment_bytes=alignment,
        allocation_alignment_bytes=alignment, subspan_byte_offset=0,
        span_capacity_bytes=prod(shape)*prod(lanes)*size,
        pool_byte_offset=alignment if offset is None else offset, pool_scope_stride_bytes=0,
        physical_buffer=name+'_allocation', owner_count=1,
        coordinate_space='canonical_global', storage=storage))


def raw_call(family, *, k=5120, n=None, dtype='bfloat16', lanes=()):
    n = n or (12288 if family == q8.FAMILIES[3] else 17408 if family == q8.FAMILIES[2] else 6144)
    g = (k+31)//32
    attrs = dict(graph_schema=q8.GRAPH_SCHEMA, plan_hash=q8.PLAN_SHA256,
        numerical_profile='H003-N-v2', numerical_profile_hash=q8.NUMERICAL_SUBJECT_SHA256,
        native_interface=q8.NATIVE_INTERFACE, q8_abi=q8.Q8_ABI,
        logical_k=k, padded_k=g*32, group_size=32, group_count=g, record_bytes=36,
        source_dtype='bfloat16', source_version_key='main:ready:step')
    qb = binding('Q', 'uint8', (1,g,36), storage='workspace')
    if family == q8.FAMILIES[0]:
        inputs = [('value', binding('X', 'bfloat16', (1,k)))]
        results = [qb]
    else:
        attrs.update(weight_block_n=128, weight_block_k=128, output_data_type=dtype,
                     output_lanes=lanes, logical_n=n)
        inputs = [('q8',qb)]
        formals = ('gate_weight','up_weight') if family == q8.FAMILIES[2] else ('weight',)
        for i,f in enumerate(formals):
            attrs[f+'_encoding']='IQ4_XS'
            inputs.extend([(f,binding('W'+str(i),'uint8',(n,k//256*136))),
                           (f+'_scale',binding('S'+str(i),'float32',((n+127)//128,(k+127)//128)))])
        if family == q8.FAMILIES[2]: attrs['activation']='silu'
        if family == q8.FAMILIES[3]:
            attrs.update(feature_shape=(24,256),field_count=2,output_lanes=((8,),(8,)))
            results=[binding('Y','bfloat16',(1,24,32),lanes=(8,)),
                     binding('Z','bfloat16',(1,24,32),lanes=(8,))]
        else: results=[binding('Y',dtype,(1,n//prod(lanes)),lanes=lanes)]
    return dict(call='native_q8', family=family, variant=q8.VARIANT, execution_kind='local_shard',
        parameters=q8.implementation_parameters(family), semantic_attrs=attrs,
        facts=dict(internal_grid_barriers=0,required_codegen_backend='triton'),
        inputs=[dict(formal=f,buffers=[b]) for f,b in inputs],
        outputs=([dict(formal='result_'+str(i),buffers=[b]) for i,b in enumerate(results)]
                 if family==q8.FAMILIES[3] else [dict(formal='result',buffers=results)]),workspaces=[])


@pytest.mark.parametrize('family',q8.FAMILIES)
def test_encoder_json_roundtrip_and_jit_lowering(family):
    raw=raw_call(family)
    q8.register();q8.register()
    call=prepare_kernel_calls([raw],function_name='main')[0]
    decoded=json.loads(json.dumps(raw))
    same=prepare_kernel_calls([decoded],function_name='main')[0]
    assert json.loads(json.dumps(call)) == json.loads(json.dumps(same))
    descriptor=q8.materialize(raw)
    assert descriptor['jit_entry']==('q8_prepare' if family==q8.FAMILIES[0] else 'dp4a_m1')
    assert descriptor['compile_options']['num_warps']==4
    assert descriptor['compile_options']['maxnreg']==dict(zip(q8.FAMILIES,(80,80,104,96)))[family]
    assert descriptor['compile_options']['ptx_options']=='--minnctapersm=1'
    assert descriptor['compile_options']['enable_static_shared']==(family!=q8.FAMILIES[0])
    assert descriptor['program_kind']=='ordinary' and descriptor['dynamic_shared_bytes']==0
    source=TritonTemplateRegistry().render_kernel(
        KernelTemplateSpec(family,q8.VARIANT,'nvidia','sm120'),{'render_calls':[call]}).source
    tree=ast.parse(source)
    invocation=next(n for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name)
                    and n.func.id.startswith('_r003_'))
    assert {kw.arg:ast.literal_eval(kw.value) for kw in invocation.keywords}==descriptor['constexprs']
    assert not any(isinstance(n,ast.Name) and n.id.startswith('S') for n in ast.walk(invocation))
    assert all(not b['device_read'] for b in descriptor['placeholder_bindings'])
    assert all('tl.full((),' in ptr for ptr in descriptor['pointers'])


@pytest.mark.parametrize('family',q8.FAMILIES)
@pytest.mark.parametrize('fault',('dtype','k','kp','g','record_stride','alignment','base','pool','subspan',
                                 'capacity','version','profile','options','geometry','missing_attrs'))
def test_validate_raw_rejects(family,fault):
    raw=raw_call(family);a=raw['semantic_attrs'];p=raw['parameters']
    qb=(raw['outputs'][0] if family==q8.FAMILIES[0] else raw['inputs'][0])['buffers'][0]['abi']
    if fault=='dtype':qb['scalar_dtype']='bfloat16'
    elif fault=='k':a['logical_k']=6145
    elif fault=='kp':a['padded_k']+=32
    elif fault=='g':a['group_count']+=1
    elif fault=='record_stride':qb['scalar_storage_strides']=(a['group_count']*40,40,1)
    elif fault=='alignment':qb['alignment_bytes']=2
    elif fault=='base':qb['allocation_alignment_bytes']=2
    elif fault=='pool':qb['pool_byte_offset']=2
    elif fault=='subspan':qb['subspan_byte_offset']=2
    elif fault=='capacity':qb['span_capacity_bytes']+=4
    elif fault=='version':del a['source_version_key']
    elif fault=='profile':a['numerical_profile']='H003-N-v1'
    elif fault=='options':p['enable_static_shared']=not p['enable_static_shared']
    elif fault=='geometry':p['num_warps']=8
    elif fault=='missing_attrs':raw.pop('semantic_attrs')
    with pytest.raises(CodegenError):q8.validate_raw(raw)


@pytest.mark.parametrize('family',q8.FAMILIES)
def test_all_mandatory_attrs_parameters_rejected_if_missing(family):
    raw=raw_call(family)
    for side in ('parameters','semantic_attrs'):
        for name in raw[side]:
            bad=deepcopy(raw);bad[side].pop(name)
            with pytest.raises(CodegenError):q8.validate_raw(bad)


def test_catalog_is_four_frozen_versioned_families():
    catalog=q8_implementations()
    assert tuple(i.family for i in catalog)==q8.FAMILIES
    for i in catalog:assert dict(i.parameters)==q8.implementation_parameters(i.family)
    for kw in ({'row_tile':4},{'num_warps':8},{'row_tile':True}):
        with pytest.raises(CodegenError):q8_implementations(**kw)


def test_registration_visible_in_normal_bootstrap_without_local_register():
    # A fresh interpreter cannot inherit test-local register() calls.
    code='''from test_iq4_q8_native_r003 import raw_call, q8, prepare_kernel_calls
from triton.flagmega.codegen.triton.kernel_call_renderers import _FAMILY_ENCODERS
assert not set(q8.FAMILIES).intersection(_FAMILY_ENCODERS)
for f in q8.FAMILIES:
 c=prepare_kernel_calls([raw_call(f)],function_name='main')[0]
 assert c['schema']==q8.SCHEMA
assert set(q8.FAMILIES).issubset(_FAMILY_ENCODERS)
'''
    result=subprocess.run([sys.executable,'-c',code],cwd=Path(__file__).parent,capture_output=True,text=True)
    assert result.returncode==0,result.stdout+result.stderr


@pytest.mark.parametrize('k',(1,31,33,129,5120,6144,17408))
def test_prepare_padding_and_grid(k):
    c=q8.materialize(raw_call(q8.FAMILIES[0],k=k))
    assert c['grid']==[((k+31)//32+3)//4,1,1]


def test_f32_source_head_n_tail_and_bf16_packets():
    raw=raw_call(q8.FAMILIES[0]);raw['semantic_attrs']['source_dtype']='float32'
    raw['inputs'][0]['buffers'][0]=binding('X','float32',(1,5120))
    q8.validate_raw(raw)
    for k in (5120,6144,17408):
        q8.validate_raw(raw_call(q8.FAMILIES[1],k=k,n=7))
    q8.validate_raw(raw_call(q8.FAMILIES[1],n=248320,dtype='float32'))
    q8.validate_raw(raw_call(q8.FAMILIES[1],lanes=(8,)))


def test_glu_order_scale_and_interleaved_all_scalar_components():
    raw=raw_call(q8.FAMILIES[2]);raw['inputs'][1],raw['inputs'][3]=raw['inputs'][3],raw['inputs'][1]
    with pytest.raises(CodegenError,match='order'):q8.validate_raw(raw)
    raw=raw_call(q8.FAMILIES[1]);raw['inputs'][2]['buffers'][0]['abi']['scalar_dtype']='uint8'
    with pytest.raises(CodegenError,match='dtype'):q8.validate_raw(raw)
    raw=raw_call(q8.FAMILIES[3]);out=[a['buffers'][0] for a in raw['outputs']]
    out[1]['abi']['physical_buffer']=out[0]['abi']['physical_buffer']
    with pytest.raises(CodegenError,match='overlapping'):q8.validate_raw(raw)
    out[1]['abi']['pool_byte_offset']+=12288
    q8.validate_raw(raw) # Disjoint fields in one pooled allocation are valid.
    counts=[[0]*6144 for _ in range(2)]
    for row in range(12288):counts[row%512//256][row//512*256+row%256]+=1
    assert all(count==1 for field in counts for count in field)
    raw['parameters']['row_origin']=256
    with pytest.raises(CodegenError,match='origin'):q8.validate_raw(raw)


def test_accepted_source_pin():
    import importlib.util
    source=Path(importlib.util.find_spec(q8.JIT_MODULE).origin)
    assert hashlib.sha256(source.read_bytes()).hexdigest()==q8.JIT_SOURCE_SHA256


@pytest.mark.parametrize('family',q8.FAMILIES[1:])
def test_semantic_op_to_normal_renderer(family,tmp_path):
    """Real OpDefinitions → target/provider → buffer ABI → owning templates."""
    from agent_optimizations.iq4_streaming.graph import configure_q8_target
    from triton.flagmega import ir as fm
    from triton.flagmega.compiler import Compiler
    from triton.flagmega.options import CompileOptions
    from triton.flagmega.codegen.triton.renderer import render_triton_package
    from triton.flagmega.codegen.triton.tir_package import describe_tir_package
    class Graph(fm.Module):
        def forward(self):
            n=12288 if family==q8.FAMILIES[3] else 17408 if family==q8.FAMILIES[2] else 6144
            x=self.input('x',fm.tensor_type('bfloat16',(1,5120)))
            w=self.input('w',fm.tensor_type('uint8',(n,2720)))
            s=self.input('scale',fm.tensor_type('float32',((n+127)//128,40)))
            attrs=dict(weight_encoding='IQ4_XS',weight_block_n=128,weight_block_k=128)
            if family==q8.FAMILIES[3]:
                y=fm.F.nn.interleaved_projection(x,w,s,feature_shape=(24,256),output_lanes=((8,),(8,)),**attrs)
                outputs=fm.F.tensors.get_items(y,0,1)
            elif family==q8.FAMILIES[2]:
                attrs.pop('weight_encoding')
                y=fm.F.nn.matmul_glu(x,w,w,s,s,activation='silu',gate_weight_encoding='IQ4_XS',up_weight_encoding='IQ4_XS',**attrs)
                outputs=(y,)
            else:
                y=fm.F.math.block_scaled_matmul(x,w,s,**attrs);outputs=(y,)
            self.function('main',(x,w,s),outputs)
    compiler=Compiler(CompileOptions(target='nvidia-sm120',codegen_backend='triton'))
    compiler.target=configure_q8_target(compiler.target)
    module=compiler.compile(Graph(dialect='high_level',stage='imported',entry='main').build()).module
    calls=describe_tir_package(module)['render_calls']
    assert {c['family'] for c in calls}=={q8.FAMILIES[0],family}
    for c in calls:
        assert c['schema']==q8.SCHEMA
        assert c['jit_entry']==('q8_prepare' if c['family']==q8.FAMILIES[0] else 'dp4a_m1')
        assert c['compile_options']==q8.materialize(c)['compile_options']
    package=render_triton_package(module,tmp_path/'generated')
    for c in calls:
        spec=next(k for k in package['kernels'] if c['call'] in k['calls'])
        assert spec['compile_options']==c['compile_options']
        assert spec['num_warps']==4 and spec['maxnreg']==c['compile_options']['maxnreg']
        assert spec['enable_static_shared']==c['compile_options']['enable_static_shared']
        assert spec['program_kind']=='ordinary'
        assert spec['grid']==c['grid']
    fm.emit_module(module,tmp_path/'bufferized.py')
    (tmp_path/'package.json').write_text(json.dumps(package,indent=2,allow_nan=False)+'\n')
    source=(tmp_path/'generated/generated_kernels.py').read_text()
    ast.parse(source)
    assert '_r003_'+family in source and '_r003_q8_1_prepare' in source


@pytest.mark.parametrize('family',q8.FAMILIES[1:])
def test_metadata_placeholder_identity_has_no_device_argument(family):
    from triton.flagmega.ir import tensor_type
    raw=raw_call(family)
    for arg in raw['inputs']:
        if arg['formal'].endswith('_scale'):
            b=arg['buffers'][0]
            arg.update(role='metadata',actual_value=b['actual'],buffers=[],
                       type=tensor_type('float32',b['abi']['logical_shape']).to_data())
    q8.register()
    call=prepare_kernel_calls([raw],function_name='main')[0]
    assert not any(a.startswith('S') for a in call['arguments'])
    assert len(call['placeholder_bindings'])==(2 if family==q8.FAMILIES[2] else 1)
    assert all(not b['device_read'] for b in call['placeholder_bindings'])
    bad=deepcopy(raw)
    next(a for a in bad['inputs'] if a['formal'].endswith('_scale'))['type']=tensor_type('bfloat16',(1,1)).to_data()
    with pytest.raises(CodegenError,match='dtype/shape'):q8.validate_raw(bad)
