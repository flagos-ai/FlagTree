# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Count physical stores from normal descriptors on a CPU CTA simulator."""
import ast
from copy import deepcopy
import ctypes
from functools import lru_cache
import re
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest
from triton.flagmega import ir as fm
from triton.flagmega.codegen.cpp import describe_cpp_package, render_cpp_source_package
from triton.flagmega.codegen.cpp.build import FTT_INCLUDE_DIR, build_cpp_package, toolchain_available
from triton.flagmega.codegen.cpp.templates import CppTemplateRegistry
from triton.flagmega.codegen.cpp.expressions import cpp_scalar_type
from triton.flagmega.codegen.triton.kernel_call_renderers import prepare_kernel_calls, _canonical_writer_active, _tensor_output_writer_active
from triton.flagmega.codegen.triton.renderer import render_triton_package
from triton.flagmega.compiler import Compiler
from triton.flagmega.evaluator import PagedAttentionStateConfig
from triton.flagmega.options import CompileOptions
from python.test.flagmega.codegen.cpp.test_native_quantized_kernels import packed_rows


@lru_cache(None)
def normal(family):
    class Graph(fm.Module):
        def forward(self):
            if family=='embedding':
                ids=self.input('ids',fm.tensor_type('int32',(2,)))
                weight=self.input('weight',fm.tensor_type('uint8',(3,272)))
                result=fm.F.nn.embedding(ids,weight,weight_encoding='IQ4_XS',output_lanes=(8,),name='embedding')
                self.function('main',(ids,weight),(result,))
            else:
                value=self.input('value',fm.tensor_type('bfloat16',(2,64)))
                state=self.input('state',PagedAttentionStateConfig(1,2,64).ref_type)
                cache=self.input('cache',fm.tensor_type('float32',(3,2,64))) if family=='rotary_cached' else None
                result=fm.F.nn.rotary_embedding(value,state,*((cache,) if cache is not None else ()),head_dim=64,theta=10000.,output_dtype='float32',output_lanes=(16,),name='rotary')
                self.function('main',(value,state,*((cache,) if cache is not None else ())),fm.F.tensors.get_items(result,0,1))
    m=Graph(dialect='high_level',stage='imported',entry='main').build()
    return Compiler(CompileOptions(target='nvidia-sm120',bufferize_opt_level='fast')).compile(m).module


def _call(family):
    d=describe_cpp_package(normal(family))
    return deepcopy(next(c for c in d['render_calls'] if c['family']==('rotary_embedding' if family=='rotary_cached' else family))),tuple(d['mesh_hierarchy'])


def counted_run(tmp_path,call,mesh,*,template_root=None,threads=128):
    """Use scalar cells that count assignments; only FTT identity hooks change."""
    if not shutil.which('c++'):pytest.skip('Host C++ compiler required')
    call=deepcopy(call)
    types={};inputs=[];outputs=[];private={}
    for section in ('inputs','outputs'):
        for parameter in call[section]:
            for binding in parameter['buffers']:
                name=f'p{len(types)}';abi=binding['abi'];binding['runtime_argument']=name
                dtype=abi['scalar_dtype']
                types[name]=('counted_bf16' if dtype=='bfloat16' else 'counted_float') if section=='outputs' else cpp_scalar_type(dtype)
                abi.update(storage='input',pooled=False,pool_byte_offset=0,pool_scope_stride_bytes=0)
                logical=abi['logical_shape'];lanes=abi.get('scalar_lane_count',1)
                capacity=int(np.prod(abi['local_capacity_shape']))*lanes
                if section=='outputs':
                    if abi.get('memory_sharing_scope')=='block':private[name]=int(np.prod(logical))*lanes
                    size=int(np.prod(logical))*lanes
                    if abi['storage_kind']=='compact_per_owner':size=capacity*int(np.prod(mesh))
                    if name in private:size*=int(np.prod(mesh))
                    output=np.zeros(size,dtype=np.dtype([('value','u2'),('pad','u2'),('writes','u4')]) if dtype=='bfloat16' else np.dtype([('value','f4'),('writes','u4')]))
                    outputs.append(output)
                else:
                    if parameter['formal']=='indices':value=np.zeros(logical,np.int32)
                    elif parameter['formal']=='weight' and dtype=='uint8':value=packed_rows(call['semantic_attrs']['weight_encoding'],rows=1,blocks=logical[1]//136)[0]
                    else:value=np.zeros(1 if parameter['formal']=='state' else max(1,int(np.prod(logical))),dtype={'int32':np.int32,'int64':np.int64,'bfloat16':np.uint16,'uint8':np.uint8,'float32':np.float32}[dtype])
                    inputs.append(value)
    call=prepare_kernel_calls([call],function_name='main')[0]
    call.update(symbol='kernel',arguments=list(types))
    context=dict(render_calls=[call],argument_types={n:{'type':t,'pointer':True} for n,t in types.items()},
                 distributed_entry=True,mesh_hierarchy=list(mesh),mesh_rank=len(mesh),block_threads=threads,
                 grid_barrier_axis_groups=[],grid_total_blocks=int(np.prod(mesh)))
    body=CppTemplateRegistry(template_root).render(f"kernels/{call['family']}/_function.cpp.jinja",context)
    # Replace architecture identity queries with simulated CPU coordinates;
    # stores, pointer arithmetic and writer predicates remain as rendered.
    body=body.replace('ftt::arch::thread_id()','test_thread').replace('ftt::arch::block_id()','test_block')
    body=re.sub(r'ftt::arch::mesh_shard_id<(\d+)>\([^)]*\)',r'test_shard[\1]',body)
    parameters=', '.join(f'{t} *{n}' for n,t in types.items())
    arguments=', '.join(f'{n} + test_block * {private[n]}' if n in private else n for n in types)
    source='''#include <cmath>
#include <cstdint>
struct float4 {float x,y,z,w;};
#include <ftt/ftt.h>
extern "C" float __cosf(float x){return std::cos(x);}
extern "C" float __sinf(float x){return std::sin(x);}
struct counted_bf16 {unsigned short value,pad;unsigned writes;counted_bf16& operator=(const counted_bf16 &v){value=v.value;++writes;return *this;}counted_bf16& operator=(const ftt::bfloat16_t &v){value=v.bits;++writes;return *this;}};
struct counted_float {float value;unsigned writes;counted_float& operator=(const counted_float &v){value=v.value;++writes;return *this;}counted_float& operator=(float v){value=v;++writes;return *this;}};
namespace ftt {
inline counted_bf16 store_as(const counted_bf16 *,float v){return {store_as(static_cast<const bfloat16_t *>(nullptr),v).bits,0,0};}
inline counted_float store_as(const counted_float *,float v){return {v,0};}
}
static ftt::dim_t test_thread,test_block,test_shard[3];
static unsigned int _ftt_grid_barrier_counters[2];
'''+body+f'''extern "C" void run({parameters}) {{
for(test_block=0;test_block<{int(np.prod(mesh))};++test_block) {{
'''+''.join(f'test_shard[{i}]=(test_block/{int(np.prod(mesh[i+1:]))})%{extent};\n' for i,extent in enumerate(mesh))+f'''
for(test_thread=0;test_thread<{threads};++test_thread) kernel({arguments});
}}
}}
'''
    path=tmp_path/'counted.cpp';path.write_text(source)
    so=tmp_path/'counted.so'
    subprocess.run(['c++','-std=c++17','-O1','-shared','-fPIC','-DFTT_ARCH_ILUVATAR',f'-I{FTT_INCLUDE_DIR}',str(path),'-o',str(so)],check=True,capture_output=True,text=True)
    lib=ctypes.CDLL(str(so));lib.run(*(a.ctypes.data_as(ctypes.c_void_p) for a in (*inputs,*outputs)))
    return outputs,call,body


@pytest.mark.parametrize('family',['embedding','rotary_embedding','rotary_cached'])
def test_normal_shared_outputs_have_one_physical_store_per_scalar(tmp_path,family):
    call,mesh=_call(family)
    outputs,_,_=counted_run(tmp_path,call,mesh)
    for output in outputs:assert np.all(output['writes']==1)


@pytest.mark.parametrize('family',['embedding','rotary_embedding','rotary_cached'])
def test_private_output_components_are_all_written(tmp_path,family):
    call,mesh=_call(family)
    for p in call['outputs']:
        for b in p['buffers']:b['abi']['memory_sharing_scope']='block'
    outputs,_,_=counted_run(tmp_path,call,mesh)
    for output in outputs:assert np.all(output['writes']==1)


@pytest.mark.parametrize('family',['rotary_embedding','rotary_cached'])
def test_rotary_tuple_elects_each_result_from_its_own_physical_storage(tmp_path,family):
    call,mesh=_call(family)
    call['outputs'][1]['buffers'][0]['abi']['memory_sharing_scope']='block'
    outputs,prepared,_=counted_run(tmp_path,call,mesh)
    assert prepared['cosine_writer_active']!='True'
    assert prepared['sine_writer_active']=='True'
    for output in outputs:assert np.all(output['writes']==1)


def test_canonical_split_does_not_collapse_partitioned_feature_writers(tmp_path):
    call,mesh=_call('embedding')
    abi=call['outputs'][0]['buffers'][0]['abi']
    axis=len(mesh)-1
    policy=fm.SBP.split_contiguous((axis,)).to_data()
    abi['distributed_type']['axis_policies'][1]=policy
    extent=abi['logical_shape'][1];local=extent//mesh[axis]
    abi['local_capacity_shape']=(abi['logical_shape'][0],local)
    abi['active_shape_expressions']=(str(abi['logical_shape'][0]),str(local))
    abi['logical_coordinate_expressions']=('local_coord_0',f'local_coord_1 + shard_coord_{axis} * {local}')
    outputs,prepared,_=counted_run(tmp_path,call,mesh)
    assert 'shard_x == 0' not in prepared['writer_active']
    assert np.all(outputs[0]['writes']==1)


@pytest.mark.parametrize('family',['embedding','rotary_embedding','rotary_cached'])
def test_compact_per_owner_outputs_keep_all_replica_components(tmp_path,family):
    call,mesh=_call(family)
    for p in call['outputs']:
        for b in p['buffers']:
            abi=b['abi'];abi.update(storage_kind='compact_per_owner',coordinate_space='local',owner_index_expression='shard_index',
                                   component_stride_scalar_elements=int(np.prod(abi['local_capacity_shape']))*abi['scalar_lane_count'])
    outputs,_,_=counted_run(tmp_path,call,mesh)
    for output in outputs:assert np.all(output['writes']==1)


def test_plain_shared_canonical_writer_and_partial_private_neighbor():
    abi={'coordinate_space':'canonical_global','storage_kind':'compact_local','distributed_type':None,'memory_sharing_scope':'chip'}
    assert _tensor_output_writer_active(abi)=='(block_index == 0)'
    abi.update(memory_sharing_scope='block',distributed_type={'partial':{'axes':[0],'reduce_op':'sum'}})
    assert _canonical_writer_active(abi)=='True'


@pytest.mark.parametrize('family',['embedding','rotary_embedding'])
def test_normal_sources_preserve_publication_and_compile_without_launch(tmp_path,family):
    module=normal(family)
    d=describe_cpp_package(module)
    render_cpp_source_package(module,tmp_path/'cpp')
    render_triton_package(module,tmp_path/'triton')
    source=(tmp_path/'triton/generated_kernels.py').read_text();ast.parse(source)
    call=next(c for c in d['render_calls'] if c['family']==family)
    predicates=[call['writer_active']] if family=='embedding' else [call['cosine_writer_active'],call['sine_writer_active']]
    for predicate in predicates:assert predicate in source
    if toolchain_available('nvidia-sm120'):
        assert build_cpp_package(tmp_path/'cpp/generated_kernels.cpp',tmp_path/'cpp/kernel.so',target='nvidia-sm120').stat().st_size
