"""Normal function-pool regrouping must retain O-proj ownership and reject bad bindings."""
from pathlib import Path
from types import SimpleNamespace
import copy,json,sys
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from local_optimizations import oproj
from triton.flagmega.errors import CodegenError

def fixture(monkeypatch,tmp_path,*,dense_width=16):
    monkeypatch.setattr(oproj,'OUT',tmp_path)
    (tmp_path/'plan.json').write_text(json.dumps({'semantic_hash':'fixture'}))
    # describe also records its owned source/template inputs.
    for name in ('oproj.py','oproj-head.py.jinja','oproj-tail.py.jinja'):
        (tmp_path/name).write_text('fixture')
    functions=[('decode_linear_attention_dense',3,dense_width),
               ('decode_linear_attention_sparse.prefix',31,16),
               ('decode_mla_attention_sparse.prefix',11,18)]
    kernels=[dict(symbol=f'actor_{fn}_{i}',entry_function=fn) for fn,count,width in functions for i in range(count)]
    d=dict(kernels=kernels,programs=[],device_functions=[],render_calls=[],
           persistent_window_resources=dict(resources=[dict(name='ordered_oproj',programs=[k['symbol'] for k in kernels],nbytes=9216,offset=0)]),
           implementation_override=dict(inputs={}))
    for index,(fn,count,width) in enumerate(functions):
        op=dict(kind='tir.kernel_call',family='partitioned_linear',variant='wgmma_local',call='o_proj_weight_apply',
                symbol=f'op_{index}',arguments='workspace, weights, normalized, value, output')
        post=dict(kind='tir.kernel_call',family='mhc_post',symbol=f'post_{index}',arguments='workspace, residual, coefficient, result')
        region=[op,dict(kind='barrier',scope='device'),dict(kind='tir.kernel_call',family='distributed_boxing'),
                dict(kind='barrier',scope='grid'),post,dict(kind='barrier',scope='grid')]
        fields=[dict(formal=f'field_{col}',storage='workspace',scalar_dtype='float32',
                     alignment_bytes=256,record_column=col,byte_offset=0) for col in range(width-1)]
        fields.append(dict(formal='workspace',storage='workspace',scalar_dtype='uint8',alignment_bytes=256,record_column=width-1,byte_offset=0))
        fields.append(dict(formal='rdata',storage='rdata',scalar_dtype='uint8',alignment_bytes=256,byte_offset=0))
        abi=dict(schema='flagmega.function-pool-abi/v2',bindings=fields,
                 pools=[dict(storage=storage,parameter='_flagmega_call_pool_'+storage) for storage in ('rdata','workspace')],
                 record_width=width,calls=[dict(caller='main',call=f'call_{i}',record_offset=i*width) for i in range(count)])
        d['device_functions'].append(dict(function=fn,function_pool_abi=abi,schedule=dict(consumer_events=region)))
        d['render_calls'].extend([
            dict(symbol=op['symbol'],signature='workspace, weights, normalized, value, output',tasks=64,rows=4096,
                 accumulation_splits=2 if fn.startswith('decode_mla') else 1),
            dict(symbol=post['symbol'],hc=4,value_first_fma=True,residual_input_type='tl.bfloat16')])
    return d

def test_unrelated_valid_record_regrouping_preserves_oproj(monkeypatch,tmp_path):
    descriptor=fixture(monkeypatch,tmp_path)
    changed=oproj.describe(SimpleNamespace(semantic_hash='fixture'),lambda module:copy.deepcopy(descriptor))
    assert len(changed['owner_matched_oproj'])==3
    assert all(len(row['numerical_region'])==6 for row in changed['owner_matched_oproj'].values())
    assert [row['function_pool_abi']['record_width'] for row in changed['device_functions']]==[16,16,18]

def test_inconsistent_affine_record_domain_is_rejected(monkeypatch,tmp_path):
    descriptor=fixture(monkeypatch,tmp_path,dense_width=15)
    abi=descriptor['device_functions'][0]['function_pool_abi']
    abi['bindings'][0]['record_affine']=dict(record_start=0,record_width=14,base=0,stride=256,records=3)
    with pytest.raises(CodegenError):
        oproj.describe(SimpleNamespace(semantic_hash='fixture'),lambda module:copy.deepcopy(descriptor))

def test_invalid_workspace_frame_dtype_is_rejected(monkeypatch,tmp_path):
    descriptor=fixture(monkeypatch,tmp_path,dense_width=15)
    abi=descriptor['device_functions'][0]['function_pool_abi']
    next(field for field in abi['bindings'] if field['formal']=='workspace')['scalar_dtype']='float32'
    with pytest.raises(CodegenError):
        oproj.describe(SimpleNamespace(semantic_hash='fixture'),lambda module:copy.deepcopy(descriptor))
