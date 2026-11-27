# Copyright 2026 FlagOS Contributors; SPDX-License-Identifier: MIT
from pathlib import Path
import pytest
from triton.flagmega.codegen.triton import iq4_q8_projection_consumer_renderer as renderer
from triton.flagmega.errors import CodegenError, IRSchemaError
from triton.flagmega.ir.launch_recipe import TaskLaunchRecipe

@pytest.mark.parametrize('qualification', renderer.SHAPES)
@pytest.mark.parametrize('config_prefix', [(4,1,40),(4,2,64),(8,4,56)])
def test_projection_recipe_roundtrip_and_tampering(qualification,config_prefix):
    n,k=renderer.SHAPES[qualification]
    config=(*config_prefix,n//4)
    data=renderer.recipe(2,qualification,config).to_data()
    assert TaskLaunchRecipe.from_data(data).to_data()==data
    for key,value in [('producer_warps',1),('weight_staging',True),('token_rows',1),
                      ('partial_sync','none'),('prefetch_distance',3),('work_size',n+1),
                      ('cta_warps',True),('qualification','logits')]:
        with pytest.raises(IRSchemaError): TaskLaunchRecipe.from_data({**data,key:value})
    with pytest.raises(CodegenError): renderer.checked_config(qualification,(4,3,64,n//4))
    with pytest.raises(IRSchemaError): renderer.recipe(1,qualification,config)

def test_single_projection_arithmetic_is_preserved():
    base=Path(renderer.__file__).parent
    new=(base/'kernels/iq4_q8_projection/r050_projection_consumer_prefetch_v1.py.jinja').read_text()
    helper=(base/'iq4_q8_prefetch_rows.py').read_text()
    body=helper[helper.index('@tr.jit\ndef accumulate_prefetched_group'):].replace('@tr.jit','@triton.jit')
    assert body in new
    r49=(base/'kernels/iq4_q8_glu/r049_glu_consumer_prefetch_v1.py.jinja').read_text()
    begin='@triton.jit\ndef packets_cached_tables'
    end='@triton.jit\ndef {{ call.symbol }}__accumulate_pair'
    assert r49[r49.index(begin):r49.index(end)] in new
    for excluded in ('__producer','_prefetch_progress','nanosleep','warp_specialize','cp.async','os.environ'):
        assert excluded not in new
    assert 'prefetch.global.L2' in new
    assert renderer.qualify('iq4_q8_projection',248320,5120) is None

@pytest.mark.parametrize('qualification,columns,grid', [('mlp_down',4,160),('in_proj_qkv',2,640),('out_proj',2,320)])
def test_existing_column_recipe(qualification,columns,grid):
    data=renderer.recipe(2,qualification,(4,1,64,grid),columns).to_data()
    assert data['tile_size']==columns
    assert TaskLaunchRecipe.from_data(data).to_data()==data
    for key,value in [('columns',True),('columns',0),('tile_size',1)]:
        with pytest.raises(IRSchemaError): TaskLaunchRecipe.from_data({**data,key:value})

def test_preserves_existing_vector_arithmetic():
    base=Path(renderer.__file__).parent
    old=(base/'iq4_q8_multi_column.py').read_text()
    new=(base/'kernels/iq4_q8_projection/r050_existing_columns.py.jinja').read_text()
    begin='@tr.jit\ndef vector_packets'
    end='@tr.jit\ndef _dp4a_columns_impl'
    assert old[old.index(begin):old.index(end)].replace('@tr.jit','@triton.jit') in new
    for excluded in ('__producer','_prefetch_progress','nanosleep','warp_specialize','cp.async','os.environ'):
        assert excluded not in new
