# Copyright 2026 FlagOS Contributors; SPDX-License-Identifier: MIT
from pathlib import Path
import pytest
from triton.flagmega.codegen.triton import iq4_q8_consumer_prefetch_renderer as renderer
from triton.flagmega.errors import CodegenError, IRSchemaError
from triton.flagmega.ir.launch_recipe import TaskLaunchRecipe

@pytest.mark.parametrize('config', [(4,1,40,432),(4,2,56,2176),(8,8,64,1088)])
def test_recipe_roundtrip_and_reject_tampering(config):
    data = renderer.recipe(config=config).to_data()
    assert TaskLaunchRecipe.from_data(data).to_data() == data
    for key,value in [('producer_warps',1),('weight_staging',True),('token_rows',1),
                      ('partial_sync','none'),('prefetch_distance',3)]:
        with pytest.raises(IRSchemaError):
            TaskLaunchRecipe.from_data({**data,key:value})
    with pytest.raises(CodegenError):
        renderer.checked_config((config[0],3,*config[2:]))

def test_preserves_shell_packet_reuse_and_arithmetic():
    base = Path(renderer.__file__).parent/'kernels/iq4_q8_glu'
    old = (base/'r038_glu_global_prefetch_v1.py.jinja').read_text()
    new = (base/'r049_glu_consumer_prefetch_v1.py.jinja').read_text()
    begin = '@triton.jit\ndef {{ call.symbol }}__accumulate_pair'
    end = '@triton.jit\ndef {{ call.symbol }}__store_rows'
    assert old[old.index(begin):old.index(end)] == new[new.index(begin):new.index(end)]
    for excluded in ('__producer','_prefetch_progress','nanosleep','warp_specialize'):
        assert excluded not in new
    assert 'prefetch.global.L2' in new
