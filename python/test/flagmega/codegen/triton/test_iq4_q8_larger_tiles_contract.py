# Copyright 2026 FlagOS Contributors; SPDX-License-Identifier: MIT
from pathlib import Path
import pytest
from triton.flagmega.codegen.triton import iq4_q8_larger_tiles_renderer as renderer
from triton.flagmega.errors import CodegenError, IRSchemaError
from triton.flagmega.ir.launch_recipe import TaskLaunchRecipe

@pytest.mark.parametrize('config', [(2,2,4,56,432),(4,3,2,48,144),(8,2,8,64,2176),(16,1,4,64,432)])
def test_tile_recipe_rejects_payload_and_synchronization_tampering(config):
    data=renderer.recipe(config=config).to_data()
    assert TaskLaunchRecipe.from_data(data).to_data()==data
    for key,value in [('columns',3),('stage_bytes',1),('token_rows',5),('partial_sync','none')]:
        with pytest.raises(IRSchemaError):
            TaskLaunchRecipe.from_data({**data,key:value})
    with pytest.raises(CodegenError):
        renderer.checked_config((config[0],0,*config[2:]))

def test_arithmetic_helpers_are_byte_identical_to_r047():
    base=Path(renderer.__file__).parent/'kernels/iq4_q8_glu'
    old=(base/'r047_glu_occupancy_v1.py.jinja').read_text()
    new=(base/'r048_glu_larger_tiles_v1.py.jinja').read_text()
    # Exclude only the CTA layout declaration; every arithmetic helper is literal.
    start='@triton.jit\ndef packets_cached_tables'
    end='@triton.jit\ndef {{ call.symbol }}__producer'
    assert old[old.index(start):old.index(end)]==new[new.index(start):new.index(end)]
