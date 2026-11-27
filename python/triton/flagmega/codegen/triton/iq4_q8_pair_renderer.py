# Copyright 2026 FlagOS Contributors; SPDX-License-Identifier: MIT
"""Static long-K pair-load schedule with the original 80-register upper bound."""
from dataclasses import replace
from . import iq4_q8_renderer as accepted
from triton.flagmega.errors import CodegenError
from triton.flagmega.ir.launch_recipe import TaskLaunchRecipe
VARIANT='r040_down_pair_columns_v1'

def implementation(base):
    if base.family!='iq4_q8_projection' or base.variant!=accepted.VARIANT:
        raise CodegenError('R040 down pair requires the accepted projection')
    return replace(base,id='tir.iq4_q8_projection.'+VARIANT,variant=VARIANT,
                   contract={**base.contract,'mapping_identity':VARIANT})

def old_view(raw):return {**raw,'variant':accepted.VARIANT}

def validate_raw(raw):
    if raw.get('variant')!=VARIANT or raw.get('family')!='iq4_q8_projection':
        raise CodegenError('wrong R040 down pair subject')
    accepted.validate_raw(old_view(raw))
    a=raw['semantic_attrs'];rows=accepted._shape(accepted._bindings(raw,'inputs','q8')[0]['abi'],'logical_shape')[0]
    if (a['logical_n'],a['logical_k'],rows)!=(5120,17408,5):
        raise CodegenError('R040 down pair requires the measured N5120 K17408 T5 bucket')
    if accepted._canonical(raw['parameters']['task_mapping'])!=accepted._canonical(TaskLaunchRecipe(5120,2,'ordinary').to_data()):
        raise CodegenError('R040 down pair task mapping differs from two independent columns')

def materialize(raw):
    validate_raw(raw);result=accepted.materialize(old_view(raw))
    return {**result,'jit_module':'triton.flagmega.codegen.triton.iq4_q8_pair_columns',
            'jit_entry':'dp4a_pair_columns','constexprs':{**result['constexprs'],'COLS':2},
            'grid':[2560,1,1]}
