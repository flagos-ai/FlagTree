# Copyright 2026 FlagOS Contributors; SPDX-License-Identifier: MIT
"""Versioned static token/group launch; preserves the accepted Q8 byte ABI."""
from dataclasses import replace
from . import iq4_q8_renderer as accepted
from triton.flagmega.errors import CodegenError
from triton.flagmega.ir.launch_recipe import TaskLaunchRecipe

VARIANT='r040_q8_prepare_rows_v1'

def implementation(base):
    if base.family!='q8_1_prepare' or base.variant!=accepted.VARIANT:
        raise CodegenError('R040 prepare requires the accepted prepare implementation')
    return replace(base,id='tir.q8_1_prepare.'+VARIANT,variant=VARIANT,
                   contract={**base.contract,'mapping_identity':VARIANT})

def old_view(raw):
    return {**raw,'variant':accepted.VARIANT}

def validate_raw(raw):
    if raw.get('variant')!=VARIANT or raw.get('family')!='q8_1_prepare':
        raise CodegenError('wrong R040 Q8 prepare subject')
    accepted.validate_raw(old_view(raw))
    rows=accepted._shape(accepted._results(raw)[0]['abi'],'logical_shape')[0]
    k=raw['semantic_attrs']['logical_k']
    if not 1<rows<=5 or k%128:
        raise CodegenError('R040 Q8 prepare requires 2..5 complete K128 token rows')
    mapping=raw['parameters']['task_mapping']
    expected=TaskLaunchRecipe(rows*(k//32),4,'ordinary').to_data()
    if accepted._canonical(mapping)!=accepted._canonical(expected):
        raise CodegenError('R040 Q8 prepare requires an independent owner for every token/group')

def materialize(raw):
    validate_raw(raw)
    result=accepted.materialize(old_view(raw))
    k=result['constexprs']['K'];rows=result['constexprs']['ROWS']
    return {**result,'jit_module':'triton.flagmega.codegen.triton.iq4_q8_prepare_rows',
            'jit_entry':'q8_prepare_rows','grid':[rows*(k//128),1,1]}
