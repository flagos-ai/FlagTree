# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Lower declared loop value transfers through ordinary scalar-buffer ABIs."""
from triton.flagmega.errors import CodegenError
from triton.flagmega.codegen.triton.kernel_call_renderers import _pointer,_scalar_local_domain,_distributed_unique_writer_active
from triton.flagmega.codegen.triton.physical_access import emit_local_scalar_offset,emit_scalar_immediate


def loop_copy(source,result):
    a,b=source['abi'],result['abi']
    if any(a[key]!=b[key] for key in ('scalar_dtype','scalar_lane_shape','logical_shape','distributed_type')):
        raise CodegenError('Loop value copies require the fixed recurrence dtype/layout/owner interface.')
    domain=_scalar_local_domain(b,'_fm_loop_offsets')
    kind=b.get('distributed_type');placement=kind.get('placement',{}) if kind else {}
    levels=placement.get('hierarchy_levels','')
    replicas=tuple(axis for axis,level in enumerate(levels) if level in {'c','d'})
    result_pointer=_pointer(result)
    result_offset=emit_local_scalar_offset(b,domain['local_coordinates'],lane_coordinate=domain['lane_coordinate'])
    if source.get('runtime_value_kind') in {'scalar','immediate'}:
        value=(emit_scalar_immediate(a,source['runtime_argument']) if source['runtime_value_kind']=='immediate'
               else source['runtime_argument'])
        source_pointer=None;source_offset=None
    else:
        value=None;source_pointer=_pointer(source)
        source_offset=emit_local_scalar_offset(a,domain['local_coordinates'],lane_coordinate=domain['lane_coordinate'])
    return dict(kind='loop_copy',source=source_pointer,source_offset=source_offset,result=result_pointer,
                result_offset=result_offset,capacity=domain['capacity'],active=domain['active'],
                writer_active=_distributed_unique_writer_active(b,allow_redundant_axes=replicas),value=value,tile=256)
