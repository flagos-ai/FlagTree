# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Native final-axis packets still execute every scalar output row."""
import pytest
from triton.flagmega.codegen.triton.kernel_call_renderers import prepare_kernel_calls
from triton.flagmega.codegen.triton.templates import TritonTemplateRegistry
from python.test.flagmega.codegen.triton.kernels.block_fp8.test_output_dtype_abi import raw_projection


@pytest.mark.parametrize('encoding',['IQ4_XS','Q4_K'])
@pytest.mark.parametrize('lanes',[(4,),(2,4)])
def test_native_template_uses_descriptor_scalar_mask_for_all_lanes(encoding,lanes):
    call,=prepare_kernel_calls((raw_projection(encoding,lanes),),function_name='main')
    call.update(symbol='native_projection',signature='value, weight, result')
    source=TritonTemplateRegistry().render('kernels/block_fp8/_function.py.jinja',dict(
        render_calls=[call],block_fp8_variant='simt',mesh_hierarchy=[1],distributed_entry=False))
    assert f"n_mask = {call['n_mask']}" in source
    assert f"n_mask = local_n_offsets < ({call['active_n']})" not in source
    assert call['local_n_capacity']==16
