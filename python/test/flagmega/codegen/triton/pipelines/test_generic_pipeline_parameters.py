# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""An agent-defined pipeline must schedule without a family-specific encoder."""
from copy import deepcopy

from triton.flagmega.codegen.triton.kernel_call_renderers import _bind_transfer_pipeline_parameters
from triton.flagmega.codegen.triton.pipeline_source import _Context, _PipelineSourceBuilder
from triton.flagmega.codegen.triton.tir_package import describe_tir_package
from triton.flagmega.ir import KernelInvoke
from triton.flagmega.ir.tir import iter_tir_children


def test_pipeline_launch_parameters_survive_generic_binding(compile_pipeline_module):
    module = compile_pipeline_module(reusable=False)
    pending = [module.execution_function_map[module.entry].body]
    while pending:
        call = pending.pop()
        if isinstance(call, KernelInvoke) and call.shared_workspace_buffers:
            break
        pending.extend(iter_tir_children(call))
    else:
        raise AssertionError('fixture has no pipeline')
    package = describe_tir_package(module)
    encoded = deepcopy(next(value for value in package['render_calls'] if value['call'] == call.call_id))
    # New family encoders supply geometry; the generic binder owns launch policy.
    for name in ('num_stages', 'producer_warps', 'producer_registers'):
        encoded.pop(name, None)
    _bind_transfer_pipeline_parameters(encoded, encoded)
    builder = _PipelineSourceBuilder(module, {}, {module.entry: [encoded]}, lambda *_: {}, lambda *_: {}, lambda *_: ())
    plan = builder._leaf_stage(_Context(module.entry, {}, {}, {}, ()), call, {'call': call.call_id})
    assert plan['producer_warps'] == encoded['parameters']['producer_warps']
    assert plan['producer_registers'] == encoded['parameters']['producer_registers']
    assert plan['channels'][0]['capacity'] == encoded['parameters']['num_stages']
