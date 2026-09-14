# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Ref bundles must not turn independent state arrays into false hazards."""

from dataclasses import replace

from triton.flagmega import ir as fm
from triton.flagmega.codegen.triton.microkernels.materialization import materialize_shared_workspace_buffers
from triton.flagmega.ir.tir.kernel_definition import replace_kernel_callables, replace_kernel_dispatch
from triton.flagmega.passes.tir import (
    bind_prim_function_buffers, materialize_execution_functions, materialize_kernel_prim_functions,
)
from triton.flagmega.passes.tir.bufferize import plan_memory_synchronization
from triton.flagmega.targets.nvidia import NvidiaSm90Machine

from python.test.flagmega.gdn.recurrent_helpers import recurrent_case


def gdn_state_calls(*, repeated_recurrent=False, distributed=False):
    config, attrs, types, _ = recurrent_case(tokens=1)
    if distributed:
        placement = fm.Placement((2, 4), "yx", "bb")
        types = {name: (value if name == "state" else fm.DistributedType(value, (
            (fm.SBP.broadcast(), fm.SBP.split_contiguous((0, 1))) if name == "z"
            else (fm.SBP.broadcast(),) * value.rank), placement)) for name, value in types.items()}
    builder = fm.IRBuilder(dialect="semantic_tir", stage="selected_tir")
    inputs = {name: builder.var(name, value, id=name) for name, value in types.items()}
    weight_type = fm.tensor_type("bfloat16", (config.conv_dim, 1, 4))
    if distributed:
        weight_type = fm.DistributedType(weight_type, (fm.SBP.broadcast(),) * 3, placement)
    weight = builder.var("conv_weight", weight_type, id="conv_weight")
    calls = []
    for index, family in enumerate(("gdn_recurrent" if repeated_recurrent else "gdn_convolution", "gdn_recurrent")):
        recurrent = family == "gdn_recurrent"
        arguments = tuple(inputs.values()) if recurrent else (inputs["qkv"], inputs["state"], weight)
        result = fm.TupleType((types["z"] if recurrent else types["qkv"], config.ref_type))
        calls.append(builder.call("tir.kernel", arguments, result, id=f"call{index}",
            effect=fm.effect("read_write", "gated_delta_net_state"), attrs={
                "semantic_op": "nn.gdn_recurrent_core" if recurrent else "nn.gdn_convolution",
                "candidate": f"tir.{family}.persistent",
                "parameters": {"family": family, "variant": "persistent"},
                "facts": {}, "semantic_attrs": attrs if recurrent else {"conv_kernel_size": 4},
            }))
    first_value = builder.call("builtin.get_item", (calls[0],), calls[0].type.fields[0],
                               id="first_value", attrs={"index": 0})
    builder.function("main", (*inputs.values(), weight), (first_value, calls[1]))
    return materialize_kernel_prim_functions(builder.build(entry="main"))


def test_independent_gdn_state_fields_have_no_false_hazard():
    module = gdn_state_calls()
    plan = fm.make_buffer_plan(module)
    synchronization = plan_memory_synchronization(module, plan)
    assert synchronization.events == ()


def test_recurrent_updates_still_synchronize_the_recurrent_field():
    module = gdn_state_calls(repeated_recurrent=True)
    plan = fm.make_buffer_plan(module)
    synchronization = plan_memory_synchronization(module, plan)
    event, = synchronization.events
    assert event.scope == "grid"
    recurrent = plan.buffer_map[dict(plan.function_map["main"].values)["state"][1]]
    assert {value.physical_id for value in event.ranges} == {str(recurrent.physical_id)}


def test_field_effects_survive_python_ir_resume():
    module = gdn_state_calls()
    namespace = {}
    exec(fm.module_source(module), namespace)
    assert namespace["MODULE"].semantic_hash == module.semantic_hash


def _scheduled(module):
    plan = fm.make_buffer_plan(module, options=NvidiaSm90Machine().bufferization_options())
    bound = bind_prim_function_buffers(module, plan=plan)
    allocated = replace(bound, stage="allocated_tir", dialect="bufferized_tir",
                        metadata={**bound.metadata, "buffer_plan": plan.to_data()})
    return materialize_execution_functions(allocated), plan


def test_execution_accesses_do_not_reinflate_field_effects():
    module, plan = _scheduled(gdn_state_calls())
    convolution, recurrent = dict(plan.function_map["main"].parameters)["state"]
    first, second = fm.execution_calls_of(module.execution_function_map["main"])
    assert convolution in first.reads and convolution in first.writes
    assert recurrent not in first.reads and recurrent not in first.writes
    assert recurrent in second.reads and recurrent in second.writes
    assert convolution not in second.reads and convolution not in second.writes


def test_state_prefetch_depends_on_only_the_selected_reference_leaf():
    module = gdn_state_calls(distributed=True)
    functions = []
    for function in module.kernel_callable_map.values():
        dispatch = fm.kernel_dispatch_of(function)
        if dispatch.semantic_op == "nn.gdn_recurrent_core":
            partition = fm.T.inplace_transfer_partition(("recurrent",), 3, 0, 1, 4)
            workspace = fm.T.shared_workspace_descriptor("state_stage", fm.tensor_type("float32", (2, 4, 8)), 16)
            pipeline = fm.T.transfer_pipeline_contract((fm.T.transfer_pipeline_channel(
                "state", (0,), (0,), 16, inplace_partition=partition),), capacity=2)
            selection = fm.T.microkernel_selection("test.state", "gdn_recurrent", "test_state",
                                                   shared_workspaces=(workspace,), transfer_pipeline=pipeline)
            dispatch = replace(dispatch, microkernel=selection,
                               shared_workspace_buffers=materialize_shared_workspace_buffers(function, selection))
            function = replace_kernel_dispatch(function, dispatch)
        functions.append(function)
    module, plan = _scheduled(replace_kernel_callables(module, functions))
    _, recurrent = dict(plan.function_map["main"].parameters)["state"]
    _, invocation = fm.execution_calls_of(module.execution_function_map["main"])
    assert invocation.transfer_sources == (recurrent,)
