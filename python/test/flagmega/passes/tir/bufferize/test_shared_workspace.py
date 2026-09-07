# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT

from dataclasses import replace

import pytest

from triton.flagmega import ir as fm
from triton.flagmega.codegen.triton.microkernels.materialization import (
    materialize_shared_workspace_buffers,
)
from triton.flagmega.errors import IRVerificationError
from triton.flagmega.ir.bufferization import (
    AllocationPolicy,
    AllocationStrategy,
    MemorySpace,
)
from triton.flagmega.ir import kernel_dispatch_of
from triton.flagmega.passes.tir.bufferize import (
    BufferizationOptions,
    NttBufferizationPolicy,
)

def _selected_module(*, source_alignment=16):
    input_type = fm.tensor_type("bfloat16", (1, 2048))
    weight_type = fm.tensor_type("bfloat16", (2048, 4096))
    output_type = fm.tensor_type("bfloat16", (1, 4096))
    workspaces = (
        fm.T.shared_workspace_descriptor(
            "rhs_stage", fm.tensor_type("bfloat16", (2, 64, 16)), 16
        ),
        fm.T.shared_workspace_descriptor(
            "accumulator", fm.tensor_type("float32", (64,)), 32
        ),
    )
    selection = fm.T.microkernel_selection(
        implementation="test.qkv.workspace_pipeline",
        family="qkv_parallel_linear",
        variant="workspace_pipeline",
        parameters={"block_n": 64, "block_k": 128, "stages": 2},
        requires=("async_matrix",),
        shared_workspaces=workspaces,
        transfer_pipeline=fm.T.transfer_pipeline_contract(
            (fm.T.transfer_pipeline_channel(
                "rhs",
                source_argument_indices=(1,),
                shared_workspace_indices=(0,),
                source_alignment_bytes=source_alignment,
            ),),
            consumer_shared_workspace_indices=(1,),
        ),
    )
    dispatch = fm.T.kernel_dispatch(
        semantic_op="ntt.packed_qkv_parallel_linear_fused_rhs",
        semantic_candidate="ntt.packed_qkv_parallel_linear",
        arguments=("input", "fused_weight"),
        outputs=("output",),
        semantic_attrs={
            "rhs_layout": "k_major",
            "projection_n_capacities": (2048, 1024, 1024),
        },
        microkernel=selection,
        reads=("input", "fused_weight"),
        writes=("output",),
    )
    parameters = (
        fm.T.prim_parameter("input", input_type, fm.T.PrimParameterRole.INPUT),
        fm.T.prim_parameter(
            "fused_weight", weight_type, fm.T.PrimParameterRole.INPUT
        ),
        fm.T.prim_parameter("output", output_type, fm.T.PrimParameterRole.OUTPUT),
    )
    function = fm.T.prim_function(
        "packed_qkv",
        "triton",
        parameters,
        fm.T.sequential((dispatch,)),
        fm.T.return_((
            fm.T.return_binding(fm.T.value_ref("output", output_type), "output"),
        )),
    )
    dispatch = replace(
        dispatch,
        shared_workspace_buffers=materialize_shared_workspace_buffers(
            function, selection
        ),
    )
    function = replace(function, body=fm.T.sequential((dispatch,)))
    builder = fm.IRBuilder(dialect="semantic_tir", stage="selected_microkernels")
    source = builder.var("source", input_type, id="source")
    weight = builder.weight(
        "fused_weight", weight_type,
        source="weights.safetensors", key="qkv", id="fused_weight",
    )
    builder.prim_function(function)
    result = builder.call(
        "tir.call", (source, weight), output_type,
        id="result", attrs={"callee": function.name},
    )
    builder.function("main", (source,), (result,))
    return fm.verify_module(builder.build(entry="main"))


def _options(shared_bytes=8192, *, include_shared=True):
    maximum = (1 << 31) - 1
    spaces = [
        MemorySpace(
            "workspace", "device", 256, maximum,
            AllocationStrategy.SAT, AllocationPolicy.GRANULARITY_ALIGNED,
            "function",
        ),
        MemorySpace(
            "rdata", "readonly_device", 256, maximum,
            AllocationStrategy.LINEAR, AllocationPolicy.GRANULARITY_ALIGNED,
            "module",
        ),
    ]
    if include_shared:
        spaces.append(MemorySpace(
            "shared", "shared", 16, shared_bytes,
            AllocationStrategy.SAT, AllocationPolicy.GRANULARITY_ALIGNED,
            "function",
        ))
    spaces.append(MemorySpace(
        "external", "external", 1, maximum,
        AllocationStrategy.EXTERNAL, AllocationPolicy.GRANULARITY_ALIGNED,
        "external",
    ))
    return BufferizationOptions(tuple(spaces))


def test_bufferize_sat_allocates_and_rebinds_microkernel_shared_buffers():
    module = NttBufferizationPolicy(_options()).bufferize(_selected_module())
    plan = fm.verify_buffer_plan(module)
    function = module.prim_function_map["packed_qkv"]
    dispatch = kernel_dispatch_of(function)
    shared = dispatch.shared_workspace_buffers
    allocations = tuple(value.mem_span.buffer for value in shared)

    assert tuple(value.memory_space for value in allocations) == (
        "shared", "shared"
    )
    assert tuple(value.function for value in allocations) == (
        "packed_qkv", "packed_qkv"
    )
    assert all(value.role == "microkernel_shared_workspace" for value in allocations)
    assert not shared[0].mem_span.may_alias(shared[1].mem_span)
    assert all(
        plan.physical_buffer_map[value.id] == value for value in allocations
    )
    assert {
        value.storage for value in plan.buffers
        if value.role == "microkernel_shared_workspace"
    } == {"shared"}


def test_shared_workspace_requires_a_target_space_and_respects_capacity():
    with pytest.raises(IRVerificationError, match="no 'shared' memory space"):
        NttBufferizationPolicy(_options(include_shared=False)).bufferize(
            _selected_module()
        )
    with pytest.raises(
        IRVerificationError,
        match="exceeds memory space 'shared'|SAT allocation",
    ):
        NttBufferizationPolicy(_options(shared_bytes=1024)).bufferize(
            _selected_module()
        )


def test_transfer_source_alignment_is_propagated_to_actual_and_formal_buffers():
    module = NttBufferizationPolicy(_options()).bufferize(
        _selected_module(source_alignment=512)
    )
    plan = fm.verify_buffer_plan(module)
    actual = plan.buffer_map["fused_weight"]
    function = module.prim_function_map["packed_qkv"]
    formal = function.parameter_map["fused_weight"].buffers[0]

    assert actual.alignment == 512
    assert actual.mem_span.buffer.alignment == 512
    assert actual.offset % 512 == 0
    assert formal.mem_span.buffer.alignment == 512
