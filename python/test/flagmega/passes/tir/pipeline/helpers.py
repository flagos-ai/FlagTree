# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT

from triton.flagmega import ir as fm


def pipeline_dispatch(
    function: str,
    index: int,
    *,
    shared_start: int,
    source: str = "source",
    writes=("output",),
    pipelined: bool = True,
):
    tensor = fm.tensor_type("float32", (16,))
    name = f"shared_{index}"
    descriptor = fm.T.shared_workspace_descriptor(name, tensor, 16)
    physical = fm.T.physical_buffer(
        f"{function}.{name}",
        "shared",
        64,
        16,
        start=shared_start,
        function=function,
        role="microkernel_shared_workspace",
    )
    buffer = fm.T.buffer(
        name,
        "float32",
        fm.T.mem_span(physical),
        (16,),
        (1,),
    )
    pipeline = (
        fm.T.transfer_pipeline_contract((
            fm.T.transfer_pipeline_channel(
                "rhs",
                source_argument_indices=(0,),
                shared_workspace_indices=(0,),
                source_alignment_bytes=16,
            ),
        ))
        if pipelined
        else None
    )
    selection = fm.T.microkernel_selection(
        implementation=f"test.kernel.{index}",
        family="test_kernel",
        variant=f"v{index}",
        shared_workspaces=(descriptor,),
        transfer_pipeline=pipeline,
    )
    return fm.T.kernel_dispatch(
        semantic_op="test.kernel",
        arguments=(source,),
        outputs=("output",),
        microkernel=selection,
        shared_workspace_buffers=(buffer,),
        reads=(source,),
        writes=writes,
    )


def prim_function(name: str, dispatches):
    tensor = fm.tensor_type("float32", (16,))
    return fm.T.prim_function(
        name,
        "triton",
        (
            fm.T.prim_parameter("source", tensor, fm.T.PrimParameterRole.INPUT),
            fm.T.prim_parameter("output", tensor, fm.T.PrimParameterRole.OUTPUT),
        ),
        fm.T.sequential(tuple(dispatches)),
        fm.T.return_((
            fm.T.return_binding(fm.T.value_ref("output", tensor), "output"),
        )),
    )


def module(function):
    return fm.IRModule(
        dialect="bufferized_tir",
        stage="synchronized_tir",
        nodes=(),
        functions=(),
        entry="main",
        prim_functions=(function,),
    )
