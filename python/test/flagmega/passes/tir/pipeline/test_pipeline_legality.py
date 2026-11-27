# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT

from dataclasses import replace

import pytest

from triton.flagmega import ir as fm
from triton.flagmega.errors import IRVerificationError
from triton.flagmega.ir.tir import For
from triton.flagmega.passes.tir import lower_transfer_pipeline_regions

from .helpers import module, pipeline_dispatch, prim_function


def test_pipeline_under_counted_loop_is_mirrored_on_both_sides():
    """A counted loop runs its body a proven number of times.

    The region mirrors the loop into both partitions, so each trip performs
    one push/pop pair and the stage order stays paired. Only the structures
    without that guarantee remain illegal.
    """
    dispatch = pipeline_dispatch("kernel", 0, shared_start=0)
    loop = fm.T.for_loop(
        fm.dim("i", minimum=0, maximum=0),
        fm.T.range(0, 1),
        fm.T.LoopMode.SERIAL,
        fm.T.sequential((dispatch,)),
    )
    function = prim_function("kernel", ())
    function = replace(function, body=fm.T.sequential((loop,)))

    lowered = lower_transfer_pipeline_regions(module(function))
    region = lowered.prim_function_map["kernel"].body.fields[0]
    for side in (region.produce_body, region.consume_body):
        carried = side.fields[0]
        assert isinstance(carried, For), side
        assert carried.body.fields, "each partition keeps the loop's stage"


def test_pipeline_under_unbounded_control_flow_fails_fast():
    dispatch = pipeline_dispatch("kernel", 0, shared_start=0)
    scoped = fm.T.block("guarded", fm.T.sequential((dispatch,)))
    function = prim_function("kernel", ())
    function = replace(function, body=fm.T.sequential((scoped,)))

    with pytest.raises(IRVerificationError, match="straight-line stage order"):
        lower_transfer_pipeline_regions(module(function))


def test_pipeline_requires_fixed_function_owned_shared_memspan():
    dispatch = pipeline_dispatch("kernel", 0, shared_start=0)
    shared = dispatch.shared_workspace_buffers[0]
    wrong_owner = replace(shared.mem_span.buffer, function="another_function")
    dispatch = replace(
        dispatch,
        shared_workspace_buffers=(
            replace(shared, mem_span=fm.T.mem_span(wrong_owner)),
        ),
    )

    with pytest.raises(IRVerificationError, match="function-owned post-Bufferize"):
        lower_transfer_pipeline_regions(
            module(prim_function("kernel", (dispatch,)))
        )


def test_ordinary_write_to_transfer_source_requires_explicit_block_barrier():
    ordinary = pipeline_dispatch(
        "kernel",
        0,
        shared_start=128,
        pipelined=False,
        writes=("source", "output"),
    )
    pipeline = pipeline_dispatch("kernel", 1, shared_start=0)

    with pytest.raises(IRVerificationError, match="first-class block barrier"):
        lower_transfer_pipeline_regions(
            module(prim_function("kernel", (ordinary, pipeline)))
        )
