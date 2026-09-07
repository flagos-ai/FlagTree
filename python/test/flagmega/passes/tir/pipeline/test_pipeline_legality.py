# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT

from dataclasses import replace

import pytest

from triton.flagmega import ir as fm
from triton.flagmega.errors import IRVerificationError
from triton.flagmega.passes.tir import lower_transfer_pipeline_regions

from .helpers import module, pipeline_dispatch, prim_function


def test_pipeline_under_structured_control_flow_fails_fast():
    dispatch = pipeline_dispatch("kernel", 0, shared_start=0)
    loop = fm.T.for_loop(
        fm.dim("i", minimum=0, maximum=0),
        fm.T.range(0, 1),
        fm.T.LoopMode.SERIAL,
        fm.T.sequential((dispatch,)),
    )
    function = prim_function("kernel", ())
    function = replace(function, body=fm.T.sequential((loop,)))

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
