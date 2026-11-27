# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""A repeated body cannot drain a pipe, so its arena reuse needs handoffs."""

from collections import defaultdict

from triton.flagmega import ir as fm
from triton.flagmega.passes.tir.lower_transfer_pipeline_regions import (
    _shared_synchronization,
)


def _owners():
    """Two stages whose shared-arena ranges overlap, in program order."""
    from triton.flagmega.passes.tir.lower_transfer_pipeline_regions import (
        _ByteRange, _Owner,
    )

    arena = ("shared", "worker")
    first = fm.KernelInvoke("first", "matmul")
    second = fm.KernelInvoke("second", "norm")
    return (
        _Owner(first, fm.PipelineStage("stage_0", first),
               (_ByteRange(arena, 0, 65536),)),
        _Owner(second, fm.PipelineStage("stage_1", second),
               (_ByteRange(arena, 0, 65536),)),
    )


def test_single_execution_uses_a_drain():
    drains, after, before = _shared_synchronization(
        _owners(), "worker", repeated=False
    )
    # A function that runs once closes its pipes in its own body, so a drain
    # is both legal and sufficient there.
    assert drains == {"stage_0"}
    assert not after and not before


def test_repeated_execution_uses_handoffs_instead():
    drains, after, before = _shared_synchronization(
        _owners(), "worker", repeated=True
    )
    # One pipe instance serves every execution, so its close lands after the
    # loop; `tle.pipe.drain` needs that close in the same async task and in
    # both tasks at once, so the drain would be hoisted out of the body and
    # stop protecting the reuse. A handoff runs per execution instead.
    assert not drains, f"a repeated body cannot drain its pipes: {drains}"
    assert after and before


def test_repeated_execution_marks_the_wrap_around_pair():
    _, after, before = _shared_synchronization(
        _owners(), "worker", repeated=True
    )
    handoffs = {
        value.handoff_id: value
        for values in after.values() for value in values
    }
    wrapped = [value for value in handoffs.values() if value.wrap_around]
    # The last owner of a byte range is the first owner's predecessor on the
    # next execution; only that pair needs priming before the loop.
    assert wrapped, f"expected a wrap-around handoff among {list(handoffs)}"
    assert len(wrapped) == 1
