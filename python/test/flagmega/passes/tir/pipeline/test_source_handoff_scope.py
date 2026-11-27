# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Source-handoff scope: every source is resolved, one edge orders them all.

``_add_source_handoffs`` (the authored/legacy ``PrimFunction`` path) resolves a
transfer stage's read-argument set against its predecessors. Two properties are
independent and both required:

* **Diagnostic completeness** -- every source must have a stage predecessor for
  its writes. An ordinary (non-stage) write to *any* source is illegal and must
  be reported, no matter how many other sources a nearer stage happens to write.
* **Ordering minimality** -- the consumer body is straight-line, so committing
  once after the *latest* stage predecessor already implies every earlier one
  ran. One handoff per stage is therefore the correct edge count, not one per
  intersecting source.

The emitter used to stop its predecessor scan at the first intersecting
statement, which satisfied minimality but silently dropped the diagnostic for
every source whose writer sat further back.
"""

import pytest

from triton.flagmega import ir as fm
from triton.flagmega.errors import IRVerificationError
from triton.flagmega.passes.tir import lower_transfer_pipeline_regions

from .helpers import module, pipeline_dispatch, prim_function


def _stage(index, *, shared_start, source, output):
    """A pipelined dispatch transferring ``source`` and writing ``output``."""
    tensor = fm.tensor_type("float32", (16,))
    name = f"shared_{index}"
    descriptor = fm.T.shared_workspace_descriptor(name, tensor, 16)
    physical = fm.T.physical_buffer(
        f"kernel.{name}", "shared", 64, 16, start=shared_start,
        function="kernel", role="microkernel_shared_workspace",
    )
    buffer = fm.T.buffer(name, "float32", fm.T.mem_span(physical), (16,), (1,))
    contract = fm.T.transfer_pipeline_contract((
        fm.T.transfer_pipeline_channel(
            "rhs", source_argument_indices=(0,),
            shared_workspace_indices=(0,), source_alignment_bytes=16,
        ),
    ))
    selection = fm.T.microkernel_selection(
        implementation=f"test.kernel.{index}", family="test_kernel",
        variant=f"v{index}", shared_workspaces=(descriptor,),
        transfer_pipeline=contract,
    )
    return fm.T.kernel_dispatch(
        semantic_op="test.kernel", arguments=(source,), outputs=(output,),
        microkernel=selection, shared_workspace_buffers=(buffer,),
        reads=(source,), writes=(output,),
    )


def _two_source_consumer(index, *, shared_start, channel_source, producer_read):
    """A stage whose read-argument set spans a channel and a producer read."""
    tensor = fm.tensor_type("float32", (16,))
    name = f"shared_{index}"
    descriptor = fm.T.shared_workspace_descriptor(name, tensor, 16)
    physical = fm.T.physical_buffer(
        f"kernel.{name}", "shared", 64, 16, start=shared_start,
        function="kernel", role="microkernel_shared_workspace",
    )
    buffer = fm.T.buffer(name, "float32", fm.T.mem_span(physical), (16,), (1,))
    contract = fm.T.transfer_pipeline_contract(
        (
            fm.T.transfer_pipeline_channel(
                "rhs", source_argument_indices=(0,),
                shared_workspace_indices=(0,), source_alignment_bytes=16,
            ),
        ),
        producer_read_argument_indices=(1,),
    )
    selection = fm.T.microkernel_selection(
        implementation=f"test.kernel.{index}", family="test_kernel",
        variant=f"v{index}", shared_workspaces=(descriptor,),
        transfer_pipeline=contract,
    )
    return fm.T.kernel_dispatch(
        semantic_op="test.kernel",
        arguments=(channel_source, producer_read),
        outputs=("output",), microkernel=selection,
        shared_workspace_buffers=(buffer,),
        reads=(channel_source, producer_read), writes=("output",),
    )


def _handoffs(region):
    return [
        value.handoff_id for value in region.produce_body.fields
        if isinstance(value, fm.PipelineHandoff)
    ]


def test_nearer_stage_write_does_not_hide_an_ordinary_write_to_another_source():
    """The decisive case: the scan must not stop at the first intersection.

    ``consumer`` reads ``source`` (channel) and ``other`` (producer read).
    ``other`` is written by a pipeline stage, so it is correctly ordered.
    ``source`` is written by an *ordinary* dispatch with no block barrier,
    which is exactly the condition
    ``test_ordinary_write_to_transfer_source_requires_explicit_block_barrier``
    rejects when ``source`` is the only source. Adding an unrelated, properly
    staged second source must not make that violation unreportable.
    """
    ordinary = pipeline_dispatch(
        "kernel", 0, shared_start=128, pipelined=False,
        writes=("source", "output"),
    )
    staged_other = _stage(1, shared_start=64, source="source2", output="other")
    consumer = _two_source_consumer(
        2, shared_start=0, channel_source="source", producer_read="other",
    )

    with pytest.raises(IRVerificationError, match="first-class block barrier"):
        lower_transfer_pipeline_regions(
            module(prim_function("kernel", (ordinary, staged_other, consumer)))
        )


def test_distinct_stage_predecessors_share_one_ordering_edge():
    """Ordering minimality: two staged sources still need only one handoff.

    Both ``source`` and ``other`` are written by pipeline stages. The consumer
    body is straight-line, so a single commit after the *later* stage already
    implies the earlier one ran. Emitting one handoff per intersecting source
    would add a redundant rendezvous with no ordering content.
    """
    first = _stage(0, shared_start=128, source="source2", output="source")
    second = _stage(1, shared_start=64, source="source3", output="other")
    consumer = _two_source_consumer(
        2, shared_start=0, channel_source="source", producer_read="other",
    )

    lowered = lower_transfer_pipeline_regions(
        module(prim_function("kernel", (first, second, consumer)))
    )
    region = lowered.prim_function_map["kernel"].body.fields[0]
    handoffs = _handoffs(region)
    assert len(handoffs) == 1, handoffs

    # The single commit must sit after the LATER predecessor; committing after
    # the earlier one would release the producer before `other` was written.
    consume = region.consume_body.fields
    stage_ids = [
        value.stage_id if isinstance(value, fm.PipelineStage) else None
        for value in consume
    ]
    commit = next(
        index for index, value in enumerate(consume)
        if isinstance(value, fm.PipelineHandoff)
        and value.handoff_id == handoffs[0]
    )
    later = stage_ids.index("kernel_transfer_stage_1")
    assert commit > later, (commit, later, stage_ids)


def test_single_staged_source_keeps_exactly_one_handoff():
    """Regression guard: the ordinary one-source case is unchanged."""
    writer = _stage(0, shared_start=64, source="source2", output="source")
    consumer = pipeline_dispatch("kernel", 1, shared_start=0)

    lowered = lower_transfer_pipeline_regions(
        module(prim_function("kernel", (writer, consumer)))
    )
    region = lowered.prim_function_map["kernel"].body.fields[0]
    assert len(_handoffs(region)) == 1


def test_staged_overwrite_discharges_an_earlier_ordinary_write():
    """Only the *nearest* writer of a source decides its readiness.

    ``source`` is written by an ordinary dispatch and then overwritten by a
    pipeline stage. The handoff orders the consumer after that stage, which
    transitively orders the earlier ordinary write, so this is legal. The
    per-source scan must keep the nearest-writer rule rather than reporting
    every historical writer.
    """
    ordinary = pipeline_dispatch(
        "kernel", 0, shared_start=128, pipelined=False,
        writes=("source", "output"),
    )
    restage = _stage(1, shared_start=64, source="source2", output="source")
    consumer = pipeline_dispatch("kernel", 2, shared_start=0)

    lowered = lower_transfer_pipeline_regions(
        module(prim_function("kernel", (ordinary, restage, consumer)))
    )
    region = lowered.prim_function_map["kernel"].body.fields[0]
    assert len(_handoffs(region)) == 1


def test_ordinary_overwrite_after_a_stage_is_still_rejected():
    """The mirror case: a later ordinary write is not excused by an earlier stage."""
    staged = _stage(0, shared_start=64, source="source2", output="source")
    ordinary = pipeline_dispatch(
        "kernel", 1, shared_start=128, pipelined=False,
        writes=("source", "output"),
    )
    consumer = pipeline_dispatch("kernel", 2, shared_start=0)

    with pytest.raises(IRVerificationError, match="first-class block barrier"):
        lower_transfer_pipeline_regions(
            module(prim_function("kernel", (staged, ordinary, consumer)))
        )


def test_unrelated_writes_emit_no_source_handoff():
    """A predecessor writing nothing the stage reads must add no edge."""
    unrelated = _stage(0, shared_start=64, source="source2", output="elsewhere")
    consumer = pipeline_dispatch("kernel", 1, shared_start=0)

    lowered = lower_transfer_pipeline_regions(
        module(prim_function("kernel", (unrelated, consumer)))
    )
    region = lowered.prim_function_map["kernel"].body.fields[0]
    assert _handoffs(region) == []
