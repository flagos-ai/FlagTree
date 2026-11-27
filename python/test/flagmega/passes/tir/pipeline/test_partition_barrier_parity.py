# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Both warp partitions are one CTA, so they must arrive at the same barriers."""

from collections import defaultdict

from triton.flagmega import ir as fm
from triton.flagmega.ir.tir import Barrier
from triton.flagmega.passes.tir.lower_transfer_pipeline_regions import _build_producer


def _barriers(body):
    result = []
    for field in body.fields:
        if isinstance(field, Barrier):
            result.append(field)
        elif hasattr(field, "fields"):
            result.extend(_barriers(field))
        elif hasattr(field, "body") and hasattr(field.body, "fields"):
            result.extend(_barriers(field.body))
    return result


def _pipelined_body():
    """A consumer body whose barriers straddle calls the producer never runs."""
    staged = fm.KernelInvoke(
        "staged", "matmul",
        arguments=(fm.PrimCallBinding("weight", "weight"),),
        transfer_sources=("weight",), reads=("weight",),
    )
    local = fm.KernelInvoke(
        "local", "elementwise",
        results=(fm.PrimCallBinding("result", "scratch"),), writes=("scratch",),
    )
    later = fm.KernelInvoke(
        "later", "norm",
        arguments=(fm.PrimCallBinding("input", "scratch"),), reads=("scratch",),
    )
    body = fm.Sequential((
        fm.T.barrier(fm.T.BarrierScope.CHIP, ("entry",), "staged"),
        staged,
        fm.T.barrier(fm.T.BarrierScope.CHIP, ("staged",), "local"),
        local,
        fm.T.barrier(fm.T.BarrierScope.BLOCK, ("local",), "later"),
        later,
    ))
    return body, {id(staged): fm.PipelineStage("stage_0", staged)}


def test_producer_partition_replays_every_consumer_barrier():
    body, stages = _pipelined_body()
    producer = _build_producer(body, stages, {}, defaultdict(list))
    consumer_barriers = _barriers(body)
    producer_barriers = _barriers(producer)
    # One `bar.sync` per barrier is CTA-wide, so unequal counts make each
    # side's k-th arrival pair with a different synchronization point.
    assert [value.scope for value in producer_barriers] == [
        value.scope for value in consumer_barriers
    ], (
        "the producer partition must arrive at the same barrier sequence as "
        f"the consumer; got {producer_barriers!r}"
    )


def test_producer_keeps_barriers_between_calls_it_does_not_run():
    body, stages = _pipelined_body()
    producer = _build_producer(body, stages, {}, defaultdict(list))
    # `local` and `later` are consumer-only, but the barriers around them are
    # still CTA rendezvous the producer has to join.
    assert len(_barriers(producer)) == 3
