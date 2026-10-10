# Copyright 2026- Xcoresigma Technology Co., Ltd

import pytest

from triton.experimental import tle
from triton.experimental.tle.language.dsa.ascend.pipe import (
    _deferred_pipe,
    _normalize_scheduler,
    _pipe_event_ranges,
    _resolve_scheduler,
    pipe_reader,
    pipe_value,
    pipe_wait_result,
    pipe_writer,
    reset_pipe_event_allocator,
)

PIPE = tle.dsa.ascend.PIPE


def make_sync_specs():
    ready_sync = tle.dsa.ascend.SyncSpec(
        sender="cube",
        receiver="vector",
        sender_pipe=PIPE.PIPE_FIX,
        receiver_pipe=PIPE.PIPE_MTE2,
    )
    free_sync = tle.dsa.ascend.SyncSpec(
        sender="vector",
        receiver="cube",
        sender_pipe=PIPE.PIPE_MTE2,
        receiver_pipe=PIPE.PIPE_FIX,
    )
    return ready_sync, free_sync


def make_scheduler(ready_sync=None, free_sync=None, event_base=None):
    return tle.dsa.ascend.pipe_scheduler(ready_sync=ready_sync, free_sync=free_sync, event_base=event_base)


def make_workspace(capacity=2, name="base"):
    return tle.dsa.workspace(name, capacity=capacity, shape=[4, 8], dtype="float32")


def make_pipe(capacity, scheduler=None, scope="cta", name=None, one_shot=False, **fields):
    return pipe_value(capacity, scope, name, scheduler, fields, one_shot=one_shot)


# Ascend event ids are a per-kernel resource: creation order auto-allocates
# non-overlapping rings, and explicitly pinned bases must not collide.
def test_event_range_allocation_and_conflicts():
    reset_pipe_event_allocator()
    scheduler = make_scheduler()

    # Creation order auto-allocates non-overlapping rings.
    first = make_pipe(capacity=2, scheduler=scheduler)
    second = make_pipe(capacity=3, scheduler=scheduler)
    assert first.event_base == 0
    assert second.event_base == 2

    # Explicit bases are honored but must not overlap each other.
    reset_pipe_event_allocator()
    make_pipe(capacity=2, scheduler=make_scheduler(event_base=4))

    with pytest.raises(ValueError, match=r"pipe event range \[5, 7\) conflicts with \[4, 6\)"):
        make_pipe(capacity=2, scheduler=make_scheduler(event_base=5))


# one_shot pipes model a single cube->vector edge: capacity is pinned to 1,
# acquire/release are rejected, and the GPU-shaped is_closed flag is constant False.
def test_one_shot_semantics():
    reset_pipe_event_allocator()

    pipe = make_pipe(capacity=1, c=make_workspace(capacity=1), one_shot=True)
    assert pipe.one_shot is True

    with pytest.raises(ValueError, match="one_shot on Ascend requires capacity=1"):
        pipe_value(2, "cta", None, None, {}, one_shot=True)

    with pytest.raises(ValueError, match="one_shot pipes do not use acquire"):
        pipe_writer.acquire.__wrapped__(pipe.writer(), 0)

    with pytest.raises(ValueError, match="one_shot pipes do not use release"):
        pipe_reader.release.__wrapped__(pipe.reader(), 0)

    # No writer.close on Ascend: the GPU-shaped flag is a compile-time constant False.
    assert pipe_wait_result("slot").is_closed is False


# The public tle.pipe entry dispatches by payload kind and keeps the GPU
# frontend's parameter signature; run_pipeline stays an Ascend-only entry point.
def test_public_pipe_entry_matches_gpu_frontend():
    import inspect
    from triton.experimental.tle.language import pipe as public_pipe

    workspace = make_workspace()

    assert public_pipe._pipe_backend({"c": workspace}) == "ascend"

    with pytest.raises(ValueError, match="must be a tle.dsa.workspace payload"):
        public_pipe._pipe_backend({"c": "not-a-workspace"})

    # scheduler= is no longer a tle.pipe parameter: unknown kwargs fall into the
    # payload fields and fail generic payload validation, scheduler descriptors
    # included.
    with pytest.raises(ValueError, match="tle.pipe field must be a tle.dsa.workspace payload"):
        public_pipe.pipe.__wrapped__(capacity=2, scheduler=make_scheduler(), c=workspace)

    parameters = inspect.signature(public_pipe.pipe.__wrapped__).parameters
    assert "scheduler" not in parameters
    assert "readers" in parameters and "one_shot" in parameters

    # run_pipeline stays an Ascend-specific entry point.
    assert callable(tle.dsa.ascend.run_pipeline)
    assert not hasattr(tle.dsa, "run_pipeline")


# run_pipeline's scheduler= accepts one PipeScheduler or (pipe_name, scheduler)
# pairs; pair mappings address pipes by name and reject unnamed/missing entries.
def test_scheduler_normalization_and_addressing():
    reset_pipe_event_allocator()
    ready_sync, free_sync = make_sync_specs()
    sched = make_scheduler(ready_sync, free_sync, event_base=4)

    single = _normalize_scheduler(make_scheduler())
    assert single is not None and isinstance(single, tle.dsa.ascend.PipeScheduler)

    pairs = _normalize_scheduler((("p0", make_scheduler(ready_sync, free_sync,
                                                        event_base=4)), ("p1", make_scheduler())))
    assert set(pairs) == {"p0", "p1"}
    assert pairs["p0"].event_base == 4

    # Addressing: single schedulers pass through; mappings need named pipes with entries.
    assert _resolve_scheduler(None, make_pipe(capacity=2), ["p0"]) is None
    assert _resolve_scheduler(sched, make_pipe(capacity=2), ["p0"]) is sched
    assert _resolve_scheduler({"p0": sched}, make_pipe(capacity=2, name="p0"), ["p0"]) is sched

    with pytest.raises(ValueError, match="require named pipes"):
        _resolve_scheduler({"p0": sched}, make_pipe(capacity=2), [])

    with pytest.raises(ValueError, match="no entry for pipe 'p1'"):
        _resolve_scheduler({"p0": sched}, make_pipe(capacity=2, name="p1"), ["p1"])


# run_pipeline injects its scheduler into not-yet-initialized pipes, rebinding
# the sync directions and event range; pinned bases and initialized pipes refuse.
def test_apply_scheduler_rebinds_directions_and_event_range():
    reset_pipe_event_allocator()

    pipe = _deferred_pipe(capacity=2, name="c_pipe", c=make_workspace())
    assert pipe.event_base == 0
    assert pipe._auto_event_range is True

    reversed_sync = tle.dsa.ascend.pipe_scheduler(
        ready_sync=tle.dsa.ascend.SyncSpec(sender="vector", receiver="cube", sender_pipe=PIPE.PIPE_MTE3,
                                           receiver_pipe=PIPE.PIPE_MTE2),
        free_sync=tle.dsa.ascend.SyncSpec(sender="cube", receiver="vector", sender_pipe=PIPE.PIPE_FIX,
                                          receiver_pipe=PIPE.PIPE_MTE3), event_base=8)
    pipe._apply_scheduler(reversed_sync)

    assert pipe.ready_sync == reversed_sync.ready_sync
    assert pipe.event_base == 8
    assert pipe._auto_event_range is False
    assert _pipe_event_ranges == [(8, 10)]

    # A pinned base cannot be moved a second time.
    with pytest.raises(ValueError, match="was pinned explicitly and cannot move"):
        pipe._apply_scheduler(tle.dsa.ascend.pipe_scheduler(event_base=12))

    pipe._inited = True
    with pytest.raises(ValueError, match="already initialized and cannot take a new scheduler"):
        pipe._apply_scheduler(tle.dsa.ascend.pipe_scheduler())


# Without a semantic context (host-side construction) the pre-arm emission is
# deferred; one_shot pipes have no free direction to pre-arm and init directly.
def test_ensure_initialized_without_semantic_defers_emission():
    reset_pipe_event_allocator()

    pipe = make_pipe(capacity=2, c=make_workspace())
    pipe.ensure_initialized(None)
    assert pipe._inited is False

    one_shot = make_pipe(capacity=1, c=make_workspace(capacity=1), one_shot=True)
    one_shot.ensure_initialized(None)
    assert one_shot._inited is True
