# Copyright 2026- Xcoresigma Technology Co., Ltd

import pytest

from triton.experimental import tle
from triton.experimental.tle.language.dsa.ascend.pipe import Pipe, PipeEndpoint, reset_pipe_event_allocator

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


def make_pipe(capacity, ready_sync, free_sync, scope="cta", event_base=None, **fields):
    return Pipe(capacity, scope, None, ready_sync, free_sync, event_base, fields)


def test_sync_spec_validates_roles_and_pipe_types():
    with pytest.raises(ValueError, match="sender must be 'cube' or 'vector'"):
        tle.dsa.ascend.SyncSpec("scalar", "vector", PIPE.PIPE_FIX, PIPE.PIPE_MTE2)

    with pytest.raises(ValueError, match="sender and receiver must be different"):
        tle.dsa.ascend.SyncSpec("cube", "cube", PIPE.PIPE_FIX, PIPE.PIPE_MTE2)

    with pytest.raises(TypeError, match="sender_pipe must be an instance of PIPE"):
        tle.dsa.ascend.SyncSpec("cube", "vector", "PIPE_FIX", PIPE.PIPE_MTE2)


def test_pipe_allocates_non_overlapping_event_ranges():
    reset_pipe_event_allocator()
    ready_sync, free_sync = make_sync_specs()

    first = make_pipe(capacity=2, ready_sync=ready_sync, free_sync=free_sync)
    second = make_pipe(capacity=3, ready_sync=ready_sync, free_sync=free_sync)

    assert first.event_base == 0
    assert second.event_base == 2


def test_pipe_rejects_conflicting_explicit_event_ranges():
    reset_pipe_event_allocator()
    ready_sync, free_sync = make_sync_specs()

    make_pipe(capacity=2, ready_sync=ready_sync, free_sync=free_sync, event_base=4)

    with pytest.raises(ValueError, match=r"pipe event range \[5, 7\) conflicts with \[4, 6\)"):
        make_pipe(capacity=2, ready_sync=ready_sync, free_sync=free_sync, event_base=5)


def test_pipe_validates_capacity_scope_and_sync_pairing():
    reset_pipe_event_allocator()
    ready_sync, free_sync = make_sync_specs()

    with pytest.raises(ValueError, match=r"capacity must be in \[1, 16\]"):
        make_pipe(capacity=0, ready_sync=ready_sync, free_sync=free_sync)

    with pytest.raises(ValueError, match="DSA pipe currently only supports scope='cta'"):
        make_pipe(capacity=1, scope="gpu", ready_sync=ready_sync, free_sync=free_sync)

    bad_free_sync = tle.dsa.ascend.SyncSpec(
        sender="cube",
        receiver="vector",
        sender_pipe=PIPE.PIPE_FIX,
        receiver_pipe=PIPE.PIPE_MTE2,
    )
    with pytest.raises(ValueError, match="ready_sync.sender must equal free_sync.receiver"):
        make_pipe(capacity=1, ready_sync=ready_sync, free_sync=bad_free_sync)


def test_pipe_exposes_named_payload_fields_and_endpoints():
    reset_pipe_event_allocator()
    ready_sync, free_sync = make_sync_specs()
    workspace = tle.dsa.workspace("base", capacity=2, shape=[4, 8], dtype="float32")

    pipe = make_pipe(capacity=2, ready_sync=ready_sync, free_sync=free_sync, c=workspace)
    writer = pipe.writer()
    reader = pipe.reader()

    assert isinstance(writer, PipeEndpoint)
    assert isinstance(reader, PipeEndpoint)
    assert writer.kind == "writer"
    assert reader.kind == "reader"
    assert pipe.fields["c"] is workspace
