# Copyright 2026- Xcoresigma Technology Co., Ltd

from dataclasses import dataclass
from typing import Any, Dict

import triton.language.core as tl
from triton.language.core import _unwrap_if_constexpr

from ..core import builtin
from .core import sync_block_set, sync_block_wait

_pipe_event_ranges = []


def reset_pipe_event_allocator():
    _pipe_event_ranges.clear()


def _unwrap_constexpr(value):
    return _unwrap_if_constexpr(value)


def _as_int(value, name):
    value = _unwrap_constexpr(value)
    if not isinstance(value, int):
        raise TypeError(f"{name} must be a compile-time integer")
    return value


def _allocate_event_base(capacity):
    # Ascend event ids are per-kernel resources; creation order gives each pipe a deterministic range.
    base = 0
    if _pipe_event_ranges:
        base = max(end for _, end in _pipe_event_ranges)
    _reserve_event_range(base, capacity)
    return base


def _reserve_event_range(event_base, capacity):
    start = event_base
    end = event_base + capacity
    if start < 0:
        raise ValueError("event_base must be non-negative")
    if end > 16:
        raise ValueError("pipe event range exceeds Ascend event id limit 16")
    for existing_start, existing_end in _pipe_event_ranges:
        if start < existing_end and existing_start < end:
            raise ValueError(f"pipe event range [{start}, {end}) conflicts with [{existing_start}, {existing_end})")
    _pipe_event_ranges.append((start, end))


@dataclass(frozen=True)
class PipeWaitResult:
    __triton_compile_time_value__ = True

    slot: Any


class PipeSlot:
    __triton_compile_time_value__ = True

    def __init__(self, fields: Dict[str, Any]):
        self._fields = fields
        for name, value in fields.items():
            setattr(self, name, value)


class PipeEndpoint:
    # Pipe endpoints are compile-time descriptors; codegen must preserve them as Python objects.
    __triton_compile_time_value__ = True

    def __init__(self, pipe, kind):
        self.pipe = pipe
        self.kind = kind

    def _stage(self, iteration, _semantic):
        if isinstance(iteration, tl.tensor):
            return iteration.__mod__(self.pipe.capacity, _semantic=_semantic)
        return iteration % self.pipe.capacity

    def _event_id(self, iteration, _semantic):
        stage = self._stage(iteration, _semantic)
        if isinstance(stage, tl.tensor):
            return stage.__add__(self.pipe.event_base, _semantic=_semantic)
        return self.pipe.event_base + stage

    def _slot(self, iteration, _semantic):
        # Payload buffers are a ring; iteration chooses the workspace stage while fields preserve user names.
        stage = self._stage(iteration, _semantic)
        fields = {name: workspace.slot(stage, _semantic) for name, workspace in self.pipe.fields.items()}
        return PipeSlot(fields)

    @builtin
    def acquire(self, iteration, _semantic=None, _generator=None):
        if self.kind != "writer":
            raise ValueError("acquire is only valid on a pipe writer")
        event_id = self._event_id(iteration, _semantic)
        sync = self.pipe.free_sync
        sync_block_wait(sync.sender, sync.receiver, event_id, sync.sender_pipe, sync.receiver_pipe, _semantic=_semantic)
        return self._slot(iteration, _semantic)

    @builtin
    def commit(self, iteration, _semantic=None, _generator=None):
        if self.kind != "writer":
            raise ValueError("commit is only valid on a pipe writer")
        event_id = self._event_id(iteration, _semantic)
        sync = self.pipe.ready_sync
        sync_block_set(sync.sender, sync.receiver, event_id, sync.sender_pipe, sync.receiver_pipe, _semantic=_semantic)

    @builtin
    def wait(self, iteration, _semantic=None, _generator=None):
        if self.kind != "reader":
            raise ValueError("wait is only valid on a pipe reader")
        event_id = self._event_id(iteration, _semantic)
        sync = self.pipe.ready_sync
        sync_block_wait(sync.sender, sync.receiver, event_id, sync.sender_pipe, sync.receiver_pipe, _semantic=_semantic)
        return PipeWaitResult(self._slot(iteration, _semantic))

    @builtin
    def release(self, iteration, _semantic=None, _generator=None):
        if self.kind != "reader":
            raise ValueError("release is only valid on a pipe reader")
        event_id = self._event_id(iteration, _semantic)
        sync = self.pipe.free_sync
        sync_block_set(sync.sender, sync.receiver, event_id, sync.sender_pipe, sync.receiver_pipe, _semantic=_semantic)


class Pipe:
    __triton_compile_time_value__ = True

    def __init__(self, capacity, scope, name, ready_sync, free_sync, event_base, fields):
        self.capacity = _as_int(capacity, "capacity")
        if self.capacity < 1 or self.capacity > 16:
            raise ValueError("capacity must be in [1, 16]")
        if scope != "cta":
            raise ValueError("DSA pipe currently only supports scope='cta'")
        if ready_sync is None or free_sync is None:
            raise ValueError("ready_sync and free_sync must be provided")
        if ready_sync.sender != free_sync.receiver:
            raise ValueError("ready_sync.sender must equal free_sync.receiver")
        if ready_sync.receiver != free_sync.sender:
            raise ValueError("ready_sync.receiver must equal free_sync.sender")
        self.scope = scope
        self.name = name
        self.ready_sync = ready_sync
        self.free_sync = free_sync
        self.fields = fields
        if event_base is None:
            self.event_base = _allocate_event_base(self.capacity)
        else:
            self.event_base = _as_int(event_base, "event_base")
            _reserve_event_range(self.event_base, self.capacity)

    @builtin
    def init(self, _semantic=None, _generator=None):
        # Producers must see every slot as free before the first acquire; hide that handshake in pipe().
        sync = self.free_sync
        for event_offset in range(self.capacity):
            sync_block_set(sync.sender, sync.receiver, self.event_base + event_offset, sync.sender_pipe,
                           sync.receiver_pipe, _semantic=_semantic)

    def writer(self):
        return PipeEndpoint(self, "writer")

    def reader(self):
        return PipeEndpoint(self, "reader")


def pipe(*, capacity, scope="cta", name=None, ready_sync=None, free_sync=None, event_base=None, _semantic=None,
         _generator=None, **fields):
    result = Pipe(capacity, scope, name, ready_sync, free_sync, event_base, fields)
    result.init(_semantic=_semantic, _generator=_generator)
    return result
