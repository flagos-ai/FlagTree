# Copyright 2026- Xcoresigma Technology Co., Ltd

from dataclasses import dataclass
from typing import Any, Dict

import triton.language.core as tl
from triton.language.core import _unwrap_if_constexpr

from ..core import TRITON_BUILTIN, TLE_BUILTIN, Workspace, builtin
from .core import PIPE, SyncSpec, sync_block_set, sync_block_wait

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


def _validate_public_name(kind, value):
    if not isinstance(value, str) or not value.isidentifier():
        raise ValueError(f"tle.pipe {kind} name must be a Python identifier, got {value!r}")
    if value.startswith("_"):
        raise ValueError(f"tle.pipe {kind} name must not start with '_', got {value!r}")
    if value in {"fields", "readers"}:
        raise ValueError(f"tle.pipe {kind} name {value!r} is reserved")


def _validate_readers(readers):
    readers = _unwrap_constexpr(readers)
    if readers is None:
        return None
    # Malformed readers= fail exactly like the GPU frontend; well-formed ones are
    # then rejected with the hardware reason.
    if isinstance(readers, str) or not isinstance(readers, (tuple, list)):
        raise ValueError("tle.pipe readers must be a compile-time tuple/list of strings")
    if not readers:
        raise ValueError("tle.pipe readers must not be empty")
    seen = set()
    for reader in readers:
        reader = _unwrap_constexpr(reader)
        _validate_public_name("reader", reader)
        if reader in seen:
            raise ValueError(f"tle.pipe readers must be unique, got duplicate {reader!r}")
        seen.add(reader)
    raise ValueError(
        "tle.pipe readers= is not supported on Ascend yet: cross-core events are directed "
        "single-cast set/wait primitives, so multi-reader pipes would need an event fan-out "
        "or arrival-count mechanism")


def _default_sync_specs():
    # The canonical CV-mix handshake: cube publishes from its fix pipeline, vector
    # consumes through its MTE2 load pipeline; the free direction mirrors it.
    ready_sync = SyncSpec(
        sender="cube",
        receiver="vector",
        sender_pipe=PIPE.PIPE_FIX,
        receiver_pipe=PIPE.PIPE_MTE2,
    )
    free_sync = SyncSpec(
        sender="vector",
        receiver="cube",
        sender_pipe=PIPE.PIPE_MTE2,
        receiver_pipe=PIPE.PIPE_FIX,
    )
    return ready_sync, free_sync


class PipeScheduler:
    """Ascend pipe scheduler descriptor.

    It bundles the Ascend-specific sync directions (which core sets/waits which
    event, through which hardware pipelines) and an optional explicit event id
    base. Build it with tle.dsa.ascend.pipe_scheduler and hand it to
    tle.dsa.ascend.run_pipeline(scheduler=...); both sync directions
    default to the canonical cube->vector CV-mix handshake and event ids are
    auto-allocated per pipe when event_base is None.
    """

    __triton_compile_time_value__ = True

    backend = "ascend"

    def __init__(self, ready_sync, free_sync, event_base=None):
        self.ready_sync = ready_sync
        self.free_sync = free_sync
        self.event_base = event_base


def pipe_scheduler(ready_sync=None, free_sync=None, event_base=None):
    """Build an Ascend pipe scheduler.

    ready_sync/free_sync describe the ready (producer->consumer) and free
    (consumer->producer) cross-core event directions; both default to the
    canonical cube->vector CV-mix handshake. event_base optionally pins the
    lowest event id of the pipe's ring; left as None it is auto-allocated.
    """
    ready_sync = _unwrap_constexpr(ready_sync)
    free_sync = _unwrap_constexpr(free_sync)
    event_base = _unwrap_constexpr(event_base)
    if ready_sync is None or free_sync is None:
        default_ready, default_free = _default_sync_specs()
        ready_sync = default_ready if ready_sync is None else ready_sync
        free_sync = default_free if free_sync is None else free_sync
    if ready_sync.sender != free_sync.receiver:
        raise ValueError("ready_sync.sender must equal free_sync.receiver")
    if ready_sync.receiver != free_sync.sender:
        raise ValueError("ready_sync.receiver must equal free_sync.sender")
    if event_base is not None:
        event_base = _as_int(event_base, "event_base")
    return PipeScheduler(ready_sync, free_sync, event_base)


# Pure descriptor constructor: callable from kernels as well as host code, so it
# only needs the builtin marker to satisfy JIT reference checks (like workspace).
setattr(pipe_scheduler, TRITON_BUILTIN, True)
setattr(pipe_scheduler, TLE_BUILTIN, True)


def _validate_reader_fields(pipe, fields):
    fields = _unwrap_constexpr(fields)
    if fields is None:
        return None
    if isinstance(fields, str) or not isinstance(fields, (tuple, list)):
        raise ValueError("tle.pipe.reader fields must be a compile-time tuple/list of field names")
    if not fields:
        raise ValueError("tle.pipe.reader fields must not be empty")

    names = []
    seen = set()
    for field in fields:
        field = _unwrap_constexpr(field)
        if not isinstance(field, str):
            raise ValueError(f"tle.pipe.reader field name must be a string, got {type(field).__name__}")
        if field not in pipe.fields:
            raise ValueError(f"tle.pipe.reader field {field!r} is not a pipe field")
        if field in seen:
            raise ValueError(f"tle.pipe.reader fields must be unique, got duplicate {field!r}")
        seen.add(field)
        names.append(field)
    return tuple(names)


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


def _release_event_range(event_base, capacity):
    _pipe_event_ranges.remove((event_base, event_base + capacity))


@dataclass(frozen=True)
class pipe_wait_result:
    __triton_compile_time_value__ = True

    slot: Any
    # Ascend events carry no closed state (no writer.close), so the GPU-shaped
    # is_closed flag is a compile-time constant False.
    is_closed: bool = False


class pipe_slot:
    __triton_compile_time_value__ = True

    def __init__(self, fields: Dict[str, Any]):
        self._fields = fields
        for name, value in fields.items():
            setattr(self, name, value)


class _pipe_endpoint:
    # Pipe endpoints are compile-time descriptors; codegen must preserve them as Python objects.
    __triton_compile_time_value__ = True

    kind = "endpoint"

    def __init__(self, pipe, field_names=None):
        self.pipe = pipe
        self.field_names = None if field_names is None else tuple(field_names)

    @property
    def capacity(self):
        return self.pipe.capacity

    @property
    def scope(self):
        return self.pipe.scope

    @property
    def name(self):
        return self.pipe.name

    @property
    def fields(self):
        if self.field_names is None:
            return self.pipe.fields
        return {name: self.pipe.fields[name] for name in self.field_names}

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
        fields = self.fields
        return pipe_slot({name: workspace.slot(stage, _semantic) for name, workspace in fields.items()})


class pipe_writer(_pipe_endpoint):
    kind = "writer"

    @builtin
    def acquire(self, iteration, _semantic=None, _generator=None):
        if self.pipe.one_shot:
            raise ValueError("tle.pipe one_shot pipes do not use acquire")
        self.pipe.ensure_initialized(_semantic, _generator)
        event_id = self._event_id(iteration, _semantic)
        sync = self.pipe.free_sync
        sync_block_wait(sync.sender, sync.receiver, event_id, sync.sender_pipe, sync.receiver_pipe, _semantic=_semantic)
        return self._slot(iteration, _semantic)

    @builtin
    def commit(self, iteration, _semantic=None, _generator=None):
        self.pipe.ensure_initialized(_semantic, _generator)
        event_id = self._event_id(iteration, _semantic)
        sync = self.pipe.ready_sync
        sync_block_set(sync.sender, sync.receiver, event_id, sync.sender_pipe, sync.receiver_pipe, _semantic=_semantic)


class pipe_reader(_pipe_endpoint):
    kind = "reader"

    @builtin
    def wait(self, iteration, _semantic=None, _generator=None):
        self.pipe.ensure_initialized(_semantic, _generator)
        event_id = self._event_id(iteration, _semantic)
        sync = self.pipe.ready_sync
        sync_block_wait(sync.sender, sync.receiver, event_id, sync.sender_pipe, sync.receiver_pipe, _semantic=_semantic)
        return pipe_wait_result(self._slot(iteration, _semantic))

    @builtin
    def release(self, iteration, _semantic=None, _generator=None):
        if self.pipe.one_shot:
            raise ValueError("tle.pipe one_shot pipes do not use release")
        self.pipe.ensure_initialized(_semantic, _generator)
        event_id = self._event_id(iteration, _semantic)
        sync = self.pipe.free_sync
        sync_block_set(sync.sender, sync.receiver, event_id, sync.sender_pipe, sync.receiver_pipe, _semantic=_semantic)


class pipe_value:
    __triton_compile_time_value__ = True

    def __init__(self, capacity, scope, name, scheduler, fields, readers=None, one_shot=False):
        self.capacity = _as_int(capacity, "capacity")
        if self.capacity < 1 or self.capacity > 16:
            raise ValueError("capacity must be in [1, 16]")
        if scope != "cta":
            raise ValueError("DSA pipe currently only supports scope='cta'")
        one_shot = _unwrap_constexpr(one_shot)
        if not isinstance(one_shot, bool):
            raise ValueError("tle.pipe one_shot must be a compile-time bool")
        self.one_shot = one_shot
        if one_shot and self.capacity != 1:
            raise ValueError("tle.pipe one_shot on Ascend requires capacity=1: one edge owns exactly one event")
        self.readers = _validate_readers(readers)
        scheduler = _unwrap_constexpr(scheduler)
        if scheduler is None:
            scheduler = pipe_scheduler()
        if not isinstance(scheduler, PipeScheduler):
            raise ValueError(
                "tle.pipe scheduler must be built by tle.dsa.ascend.pipe_scheduler, "
                f"got {type(scheduler).__name__}")
        self.scope = scope
        self.name = name
        self.scheduler = scheduler
        self.ready_sync = scheduler.ready_sync
        self.free_sync = scheduler.free_sync
        self.fields = fields
        # Deferred init state: the public tle.pipe factory captures its creation
        # point and emits the pre-arm on first use, after run_pipeline has
        # had a chance to inject a scheduler.
        self._inited = False
        self._create_ip = None
        if scheduler.event_base is None:
            self.event_base = _allocate_event_base(self.capacity)
            self._auto_event_range = True
        else:
            self.event_base = scheduler.event_base
            _reserve_event_range(self.event_base, self.capacity)
            self._auto_event_range = False

    @builtin
    def init(self, _semantic=None, _generator=None):
        # Producers must see every slot as free before the first acquire; hide that handshake in pipe().
        if self.one_shot:
            # A one-shot edge has no acquire/release cycle, so there is no free direction to pre-arm.
            return
        sync = self.free_sync
        for event_offset in range(self.capacity):
            sync_block_set(sync.sender, sync.receiver, self.event_base + event_offset, sync.sender_pipe,
                           sync.receiver_pipe, _semantic=_semantic)

    def _apply_scheduler(self, scheduler):
        """Rebind this pipe to scheduler; only allowed before the pre-arm is emitted."""
        if not isinstance(scheduler, PipeScheduler):
            raise ValueError(
                f"scheduler must be a tle.dsa.ascend.pipe_scheduler value, got {type(scheduler).__name__}")
        if self._inited:
            raise ValueError("pipe is already initialized and cannot take a new scheduler")
        if scheduler.event_base is not None and scheduler.event_base != self.event_base:
            if not self._auto_event_range:
                raise ValueError(
                    f"pipe event_base {self.event_base} was pinned explicitly and cannot move to "
                    f"{scheduler.event_base}")
            _release_event_range(self.event_base, self.capacity)
            _reserve_event_range(scheduler.event_base, self.capacity)
            self.event_base = scheduler.event_base
            self._auto_event_range = False
        self.scheduler = scheduler
        self.ready_sync = scheduler.ready_sync
        self.free_sync = scheduler.free_sync

    def _scheduler_compatible(self, scheduler):
        return (scheduler.ready_sync == self.ready_sync
                and scheduler.free_sync == self.free_sync
                and (scheduler.event_base is None or scheduler.event_base == self.event_base))

    def ensure_initialized(self, _semantic=None, _generator=None):
        """Emit the free-direction pre-arm once, at the pipe's creation point.

        tle.pipe defers init so tle.dsa.ascend.run_pipeline can inject a
        scheduler first; the insertion point saved at creation keeps the pre-arm
        ahead of every loop using the pipe.
        """
        if self._inited:
            return
        if self.one_shot:
            self._inited = True
            return
        if _semantic is None:
            # Descriptor-only context (host-side construction): nothing to emit.
            return
        builder = _semantic.builder
        saved_ip = builder.get_insertion_point()
        try:
            if self._create_ip is not None:
                builder.restore_insertion_point(self._create_ip)
            self.init(_semantic=_semantic, _generator=_generator)
        finally:
            builder.restore_insertion_point(saved_ip)
        self._inited = True

    def writer(self):
        return pipe_writer(self)

    def reader(self, name=None, fields=None):
        reader_name = _unwrap_constexpr(name)
        if reader_name is not None:
            raise ValueError("tle.pipe.reader name requires pipe readers=..., which Ascend does not support yet")
        field_names = _validate_reader_fields(self, fields)
        return pipe_reader(self, field_names=field_names)


def _validated_pipe_value(*, capacity, scope="cta", name=None, readers=None, one_shot=False, scheduler=None,
                          **fields):
    capacity = _unwrap_constexpr(capacity)
    if not isinstance(capacity, int):
        raise ValueError(f"tle.pipe capacity must be a compile-time int, got {type(capacity).__name__}")
    scope = _unwrap_constexpr(scope)
    name = _unwrap_constexpr(name)
    if name is not None and not isinstance(name, str):
        raise ValueError(f"tle.pipe name must be a string or None, got {type(name).__name__}")
    if not fields:
        raise ValueError("tle.pipe requires at least one payload field")
    for field_name, field in fields.items():
        _validate_public_name("field", field_name)
        field = _unwrap_constexpr(field)
        if not isinstance(field, Workspace):
            raise ValueError(
                f"tle.pipe field {field_name!r} must be a tle.dsa.workspace payload, got {type(field).__name__}")
        field_capacity = _as_int(field.capacity, f"field {field_name!r} capacity")
        if field_capacity != capacity:
            raise ValueError(
                f"tle.pipe field {field_name!r} capacity {field_capacity} must equal pipe capacity {capacity}")
    return pipe_value(capacity, scope, name, scheduler, fields, readers=readers, one_shot=one_shot)


def _deferred_pipe(*, capacity, scope="cta", name=None, readers=None, one_shot=False, scheduler=None, _semantic=None,
                   _generator=None, **fields):
    """Public tle.pipe path: capture the creation point, defer init to first use.

    Deferring lets tle.dsa.ascend.run_pipeline inject a scheduler before
    the pre-arm is emitted; ensure_initialized then restores the saved creation
    point so the pre-arm still lands outside every loop using the pipe.
    """
    result = _validated_pipe_value(capacity=capacity, scope=scope, name=name, readers=readers, one_shot=one_shot,
                                   scheduler=scheduler, **fields)
    if _semantic is not None:
        result._create_ip = _semantic.builder.get_insertion_point()
    return result


def _unwrap_ct_value(x):
    # language.tuple exposes .values (a real attribute); probing other attributes
    # on it raises ValueError, so touch that first. Compile-time descriptors may
    # additionally arrive wrapped in constexpr.
    if hasattr(x, "values") or hasattr(x, "handle"):
        return x
    value = getattr(x, "value", None)
    return value if value is not None else x


def _normalize_scheduler(scheduler):
    """Normalize scheduler= into None, one PipeScheduler, or {pipe_name: PipeScheduler}.

    Per-pipe configs arrive as (pipe_name, scheduler) pairs because kernel code
    cannot build dict literals. Elements are unwrapped one by one: a wholesale
    _unwrap_if_constexpr on a language.tuple rebuilds it and rejects compile-time
    string members.
    """
    if scheduler is None or isinstance(scheduler, PipeScheduler):
        return scheduler
    pairs = getattr(scheduler, "values", scheduler)
    if not isinstance(pairs, (tuple, list)):
        raise ValueError(
            "scheduler must be None, a tle.dsa.ascend.pipe_scheduler value, or (pipe_name, scheduler) pairs, "
            f"got {type(scheduler).__name__}")
    mapping = {}
    for item in pairs:
        item = _unwrap_ct_value(item)
        item = getattr(item, "values", item)
        if not isinstance(item, (list, tuple)) or len(item) != 2:
            raise ValueError("scheduler pair entries must be (pipe_name, scheduler)")
        name, sched = _unwrap_ct_value(item[0]), _unwrap_ct_value(item[1])
        if not isinstance(name, str):
            raise ValueError("scheduler pair name must be a compile-time string")
        if not isinstance(sched, PipeScheduler):
            raise ValueError(
                f"scheduler pair for {name!r} must be a tle.dsa.ascend.pipe_scheduler value, "
                f"got {type(sched).__name__}")
        if name in mapping:
            raise ValueError(f"scheduler pairs must have unique pipe names, got duplicate {name!r}")
        mapping[name] = sched
    return mapping


def _resolve_scheduler(scheduler, pipe, valid_names):
    """Pick one pipe's config from a normalized scheduler (None / PipeScheduler / name map)."""
    if scheduler is None or isinstance(scheduler, PipeScheduler):
        return scheduler
    if pipe.name is None:
        raise ValueError(
            "scheduler pairs require named pipes: an unnamed pipe cannot be addressed "
            f"(named pipes: {sorted(valid_names)})")
    try:
        return scheduler[pipe.name]
    except KeyError:
        raise ValueError(
            f"scheduler pairs have no entry for pipe {pipe.name!r} "
            f"(provided: {sorted(scheduler)}, pipes: {sorted(valid_names)})") from None


def _collect_pipes(entries):
    """Dedup pipes reachable from already-unwrapped role args (endpoints or values)."""
    pipes = []
    seen = set()
    for _, args in entries:
        for arg in args:
            if isinstance(arg, _pipe_endpoint):
                pipe = arg.pipe
            elif isinstance(arg, pipe_value):
                pipe = arg
            else:
                pipe = None
            if pipe is not None and id(pipe) not in seen:
                seen.add(id(pipe))
                pipes.append(pipe)
    return pipes


@builtin
def run_pipeline(functions_and_args, scheduler=None, _semantic=None, _generator=None):
    """Expand CV-mix role functions in place and bind their pipes' scheduling.

    functions_and_args is a [(role_fn, (args, ...)), ...] list; each role is
    inlined at the call site in order. scheduler optionally overrides the pipes
    reached through the role args before their first use: one
    tle.dsa.ascend.pipe_scheduler value applied to every pipe, or
    (pipe_name, scheduler) pairs addressing individual pipes by name.
    Already-initialized pipes reject incompatible configs.
    """
    if _generator is None:
        raise ValueError("run_pipeline requires code generator context")

    scheduler = _normalize_scheduler(scheduler)
    entries = []
    entries_src = getattr(functions_and_args, "values", functions_and_args)
    for item in entries_src:
        item = _unwrap_ct_value(item)
        if hasattr(item, "values"):
            item = item.values
        if not isinstance(item, (list, tuple)) or len(item) != 2:
            raise TypeError("run_pipeline entries must be (fn, args)")
        fn, args = _unwrap_ct_value(item[0]), _unwrap_ct_value(item[1])
        if hasattr(args, "values"):
            args = args.values
        if not isinstance(args, (list, tuple)):
            raise TypeError("run_pipeline args must be a tuple")
        args = [_unwrap_ct_value(arg) for arg in args]
        entries.append((fn, args))

    # Bind scheduling before the first inline: inject configs into pipes that
    # have not emitted their pre-arm yet, then initialize every pipe so the
    # free-direction events exist before any role body runs.
    pipes = _collect_pipes(entries)
    valid_names = [pipe.name for pipe in pipes if pipe.name is not None]
    for pipe in pipes:
        pipe_scheduler_cfg = _resolve_scheduler(scheduler, pipe, valid_names)
        if pipe._inited:
            if pipe_scheduler_cfg is not None and not pipe._scheduler_compatible(pipe_scheduler_cfg):
                raise ValueError(
                    f"pipe {pipe.name!r} is already initialized; pass the scheduler on the first "
                    "run_pipeline call")
            continue
        if pipe_scheduler_cfg is not None:
            pipe._apply_scheduler(pipe_scheduler_cfg)
        pipe.ensure_initialized(_semantic=_semantic, _generator=_generator)

    # The scheduler only expands role functions; ordering and synchronization stay explicit in pipe calls.
    for fn, args in entries:
        _generator.inline_JitFunction(fn, list(args), {})
