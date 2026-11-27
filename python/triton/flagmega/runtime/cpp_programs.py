# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Verified native serial programs, typed binding, admission and frame lifetime.

Preparation is the only path that queries resources or allocates slot arrays.
Enqueue consumes prepared arrays on one caller stream; CUDA graph ownership is
explicit and no tensor scalar is ever converted into a host value.
"""
import ctypes
import hashlib
import json
import threading
import weakref
from contextlib import contextmanager
from math import prod
from pathlib import Path

from triton.flagmega.errors import ArtifactError, RuntimeContractError
from triton.flagmega.codegen.cpp.program_abi import (
    ABI_VERSION, PROGRAM_SCHEMA, ArgTag, ArgSlot, ResourceRecord, canonical_json,
    identity, slot, validate_slots, SCALAR_TAGS, ProgramRow, PlanResult, PlanStage,
)


class ProgramTrace(ctypes.Structure):
    _fields_ = [('buffer', ctypes.c_uint64), ('capacity', ctypes.c_uint64), ('epoch', ctypes.c_uint64),
                ('entry_id', ctypes.c_uint32), ('invocation', ctypes.c_uint32)]


def verified_cpp_programs(codegen, module):
    from triton.flagmega.codegen.cpp.renderer import describe_cpp_package
    descriptor = describe_cpp_package(module)
    expected = descriptor.get('cpp_programs')
    actual = codegen.get('cpp_programs')
    if expected is None or canonical_json(actual) != canonical_json(expected):
        raise ArtifactError('Native program manifest differs from authoritative typed TIR.')
    if codegen.get('program_profile'):
        from triton.flagmega.codegen.cpp.program_profile import configure_descriptor, MODE
        profile = configure_descriptor(describe_cpp_package(module), MODE)['program_profile']
        if canonical_json(profile) != canonical_json(codegen['program_profile']):
            raise ArtifactError('Native trace profile differs from its typed entry census.')
    key = 'cpp_counter_epoch_plan'
    if (key in codegen) != (key in descriptor) or canonical_json(codegen.get(key)) != canonical_json(descriptor.get(key)):
        raise ArtifactError('Counter epoch plan differs from authoritative typed TIR.')
    return actual


def _hash_file(path, expected):
    if not isinstance(expected, str) or len(expected) != 64 or hashlib.sha256(Path(path).read_bytes()).hexdigest() != expected:
        raise ArtifactError('Native source/library identity mismatch: '+str(path))


class CppProgramLibrary:
    def __init__(self, path, manifest, *, library_sha256, source_path, source_sha256, library=None, profile=None, counter_epoch_plan=None):
        _hash_file(path, library_sha256)
        _hash_file(source_path, source_sha256)
        self.path = Path(path)
        self.manifest = json.loads(canonical_json(manifest))
        if manifest.get('schema') != PROGRAM_SCHEMA or identity({k: v for k, v in manifest.items() if k != 'plan_sha256'}) != manifest.get('plan_sha256'):
            raise ArtifactError('Invalid native program descriptor identity.')
        self.identity = (library_sha256, source_sha256, manifest['plan_sha256'])
        self._library = ctypes.CDLL(str(path)) if library is None else library
        for name, result, args in (
            ('flagmega_program_count_v3', ctypes.c_uint32, []),
            ('flagmega_program_manifest_v3', ctypes.c_char_p, []),
            ('flagmega_program_resource_info_v3', ctypes.c_int,
             [ctypes.c_uint32, ctypes.POINTER(ResourceRecord), ctypes.c_uint32]),
            ('flagmega_launch_program_v3', ctypes.c_int,
             [ctypes.c_uint32, ctypes.POINTER(ArgSlot), ctypes.c_uint32, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint64]),
        ):
            try:
                fn = getattr(self._library, name)
                fn.restype, fn.argtypes = result, args
            except AttributeError as error:
                raise ArtifactError('Missing native program export '+name) from error
        try:
            actual = json.loads(self._library.flagmega_program_manifest_v3())
        except (ValueError, TypeError) as error:
            raise ArtifactError('Malformed native immutable manifest.') from error
        if (self._library.flagmega_program_count_v3() != len(manifest['entries']) or
                canonical_json(actual) != canonical_json(manifest)):
            raise ArtifactError('Native definition table differs from the artifact manifest.')
        self.profile = profile
        if profile is not None:
            try:
                fn = self._library.flagmega_launch_program_profile_v3
                fn.restype = ctypes.c_int
                fn.argtypes = [ctypes.c_uint32, ctypes.POINTER(ArgSlot), ctypes.c_uint32, ctypes.c_void_p,
                               ctypes.c_void_p, ctypes.c_uint64, ctypes.POINTER(ProgramTrace)]
            except AttributeError as error:
                raise ArtifactError('Missing native variable-grid trace launch export.') from error
        self.counter_epoch_plan = None
        if counter_epoch_plan is not None:
            if profile is not None:
                raise ArtifactError('Normal counter epoch plans cannot inherit diagnostic generations.')
            from triton.flagmega.ir.counter_epoch_plan import CounterEpochPlan, verify_counter_epoch_plan
            typed = CounterEpochPlan.from_data(counter_epoch_plan)
            verify_counter_epoch_plan(typed, self.manifest, counter_epoch_plan['device_source_sha256'])
            for name, result, args in (
                ('flagmega_counter_epoch_manifest_v1', ctypes.c_char_p, []),
                ('flagmega_launch_counter_epoch_plan_v1', ctypes.c_int,
                 [ctypes.c_char_p, ctypes.POINTER(ProgramRow), ctypes.c_uint32, ctypes.c_void_p,
                  ctypes.c_void_p, ctypes.c_uint64, ctypes.POINTER(PlanResult)]),
            ):
                try:
                    fn = getattr(self._library, name)
                    fn.restype, fn.argtypes = result, args
                except AttributeError as error:
                    raise ArtifactError('Missing counter epoch export '+name) from error
            try:
                native = json.loads(self._library.flagmega_counter_epoch_manifest_v1())
            except (ValueError, TypeError) as error:
                raise ArtifactError('Malformed native counter epoch manifest.') from error
            if canonical_json(native) != canonical_json(counter_epoch_plan):
                raise ArtifactError('Native counter epoch table differs from the typed companion.')
            self.counter_epoch_plan = json.loads(canonical_json(counter_epoch_plan))
            self._counter_plan_hash = counter_epoch_plan['plan_sha256'].encode('ascii')
            self.identity += (counter_epoch_plan['plan_sha256'],)
        self.admitted = {}
        self._epoch_admission = None

    def invalidate(self):
        self.admitted.clear()
        self._epoch_admission = None

    def prepare(self, *, device, capturing=False):
        if capturing:
            raise RuntimeContractError('Native resource preparation is forbidden during capture.')
        self.invalidate()
        records = []
        try:
            for entry in self.manifest['entries']:
                record = ResourceRecord()
                status = self._library.flagmega_program_resource_info_v3(entry['entry_id'], ctypes.byref(record), ctypes.sizeof(record))
                if status:
                    raise RuntimeContractError(f"Native entry {entry['entry_id']} resource query failed: {status}.")
                self._admit(entry, record, device)
                records.append({key: getattr(record, key) for key, _ in record._fields_})
        except BaseException:
            self.invalidate()
            raise
        if self.counter_epoch_plan is not None:
            self._epoch_admission = (self.identity, device, tuple(self.admitted.items()))
        return records

    def _admit(self, entry, r, device):
        limit = entry['launch_resource_limits']
        if (r.version != ABI_VERSION or r.record_bytes != ctypes.sizeof(r) or r.reserved or
            r.entry_id != entry['entry_id'] or r.device != device or not r.sm_count or
            [r.cc_major, r.cc_minor] != limit['compute_capability'] or r.warp_size != entry['warp_size'] or
            r.block_threads != entry['block_threads'] or r.dynamic_shared_bytes != entry['dynamic_shared_bytes'] or
            r.block_threads > min(r.max_threads_per_block, limit['max_threads_per_block']) or
            r.static_shared_bytes + r.dynamic_shared_bytes > limit['max_shared_memory_bytes'] or
            r.registers_per_thread * r.block_threads > limit['max_registers_per_sm'] or not r.resident_blocks_per_sm):
            raise RuntimeContractError(f"Native entry {entry['entry_id']} failed resource admission.")
        if entry['launch_kind'] == 'cooperative' and (not r.cooperative_supported or
                prod(entry['grid']) > r.sm_count * r.resident_blocks_per_sm):
            raise RuntimeContractError('Cooperative native entry exceeds whole-grid residency.')
        self.admitted[entry['entry_id']] = (self.identity, device, identity(entry))

    def launch(self, entry, slots, stream, *, device, scratch=0, scratch_bytes=0, trace=None):
        if self.admitted.get(entry['entry_id']) != (self.identity, device, identity(entry)):
            raise RuntimeContractError('Native entry is not admitted for this exact library/device/plan.')
        validate_slots(slots, entry['signature'])
        required = entry['scratch_nbytes']
        if required and (not scratch or scratch % 4 or scratch_bytes != required) or not required and (scratch or scratch_bytes):
            raise RuntimeContractError('Native entry scratch ABI mismatch.')
        if trace is None:
            status = self._library.flagmega_launch_program_v3(entry['entry_id'], slots, len(slots), stream, scratch, scratch_bytes)
        else:
            if self.profile is None or not trace.epoch:
                raise RuntimeContractError('Program trace requires its prepared diagnostic library and epoch.')
            status = self._library.flagmega_launch_program_profile_v3(entry['entry_id'], slots, len(slots), stream,
                scratch, scratch_bytes, ctypes.byref(trace))
        if status:
            raise RuntimeContractError(f"Native entry {entry['entry_id']} launch failed: {status}.")


    def validate_counter_epoch_plan(self, cells, rows, *, device, scratch, scratch_bytes):
        """CPU-only preflight, also used before controlled capture prefixes."""
        admission = self._epoch_admission
        if (self.counter_epoch_plan is None or admission is None or
                admission[:2] != (self.identity, device) or
                any(self.admitted.get(k) != v for k, v in admission[2])):
            raise RuntimeContractError('Whole plan lacks current complete resource admission.')
        plan = self.counter_epoch_plan
        if (len(rows) != len(plan['rows']) or len(cells) != len(rows) or
                type(scratch) is not int or type(scratch_bytes) is not int or
                scratch_bytes != plan['frame_nbytes'] or not 0 < scratch <= 2**64 - scratch_bytes or scratch % 4):
            raise RuntimeContractError('Whole plan row/frame capacity mismatch.')
        for row, values, invocation in zip(rows, cells, plan['rows'], strict=True):
            if (row.reserved or row.arg_count != len(values) or
                    ctypes.cast(row.args, ctypes.c_void_p).value != ctypes.addressof(values)):
                raise RuntimeContractError('Whole plan prepared row identity mismatch.')
            validate_slots(values, self.manifest['entries'][invocation['entry_id']]['signature'])
    def launch_counter_epoch_plan(self, cells, rows, result, stream, *, device, scratch, scratch_bytes):
        self.validate_counter_epoch_plan(cells, rows, device=device, scratch=scratch, scratch_bytes=scratch_bytes)
        result.version, result.record_bytes = 1, ctypes.sizeof(PlanResult)
        status = self._library.flagmega_launch_counter_epoch_plan_v1(
            self._counter_plan_hash, rows, len(rows), stream, scratch, scratch_bytes, ctypes.byref(result))
        if status:
            raise CounterEpochLaunchError(status, result)
        if (result.version != 1 or result.record_bytes != ctypes.sizeof(PlanResult) or
                result.stage != PlanStage.COMPLETE or result.ordinal != len(rows) or
                result.submitted != len(rows) or result.status or result.reserved0 or result.reserved1):
            raise RuntimeContractError('Native whole-plan completion result is inconsistent.')


class CounterEpochLaunchError(RuntimeContractError):
    def __init__(self, status, result):
        super().__init__(f'Counter epoch plan stopped at ordinal {result.ordinal}, stage {result.stage}: {status}.')
        self.status, self.ordinal, self.stage, self.submitted = status, result.ordinal, result.stage, result.submitted


def _pointer(value):
    if not callable(getattr(value, 'data_ptr', None)):
        raise RuntimeContractError('Native pointer requires a physical tensor root.')
    address = value.data_ptr()
    nbytes = value.numel() * value.element_size()
    if type(address) is not int or type(nbytes) is not int or address <= 0 or nbytes < 0:
        raise RuntimeContractError('Invalid native tensor root metadata.')
    return address, nbytes


def bind_slots(bindings, roots, signature, output):
    if len(bindings) != len(signature) or len(output) != len(signature):
        raise RuntimeContractError('Native binding count mismatch.')
    for i, (binding, formal) in enumerate(zip(bindings, signature, strict=True)):
        kind = binding.get('kind')
        fields = {'argument': {'kind', 'source'}, 'pointer': {'kind', 'source', 'dtype', 'byte_offset', 'scope_stride_bytes'},
                  'immediate': {'kind', 'dtype', 'value'}, 'scalar_argument': {'kind', 'source', 'dtype', 'add'}}
        if kind not in fields or set(binding) - fields[kind]:
            raise RuntimeContractError('Unknown native binding schema/fields.')
        if kind in {'immediate', 'scalar_argument'} and SCALAR_TAGS.get(binding.get('dtype')) != formal['tag']:
            raise RuntimeContractError('Native binding scalar dtype differs from signature tag.')
        if kind == 'pointer' and (type(binding.get('scope_stride_bytes', 0)) is not int or binding.get('scope_stride_bytes', 0) < 0):
            raise RuntimeContractError('Invalid native pointer scope stride.')
        if kind == 'immediate':
            value = binding['value']
        else:
            source = binding.get('source')
            if type(source) is not int or not 0 <= source < len(roots):
                raise RuntimeContractError('Native binding root index is invalid.')
            value = roots[source]
        if formal['tag'] == ArgTag.POINTER:
            if kind not in {'argument', 'pointer'}:
                raise RuntimeContractError('Native pointer needs a root-addressable binding.')
            address, nbytes = _pointer(value)
            offset = binding.get('byte_offset', 0)
            required = formal.get('min_nbytes', 1)
            alignment = formal.get('alignment', 1)
            if (type(offset) is not int or offset < 0 or offset + required > nbytes or
                    (address + offset) % alignment):
                raise RuntimeContractError('Native pointer binding exceeds its root span/alignment.')
            value = address + offset
        elif kind not in {'argument', 'immediate', 'scalar_argument'}:
            raise RuntimeContractError('Native scalar cannot read a device pointer.')
        elif kind == 'scalar_argument' and binding.get('add', 0):
            if type(value) not in (int, float) or type(binding['add']) not in (int, float):
                raise RuntimeContractError('Native scalar arithmetic requires host immediate operands.')
            value += binding['add']
        output[i] = slot(formal['tag'], value)
    validate_slots(output, signature)


class CppProgramFrame:
    """Runtime-owned typed cells/scratch retained until declared completion.

    Same-stream reuse is ordered by native launches. Cross-stream reuse,
    release and rebinding captured roots require completion acknowledgement.
    A failed partial enqueue remains quarantined until its owned stream completes.
    """
    def __init__(self, library, roots, *, device, scratch=0, scratch_bytes=0):
        self.library, self.device = library, device
        self.entries = library.manifest['entries']
        self.invocations = library.manifest['invocations']
        self.cells = [(ArgSlot * len(self.entries[v['entry_id']]['signature']))() for v in self.invocations]
        self.scratch, self.scratch_bytes = scratch, scratch_bytes
        required = max((e['scratch_nbytes'] for e in self.entries), default=0)
        if scratch_bytes < required or required and (not scratch or scratch % 4):
            raise RuntimeContractError('Program frame has insufficient aligned scratch capacity.')
        self.stream = None
        self.pending = False
        self.poisoned = None
        self.captured = False
        self.roots = tuple(roots)
        self.retained = {id(v): v for v in roots}
        self.trace_cells = None
        self.trace_pending = False
        self.bind(roots)

    def bind(self, roots):
        if self.captured and any(a is not b for a, b in zip(roots, self.roots, strict=True)):
            raise RuntimeContractError('Captured frame roots remain pinned through graph lifetime.')
        for invocation, cells in zip(self.invocations, self.cells, strict=True):
            bind_slots(invocation['argument_bindings'], roots, self.entries[invocation['entry_id']]['signature'], cells)
        self.roots = tuple(roots)
        self.retained.update({id(v): v for v in roots})

    def configure_trace(self, pointer, nbytes):
        profile = self.library.profile
        if self.pending or profile is None or nbytes != profile['trace_nbytes'] or not pointer or pointer % 8:
            raise RuntimeContractError('Program trace requires exact preallocated capacity before enqueue.')
        self.trace_cells = [ProgramTrace(pointer+s['offset'], s['capacity'], 0, s['entry_id'], s['invocation'])
                            for s in profile['ranges']]

    def enqueue(self, stream, *, capturing=False, trace_epoch=None):
        if self.poisoned is not None:
            raise RuntimeContractError('Failed program frame is quarantined.') from self.poisoned
        if self.stream is not None and self.stream != stream and (self.pending or self.captured):
            raise RuntimeContractError('Program frame reuse requires declared stream/graph completion.')
        if trace_epoch is not None:
            if self.trace_cells is None or self.trace_pending or type(trace_epoch) is not int or not 0 < trace_epoch < 2**64:
                raise RuntimeContractError('Program trace epoch/capacity is absent or still pending consumption.')
            for cell in self.trace_cells:
                cell.epoch = trace_epoch
            self.trace_pending = True
        self.stream = stream
        self.pending = True
        self.captured |= capturing
        for index, (invocation, cells) in enumerate(zip(self.invocations, self.cells, strict=True)):
            entry = self.entries[invocation['entry_id']]
            size = entry['scratch_nbytes']
            try:
                self.library.launch(entry, cells, stream, device=self.device,
                    scratch=self.scratch if size else 0, scratch_bytes=size,
                    trace=self.trace_cells[index] if trace_epoch is not None else None)
            except BaseException as error:
                self.poisoned = error
                raise RuntimeContractError(f"Native invocation {invocation['ordinal']} path {invocation['call_path']} stopped: {error}") from error

    def complete(self, wait, *, graph_released=False):
        """Called only outside capture with the owned execution's completion wait."""
        wait()  # If completion fails, retain every resource and error.
        if self.captured and not graph_released:
            raise RuntimeContractError('Captured frame cannot release while its CUDA graph remains executable.')
        self.pending = self.captured = False
        self.stream = None
        self.retained = {id(v): v for v in self.roots}

    def require_releasable(self):
        if self.pending or self.captured or self.trace_pending:
            raise RuntimeContractError('Program frame resources retained until declared completion.')


# Weak leases exclude scratch aliases even between quiescent prepared frames.
# Strong holds prevent Python finalization from freeing pending CUDA resources.
_epoch_registry_lock = threading.Lock()
_epoch_scratch_leases = weakref.WeakValueDictionary()
_epoch_pending_holds = {}


class CppCounterEpochFrame:
    """Runtime-owned normal plan, with atomic bind/submission and owned graphs.

    This internal class receives real streams/roots from _CppLaunchMixin. Raw
    native callers and externally created graphs are outside its ownership API.
    A scratch lease never grants concurrent access to shared model state.
    """
    def __init__(self, library, roots, *, device, scratch, scratch_bytes, keepalive):
        plan = library.counter_epoch_plan
        if (plan is None or type(scratch) is not int or type(scratch_bytes) is not int or
                scratch_bytes != plan['frame_nbytes'] or not 0 < scratch <= 2**64 - scratch_bytes or scratch % 4):
            raise RuntimeContractError('Counter epoch frame requires its exact aligned capacity.')
        self.library, self.device = library, device
        self.entries, self.invocations = library.manifest['entries'], library.manifest['invocations']
        self.scratch, self.scratch_bytes = scratch, scratch_bytes
        self.cells = [(ArgSlot * len(self.entries[v['entry_id']]['signature']))() for v in self.invocations]
        self.rows = (ProgramRow * len(self.cells))(*(ProgramRow(v, len(v), 0) for v in self.cells))
        self.result = PlanResult(1, ctypes.sizeof(PlanResult), 0, 0, 0, 0, 0, 0)
        self._lock = threading.Lock()
        self._stream = None
        self.stream = None
        self.pending = self.captured = self.trace_pending = False
        self.poisoned = None
        self.cleanup_failures = []
        self.released = False
        self._graphs = []
        self._capture_in_progress = False
        self.roots = ()
        self.retained = {}
        self._keepalive = tuple(keepalive)
        self._bind(roots)
        with _epoch_registry_lock:
            for frame in list(_epoch_scratch_leases.values()):
                if frame.device == device and scratch < frame.scratch + frame.scratch_bytes and frame.scratch < scratch + scratch_bytes:
                    raise RuntimeContractError('Counter epoch scratch aliases another prepared frame.')
            _epoch_scratch_leases[id(self)] = self

    @contextmanager
    def _exclusive(self):
        if not self._lock.acquire(blocking=False):
            raise RuntimeContractError('Counter epoch frame is busy or reentered.')
        try:
            if self.released:
                raise RuntimeContractError('Counter epoch frame has been released.')
            yield
        finally:
            self._lock.release()

    def _healthy(self):
        if self.poisoned is not None:
            raise RuntimeContractError('Failed counter epoch frame is quarantined; prepare a fresh frame.') from self.poisoned

    def _own_stream(self, stream):
        handle = int(stream.cuda_stream)
        if self.stream is not None and self.stream != handle:
            raise RuntimeContractError('Counter epoch frame is bound to one owned stream.')
        if self._stream is None:
            self._stream, self.stream = stream, handle

    def _hold(self):
        with _epoch_registry_lock:
            _epoch_pending_holds[id(self)] = self

    def _drop_hold_if_complete(self):
        if not self.pending and not self._graphs:
            with _epoch_registry_lock:
                _epoch_pending_holds.pop(id(self), None)
            self.retained = {id(v): v for v in self.roots}
            # Preserve poison/type/message/stage, but completed failures must
            # not keep captured callback tensors alive through Python frames.
            errors, seen = [self.poisoned, *(error for _, error in self.cleanup_failures)], set()
            while errors:
                error = errors.pop()
                if error is None or id(error) in seen:
                    continue
                seen.add(id(error))
                error.__traceback__ = None
                errors.extend((error.__cause__, error.__context__))

    def _bind(self, roots):
        self._healthy()
        roots = tuple(roots)
        if self._graphs and (len(roots) != len(self.roots) or any(a is not b for a, b in zip(roots, self.roots))):
            raise RuntimeContractError('Captured frame roots remain pinned through graph retirement.')
        for root in roots:
            if callable(getattr(root, 'data_ptr', None)):
                address, size = _pointer(root)
                if address < self.scratch + self.scratch_bytes and self.scratch < address + size:
                    raise RuntimeContractError('Counter scratch overlaps a model root.')
        for invocation, cells in zip(self.invocations, self.cells, strict=True):
            bind_slots(invocation['argument_bindings'], roots, self.entries[invocation['entry_id']]['signature'], cells)
        self.roots = roots
        self.retained.update({id(v): v for v in roots})

    def _submit(self):
        self.pending = True
        self._hold()
        try:
            self.library.launch_counter_epoch_plan(self.cells, self.rows, self.result, self.stream,
                device=self.device, scratch=self.scratch, scratch_bytes=self.scratch_bytes)
        except BaseException as error:
            self.poisoned = error
            raise

    def submit(self, roots, stream, *, capturing=False):
        with self._exclusive():
            self._healthy()
            if capturing:
                raise RuntimeContractError('Use capture_cpp_programs; unmanaged graph capture is unsupported for counter epoch plans.')
            self._own_stream(stream)
            self._bind(roots)
            self._submit()

    def capture(self, roots, stream, torch, *, before=None, after=None, keepalive=()):
        """Capture original prefix, complete plan, and feedback on the owned stream.

        Callbacks are trusted runtime integration operations. Their additional
        tensor roots must be retained explicitly; callbacks themselves stay
        alive as well. The plan never exposes its scratch to these operations.
        """
        with self._exclusive():
            self._healthy()
            self._own_stream(stream)
            if self._graphs or self._capture_in_progress:
                raise RuntimeContractError('Retire the existing owned graph before capturing another.')
            if any(callback is not None and not callable(callback) for callback in (before, after)):
                raise RuntimeContractError('Capture prefix/feedback must be callable.')
            self._bind(roots)
            self.library.validate_counter_epoch_plan(self.cells, self.rows, device=self.device,
                scratch=self.scratch, scratch_bytes=self.scratch_bytes)
            keepalive = tuple(keepalive)
            retained = (before, after, *keepalive)
            for value in keepalive:
                if callable(getattr(value, 'data_ptr', None)):
                    address, size = _pointer(value)
                    if address < self.scratch + self.scratch_bytes and self.scratch < address + size:
                        raise RuntimeContractError('Capture feedback root overlaps counter scratch.')
            graph = _CppCounterEpochGraph(self, torch.cuda.CUDAGraph(), retained)
            self._graphs.append(graph)
            self.pending = self.captured = self._capture_in_progress = True
            self._hold()
            try:
                with torch.cuda.graph(graph._graph, stream=stream):
                    if before is not None:
                        before()
                    self._submit()
                    if after is not None:
                        after()
            except BaseException as error:
                self.poisoned = error
                raise
            finally:
                self._capture_in_progress = False
            return graph

    def _quarantine_cleanup_failure(self, error, stage):
        # Cleanup may be retried, but its failure must never reopen submission.
        # Keep the first failure (including its native ordinal/stage) intact.
        if self.poisoned is None:
            self.poisoned = error
        self.cleanup_failures.append((stage, error))
        if hasattr(error, 'add_note'):
            error.add_note(f'Counter epoch {stage} failed; the frame remains quarantined.')
        self.pending = True
        self._hold()

    def synchronize(self):
        """Use the actual owned stream; callers cannot certify completion."""
        with self._exclusive():
            if self._capture_in_progress:
                raise RuntimeContractError('Cannot synchronize during capture.')
            try:
                if self._stream is not None:
                    self._stream.synchronize()
            except BaseException as error:
                self._quarantine_cleanup_failure(error, 'completion stream synchronize')
                raise
            self.pending = False
            self._drop_hold_if_complete()

    def retire_graphs(self):
        """Wait for all replays, then actually reset graphs, including failed capture."""
        with self._exclusive():
            if self._capture_in_progress:
                raise RuntimeContractError('Cannot retire a graph during capture.')
            stage = 'retirement stream synchronize'
            try:
                if self._stream is not None:
                    self._stream.synchronize()
                # A failed reset keeps the remaining owner and every strong hold.
                for graph in list(self._graphs):
                    stage = 'retirement graph reset'
                    graph._graph.reset()
                    graph._closed = True
                    graph._graph = None
                    graph._keepalive = ()
                    self._graphs.remove(graph)
            except BaseException as error:
                self._quarantine_cleanup_failure(error, stage)
                raise
            self.pending = self.captured = False
            self._drop_hold_if_complete()

    def require_releasable(self):
        with self._exclusive():
            if self.pending or self._graphs or self._capture_in_progress:
                raise RuntimeContractError('Counter epoch resources retained until actual completion and graph retirement.')

    def release(self):
        with self._exclusive():
            if self.pending or self._graphs:
                raise RuntimeContractError('Counter epoch resources remain pending.')
            with _epoch_registry_lock:
                _epoch_scratch_leases.pop(id(self), None)
                _epoch_pending_holds.pop(id(self), None)
            self.released = True
            self._keepalive = ()
            self.retained.clear()


class _CppCounterEpochGraph:
    """Only this wrapper's replay is supported; no public raw graph escape."""
    def __init__(self, frame, graph, keepalive=()):
        self._frame, self._graph, self._closed = frame, graph, False
        self._keepalive = tuple(keepalive)

    def replay(self, *, stream=None):
        frame = self._frame
        with frame._exclusive():
            frame._healthy()
            if self._closed or self not in frame._graphs:
                raise RuntimeContractError('Owned counter epoch graph is retired.')
            if stream is not None and int(stream.cuda_stream) != frame.stream:
                raise RuntimeContractError('Owned graph replay cannot switch streams.')
            # Torch graph replay uses the current stream, so select the retained
            # owned stream even when the caller's current stream is different.
            import torch
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeContractError('Owned replay cannot be embedded in an unmanaged graph capture.')
            frame.pending = True
            frame._hold()
            try:
                with torch.cuda.stream(frame._stream):
                    self._graph.replay()
            except BaseException as error:
                frame.poisoned = error
                raise

    def close(self):
        if not self._closed:
            import torch
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeContractError('Owned graph close requires execution outside capture.')
            self._frame.retire_graphs()
