# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Runtime adapter for artifacts whose package was generated as C++.

A Triton package is launched by compiling the generated Python at prepare time
and calling into Triton's launcher. A C++ package is already compiled: the
artifact carries a shared library, and launching means calling its plain-C
``flagmega_launch`` with an array of device addresses.

Everything above the launch -- pool allocation, rdata, external-argument
validation, workspace diagnostics, the paged-attention state ABI -- is backend
independent. This subclasses the Triton call-graph module for launch and applies
the C++ package's readonly device-layout contract after canonical rdata loading.
"""

from __future__ import annotations

import ctypes
from contextlib import contextmanager
from pathlib import Path
from typing import Mapping

from triton.flagmega.artifacts.manifest import resolve_artifact_path
from triton.flagmega.errors import ArtifactError, RuntimeContractError


class _CppPackageLibrary:
    """One loaded generated library and its validated C entry points."""

    def __init__(self, path: Path, *, expected_arguments: int, abi_version: int = 1,
                 resource_admission: bool = False, operator_profile: Mapping | None = None) -> None:
        self.path = path
        try:
            self._library = ctypes.CDLL(str(path))
        except OSError as error:
            raise ArtifactError(
                f"Cannot load generated C++ package {path}: {error}."
            ) from error
        for name, restype, argtypes in (
            ("flagmega_launch", ctypes.c_int,
             (ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p) +
             ((ctypes.c_void_p,) if abi_version == 2 else ())),
            ("flagmega_argument_count", ctypes.c_int, ()),
            ("flagmega_abi_version", ctypes.c_int, ()),
        ):
            try:
                symbol = getattr(self._library, name)
            except AttributeError as error:
                raise ArtifactError(
                    f"Generated C++ package {path} has no {name!r}; it was not "
                    "built from a FlagMega C++ package source."
                ) from error
            symbol.restype = restype
            symbol.argtypes = list(argtypes)
        from triton.flagmega.codegen.cpp.renderer import CPP_RENDERER_VERSION

        abi = int(self._library.flagmega_abi_version())
        if abi_version not in {1, 2} or abi != abi_version:
            raise ArtifactError(
                f"Generated C++ package {path} declares launch ABI version "
                f"{abi}; this runtime ({CPP_RENDERER_VERSION}) speaks version "
                f"{abi_version}. Recompile the artifact."
            )
        count = int(self._library.flagmega_argument_count())
        if count != expected_arguments:
            raise ArtifactError(
                f"Generated C++ package {path} takes {count} argument(s) but "
                f"the artifact's runtime binding has {expected_arguments}. The "
                "library and the manifest are out of sync."
            )
        self.argument_count = count
        self.abi_version = abi
        self._resource_info = None
        if resource_admission:
            try:
                self._resource_info = self._library.flagmega_resource_info
            except AttributeError as error:
                raise ArtifactError(f"C++ package {path} has no compiled launch admission entry point.") from error
            self._resource_info.restype = ctypes.c_int
            self._resource_info.argtypes = [ctypes.POINTER(ctypes.c_int)]

        self._launch_profile = None
        if operator_profile is not None:
            from .operator_profile import ProfileSession
            ProfileSession(operator_profile)
            try:
                self._launch_profile = self._library.flagmega_launch_profile
            except AttributeError as error:
                raise ArtifactError("Diagnostic library missing flagmega_launch_profile.") from error
            self._launch_profile.restype = ctypes.c_int
            self._launch_profile.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p,
                ctypes.c_void_p, ctypes.c_void_p, ctypes.c_ulonglong, ctypes.c_ulonglong]

            for name, types in (
                ('flagmega_launch_profile_events', self._launch_profile.argtypes+[ctypes.c_void_p]*3),
                ('flagmega_profile_device_info', [ctypes.POINTER(ctypes.c_ubyte),ctypes.POINTER(ctypes.c_int)]),
                ('flagmega_profile_event_create', [ctypes.POINTER(ctypes.c_void_p)]),
                ('flagmega_profile_event_destroy', [ctypes.c_void_p]),
                ('flagmega_profile_event_elapsed', [ctypes.c_void_p,ctypes.c_void_p,ctypes.POINTER(ctypes.c_float)]),
                ('flagmega_profile_clock_calibrate', [ctypes.c_void_p,ctypes.c_ulonglong,ctypes.c_ulonglong,ctypes.c_uint,ctypes.c_uint,ctypes.c_void_p,ctypes.c_void_p,ctypes.c_void_p]),
                ('flagmega_profile_clock_resource_info', [ctypes.POINTER(ctypes.c_int)]),
            ):
                try: entry=getattr(self._library,name)
                except AttributeError as error: raise ArtifactError('Missing normal diagnostic export '+name) from error
                entry.argtypes=types;entry.restype=ctypes.c_int

    def profile_device_identity(self):
        uuid=(ctypes.c_ubyte*16)();values=(ctypes.c_int*5)()
        status=self._library.flagmega_profile_device_info(uuid,values)
        if status:raise RuntimeContractError(f'Diagnostic device query failed {status}.')
        text=bytes(uuid).hex();text='GPU-'+text[:8]+'-'+text[8:12]+'-'+text[12:16]+'-'+text[16:20]+'-'+text[20:]
        return {'device_UUID':text,'device_ordinal':values[0],'compute_capability':[values[1],values[2]],'SM_count':values[3],'driver_version':values[4]}

    def profile_event_create(self):
        event=ctypes.c_void_p();status=self._library.flagmega_profile_event_create(ctypes.byref(event))
        if status:raise RuntimeContractError(f'Diagnostic event create failed {status}.')
        return event

    def profile_event_destroy(self,event):
        status=self._library.flagmega_profile_event_destroy(event)
        if status:raise RuntimeContractError(f'Diagnostic event destroy failed {status}.')

    def profile_event_elapsed(self,start,end):
        value=ctypes.c_float();status=self._library.flagmega_profile_event_elapsed(start,end,ctypes.byref(value))
        if status:raise RuntimeContractError(f'Diagnostic event elapsed/completion failed {status}.')
        return float(value.value)

    def launch_profile(self, pointers, stream, *, scratch, trace, capacity, epoch, events=None):
        if self._launch_profile is None or len(pointers) != self.argument_count:
            raise RuntimeContractError("Profiling launch ABI does not match the diagnostic artifact.")
        array = (ctypes.c_void_p * self.argument_count)(*pointers)
        status = (self._launch_profile(array, stream, scratch, trace, capacity, epoch) if events is None else
                  self._library.flagmega_launch_profile_events(array,stream,scratch,trace,capacity,epoch,*events))
        if status:
            raise RuntimeContractError(f"Diagnostic launch error {status}; preserve pending trace.")

    def validate_resources(self) -> dict[str, object] | None:
        if self._resource_info is None:
            return None
        values = (ctypes.c_int * 6)()
        status = int(self._resource_info(values))
        if status != 0:
            raise RuntimeContractError(f"C++ compiled launch resource admission failed with CUDA error {status}.")
        return dict(zip(("available_sm_count", "registers_per_thread", "shared_memory_bytes",
                         "local_memory_bytes", "resident_blocks_per_sm", "max_threads_per_block"), values))

    def launch(self, pointers: tuple[int, ...], stream: int, *, scratch: int = 0) -> None:
        if len(pointers) != self.argument_count:
            raise RuntimeContractError(
                f"Generated C++ package expects {self.argument_count} device "
                f"pointer(s), got {len(pointers)}."
            )
        array = (ctypes.c_void_p * self.argument_count)(*pointers)
        status = int(
            self._library.flagmega_launch(array, ctypes.c_void_p(stream),
                                         *((ctypes.c_void_p(scratch),) if self.abi_version == 2 else ()))
        )
        if status != 0:
            raise RuntimeContractError(
                f"Launching the generated C++ package failed with backend "
                f"error {status}."
            )


@contextmanager
def _cpp_program_stream(torch, device, stream):
    """Select the launch stream for capture inspection, then restore the caller.

    ExternalStream wraps an existing handle without allocating a CUDA stream.
    Handle zero needs the existing default stream, not an ExternalStream
    constructor (which may interpret zero as a request for a new stream).
    """
    with torch.cuda.device(device):
        current = torch.cuda.current_stream(device)
        selected = current
        if stream is not None:
            stream_device = getattr(stream, 'device', None)
            if stream_device is not None and torch.device(stream_device) != current.device:
                raise RuntimeContractError('Native program stream belongs to another device.')
            handle = int(getattr(stream, 'cuda_stream', stream))
            if handle != int(current.cuda_stream):
                selected = (torch.cuda.default_stream(device) if handle == 0 else
                            torch.cuda.ExternalStream(handle, device=device))
        with torch.cuda.stream(selected):
            yield selected


class _CppLaunchMixin:
    """Replace Triton JIT prepare/launch with the generated library's."""

    def _load_rdata(self, torch, device: str, expected_nbytes: int, *, shared_rdata=None):
        pool = super()._load_rdata(torch, device, expected_nbytes, shared_rdata=shared_rdata)
        contract = self.codegen.get("readonly_device_layout")
        if contract is not None:
            from triton.flagmega.codegen.cpp.readonly_layout import apply_readonly_layout
            apply_readonly_layout(pool, contract, self.ir_module)
        return pool

    def _cpp_library(self) -> _CppPackageLibrary:
        library = getattr(self, "_library_handle", None)
        if library is None:
            name = self.codegen.get("library")
            if not isinstance(name, str) or not name:
                raise ArtifactError(
                    "C++ artifact codegen descriptor names no built library."
                )
            if self.codegen.get('cpp_programs') is not None:
                from .cpp_programs import CppProgramLibrary, verified_cpp_programs
                programs = verified_cpp_programs(self.codegen, self.ir_module)
                sections = {v['name']: v for v in self.manifest['sections']}
                library = CppProgramLibrary(resolve_artifact_path(Path(self.artifact), name), programs,
                    library_sha256=sections['codegen.library']['sha256'],
                    source_path=resolve_artifact_path(Path(self.artifact), self.codegen['source']),
                    source_sha256=self.codegen['source_sha256'], profile=self.codegen.get('program_profile'),
                    counter_epoch_plan=self.codegen.get('cpp_counter_epoch_plan'))
                self._library_handle = library
                return library
            library = _CppPackageLibrary(
                resolve_artifact_path(Path(self.artifact), name),
                expected_arguments=len(self.runtime_binding["signature"]),
                abi_version=int(self.codegen.get("launch_abi_version", 1)),
                resource_admission=self.codegen.get("launch_admission") == "cuda-occupancy-cooperative/v1",
                operator_profile=self.codegen.get("operator_profile"),
            )
            self._library_handle = library
        return library

    def _prepare_external(self, external_arguments: tuple[object, ...]) -> None:
        """Validate arguments and bind the library; nothing is JIT-compiled.

        NVIDIA queries the compiled function and occupancy before preparation
        and again before launch. COREX retains its software SM-count bound.
        """

        if self.codegen.get('cpp_programs') is not None:
            return self._prepare_cpp_programs(external_arguments)
        self._require_loaded()
        self._validate_external_arguments(external_arguments)
        import torch
        with torch.cuda.device(self._device):
            self._cpp_resource_report = self._cpp_library().validate_resources()
            contract = self.codegen.get("launch_scratch")
            if contract is not None and getattr(self, "_cpp_scratch", None) is None:
                if (not isinstance(contract, Mapping) or contract.get("kind") != "grid-barrier-counters/v1"
                        or contract.get("dtype") != "int32" or contract.get("reset") != "per-launch-stream"
                        or not isinstance(contract.get("count"), int) or contract["count"] <= 0
                        or contract.get("nbytes") != 4 * contract["count"]
                        or contract["count"] != self.codegen.get("grid_barrier_counter_slots")):
                    raise ArtifactError("C++ launch scratch does not match its barrier counter contract.")
                self._cpp_scratch = torch.empty((contract["count"],), dtype=torch.int32, device=self._device)
                self._cpp_scratch_stream = None
            if self._cpp_resource_report is not None:
                self._cpp_resource_report.update(
                    grid=list(self.codegen["grid"]), block_threads=self.codegen["block_threads"],
                    cooperative_grid=bool(self.codegen.get("distributed_entry")),
                    launch_scratch_bytes=0 if contract is None else contract["nbytes"],
                )
        profile = self.codegen.get("operator_profile")
        if profile is not None:
            from .operator_profile import ProfileSession
            self._operator_profile_session = ProfileSession(
                profile, grid=self.codegen["grid"], block_threads=self.codegen["block_threads"])
            self._operator_profile_trace = torch.empty((profile["trace_nbytes"] // 8,),
                dtype=torch.uint64, device=self._device)
            with torch.cuda.device(self._device):
                self._operator_profile_device_identity=self._cpp_library().profile_device_identity()
                self._operator_profile_events=[]
                try:
                    for _ in range(3):self._operator_profile_events.append(self._cpp_library().profile_event_create())
                except BaseException:
                    for event in self._operator_profile_events:self._cpp_library().profile_event_destroy(event)
                    self._operator_profile_events=[]
                    raise
                self._operator_clock_output=torch.empty((4,),dtype=torch.uint64,device=self._device)
                self._operator_clock_state={'known_complete':True,'pending':False,'raw':None,'error':None}
        self._prepared = ()
        self.prepare_count += 1
        self._mark_prepared()

    @property
    def prepared(self) -> bool:
        # The base class treats `()` as "not prepared" because a Triton
        # artifact always has at least one prepared kernel. A C++ artifact has
        # none, so prepared-ness is tracked by the library binding instead.
        return self._prepared is not None

    @property
    def resource_report(self):
        return getattr(self, "_cpp_resource_report", None)

    def _launch_external(
        self, external_arguments: tuple[object, ...], *, stream=None
    ) -> None:
        self._require_prepared()
        if self.codegen.get('cpp_programs') is not None:
            return self._launch_cpp_programs(external_arguments, stream=stream)
        if self.codegen.get("operator_profile") is not None:
            sink = getattr(self, "_operator_profile_sink", None)
            if not callable(sink) or getattr(self, "_operator_profile_sink_failed", False):
                raise RuntimeContractError("Diagnostic artifact requires an explicit healthy eager token sink; graph profiling unsupported.")
            epoch, raw = self.launch_profiled(external_arguments, stream=stream)
            try:
                ack=sink(epoch, raw)  # Binary+index persistence before buffer reuse.
                self._operator_profile_session.consume(epoch,ack)
            except BaseException as error:
                self._operator_profile_sink_failed = True
                self._operator_profile_session.fail(error,raw=raw,stage='durable_sink',failure_sink=getattr(self,'_operator_profile_failure_sink',None))
                raise
            return
        self._validate_external_arguments(external_arguments)
        pool_arguments = tuple(
            self._pool_values[str(pool["name"])]
            for pool in self.runtime_binding["pools"]
        )
        if stream is None:
            import torch

            stream = torch.cuda.current_stream(self._device).cuda_stream
        else:
            stream = getattr(stream, "cuda_stream", stream)
        scratch = getattr(self, "_cpp_scratch", None)
        if scratch is not None:
            previous = self._cpp_scratch_stream
            if previous is not None and previous != int(stream):
                raise RuntimeContractError("A C++ runtime with launch scratch is bound to its first CUDA stream; "
                                           "prepare another runtime instance for a different or concurrent stream.")
            self._cpp_scratch_stream = int(stream)
        import torch
        with torch.cuda.device(self._device):
            self._cpp_library().launch(
                tuple(_device_pointer(value) for value in (*external_arguments, *pool_arguments)),
                int(stream), scratch=0 if scratch is None else _device_pointer(scratch),
            )

    def _prepare_cpp_programs(self, external_arguments):
        from .cpp_programs import CppProgramFrame, CppCounterEpochFrame
        import torch
        self._require_loaded()
        self._validate_external_arguments(external_arguments)
        old = getattr(self, '_cpp_program_frame', None)
        if old is not None:
            old.require_releasable()
        with torch.cuda.device(self._device):
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeContractError('Native program preparation is forbidden during capture.')
            if isinstance(old, CppCounterEpochFrame):
                old.release()
            library = self._cpp_library()
            device = torch.cuda.current_device()
            self._cpp_resource_report = library.prepare(device=device)
            size = max((e['scratch_nbytes'] for e in library.manifest['entries']), default=0)
            self._cpp_scratch = torch.empty((size,), dtype=torch.uint8, device=self._device) if size else None
            roots = (*external_arguments, *(self._pool_values[str(p['name'])] for p in self.runtime_binding['pools']))
            if library.counter_epoch_plan is not None:
                self._cpp_program_frame = CppCounterEpochFrame(library, roots, device=device,
                    scratch=self._cpp_scratch.data_ptr(), scratch_bytes=size,
                    keepalive=(self, self._cpp_scratch, *self._pool_values.values()))
            else:
                self._cpp_program_frame = CppProgramFrame(library, roots, device=device,
                    scratch=self._cpp_scratch.data_ptr() if size else 0, scratch_bytes=size)
            if library.profile is not None:
                trace_bytes = library.profile['trace_nbytes']
                self._cpp_program_trace = torch.empty((trace_bytes // 8,), dtype=torch.uint64, device=self._device)
                self._cpp_program_frame.configure_trace(self._cpp_program_trace.data_ptr(), trace_bytes)
        self._prepared = ()
        self.prepare_count += 1
        self._mark_prepared()

    def _launch_cpp_programs(self, external_arguments, *, stream=None):
        import torch
        self._validate_external_arguments(external_arguments)
        with _cpp_program_stream(torch, self._device, stream) as selected:
            frame = self._cpp_program_frame
            roots = (*external_arguments, *(self._pool_values[str(p['name'])] for p in self.runtime_binding['pools']))
            if frame.library.counter_epoch_plan is not None:
                frame.submit(roots, selected, capturing=torch.cuda.is_current_stream_capturing())
            else:
                frame.bind(roots)
                frame.enqueue(int(selected.cuda_stream), capturing=torch.cuda.is_current_stream_capturing())

    def capture_cpp_programs(self, external_arguments, *, stream=None, before=None, after=None, keepalive=()):
        """Capture the complete normal plan and return its managed replay owner.

        Raw torch.cuda.graph around a counter-epoch launch is rejected. Update
        captured tensor contents in place and use this returned wrapper only.
        before/after retain original same-graph input updates and GPU feedback;
        keepalive must contain their additional tensor roots.
        """
        self._require_prepared()
        import torch
        frame = self._cpp_program_frame
        if frame.library.counter_epoch_plan is None:
            raise RuntimeContractError('Managed counter epoch capture requires an explicit normal plan.')
        self._validate_external_arguments(external_arguments)
        with _cpp_program_stream(torch, self._device, stream) as selected:
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeContractError('Nested counter epoch graph capture is unsupported.')
            roots = (*external_arguments, *(self._pool_values[str(p['name'])] for p in self.runtime_binding['pools']))
            return frame.capture(roots, selected, torch, before=before, after=after, keepalive=keepalive)

    def retire_cpp_program_graphs(self):
        """Actually retire owned graphs, including a failed capture quarantine."""
        import torch
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeContractError('Graph retirement requires execution outside capture.')
        frame = self._cpp_program_frame
        if frame.library.counter_epoch_plan is None:
            raise RuntimeContractError('No managed counter epoch graph owner.')
        frame.retire_graphs()

    def enqueue_cpp_program_trace(self, external_arguments, *, stream, epoch):
        """Enqueue into preallocated ranges; collection is a separate completion boundary."""
        self._require_prepared()
        import torch
        self._validate_external_arguments(external_arguments)
        with _cpp_program_stream(torch, self._device, stream) as selected:
            frame = self._cpp_program_frame
            roots = (*external_arguments, *(self._pool_values[str(p['name'])] for p in self.runtime_binding['pools']))
            frame.bind(roots)
            frame.enqueue(int(selected.cuda_stream), capturing=torch.cuda.is_current_stream_capturing(), trace_epoch=epoch)

    def collect_cpp_program_trace(self, stream, *, epoch):
        import struct
        import torch
        from triton.flagmega.codegen.cpp.program_profile import validate_observation
        frame = self._cpp_program_frame
        if torch.cuda.is_current_stream_capturing() or not frame.trace_pending or int(stream.cuda_stream) != frame.stream:
            raise RuntimeContractError('Trace collection requires its owned stream outside capture.')
        stream.synchronize()
        raw = self._cpp_program_trace.view(torch.uint8).cpu().numpy().tobytes()
        profile = self._cpp_library().profile
        observations = []
        for scope in profile['ranges']:
            count, dropped = struct.unpack_from('<QQ', raw, scope['offset'])
            records = []
            for index in range(min(count, scope['capacity'])):
                values = struct.unpack_from('<QQIIII', raw, scope['offset'] + 16 + index * 32)
                record = dict(zip(('tick','epoch','entry_id','invocation','warp','event'),values))
                if record['epoch'] != epoch:
                    raise RuntimeContractError('Trace contains a different capture epoch.')
                records.append(record)
            validate_observation(profile, scope, count=count, dropped=dropped,
                observed_warps=[r['warp'] for r in records], grid=scope['grid'], records=records)
            observations.append(dict(scope=scope, count=count, dropped=dropped, records=records))
        # Preserve cells until the caller durably consumes this exact raw capture.
        self._cpp_trace_observation = dict(epoch=epoch, raw=raw, observations=observations,
            source_library_plan_identity=self._cpp_library().identity, resources=self._cpp_resource_report,
            profile=profile, timing_eligible=False)
        return self._cpp_trace_observation

    def acknowledge_cpp_program_trace(self, epoch, durable_pin):
        import hashlib
        observation = getattr(self, '_cpp_trace_observation', None)
        raw = Path(durable_pin['path']).read_bytes()
        digest = hashlib.sha256(raw).hexdigest()
        if (observation is None or observation['epoch'] != epoch or raw != observation['raw'] or
                digest != durable_pin['sha256'] or len(raw) != durable_pin['nbytes']):
            raise RuntimeContractError('Trace acknowledgement does not bind the durably retained capture.')
        self._cpp_program_frame.trace_pending = False
        self._cpp_trace_observation = None

    def complete_cpp_programs(self, stream, *, graph_released=False):
        """Retire a native frame after its owned stream and graph lifetime."""
        import torch
        frame = self._cpp_program_frame
        if torch.cuda.is_current_stream_capturing() or not hasattr(stream, 'synchronize') or int(stream.cuda_stream) != frame.stream:
            raise RuntimeContractError('Native completion requires its owned stream outside capture.')
        if frame.library.counter_epoch_plan is not None:
            if graph_released:
                raise RuntimeContractError('A boolean cannot certify graph retirement; close its owned wrapper.')
            frame.synchronize()
        else:
            frame.complete(stream.synchronize, graph_released=graph_released)

    @property
    def profile_identity(self):
        from .operator_profile import loaded_identity
        return {**loaded_identity(self.artifact,self.manifest,self._cpp_library().path),**self._operator_profile_device_identity}

    @property
    def profile_event_observation(self):
        import copy
        return copy.deepcopy(getattr(self,'_operator_profile_event_observation',None))

    @property
    def profile_failure(self):
        session=getattr(self,'_operator_profile_session',None)
        return None if session is None else session.failure

    def attach_profile_sink(self, sink, *, failure_sink=None, stream_id=None):
        if self.codegen.get("operator_profile") is None or not callable(sink):
            raise RuntimeContractError("Profile sink requires a diagnostic artifact and callable collector.")
        if getattr(self, "_operator_profile_sink", None) is not None:
            raise RuntimeContractError("Profile sink is immutable; use a new prepared runtime.")
        if failure_sink is not None and not callable(failure_sink):raise RuntimeContractError('Failure sink must be callable.')
        self._operator_profile_sink = sink
        self._operator_profile_failure_sink = failure_sink
        self._operator_profile_expected_stream=stream_id
        self._operator_profile_sink_failed = False

    def acknowledge_profile(self, epoch, durable_index_pin):
        self._operator_profile_session.consume(epoch,durable_index_pin)

    def launch_profiled(self, external_arguments, *, stream=None):
        """One explicit eager token launch; returns complete raw bytes after stream completion.

        The caller must persist this token before the next launch. No graph path,
        timing eligibility or independent aggregation receipt is provided here.
        """
        self._require_prepared()
        self._validate_external_arguments(external_arguments)
        if self.codegen.get("operator_profile") is None:
            raise RuntimeContractError("Ordinary artifact has no diagnostic trace ABI.")
        if getattr(self,'_operator_clock_state',{}).get('pending'):raise RuntimeContractError('Clock observation not durably consumed; no model enqueue.')
        import torch
        with torch.cuda.device(self._device):
            stream = torch.cuda.current_stream(self._device) if stream is None else stream
            if not hasattr(stream, "synchronize") or not hasattr(stream, "cuda_stream"):
                raise RuntimeContractError("Diagnostic collection requires an owned CUDA stream object.")
            trace = self._operator_profile_trace
            session = self._operator_profile_session
            if str(stream.device)!=str(torch.device(self._device)):raise RuntimeContractError('Diagnostic stream belongs to another device.')
            expected=getattr(self,'_operator_profile_expected_stream',None)
            if expected is not None and expected!=int(stream.cuda_stream):raise RuntimeContractError('Capture archive belongs to another stream.')
            epoch = session.reserve(int(stream.cuda_stream), pointer=trace.data_ptr(),
                capacity=trace.numel()*trace.element_size(), capturing=torch.cuda.is_current_stream_capturing())
            pools = tuple(self._pool_values[str(p["name"])] for p in self.runtime_binding["pools"])
            scratch = getattr(self, "_cpp_scratch", None)
            if str(stream.device)!=str(torch.device(self._device)):raise RuntimeContractError('Diagnostic stream belongs to another device.')
            events=self._operator_profile_events
            def enqueue():
                self._cpp_library().launch_profile(
                    tuple(_device_pointer(v) for v in (*external_arguments,*pools)),int(stream.cuda_stream),
                    scratch=0 if scratch is None else scratch.data_ptr(),trace=trace.data_ptr(),
                    capacity=trace.numel()*trace.element_size(),epoch=epoch,events=events)
            raw=session.capture(enqueue,stream.synchronize,lambda: trace.view(torch.uint8).cpu().numpy().tobytes(),
                                failure_sink=getattr(self,'_operator_profile_failure_sink',None))
            try:
                lib=self._cpp_library()
                self._operator_profile_event_observation={'event_launch_abi':'flagmega_launch_profile_events/v1',
                    'epoch':str(epoch),'stream':str(stream.cuda_stream),'device':str(self._device),'event_status':0,
                    'reset_inclusive_ms':lib.profile_event_elapsed(events[0],events[2]),
                    'kernel_only_ms':lib.profile_event_elapsed(events[1],events[2]),
                    'reset_ms':lib.profile_event_elapsed(events[0],events[1]),'accepted_timing_eligible':False}
            except BaseException as error:
                session.fail(error,raw=raw,stage='event_elapsed',failure_sink=getattr(self,'_operator_profile_failure_sink',None));raise
            return epoch,raw

    def calibrate_profile_clock(self,iterations,seed,epoch,*,stream,failure_sink=None):
        """Normal auxiliary kernel, owned events; returns raw observation, never a passed receipt."""
        self._require_prepared()
        from .profile_clock import LADDER
        if iterations not in LADDER or type(seed) is not int or not 0<=seed<(1<<32) or type(epoch) is not int or not 0<epoch<(1<<64):raise RuntimeContractError('Bounded clock controls required.')
        import torch
        with torch.cuda.device(self._device):
            if not hasattr(stream,'synchronize') or str(stream.device)!=str(torch.device(self._device)):raise RuntimeContractError('Owned same-device stream required.')
            if self._operator_profile_session.pending or self._operator_profile_session.poisoned:raise RuntimeContractError('Model trace still pending/poisoned.')
            state=self._operator_clock_state
            if state['pending']:raise RuntimeContractError('Clock resources pending/quarantined.')
            out=self._operator_clock_output;events=self._operator_profile_events;lib=self._cpp_library()
            state.update(pending=True,known_complete=False,raw=None,error=None)
            try:
                status=lib._library.flagmega_profile_clock_calibrate(out.data_ptr(),32,epoch,iterations,seed,int(stream.cuda_stream),events[0],events[1])
                if status:raise RuntimeContractError(f'Clock enqueue failed {status}.')
                stream.synchronize();state['known_complete']=True
                raw=out.view(torch.uint8).cpu().numpy().tobytes();state['raw']=raw
                import struct
                start,end,observed_epoch,check=struct.unpack('<QQQQ',raw)
                if observed_epoch!=epoch or end<=start:raise RuntimeContractError('Clock output epoch/order failed.')
                observation={'iterations':iterations,'seed':seed,'epoch':str(epoch),'start_tick':str(start),'end_tick':str(end),'checksum':str(check),
                    'event_elapsed_ms':lib.profile_event_elapsed(events[0],events[1]),'event_status':0,'identity':self.profile_identity,'raw':raw}
                state['epoch']=epoch;state['observation']={k:v for k,v in observation.items() if k!='raw'};return observation
            except BaseException as error:
                state['error']=error
                if not state['known_complete']:
                    try:stream.synchronize();state['known_complete']=True
                    except BaseException as sync_error:
                        if hasattr(error,'add_note'):error.add_note('Clock completion unknown: '+str(sync_error))
                if state['known_complete'] and state['raw'] is None:
                    try:state['raw']=out.view(torch.uint8).cpu().numpy().tobytes()
                    except BaseException:pass
                if callable(failure_sink):
                    try:state['failure_receipt']=failure_sink(epoch,state['raw'],error,dict(state))
                    except BaseException as archive_error:
                        if hasattr(error,'add_note'):error.add_note('Clock failure archival failed: '+str(archive_error))
                raise

    def acknowledge_clock(self,epoch,durable_pin):
        """Release calibration cells only after the exact observation is persisted."""
        import hashlib,json
        state=self._operator_clock_state
        if not state['pending'] or not state['known_complete'] or state.get('epoch')!=epoch:raise RuntimeContractError('No completed matching clock observation.')
        p=Path(durable_pin['path']);data=p.read_bytes()
        if len(data)!=durable_pin['nbytes'] or hashlib.sha256(data).hexdigest()!=durable_pin['sha256']:raise RuntimeContractError('Clock persistence pin drift.')
        observation=json.loads(data)
        if observation.get('schema')!='flagmega.profile-clock-observation/v1' or observation.get('epoch')!=str(epoch) or observation.get('raw_sha256')!=hashlib.sha256(state['raw']).hexdigest():raise RuntimeContractError('Wrong archived clock cells.')
        if state.get('error') is not None:raise RuntimeContractError('Failed calibration cannot acknowledge success.')
        for key,value in state['observation'].items():
            if observation.get(key)!=value:raise RuntimeContractError('Persisted clock observation differs '+key)
        state['pending']=False

    def close(self, *, synchronize=True):
        frame = getattr(self, '_cpp_program_frame', None)
        if frame is not None and frame.pending and synchronize:
            import torch
            if torch.cuda.is_current_stream_capturing() or frame.captured:
                raise RuntimeContractError('Complete and release the owned graph before closing its native frame.')
            if frame.library.counter_epoch_plan is not None:
                frame.synchronize()
            else:
                frame.complete(lambda: torch.cuda.synchronize(torch.device(self._device)))
            synchronize = False
        return super().close(synchronize=synchronize)

    def _release_resources(self) -> None:
        frame = getattr(self, '_cpp_program_frame', None)
        if frame is not None:
            frame.require_releasable()
            if frame.library.counter_epoch_plan is not None:
                frame.release()
        session=getattr(self,'_operator_profile_session',None)
        clock=getattr(self,'_operator_clock_state',None)
        if session is not None and not session.safe_release:
            raise RuntimeContractError('Pending/failed diagnostic resources quarantined; archive/owned stream completion required before release.')
        if clock and clock['pending']:
            raise RuntimeContractError('Clock resources quarantined after failure; owned child completion/archive required.')
        for event in getattr(self,'_operator_profile_events',[]):self._cpp_library().profile_event_destroy(event)
        self._operator_profile_events=[]
        super()._release_resources()
        self._cpp_program_frame = None
        self._cpp_program_trace = None
        self._cpp_trace_observation = None
        self._operator_profile_sink = None
        self._operator_profile_sink_failed = False
        self._operator_profile_trace = None
        self._operator_profile_session = None
        self._cpp_scratch = None
        self._cpp_resource_report = None
        self._cpp_scratch_stream = None


def _device_pointer(value: object) -> int:
    pointer = getattr(value, "data_ptr", None)
    if pointer is None:
        raise RuntimeContractError(
            "A C++ package launch argument must be a device tensor; got "
            f"{type(value).__name__}."
        )
    return int(pointer())


def create_cpp_tir_runtime(artifact, manifest: Mapping[str, object], ir_module, kernel):
    """Build the C++ runtime adapter matching the artifact's entry ABI.

    Which adapter an entry needs (raw call graph, paged-attention model, gated
    delta net, single tensor) depends only on that entry's structural ABI, so
    this reuses ``tir_runtime_type`` and mixes the C++ launch path into
    whichever class it names. Enumerating backend-specific subclasses instead
    would silently lose any adapter added to the Triton path later.

    `kernel` is None for a C++ package: there is no JIT function to resolve.
    The loader passes it through only to keep one factory signature across both
    backends.
    """

    if kernel is not None:
        raise ArtifactError(
            "A C++ package has no JIT kernel object; the loader must pass None."
        )
    from triton.flagmega.runtime.module import tir_runtime_type

    base = tir_runtime_type(ir_module)
    adapter = _CPP_ADAPTERS.get(base)
    if adapter is None:
        adapter = type(f"GeneratedCpp{base.__name__.removeprefix('Generated')}",
                       (_CppLaunchMixin, base), {})
        _CPP_ADAPTERS[base] = adapter
    return adapter(artifact, manifest, ir_module, kernel)


# Composed adapters are cached so repeated loads of one artifact shape produce
# one class, keeping `isinstance` checks and tracebacks stable.
_CPP_ADAPTERS: dict[type, type] = {}


__all__ = ["create_cpp_tir_runtime"]
