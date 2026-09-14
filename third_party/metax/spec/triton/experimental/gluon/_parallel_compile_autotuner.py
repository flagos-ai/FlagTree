from __future__ import annotations

import logging
import multiprocessing
import queue
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Iterator, Mapping, Sequence

from triton.runtime.autotuner import Autotuner

from ._layout_autotune import (
    CompilerLayoutAutotuner,
    GluonLayoutCompiledVariant,
    GluonLayoutVariantBundle,
    LayoutAutotuneError,
    LayoutManifest,
    LayoutVariant,
    _AUTOTUNE_CANDIDATE_TIMEOUT_SECONDS,
    _AUTOTUNE_WORKER_SETUP_TIMEOUT_SECONDS,
    _CandidateExecutionFailure,
    _DomainMeasurements,
    _IsolatedMeasurementRequest,
    _ScratchReplay,
    _close_isolated_worker,
    _device_event_benchmarker,
    _kernel_resources,
    _make_kernel_reload_descriptor,
    _measure_variant,
    _receive_worker_message,
    _validated_measurement_result,
)
from ._layout_autotune_factory import register_layout_autotuner_factory


_LOGGER = logging.getLogger(__name__)
_DEFAULT_BATCH_SIZE = 16
_DEFAULT_COMPILE_WORKERS = 16
_DEFAULT_READY_BATCHES = 1


@dataclass(frozen=True)
class Candidate:
    order_index: int
    specification: LayoutVariant


@dataclass(frozen=True)
class CandidateDomain:
    """Closed compiler-manifest domain with no runtime generation or pruning."""

    fallback: Candidate
    nonfallback: tuple[Candidate, ...]

    @classmethod
    def from_manifest(cls, manifest: LayoutManifest) -> CandidateDomain:
        ordered = (manifest.fallback,) + tuple(
            variant
            for variant in manifest.variants
            if variant != manifest.fallback
        )
        candidates = tuple(
            Candidate(order_index, specification)
            for order_index, specification in enumerate(ordered)
        )
        return cls(candidates[0], candidates[1:])

    def batches(self, batch_size: int) -> tuple[tuple[Candidate, ...], ...]:
        if batch_size <= 0:
            raise ValueError("compile batch size must be positive")
        return tuple(
            self.nonfallback[offset:offset + batch_size]
            for offset in range(0, len(self.nonfallback), batch_size)
        )


@dataclass(frozen=True)
class CompiledCandidate:
    candidate: Candidate
    compiled: GluonLayoutCompiledVariant | None
    error: Exception | None

    @property
    def order_index(self) -> int:
        return self.candidate.order_index

    @property
    def specification(self) -> LayoutVariant:
        return self.candidate.specification


@dataclass(frozen=True)
class CompiledBatch:
    batch_index: int
    results: tuple[CompiledCandidate, ...]


class IsolatedCompileBackend:
    """Compile through the bundle's existing per-candidate isolated backend."""

    def compile(
        self,
        bundle: GluonLayoutVariantBundle,
        candidate: Candidate,
    ) -> GluonLayoutCompiledVariant:
        return bundle.get_or_compile(candidate.specification)


@dataclass(frozen=True)
class _ProducerFailure:
    error: BaseException


class _EndOfBatches:
    pass


_END_OF_BATCHES = _EndOfBatches()


class _CompiledBatchStream:
    """Iterator whose explicit close works even before the first next()."""

    def __init__(
        self,
        ready: queue.Queue,
        stop: threading.Event,
        producer: threading.Thread,
    ):
        self._ready = ready
        self._stop = stop
        self._producer = producer
        self._closed = False

    def __iter__(self):
        return self

    def __next__(self) -> CompiledBatch:
        if self._closed:
            raise StopIteration
        item = self._ready.get()
        if item is _END_OF_BATCHES:
            self.close()
            raise StopIteration
        if isinstance(item, _ProducerFailure):
            self.close()
            raise item.error
        return item

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._stop.set()
        self._producer.join()


class BatchCompileScheduler:
    """Produce complete, manifest-ordered batches under bounded backpressure."""

    def __init__(
        self,
        *,
        batch_size: int = _DEFAULT_BATCH_SIZE,
        max_workers: int = _DEFAULT_COMPILE_WORKERS,
        ready_batches: int = _DEFAULT_READY_BATCHES,
        backend: IsolatedCompileBackend | None = None,
    ):
        if batch_size <= 0 or batch_size > _DEFAULT_BATCH_SIZE:
            raise ValueError("compile batch size must be in [1, 16]")
        if max_workers <= 0 or max_workers > _DEFAULT_COMPILE_WORKERS:
            raise ValueError("compile workers must be in [1, 16]")
        if ready_batches <= 0:
            raise ValueError("ready batch capacity must be positive")
        self.batch_size = batch_size
        self.max_workers = max_workers
        self.ready_batches = ready_batches
        self.backend = backend or IsolatedCompileBackend()

    @staticmethod
    def _put(
        ready: queue.Queue,
        item: CompiledBatch | _ProducerFailure | _EndOfBatches,
        stop: threading.Event,
    ) -> bool:
        while not stop.is_set():
            try:
                ready.put(item, timeout=0.05)
                return True
            except queue.Full:
                continue
        return False

    def _compile_batch(
        self,
        executor: ThreadPoolExecutor,
        bundle: GluonLayoutVariantBundle,
        batch_index: int,
        candidates: tuple[Candidate, ...],
    ) -> CompiledBatch:
        futures = {
            executor.submit(self.backend.compile, bundle, candidate): candidate
            for candidate in candidates
        }
        completed: dict[int, CompiledCandidate] = {}
        for future in as_completed(futures):
            candidate = futures[future]
            try:
                compiled = future.result()
            except Exception as error:
                result = CompiledCandidate(candidate, None, error)
            else:
                result = CompiledCandidate(candidate, compiled, None)
            completed[candidate.order_index] = result
        return CompiledBatch(
            batch_index,
            tuple(completed[candidate.order_index] for candidate in candidates),
        )

    def stream(
        self,
        bundle: GluonLayoutVariantBundle,
        domain: CandidateDomain,
    ) -> Iterator[CompiledBatch]:
        batches = domain.batches(self.batch_size)
        if not batches:
            return iter(())

        ready: queue.Queue = queue.Queue(maxsize=self.ready_batches)
        stop = threading.Event()

        def produce() -> None:
            try:
                with ThreadPoolExecutor(
                    max_workers=self.max_workers,
                    thread_name_prefix="gluon-layout-compile",
                ) as executor:
                    for batch_index, candidates in enumerate(batches):
                        if stop.is_set():
                            return
                        compiled = self._compile_batch(
                            executor, bundle, batch_index, candidates
                        )
                        if not self._put(ready, compiled, stop):
                            return
            except BaseException as error:
                self._put(ready, _ProducerFailure(error), stop)
            finally:
                self._put(ready, _END_OF_BATCHES, stop)

        producer = threading.Thread(
            target=produce,
            name="gluon-layout-batch-producer",
            daemon=True,
        )
        producer.start()

        return _CompiledBatchStream(ready, stop, producer)


def _persistent_measurement_worker(connection, device: int) -> None:
    """Measure serial candidates while reusing one framework/device context."""

    phase = "session-setup"
    try:
        from types import SimpleNamespace

        from triton.compiler.compiler import CompiledKernel
        from triton.runtime.driver import driver

        active = driver.active
        device_interface = active.get_device_interface()
        device_interface.set_device(device)
        stream = active.get_current_stream(device)
        connection.send({"status": "session-ready"})
        while True:
            command = connection.recv()
            if command == "stop":
                return
            if (
                not isinstance(command, Mapping)
                or command.get("command") != "prepare"
                or not isinstance(
                    command.get("request"), _IsolatedMeasurementRequest
                )
            ):
                raise RuntimeError(
                    "persistent measurement worker received an invalid command"
                )
            request = command["request"]
            phase = "candidate-setup"
            function = SimpleNamespace(
                arg_names=list(request.reload.argument_names),
                constexprs=request.reload.constexprs,
            )
            source = SimpleNamespace(
                fn=function,
                signature=dict(request.reload.signature),
                constants={
                    path: None for path in request.reload.constant_paths
                },
            )
            kernel = CompiledKernel(
                source,
                dict(request.reload.metadata_group),
                request.reload.compilation_hash,
            )
            _ = kernel.run
            connection.send({"status": "candidate-ready"})
            if connection.recv() != "run":
                raise RuntimeError(
                    "persistent measurement worker received an invalid run command"
                )
            phase = "measurement"

            def reset() -> None:
                for scratch, original in request.reset_pairs:
                    scratch.copy_(original)

            measurement = _measure_variant(
                kernel,
                request.grid,
                stream,
                request.args,
                _device_event_benchmarker,
                reset if request.reset_pairs else None,
            )
            connection.send(
                {
                    "status": "ok",
                    "timing": measurement.timing,
                    "device_samples_ms": measurement.device_samples_ms,
                    "resources": _kernel_resources(kernel),
                }
            )
            phase = "candidate-wait"
    except EOFError:
        pass
    except BaseException as error:
        try:
            connection.send(
                {
                    "status": "error",
                    "phase": phase,
                    "error_type": (
                        f"{type(error).__module__}.{type(error).__qualname__}"
                    ),
                    "message": str(error),
                }
            )
        except BaseException:
            pass
    finally:
        connection.close()


class _PersistentMeasurementSession:
    """One killable serial worker for one workload autotune transaction."""

    def __init__(self):
        self._process = None
        self._connection = None

    def _discard(self) -> None:
        process = self._process
        connection = self._connection
        self._process = None
        self._connection = None
        if process is not None and connection is not None:
            _close_isolated_worker(process, connection)

    def _ensure_started(self) -> None:
        if self._process is not None:
            return
        from triton.runtime.driver import driver

        if "forkserver" not in multiprocessing.get_all_start_methods():
            raise LayoutAutotuneError(
                "persistent layout measurement requires forkserver support"
            )
        context = multiprocessing.get_context("forkserver")
        parent_connection, child_connection = context.Pipe(duplex=True)
        process = context.Process(
            target=_persistent_measurement_worker,
            args=(
                child_connection,
                int(driver.active.get_current_device()),
            ),
            daemon=True,
        )
        self._process = process
        self._connection = parent_connection
        try:
            process.start()
        finally:
            child_connection.close()
        ready = _receive_worker_message(
            process,
            parent_connection,
            _AUTOTUNE_WORKER_SETUP_TIMEOUT_SECONDS,
            "session-setup",
        )
        if ready.get("status") == "error":
            raise _CandidateExecutionFailure(
                "setup-error",
                f"{ready.get('error_type', 'unknown')}: "
                f"{ready.get('message', '')}",
            )
        if ready.get("status") != "session-ready":
            raise _CandidateExecutionFailure(
                "setup-protocol",
                "persistent measurement worker did not report SESSION_READY",
            )

    @staticmethod
    def _request(
        kernel: Any,
        grid: Sequence[int],
        replay: _ScratchReplay,
    ) -> _IsolatedMeasurementRequest:
        from triton.runtime.driver import driver

        reload = _make_kernel_reload_descriptor(kernel)
        args = list(replay.args)
        for path in reload.constant_paths:
            args[path[0]] = None
        return _IsolatedMeasurementRequest(
            reload,
            tuple(int(item) for item in grid),
            int(driver.active.get_current_device()),
            tuple(args),
            replay.reset_pairs,
        )

    def measure(
        self,
        kernel: Any,
        grid: Sequence[int],
        replay: _ScratchReplay,
    ):
        setup_started_ns = time.perf_counter_ns()
        try:
            self._ensure_started()
            assert self._process is not None
            assert self._connection is not None
            request = self._request(kernel, grid, replay)
            self._connection.send(
                {"command": "prepare", "request": request}
            )
            ready = _receive_worker_message(
                self._process,
                self._connection,
                _AUTOTUNE_WORKER_SETUP_TIMEOUT_SECONDS,
                "candidate-setup",
            )
            setup_wall_us = (
                time.perf_counter_ns() - setup_started_ns
            ) // 1000
            if ready.get("status") == "error":
                raise _CandidateExecutionFailure(
                    "setup-error",
                    f"{ready.get('error_type', 'unknown')}: "
                    f"{ready.get('message', '')}",
                )
            if ready.get("status") != "candidate-ready":
                raise _CandidateExecutionFailure(
                    "setup-protocol",
                    "persistent measurement worker did not report "
                    "CANDIDATE_READY",
                )
            execution_started_ns = time.perf_counter_ns()
            self._connection.send("run")
            measured = _receive_worker_message(
                self._process,
                self._connection,
                _AUTOTUNE_CANDIDATE_TIMEOUT_SECONDS,
                "execution",
            )
            execution_wall_us = (
                time.perf_counter_ns() - execution_started_ns
            ) // 1000
            if measured.get("status") == "error":
                raise _CandidateExecutionFailure(
                    "execution-error",
                    f"{measured.get('error_type', 'unknown')}: "
                    f"{measured.get('message', '')}",
                )
            return _validated_measurement_result(
                measured, setup_wall_us, execution_wall_us
            )
        except BaseException:
            self._discard()
            raise

    def close(self) -> None:
        if self._process is None or self._connection is None:
            return
        try:
            self._connection.send("stop")
        except (BrokenPipeError, EOFError, OSError):
            pass
        self._discard()


class MeasurementBackend:
    """Keep device measurement on the consumer thread."""

    def __init__(self):
        self._session: _PersistentMeasurementSession | None = None

    @contextmanager
    def scope(self):
        if self._session is not None:
            raise RuntimeError("persistent measurement scopes cannot nest")
        try:
            yield
        finally:
            if self._session is not None:
                self._session.close()
                self._session = None

    def measure_kernel(
        self,
        kernel: Any,
        grid: Sequence[int],
        replay: _ScratchReplay,
    ):
        if self._session is None:
            self._session = _PersistentMeasurementSession()
        return self._session.measure(kernel, grid, replay)

    def measure(
        self,
        controller: CompilerLayoutAutotuner,
        outcomes: _DomainMeasurements,
        manifest: LayoutManifest,
        candidate: Candidate,
        compiled: GluonLayoutCompiledVariant,
        grid: Sequence[int],
        stream: Any,
        replay: _ScratchReplay,
    ) -> None:
        controller._measure_compiled_candidate(
            outcomes,
            manifest,
            candidate.specification,
            compiled,
            candidate.order_index,
            grid,
            stream,
            replay,
        )


class SerialMeasurementConsumer:
    """Consume only completed batches and never overlap device measurements."""

    def __init__(self, backend: MeasurementBackend | None = None):
        self.backend = backend or MeasurementBackend()

    def consume_candidate(
        self,
        controller: CompilerLayoutAutotuner,
        outcomes: _DomainMeasurements,
        manifest: LayoutManifest,
        candidate: Candidate,
        compiled: GluonLayoutCompiledVariant,
        grid: Sequence[int],
        stream: Any,
        replay: _ScratchReplay,
    ) -> None:
        self.backend.measure(
            controller,
            outcomes,
            manifest,
            candidate,
            compiled,
            grid,
            stream,
            replay,
        )


class SelectionPolicy:
    """Named boundary for the controller's existing stable-winner policy."""

    def remeasure(self, controller: CompilerLayoutAutotuner, *args, **kwargs):
        return controller._remeasure_close_variants(*args, **kwargs)


class RecordStore:
    """Named boundary for the controller's existing exact-identity records."""

    def read(self, controller: CompilerLayoutAutotuner, *args, **kwargs):
        return controller._read_cache(*args, **kwargs)

    def write(self, controller: CompilerLayoutAutotuner, *args, **kwargs):
        return controller._write_cache(*args, **kwargs)


class ParallelCompileAutotuner(CompilerLayoutAutotuner, Autotuner):
    """Native Triton autotuner plus closed-domain parallel layout compilation."""

    def __init__(
        self,
        fn,
        arg_names=None,
        configs=None,
        key=None,
        reset_to_zero=None,
        restore_value=None,
        pre_hook=None,
        post_hook=None,
        prune_configs_by=None,
        warmup=None,
        rep=None,
        use_cuda_graph=False,
        do_bench=None,
        cache_results=False,
        *,
        benchmarker=None,
        cache_manager_factory=None,
        measurer=None,
        compile_batch_size: int = _DEFAULT_BATCH_SIZE,
        compile_workers: int = _DEFAULT_COMPILE_WORKERS,
        ready_batches: int = _DEFAULT_READY_BATCHES,
        compile_backend: IsolatedCompileBackend | None = None,
        measurement_backend: MeasurementBackend | None = None,
        selection_policy: SelectionPolicy | None = None,
        record_store: RecordStore | None = None,
    ):
        Autotuner.__init__(
            self,
            fn,
            tuple(arg_names if arg_names is not None else getattr(fn, "arg_names", ())),
            [] if configs is None else configs,
            [] if key is None else key,
            reset_to_zero,
            restore_value,
            pre_hook,
            post_hook,
            prune_configs_by,
            warmup,
            rep,
            use_cuda_graph,
            do_bench,
            cache_results,
        )
        CompilerLayoutAutotuner.__init__(
            self,
            benchmarker=benchmarker,
            cache_manager_factory=cache_manager_factory,
            measurer=measurer,
        )
        self._compile_scheduler = BatchCompileScheduler(
            batch_size=compile_batch_size,
            max_workers=compile_workers,
            ready_batches=ready_batches,
            backend=compile_backend,
        )
        self._measurement_backend = measurement_backend or MeasurementBackend()
        self._measurement_consumer = SerialMeasurementConsumer(
            self._measurement_backend
        )
        self._selection_policy = selection_policy or SelectionPolicy()
        self._record_store = record_store or RecordStore()

    def _prepare_bundle(self, *args, **kwargs):
        # The scope is lazy: memory and persistent winner hits never start a
        # measurement process.
        with self._measurement_backend.scope():
            return super()._prepare_bundle(*args, **kwargs)

    def _measure(
        self,
        kernel: Any,
        grid: Sequence[int],
        stream: Any,
        replay: _ScratchReplay,
    ):
        if self._measurer is not None or self._benchmarker is not None:
            return super()._measure(kernel, grid, stream, replay)
        return self._measurement_backend.measure_kernel(kernel, grid, replay)

    def _evaluate_domain(
        self,
        bundle: GluonLayoutVariantBundle,
        manifest: LayoutManifest,
        grid: Sequence[int],
        stream: Any,
        replay: _ScratchReplay,
    ) -> _DomainMeasurements:
        domain = CandidateDomain.from_manifest(manifest)
        outcomes = _DomainMeasurements()
        batches = self._compile_scheduler.stream(bundle, domain)
        try:
            # Starting the producer before consuming the already-compiled
            # fallback overlaps nonfallback compilation with fallback
            # measurement.
            try:
                fallback = bundle.get_or_compile(
                    domain.fallback.specification
                )
            except Exception as error:
                self._record_candidate_failure(
                    outcomes, manifest, domain.fallback.specification, error
                )
            else:
                self._measurement_consumer.consume_candidate(
                    self,
                    outcomes,
                    manifest,
                    domain.fallback,
                    fallback,
                    grid,
                    stream,
                    replay,
                )

            for batch in batches:
                _LOGGER.debug(
                    "consuming complete Gluon layout compile batch "
                    "index=%d size=%d",
                    batch.batch_index,
                    len(batch.results),
                )
                for result in batch.results:
                    if result.error is not None:
                        self._record_candidate_failure(
                            outcomes,
                            manifest,
                            result.candidate.specification,
                            result.error,
                        )
                        continue
                    assert result.compiled is not None
                    self._measurement_consumer.consume_candidate(
                        self,
                        outcomes,
                        manifest,
                        result.candidate,
                        result.compiled,
                        grid,
                        stream,
                        replay,
                    )
        finally:
            close = getattr(batches, "close", None)
            if close is not None:
                close()
        return outcomes


def _create_parallel_compile_autotuner(**kwargs):
    return ParallelCompileAutotuner(**kwargs)


register_layout_autotuner_factory(
    "parallel-compile", _create_parallel_compile_autotuner
)


__all__ = [
    "BatchCompileScheduler",
    "CandidateDomain",
    "CompiledBatch",
    "IsolatedCompileBackend",
    "MeasurementBackend",
    "ParallelCompileAutotuner",
    "RecordStore",
    "SelectionPolicy",
    "SerialMeasurementConsumer",
]
