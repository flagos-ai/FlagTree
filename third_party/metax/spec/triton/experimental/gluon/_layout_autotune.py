from __future__ import annotations

import hashlib
import json
import logging
import math
import multiprocessing
import re
import statistics
import threading
import time
import warnings
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Sequence


_MANIFEST_VERSION = 5
_RUNTIME_CONTRACT_VERSION = 2
_MEASUREMENT_PROTOCOL_VERSION = 8
_CACHE_FILENAME = "gluon-layout-autotune.json"
_INTERLEAVED_REMEASUREMENTS = 2
_AUTOTUNE_WARMUP_LAUNCHES = 5
_AUTOTUNE_REPEAT_LAUNCHES = 5
_AUTOTUNE_WORKER_SETUP_TIMEOUT_SECONDS = 60
_AUTOTUNE_CANDIDATE_COMPILE_TIMEOUT_SECONDS = 300
_AUTOTUNE_CANDIDATE_TIMEOUT_SECONDS = 60
_MEASUREMENT_PROTOCOL = {
    "version": _MEASUREMENT_PROTOCOL_VERSION,
    "timer": "device-event-isolated-v1",
    "executor": "forkserver-parallel-compile-persistent-serial-measure-v1",
    "cache_policy": "restore-scratch-then-clear-before-each-warmup-and-sample",
    "quantiles": [0.5, 0.2, 0.8],
    "warmup_launches": _AUTOTUNE_WARMUP_LAUNCHES,
    "repeat_launches": _AUTOTUNE_REPEAT_LAUNCHES,
    "worker_setup_timeout_seconds": _AUTOTUNE_WORKER_SETUP_TIMEOUT_SECONDS,
    "candidate_compile_timeout_seconds": _AUTOTUNE_CANDIDATE_COMPILE_TIMEOUT_SECONDS,
    "candidate_timeout_seconds": _AUTOTUNE_CANDIDATE_TIMEOUT_SECONDS,
}

_LOGGER = logging.getLogger(__name__)


class LayoutAutotuneError(RuntimeError):
    """Raised when runtime layout tuning cannot be proven safe."""


class FallbackVariantExecutionError(RuntimeError):
    """Raised when the mandatory compiler fallback cannot be benchmarked."""


class _CandidateExecutionFailure(RuntimeError):
    """One isolated candidate failed without invalidating the closed bundle."""

    def __init__(self, reason: str, detail: str):
        super().__init__(f"{reason}: {detail}")
        self.reason = reason
        self.detail = detail


class _CandidateCompileFailure(RuntimeError):
    """One closed-domain experiment failed standalone compilation."""

    def __init__(
        self,
        reason: str,
        detail: str,
        compile_metrics: Mapping[str, Any] | None = None,
    ):
        super().__init__(f"{reason}: {detail}")
        self.reason = reason
        self.detail = detail
        self.compile_metrics = compile_metrics


@dataclass(frozen=True)
class _CandidateFailureRecord:
    reason: str
    detail: str


@dataclass(frozen=True)
class LayoutVariant:
    digest: str
    mlir_file: str
    mlir_sha256: str
    stage: str


@dataclass(frozen=True)
class LayoutManifest:
    digest: str
    fallback: LayoutVariant
    fallback_only: bool
    variants: tuple[LayoutVariant, ...]


@dataclass(frozen=True)
class RuntimeContract:
    tensor_args: tuple[int, ...]
    read_args: tuple[int, ...]
    written_args: tuple[int, ...]
    write_only_args: tuple[int, ...]
    atomic_args: tuple[int, ...]
    require_noalias: tuple[tuple[int, int], ...]
    deterministic: bool


@dataclass(frozen=True)
class _ScratchReplay:
    args: tuple[Any, ...]
    reset: Callable[[], None]
    reset_pairs: tuple[tuple[Any, Any], ...]


@dataclass(frozen=True)
class _KernelReloadDescriptor:
    metadata_group: tuple[tuple[str, str], ...]
    compilation_hash: str
    argument_names: tuple[str, ...]
    constexprs: tuple[Any, ...]
    signature: tuple[tuple[str, Any], ...]
    constant_paths: tuple[tuple[int, ...], ...]


@dataclass(frozen=True)
class _LauncherABIDescriptor:
    argument_names: tuple[str, ...]
    constexprs: tuple[Any, ...]
    signature: tuple[tuple[str, Any], ...]
    constant_paths: tuple[tuple[int, ...], ...]


@dataclass(frozen=True)
class _IsolatedCompileRequest:
    source_path: str
    target: Any
    options: Mapping[str, Any]
    env_vars: Mapping[str, Any] | None
    domain_digest: str
    variant_digest: str
    runtime_contract: Mapping[str, Any]
    launcher: _LauncherABIDescriptor


@dataclass(frozen=True)
class _CompiledArtifact:
    reload: _KernelReloadDescriptor
    device_signature: tuple[str, ...]
    compile_metrics: Mapping[str, Any]


@dataclass(frozen=True)
class _IsolatedMeasurementRequest:
    reload: _KernelReloadDescriptor
    grid: tuple[int, int, int]
    device: int
    args: tuple[Any, ...]
    reset_pairs: tuple[tuple[Any, Any], ...]


@dataclass(frozen=True)
class _MeasurementResult:
    timing: tuple[float, float, float]
    resources: Mapping[str, int | None]
    device_samples_ms: tuple[float, ...] | None
    setup_wall_us: int | None
    execution_wall_us: int | None


@dataclass(frozen=True)
class _TimingResult:
    timing: tuple[float, float, float]
    device_samples_ms: tuple[float, ...] | None


@dataclass(frozen=True)
class _MeasurementRound:
    round_index: int
    order_index: int
    timing: tuple[float, float, float]
    device_samples_ms: tuple[float, ...] | None
    setup_wall_us: int | None
    execution_wall_us: int | None


@dataclass
class _DomainMeasurements:
    """Mutable outcomes for one closed, manifest-ordered candidate domain."""

    timings: dict[LayoutVariant, tuple[float, float, float]] = field(default_factory=dict)
    resources: dict[LayoutVariant, Mapping[str, int | None]] = field(default_factory=dict)
    failures: dict[LayoutVariant, str] = field(default_factory=dict)
    failure_details: dict[LayoutVariant, str] = field(default_factory=dict)
    rounds: dict[LayoutVariant, list[_MeasurementRound]] = field(default_factory=dict)


@dataclass(frozen=True)
class _StabilityResult:
    initial_winner: LayoutVariant
    close: tuple[LayoutVariant, ...]
    stable: frozenset[LayoutVariant]


@dataclass(frozen=True)
class _FastWinner:
    """Process-local winner guarded by exact bundle and launch identities."""

    digest: str
    compiled: GluonLayoutCompiledVariant


@dataclass(frozen=True)
class GluonLayoutCompiledVariant:
    digest: str
    mlir_file: str
    mlir_sha256: str
    device_signature: tuple[str, ...]
    kernel: Any


@dataclass(frozen=True)
class GluonLayoutVariantBundle:
    """Closed compiler-produced source domain for one specialization.

    Layout formulas remain compiler-owned.  The runtime only consumes opaque
    digest/file identities emitted by Transform. Incremental compilation may
    consume one exact source at a time, but cannot add a candidate, rerun
    candidate generation, or select by ordinal.
    """

    manifest: Mapping[str, Any]
    variants: Mapping[str, GluonLayoutCompiledVariant]
    runtime_contract: Mapping[str, Any] | None = None
    source_kernel: Any | None = None
    sources: Mapping[str, str] = field(default_factory=dict)
    compile_variant: Callable[[LayoutVariant], GluonLayoutCompiledVariant] | None = None
    compile_failures: dict[str, _CandidateFailureRecord] = field(default_factory=dict)
    compile_metrics: dict[str, Mapping[str, Any]] = field(default_factory=dict)
    compile_state_lock: Any = field(default_factory=threading.RLock, repr=False, compare=False)
    compile_locks: dict[str, Any] = field(default_factory=dict, repr=False, compare=False)

    @property
    def fallback_kernel(self):
        manifest = parse_layout_manifest(self.manifest)
        return self.variants[manifest.fallback.digest].kernel

    @property
    def specialization_kernel(self):
        """Outer AST compilation used only for ABI and specialization identity."""
        return self.source_kernel if self.source_kernel is not None else self.fallback_kernel

    def get_or_compile(self, specification: LayoutVariant) -> GluonLayoutCompiledVariant:
        with self.compile_state_lock:
            compiled = self.variants.get(specification.digest)
            if compiled is not None:
                return compiled
            previous_failure = self.compile_failures.get(specification.digest)
            if previous_failure is not None:
                raise _CandidateCompileFailure(
                    previous_failure.reason,
                    previous_failure.detail,
                    self.compile_metrics.get(specification.digest),
                )
            if self.compile_variant is None:
                raise _CandidateCompileFailure(
                    "compile-unavailable",
                    "closed source domain has no incremental compiler",
                )
            compile_lock = self.compile_locks.setdefault(
                specification.digest, threading.Lock()
            )

        # Compilation is isolated by candidate digest. Different candidates
        # may compile concurrently, while duplicate requests for one digest
        # share the exact same terminal result.
        with compile_lock:
            with self.compile_state_lock:
                compiled = self.variants.get(specification.digest)
                if compiled is not None:
                    return compiled
                previous_failure = self.compile_failures.get(
                    specification.digest
                )
                if previous_failure is not None:
                    raise _CandidateCompileFailure(
                        previous_failure.reason,
                        previous_failure.detail,
                        self.compile_metrics.get(specification.digest),
                    )
            try:
                compiled = self.compile_variant(specification)
            except _CandidateCompileFailure as error:
                with self.compile_state_lock:
                    self.compile_failures[
                        specification.digest
                    ] = _CandidateFailureRecord(
                        error.reason, _compact_candidate_error(error.detail)
                    )
                    if error.compile_metrics is not None:
                        self.compile_metrics[specification.digest] = dict(
                            error.compile_metrics
                        )
                raise
            with self.compile_state_lock:
                self.variants[specification.digest] = compiled
            return compiled


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _is_digest(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(char in "0123456789abcdef" for char in value)


def parse_layout_manifest(raw: Mapping[str, Any]) -> LayoutManifest:
    if (
        not isinstance(raw, Mapping)
        or type(raw.get("version")) is not int
        or raw.get("version") != _MANIFEST_VERSION
    ):
        raise LayoutAutotuneError("unsupported or missing Gluon layout manifest version")
    if set(raw) != {"version", "digest", "fallback", "fallback_only", "variants"}:
        raise LayoutAutotuneError("Gluon layout manifest has unknown or missing top-level fields")
    digest = raw.get("digest")
    fallback_digest = raw.get("fallback")
    fallback_only = raw.get("fallback_only")
    variants_raw = raw.get("variants")
    if (
        not _is_digest(digest)
        or not _is_digest(fallback_digest)
        or not isinstance(fallback_only, bool)
        or not isinstance(variants_raw, list)
        or not variants_raw
    ):
        raise LayoutAutotuneError("malformed Gluon layout manifest")

    variants = []
    digests = set()
    files = set()
    for variant in variants_raw:
        if not isinstance(variant, Mapping):
            raise LayoutAutotuneError("layout variant entry must be a mapping")
        if set(variant) != {"digest", "mlir_file", "mlir_sha256", "stage"}:
            raise LayoutAutotuneError("layout variant has unknown or missing fields")
        variant_digest = variant.get("digest")
        mlir_file = variant.get("mlir_file")
        mlir_sha256 = variant.get("mlir_sha256")
        stage = variant.get("stage")
        if (
            not _is_digest(variant_digest)
            or not isinstance(mlir_file, str)
            or not mlir_file
            or mlir_file != f"{variant_digest}.ttgir"
            or not _is_digest(mlir_sha256)
            or stage != "final-ttgir"
        ):
            raise LayoutAutotuneError("layout variant has an invalid digest, MLIR file, or MLIR SHA")
        if variant_digest in digests or mlir_file in files:
            raise LayoutAutotuneError("layout variant digests and MLIR files must be unique")
        digests.add(variant_digest)
        files.add(mlir_file)
        variants.append(LayoutVariant(variant_digest, mlir_file, mlir_sha256, stage))

    fallback = variants[0]
    if fallback.digest != fallback_digest:
        raise LayoutAutotuneError(
            "the first layout manifest variant must be the unique fallback"
        )
    if any(variant.stage != "final-ttgir" for variant in variants):
        raise LayoutAutotuneError("layout manifest has inconsistent finalization stages")
    if fallback_only != (len(variants) == 1):
        raise LayoutAutotuneError("layout manifest fallback_only does not match its closed variant set")
    return LayoutManifest(digest, fallback, fallback_only, tuple(variants))


def parse_runtime_contract(raw: Mapping[str, Any] | None, num_args: int) -> RuntimeContract:
    if not isinstance(raw, Mapping) or raw.get("version") != _RUNTIME_CONTRACT_VERSION:
        raise LayoutAutotuneError("compiler did not provide a supported runtime validation contract")
    if raw.get("argument_index_space") != "lowered-tt-func-runtime-abi":
        raise LayoutAutotuneError("runtime validation contract uses an unknown argument index space")
    if raw.get("replayable") is not True:
        raise LayoutAutotuneError("compiler could not prove a runtime replay contract")
    expected_fields = {
        "version", "argument_index_space", "replayable", "deterministic",
        "tensor_args", "read_args", "written_args", "write_only_args",
        "atomic_args", "required_noalias",
    }
    if set(raw) != expected_fields:
        raise LayoutAutotuneError("runtime validation contract has unknown or missing fields")

    def parse_indices(name: str) -> tuple[int, ...]:
        values = raw.get(name)
        if not isinstance(values, list) or any(not isinstance(value, int) for value in values):
            raise LayoutAutotuneError(f"runtime validation contract has invalid {name}")
        result = tuple(sorted(set(values)))
        if len(result) != len(values) or any(value < 0 or value >= num_args for value in result):
            raise LayoutAutotuneError(f"runtime validation contract has out-of-range or duplicate {name}")
        return result

    tensor_args = parse_indices("tensor_args")
    read_args = parse_indices("read_args")
    written_args = parse_indices("written_args")
    write_only_args = parse_indices("write_only_args")
    atomic_args = parse_indices("atomic_args")
    deterministic = raw.get("deterministic")
    noalias_raw = raw.get("required_noalias")
    if not isinstance(deterministic, bool) or not isinstance(noalias_raw, list):
        raise LayoutAutotuneError("runtime validation contract must state determinism and no-alias pairs")
    tensor_arg_set = set(tensor_args)
    if any(
        not set(indices).issubset(tensor_arg_set)
        for indices in (read_args, written_args, write_only_args, atomic_args)
    ):
        raise LayoutAutotuneError("every memory effect must refer to a tensor argument")
    read_arg_set = set(read_args)
    written_arg_set = set(written_args)
    atomic_arg_set = set(atomic_args)
    if set(write_only_args) != written_arg_set - read_arg_set:
        raise LayoutAutotuneError("write-only arguments must equal written arguments that are not read")
    if not atomic_arg_set.issubset(read_arg_set & written_arg_set):
        raise LayoutAutotuneError("atomic arguments must be published as both read and written")
    if deterministic != (not atomic_args):
        raise LayoutAutotuneError("runtime validation contract has inconsistent atomic determinism")

    noalias = []
    seen_pairs = set()
    for pair in noalias_raw:
        if not isinstance(pair, list) or len(pair) != 2 or any(not isinstance(index, int) for index in pair):
            raise LayoutAutotuneError("runtime validation contract has an invalid no-alias pair")
        lhs, rhs = sorted(pair)
        if lhs == rhs or lhs < 0 or rhs >= num_args or lhs not in tensor_args or rhs not in tensor_args:
            raise LayoutAutotuneError("runtime validation contract has an out-of-range no-alias pair")
        if (lhs, rhs) in seen_pairs:
            raise LayoutAutotuneError("runtime validation contract has a duplicate no-alias pair")
        seen_pairs.add((lhs, rhs))
        noalias.append((lhs, rhs))
    return RuntimeContract(
        tensor_args, read_args, written_args, write_only_args, atomic_args, tuple(sorted(noalias)), deterministic,
    )


def _pointer_alignment(pointer: int) -> int:
    if pointer == 0:
        return 0
    return min(pointer & -pointer, 4096)


def _runtime_argument_signature(value: Any) -> Mapping[str, Any]:
    if isinstance(value, bool):
        return {"kind": "bool", "value": value}
    if isinstance(value, int):
        return {"kind": "int", "value": value}
    if isinstance(value, float):
        if not math.isfinite(value):
            raise LayoutAutotuneError("non-finite runtime scalars are not supported by layout autotuning")
        return {"kind": "float", "value": value.hex()}
    if value is None:
        return {"kind": "none"}

    data_ptr = getattr(value, "data_ptr", None)
    shape = getattr(value, "shape", None)
    stride = getattr(value, "stride", None)
    dtype = getattr(value, "dtype", None)
    if not callable(data_ptr) or shape is None or not callable(stride) or dtype is None:
        raise LayoutAutotuneError(f"unsupported runtime argument type: {type(value).__qualname__}")
    return {
        "kind": "tensor",
        "type": type(value).__qualname__,
        "dtype": str(dtype),
        "device": str(getattr(value, "device", "unknown")),
        "shape": [int(dim) for dim in shape],
        "stride": [int(item) for item in stride()],
        "alignment": _pointer_alignment(int(data_ptr())),
    }


def _normalize_grid(grid: Sequence[int]) -> tuple[int, int, int]:
    if not 1 <= len(grid) <= 3:
        raise LayoutAutotuneError("Gluon layout autotuning requires a one-, two-, or three-dimensional grid")
    result = tuple(int(dim) for dim in grid)
    if any(dim <= 0 for dim in result):
        raise LayoutAutotuneError("Gluon layout autotuning requires positive grid dimensions")
    return result + (1,) * (3 - len(result))


def _fast_scalar_identity(value: Any) -> tuple[Any, ...] | None:
    """Return an exact, hashable scalar identity without serialization."""

    if isinstance(value, bool):
        return ("bool", value)
    if isinstance(value, int):
        return ("int", value)
    if isinstance(value, float):
        return ("float", value.hex()) if math.isfinite(value) else None
    if isinstance(value, str):
        return ("str", value)
    if isinstance(value, bytes):
        return ("bytes", value)
    if value is None:
        return ("none", )
    return None


def _fast_runtime_argument_identity(value: Any) -> tuple[Any, ...] | None:
    """Build an over-specific tensor identity for the process-local fast path.

    The persistent workload key intentionally ignores allocation identity so a
    measured winner can be reused by equivalent future tensors. The launch
    fast path has a different contract: it may bypass every validator only for
    the exact live tensor object and physical view that was validated before.
    Pointer, shape, stride, element width, dtype, and device are a complete
    description of the physical view used by the persistent key.  Alignment
    and byte intervals are deterministic functions of those fields, so do not
    recompute them on every launch: doing so would put avoidable Python work
    between a caller's device events.
    """

    scalar = _fast_scalar_identity(value)
    if scalar is not None:
        return scalar

    data_ptr = getattr(value, "data_ptr", None)
    shape = getattr(value, "shape", None)
    stride = getattr(value, "stride", None)
    dtype = getattr(value, "dtype", None)
    element_size = getattr(value, "element_size", None)
    if (
        not callable(data_ptr)
        or shape is None
        or not callable(stride)
        or dtype is None
        or not callable(element_size)
    ):
        return None
    try:
        pointer = int(data_ptr())
        dimensions = tuple(int(dim) for dim in shape)
        strides = tuple(int(item) for item in stride())
        element_bytes = int(element_size())
    except (TypeError, ValueError, OverflowError, AttributeError, RuntimeError):
        return None
    if (
        element_bytes <= 0
        or len(dimensions) != len(strides)
        or any(dim < 0 for dim in dimensions)
    ):
        return None
    device = getattr(value, "device", "unknown")
    try:
        hash(dtype)
        hash(device)
    except TypeError:
        return None
    return (
        "tensor",
        id(value),
        type(value),
        pointer,
        dimensions,
        strides,
        element_bytes,
        dtype,
        device,
    )


def _fast_launch_identity(
    grid: Sequence[int], stream: Any, bound_args: Mapping[str, Any],
) -> tuple[Any, ...] | None:
    """Return a cheap exact launch key, or disable the fast path conservatively."""

    try:
        normalized_grid = _normalize_grid(grid)
    except (LayoutAutotuneError, TypeError, ValueError, OverflowError):
        return None
    stream_identity = _fast_scalar_identity(stream)
    if stream_identity is None:
        stream_identity = ("object", id(stream))

    arguments = []
    for name, value in bound_args.items():
        identity = _fast_runtime_argument_identity(value)
        if identity is None:
            return None
        arguments.append((str(name), identity))
    return normalized_grid, stream_identity, tuple(arguments)


def build_workload_key(
    grid: Sequence[int],
    bound_args: Mapping[str, Any],
    *,
    source_identity: Mapping[str, Any],
    target_identity: Mapping[str, Any],
    device_identity: Mapping[str, Any],
    domain_digest: str,
    effect_digest: str,
) -> tuple[str, Mapping[str, Any]]:
    if any(not _is_digest(digest) for digest in (domain_digest, effect_digest)):
        raise LayoutAutotuneError("workload identity requires valid domain and effect digests")
    arguments = [
        {"name": str(name), "signature": _runtime_argument_signature(value)}
        for name, value in bound_args.items()
    ]
    payload = {
        "measurement": _MEASUREMENT_PROTOCOL,
        "payload": source_identity,
        "target": target_identity,
        "device": device_identity,
        "domain": domain_digest,
        "effect": effect_digest,
        "grid": list(_normalize_grid(grid)),
        "arguments": arguments,
        "alias_classes": _tensor_alias_classes(tuple(bound_args.values()), arguments),
    }
    return _sha256_json(payload), payload


def select_measured_variant(
    timings: Mapping[LayoutVariant, Sequence[float]], fallback: LayoutVariant,
) -> LayoutVariant:
    fallback_timing = timings.get(fallback)
    if not fallback_timing or not math.isfinite(float(fallback_timing[0])):
        raise LayoutAutotuneError("fallback candidate did not produce a finite median")
    fallback_median = float(fallback_timing[0])
    eligible = [candidate for candidate, timing in timings.items()
                if timing and math.isfinite(float(timing[0]))]
    if not eligible:
        return fallback
    winner = min(
        eligible,
        key=lambda variant: (float(timings[variant][0]), variant.digest, variant.mlir_file),
    )
    # The 3% allowance belongs to end-to-end regression acceptance, not to
    # candidate selection.  A candidate must beat the fixed compiler fallback;
    # close/noisy results are interleaved and remeasured before this decision.
    return winner if float(timings[winner][0]) < fallback_median else fallback


def _compiled_binary_identity(kernel: Any) -> Mapping[str, str]:
    binary = getattr(kernel, "kernel", None)
    if not isinstance(binary, (bytes, bytearray, memoryview)):
        raise LayoutAutotuneError("compiled candidate does not expose a stable device binary")

    # ``CompiledKernel.hash`` identifies one compiler-cache instance.  It may
    # change when TRITON_ALWAYS_COMPILE rebuilds byte-identical executables, so
    # including it would turn a persistent winner into a per-process cache.
    # The executable content hash is stable across such rebuilds.  Staleness is
    # still guarded by the source, target, layout-domain/TTGIR identities and
    # launch resources carried alongside this value.
    return {"binary_sha256": hashlib.sha256(bytes(binary)).hexdigest()}


def _source_identity(kernel: Any) -> Mapping[str, Any]:
    source = getattr(kernel, "src", None)
    source_hash = getattr(source, "hash", None)
    if not callable(source_hash):
        raise LayoutAutotuneError("compiled fallback does not expose its source identity")
    value = source_hash()
    if not isinstance(value, str) or not value:
        raise LayoutAutotuneError("compiled fallback has an invalid source identity")
    return {"source_hash": value, "fallback_binary": _compiled_binary_identity(kernel)}


def _target_identity(kernel: Any) -> Mapping[str, Any]:
    metadata = getattr(kernel, "metadata", None)
    target = getattr(metadata, "target", None)
    if target is None:
        raise LayoutAutotuneError("compiled fallback does not expose its target identity")
    if hasattr(target, "_asdict"):
        target = target._asdict()
    elif not isinstance(target, Mapping):
        target = {
            "backend": getattr(target, "backend", None),
            "arch": getattr(target, "arch", None),
            "warp_size": getattr(target, "warp_size", None),
        }
    result = {str(name): _make_json_value(value) for name, value in target.items()}
    if not result.get("backend") or result.get("arch") is None:
        raise LayoutAutotuneError("compiled fallback has an incomplete target identity")
    return result


def _make_json_value(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Mapping):
        return {str(name): _make_json_value(item) for name, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_make_json_value(item) for item in value]
    return str(value)


def _device_identity_from_active_driver(active: Any) -> Mapping[str, Any]:
    device = active.get_current_device()
    target = active.get_current_target()
    if hasattr(target, "_asdict"):
        target = target._asdict()
    elif not isinstance(target, Mapping):
        target = {
            "backend": getattr(target, "backend", None),
            "arch": getattr(target, "arch", None),
            "warp_size": getattr(target, "warp_size", None),
        }
    utils = getattr(active, "utils", None)
    get_properties = getattr(utils, "get_device_properties", None)
    if not callable(get_properties):
        raise LayoutAutotuneError("active backend does not expose stable device properties")
    properties = get_properties(device)
    if not isinstance(properties, Mapping) or not properties:
        raise LayoutAutotuneError("active backend returned an invalid device property set")

    framework_properties = {}
    device_interface = active.get_device_interface()
    get_framework_properties = getattr(device_interface, "get_device_properties", None)
    if callable(get_framework_properties):
        framework = get_framework_properties(device)
        for name in ("name", "uuid", "total_memory", "major", "minor", "multi_processor_count"):
            if hasattr(framework, name):
                framework_properties[name] = _make_json_value(getattr(framework, name))

    return {
        "driver": f"{type(active).__module__}.{type(active).__qualname__}",
        "device_index": _make_json_value(device),
        "target": {str(name): _make_json_value(value) for name, value in target.items()},
        "properties": {str(name): _make_json_value(value) for name, value in properties.items()},
        "framework_properties": framework_properties,
    }


def _device_identity() -> Mapping[str, Any]:
    from triton.runtime.driver import driver

    return _device_identity_from_active_driver(driver.active)


def bind_runtime_arguments(kernel: Any, bound_args: Mapping[str, Any]) -> tuple[tuple[str, Any], ...]:
    """Map MLIR entry arguments back to concrete Python launch arguments.

    The compiler effect contract numbers arguments after constexpr erasure.  A
    positional walk over ``bound_args`` is therefore incorrect.  The initial
    AST source records both the original parameter order and the constant
    paths, which is sufficient for ordinary scalar/pointer kernel signatures.
    Aggregate and descriptor arguments may flatten to multiple MLIR arguments;
    until the compiler publishes their leaf mapping, reject them conservatively
    instead of guessing.
    """

    source = getattr(kernel, "src", None)
    fn = getattr(source, "fn", None)
    arg_names = getattr(fn, "arg_names", None)
    signature = getattr(source, "signature", None)
    constants = getattr(source, "constants", None)
    if not isinstance(arg_names, list) or not isinstance(signature, Mapping) or not isinstance(constants, Mapping):
        raise LayoutAutotuneError("compiled fallback does not expose its runtime argument mapping")

    constant_paths = set(constants)
    if any(not isinstance(path, tuple) or not path or not isinstance(path[0], int) for path in constant_paths):
        raise LayoutAutotuneError("compiled fallback has malformed constexpr paths")

    result = []
    for index, name in enumerate(arg_names):
        paths = [path for path in constant_paths if path[0] == index]
        if any(len(path) != 1 for path in paths):
            raise LayoutAutotuneError("aggregate constexpr arguments are not supported by layout autotuning")
        if (index, ) in constant_paths:
            continue
        if name not in signature or name not in bound_args:
            raise LayoutAutotuneError(f"runtime argument {name!r} is missing from the compiled specialization")
        type_spec = signature[name]
        if isinstance(type_spec, tuple) or not isinstance(type_spec, str) or type_spec.startswith("tensordesc"):
            raise LayoutAutotuneError("flattened aggregate or descriptor arguments require compiler leaf metadata")
        if type_spec.startswith("constexpr"):
            raise LayoutAutotuneError("constexpr argument is missing from the source constant map")
        result.append((name, bound_args[name]))
    return tuple(result)


def _bind_launch_arguments(kernel: Any, bound_args: Mapping[str, Any]) -> tuple[tuple[str, Any], ...]:
    source = getattr(kernel, "src", None)
    fn = getattr(source, "fn", None)
    arg_names = getattr(fn, "arg_names", None)
    if not isinstance(arg_names, list) or any(name not in bound_args for name in arg_names):
        raise LayoutAutotuneError("compiled fallback does not expose its complete launch argument mapping")
    return tuple((name, bound_args[name]) for name in arg_names)


def _remap_runtime_contract(
    contract: RuntimeContract,
    runtime_arguments: Sequence[tuple[str, Any]],
    launch_arguments: Sequence[tuple[str, Any]],
) -> RuntimeContract:
    launch_positions = {name: index for index, (name, _) in enumerate(launch_arguments)}
    if len(launch_positions) != len(launch_arguments):
        raise LayoutAutotuneError("kernel launch argument names must be unique")
    runtime_to_launch = []
    for name, _ in runtime_arguments:
        if name not in launch_positions:
            raise LayoutAutotuneError(f"runtime argument {name!r} has no launch argument")
        runtime_to_launch.append(launch_positions[name])

    def remap(indices: Sequence[int]) -> tuple[int, ...]:
        return tuple(runtime_to_launch[index] for index in indices)

    return RuntimeContract(
        remap(contract.tensor_args),
        remap(contract.read_args),
        remap(contract.written_args),
        remap(contract.write_only_args),
        remap(contract.atomic_args),
        tuple(sorted((runtime_to_launch[lhs], runtime_to_launch[rhs]) for lhs, rhs in contract.require_noalias)),
        contract.deterministic,
    )


def _runtime_contract_digest(contract: Mapping[str, Any] | None) -> str:
    if not isinstance(contract, Mapping):
        raise LayoutAutotuneError("compiler did not provide a runtime validation contract")
    return _sha256_json(contract)


def _kernel_resources(kernel: Any) -> Mapping[str, int | None]:
    metadata = getattr(kernel, "metadata", None)
    shared = int(getattr(metadata, "shared", 0)) if metadata is not None else None
    shared_tier = None
    if shared is not None:
        shared_tier = next((tier for tier in (16, 32, 64) if shared <= tier * 1024), None)
    return {
        "shared": shared,
        "shared_residency_tier_kib": shared_tier,
        "registers": int(kernel.n_regs) if hasattr(kernel, "n_regs") else None,
        # MetaX reports this field as private-memory words per thread.  It is
        # not a count of generated spill load/store instructions.
        "private_words32": int(kernel.n_spills) if hasattr(kernel, "n_spills") else None,
    }


def _variant_executable_identity(compiled: GluonLayoutCompiledVariant) -> Mapping[str, Any]:
    metadata = getattr(compiled.kernel, "metadata", None)
    shared = getattr(metadata, "shared", None)
    if not isinstance(shared, int) or shared < 0:
        raise LayoutAutotuneError(f"layout variant {compiled.digest} has no valid shared-memory footprint")
    launch_resources = {"shared": shared}
    for name in ("num_warps", "num_ctas", "num_stages", "cluster_dims"):
        value = getattr(metadata, name, None)
        if value is not None:
            launch_resources[name] = _make_json_value(value)
    return {
        "digest": compiled.digest,
        "mlir_sha256": compiled.mlir_sha256,
        "device_signature": list(compiled.device_signature),
        "binary": _compiled_binary_identity(compiled.kernel),
        "launch_resources": launch_resources,
    }


def _bundle_executable_identity(
    manifest: LayoutManifest, bundle: GluonLayoutVariantBundle,
) -> list[Mapping[str, Any]]:
    return [
        _variant_executable_identity(bundle.variants[variant.digest])
        for variant in manifest.variants
        if variant.digest in bundle.variants
    ]


def _source_domain_identity(manifest: LayoutManifest) -> list[Mapping[str, Any]]:
    """Return a stable identity without requiring experimental binaries."""

    return [
        {
            "digest": variant.digest,
            "mlir_file": variant.mlir_file,
            "mlir_sha256": variant.mlir_sha256,
            "stage": variant.stage,
        }
        for variant in manifest.variants
    ]


def _tensor_byte_interval(value: Any) -> tuple[int, int]:
    shape = tuple(int(dim) for dim in value.shape)
    strides = tuple(int(stride) for stride in value.stride())
    if len(shape) != len(strides) or any(dim < 0 for dim in shape):
        raise LayoutAutotuneError("cannot determine a tensor argument's storage interval")
    if any(dim == 0 for dim in shape):
        pointer = int(value.data_ptr())
        return pointer, pointer
    min_element = 0
    max_element = 0
    for dim, stride in zip(shape, strides):
        extent = (dim - 1) * stride
        min_element += min(0, extent)
        max_element += max(0, extent)
    element_size = int(value.element_size())
    pointer = int(value.data_ptr())
    return pointer + min_element * element_size, pointer + (max_element + 1) * element_size


def _tensor_alias_classes(
    args: Sequence[Any], argument_records: Sequence[Mapping[str, Any]],
) -> list[Mapping[str, Any]]:
    tensor_indices = [
        index for index, record in enumerate(argument_records)
        if record.get("signature", {}).get("kind") == "tensor"
    ]
    intervals = {index: _tensor_byte_interval(args[index]) for index in tensor_indices}
    adjacency = {index: set() for index in tensor_indices}
    overlaps = []
    for position, lhs in enumerate(tensor_indices):
        for rhs in tensor_indices[position + 1:]:
            lhs_interval = intervals[lhs]
            rhs_interval = intervals[rhs]
            if max(lhs_interval[0], rhs_interval[0]) >= min(lhs_interval[1], rhs_interval[1]):
                continue
            adjacency[lhs].add(rhs)
            adjacency[rhs].add(lhs)
            overlaps.append((lhs, rhs))

    classes = []
    visited = set()
    for root in tensor_indices:
        if root in visited:
            continue
        stack = [root]
        members = []
        while stack:
            current = stack.pop()
            if current in visited:
                continue
            visited.add(current)
            members.append(current)
            stack.extend(sorted(adjacency[current] - visited, reverse=True))
        members.sort()
        member_set = set(members)
        classes.append({
            "arguments": members,
            "overlaps": [[lhs, rhs] for lhs, rhs in overlaps if lhs in member_set and rhs in member_set],
        })
    return classes


def _check_required_noalias(args: Sequence[Any], pairs: Sequence[tuple[int, int]]) -> None:
    intervals = {}
    for lhs, rhs in pairs:
        lhs_interval = intervals.setdefault(lhs, _tensor_byte_interval(args[lhs]))
        rhs_interval = intervals.setdefault(rhs, _tensor_byte_interval(args[rhs]))
        if max(lhs_interval[0], rhs_interval[0]) < min(lhs_interval[1], rhs_interval[1]):
            raise LayoutAutotuneError(f"runtime arguments {lhs} and {rhs} violate a required no-alias contract")


def _allocate_scratch_argument(value: Any) -> Any:
    """Allocate an uninitialized tensor with the same logical strided view."""

    shape = getattr(value, "shape", None)
    stride = getattr(value, "stride", None)
    if shape is None or not callable(stride):
        raise LayoutAutotuneError("a written runtime argument is not a strided tensor")
    shape = tuple(int(dim) for dim in shape)
    strides = tuple(int(item) for item in stride())
    if len(shape) != len(strides) or any(dim < 0 for dim in shape) or any(item < 0 for item in strides):
        raise LayoutAutotuneError("scratch replay requires a non-negative strided tensor view")

    create = getattr(value, "new_empty_strided", None)
    if callable(create):
        scratch = create(shape, strides)
    else:
        # TensorWrapper intentionally exposes the wrapped framework tensor as
        # `base`. Reconstructing the same wrapper preserves its logical Triton
        # dtype while the underlying allocation preserves the physical view.
        base = getattr(value, "base", None)
        create_base = getattr(base, "new_empty_strided", None)
        if not callable(create_base):
            raise LayoutAutotuneError("a written tensor cannot allocate layout-preserving scratch")
        try:
            scratch = type(value)(create_base(shape, strides), value.dtype)
        except (TypeError, ValueError, AttributeError) as error:
            raise LayoutAutotuneError("a wrapped tensor cannot preserve its scratch view") from error

    scratch_stride = getattr(scratch, "stride", None)
    scratch_element_size = getattr(scratch, "element_size", None)
    scratch_data_ptr = getattr(scratch, "data_ptr", None)
    try:
        preserved = (
            scratch is not value
            and tuple(int(dim) for dim in scratch.shape) == shape
            and callable(scratch_stride)
            and tuple(int(item) for item in scratch_stride()) == strides
            and str(scratch.dtype) == str(value.dtype)
            and str(getattr(scratch, "device", "unknown")) == str(getattr(value, "device", "unknown"))
            and callable(scratch_element_size)
            and int(scratch_element_size()) == int(value.element_size())
            and callable(scratch_data_ptr)
        )
    except (TypeError, ValueError, OverflowError, AttributeError, RuntimeError):
        preserved = False
    if not preserved:
        raise LayoutAutotuneError("scratch allocation did not preserve the tensor view and device contract")
    if not callable(getattr(scratch, "copy_", None)):
        raise LayoutAutotuneError("scratch tensor does not support state restoration")
    return scratch


def _prepare_scratch_replay(args: Sequence[Any], contract: RuntimeContract) -> _ScratchReplay:
    """Redirect every external write and build an out-of-band reset action.

    Write-only buffers intentionally remain uninitialized. Read/write and
    atomic buffers are restored from the real user arguments before each raw
    launch; the benchmarker invokes this reset before recording its start
    event, so restoration is never charged to a layout candidate.
    """

    replay_args = list(args)
    originals = {}
    scratch = {}
    for index in contract.written_args:
        originals[index] = args[index]
        scratch[index] = _allocate_scratch_argument(args[index])
        replay_args[index] = scratch[index]

    # A framework allocator is expected to return fresh storage, but replay
    # safety must not depend on that convention. Prove that every live scratch
    # view is disjoint from all real tensor arguments and from every other
    # scratch view before launching even the fallback candidate.
    real_intervals = {
        index: _tensor_byte_interval(args[index])
        for index in contract.tensor_args
    }
    scratch_intervals = {}
    for index, value in scratch.items():
        interval = _tensor_byte_interval(value)
        if any(
            max(interval[0], real[0]) < min(interval[1], real[1])
            for real in real_intervals.values()
        ) or any(
            max(interval[0], other[0]) < min(interval[1], other[1])
            for other in scratch_intervals.values()
        ):
            raise LayoutAutotuneError("scratch replay allocation aliases a live runtime tensor")
        scratch_intervals[index] = interval

    write_only = set(contract.write_only_args)
    reset_indices = tuple(
        index for index in contract.written_args
        if index not in write_only
    )

    def reset() -> None:
        for index in reset_indices:
            try:
                scratch[index].copy_(originals[index])
            except (RuntimeError, TypeError, ValueError, AttributeError) as error:
                raise LayoutAutotuneError(
                    f"failed to restore scratch for runtime argument {index}"
                ) from error

    reset_pairs = tuple((scratch[index], originals[index]) for index in reset_indices)
    return _ScratchReplay(tuple(replay_args), reset, reset_pairs)


def _launch_compiled_kernel(kernel: Any, grid: Sequence[int], stream: Any, args: Sequence[Any]) -> None:
    # Accessing ``run`` initializes lazy driver handles before ``function`` is
    # read.  Invoke the raw launcher with the same complete host argument list
    # as generic JIT launch, but deliberately suppress launch metadata and
    # process-global enter/exit hooks so profiler configuration cannot bias the
    # layout winner.
    run = kernel.run
    run(
        grid[0],
        grid[1],
        grid[2],
        stream,
        kernel.function,
        kernel.packed_metadata,
        None,
        None,
        None,
        *args,
    )


def _device_event_benchmarker(
    fn: Callable, *, quantiles: Sequence[float], warmup: int, rep: int,
    before_each: Callable[[], None] | None = None,
) -> _TimingResult:
    """Benchmark exclusively with backend device events.

    This intentionally does not call ``triton.testing.do_bench``: that public
    helper may switch to profiler timing through process environment.  Layout
    winner cache entries must instead be produced by one explicit, versioned
    timing protocol.
    """

    from triton.runtime.driver import driver

    if warmup <= 0 or rep <= 0:
        raise LayoutAutotuneError("device-event benchmark requires positive launch counts")

    active = driver.active
    device_interface = active.get_device_interface()
    cache = active.get_empty_cache_for_benchmark()
    _LOGGER.debug(
        "running Gluon layout device-event benchmark warmup_launches=%d repeat_launches=%d",
        warmup,
        rep,
    )

    for _ in range(warmup):
        if before_each is not None:
            before_each()
        active.clear_cache(cache)
        fn()
    device_interface.synchronize()
    starts = [device_interface.Event(enable_timing=True) for _ in range(rep)]
    ends = [device_interface.Event(enable_timing=True) for _ in range(rep)]
    for sample_start, sample_end in zip(starts, ends):
        if before_each is not None:
            before_each()
        active.clear_cache(cache)
        sample_start.record()
        fn()
        sample_end.record()
    device_interface.synchronize()
    from triton.testing import _summarize_statistics

    times = tuple(
        float(sample_start.elapsed_time(sample_end))
        for sample_start, sample_end in zip(starts, ends)
    )
    timing = tuple(
        float(value)
        for value in _summarize_statistics(times, list(quantiles), "mean")
    )
    return _TimingResult(timing, times)


def _measure_variant(
    kernel: Any, grid: Sequence[int], stream: Any, args: Sequence[Any], benchmarker: Callable,
    before_each: Callable[[], None] | None = None,
) -> _TimingResult:
    benchmark_kwargs = {
        "quantiles": [0.5, 0.2, 0.8],
        "warmup": _AUTOTUNE_WARMUP_LAUNCHES,
        "rep": _AUTOTUNE_REPEAT_LAUNCHES,
    }
    if before_each is not None:
        benchmark_kwargs["before_each"] = before_each
    result = benchmarker(
        lambda: _launch_compiled_kernel(kernel, grid, stream, args),
        **benchmark_kwargs,
    )
    timing = result.timing if isinstance(result, _TimingResult) else result
    values = timing if isinstance(timing, Sequence) else [timing]
    if len(values) != 3 or any(not math.isfinite(float(value)) for value in values):
        raise LayoutAutotuneError("variant benchmark did not return finite median/p20/p80 values")
    normalized = tuple(float(value) for value in values)
    if not normalized[1] <= normalized[0] <= normalized[2]:
        raise LayoutAutotuneError("variant benchmark returned unordered median/p20/p80 values")
    if not isinstance(result, _TimingResult):
        return _TimingResult(normalized, None)
    samples = tuple(float(value) for value in result.device_samples_ms or ())
    if (
        len(samples) != _AUTOTUNE_REPEAT_LAUNCHES
        or any(not math.isfinite(value) or value < 0.0 for value in samples)
    ):
        raise LayoutAutotuneError("device-event benchmark returned invalid raw samples")
    from triton.testing import _summarize_statistics

    summarized = tuple(
        float(value)
        for value in _summarize_statistics(
            samples,
            list(_MEASUREMENT_PROTOCOL["quantiles"]),
            "mean",
        )
    )
    if not _timing_summaries_match(normalized, summarized):
        raise LayoutAutotuneError(
            "device-event timing does not summarize its raw samples"
        )
    return _TimingResult(normalized, samples)


def _make_launcher_abi_descriptor(kernel: Any) -> _LauncherABIDescriptor:
    """Capture the outer AST launch ABI without any executable state."""
    source = getattr(kernel, "src", None)
    function = getattr(source, "fn", None)
    argument_names = getattr(function, "arg_names", None)
    constexprs = getattr(function, "constexprs", None)
    signature = getattr(source, "signature", None)
    constants = getattr(source, "constants", None)
    if (
        not isinstance(argument_names, list)
        or not isinstance(constexprs, (list, tuple))
        or not isinstance(signature, Mapping)
        or not isinstance(constants, Mapping)
    ):
        raise LayoutAutotuneError("compiled specialization has no launcher ABI descriptor")
    constant_paths = tuple(constants)
    if any(
        not isinstance(path, tuple)
        or len(path) != 1
        or not isinstance(path[0], int)
        or path[0] < 0
        or path[0] >= len(argument_names)
        for path in constant_paths
    ):
        raise LayoutAutotuneError("isolated execution does not support aggregate constexpr arguments")
    if set(signature) != set(argument_names):
        raise LayoutAutotuneError("compiled variant launcher signature does not match its argument names")
    return _LauncherABIDescriptor(
        tuple(argument_names),
        tuple(constexprs),
        tuple((name, signature[name]) for name in argument_names),
        tuple(constant_paths),
    )


def _make_kernel_reload_descriptor(kernel: Any) -> _KernelReloadDescriptor:
    """Capture only cache identity and launcher ABI needed by a fresh worker."""

    launcher = _make_launcher_abi_descriptor(kernel)
    metadata_group = getattr(kernel, "metadata_group", None)
    compilation_hash = getattr(kernel, "hash", None)
    if (
        not isinstance(metadata_group, Mapping)
        or not isinstance(compilation_hash, str)
        or not compilation_hash
    ):
        raise LayoutAutotuneError("compiled variant has no reloadable cache descriptor")
    return _KernelReloadDescriptor(
        tuple(sorted((str(name), str(path)) for name, path in metadata_group.items())),
        compilation_hash,
        launcher.argument_names,
        launcher.constexprs,
        launcher.signature,
        launcher.constant_paths,
    )


def _compact_candidate_error(error: BaseException | str) -> str:
    """Return a byte-bounded diagnostic with a stable normalized-message id."""

    fallback = type(error).__name__ if isinstance(error, BaseException) else "error"
    message = " ".join(str(error).split()) or fallback
    encoded = message.encode("utf-8", errors="replace")
    if len(encoded) <= 1024:
        return message
    marker = re.search(
        r"\s*\.\.\. \[truncated normalized-message sha256=([0-9a-f]{64})\]",
        message,
    )
    if marker is None:
        digest = hashlib.sha256(encoded).hexdigest()
        prefix_text = message
    else:
        # Parent layers may add a phase or exception type around an already
        # bounded child diagnostic. Preserve the original complete-message id
        # instead of hashing a wrapper around a truncated prefix.
        digest = marker.group(1)
        prefix_text = message[:marker.start()].rstrip()
    suffix = f" ... [truncated normalized-message sha256={digest}]"
    prefix_budget = 1024 - len(suffix.encode("utf-8"))
    prefix = prefix_text.encode("utf-8", errors="replace")[:prefix_budget]
    return prefix.decode("utf-8", errors="ignore") + suffix


def _is_nonnegative_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _validate_compile_metrics(metrics: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the version-six compile-stage timing schema."""

    if not isinstance(metrics, Mapping):
        raise LayoutAutotuneError("compile metrics must be a mapping")
    mode = metrics.get("mode")
    outcome = metrics.get("outcome")
    if mode == "parent-fallback" and outcome == "success":
        if set(metrics) != {"mode", "outcome", "wall_us"} or not _is_nonnegative_int(
            metrics.get("wall_us")
        ):
            raise LayoutAutotuneError("malformed parent-fallback compile metrics")
        return dict(metrics)
    if mode != "isolated-candidate" or outcome not in {"success", "failure"}:
        raise LayoutAutotuneError("unknown compile metrics mode or outcome")
    if outcome == "failure":
        if set(metrics) != {"mode", "outcome", "isolated_wall_us"} or not _is_nonnegative_int(
            metrics.get("isolated_wall_us")
        ):
            raise LayoutAutotuneError("malformed failed-candidate compile metrics")
        return dict(metrics)

    expected = {
        "mode",
        "outcome",
        "cache_hit",
        "ir_initialization_us",
        "lowering_stages_us",
        "store_results_us",
        "total_us",
        "isolated_wall_us",
    }
    stages = metrics.get("lowering_stages_us")
    if (
        set(metrics) != expected
        or not isinstance(metrics.get("cache_hit"), bool)
        or any(
            not _is_nonnegative_int(metrics.get(name))
            for name in (
                "ir_initialization_us",
                "store_results_us",
                "total_us",
                "isolated_wall_us",
            )
        )
        or not isinstance(stages, list)
    ):
        raise LayoutAutotuneError("malformed successful-candidate compile metrics")
    stage_names = set()
    for entry in stages:
        if (
            not isinstance(entry, list)
            or len(entry) != 2
            or not isinstance(entry[0], str)
            or not entry[0]
            or entry[0] in stage_names
            or not _is_nonnegative_int(entry[1])
        ):
            raise LayoutAutotuneError("malformed or duplicate compile stage timing")
        stage_names.add(entry[0])
    accounted_us = (
        metrics["ir_initialization_us"]
        + sum(entry[1] for entry in stages)
        + metrics["store_results_us"]
    )
    if metrics["total_us"] != accounted_us:
        raise LayoutAutotuneError(
            "compile total does not equal compile stage timings"
        )
    return dict(metrics)


def _isolated_compile_worker(connection, request: _IsolatedCompileRequest) -> None:
    """Compile one exact manifest source and publish only its cache artifact."""

    try:
        from types import SimpleNamespace

        from triton import knobs
        from triton.compiler.compiler import compile as triton_compile

        compile_metrics = {
            "mode": "isolated-candidate",
            "outcome": "success",
        }

        def record_compile_times(*, times, cache_hit, **_kwargs):
            compile_metrics.update({
                "cache_hit": bool(cache_hit),
                "ir_initialization_us": int(times.ir_initialization),
                "lowering_stages_us": [
                    [str(stage), int(duration)]
                    for stage, duration in times.lowering_stages
                ],
                "store_results_us": int(times.store_results),
                "total_us": int(times.total),
            })

        compile_kwargs = {
            "target": request.target,
            "options": dict(request.options),
        }
        if request.env_vars is not None:
            compile_kwargs["_env_vars"] = dict(request.env_vars)
        previous_listener = knobs.compilation.listener
        # The candidate worker is a disposable instrumentation boundary. Do
        # not invoke a process-global listener inherited from the parent: its
        # side effects cannot be observed reliably across the forkserver IPC.
        knobs.compilation.listener = record_compile_times
        try:
            kernel = triton_compile(request.source_path, **compile_kwargs)
        finally:
            knobs.compilation.listener = previous_listener
        metadata = getattr(kernel, "metadata", None)
        if (
            getattr(metadata, "gluon_layout_domain_digest", None)
            != request.domain_digest
            or getattr(metadata, "gluon_layout_variant_digest", None)
            != request.variant_digest
            or getattr(metadata, "gluon_layout_runtime_contract", None)
            != request.runtime_contract
        ):
            raise RuntimeError(
                "standalone candidate changed its domain, variant, or effect identity"
            )
        device_signature = getattr(getattr(kernel, "src", None), "signature", None)
        if (
            not isinstance(device_signature, Mapping)
            or set(device_signature) != set(range(len(device_signature)))
        ):
            raise RuntimeError("standalone candidate has an invalid device ABI")
        function = SimpleNamespace(
            arg_names=list(request.launcher.argument_names),
            constexprs=request.launcher.constexprs,
        )
        kernel.src = SimpleNamespace(
            fn=function,
            signature=dict(request.launcher.signature),
            constants={path: None for path in request.launcher.constant_paths},
        )
        connection.send({
            "status": "ok",
            "artifact": _CompiledArtifact(
                _make_kernel_reload_descriptor(kernel),
                tuple(str(device_signature[index]) for index in range(len(device_signature))),
                _validate_compile_metrics({
                    **compile_metrics,
                    "isolated_wall_us": 0,
                }),
            ),
        })
    except BaseException as error:
        try:
            connection.send({
                "status": "error",
                "error_type": f"{type(error).__module__}.{type(error).__qualname__}",
                "message": _compact_candidate_error(error),
            })
        except BaseException:
            pass
    finally:
        connection.close()


def _compile_variant_isolated(
    request: _IsolatedCompileRequest, source_kernel: Any,
) -> tuple[Any, tuple[str, ...], Mapping[str, Any]]:
    """Compile behind a killable process and reload without recompilation."""

    started_ns = time.perf_counter_ns()
    parent_connection = None
    child_connection = None
    process = None
    phase = "compile-isolation"
    try:
        context = multiprocessing.get_context("forkserver")
        parent_connection, child_connection = context.Pipe(duplex=False)
        process = context.Process(
            target=_isolated_compile_worker,
            args=(child_connection, request),
            daemon=False,
        )
        phase = "compile-start"
        process.start()
        child_connection.close()
        phase = "compile-worker"
        message = _receive_worker_message(
            process,
            parent_connection,
            _AUTOTUNE_CANDIDATE_COMPILE_TIMEOUT_SECONDS,
            "compile",
        )
        if message["status"] != "ok" or not isinstance(
            message.get("artifact"), _CompiledArtifact
        ):
            raise _CandidateCompileFailure(
                "compile-error",
                _compact_candidate_error(
                    f"{message.get('error_type', 'unknown')}: "
                    f"{message.get('message', '')}"
                ),
            )
        artifact = message["artifact"]
        phase = "compile-artifact-reload"
        from triton.compiler.compiler import CompiledKernel

        source = getattr(source_kernel, "src", None)
        if source is None:
            raise _CandidateCompileFailure(
                "compile-artifact-error",
                "outer specialization lost its launcher source",
            )
        kernel = CompiledKernel(
            source,
            dict(artifact.reload.metadata_group),
            artifact.reload.compilation_hash,
        )
        compile_metrics = dict(artifact.compile_metrics)
        compile_metrics["isolated_wall_us"] = (
            time.perf_counter_ns() - started_ns
        ) // 1000
        return kernel, artifact.device_signature, _validate_compile_metrics(
            compile_metrics
        )
    except Exception as error:
        metrics = _validate_compile_metrics({
            "mode": "isolated-candidate",
            "outcome": "failure",
            "isolated_wall_us": (time.perf_counter_ns() - started_ns) // 1000,
        })
        if isinstance(error, _CandidateCompileFailure):
            reason = error.reason
            detail = error.detail
        elif isinstance(error, _CandidateExecutionFailure):
            reason = error.reason
            detail = error.detail
        elif phase == "compile-isolation" and isinstance(error, ValueError):
            reason = "compile-isolation-unavailable"
            detail = str(error)
        elif phase == "compile-artifact-reload":
            reason = "compile-artifact-error"
            detail = str(error)
        else:
            reason = f"{phase}-error"
            detail = str(error)
        raise _CandidateCompileFailure(
            reason, _compact_candidate_error(detail), metrics
        ) from error
    finally:
        if child_connection is not None and (
            process is None or process.pid is None
        ):
            child_connection.close()
        if parent_connection is not None:
            if process is None:
                parent_connection.close()
            else:
                _close_isolated_worker(process, parent_connection)


def _close_isolated_worker(process, connection) -> None:
    connection.close()
    if process.pid is None:
        return
    process.join(timeout=5)
    if not process.is_alive():
        return
    process.terminate()
    process.join(timeout=5)
    if process.is_alive() and hasattr(process, "kill"):
        process.kill()
        process.join(timeout=5)


def _receive_worker_message(process, connection, timeout: int, phase: str) -> Mapping[str, Any]:
    if not connection.poll(timeout):
        raise _CandidateExecutionFailure(
            f"{phase}-timeout",
            f"isolated worker exceeded {timeout} seconds",
        )
    try:
        message = connection.recv()
    except EOFError as error:
        raise _CandidateExecutionFailure(
            f"{phase}-crash",
            f"isolated worker exited with code {process.exitcode}",
        ) from error
    if not isinstance(message, Mapping) or not isinstance(message.get("status"), str):
        raise _CandidateExecutionFailure(f"{phase}-protocol", "isolated worker returned a malformed response")
    return message


def _isolated_measurement_worker(connection, request: _IsolatedMeasurementRequest) -> None:
    """Reload and measure exactly one compiled candidate in a fresh process."""

    phase = "setup"
    try:
        from types import SimpleNamespace

        from triton.compiler.compiler import CompiledKernel
        from triton.runtime.driver import driver

        active = driver.active
        device_interface = active.get_device_interface()
        device_interface.set_device(request.device)
        function = SimpleNamespace(
            arg_names=list(request.reload.argument_names),
            constexprs=request.reload.constexprs,
        )
        source = SimpleNamespace(
            fn=function,
            signature=dict(request.reload.signature),
            constants={path: None for path in request.reload.constant_paths},
        )
        kernel = CompiledKernel(
            source,
            dict(request.reload.metadata_group),
            request.reload.compilation_hash,
        )
        stream = active.get_current_stream(request.device)
        # Load the binary and build the launcher before READY. The parent starts
        # the device-execution timeout only after this setup is complete.
        _ = kernel.run
        connection.send({"status": "ready"})
        command = connection.recv()
        if command != "run":
            raise RuntimeError("isolated worker received an invalid command")

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
        connection.send({
            "status": "ok",
            "timing": measurement.timing,
            "device_samples_ms": measurement.device_samples_ms,
            "resources": _kernel_resources(kernel),
        })
    except BaseException as error:
        try:
            connection.send({
                "status": "error",
                "phase": phase,
                "error_type": f"{type(error).__module__}.{type(error).__qualname__}",
                "message": str(error),
            })
        except BaseException:
            pass
    finally:
        connection.close()


def _measure_variant_isolated(
    kernel: Any, grid: Sequence[int], replay: _ScratchReplay,
) -> _MeasurementResult:
    """Measure one candidate behind a killable process boundary."""

    from triton.runtime.driver import driver

    reload = _make_kernel_reload_descriptor(kernel)
    args = list(replay.args)
    for path in reload.constant_paths:
        # The launcher parses constexpr positions as Python objects but never
        # forwards them to the device kernel. Avoid transporting arbitrary
        # compiler objects through multiprocessing.
        args[path[0]] = None
    request = _IsolatedMeasurementRequest(
        reload,
        tuple(int(item) for item in grid),
        int(driver.active.get_current_device()),
        tuple(args),
        replay.reset_pairs,
    )
    methods = multiprocessing.get_all_start_methods()
    if "forkserver" not in methods:
        raise LayoutAutotuneError("isolated layout autotuning requires multiprocessing forkserver support")
    context = multiprocessing.get_context("forkserver")
    parent_connection, child_connection = context.Pipe(duplex=True)
    process = context.Process(
        target=_isolated_measurement_worker,
        args=(child_connection, request),
        daemon=True,
    )
    setup_started_ns = time.perf_counter_ns()
    try:
        process.start()
        child_connection.close()
        ready = _receive_worker_message(
            process,
            parent_connection,
            _AUTOTUNE_WORKER_SETUP_TIMEOUT_SECONDS,
            "setup",
        )
        setup_wall_us = (time.perf_counter_ns() - setup_started_ns) // 1000
        if ready["status"] == "error":
            raise _CandidateExecutionFailure(
                "setup-error",
                f"{ready.get('error_type', 'unknown')}: {ready.get('message', '')}",
            )
        if ready["status"] != "ready":
            raise _CandidateExecutionFailure("setup-protocol", "isolated worker did not report READY")
        execution_started_ns = time.perf_counter_ns()
        parent_connection.send("run")
        measured = _receive_worker_message(
            process,
            parent_connection,
            _AUTOTUNE_CANDIDATE_TIMEOUT_SECONDS,
            "execution",
        )
        execution_wall_us = (
            time.perf_counter_ns() - execution_started_ns
        ) // 1000
        if measured["status"] == "error":
            raise _CandidateExecutionFailure(
                "execution-error",
                f"{measured.get('error_type', 'unknown')}: {measured.get('message', '')}",
            )
        return _validated_measurement_result(
            measured, setup_wall_us, execution_wall_us
        )
    finally:
        _close_isolated_worker(process, parent_connection)


def _validated_measurement_result(
    measured: Mapping[str, Any],
    setup_wall_us: int,
    execution_wall_us: int,
) -> _MeasurementResult:
    """Validate one worker response independently of worker lifetime."""

    timing = measured.get("timing")
    device_samples = measured.get("device_samples_ms")
    resources = measured.get("resources")
    if (
        measured.get("status") != "ok"
        or not isinstance(timing, Sequence)
        or not isinstance(device_samples, Sequence)
        or not isinstance(resources, Mapping)
    ):
        raise _CandidateExecutionFailure(
            "execution-protocol", "isolated worker returned no measurement"
        )
    values = tuple(float(value) for value in timing)
    if (
        len(values) != 3
        or any(not math.isfinite(value) or value < 0.0 for value in values)
        or not values[1] <= values[0] <= values[2]
    ):
        raise _CandidateExecutionFailure(
            "execution-timing", "isolated worker returned invalid timing"
        )
    samples = tuple(float(value) for value in device_samples)
    if (
        len(samples) != _AUTOTUNE_REPEAT_LAUNCHES
        or any(not math.isfinite(value) or value < 0.0 for value in samples)
    ):
        raise _CandidateExecutionFailure(
            "execution-timing",
            "isolated worker returned invalid raw device-event samples",
        )
    from triton.testing import _summarize_statistics

    summarized = tuple(
        float(value)
        for value in _summarize_statistics(
            samples,
            list(_MEASUREMENT_PROTOCOL["quantiles"]),
            "mean",
        )
    )
    if not _timing_summaries_match(values, summarized):
        raise _CandidateExecutionFailure(
            "execution-timing",
            "isolated worker timing does not summarize its raw samples",
        )
    return _MeasurementResult(
        values,
        dict(resources),
        samples,
        setup_wall_us,
        execution_wall_us,
    )


def _timing_intervals_overlap(lhs: Sequence[float], rhs: Sequence[float]) -> bool:
    return float(lhs[1]) <= float(rhs[2]) and float(rhs[1]) <= float(lhs[2])


def _aggregate_timing_samples(samples: Sequence[Sequence[float]]) -> tuple[float, float, float]:
    return tuple(statistics.median(float(sample[index]) for sample in samples) for index in range(3))


def _normalize_timing_summary(
    timing: Sequence[float], description: str,
) -> tuple[float, float, float]:
    if (
        not isinstance(timing, Sequence)
        or isinstance(timing, (str, bytes))
        or len(timing) != 3
    ):
        raise LayoutAutotuneError(
            f"{description} must be a median/p20/p80 triple"
        )
    if any(isinstance(value, bool) for value in timing):
        raise LayoutAutotuneError(f"{description} contains a boolean value")
    try:
        values = tuple(float(value) for value in timing)
    except (TypeError, ValueError, OverflowError) as error:
        raise LayoutAutotuneError(
            f"{description} contains a non-numeric value"
        ) from error
    if (
        any(not math.isfinite(value) or value < 0.0 for value in values)
        or not values[1] <= values[0] <= values[2]
    ):
        raise LayoutAutotuneError(
            f"{description} has non-finite, negative, or unordered values"
        )
    return values


def _timing_summaries_match(
    lhs: Sequence[float], rhs: Sequence[float],
) -> bool:
    return all(
        math.isclose(float(lhs_value), float(rhs_value), rel_tol=1e-12, abs_tol=1e-12)
        for lhs_value, rhs_value in zip(lhs, rhs)
    )


def _serialize_measurement_rounds(
    rounds: Sequence[_MeasurementRound],
    final_timing: Sequence[float] | None,
) -> list[Mapping[str, Any]]:
    records = []
    seen_rounds = set()
    for expected_round_index, measurement in enumerate(rounds):
        if not isinstance(measurement, _MeasurementRound):
            raise LayoutAutotuneError("measurement round has an invalid representation")
        if (
            not _is_nonnegative_int(measurement.round_index)
            or measurement.round_index > _INTERLEAVED_REMEASUREMENTS
            or measurement.round_index != expected_round_index
            or measurement.round_index in seen_rounds
            or not _is_nonnegative_int(measurement.order_index)
        ):
            raise LayoutAutotuneError("measurement rounds are not uniquely and consecutively indexed")
        seen_rounds.add(measurement.round_index)
        timing = _normalize_timing_summary(
            measurement.timing,
            f"measurement round {measurement.round_index} timing",
        )
        samples = measurement.device_samples_ms
        walls = (measurement.setup_wall_us, measurement.execution_wall_us)
        telemetry_available = samples is not None
        if telemetry_available != all(wall is not None for wall in walls):
            raise LayoutAutotuneError(
                "measurement round must provide raw samples and both wall times together"
            )
        if telemetry_available:
            if (
                not isinstance(samples, Sequence)
                or isinstance(samples, (str, bytes))
                or any(isinstance(value, bool) for value in samples)
            ):
                raise LayoutAutotuneError(
                    "measurement round has invalid raw device samples"
                )
            try:
                normalized_samples = tuple(float(value) for value in samples)
            except (TypeError, ValueError, OverflowError) as error:
                raise LayoutAutotuneError(
                    "measurement round has non-numeric raw device samples"
                ) from error
            if (
                len(normalized_samples) != _AUTOTUNE_REPEAT_LAUNCHES
                or any(
                    not math.isfinite(value) or value < 0.0
                    for value in normalized_samples
                )
                or any(not _is_nonnegative_int(wall) for wall in walls)
            ):
                raise LayoutAutotuneError(
                    "measurement round has invalid device samples or wall times"
                )
            from triton.testing import _summarize_statistics

            summarized = tuple(
                float(value)
                for value in _summarize_statistics(
                    normalized_samples,
                    list(_MEASUREMENT_PROTOCOL["quantiles"]),
                    "mean",
                )
            )
            if not _timing_summaries_match(timing, summarized):
                raise LayoutAutotuneError(
                    "measurement round timing does not summarize its raw device samples"
                )
            samples_record = list(normalized_samples)
        else:
            if any(wall is not None for wall in walls):
                raise LayoutAutotuneError(
                    "measurement round has partial unavailable telemetry"
                )
            samples_record = None
        records.append({
            "round_index": measurement.round_index,
            "order_index": measurement.order_index,
            "timing_ms": list(timing),
            "device_samples_ms": samples_record,
            "setup_wall_us": measurement.setup_wall_us,
            "execution_wall_us": measurement.execution_wall_us,
        })
    if final_timing is not None:
        normalized_final = _normalize_timing_summary(
            final_timing,
            "final variant timing",
        )
        if not records:
            raise LayoutAutotuneError(
                "timed layout variant has no measurement rounds"
            )
        aggregated = _aggregate_timing_samples(
            [record["timing_ms"] for record in records]
        )
        if not _timing_summaries_match(normalized_final, aggregated):
            raise LayoutAutotuneError(
                "final variant timing does not aggregate its measurement rounds"
            )
    return records


def _validate_cached_measurement_rounds(
    records: Any,
    final_timing: Sequence[float] | None,
) -> list[Mapping[str, Any]]:
    expected = {
        "round_index",
        "order_index",
        "timing_ms",
        "device_samples_ms",
        "setup_wall_us",
        "execution_wall_us",
    }
    if not isinstance(records, list):
        raise LayoutAutotuneError("cached measurement rounds must be a list")
    rounds = []
    for record in records:
        if not isinstance(record, Mapping) or set(record) != expected:
            raise LayoutAutotuneError(
                "cached measurement round has an invalid schema"
            )
        samples = record["device_samples_ms"]
        if samples is not None and (
            not isinstance(samples, list)
            or any(isinstance(value, bool) for value in samples)
        ):
            raise LayoutAutotuneError(
                "cached measurement round has invalid raw samples"
            )
        rounds.append(
            _MeasurementRound(
                record["round_index"],
                record["order_index"],
                record["timing_ms"],
                tuple(samples) if samples is not None else None,
                record["setup_wall_us"],
                record["execution_wall_us"],
            )
        )
    return _serialize_measurement_rounds(rounds, final_timing)


def _validate_selection_trace(
    selection: Mapping[str, Any],
    manifest: LayoutManifest,
    winner: LayoutVariant,
    timings: Mapping[LayoutVariant, Sequence[float]],
) -> dict[str, Any]:
    expected = {
        "initial_winner",
        "close",
        "stable",
        "selected",
        "fallback_forced",
    }
    if not isinstance(selection, Mapping) or set(selection) != expected:
        raise LayoutAutotuneError("layout selection trace has an invalid schema")
    known = {variant.digest for variant in manifest.variants}
    initial = selection["initial_winner"]
    selected = selection["selected"]
    close = selection["close"]
    stable = selection["stable"]
    if (
        not isinstance(initial, str)
        or not isinstance(selected, str)
        or initial not in known
        or selected != winner.digest
        or selected not in known
        or not isinstance(close, list)
        or not isinstance(stable, list)
        or any(not isinstance(digest, str) for digest in (*close, *stable))
        or any(digest not in known for digest in (*close, *stable))
        or len(set(close)) != len(close)
        or len(set(stable)) != len(stable)
        or manifest.fallback.digest not in stable
        or selected not in stable
        or not isinstance(selection["fallback_forced"], bool)
        or (
            selection["fallback_forced"]
            and selected != manifest.fallback.digest
        )
        or any(
            variant.digest not in {candidate.digest for candidate in timings}
            for variant in manifest.variants
            if variant.digest in stable
        )
    ):
        raise LayoutAutotuneError("layout selection trace is inconsistent")
    return {
        "initial_winner": initial,
        "close": list(close),
        "stable": list(stable),
        "selected": selected,
        "fallback_forced": selection["fallback_forced"],
    }


def _validate_cache_record(
    record: Mapping[str, Any],
    manifest: LayoutManifest,
    workload_digest: str,
    effect_digest: str,
    bundle_identity: Sequence[Mapping[str, Any]],
) -> tuple[LayoutVariant, Mapping[str, Any]]:
    top_level = {
        "version",
        "manifest_digest",
        "workload_digest",
        "workload",
        "effect_digest",
        "bundle_identity",
        "winner",
        "selection",
        "variants",
    }
    if not isinstance(record, Mapping):
        raise LayoutAutotuneError("cached layout autotune record is not an object")
    workload = record.get("workload")
    if (
        set(record) != top_level
        or record.get("version") != _MEASUREMENT_PROTOCOL_VERSION
        or record.get("manifest_digest") != manifest.digest
        or record.get("workload_digest") != workload_digest
        or not isinstance(workload, Mapping)
        or workload.get("measurement") != _MEASUREMENT_PROTOCOL
        or _sha256_json(workload) != workload_digest
        or record.get("effect_digest") != effect_digest
        or record.get("bundle_identity") != bundle_identity
    ):
        raise LayoutAutotuneError("cached layout autotune identity is invalid")

    variants = record.get("variants")
    variant_fields = {
        "digest",
        "mlir_file",
        "mlir_sha256",
        "timing_ms",
        "failure",
        "failure_detail",
        "compile_metrics",
        "measurement_rounds",
        "executable",
        "resources",
    }
    if not isinstance(variants, list) or len(variants) != len(manifest.variants):
        raise LayoutAutotuneError("cached layout variants do not match the manifest")
    timings = {}
    occupied_positions = set()
    by_digest = {}
    for specification, variant in zip(manifest.variants, variants):
        if (
            not isinstance(variant, Mapping)
            or set(variant) != variant_fields
            or variant.get("digest") != specification.digest
            or variant.get("mlir_file") != specification.mlir_file
            or variant.get("mlir_sha256") != specification.mlir_sha256
        ):
            raise LayoutAutotuneError("cached layout variant identity is invalid")
        timing = variant.get("timing_ms")
        normalized_timing = (
            _normalize_timing_summary(timing, "cached final variant timing")
            if timing is not None
            else None
        )
        rounds = _validate_cached_measurement_rounds(
            variant.get("measurement_rounds"),
            normalized_timing,
        )
        for round_record in rounds:
            position = (
                round_record["round_index"],
                round_record["order_index"],
            )
            if position in occupied_positions:
                raise LayoutAutotuneError(
                    "cached measurement rounds contain duplicate order positions"
                )
            occupied_positions.add(position)
        failure = variant.get("failure")
        failure_detail = variant.get("failure_detail")
        executable = variant.get("executable")
        resources = variant.get("resources")
        compile_metrics = _validate_compile_metrics(
            variant.get("compile_metrics")
        )
        has_failure = isinstance(failure, str) and bool(failure)
        if (
            has_failure != (
                isinstance(failure_detail, str) and bool(failure_detail)
            )
            or (normalized_timing is not None) == has_failure
            or (
                executable is None
                and (
                    compile_metrics["outcome"] != "failure"
                    or resources is not None
                    or rounds
                )
            )
            or (
                executable is not None
                and (
                    not isinstance(executable, Mapping)
                    or not isinstance(resources, Mapping)
                    or compile_metrics["outcome"] != "success"
                )
            )
        ):
            raise LayoutAutotuneError("cached layout variant outcome is invalid")
        if normalized_timing is not None:
            timings[specification] = normalized_timing
        by_digest[specification.digest] = variant

    winner = record.get("winner")
    if not isinstance(winner, Mapping) or set(winner) != {
        "digest",
        "mlir_sha256",
        "executable",
    }:
        raise LayoutAutotuneError("cached layout winner has an invalid schema")
    specification = next(
        (
            variant
            for variant in manifest.variants
            if variant.digest == winner.get("digest")
        ),
        None,
    )
    winner_variant = (
        by_digest.get(specification.digest)
        if specification is not None
        else None
    )
    if (
        specification is None
        or winner.get("mlir_sha256") != specification.mlir_sha256
        or not isinstance(winner.get("executable"), Mapping)
        or winner_variant is None
        or winner_variant["timing_ms"] is None
        or winner_variant["failure"] is not None
        or winner_variant["executable"] != winner["executable"]
    ):
        raise LayoutAutotuneError("cached layout winner is inconsistent")
    _validate_selection_trace(
        record.get("selection"),
        manifest,
        specification,
        timings,
    )
    return specification, winner["executable"]


def _make_measurement_round(
    round_index: int, order_index: int, measurement: _MeasurementResult,
) -> _MeasurementRound:
    return _MeasurementRound(
        round_index,
        order_index,
        measurement.timing,
        measurement.device_samples_ms,
        measurement.setup_wall_us,
        measurement.execution_wall_us,
    )


class CompilerLayoutAutotuner:
    """Compile, measure, and select within one closed Gluon source domain."""

    def __init__(
        self,
        *,
        benchmarker: Callable | None = None,
        cache_manager_factory: Callable[[str], Any] | None = None,
        measurer: Callable | None = None,
    ):
        self._benchmarker = benchmarker
        self._cache_manager_factory = cache_manager_factory
        self._measurer = measurer
        self._memory_cache: dict[tuple[str, str, str], str] = {}
        # Keep a strong reference to each bundle so an object id cannot be
        # recycled into an unrelated specialization. Each inner key is an
        # exact launch identity, not the allocation-independent persistent
        # workload key.
        self._fast_winners: dict[
            int,
            tuple[GluonLayoutVariantBundle, dict[tuple[Any, ...], _FastWinner]],
        ] = {}
        self._lock = threading.RLock()

    def _get_benchmarker(self):
        return self._benchmarker or _device_event_benchmarker

    def _get_cache_manager(self, key: str):
        if self._cache_manager_factory is not None:
            return self._cache_manager_factory(key)
        from triton.runtime.cache import get_cache_manager

        return get_cache_manager(key)

    def _measure(self, kernel, grid, stream, replay: _ScratchReplay):
        if self._measurer is not None:
            # A custom internal measurer is an opaque single measurement. Its
            # reset cannot be interposed between launches, so initialize once
            # immediately before handing it the scratch arguments.
            replay.reset()
            timing = _normalize_timing_summary(
                self._measurer(kernel, grid, stream, replay.args),
                "custom layout measurement",
            )
            return _MeasurementResult(
                timing,
                _kernel_resources(kernel),
                None,
                None,
                None,
            )
        if self._benchmarker is not None:
            timing = _measure_variant(
                kernel, grid, stream, replay.args, self._get_benchmarker(), replay.reset,
            )
            return _MeasurementResult(
                timing.timing,
                _kernel_resources(kernel),
                None,
                None,
                None,
            )
        return _measure_variant_isolated(kernel, grid, replay)

    @staticmethod
    def _is_known_failure(error: Exception) -> bool:
        from triton.compiler.errors import CompileTimeAssertionFailure
        from triton.runtime.errors import OutOfResources, PTXASError

        return isinstance(error, (
            _CandidateCompileFailure,
            _CandidateExecutionFailure,
            OutOfResources,
            CompileTimeAssertionFailure,
            PTXASError,
        ))

    def _record_candidate_failure(
        self,
        outcomes: _DomainMeasurements,
        manifest: LayoutManifest,
        variant: LayoutVariant,
        error: Exception,
    ) -> None:
        if isinstance(error, LayoutAutotuneError):
            raise error
        if variant == manifest.fallback:
            raise FallbackVariantExecutionError(
                "the mandatory Gluon layout fallback failed during "
                "compilation or device-event measurement"
            ) from error
        if not self._is_known_failure(error):
            raise error
        warnings.warn(
            f"rejecting Gluon layout variant {variant.digest}: {error}",
            RuntimeWarning,
        )
        outcomes.failures[variant] = getattr(
            error, "reason", "execution-error"
        )
        outcomes.failure_details[variant] = _compact_candidate_error(
            getattr(error, "detail", error)
        )

    def _measure_compiled_candidate(
        self,
        outcomes: _DomainMeasurements,
        manifest: LayoutManifest,
        variant: LayoutVariant,
        compiled: GluonLayoutCompiledVariant,
        order_index: int,
        grid: Sequence[int],
        stream: Any,
        replay: _ScratchReplay,
    ) -> None:
        try:
            measurement = self._measure(
                compiled.kernel, grid, stream, replay
            )
        except Exception as error:
            self._record_candidate_failure(
                outcomes, manifest, variant, error
            )
            return
        outcomes.timings[variant] = measurement.timing
        outcomes.rounds.setdefault(variant, []).append(
            _make_measurement_round(0, order_index, measurement)
        )
        outcomes.resources[variant] = measurement.resources
        _LOGGER.debug(
            "measured Gluon layout variant order=%d digest=%s file=%s "
            "timing_ms=%s resources=%s",
            order_index,
            variant.digest,
            variant.mlir_file,
            measurement.timing,
            measurement.resources,
        )

    def _evaluate_domain(
        self,
        bundle: GluonLayoutVariantBundle,
        manifest: LayoutManifest,
        grid: Sequence[int],
        stream: Any,
        replay: _ScratchReplay,
    ) -> _DomainMeasurements:
        """Compile and measure candidates serially in manifest order.

        This baseline remains available for tests and custom controllers.
        Production registers a parallel compiler that overrides this hook
        without changing effect replay, measurement, selection, or cache
        semantics.
        """

        outcomes = _DomainMeasurements()
        ordered = [manifest.fallback]
        ordered.extend(
            variant
            for variant in manifest.variants
            if variant != manifest.fallback
        )
        for order_index, variant in enumerate(ordered):
            try:
                compiled = bundle.get_or_compile(variant)
            except Exception as error:
                self._record_candidate_failure(
                    outcomes, manifest, variant, error
                )
                continue
            self._measure_compiled_candidate(
                outcomes,
                manifest,
                variant,
                compiled,
                order_index,
                grid,
                stream,
                replay,
            )
        return outcomes

    def _remeasure_close_variants(
        self, timings, resources, failures, failure_details, measurement_rounds,
        bundle, manifest, grid, stream, replay,
    ):
        initial_winner = select_measured_variant(timings, manifest.fallback)
        best_timing = timings[initial_winner]
        fallback_timing = timings[manifest.fallback]
        stable = {manifest.fallback}
        stable.update(
            variant for variant, timing in timings.items()
            if variant != manifest.fallback
            and float(timing[0]) < float(fallback_timing[0])
            and not _timing_intervals_overlap(timing, fallback_timing)
        )
        close = sorted(
            (variant for variant, timing in timings.items() if _timing_intervals_overlap(timing, best_timing)),
            key=lambda variant: (variant.digest, variant.mlir_file),
        )
        if len(close) <= 1:
            return _StabilityResult(
                initial_winner,
                tuple(close),
                frozenset(stable),
            )
        _LOGGER.debug("remeasuring close Gluon layout variants digests=%s", [item.digest for item in close])

        samples = {variant: [timings[variant]] for variant in close}
        rejected = set()
        for iteration in range(_INTERLEAVED_REMEASUREMENTS):
            order = close if iteration % 2 == 0 else list(reversed(close))
            for order_index, variant in enumerate(order):
                if variant in rejected:
                    continue
                kernel = bundle.variants[variant.digest].kernel
                try:
                    measurement = self._measure(kernel, grid, stream, replay)
                    samples[variant].append(measurement.timing)
                    measurement_rounds.setdefault(variant, []).append(
                        _make_measurement_round(
                            iteration + 1,
                            order_index,
                            measurement,
                        )
                    )
                    resources[variant] = measurement.resources
                except LayoutAutotuneError:
                    # Scratch restoration and timing-protocol failures make
                    # autotuning unavailable; they do not prove that the real
                    # fallback executable cannot service the user launch.
                    raise
                except Exception as error:
                    if variant == manifest.fallback:
                        raise FallbackVariantExecutionError(
                            "the mandatory Gluon layout fallback failed during device-event measurement"
                        ) from error
                    if not self._is_known_failure(error):
                        raise
                    warnings.warn(
                        f"rejecting Gluon layout variant {variant.digest} during remeasurement: {error}",
                        RuntimeWarning,
                    )
                    failures[variant] = getattr(error, "reason", "execution-error")
                    failure_details[variant] = _compact_candidate_error(
                        getattr(error, "detail", error)
                    )
                    stable.discard(variant)
                    rejected.add(variant)
        for variant in close:
            if variant in rejected:
                timings.pop(variant, None)
            else:
                timings[variant] = _aggregate_timing_samples(samples[variant])
        fallback_samples = samples.get(manifest.fallback)
        if fallback_samples is not None:
            for variant in close:
                variant_samples = samples.get(variant)
                if variant == manifest.fallback or variant_samples is None:
                    continue
                aligned_gain = len(variant_samples) == len(fallback_samples) and all(
                    float(variant_sample[0]) < float(fallback_sample[0])
                    for variant_sample, fallback_sample in zip(variant_samples, fallback_samples)
                )
                if aligned_gain:
                    stable.add(variant)
                else:
                    stable.discard(variant)
        return _StabilityResult(
            initial_winner,
            tuple(close),
            frozenset(stable),
        )

    @staticmethod
    def _validate_bundle(bundle: GluonLayoutVariantBundle, manifest: LayoutManifest) -> list[Mapping[str, Any]]:
        expected = {variant.digest: variant for variant in manifest.variants}
        active = set(bundle.variants)
        if manifest.fallback.digest not in active or not active.issubset(expected):
            raise LayoutAutotuneError(
                "compiled variant bundle must contain its fallback and only variants published by its manifest"
            )
        if bundle.sources and set(bundle.sources) != set(expected):
            raise LayoutAutotuneError(
                "closed source domain does not exactly match its manifest"
            )

        effect_digest = _runtime_contract_digest(bundle.runtime_contract)
        fallback_signature = bundle.variants[manifest.fallback.digest].device_signature
        for digest, compiled in bundle.variants.items():
            specification = expected[digest]
            if not isinstance(compiled, GluonLayoutCompiledVariant):
                raise LayoutAutotuneError(f"layout variant {digest} has no compiled wrapper")
            if (
                compiled.digest != digest
                or compiled.mlir_file != specification.mlir_file
                or compiled.mlir_sha256 != specification.mlir_sha256
                or not isinstance(compiled.device_signature, tuple)
                or any(not isinstance(item, str) for item in compiled.device_signature)
                or compiled.device_signature != fallback_signature
                or compiled.kernel is None
            ):
                raise LayoutAutotuneError(f"layout variant {digest} does not match its manifest identity")
            metadata = getattr(compiled.kernel, "metadata", None)
            domain_digest = getattr(metadata, "gluon_layout_domain_digest", None)
            selected_digest = getattr(metadata, "gluon_layout_variant_digest", None)
            if domain_digest != manifest.digest or selected_digest != digest:
                raise LayoutAutotuneError(f"layout variant {digest} changed its compiler-selected identity")
            variant_contract = getattr(metadata, "gluon_layout_runtime_contract", None)
            if _runtime_contract_digest(variant_contract) != effect_digest:
                raise LayoutAutotuneError(f"layout variant {digest} changed the runtime effect contract")
        return _source_domain_identity(manifest)

    @staticmethod
    def _cache_key(manifest: LayoutManifest, workload_digest: str) -> str:
        return _sha256_json({
            "protocol": _MEASUREMENT_PROTOCOL_VERSION,
            "manifest": manifest.digest,
            "workload": workload_digest,
        })

    def _lookup_fast_winner(
        self, bundle: GluonLayoutVariantBundle, launch_identity: tuple[Any, ...] | None,
    ):
        if launch_identity is None:
            return None
        bundle_entry = self._fast_winners.get(id(bundle))
        if bundle_entry is None or bundle_entry[0] is not bundle:
            return None
        winner = bundle_entry[1].get(launch_identity)
        if winner is None:
            return None

        # The bundle is compiler-produced and closed. Still guard the selected
        # mapping entry in O(1), so accidental replacement or deletion cannot
        # make this fast path launch an object that is no longer the winner in
        # the bundle.
        compiled = bundle.variants.get(winner.digest)
        if compiled is not winner.compiled:
            bundle_entry[1].pop(launch_identity, None)
            return None
        return compiled.kernel

    def _remember_fast_winner(
        self,
        bundle: GluonLayoutVariantBundle,
        launch_identity: tuple[Any, ...] | None,
        digest: str,
    ) -> None:
        if launch_identity is None:
            return
        compiled = bundle.variants.get(digest)
        if compiled is None:
            return
        bundle_id = id(bundle)
        bundle_entry = self._fast_winners.get(bundle_id)
        if bundle_entry is None or bundle_entry[0] is not bundle:
            bundle_entry = (bundle, {})
            self._fast_winners[bundle_id] = bundle_entry
        bundle_entry[1][launch_identity] = _FastWinner(digest, compiled)

    @staticmethod
    def _read_cache(cache, manifest, workload_digest, effect_digest, bundle_identity):
        path = cache.get_file(_CACHE_FILENAME)
        if not path:
            return None
        try:
            with open(path) as handle:
                record = json.load(handle)
            return _validate_cache_record(
                record,
                manifest,
                workload_digest,
                effect_digest,
                bundle_identity,
            )
        except (
            OSError,
            json.JSONDecodeError,
            LayoutAutotuneError,
            TypeError,
            ValueError,
        ):
            return None

    @staticmethod
    def _write_cache(
        cache, manifest, workload_digest, workload, winner, timings, resources,
        failures, failure_details, bundle, effect_digest, bundle_identity,
        measurement_rounds, selection,
    ) -> None:
        serialized_rounds = {}
        occupied_positions = set()
        for variant in manifest.variants:
            rounds = _serialize_measurement_rounds(
                measurement_rounds.get(variant, ()),
                timings.get(variant),
            )
            for round_record in rounds:
                position = (
                    round_record["round_index"],
                    round_record["order_index"],
                )
                if position in occupied_positions:
                    raise LayoutAutotuneError(
                        "measurement rounds contain duplicate execution order positions"
                    )
                occupied_positions.add(position)
            serialized_rounds[variant] = rounds
        selection = _validate_selection_trace(
            selection,
            manifest,
            winner,
            timings,
        )
        records = []
        for variant in manifest.variants:
            compiled = bundle.variants.get(variant.digest)
            timing = timings.get(variant)
            compile_failure = bundle.compile_failures.get(variant.digest)
            compile_metrics = bundle.compile_metrics.get(variant.digest)
            if compile_metrics is None:
                raise LayoutAutotuneError(
                    f"layout variant {variant.digest} has no compile metrics"
                )
            compile_metrics = _validate_compile_metrics(compile_metrics)
            failure = failures.get(variant) or (
                compile_failure.reason if compile_failure is not None else None
            )
            failure_detail = failure_details.get(variant) or (
                compile_failure.detail if compile_failure is not None else None
            )
            if (failure is None) != (failure_detail is None):
                raise LayoutAutotuneError(
                    f"layout variant {variant.digest} has an incomplete failure record"
                )
            if failure is not None and timing is not None:
                raise LayoutAutotuneError(
                    f"layout variant {variant.digest} has both a failure and timing"
                )
            if compiled is None and compile_metrics.get("outcome") != "failure":
                raise LayoutAutotuneError(
                    f"uncompiled layout variant {variant.digest} has successful metrics"
                )
            if compiled is not None and compile_metrics.get("outcome") != "success":
                raise LayoutAutotuneError(
                    f"compiled layout variant {variant.digest} has failed metrics"
                )
            if compiled is not None and timing is None and failure is None:
                raise LayoutAutotuneError(
                    f"compiled layout variant {variant.digest} has no measurement outcome"
                )
            record = {
                "digest": variant.digest,
                "mlir_file": variant.mlir_file,
                "mlir_sha256": variant.mlir_sha256,
                "timing_ms": list(timing) if timing is not None else None,
                "failure": failure,
                "failure_detail": failure_detail,
                "compile_metrics": compile_metrics,
                "measurement_rounds": serialized_rounds[variant],
                "executable": (
                    _variant_executable_identity(compiled)
                    if compiled is not None
                    else None
                ),
                "resources": (
                    resources.get(variant, _kernel_resources(compiled.kernel))
                    if compiled is not None
                    else None
                ),
            }
            records.append(record)
        record = {
            "version": _MEASUREMENT_PROTOCOL_VERSION,
            "manifest_digest": manifest.digest,
            "workload_digest": workload_digest,
            "workload": workload,
            "effect_digest": effect_digest,
            "bundle_identity": bundle_identity,
            "winner": {
                "digest": winner.digest,
                "mlir_sha256": winner.mlir_sha256,
                "executable": _variant_executable_identity(
                    bundle.variants[winner.digest]
                ),
            },
            "selection": selection,
            "variants": records,
        }
        _validate_cache_record(
            record,
            manifest,
            workload_digest,
            effect_digest,
            bundle_identity,
        )
        cache.put(_canonical_json(record), _CACHE_FILENAME, binary=False)

    def _read_record(self, *args, **kwargs):
        store = getattr(self, "_record_store", None)
        if store is None:
            return self._read_cache(*args, **kwargs)
        return store.read(self, *args, **kwargs)

    def _write_record(self, *args, **kwargs):
        store = getattr(self, "_record_store", None)
        if store is None:
            return self._write_cache(*args, **kwargs)
        return store.write(self, *args, **kwargs)

    def _apply_selection_policy(self, *args, **kwargs):
        policy = getattr(self, "_selection_policy", None)
        if policy is None:
            return self._remeasure_close_variants(*args, **kwargs)
        return policy.remeasure(self, *args, **kwargs)

    def prepare_for_launch(
        self,
        kernel_or_bundle: Any,
        grid: Sequence[int],
        stream: Any,
        bound_args: Mapping[str, Any],
    ):
        if not isinstance(kernel_or_bundle, GluonLayoutVariantBundle):
            return kernel_or_bundle
        launch_identity = _fast_launch_identity(grid, stream, bound_args)
        winner = self._lookup_fast_winner(kernel_or_bundle, launch_identity)
        if winner is not None:
            return winner
        with self._lock:
            # A different thread may have populated the exact winner while this
            # caller waited. Recheck before entering the serialized miss path.
            winner = self._lookup_fast_winner(kernel_or_bundle, launch_identity)
            if winner is not None:
                return winner
            return self._prepare_bundle(
                kernel_or_bundle, grid, stream, bound_args, launch_identity,
            )

    def _prepare_bundle(self, bundle, grid, stream, bound_args, launch_identity=None):
        try:
            manifest = parse_layout_manifest(bundle.manifest)
            fallback_kernel = bundle.fallback_kernel
            specialization_kernel = bundle.specialization_kernel
            bundle_identity = self._validate_bundle(bundle, manifest)
            if manifest.fallback_only:
                self._remember_fast_winner(
                    bundle, launch_identity, manifest.fallback.digest,
                )
                return fallback_kernel

            # Workload identity and both cache levels precede effect-contract
            # parsing and no-alias checks. Alias topology and the raw
            # effect-contract digest are already part of the key.
            runtime_arguments = bind_runtime_arguments(specialization_kernel, bound_args)
            launch_arguments = _bind_launch_arguments(specialization_kernel, bound_args)
            runtime_bound_args = dict(runtime_arguments)
            effect_digest = _runtime_contract_digest(bundle.runtime_contract)
            workload_digest, workload = build_workload_key(
                grid,
                runtime_bound_args,
                source_identity=_source_identity(specialization_kernel),
                target_identity=_target_identity(specialization_kernel),
                device_identity=_device_identity(),
                domain_digest=manifest.digest,
                effect_digest=effect_digest,
            )
            memory_key = (manifest.digest, workload_digest, _sha256_json(bundle_identity))
            cached_digest = self._memory_cache.get(memory_key)
            cached_specification = next(
                (variant for variant in manifest.variants
                 if variant.digest == cached_digest),
                None,
            )
            if cached_specification is not None:
                compiled = bundle.get_or_compile(cached_specification)
                self._remember_fast_winner(bundle, launch_identity, cached_digest)
                _LOGGER.debug(
                    "replaying in-memory Gluon layout variant digest=%s without effect preflight",
                    cached_digest,
                )
                return compiled.kernel

            cache = self._get_cache_manager(self._cache_key(manifest, workload_digest))
            cached = self._read_record(
                cache, manifest, workload_digest, effect_digest, bundle_identity,
            )
            if cached is not None:
                cached_specification, cached_executable = cached
                compiled = bundle.get_or_compile(cached_specification)
                if _variant_executable_identity(compiled) == cached_executable:
                    self._memory_cache[memory_key] = cached_specification.digest
                    self._remember_fast_winner(
                        bundle, launch_identity, cached_specification.digest
                    )
                    _LOGGER.debug(
                        "replaying persistent Gluon layout variant digest=%s without effect preflight",
                        cached_specification.digest,
                    )
                    return compiled.kernel
                _LOGGER.warning(
                    "invalidating Gluon layout winner digest=%s because its executable identity changed",
                    cached_specification.digest,
                )

            contract = parse_runtime_contract(bundle.runtime_contract, len(runtime_arguments))
            contract = _remap_runtime_contract(contract, runtime_arguments, launch_arguments)
            args = [value for _, value in launch_arguments]
            _check_required_noalias(args, contract.require_noalias)
            replay = _prepare_scratch_replay(args, contract)
            normalized_grid = _normalize_grid(grid)
            if self._measurer is None and self._benchmarker is None:
                # Inputs may have been produced on the caller's stream. A fresh
                # worker owns a distinct stream/context, so establish visibility
                # once before exporting tensor storage through framework IPC.
                from triton.runtime.driver import driver

                driver.active.get_device_interface().synchronize()
            outcomes = self._evaluate_domain(
                bundle, manifest, normalized_grid, stream, replay
            )
            timings = outcomes.timings
            resources = outcomes.resources
            failures = outcomes.failures
            failure_details = outcomes.failure_details
            measurement_rounds = outcomes.rounds

            stability = self._apply_selection_policy(
                timings, resources, failures, failure_details,
                measurement_rounds, bundle, manifest, normalized_grid, stream,
                replay,
            )
            winner = select_measured_variant(timings, manifest.fallback)
            fallback_forced = winner not in stability.stable
            if fallback_forced:
                _LOGGER.debug("selecting fallback because variant digest=%s lacked stable gain", winner.digest)
                winner = manifest.fallback
            selection = {
                "initial_winner": stability.initial_winner.digest,
                "close": [variant.digest for variant in stability.close],
                "stable": [
                    variant.digest
                    for variant in manifest.variants
                    if variant in stability.stable
                ],
                "selected": winner.digest,
                "fallback_forced": fallback_forced,
            }
            self._write_record(
                cache, manifest, workload_digest, workload, winner, timings,
                resources, failures, failure_details, bundle, effect_digest,
                bundle_identity, measurement_rounds, selection,
            )
            self._memory_cache[memory_key] = winner.digest
            self._remember_fast_winner(bundle, launch_identity, winner.digest)
            _LOGGER.debug(
                "selected Gluon layout variant digest=%s file=%s workload=%s",
                winner.digest,
                winner.mlir_file,
                workload_digest,
            )
            return bundle.variants[winner.digest].kernel
        except FallbackVariantExecutionError:
            raise
        except Exception as error:
            warnings.warn(f"Gluon layout autotuning fell back to the compiler default: {error}", RuntimeWarning)
            return bundle.fallback_kernel
