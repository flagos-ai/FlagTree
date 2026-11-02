"""Public runtime adapter for artifacts produced by :mod:`.core_ir`.

The adapter reuses the existing TXDA driver and launcher instead of duplicating
argument packing or device-runtime calls. Importing this module is hardware
free; native launcher compilation and module loading are lazy.
"""

from __future__ import annotations

import os
import threading
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

from .core_ir import CompileArtifact


class CoreIRRuntimeError(RuntimeError):
    """A compiled CoreIR artifact could not be loaded or launched."""


@dataclass(frozen=True)
class LaunchDiagnostics:
    device: int
    stream: int
    grid: tuple[int, int, int]
    result_code: int
    log_buffer: str
    kernel_time_ns: int
    ap_duration_ns: int
    profiling_requested: bool
    timing_valid: bool
    timing_source: str


def _device_timing_requested() -> bool:
    return any(
        os.getenv(name, "0").lower() in ("1", "true", "yes")
        for name in ("TILELANG_TX_EVENT_TIMING", "TSM_PROFILER_EN"))


def _tsm_profiling_requested() -> bool:
    """Compatibility helper retained for the original profiler contract."""
    return os.getenv("TSM_PROFILER_EN", "0").lower() in ("1", "true", "yes")


def _signature_type(dtype: str, kind: str) -> str:
    if kind == "buffer":
        return "*" + {"float16": "fp16", "float32": "fp32"}.get(dtype, dtype)
    return {
        "bool": "i1",
        "int8": "i8",
        "int16": "i16",
        "int32": "i32",
        "int64": "i64",
        "uint8": "u8",
        "uint16": "u16",
        "uint32": "u32",
        "uint64": "u64",
        "float16": "fp16",
        "float32": "fp32",
        "float64": "fp64",
    }[dtype]


class CoreIRRuntime:
    """Lazy TXDA loader/launcher for one immutable compile artifact."""

    def __init__(self, artifact: CompileArtifact):
        if not isinstance(artifact, CompileArtifact):
            raise TypeError("artifact must be a CompileArtifact")
        self.artifact = artifact
        self._lock = threading.RLock()
        self._launcher = None
        self._utils = None
        self._loaded: dict[int, tuple[Any, int]] = {}

    def _ensure_launcher(self):
        with self._lock:
            if self._launcher is not None:
                return
            from .driver import TXDALauncher, TXDAUtils

            def kernel_placeholder():
                return None

            kernel_placeholder.__name__ = self.artifact.kernel_name
            kernel_placeholder.arg_names = [argument.name for argument in self.artifact.signature.arguments]
            signature = {
                index: _signature_type(argument.dtype, argument.kind)
                for index, argument in enumerate(self.artifact.signature.arguments)
            }
            source = SimpleNamespace(constants={}, signature=signature, fn=kernel_placeholder)
            metadata = SimpleNamespace(
                kernel_path=self.artifact.kernel_path,
                so_key=self.artifact.source_hash,
            )
            self._launcher = TXDALauncher(source, metadata)
            self._utils = TXDAUtils()

    def _load(self, device: int) -> int:
        self._ensure_launcher()
        with self._lock:
            loaded = self._loaded.get(device)
            if loaded is None:
                module, function, _registers, _spills, _threads = self._utils.load_binary(
                    self.artifact.kernel_name,
                    self.artifact.binary,
                    self.artifact.shared_memory_bytes,
                    device,
                )
                loaded = (module, int(function))
                self._loaded[device] = loaded
            return loaded[1]

    def launch(
        self,
        *arguments: Any,
        device: int | None = None,
        stream: int | None = None,
        grid: tuple[int, int, int] | None = None,
    ) -> LaunchDiagnostics:
        if len(arguments) != len(self.artifact.signature.arguments):
            raise CoreIRRuntimeError(
                f"@{self.artifact.kernel_name} expects {len(self.artifact.signature.arguments)} arguments, "
                f"got {len(arguments)}")
        try:
            import torch
            import torch_txda  # noqa: F401 - registers torch.txda

            if not hasattr(torch, "txda") or not torch.txda.is_available():
                raise CoreIRRuntimeError("torch.txda runtime is unavailable")
            resolved_device = torch.txda.current_device() if device is None else int(device)
            torch.txda.set_device(resolved_device)
            resolved_stream = (int(torch.txda.current_stream(resolved_device).txda_stream)
                               if stream is None else int(stream))
            function = self._load(resolved_device)
            # A launch-time grid (a dynamic shape's ceildiv) overrides the
            # artifact's placeholder.
            artifact_grid = self.artifact.grid
            launch_grid = grid if grid is not None else (artifact_grid.x, artifact_grid.y, artifact_grid.z)
            # TXDALauncher preserves Triton's four optional launch metadata/hook
            # slots even for external CoreIR producers.
            result = self._launcher(
                launch_grid[0],
                launch_grid[1],
                launch_grid[2],
                resolved_stream,
                function,
                None,
                None,
                None,
                None,
                *arguments,
            )
        except CoreIRRuntimeError:
            raise
        except Exception as err:
            raise CoreIRRuntimeError(f"failed launching @{self.artifact.kernel_name}: {err}") from err

        kernel_time_ns = int(result.kernel_time_ns)
        ap_duration_ns = int(result.ap_duration_ns)
        timing_kind = int(getattr(result, "timing_kind", 0))
        profiling_requested = _device_timing_requested()
        timing_valid = profiling_requested and kernel_time_ns > 0
        timing_sources = {1: "tx-event", 2: "tsm-profiler"}
        diagnostics = LaunchDiagnostics(
            device=resolved_device,
            stream=resolved_stream,
            grid=launch_grid,
            result_code=int(result.res),
            log_buffer=str(result.log_buffer),
            kernel_time_ns=kernel_time_ns,
            ap_duration_ns=ap_duration_ns,
            profiling_requested=profiling_requested,
            timing_valid=timing_valid,
            timing_source=(timing_sources.get(timing_kind, "device-timing-unavailable")
                           if profiling_requested else "disabled"),
        )
        if diagnostics.result_code != 0:
            raise CoreIRRuntimeError(f"@{self.artifact.kernel_name} launch failed with code {diagnostics.result_code}: "
                                     f"{diagnostics.log_buffer}")
        return diagnostics


def load_core_ir_artifact(artifact: CompileArtifact) -> CoreIRRuntime:
    """Create a lazy runtime object without selecting or touching a device."""

    return CoreIRRuntime(artifact)
