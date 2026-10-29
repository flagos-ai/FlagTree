"""Versioned public compiler API for external TsingMicro CoreIR producers.

This module intentionally has no eager dependency on Triton or MLIR. Contract
objects can therefore be inspected, serialized, and unit-tested on a host that
does not have the TX81 toolchain. Compiler dependencies are imported only when
``compile_core_ir`` is called.
"""

from __future__ import annotations

import hashlib
import json
import re
import threading
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Literal

CORE_IR_VERSION = "tsingmicro.coreir/1"
KERNEL_ABI_VERSION = "tsingmicro.kernel-abi/1"

# The per-core SPM capacity, as the runtime reports it in
# `max_shared_mem` (driver.cpp getDeviceProperties). Single source of truth for
# every capacity check: Triton's own OutOfResources gate reads the same number
# through get_device_properties, and external CoreIR producers (TileLang)
# import it from here instead of pinning a private copy. A kernel allocated
# past this does not fail cleanly -- the launch hangs or faults the device.
SPM_CAPACITY_BYTES = 1024 * 1024 * 3 - 0x10000 - 0x10000
_SYMBOL_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_.$-]*$")
_DTYPE_RE = re.compile(r"^(?:[suf]?int(?:8|16|32|64)|float(?:16|32|64)|bfloat16|bool|index)$")
_COMPILE_LOCK = threading.RLock()


class CoreIRContractError(ValueError):
    """The request violates the public CoreIR contract."""


class CoreIRCompileError(RuntimeError):
    """FlagTree failed to lower a valid public CoreIR request."""


@dataclass(frozen=True)
class ArgumentSpec:
    name: str
    dtype: str
    kind: Literal["buffer", "scalar"] = "buffer"
    shape: tuple[int, ...] = ()
    address_space: Literal["global", "spm", "register"] = "global"
    readonly: bool = False

    def __post_init__(self) -> None:
        if not _SYMBOL_RE.fullmatch(self.name):
            raise CoreIRContractError(f"invalid argument name: {self.name!r}")
        if not _DTYPE_RE.fullmatch(self.dtype):
            raise CoreIRContractError(f"unsupported ABI dtype: {self.dtype!r}")
        if self.kind == "scalar" and self.shape:
            raise CoreIRContractError("scalar arguments must not declare a shape")
        if self.kind == "buffer" and not self.shape:
            raise CoreIRContractError("buffer arguments require a static shape")
        if any(not isinstance(extent, int) or extent == 0 or (extent < -1) for extent in self.shape):
            # A buffer dimension may be -1 (dynamic): the ABI carries an
            # unranked memref, so the shape is descriptive and its real value
            # travels as a scalar argument filled from the runtime tensor.
            raise CoreIRContractError(f"shape must contain positive static extents or -1: {self.shape!r}")
        if self.kind == "scalar" and self.address_space != "global":
            raise CoreIRContractError("scalar arguments use the global ABI value space")


@dataclass(frozen=True)
class KernelSignature:
    arguments: tuple[ArgumentSpec, ...]

    def __post_init__(self) -> None:
        names = [argument.name for argument in self.arguments]
        if len(names) != len(set(names)):
            raise CoreIRContractError(f"argument names must be unique: {names!r}")


@dataclass(frozen=True)
class Grid:
    x: int
    y: int = 1
    z: int = 1

    def __post_init__(self) -> None:
        # -1 marks a launch-time axis: the external producer computes the real
        # extent per launch (a dynamic shape's ceildiv) and overrides the grid
        # in CoreIRRuntime.launch. 0 and negatives other than -1 stay invalid.
        if any(dim != -1 and dim <= 0 for dim in (self.x, self.y, self.z)):
            raise CoreIRContractError("grid dimensions must be positive, or -1 for launch-time")


@dataclass(frozen=True)
class CompileOptions:
    arch: str = "tx81"
    num_stages: int = 2
    precision_mode: int = 0

    def __post_init__(self) -> None:
        if self.arch != "tx81":
            raise CoreIRContractError(f"unsupported TsingMicro architecture: {self.arch!r}")
        if self.num_stages not in (1, 2):
            raise CoreIRContractError("num_stages must be 1 or 2")
        if self.precision_mode not in (0, 1, 2):
            raise CoreIRContractError("precision_mode must be 0, 1, or 2")


@dataclass(frozen=True)
class CompileArtifact:
    kernel_name: str
    signature: KernelSignature
    grid: Grid
    options: CompileOptions
    binary: bytes = field(repr=False)
    kernel_path: str
    shared_memory_bytes: int
    compiler_hash: str
    source_hash: str
    core_ir_version: str = CORE_IR_VERSION
    kernel_abi_version: str = KERNEL_ABI_VERSION
    dumps: dict[str, str] = field(default_factory=dict, repr=False)
    diagnostics: tuple[str, ...] = ()

    def manifest(self) -> dict[str, object]:
        """Return stable JSON-compatible metadata; binary bytes are excluded."""
        return {
            "kernel_name": self.kernel_name,
            "signature": asdict(self.signature),
            "grid": asdict(self.grid),
            "options": asdict(self.options),
            "kernel_path": self.kernel_path,
            "shared_memory_bytes": self.shared_memory_bytes,
            "compiler_hash": self.compiler_hash,
            "source_hash": self.source_hash,
            "core_ir_version": self.core_ir_version,
            "kernel_abi_version": self.kernel_abi_version,
            "diagnostics": list(self.diagnostics),
        }

    def manifest_json(self) -> str:
        return json.dumps(self.manifest(), sort_keys=True, separators=(",", ":"))


def _validate_module(module: str, kernel_name: str) -> None:
    if not isinstance(module, str) or not module.strip():
        raise CoreIRContractError("module must be non-empty textual MLIR")
    if not _SYMBOL_RE.fullmatch(kernel_name):
        raise CoreIRContractError(f"invalid kernel name: {kernel_name!r}")
    escaped = re.escape(kernel_name)
    if re.search(rf"func\.func(?:\s+\w+)*\s+@{escaped}(?![A-Za-z0-9_.$-])", module) is None:
        raise CoreIRContractError(f"module does not define func.func @{kernel_name}")


def _compiler_hash() -> str:
    digest = hashlib.sha256()
    for source in (Path(__file__), Path(__file__).with_name("compiler.py")):
        digest.update(source.read_bytes())
    digest.update(CORE_IR_VERSION.encode())
    digest.update(KERNEL_ABI_VERSION.encode())
    return digest.hexdigest()


def compile_core_ir(
    *,
    module: str,
    kernel_name: str,
    signature: KernelSignature,
    grid: Grid,
    options: CompileOptions | None = None,
) -> CompileArtifact:
    """Compile version-1 CoreIR into a TX81 shared-object artifact.

    The input boundary is textual MLIR after frontend-specific conversion and
    before ``core-dialects-to-mk``. The compiler owns all subsequent MK, TX81,
    LLVM, object, and link stages.
    """
    options = options or CompileOptions()
    if not isinstance(signature, KernelSignature):
        raise CoreIRContractError("signature must be a KernelSignature")
    if not isinstance(grid, Grid):
        raise CoreIRContractError("grid must be a Grid")
    if not isinstance(options, CompileOptions):
        raise CoreIRContractError("options must be CompileOptions")
    _validate_module(module, kernel_name)

    source_hash = hashlib.sha256(module.encode("utf-8")).hexdigest()
    metadata: dict[str, object] = {"name": kernel_name, "shared": 0}
    try:
        # The legacy backend keeps dump indices and one pipeline feature bit in
        # process globals. Serialize this public entry until those internals are
        # made request-local, so concurrent callers cannot mix artifacts.
        with _COMPILE_LOCK:
            # Lazy import keeps contract discovery usable without the TX81 stack.
            from . import compiler as implementation

            implementation._ensure_tx81_llvm_environment()
            implementation.ir_index += 1
            prepared = implementation._prepare_external_coreir(
                module,
                precision_mode=options.precision_mode,
                num_stages=options.num_stages,
            )
            txir = implementation._optimize_txir(implementation._coreir_to_txir(prepared))
            llir = implementation._optimize_llir(implementation._txir_to_llir(txir, metadata))
            binary = implementation._llir_to_bin(llir, metadata)
    except Exception as err:
        raise CoreIRCompileError(f"FlagTree failed compiling @{kernel_name} for {options.arch}: {err}") from err

    actual_name = str(metadata.get("name", ""))
    if actual_name != kernel_name:
        raise CoreIRCompileError(f"compiled symbol mismatch: requested {kernel_name!r}, got {actual_name!r}")
    return CompileArtifact(
        kernel_name=kernel_name,
        signature=signature,
        grid=grid,
        options=options,
        binary=bytes(binary),
        kernel_path=str(metadata.get("kernel_path", "")),
        shared_memory_bytes=int(metadata.get("shared", 0)),
        compiler_hash=_compiler_hash(),
        source_hash=source_hash,
        dumps={"prepared_coreir": prepared, "txir": txir, "llir": llir},
    )
