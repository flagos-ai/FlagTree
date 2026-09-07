# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Prepared direct C-launcher wrapper for FlagTree Triton kernels.

This is a Python-native restructuring of the prepared-launch contract used by
``../nncase/pyntt/pyntt/runtime/triton.py`` at nncase commit 2823f485.  It does
not require a fork-only ``JITFunction.prepare`` method: compilation happens
once through the local JIT API, then steady-state launches call the compiled
C launcher directly.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import prod
from typing import Any, Sequence

from triton.flagmega.errors import RuntimeContractError


@dataclass(frozen=True)
class ResourceContract:
    compute_num_warps: int
    resident_blocks_per_sm: int
    registers_per_thread_limit: int = 255
    register_file_capacity_per_sm: int = 65536
    shared_memory_capacity_bytes: int = 227328
    forbid_spills: bool = True

    def __post_init__(self) -> None:
        positive = (
            self.compute_num_warps,
            self.resident_blocks_per_sm,
            self.registers_per_thread_limit,
            self.register_file_capacity_per_sm,
            self.shared_memory_capacity_bytes,
        )
        if any(value <= 0 for value in positive):
            raise RuntimeContractError("Prepared launch resource requirements and capacities must be positive.")


class PreparedKernel:
    """A compiled specialization with immutable static argument bindings.

    Compiler-requested global scratch is allocated once with the specialization
    and scoped to each direct launch.  A prepared specialization is therefore a
    single-stream object when it owns scratch; callers must prepare a separate
    instance for concurrent streams.
    """

    def __init__(
        self,
        compiled_kernel,
        arguments: Sequence[object],
        dynamic_argument_indices: Sequence[int],
        *,
        grid: Sequence[int],
        contract: ResourceContract,
    ) -> None:
        self.compiled_kernel = compiled_kernel
        self._arguments = tuple(arguments)
        self._dynamic_indices = tuple(int(value) for value in dynamic_argument_indices)
        self.grid = _normalize_grid(grid)
        self.contract = contract
        self._global_scratch = _prepare_global_scratch(compiled_kernel, self._arguments, self.grid)
        self._scratch_stream = None
        if len(set(self._dynamic_indices)) != len(self._dynamic_indices):
            raise RuntimeContractError("Prepared dynamic argument indices must be unique.")
        if any(value < 0 or value >= len(self._arguments) for value in self._dynamic_indices):
            raise RuntimeContractError("Prepared dynamic argument index is outside the kernel signature.")

    @property
    def dynamic_argument_count(self) -> int:
        return len(self._dynamic_indices)

    @property
    def static_argument_indices(self) -> tuple[int, ...]:
        dynamic = set(self._dynamic_indices)
        return tuple(index for index in range(len(self._arguments)) if index not in dynamic)

    @property
    def resource_report(self) -> dict[str, object]:
        compiled = self.compiled_kernel
        registers = int(compiled.n_regs)
        shared = int(compiled.metadata.shared)
        return {
            "name": str(compiled.name),
            "num_warps": int(compiled.metadata.num_warps),
            "registers_per_thread": registers,
            "shared_memory_bytes": shared,
            **_spill_resource_report(compiled),
            "required_resident_blocks_per_sm": self.contract.resident_blocks_per_sm,
            "resident_registers": registers * int(compiled.metadata.num_warps) * 32
            * self.contract.resident_blocks_per_sm,
            "resident_shared_memory_bytes": shared * self.contract.resident_blocks_per_sm,
            "global_scratch_bytes": 0 if self._global_scratch is None else self._global_scratch.nbytes,
            "cooperative_grid": bool(getattr(compiled.metadata, "launch_cooperative_grid", False)),
            "grid": list(self.grid),
        }

    def launch(self, *dynamic_arguments: object, stream=None) -> None:
        if len(dynamic_arguments) != len(self._dynamic_indices):
            raise RuntimeContractError(
                f"Prepared kernel expects {len(self._dynamic_indices)} dynamic arguments, "
                f"got {len(dynamic_arguments)}.")
        arguments = list(self._arguments)
        for index, value in zip(self._dynamic_indices, dynamic_arguments):
            arguments[index] = value
        compiled = self.compiled_kernel
        if stream is None:
            from triton.runtime.driver import driver

            device = driver.active.get_current_device()
            stream = driver.active.get_current_stream(device)
        if self._global_scratch is not None:
            stream_identity = _stream_identity(stream)
            if self._scratch_stream is None:
                self._scratch_stream = stream_identity
            elif self._scratch_stream != stream_identity:
                raise RuntimeContractError(
                    "A prepared kernel with compiler global scratch is bound to its first launch stream; "
                    "prepare another instance for concurrent or different-stream execution.")
        from triton import knobs
        from triton.runtime._distributed import DistributedRtContext

        launch_metadata = compiled.launch_metadata(self.grid, stream, *arguments)
        distributed_arguments = []
        distributed = DistributedRtContext()
        if distributed.is_lite_mode:
            distributed_arguments.extend((distributed.comm_ptr, distributed.mem_ptr))
        from triton.runtime import _allocation

        token = None
        if self._global_scratch is not None:
            token = _allocation._allocator.set(self._global_scratch)
        try:
            compiled.run(
                self.grid[0],
                self.grid[1],
                self.grid[2],
                stream,
                compiled.function,
                compiled.packed_metadata,
                launch_metadata,
                knobs.runtime.launch_enter_hook,
                knobs.runtime.launch_exit_hook,
                *distributed_arguments,
                *arguments,
            )
        finally:
            if token is not None:
                _allocation._allocator.reset(token)


class _PreparedGlobalScratch:
    def __init__(self, storage, buffer, nbytes: int, alignment: int) -> None:
        self.storage = storage
        self.buffer = buffer
        self.nbytes = int(nbytes)
        self.alignment = int(alignment)

    def __call__(self, size: int, alignment: int, stream):
        if int(size) != self.nbytes:
            raise RuntimeContractError(
                f"Compiled launch requested {size} scratch bytes; prepared allocation has {self.nbytes}.")
        if int(alignment) > self.alignment or self.buffer.data_ptr() % int(alignment):
            raise RuntimeContractError(
                f"Compiled launch requested scratch alignment {alignment}; "
                f"prepared allocation guarantees {self.alignment}.")
        return self.buffer


def _prepare_global_scratch(compiled, arguments: Sequence[object], grid: Sequence[int]):
    metadata = compiled.metadata
    per_program = int(getattr(metadata, "global_scratch_size", 0) or 0)
    if per_program <= 0:
        return None
    num_ctas = int(getattr(metadata, "num_ctas", 1) or 1)
    nbytes = prod(int(value) for value in grid) * num_ctas * per_program
    alignment = max(1, int(getattr(metadata, "global_scratch_align", 1) or 1))
    device = next(
        (
            value.device
            for value in arguments
            if getattr(getattr(value, "device", None), "type", None) == "cuda"
        ),
        None,
    )
    if device is None:
        raise RuntimeContractError("Compiler global scratch requires at least one CUDA tensor argument.")
    try:
        import torch
    except ImportError as error:
        raise RuntimeContractError("Compiler global scratch allocation requires PyTorch.") from error
    # TLE's grid barrier is a reusable sense-reversing counter.  Its atomic
    # protocol preserves the phase bit between barriers, but requires the
    # arrival-count low bits to start at zero.  An uninitialized CUDA
    # allocation can therefore release a subset of CTAs before every owner has
    # published its workspace writes.  Initialize the complete compiler-owned
    # scratch allocation once at prepare time; subsequent launches reuse the
    # phase-safe counter state on the same stream.
    storage = torch.zeros(
        nbytes + alignment - 1, dtype=torch.uint8, device=device
    )
    # ``prepare`` may be followed by a first launch on a different CUDA
    # stream.  Make the one-time counter initialization globally visible
    # before the prepared object is published; steady-state launches remain
    # asynchronous and are still restricted to one stream below.
    torch.cuda.synchronize(device)
    byte_offset = (-storage.data_ptr()) % alignment
    buffer = storage.narrow(0, byte_offset, nbytes)
    return _PreparedGlobalScratch(storage, buffer, nbytes, alignment)


def prepare_jit_kernel(
    kernel,
    arguments: Sequence[object],
    dynamic_argument_indices: Sequence[int],
    *,
    grid: Sequence[int],
    contract: ResourceContract,
    **launch_options: object,
) -> PreparedKernel:
    """Compile one JIT specialization, validate resources, and bind statics."""

    requested_warps = int(launch_options.get("num_warps", contract.compute_num_warps))
    if requested_warps != contract.compute_num_warps:
        raise RuntimeContractError(
            f"Launch requests {requested_warps} warps; contract requires {contract.compute_num_warps}.")
    normalized_grid = _normalize_grid(grid)
    compiled = kernel.run(
        *arguments,
        grid=normalized_grid,
        warmup=True,
        **launch_options,
    )
    if compiled is None:
        raise RuntimeContractError("Triton compilation returned no compiled kernel.")
    if hasattr(compiled, "result"):
        compiled = compiled.result()
    _validate_resources(compiled, contract)
    return PreparedKernel(
        compiled,
        arguments,
        dynamic_argument_indices,
        grid=normalized_grid,
        contract=contract,
    )


def _validate_resources(compiled, contract: ResourceContract) -> None:
    # Materialize the module/function and resource counters without launching.
    _ = compiled.run
    actual_warps = int(compiled.metadata.num_warps)
    if actual_warps < contract.compute_num_warps:
        raise RuntimeContractError(
            f"Compiled kernel has {actual_warps} warps; contract requires at least {contract.compute_num_warps}.")
    registers = int(compiled.n_regs)
    if registers > contract.registers_per_thread_limit:
        raise RuntimeContractError(
            f"Compiled kernel uses {registers} registers/thread; limit is {contract.registers_per_thread_limit}.")
    resident_registers = registers * actual_warps * 32 * contract.resident_blocks_per_sm
    if resident_registers > contract.register_file_capacity_per_sm:
        raise RuntimeContractError(
            f"Compiled kernel needs at least {resident_registers} registers for "
            f"{contract.resident_blocks_per_sm} resident block(s); SM capacity is "
            f"{contract.register_file_capacity_per_sm}.")
    shared = int(compiled.metadata.shared)
    if shared > contract.shared_memory_capacity_bytes:
        raise RuntimeContractError(
            f"Compiled kernel uses {shared} shared-memory bytes; limit is {contract.shared_memory_capacity_bytes}.")
    resident_shared = shared * contract.resident_blocks_per_sm
    if resident_shared > contract.shared_memory_capacity_bytes:
        raise RuntimeContractError(
            f"Compiled kernel needs {resident_shared} shared-memory bytes for "
            f"{contract.resident_blocks_per_sm} resident block(s); SM capacity is "
            f"{contract.shared_memory_capacity_bytes}.")
    resources = _spill_resource_report(compiled)
    if contract.forbid_spills and resources["spill_bytes"]:
        raise RuntimeContractError(
            f"Compiled kernel has {resources['spill_store_bytes']} spill-store bytes and "
            f"{resources['spill_load_bytes']} spill-load bytes with {registers} registers/thread, "
            f"{shared} shared-memory bytes, {resources['stack_frame_bytes']} stack-frame bytes, and "
            f"{resources['local_memory_bytes']} local-memory bytes; the contract forbids spills.")


def _spill_resource_report(compiled) -> dict[str, int]:
    metadata = compiled.metadata
    fields = ("stack_frame_bytes", "spill_store_bytes", "spill_load_bytes")
    if any(not hasattr(metadata, f"ptxas_{field}") for field in fields):
        raise RuntimeContractError(
            "Compiled kernel lacks assembler resource metadata; recompile with a TLE backend "
            "that records per-function stack and spill usage.")
    report = {field: int(getattr(metadata, f"ptxas_{field}")) for field in fields}
    # The NVIDIA driver wrapper returns LOCAL_SIZE_BYTES / sizeof(int), not
    # spill bytes. Keep the historical report key for the *sum of assembler
    # spill-store/load byte counts*, and expose each component explicitly.
    report["local_memory_bytes"] = int(compiled.n_spills) * 4
    report["spill_bytes"] = report["spill_store_bytes"] + report["spill_load_bytes"]
    return report


def _normalize_grid(grid: Sequence[int]) -> tuple[int, int, int]:
    values = tuple(int(value) for value in grid)
    if not values or len(values) > 3 or any(value <= 0 for value in values):
        raise RuntimeContractError(f"Prepared grid must contain one to three positive dimensions, got {values}.")
    return values + (1, ) * (3 - len(values))


def _stream_identity(stream: object) -> object:
    value = getattr(stream, "cuda_stream", stream)
    try:
        return int(value)
    except (TypeError, ValueError):
        return value
