# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Serializable module-level physical bufferization plan."""

from __future__ import annotations

from dataclasses import dataclass, replace
from functools import cached_property
from typing import Any, Mapping

from triton.flagmega.errors import IRVerificationError
from triton.flagmega.ir.bufferization.alias import AliasInfo, AliasKind
from triton.flagmega.ir.bufferization.descriptor import BufferDescriptor
from triton.flagmega.ir.bufferization.function_plan import FunctionBufferPlan
from triton.flagmega.ir.bufferization.mem_span import MemSpan
from triton.flagmega.ir.bufferization.memory import MemorySpace
from triton.flagmega.ir.bufferization.physical_buffer import PhysicalBuffer


BUFFER_PLAN_SCHEMA = "flagmega.buffer-plan/v6"
LEGACY_BUFFER_PLAN_SCHEMA = "flagmega.buffer-plan/v5"


@dataclass(frozen=True)
class BufferPlan:
    buffers: tuple[BufferDescriptor, ...]
    physical_buffers: tuple[PhysicalBuffer, ...]
    memory_spaces: tuple[MemorySpace, ...]
    functions: tuple[FunctionBufferPlan, ...]
    workspace_bytes: int
    rdata_bytes: int
    alignment: int
    entry_inputs: tuple[tuple[str, tuple[str, ...]], ...]
    entry_outputs: tuple[tuple[str, tuple[str, ...]], ...]
    allocator: str = "ortools-cp-sat"
    default_workspace: str = "workspace"

    @cached_property
    def buffer_map(self) -> Mapping[str, BufferDescriptor]:
        return {value.id: value for value in self.buffers}

    @cached_property
    def physical_buffer_map(self) -> Mapping[str, PhysicalBuffer]:
        return {value.id: value for value in self.physical_buffers}

    @property
    def allocations(self) -> tuple[PhysicalBuffer, ...]:
        """Compatibility view for buffer-plan/v2 consumers."""

        return self.physical_buffers

    @property
    def allocation_map(self) -> Mapping[str, PhysicalBuffer]:
        return self.physical_buffer_map

    @cached_property
    def function_map(self) -> Mapping[str, FunctionBufferPlan]:
        return {value.name: value for value in self.functions}

    @cached_property
    def memory_space_map(self) -> Mapping[str, MemorySpace]:
        return {value.name: value for value in self.memory_spaces}

    @cached_property
    def workspace_memory_space(self) -> MemorySpace:
        try:
            return self.memory_space_map[self.default_workspace]
        except KeyError as error:
            raise IRVerificationError(
                f"Default workspace {self.default_workspace!r} is not a declared "
                "memory space."
            ) from error

    @cached_property
    def call_map(self):
        return {call.call: call for function in self.functions for call in function.calls}

    @cached_property
    def kernel_call_map(self):
        return {
            call.call: call
            for function in self.functions
            for call in function.kernel_calls
        }

    def resolve_kernel_workspace(self, call_id: str, formal: str) -> BufferDescriptor:
        """Resolve a selected PrimFunction workspace parameter in its caller."""

        try:
            call = self.kernel_call_map[call_id]
            actual_id = dict(call.workspaces)[formal]
            return self.buffer_map[actual_id]
        except KeyError as error:
            raise IRVerificationError(
                f"Kernel call {call_id!r} has no workspace parameter {formal!r}."
            ) from error

    def function_memory_space_bytes(
        self, function_name: str, memory_space: str
    ) -> int:
        """Return the per-scope high-water mark for one function pool."""

        try:
            function = self.function_map[function_name]
            space = self.memory_space_map[memory_space]
        except KeyError as error:
            raise IRVerificationError(
                f"Unknown function/memory space {function_name!r}/{memory_space!r}."
            ) from error
        pool = function.memory_pool_map.get(memory_space)
        if pool is not None:
            return pool.scope_bytes
        high_water = max(
            (
                value.offset + value.nbytes
                for value in self.physical_buffers
                if value.function == function_name
                and value.memory_space == memory_space
            ),
            default=0,
        )
        return space.allocation_bytes(high_water)

    def resolve_call_buffer(self, call_id: str, buffer_id: str) -> BufferDescriptor:
        """Resolve one callee-local logical buffer at a concrete call site."""

        try:
            call = self.call_map[call_id]
            descriptor = self.buffer_map[buffer_id]
        except KeyError as error:
            raise IRVerificationError(
                f"Cannot resolve buffer {buffer_id!r} at unknown/incomplete call {call_id!r}."
            ) from error
        if descriptor.function != call.callee:
            raise IRVerificationError(
                f"Buffer {buffer_id!r} belongs to @{descriptor.function}, not @{call.callee}."
            )
        result_map = dict(call.results)
        if buffer_id in result_map:
            return self.buffer_map[result_map[buffer_id]]
        argument_map = dict(call.arguments)
        if buffer_id in argument_map:
            return self.buffer_map[argument_map[buffer_id]]
        parameter_roots = {
            self.buffer_map[formal].mem_span.buffer.id: (formal, actual)
            for formal, actual in call.arguments
        }
        if descriptor.mem_span.buffer.id in parameter_roots:
            formal_id, actual_id = parameter_roots[descriptor.mem_span.buffer.id]
            formal = self.buffer_map[formal_id]
            actual = self.buffer_map[actual_id]
            relative = (descriptor.mem_span.start - formal.mem_span.start).simplify()
            return replace(
                descriptor,
                storage=actual.storage,
                mem_span=MemSpan(
                    actual.mem_span.buffer,
                    (actual.mem_span.start + relative).simplify(),
                    descriptor.mem_span.size,
                ),
                alias=AliasInfo(actual.id, AliasKind.PARAMETER),
                function=call.caller,
                role="call_parameter_alias",
            )
        output_roots = {
            self.buffer_map[formal].mem_span.buffer.id: (formal, actual)
            for formal, actual in call.results
        }
        if descriptor.mem_span.buffer.id in output_roots:
            formal_id, actual_id = output_roots[descriptor.mem_span.buffer.id]
            formal = self.buffer_map[formal_id]
            actual = self.buffer_map[actual_id]
            relative = (descriptor.mem_span.start - formal.mem_span.start).simplify()
            return replace(
                descriptor,
                storage=actual.storage,
                mem_span=MemSpan(
                    actual.mem_span.buffer,
                    (actual.mem_span.start + relative).simplify(),
                    descriptor.mem_span.size,
                ),
                alias=AliasInfo(actual.id, AliasKind.RESULT),
                function=call.caller,
                role="call_result_alias",
            )
        memory_space = descriptor.mem_span.buffer.memory_space
        space = self.memory_space_map.get(memory_space)
        if (
            space is not None
            and space.allocation_scope.value == "function"
            and space.kind != "shared"
        ):
            pool = call.memory_pool_map.get(memory_space)
            if pool is None or pool.allocation is None:
                raise IRVerificationError(
                    f"Call {call_id!r} has no {memory_space!r} PhysicalBuffer "
                    f"for {buffer_id!r}."
                )
            frame = self.physical_buffer_map[pool.allocation]
            return replace(
                descriptor,
                mem_span=MemSpan(frame, descriptor.mem_span.absolute_start, descriptor.mem_span.size),
                function=call.caller,
                role="call_memory_pool",
            )
        # Rdata/state do not acquire a call-frame base.
        return descriptor

    def to_data(self) -> dict[str, object]:
        return {
            "schema": BUFFER_PLAN_SCHEMA,
            "allocator": self.allocator,
            "buffers": [buffer.to_data() for buffer in self.buffers],
            "physical_buffers": [value.to_data() for value in self.physical_buffers],
            "memory_spaces": [value.to_data() for value in self.memory_spaces],
            "functions": [value.to_data() for value in self.functions],
            "workspace_bytes": self.workspace_bytes,
            "rdata_bytes": self.rdata_bytes,
            "alignment": self.alignment,
            "entry_inputs": [
                {"node": node, "buffers": list(buffers)} for node, buffers in self.entry_inputs
            ],
            "entry_outputs": [
                {"node": node, "buffers": list(buffers)} for node, buffers in self.entry_outputs
            ],
            "default_workspace": self.default_workspace,
        }

    @classmethod
    def from_data(cls, data: Mapping[str, Any]) -> BufferPlan:
        if data.get("schema") not in {BUFFER_PLAN_SCHEMA, LEGACY_BUFFER_PLAN_SCHEMA}:
            raise IRVerificationError(
                f"Unsupported buffer-plan schema {data.get('schema')!r}; expected {BUFFER_PLAN_SCHEMA}."
            )
        bindings = lambda values: tuple(
            (str(value["node"]), tuple(str(item) for item in value.get("buffers", ())))
            for value in values
        )
        physical_buffers = tuple(
            PhysicalBuffer.from_data(value) for value in data.get("physical_buffers", ())
        )
        physical_buffer_map = {value.id: value for value in physical_buffers}
        memory_spaces = tuple(
            MemorySpace.from_data(value) for value in data.get("memory_spaces", ())
        )
        schema = str(data.get("schema"))
        if schema == LEGACY_BUFFER_PLAN_SCHEMA:
            candidates = tuple(
                value
                for value in memory_spaces
                if value.strategy.value == "sat"
                and value.allocation_scope.value == "function"
                and value.kind != "shared"
            )
            named = tuple(value for value in candidates if value.name == "workspace")
            if len(named) == 1:
                default_workspace = named[0].name
            elif len(candidates) == 1:
                default_workspace = candidates[0].name
            else:
                raise IRVerificationError(
                    "Legacy buffer-plan/v5 requires exactly one function SAT "
                    "workspace memory space."
                )
        else:
            default_workspace = str(data.get("default_workspace", ""))
        return cls(
            buffers=tuple(
                BufferDescriptor.from_data(value, physical_buffer_map)
                for value in data.get("buffers", ())
            ),
            physical_buffers=physical_buffers,
            memory_spaces=memory_spaces,
            functions=tuple(
                FunctionBufferPlan.from_data(
                    value, legacy_memory_space=default_workspace
                )
                for value in data.get("functions", ())
            ),
            workspace_bytes=int(data.get("workspace_bytes", -1)),
            rdata_bytes=int(data.get("rdata_bytes", -1)),
            alignment=int(data.get("alignment", 0)),
            entry_inputs=bindings(data.get("entry_inputs", ())),
            entry_outputs=bindings(data.get("entry_outputs", ())),
            allocator=str(data.get("allocator", "")),
            default_workspace=default_workspace,
        )


__all__ = ["BUFFER_PLAN_SCHEMA", "LEGACY_BUFFER_PLAN_SCHEMA", "BufferPlan"]
