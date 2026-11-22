# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Typed CPU/device execution programs and their publication protocol."""

from __future__ import annotations

from dataclasses import dataclass

from triton.flagmega.errors import IRSchemaError
from triton.flagmega.ir.tir.base import TIRCost, TIRNode, TIRStmt, tir_node
from triton.flagmega.ir.tir.buffer import Buffer
from triton.flagmega.ir.tir.execution_function import ExecutionFunction
from triton.flagmega.ir.tir.kernel_invoke import KernelInvoke
from triton.flagmega.ir.tir.prim_function_call import PrimFunctionCall
from triton.flagmega.ir.tir.synchronization_range import SynchronizationRange
from triton.flagmega.ir.tir.workspace_requirement import WorkspaceRequirement


def _name(value, field):
    if not isinstance(value, str) or not value.strip():
        raise IRSchemaError(f"Heterogeneous {field} requires a non-empty string.")


def _uint(value, field, *, positive=False):
    if type(value) is not int or not (int(positive) <= value < 2 ** 32):
        raise IRSchemaError(f"Heterogeneous {field} requires a bounded uint32.")


def _values(owner, field, kind):
    values = tuple(getattr(owner, field))
    if any(not isinstance(value, kind) for value in values):
        raise IRSchemaError(f"Heterogeneous {field} requires {kind.__name__} values.")
    object.__setattr__(owner, field, values)
    return values


@tir_node("execution_placement")
@dataclass(frozen=True)
class ExecutionPlacement(TIRNode):
    source_function: str
    call_id: str
    kernel: str
    domain: str
    implementation: str
    projections: int = 0

    def __post_init__(self):
        for field in ("source_function", "call_id", "kernel", "domain", "implementation"):
            _name(getattr(self, field), field)
        _uint(self.projections, "projection count")


@tir_node("heterogeneous_plan")
@dataclass(frozen=True)
class HeterogeneousPlan(TIRNode):
    name: str
    source_entry: str
    protocol: str
    placements: tuple[ExecutionPlacement, ...]
    resources: tuple[WorkspaceRequirement, ...] = ()
    cpu_domain: str = "cpu0"
    device_domain: str = "kpu"
    slots: int = 2
    source_hash: str | None = None

    def __post_init__(self):
        for field in ("name", "source_entry", "protocol", "cpu_domain", "device_domain"):
            _name(getattr(self, field), field)
        _values(self, "placements", ExecutionPlacement)
        _values(self, "resources", WorkspaceRequirement)
        if self.cpu_domain == self.device_domain or self.slots != 2 or type(self.slots) is not int:
            raise IRSchemaError("Heterogeneous v1 requires two distinct domains and two slots.")
        if self.source_hash is not None and (
            not isinstance(self.source_hash, str) or len(self.source_hash) != 64
            or any(value not in "0123456789abcdef" for value in self.source_hash)
        ):
            raise IRSchemaError("Heterogeneous source_hash requires a lowercase SHA256.")


@tir_node("kernel_phase_invoke")
@dataclass(frozen=True)
class KernelPhaseInvoke(TIRStmt):
    call_id: str
    origin_call_id: str
    phase: str
    projection: int
    exchange: str
    invocation: KernelInvoke
    dependencies: tuple[str, ...] = ()

    def __post_init__(self):
        for field in ("call_id", "origin_call_id", "exchange"):
            _name(getattr(self, field), field)
        if self.phase not in {"prepare", "execute", "finish"}:
            raise IRSchemaError("Kernel phase must be prepare, execute or finish.")
        _uint(self.projection, "projection index")
        if not isinstance(self.invocation, KernelInvoke) or self.invocation.call_id != self.origin_call_id:
            raise IRSchemaError("Kernel phase must retain its exact source KernelInvoke.")
        dependencies = _values(self, "dependencies", str)
        if len(set(dependencies)) != len(dependencies) or self.call_id in dependencies or any(not v for v in dependencies):
            raise IRSchemaError("Kernel phase dependencies must be unique other call identities.")

    @property
    def local_cost(self):
        return TIRCost(flops=None, bytes_read=None, bytes_written=None)


@tir_node("pipeline_exchange")
@dataclass(frozen=True)
class PipelineExchange(TIRNode):
    name: str
    source_function: str
    origin_call_id: str
    projection: int
    inputs: tuple[Buffer, ...]
    outputs: tuple[Buffer, ...]
    cache_line_bytes: int = 64
    slot_stride_bytes: int = 0

    def __post_init__(self):
        for field in ("name", "source_function", "origin_call_id"):
            _name(getattr(self, field), field)
        _uint(self.projection, "projection index")
        inputs = _values(self, "inputs", Buffer)
        outputs = _values(self, "outputs", Buffer)
        if not inputs or not outputs:
            raise IRSchemaError("Pipeline exchange requires typed input and output storage.")
        line = self.cache_line_bytes
        if type(line) is not int or line <= 0 or line & (line - 1):
            raise IRSchemaError("Pipeline exchange cache line must be a positive power of two.")
        _uint(self.slot_stride_bytes, "slot stride")
        if self.slot_stride_bytes % line:
            raise IRSchemaError("Pipeline exchange slot stride must be cache-line aligned.")
        names = [value.name for value in (*inputs, *outputs)]
        if len(set(names)) != len(names):
            raise IRSchemaError("Pipeline exchange field names must be unique.")
        for value in (*inputs, *outputs):
            span = value.mem_span
            if (not span.absolute_start.is_fixed or not span.size.is_fixed
                    or span.size.fixed_value <= 0 or span.absolute_start.fixed_value % line
                    or span.size.fixed_value % line or span.buffer.alignment < line
                    or value.distributed_type is not None
                    or any(not dim.is_fixed for dim in (*value.dimensions, *value.strides))):
                raise IRSchemaError("Pipeline exchange requires static, cache-line-isolated physical storage.")
            if any(dim.fixed_value <= 0 for dim in value.dimensions):
                raise IRSchemaError("Pipeline exchange dimensions must be positive.")
            extent = (1 + sum((dim.fixed_value - 1) * stride.fixed_value
                             for dim, stride in zip(value.dimensions, value.strides))) * value.elem_type.itemsize
            if extent > span.size.fixed_value:
                raise IRSchemaError("Pipeline exchange tensor footprint exceeds its physical storage.")
            if self.slot_stride_bytes and span.absolute_end.fixed_value > self.slot_stride_bytes:
                raise IRSchemaError("Pipeline exchange field exceeds one slot stride.")


@tir_node("value_exchange")
@dataclass(frozen=True)
class ValueExchange(TIRNode):
    """A source dependency published through one shared atomic state word."""

    name: str
    source_function: str
    producer: str
    consumer: str
    producer_domain: str
    consumer_domain: str
    values: tuple[Buffer, ...]
    state_word: Buffer
    producer_ranges: tuple[SynchronizationRange, ...]
    consumer_ranges: tuple[SynchronizationRange, ...]

    def __post_init__(self):
        for field in ("name", "source_function", "producer", "consumer", "producer_domain", "consumer_domain"):
            _name(getattr(self, field), field)
        if self.producer == self.consumer or self.producer_domain == self.consumer_domain:
            raise IRSchemaError("Value exchange requires distinct source events and devices.")
        _values(self, "values", Buffer)
        _values(self, "producer_ranges", SynchronizationRange)
        _values(self, "consumer_ranges", SynchronizationRange)
        word = self.state_word
        if not isinstance(word, Buffer):
            raise IRSchemaError("Value exchange requires a typed state word.")
        span = word.mem_span
        if (getattr(word.elem_type, "value", None) != "int32"
                or any(not d.is_fixed for d in (*word.dimensions, *word.strides))
                or tuple(d.fixed_value for d in word.dimensions) != (1,)
                or tuple(d.fixed_value for d in word.strides) != (1,)
                or not span.absolute_start.is_fixed or span.absolute_start.fixed_value % 4
                or not span.size.is_fixed or span.size.fixed_value != 4 or span.buffer.alignment < 4):
            raise IRSchemaError("Value exchange state word must be one aligned int32 storage word with uint32 ticket bits.")
        for value in self.values:
            if (not value.mem_span.absolute_start.is_fixed or not value.mem_span.size.is_fixed
                    or any(not d.is_fixed for d in (*value.dimensions, *value.strides))):
                raise IRSchemaError("Value exchange requires static typed payload ranges.")


@tir_node("channel_barrier")
@dataclass(frozen=True)
class ChannelBarrier(TIRStmt):
    call_id: str
    exchange: str
    event: str
    sequence: int
    after: tuple[str, ...]
    ranges: tuple[SynchronizationRange, ...]
    epoch_resource: str = "invocation"

    def __post_init__(self):
        for field in ("call_id", "exchange", "epoch_resource"):
            _name(getattr(self, field), field)
        if self.event not in {"publish", "acquire", "complete", "observe", "reclaim"}:
            raise IRSchemaError("Unknown heterogeneous channel barrier event.")
        _uint(self.sequence, "phase sequence", positive=True)
        after = _values(self, "after", str)
        _values(self, "ranges", SynchronizationRange)
        if len(set(after)) != len(after) or self.call_id in after or any(not value for value in after):
            raise IRSchemaError("Channel barrier dependencies must name distinct other events.")
        if not self.ranges:
            raise IRSchemaError("Channel barrier must protect explicit physical ranges.")

    @property
    def local_cost(self):
        return TIRCost(synchronizations=1)


@tir_node("worker_call")
@dataclass(frozen=True)
class WorkerCall(TIRStmt):
    invocation: PrimFunctionCall
    phase_offset: int

    def __post_init__(self):
        if not isinstance(self.invocation, PrimFunctionCall):
            raise IRSchemaError("WorkerCall requires a bound PrimFunctionCall.")
        _uint(self.phase_offset, "callee phase offset")

    @property
    def call_id(self):
        return self.invocation.call_id


@tir_node("worker_function")
@dataclass(frozen=True)
class WorkerFunction(TIRNode):
    source_function: str
    domain: str
    function: ExecutionFunction
    phase_count: int

    def __post_init__(self):
        _name(self.source_function, "source function")
        _name(self.domain, "worker domain")
        if not isinstance(self.function, ExecutionFunction):
            raise IRSchemaError("WorkerFunction requires an ExecutionFunction.")
        _uint(self.phase_count, "function phase count")


@tir_node("heterogeneous_launch")
@dataclass(frozen=True)
class HeterogeneousLaunch(TIRStmt):
    cpu_entry: str
    device_entry: str
    epoch_resource: str = "invocation"

    def __post_init__(self):
        for field in ("cpu_entry", "device_entry", "epoch_resource"):
            _name(getattr(self, field), field)
        if self.cpu_entry == self.device_entry:
            raise IRSchemaError("Heterogeneous launch requires distinct worker entries.")


@tir_node("heterogeneous_drain")
@dataclass(frozen=True)
class HeterogeneousDrain(TIRStmt):
    after: tuple[str, ...]
    epoch_resource: str = "invocation"

    def __post_init__(self):
        _name(self.epoch_resource, "epoch resource")
        after = _values(self, "after", str)
        if len(after) != 2 or len(set(after)) != 2 or any(not value for value in after):
            raise IRSchemaError("Heterogeneous drain requires both worker entries.")

    @property
    def local_cost(self):
        return TIRCost(synchronizations=1)


@tir_node("heterogeneous_program")
@dataclass(frozen=True)
class HeterogeneousProgram(TIRNode):
    name: str
    plan: str
    functions: tuple[WorkerFunction, ...]
    exchanges: tuple[PipelineExchange | ValueExchange, ...]
    launch: HeterogeneousLaunch
    drain: HeterogeneousDrain

    def __post_init__(self):
        _name(self.name, "program name")
        _name(self.plan, "program plan")
        _values(self, "functions", WorkerFunction)
        exchanges = tuple(self.exchanges)
        if any(not isinstance(value, (PipelineExchange, ValueExchange)) for value in exchanges):
            raise IRSchemaError("Heterogeneous exchanges require typed channels.")
        object.__setattr__(self, "exchanges", exchanges)
        if not isinstance(self.launch, HeterogeneousLaunch) or not isinstance(self.drain, HeterogeneousDrain):
            raise IRSchemaError("Heterogeneous program requires a typed launch and final drain.")


def projected_function_attrs(source, *, cpu):
    attrs = dict(source.attrs)
    if not cpu:
        for key in ("written_parameters", "transfer_source_parameters"):
            if key in attrs:
                attrs[key] = ()
    return attrs


def verify_heterogeneous_programs(module):
    from triton.flagmega.ir.tir.heterogeneous_verify import verify_heterogeneous_programs as verify
    return verify(module)


def heterogeneous_source_hash(module):
    from triton.flagmega.ir.tir.heterogeneous_verify import heterogeneous_source_hash as fingerprint
    return fingerprint(module)


def exchange_ranges(exchange, event, *, sequence=None):
    from triton.flagmega.ir.tir.heterogeneous_verify import exchange_ranges as ranges
    return ranges(exchange, event, sequence=sequence)


__all__ = [
    "ChannelBarrier", "ExecutionPlacement", "HeterogeneousDrain", "HeterogeneousLaunch",
    "HeterogeneousPlan", "HeterogeneousProgram", "KernelPhaseInvoke", "PipelineExchange", "ValueExchange",
    "WorkerCall", "WorkerFunction", "exchange_ranges", "heterogeneous_source_hash", "verify_heterogeneous_programs",
]
