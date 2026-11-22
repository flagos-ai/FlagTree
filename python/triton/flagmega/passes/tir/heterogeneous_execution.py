# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Plan external stage storage, then project typed CPU/device worker graphs."""

from dataclasses import dataclass, replace
from math import prod

from triton.flagmega.errors import IRVerificationError
from triton.flagmega.ir import (
    Buffer, ExecutionFunction, KernelInvoke, MemSpan, PhysicalBuffer, PrimFunctionCall,
    Sequential, WorkspaceLifetime, WorkspaceRequirement, tensor_type,
)
from triton.flagmega.ir.tir.barrier import Barrier
from triton.flagmega.ir.tir.heterogeneous import (
    ChannelBarrier, ExecutionPlacement, HeterogeneousDrain, HeterogeneousLaunch,
    HeterogeneousPlan, HeterogeneousProgram, KernelPhaseInvoke, PipelineExchange,
    WorkerCall, WorkerFunction, exchange_ranges, heterogeneous_source_hash,
    verify_heterogeneous_programs, projected_function_attrs,
)
from triton.flagmega.passes.functions.graph import callee_first_functions, function_nodes


@dataclass(frozen=True)
class ProjectionStorage:
    prefix_bytes: int
    output_bytes: int
    quantized_bytes: int
    command_count: int


@dataclass(frozen=True)
class KernelPhaseContract:
    implementation: str
    domain: str
    projections: tuple[ProjectionStorage, ...] = ()
    cpu_overlap_safe: bool = False
    legal_domains: tuple[str, ...] = ()
    numerical_profile: str | None = None


def _reject(module, message):
    raise IRVerificationError("Heterogeneous " + message, stage=module.stage)


def _contracts(module, target):
    resolve = getattr(target, "heterogeneous_kernel_contract", None)
    if resolve is None or not getattr(target, "heterogeneous_protocol", None):
        _reject(module, "target has no enabled phase planning contract.")
    result = {}
    for function in module.functions:
        for node in function_nodes(module, function):
            definition = module.kernel_callable_map.get(str(node.attrs.get("callee", "")))
            if definition is None:
                if node.op == "control.for":
                    _reject(module, "initial partition requires static acyclic function calls.")
                continue
            contract = resolve(module, definition)
            if not isinstance(contract, KernelPhaseContract):
                _reject(module, "selected kernel has no typed phase contract.")
            result[(function.name, node.id)] = (definition, contract)
    if not result:
        _reject(module, "planning requires selected kernel calls.")
    return result


def plan_heterogeneous_execution(module, target):
    if module.heterogeneous_programs or module.execution_functions:
        _reject(module, "PlanHeterogeneousExecution must run before bufferization.")
    contracts = _contracts(module, target)
    if target.heterogeneous_protocol in {"k230.glb-values/v1", "k230.functions/v1"}:
        from .heterogeneous_values import plan_value_execution
        return plan_value_execution(module, target, contracts)
    placements = tuple(ExecutionPlacement(function, call_id, definition.name,
        contract.domain, contract.implementation, len(contract.projections))
        for (function, call_id), (definition, contract) in contracts.items())
    resources = target.heterogeneous_resource_requirements(tuple(value[1] for value in contracts.values()))
    result = HeterogeneousPlan("heterogeneous", module.entry, target.heterogeneous_protocol, placements, resources)
    if module.heterogeneous_plans:
        if len(module.heterogeneous_plans) != 1:
            _reject(module, "initial target admits one program plan.")
        existing = module.heterogeneous_plans[0]
        if existing != result:
            _reject(module, "edited placement or resources are unsupported by the selected phase contracts.")
        return module
    return replace(module, heterogeneous_plans=(result,))


def validate_heterogeneous_resources(module, options):
    for plan in module.heterogeneous_plans:
        for resource in plan.resources:
            space = options.memory_space_map.get(resource.memory_space)
            if space is None or any(not value.is_fixed for value in resource.type.shape):
                _reject(module, "stage resources need a declared static target memory space.")
            size = prod(value.fixed_value for value in resource.type.shape) * resource.type.dtype.itemsize
            if (size <= 0 or size > space.maximum_bytes or resource.alignment < space.granularity
                    or resource.lifetime is not WorkspaceLifetime.FUNCTION):
                _reject(module, "stage resource exceeds its target allocation contract.")


def projection_exchange(function, call_id, index, storage, allocation, prefix_extent, stride):
    name = f"{function}.{call_id}.{index}"
    prefix = Buffer(name + ".prefix", "uint8", MemSpan(allocation, 0, storage.prefix_bytes),
                    (storage.prefix_bytes,), (1,))
    output = Buffer(name + ".output", "float16", MemSpan(allocation, prefix_extent, storage.output_bytes),
                    (storage.output_bytes // 2,), (1,))
    return PipelineExchange(name, function, call_id, index, (prefix,), (output,), slot_stride_bytes=stride)


def form_heterogeneous_execution(module, target):
    if module.heterogeneous_programs:
        _reject(module, "program is already formed; resume its editable output stage.")
    if len(module.heterogeneous_plans) != 1 or not module.execution_functions:
        _reject(module, "FormHeterogeneousExecution requires a planned, bufferized execution graph.")
    plan = module.heterogeneous_plans[0]
    contracts = _contracts(module, target)
    if plan.protocol == "k230.functions/v1":
        from .k230_functions import form_function_execution
        return form_function_execution(module, target, contracts)
    if plan.protocol == "k230.glb-values/v1":
        from .heterogeneous_values import form_value_execution
        return form_value_execution(module, target, contracts)
    expected = {(f, c): ExecutionPlacement(f, c, definition.name, contract.domain,
        contract.implementation, len(contract.projections)) for (f, c), (definition, contract) in contracts.items()}
    if {(value.source_function, value.call_id): value for value in plan.placements} != expected:
        _reject(module, "placement differs from the registered phase contracts.")
    resources = target.heterogeneous_resource_requirements(tuple(value[1] for value in contracts.values()))
    if plan.resources != resources or plan.protocol != target.heterogeneous_protocol:
        _reject(module, "resource plan or protocol changed after allocation planning.")
    staging = next((value for value in resources if value.name == "pipeline.mmz"), None)
    if staging is None:
        _reject(module, "target did not declare pipeline.mmz stage storage.")
    capacity = prod(value.fixed_value for value in staging.type.shape) * staging.type.dtype.itemsize
    stride = capacity // plan.slots
    prefix_extent = max(storage.prefix_bytes for _, contract in contracts.values() for storage in contract.projections)
    allocation = PhysicalBuffer(staging.name, staging.memory_space, capacity, staging.alignment)
    from triton.flagmega.ir.bufferization import BufferPlan
    from triton.flagmega.ir.tir.heterogeneous_verify import source_call_conflicts
    buffer_plan = module.metadata.get("buffer_plan")
    if buffer_plan is not None and not isinstance(buffer_plan, BufferPlan):
        buffer_plan = BufferPlan.from_data(buffer_plan)
    workers, exchanges, counts = [], [], {}
    for source_function in callee_first_functions(module):
        source = module.execution_function_map[source_function.name]
        bodies = {plan.cpu_domain: [], plan.device_domain: []}
        sequence = 0
        cpu = bodies[plan.cpu_domain]
        pending = []
        def flush():
            for _, finish in pending:
                cpu.extend(finish)
            pending.clear()
        for call in source.body.fields:
            if isinstance(call, PrimFunctionCall):
                flush()
                for domain, body in bodies.items():
                    body.append(WorkerCall(replace(call, callee=f"{call.callee}.{domain}"), sequence))
                sequence += counts[call.callee]
            elif isinstance(call, Barrier):
                flush()
                bodies[plan.cpu_domain].append(call)
            elif isinstance(call, KernelInvoke):
                contract = contracts[(source.name, call.call_id)][1]
                if any(prior.call_id in call.dependencies or source_call_conflicts(
                        module, source.name, prior, call, buffer_plan=buffer_plan) for prior, _ in pending):
                    flush()
                if not contract.projections:
                    if not contract.cpu_overlap_safe:
                        flush()
                    bodies[plan.cpu_domain].append(call)
                    continue
                for index, storage in enumerate(contract.projections):
                    sequence += 1
                    name = f"{source.name}.{call.call_id}.{index}"
                    exchange = projection_exchange(source.name, call.call_id, index, storage, allocation, prefix_extent, stride)
                    exchanges.append(exchange)
                    def phase(kind):
                        return KernelPhaseInvoke(name + "." + kind, call.call_id, kind, index, name, call)
                    def event(kind):
                        return ChannelBarrier(name + "." + kind, name, kind, sequence, (), exchange_ranges(exchange, kind))
                    if len(pending) == plan.slots:
                        cpu.extend(pending.pop(0)[1])
                    cpu.extend((phase("prepare"), event("publish")))
                    pending.append((call, (event("observe"), phase("finish"), event("reclaim"))))
                    bodies[plan.device_domain].extend((event("acquire"), phase("execute"), event("complete")))
            else:
                _reject(module, f"unsupported source schedule statement {type(call).__name__}.")
        flush()
        counts[source.name] = sequence
        if sequence > target.heterogeneous_sequence_max:
            _reject(module, "phase sequence exceeds the target protocol limit.")
        for domain, body in bodies.items():
            function = ExecutionFunction(f"{source.name}.{domain}", source.parameters, source.results, Sequential(tuple(body)),
                                         projected_function_attrs(source, cpu=domain == plan.cpu_domain))
            workers.append(WorkerFunction(source.name, domain, function, sequence))
    entries = (f"{module.entry}.{plan.cpu_domain}", f"{module.entry}.{plan.device_domain}")
    program = HeterogeneousProgram("heterogeneous", plan.name, tuple(workers), tuple(exchanges),
                                  HeterogeneousLaunch(*entries), HeterogeneousDrain(entries))
    result = replace(module, heterogeneous_plans=(replace(plan, source_hash=heterogeneous_source_hash(module)),),
                     heterogeneous_programs=(program,))
    verify_heterogeneous_programs(result)
    target.verify_heterogeneous_program(result)
    return result


@dataclass(frozen=True)
class PlanHeterogeneousExecutionPass:
    target: object
    name: str = "PlanHeterogeneousExecution"
    preserves: frozenset[str] = frozenset()

    def run(self, module):
        return plan_heterogeneous_execution(module, self.target)


@dataclass(frozen=True)
class FormHeterogeneousExecutionPass:
    target: object
    name: str = "FormHeterogeneousExecution"
    preserves: frozenset[str] = frozenset()

    def run(self, module):
        return form_heterogeneous_execution(module, self.target)
