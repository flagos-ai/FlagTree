# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""CP-SAT candidate graph and extractor for AutoDistributed."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import cached_property
from itertools import product
from typing import Mapping

from triton.flagmega.diagnostics import DumpFlags, DumpScope
from triton.flagmega.errors import IRVerificationError
from triton.flagmega.ir import (
    BlockCyclicSplit,
    DistributedType,
    IRModule,
    IRType,
    Placement,
    ParameterKind,
    SBPSplit,
    TupleType,
)
from triton.flagmega.ir import get_definition
from triton.flagmega.passes.functions import (
    function_nodes,
    static_function_invocation_counts,
    static_node_invocation_counts,
)
from triton.flagmega.ir.distributed_inference import broadcast_ir_type
from triton.flagmega.passes.auto_distributed.candidates import (
    DistributedCandidate,
    DistributedCandidateContext,
    DistributedCandidateProvider,
    DistributedCandidateProviderRegistry,
)
from triton.flagmega.passes.auto_distributed.candidate_identity import (
    distributed_candidate_id,
)
from triton.flagmega.passes.auto_distributed.reshard import (
    DistributedReshardPlan,
    DistributedReshardPlanner,
)
from triton.flagmega.passes.auto_distributed.reshard_cost import (
    DistributedReshardCostModel,
)
from triton.flagmega.passes.auto_distributed.operation_cost import (
    DistributedOperationCostModel,
)
from triton.flagmega.passes.auto_distributed.realization import (
    DistributedReshardRealization,
    DistributedReshardRealizationContext,
    DistributedReshardRealizationPolicy,
    DistributedReshardSourceKind,
    DistributedReshardUsageKind,
)
from triton.flagmega.passes.constants import ConstnessAnalysis


@dataclass(frozen=True)
class CandidateBucket:
    node_id: str
    candidates: tuple[DistributedCandidate, ...]
    executable: bool


@dataclass(frozen=True)
class ReshardSite:
    producer_id: str
    producer_index: int
    consumer_id: str
    consumer_index: int | None
    input_index: int
    target_type: IRType
    plans: tuple[DistributedReshardPlan, ...]
    usage: str
    invocation_count: int

    @cached_property
    def key(self) -> tuple[str, str, int]:
        return (self.producer_id, self.consumer_id, self.input_index)

    @cached_property
    def id(self) -> str:
        consumer_index = "output" if self.consumer_index is None else str(self.consumer_index)
        return (
            f"{self.producer_id}_{self.producer_index}__{self.consumer_id}_"
            f"{consumer_index}_{self.input_index}"
        )


@dataclass(frozen=True)
class SearchGraph:
    module: IRModule
    placement: Placement
    buckets: tuple[CandidateBucket, ...]
    reshard_sites: tuple[ReshardSite, ...]
    realization_policy: DistributedReshardRealizationPolicy
    invocation_counts: Mapping[str, int]
    constant_ids: frozenset[str] = frozenset()
    reshard_cost_model: DistributedReshardCostModel = DistributedReshardCostModel()
    operation_cost_model: DistributedOperationCostModel = DistributedOperationCostModel()
    # A graph is one immutable policy/IR snapshot. Replacing the graph resets
    # analysis caches; no type/cost decisions leak into another agent trial.
    _realized_costs: dict = field(default_factory=dict, init=False, compare=False, repr=False)

    @cached_property
    def bucket_map(self) -> dict[str, CandidateBucket]:
        return {bucket.node_id: bucket for bucket in self.buckets}


@dataclass(frozen=True)
class SearchResult:
    graph: SearchGraph
    selected: Mapping[str, DistributedCandidate]
    selected_reshards: Mapping[tuple[str, str, int], DistributedReshardPlan]
    objective: int
    status: str


def build_search_graph(
    module: IRModule,
    placement: Placement,
    registry: DistributedCandidateProviderRegistry,
    realization_policy: DistributedReshardRealizationPolicy,
    reshard_cost_model: DistributedReshardCostModel | None = None,
    operation_cost_model: DistributedOperationCostModel | None = None,
) -> SearchGraph:
    buckets: list[CandidateBucket] = []
    reshard_memo = {}
    type_inference_memo = {}

    def reshard_plans(source, target, policy, source_kind, usage):
        key = (source, target, source_kind, usage)
        if key not in reshard_memo:
            reshard_memo[key] = _reshard_plans(source, target, policy, source_kind, usage)
        return reshard_memo[key]
    function_invocations = static_function_invocation_counts(module)
    node_invocations = static_node_invocation_counts(module)
    available: dict[str, tuple[DistributedCandidate, ...]] = {}
    constant_ids = ConstnessAnalysis.analyze(module).constants
    function_parameter_kinds = _infer_function_parameter_kinds(module)
    for node in module.nodes:
        input_candidates = tuple(
            tuple(candidate.return_type for candidate in available[input_id])
            for input_id in node.inputs
        )
        if node.op == "builtin.tuple":
            values: list[DistributedCandidate] = []
            for input_types in product(*input_candidates):
                return_type = TupleType(tuple(input_types))
                if any(
                    candidate.return_type == return_type
                    and candidate.input_types == tuple(input_types)
                    for candidate in values
                ):
                    continue
                values.append(DistributedCandidate(
                    distributed_candidate_id(
                        node.id,
                        "tuple-structural-construction",
                        return_type,
                        tuple(input_types),
                    ),
                    return_type,
                    tuple(input_types),
                    0,
                    "tuple-structural-construction",
                    objective_kind="analytic",
                    objective_model="flagmega.structural-zero/v1",
                    objective_evidence=("structural-tuple",),
                ))
            candidates = tuple(values)
            executable = False
        elif node.op == "builtin.get_item":
            index = int(node.attrs["index"])
            values: list[DistributedCandidate] = []
            for source in available[node.inputs[0]]:
                if not isinstance(source.return_type, TupleType):
                    continue
                return_type = source.return_type.fields[index]
                if not any(
                    candidate.return_type == return_type and candidate.input_types == (source.return_type,)
                    for candidate in values
                ):
                    values.append(DistributedCandidate(
                        distributed_candidate_id(
                            node.id,
                            f"tuple-field-{index}",
                            return_type,
                            (source.return_type,),
                        ),
                        return_type,
                        (source.return_type,),
                        0,
                        f"tuple-field-{index}",
                    ))
            candidates = tuple(values)
            executable = False
        else:
            provider = registry.try_get(node.op)
            if provider is None:
                is_originator = not node.inputs
                is_structural_call = node.op == "builtin.call"
                if not is_originator and not is_structural_call:
                    raise IRVerificationError(
                        f"AutoDistributed has no reviewed candidate provider for "
                        f"compute op {node.op!r} at {node.id!r}.",
                        stage=module.stage,
                        node_id=node.id,
                    )
                return_type = node.type if is_originator else broadcast_ir_type(
                    node.type, placement
                )
                if is_structural_call:
                    callee = str(node.attrs["callee"])
                    parameter_kinds = function_parameter_kinds[callee]
                    input_types = tuple(
                        module.node_map[value].type
                        if parameter_kind == ParameterKind.ATTRIBUTE
                        else broadcast_ir_type(module.node_map[value].type, placement)
                        for value, parameter_kind in zip(
                            node.inputs, parameter_kinds
                        )
                    )
                else:
                    input_types = ()
                candidates = (DistributedCandidate(
                    f"distribution.{node.id}.{'originator' if is_originator else 'broadcast'}",
                    return_type,
                    input_types,
                    0,
                    "logical-originator" if is_originator else "function-call-abi-propagation",
                    objective_model="flagmega.structural-distribution/v1",
                ),)
                executable = False
            else:
                context = DistributedCandidateContext(
                    module,
                    node,
                    placement,
                    input_candidates,
                    registry.split_candidate_provider,
                    reshard_cost_model or DistributedReshardCostModel(),
                    operation_cost_model or DistributedOperationCostModel(),
                    type_inference_memo=type_inference_memo,
                )
                candidates = _provider_candidates(provider, context, node.type)
                if not candidates and not provider.is_exhaustive:
                    candidates = (DistributedCandidate(
                        f"distribution.{node.id}.logical",
                        node.type,
                        tuple(module.node_map[value].type for value in node.inputs),
                        0,
                        "non-exhaustive-provider-fallback",
                        node.op,
                    ),)
                executable = True
        if not candidates:
            raise IRVerificationError(
                f"AutoDistributed produced no candidates for {node.id!r}.", stage=module.stage, node_id=node.id)
        available[node.id] = candidates
        buckets.append(CandidateBucket(node.id, candidates, executable))
    # Candidate providers describe the exact input types they can consume.
    # Propagate a demanded TupleType through an explicit structural tuple so
    # each field can acquire its own Boxing/ShardedView edge.  This is the
    # compact equivalent of nncase's provider-input adaptation buckets and is
    # essential for heterogeneous tuples such as Q/K/V, whose head axes use
    # independent mesh dimensions.  The closure is demand driven: it does not
    # enumerate arbitrary tuple layouts that no operation requested.
    buckets = list(_complete_structural_tuple_demands(module, tuple(buckets)))
    bucket_map = {bucket.node_id: bucket for bucket in buckets}
    sites: list[ReshardSite] = []
    for consumer in module.nodes:
        consumer_bucket = bucket_map[consumer.id]
        for consumer_index, candidate in enumerate(consumer_bucket.candidates):
            for input_index, producer_id in enumerate(consumer.inputs):
                target_type = candidate.input_types[input_index]
                for producer_index, producer in enumerate(bucket_map[producer_id].candidates):
                    # Native structural candidates encode one exact Cartesian
                    # product of their producer choices and must not be
                    # weakened by an unrelated adapter edge.  Only the
                    # demand-closure candidates above intentionally admit
                    # field-wise adaptation.
                    if (
                        consumer.op == "builtin.tuple"
                        and "provider-demanded-tuple-layout"
                        not in candidate.objective_evidence
                    ):
                        continue
                    plans = reshard_plans(
                        producer.return_type,
                        target_type,
                        realization_policy,
                        source_kind_for_node(module.node_map[producer_id]),
                        DistributedReshardUsageKind.INTERNAL,
                    )
                    if plans:
                        sites.append(ReshardSite(
                            producer_id,
                            producer_index,
                            consumer.id,
                            consumer_index,
                            input_index,
                            target_type,
                            plans,
                            DistributedReshardUsageKind.INTERNAL.value,
                            node_invocations.get(consumer.id, 1),
                        ))
    for function in module.functions:
        consumer_id = function_boundary_id(function.name)
        for output_index, output_id in enumerate(function.outputs):
            target_type = function_output_type(module, function.name, output_id, placement)
            for producer_index, producer in enumerate(bucket_map[output_id].candidates):
                usage = (
                    DistributedReshardUsageKind.PROGRAM_OUTPUT
                    if function.name == module.entry
                    else DistributedReshardUsageKind.FUNCTION_BOUNDARY
                )
                plans = reshard_plans(
                    producer.return_type,
                    target_type,
                    realization_policy,
                    source_kind_for_node(module.node_map[output_id]),
                    usage,
                )
                if plans:
                    sites.append(ReshardSite(
                        output_id,
                        producer_index,
                        consumer_id,
                        None,
                        output_index,
                        target_type,
                        plans,
                        usage.value,
                        function_invocations.get(function.name, 1),
                    ))
    return SearchGraph(
        module,
        placement,
        tuple(buckets),
        tuple(sites),
        realization_policy,
        node_invocations,
        constant_ids,
        reshard_cost_model or DistributedReshardCostModel(),
        operation_cost_model or DistributedOperationCostModel(),
    )


def _complete_structural_tuple_demands(
    module: IRModule,
    buckets: tuple[CandidateBucket, ...],
) -> tuple[CandidateBucket, ...]:
    """Thread provider tuple demands to their explicit field producers.

    A tuple itself has no data movement.  Its candidate records one exact
    combination of field types; any physical adaptation remains an ordinary
    per-field reshard site and therefore stays visible to CP-SAT, dumps, and
    the materializer.  Requiring a real downstream demand avoids the former
    alternative of inventing every Cartesian product of leaf distributions.
    """

    by_id = {bucket.node_id: bucket for bucket in buckets}
    changed = True
    while changed:
        changed = False
        demands: dict[str, list[TupleType]] = {}
        for consumer in module.nodes:
            consumer_bucket = by_id[consumer.id]
            for candidate in consumer_bucket.candidates:
                for producer_id, target_type in zip(
                    consumer.inputs, candidate.input_types, strict=True
                ):
                    producer = module.node_map[producer_id]
                    if producer.op != "builtin.tuple" or not isinstance(
                        target_type, TupleType
                    ):
                        continue
                    if len(producer.inputs) != len(target_type.fields):
                        continue
                    values = demands.setdefault(producer_id, [])
                    if target_type not in values:
                        values.append(target_type)

        for producer_id, target_types in demands.items():
            bucket = by_id[producer_id]
            values = list(bucket.candidates)
            for target_type in target_types:
                input_types = tuple(target_type.fields)
                if any(
                    candidate.return_type == target_type
                    and candidate.input_types == input_types
                    for candidate in values
                ):
                    continue
                values.append(DistributedCandidate(
                    f"distribution.{producer_id}.tuple.demand_{len(values)}",
                    target_type,
                    input_types,
                    0,
                    "tuple-structural-provider-demand",
                    objective_kind="analytic",
                    objective_model="flagmega.structural-zero/v1",
                    objective_evidence=(
                        "provider-demanded-tuple-layout",
                        "field-wise-reshard-preserves-producer-contract",
                    ),
                ))
                changed = True
            if len(values) != len(bucket.candidates):
                by_id[producer_id] = CandidateBucket(
                    bucket.node_id,
                    tuple(values),
                    bucket.executable,
                )
    return tuple(by_id[bucket.node_id] for bucket in buckets)


def _infer_function_parameter_kinds(
    module: IRModule,
) -> dict[str, tuple[ParameterKind, ...]]:
    """Infer nncase ParameterKind at reusable-function boundaries.

    A graph function has no separate signature object for semantic operand
    kinds.  Derive it from every direct use of each formal.  Attribute-only
    formals (dimensions, state handles, launch controls) stay ordinary values;
    any data/structural use makes the formal an Input.  Recursive calls are
    handled callee-first and cycles conservatively become Inputs.
    """

    users: dict[str, list[tuple[object, int]]] = {}
    for function in module.functions:
        for node in function_nodes(module, function):
            for index, input_id in enumerate(node.inputs):
                users.setdefault(input_id, []).append((node, index))

    memo: dict[str, tuple[ParameterKind, ...]] = {}
    active: set[str] = set()

    def classify_function(name: str) -> tuple[ParameterKind, ...]:
        if name in memo:
            return memo[name]
        function = module.function_map[name]
        if name in active:
            return tuple(ParameterKind.INPUT for _ in function.parameters)
        active.add(name)
        result: list[ParameterKind] = []
        for parameter_id in function.parameters:
            kinds: list[ParameterKind] = []
            for user, input_index in users.get(parameter_id, ()):
                if user.op == "builtin.call":
                    callee_name = str(user.attrs["callee"])
                    callee_kinds = classify_function(callee_name)
                    kinds.append(callee_kinds[input_index])
                    continue
                try:
                    definition = get_definition(user.op)
                except KeyError:
                    kinds.append(ParameterKind.INPUT)
                    continue
                if input_index >= len(definition.input_parameters):
                    kinds.append(ParameterKind.INPUT)
                    continue
                kinds.append(
                    definition.input_parameters[input_index].parameter_kind
                )
            result.append(
                ParameterKind.ATTRIBUTE
                if kinds and all(kind == ParameterKind.ATTRIBUTE for kind in kinds)
                else ParameterKind.INPUT
            )
        active.remove(name)
        memo[name] = tuple(result)
        return memo[name]

    for function in module.functions:
        classify_function(function.name)
    return memo


def solve_search_graph(
    graph: SearchGraph,
    *,
    fixed_selections: Mapping[str, str] | None = None,
    dump_subdirectory: str | None = None,
) -> SearchResult:
    cp_model = _load_cp_model()
    model = cp_model.CpModel()
    bucket_map = graph.bucket_map
    site_map = {
        (site.producer_id, site.producer_index, site.consumer_id, site.consumer_index, site.input_index): site
        for site in graph.reshard_sites
    }
    from .search_domains import propagate_domains
    domains = propagate_domains(graph, dict(fixed_selections or {}), site_map)
    variables = {
        (bucket.node_id, index): model.NewBoolVar(f"{bucket.node_id}__{index}")
        for bucket in graph.buckets
        for index in domains[bucket.node_id]
    }
    for bucket in graph.buckets:
        model.AddExactlyOne([variables[(bucket.node_id, index)] for index in domains[bucket.node_id]])

    objective_terms = []
    objective_weights = []
    simplicity_weights = []
    for bucket in graph.buckets:
        for index, candidate in enumerate(bucket.candidates):
            if index not in domains[bucket.node_id]:
                continue
            objective_terms.append(variables[(bucket.node_id, index)])
            # Candidate construction rejects invalid ranges.  Do not clamp:
            # silently turning a broken negative/overflow estimate into a
            # legal objective would change compiler decisions.
            objective_weights.append(
                candidate.operation_cost
                * graph.invocation_counts.get(bucket.node_id, 1)
            )
            simplicity_weights.append(_candidate_distribution_complexity(candidate))

    for consumer in graph.module.nodes:
        consumer_bucket = bucket_map[consumer.id]
        for consumer_index, candidate in enumerate(consumer_bucket.candidates):
            if consumer_index not in domains[consumer.id]:
                continue
            consumer_var = variables[(consumer.id, consumer_index)]
            for input_index, producer_id in enumerate(consumer.inputs):
                required = candidate.input_types[input_index]
                producer_bucket = bucket_map[producer_id]
                compatible = []
                for producer_index, producer in enumerate(producer_bucket.candidates):
                    if producer_index not in domains[producer_id]:
                        continue
                    if producer.return_type == required or (
                        producer_id,
                        producer_index,
                        consumer.id,
                        consumer_index,
                        input_index,
                    ) in site_map:
                        compatible.append(producer_index)
                if not compatible:
                    model.Add(consumer_var == 0)
                    continue
                model.Add(
                    consumer_var
                    <= sum(variables[(producer_id, producer_index)] for producer_index in compatible)
                )

    plan_variables = {}
    for site in graph.reshard_sites:
        if site.producer_index not in domains[site.producer_id] or (
            site.consumer_index is not None and site.consumer_index not in domains[site.consumer_id]
        ):
            continue
        producer_var = variables[(site.producer_id, site.producer_index)]
        if site.consumer_index is None:
            active = producer_var
        else:
            consumer_var = variables[(site.consumer_id, site.consumer_index)]
            if len(domains[site.producer_id]) == 1:
                active = consumer_var
            elif len(domains[site.consumer_id]) == 1:
                active = producer_var
            else:
                active = model.NewBoolVar(f"reshard_active__{site.id}")
                model.Add(active <= producer_var)
                model.Add(active <= consumer_var)
                model.Add(active >= producer_var + consumer_var - 1)
        choices = []
        source_type = bucket_map[site.producer_id].candidates[site.producer_index].return_type
        for plan_index, plan in enumerate(site.plans):
            variable = (active if len(site.plans) == 1
                        else model.NewBoolVar(f"reshard_plan__{site.id}_{plan_index}"))
            plan_variables[(site.id, plan_index)] = variable
            choices.append(variable)
            objective_terms.append(variable)
            objective_weights.append(
                _realized_reshard_plan_cost(graph, site, source_type, plan)
                * site.invocation_count
            )
            simplicity_weights.append(0)
        if len(site.plans) > 1:
            model.Add(sum(choices) == active)

    for function in graph.module.functions:
        consumer_id = function_boundary_id(function.name)
        for output_index, output_id in enumerate(function.outputs):
            target = function_output_type(
                graph.module, function.name, output_id, graph.placement)
            bucket = bucket_map[output_id]
            for producer_index, candidate in enumerate(bucket.candidates):
                if producer_index not in domains[output_id]:
                    continue
                if candidate.return_type == target:
                    continue
                if (output_id, producer_index, consumer_id, None, output_index) not in site_map:
                    model.Add(variables[(output_id, producer_index)] == 0)

    # Keep the analytic/agent supplied objective primary.  When it cannot
    # distinguish two complete plans, prefer the simpler contiguous storage
    # contract deterministically.  Block-cyclic candidates stay in the graph
    # and remain selectable by editing the checkpoint or supplying stronger
    # objective evidence; this is a default tie-break, not target legality.
    maximum_simplicity = sum(
        max(
            _candidate_distribution_complexity(candidate)
            for candidate in bucket.candidates
        )
        for bucket in graph.buckets
    )
    primary_scale = maximum_simplicity + 1
    objective_coefficients: list[int] = []
    maximum_objective = 0
    for index, (primary, simplicity) in enumerate(zip(
        objective_weights, simplicity_weights
    )):
        coefficient = primary_scale * primary + simplicity
        _require_cp_sat_int64(
            coefficient, f"objective coefficient {index}"
        )
        maximum_objective += coefficient
        _require_cp_sat_int64(
            maximum_objective, "maximum aggregate objective"
        )
        objective_coefficients.append(coefficient)
    model.Minimize(sum(
        coefficient * variable
        for coefficient, variable in zip(
            objective_coefficients, objective_terms
        )
    ))
    validation = model.Validate()
    if validation:
        raise IRVerificationError(f"AutoDistributed CP-SAT model is invalid: {validation}", stage=graph.module.stage)
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = _positive_env_float("FLAGMEGA_AD_SOLVE_MAX_TIME", 120.0)
    solver.parameters.num_search_workers = _positive_env_int(
        "FLAGMEGA_AD_SOLVE_WORKERS", max((os.cpu_count() or 2) // 2, 1))
    status_value = solver.Solve(model)
    status = solver.StatusName(status_value)
    if status_value not in {cp_model.OPTIMAL, cp_model.FEASIBLE}:
        raise IRVerificationError(f"AutoDistributed CP-SAT extraction failed: {status}", stage=graph.module.stage)
    selected = {
        bucket.node_id: next(
            candidate
            for index, candidate in enumerate(bucket.candidates)
            if index in domains[bucket.node_id] and solver.BooleanValue(variables[(bucket.node_id, index)])
        )
        for bucket in graph.buckets
    }
    selected_indexes = {
        bucket.node_id: next(
            index for index, _ in enumerate(bucket.candidates)
            if index in domains[bucket.node_id] and solver.BooleanValue(variables[(bucket.node_id, index)])
        )
        for bucket in graph.buckets
    }
    selected_reshards = {}
    selected_reshard_sites: dict[
        tuple[str, str, int], ReshardSite
    ] = {}
    for site in graph.reshard_sites:
        if selected_indexes[site.producer_id] != site.producer_index:
            continue
        if site.consumer_index is not None and selected_indexes[site.consumer_id] != site.consumer_index:
            continue
        for plan_index, plan in enumerate(site.plans):
            if solver.BooleanValue(plan_variables[(site.id, plan_index)]):
                selected_reshards[site.key] = plan
                selected_reshard_sites[site.key] = site
                break
    primary_objective = sum(
        candidate.operation_cost * graph.invocation_counts.get(node_id, 1)
        for node_id, candidate in selected.items()
    )
    primary_objective += sum(
        _realized_reshard_plan_cost(
            graph,
            selected_reshard_sites[
                (producer_id, consumer_id, input_index)
            ],
            selected[producer_id].return_type,
            plan,
        )
        * selected_reshard_sites[
            (producer_id, consumer_id, input_index)
        ].invocation_count
        for (producer_id, consumer_id, input_index), plan
        in selected_reshards.items()
    )
    result = SearchResult(
        graph,
        selected,
        selected_reshards,
        primary_objective,
        status,
    )
    if dump_subdirectory is None:
        _dump_search(result)
    else:
        with DumpScope(dump_subdirectory):
            _dump_search(result)
    return result


def _require_cp_sat_int64(value: int, description: str) -> None:
    """Reject an objective that OR-Tools cannot represent exactly."""

    maximum = (1 << 63) - 1
    if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= maximum:
        raise IRVerificationError(
            f"AutoDistributed {description} exceeds the signed 64-bit CP-SAT "
            f"range: {value}.",
        )


def _candidate_distribution_complexity(candidate: DistributedCandidate) -> int:
    """Count non-contiguous split stages in one candidate's storage ABI."""

    return sum(
        _type_distribution_complexity(value)
        for value in (candidate.return_type, *candidate.input_types)
    )


def _type_distribution_complexity(value: IRType) -> int:
    if isinstance(value, TupleType):
        return sum(_type_distribution_complexity(field) for field in value.fields)
    if not isinstance(value, DistributedType):
        return 0
    return sum(
        1
        for policy in value.axis_policies
        if isinstance(policy, SBPSplit)
        for stage in policy.stages
        if isinstance(stage.distribution, BlockCyclicSplit)
    )


def graph_dot(
    graph: SearchGraph,
    selected: Mapping[str, DistributedCandidate] | None = None,
    selected_reshards: Mapping[tuple[str, str, int], DistributedReshardPlan] | None = None,
) -> str:
    lines = ["digraph AutoDistributed {", "  rankdir=LR;"]
    for bucket in graph.buckets:
        lines.append(f'  subgraph "cluster_{_dot(bucket.node_id)}" {{')
        lines.append(f'    label="{_dot(bucket.node_id)}";')
        for index, candidate in enumerate(bucket.candidates):
            picked = selected is not None and selected.get(bucket.node_id) == candidate
            if selected is not None and not picked:
                continue
            color = "green" if picked else "black"
            target = "" if candidate.target_op is None else f"\\ntarget={candidate.target_op}"
            label = (
                f"{candidate.id}\\n{_type_text(candidate.return_type)}"
                f"{target}\\ncost={candidate.operation_cost}"
                f" x{graph.invocation_counts.get(bucket.node_id, 1)}"
                f"\\nmodel={candidate.objective_kind}:{candidate.objective_model}"
            )
            lines.append(f'    "{_dot(bucket.node_id)}_{index}" [label="{_dot(label)}", color={color}];')
        lines.append("  }")
    bucket_map = graph.bucket_map
    for node in graph.module.nodes:
        for input_index, producer_id in enumerate(node.inputs):
            for producer_index, producer in enumerate(bucket_map[producer_id].candidates):
                if selected is not None and selected.get(producer_id) != producer:
                    continue
                for consumer_index, candidate in enumerate(bucket_map[node.id].candidates):
                    if selected is not None and selected.get(node.id) != candidate:
                        continue
                    if (
                        bucket_map[producer_id].candidates[producer_index].return_type
                        == candidate.input_types[input_index]
                    ):
                        lines.append(
                            f'  "{_dot(producer_id)}_{producer_index}" -> '
                            f'"{_dot(node.id)}_{consumer_index}" [label="arg{input_index}"];')
    output_nodes: set[str] = set()
    for site in graph.reshard_sites:
        if selected is not None and (
            selected.get(site.producer_id) != bucket_map[site.producer_id].candidates[site.producer_index]
            or (site.consumer_index is not None
                and selected.get(site.consumer_id) != bucket_map[site.consumer_id].candidates[site.consumer_index])
        ):
            continue
        if site.consumer_index is None and site.consumer_id not in output_nodes:
            output_nodes.add(site.consumer_id)
            lines.append(f'  "{_dot(site.consumer_id)}" [shape=box, label="{_dot(site.consumer_id)}"];')
        for plan_index, plan in enumerate(site.plans):
            picked = (
                selected_reshards is not None
                and selected_reshards.get(site.key) == plan
                and selected is not None
                and selected.get(site.producer_id) == bucket_map[site.producer_id].candidates[site.producer_index]
                and (
                    site.consumer_index is None
                    or selected.get(site.consumer_id) == bucket_map[site.consumer_id].candidates[site.consumer_index]
                )
            )
            if selected is not None and not picked:
                continue
            color = "green" if picked else "gray"
            plan_node = f"reshard_{site.id}_{plan_index}"
            source_type = bucket_map[site.producer_id].candidates[site.producer_index].return_type
            label = (
                f"reshard {site.usage}\\n{_plan_text(plan)}\\n"
                f"cost={_realized_reshard_plan_cost(graph, site, source_type, plan)}"
                f" x{site.invocation_count}"
            )
            lines.append(f'  "{_dot(plan_node)}" [shape=diamond, label="{_dot(label)}", color={color}];')
            lines.append(
                f'  "{_dot(site.producer_id)}_{site.producer_index}" -> "{_dot(plan_node)}";')
            consumer_node = (
                site.consumer_id
                if site.consumer_index is None
                else f"{site.consumer_id}_{site.consumer_index}"
            )
            lines.append(
                f'  "{_dot(plan_node)}" -> "{_dot(consumer_node)}" [label="arg{site.input_index}"];')
    lines.append("}")
    return "\n".join(lines) + "\n"


def _dump_search(result: SearchResult) -> None:
    dumper = DumpScope.current()
    enabled = dumper.is_enabled(DumpFlags.COMPILE) or dumper.is_enabled(DumpFlags.EGRAPH_COST)
    if not enabled or dumper.directory is None:
        return
    category = (
        DumpFlags.EGRAPH_COST
        if dumper.is_enabled(DumpFlags.EGRAPH_COST)
        else DumpFlags.COMPILE
    )
    source_hash = result.graph.module.semantic_hash
    with dumper.open_artifact(
        "DistributedSearchGraph.dot", category=category,
        kind="distributed-search-dot", producer="AutoDistributed",
        source_semantic_hash=source_hash, encoding="utf-8",
    ) as stream:
        stream.write(graph_dot(result.graph))
    with dumper.open_artifact(
        "Costs/Solve.txt", category=category,
        kind="solver-log", producer="AutoDistributed",
        source_semantic_hash=source_hash, encoding="utf-8",
    ) as stream:
        stream.write(f"Status : {result.status}\n")
        stream.write(f"Objective : {result.objective}\n")
        stream.write(f"Buckets : {len(result.graph.buckets)}\n")
        stream.write(f"Candidates : {sum(len(bucket.candidates) for bucket in result.graph.buckets)}\n")
        stream.write(f"Reshard sites : {len(result.graph.reshard_sites)}\n")
        stream.write(f"Reshard programs : {sum(len(site.plans) for site in result.graph.reshard_sites)}\n")
    with dumper.open_artifact(
        "Costs/Pick.dot", category=category,
        kind="distributed-pick-dot", producer="AutoDistributed",
        source_semantic_hash=source_hash, encoding="utf-8",
    ) as stream:
        stream.write(graph_dot(result.graph, result.selected, result.selected_reshards))
    with dumper.open_artifact(
        "Costs/Pick.txt", category=category,
        kind="selection-report", producer="AutoDistributed",
        source_semantic_hash=source_hash, encoding="utf-8",
    ) as stream:
        for bucket in result.graph.buckets:
            picked = result.selected[bucket.node_id]
            stream.write(
                f"{bucket.node_id}: {picked.id} type={_type_text(picked.return_type)} "
                f"target={picked.target_op or result.graph.module.node_map[bucket.node_id].op} "
                f"cost={picked.operation_cost} objective={picked.objective_kind}:"
                f"{picked.objective_model} reason={picked.reason} "
                f"evidence={picked.objective_evidence} "
                f"invocations={result.graph.invocation_counts.get(bucket.node_id, 1)}\n")
        stream.write("Reshards:\n")
        invocations_by_site = {site.key: site.invocation_count for site in result.graph.reshard_sites}
        for (producer_id, consumer_id, input_index), plan in sorted(result.selected_reshards.items()):
            invocation_count = invocations_by_site[(producer_id, consumer_id, input_index)]
            stream.write(
                f"  {producer_id} -> {consumer_id}[{input_index}]: "
                f"{_plan_text(plan)} invocations={invocation_count}\n")


def _reshard_plans(
    source: IRType,
    target: IRType,
    policy: DistributedReshardRealizationPolicy,
    source_kind: DistributedReshardSourceKind,
    usage_kind: DistributedReshardUsageKind,
) -> tuple[DistributedReshardPlan, ...]:
    if source == target:
        return ()
    if isinstance(source, TupleType) or isinstance(target, TupleType):
        return ()
    def can_realize(edge_source: IRType, edge_target: IRType) -> bool:
        edge_source_kind = source_kind if edge_source == source else DistributedReshardSourceKind.INTERNAL
        edge_usage_kind = usage_kind if edge_target == target else DistributedReshardUsageKind.INTERNAL
        return policy.classify(DistributedReshardRealizationContext(
            edge_source,
            edge_target,
            edge_source_kind,
            edge_usage_kind,
        )) != DistributedReshardRealization.UNSUPPORTED

    return DistributedReshardPlanner.plan(source, target, can_realize)


def _realized_reshard_plan_cost(
    graph: SearchGraph,
    site: ReshardSite,
    source_type: IRType,
    plan: DistributedReshardPlan,
) -> int:
    """Cost the physical realization selected for every reshard edge.

    A semantic type transition is not necessarily a byte transfer. A local
    ``ShardedView`` is free, while a widening internal view pays the target's
    grid-publication cost; ``Boxing`` pays the regular transfer estimate.
    """

    total = 0
    previous = source_type
    source_kind = (
        DistributedReshardSourceKind.CONSTANT
        if site.producer_id in graph.constant_ids
        else source_kind_for_node(graph.module.node_map[site.producer_id])
    )
    usage = DistributedReshardUsageKind(site.usage)
    key = (source_type, plan, source_kind, usage)
    if key in graph._realized_costs:
        return graph._realized_costs[key]
    for index, step in enumerate(plan.step_types):
        realization = graph.realization_policy.classify(
            DistributedReshardRealizationContext(
                previous,
                step,
                source_kind if index == 0 else DistributedReshardSourceKind.INTERNAL,
                usage if index + 1 == len(plan.step_types)
                else DistributedReshardUsageKind.INTERNAL,
            )
        )
        total = min(
            total + graph.reshard_cost_model.realization_cost(
                DistributedReshardRealizationContext(
                    previous,
                    step,
                    source_kind
                    if index == 0
                    else DistributedReshardSourceKind.INTERNAL,
                    usage
                    if index + 1 == len(plan.step_types)
                    else DistributedReshardUsageKind.INTERNAL,
                ),
                realization,
            ),
            2_000_000_000,
        )
        previous = step
    graph._realized_costs[key] = total
    return total


def _provider_candidates(
    provider: DistributedCandidateProvider,
    context: DistributedCandidateContext,
    default_type: IRType,
) -> tuple[DistributedCandidate, ...]:
    values: list[DistributedCandidate] = []
    for return_type in provider.get_return_candidate_types(context, (default_type,)):
        for input_tuple in provider.try_get_input_type_tuples(context, return_type) or ():
            if not provider.allows_partial_inputs and any(
                _contains_partial(value) for value in input_tuple.input_types
            ):
                continue
            values.append(provider.create_candidate(context, return_type, input_tuple))
    return tuple(values)


def _contains_partial(value: IRType) -> bool:
    if isinstance(value, DistributedType):
        return value.partial is not None
    if isinstance(value, TupleType):
        return any(_contains_partial(field) for field in value.fields)
    return False


def source_kind_for_node(node) -> DistributedReshardSourceKind:
    if node.op in {"builtin.weight", "builtin.const_asset"}:
        return DistributedReshardSourceKind.CONSTANT
    if node.op == "builtin.var":
        return DistributedReshardSourceKind.FUNCTION_PARAMETER
    return DistributedReshardSourceKind.INTERNAL


def function_boundary_id(function_name: str) -> str:
    return f"@function:{function_name}"


def function_output_type(
    module: IRModule,
    function_name: str,
    output_id: str,
    placement: Placement,
) -> IRType:
    """Return the boundary type demanded after AutoDistributed.

    Like nncase, function calls and their callees remain in one distributed
    domain.  ``builtin.call`` candidates expose a recursively broadcast ABI,
    so every callee output boundary must demand that same type.  Returning an
    internal function to its imported logical type would select a reshard plan
    that cannot terminate at the distributed call ABI materialized later.
    """

    output_type = module.node_map[output_id].type
    return broadcast_ir_type(output_type, placement)


def _plan_text(plan: DistributedReshardPlan) -> str:
    return " -> ".join(_type_text(step) for step in plan.step_types)


def _type_text(value: IRType) -> str:
    if isinstance(value, DistributedType):
        return f"Dist({','.join(str(item) for item in value.axis_policies)};{value.placement})"
    if isinstance(value, TupleType):
        return "(" + ",".join(_type_text(field) for field in value.fields) + ")"
    return type(value).__name__


def _dot(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")


def _load_cp_model():
    try:
        from ortools.sat.python import cp_model
    except ImportError as error:
        raise RuntimeError(
            "FlagMega AutoDistributed requires OR-Tools CP-SAT; install ortools==9.10.4067.") from error
    return cp_model


def _positive_env_float(name: str, default: float) -> float:
    try:
        value = float(os.environ.get(name, str(default)))
    except ValueError:
        return default
    return value if value > 0 else default


def _positive_env_int(name: str, default: int) -> int:
    try:
        value = int(os.environ.get(name, str(default)))
    except ValueError:
        return default
    return value if value > 0 else default


__all__ = [
    "CandidateBucket",
    "ReshardSite",
    "SearchGraph",
    "SearchResult",
    "build_search_graph",
    "function_boundary_id",
    "function_output_type",
    "source_kind_for_node",
    "graph_dot",
    "solve_search_graph",
]
