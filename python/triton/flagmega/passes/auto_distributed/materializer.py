# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Build the expression selected by AutoDistributed CP-SAT."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace

from triton.flagmega.errors import IRVerificationError
from triton.flagmega.ir import (
    Candidate,
    DistributedType,
    Function,
    IRModule,
    IRType,
    Node,
    SelectionPoint,
    SelectionRecord,
    TupleType,
)
from triton.flagmega.passes.auto_distributed.reshard import DistributedReshardPlan
from triton.flagmega.passes.auto_distributed.realization import (
    DistributedReshardRealization,
    DistributedReshardRealizationContext,
    DistributedReshardSourceKind,
    DistributedReshardUsageKind,
)
from triton.flagmega.passes.auto_distributed.search import (
    SearchResult,
    function_boundary_id,
    function_output_type,
    source_kind_for_node,
)


class DistributedMaterializer:
    def __init__(
        self,
        result: SearchResult,
        *,
        policy: str,
        proposal_points: Mapping[str, SelectionPoint] | None = None,
        proposal_records: Mapping[str, SelectionRecord] | None = None,
    ) -> None:
        self.result = result
        self.module = result.graph.module
        self.policy = policy
        self.proposal_points = dict(proposal_points or {})
        self.proposal_records = dict(proposal_records or {})
        self.nodes: list[Node] = []
        self.realized: dict[str, Node] = {}
        self.reshards: dict[tuple[str, IRType, tuple[IRType, ...]], Node] = {}
        self._ordinal = 0

    def run(self) -> IRModule:
        buckets = self.result.graph.bucket_map
        for source in self.module.nodes:
            candidate = self.result.selected[source.id]
            inputs = tuple(
                self._require_type(
                    self.realized[input_id],
                    candidate.input_types[index],
                    source.id,
                    index,
                    self.result.selected_reshards.get((input_id, source.id, index)),
                    DistributedReshardUsageKind.INTERNAL,
                )
                for index, input_id in enumerate(source.inputs)
            )
            metadata = dict(source.metadata)
            if buckets[source.id].executable and len(buckets[source.id].candidates) > 1:
                metadata.update({
                    "distributed_candidate": candidate.id,
                    "distributed_reason": candidate.reason,
                    "distributed_placement": str(self.result.graph.placement),
                })
            node = replace(
                source,
                op=candidate.target_op or source.op,
                inputs=tuple(value.id for value in inputs),
                type=candidate.return_type,
                attrs=(
                    source.attrs
                    if candidate.target_attrs is None
                    else candidate.target_attrs
                ),
                metadata=metadata,
            )
            self.nodes.append(node)
            self.realized[source.id] = node

        self._reconcile_function_parameters()
        self._fold_reconciled_identity_reshards()

        calls_by_callee: dict[str, list[Node]] = {}
        for node in self.nodes:
            if node.op == "builtin.call":
                calls_by_callee.setdefault(str(node.attrs["callee"]), []).append(node)

        functions: list[Function] = []
        for function in self.module.functions:
            call_output_types: tuple[IRType, ...] | None = None
            calls = calls_by_callee.get(function.name, ())
            if calls:
                result_types = {call.type for call in calls}
                if len(result_types) != 1:
                    raise IRVerificationError(
                        f"AutoDistributed selected incompatible result ABIs for @{function.name}.",
                        stage=self.module.stage,
                    )
                call_type = next(iter(result_types))
                call_output_types = (
                    call_type.fields if isinstance(call_type, TupleType) else (call_type,)
                )
                if len(call_output_types) != len(function.outputs):
                    raise IRVerificationError(
                        f"Call result ABI arity does not match @{function.name}.",
                        stage=self.module.stage,
                    )
            outputs: list[str] = []
            for output_index, output_id in enumerate(function.outputs):
                source = self.realized[output_id]
                target_type = (
                    call_output_types[output_index]
                    if call_output_types is not None
                    else function_output_type(
                        self.module,
                        function.name,
                        output_id,
                        self.result.graph.placement,
                    )
                )
                outputs.append(self._require_type(
                    source,
                    target_type,
                    function.name,
                    output_index,
                    self.result.selected_reshards.get(
                        (output_id, function_boundary_id(function.name), output_index)
                    ),
                    (
                        DistributedReshardUsageKind.PROGRAM_OUTPUT
                        if function.name == self.module.entry
                        else DistributedReshardUsageKind.FUNCTION_BOUNDARY
                    ),
                ).id)
            functions.append(replace(function, outputs=tuple(outputs)))

        points, selections = distribution_selection_state(
            self.result,
            policy=self.policy,
            proposal_points=self.proposal_points,
            proposal_records=self.proposal_records,
        )

        old_distribution_ids = {point.id for point in self.module.selection_points if point.kind == "distribution"}
        metadata = dict(self.module.metadata)
        metadata.pop("auto_distribution_proposal", None)
        metadata.update({
            "auto_distribution": {
                "schema": "flagmega.auto-distributed/v1",
                "placement": self.result.graph.placement.to_data(),
                "solver": "ortools-cp-sat",
                "status": self.result.status,
                "objective": self.result.objective,
                "reshard_nodes": sum(node.op.startswith("distributed.") for node in self.nodes),
            },
        })
        return replace(
            self.module,
            nodes=tuple(self.nodes),
            functions=tuple(functions),
            metadata=metadata,
            selection_points=tuple(
                point for point in self.module.selection_points if point.id not in old_distribution_ids
            ) + tuple(points),
            selections=tuple(
                record for record in self.module.selections if record.point_id not in old_distribution_ids
            ) + tuple(selections),
        )

    def _reconcile_function_parameters(self) -> None:
        """Materialize one physical ABI shared by every call of a function.

        Imported function parameters are logical originators during the search,
        while call operands acquire physical distributed types.  A callee has
        one ABI, so all call sites must agree and its parameter variables are
        refined to those selected operand types before verification.  The body
        keeps its already-selected reshard edges from that ABI into internal
        layouts.
        """

        calls_by_callee: dict[str, list[Node]] = {}
        physical_nodes = {node.id: node for node in self.nodes}
        for node in self.nodes:
            if node.op == "builtin.call":
                calls_by_callee.setdefault(str(node.attrs["callee"]), []).append(node)
        replacements: dict[str, Node] = {}
        for function in self.module.functions:
            calls = calls_by_callee.get(function.name, ())
            if not calls:
                continue
            signatures = {
                tuple(physical_nodes[input_id].type for input_id in call.inputs)
                for call in calls
            }
            if len(signatures) != 1:
                raise IRVerificationError(
                    f"AutoDistributed selected incompatible call ABIs for @{function.name}.",
                    stage=self.module.stage,
                )
            parameter_types = next(iter(signatures))
            if len(parameter_types) != len(function.parameters):
                raise IRVerificationError(
                    f"Call ABI arity does not match @{function.name}.",
                    stage=self.module.stage,
                )
            for parameter_id, parameter_type in zip(function.parameters, parameter_types):
                parameter = self.realized[parameter_id]
                replacement = replace(parameter, type=parameter_type)
                self.realized[parameter_id] = replacement
                replacements[parameter_id] = replacement
        if replacements:
            self.nodes = [replacements.get(node.id, node) for node in self.nodes]

    def _fold_reconciled_identity_reshards(self) -> None:
        """Remove adapters made redundant by the finalized callee ABI.

        Search treats imported callee parameters as logical originators, so a
        body use can initially require a logical-to-distributed adapter.  Once
        all call sites agree, ``_reconcile_function_parameters`` refines that
        parameter to the distributed call ABI.  The old adapter then has
        identical source/result types and is neither a Boxing operation nor a
        ShardedView.  Keeping it creates an executable-looking node for which
        no legal TIR candidate exists.
        """

        by_id = {node.id: node for node in self.nodes}
        aliases: dict[str, str] = {}

        def resolve(value: str) -> str:
            while value in aliases:
                value = aliases[value]
            return value

        rewritten: list[Node] = []
        for node in self.nodes:
            inputs = tuple(resolve(value) for value in node.inputs)
            prepared = replace(node, inputs=inputs) if inputs != node.inputs else node
            if prepared.op in {"distributed.boxing", "distributed.sharded_view"}:
                if len(inputs) != 1:
                    raise IRVerificationError(
                        f"Distributed adapter {prepared.id!r} must have one input.",
                        stage=self.module.stage,
                        node_id=prepared.id,
                    )
                source = by_id[inputs[0]]
                if source.type == prepared.type:
                    aliases[prepared.id] = inputs[0]
                    continue
            rewritten.append(prepared)
            by_id[prepared.id] = prepared

        if not aliases:
            return
        self.nodes = rewritten
        self.realized = {
            node_id: (
                by_id[resolve(node.id)]
                if node.id in aliases
                else by_id[node.id]
            )
            for node_id, node in self.realized.items()
        }
        self.reshards = {
            key: by_id[resolve(node.id)]
            for key, node in self.reshards.items()
        }

    def _require_type(
        self,
        source: Node,
        target_type: IRType,
        consumer: str,
        input_index: int,
        plan: DistributedReshardPlan | None,
        usage_kind: DistributedReshardUsageKind,
    ) -> Node:
        if source.type == target_type:
            return source
        if plan is None:
            raise IRVerificationError(
                f"CP-SAT did not select a reshard path from {source.type!r} to {target_type!r} "
                f"for {consumer} input {input_index}.",
                stage=self.module.stage,
                node_id=consumer,
            )
        key = (source.id, target_type, plan.step_types)
        cached = self.reshards.get(key)
        if cached is not None:
            return cached
        if not plan.step_types or plan.step_types[-1] != target_type:
            raise IRVerificationError(
                f"Selected reshard path does not terminate at {target_type!r} for {consumer} input {input_index}.",
                stage=self.module.stage,
                node_id=consumer,
            )
        current = source
        original_source_kind = (
            DistributedReshardSourceKind.CONSTANT
            if source.id in self.result.graph.constant_ids
            else source_kind_for_node(source)
        )
        for step_index, step in enumerate(plan.step_types):
            self._ordinal += 1
            context = DistributedReshardRealizationContext(
                current.type,
                step,
                original_source_kind if step_index == 0 else DistributedReshardSourceKind.INTERNAL,
                usage_kind if step_index == len(plan.step_types) - 1 else DistributedReshardUsageKind.INTERNAL,
            )
            realization = self.result.graph.realization_policy.classify(context)
            if realization == DistributedReshardRealization.UNSUPPORTED:
                raise IRVerificationError(
                    f"Target realization policy rejected the selected reshard step {current.type!r} -> {step!r}.",
                    stage=self.module.stage,
                    node_id=consumer,
                )
            op = (
                "distributed.sharded_view"
                if realization == DistributedReshardRealization.SHARDED_VIEW
                else "distributed.boxing"
            )
            node = Node(
                id=f"{source.id}.reshard{self._ordinal}",
                op=op,
                inputs=(current.id,),
                type=step,
                attrs={"new_type": step},
                metadata={
                    "source_type": _type_summary(current.type),
                    "target_type": _type_summary(step),
                    "consumer": consumer,
                    "input_index": input_index,
                    "realization": realization.value,
                },
            )
            self.nodes.append(node)
            current = node
        self.reshards[key] = current
        return current


def _type_summary(value: IRType) -> str:
    if isinstance(value, DistributedType):
        policies = ",".join(str(policy) for policy in value.axis_policies)
        partial = "" if value.partial is None else f";partial={value.partial}"
        return f"Dist({value.placement};{policies}{partial})"
    return type(value).__name__


def distribution_selection_state(
    result: SearchResult,
    *,
    policy: str,
    proposal_points: Mapping[str, SelectionPoint] | None = None,
    proposal_records: Mapping[str, SelectionRecord] | None = None,
) -> tuple[tuple[SelectionPoint, ...], tuple[SelectionRecord, ...]]:
    """Describe globally solved distribution choices as editable IR state."""

    prior_points = dict(proposal_points or {})
    prior_records = dict(proposal_records or {})
    points: list[SelectionPoint] = []
    records: list[SelectionRecord] = []
    for bucket in result.graph.buckets:
        if not bucket.executable or len(bucket.candidates) <= 1:
            continue
        selected = result.selected[bucket.node_id]
        point_id = f"distribution.{bucket.node_id}"
        generated = SelectionPoint(
            id=point_id,
            kind="distribution",
            candidates=tuple(
                Candidate(
                    candidate.id,
                    {
                        "return_type": _type_summary(candidate.return_type),
                        "reason": candidate.reason,
                        "target_op": candidate.target_op
                        or result.graph.module.node_map[bucket.node_id].op,
                        "placement": str(result.graph.placement),
                    },
                    {
                        "objective": {
                            "value": candidate.operation_cost,
                            "kind": candidate.objective_kind,
                            "model": candidate.objective_model,
                            "evidence": candidate.objective_evidence,
                        },
                    },
                )
                for candidate in bucket.candidates
            ),
            default_candidate=selected.id,
            owner=bucket.node_id,
        )
        point = prior_points.get(point_id, generated)
        if (
            point.kind != "distribution"
            or point.owner != bucket.node_id
            or tuple(candidate.id for candidate in point.candidates)
            != tuple(candidate.id for candidate in generated.candidates)
        ):
            raise IRVerificationError(
                f"AutoDistributed proposal {point_id!r} no longer matches the "
                "reconstructed candidate graph.",
                stage=result.graph.module.stage,
                node_id=bucket.node_id,
            )
        record = prior_records.get(point_id)
        if record is None:
            record = SelectionRecord(
                point_id=point.id,
                candidate_id=selected.id,
                origin="ortools-cp-sat",
                policy=policy,
                rationale=selected.reason,
                evidence=(
                    "AutoDistributed/Costs/Pick.txt",
                    *selected.objective_evidence,
                ),
            )
        elif record.candidate_id != selected.id:
            raise IRVerificationError(
                f"AutoDistributed selection {point_id!r} does not match the "
                "constrained solver result.",
                stage=result.graph.module.stage,
                node_id=bucket.node_id,
            )
        points.append(point)
        records.append(record)
    return tuple(points), tuple(records)


__all__ = ["DistributedMaterializer", "distribution_selection_state"]
