# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Independent SM90 legality verifier for candidate and physical target IR."""

from __future__ import annotations

from collections.abc import Mapping

from triton.flagmega.errors import IRSchemaError, IRVerificationError
from triton.flagmega.ir import (
    IRModule,
    ProducerConsumerRegion,
    execution_calls_of,
    kernel_dispatch_of,
    verify_buffer_plan,
)
from triton.flagmega.ir.bufferization import (
    MemorySynchronizationPlan,
    SYNCHRONIZATION_SCHEMA,
)
from triton.flagmega.passes.tir.bufferize import plan_memory_synchronization
from triton.flagmega.passes.tir import (
    memory_synchronization_from_execution_functions,
)
from triton.flagmega.targets.nvidia.capability import Sm90Capability
from triton.flagmega.targets.ntt_options import NttTargetOptions
from triton.flagmega.codegen.triton.implementation import TritonImplementationModel
from triton.flagmega.codegen.triton.kernel_dispatch import selected_kernel_nodes
from triton.flagmega.targets.nvidia.package import sm90_codegen_package_plan


def candidate_requirements(candidate) -> tuple[str, ...]:
    return _requirements(candidate.facts.get("requires", ()))


def verify_sm90_module(
    module: IRModule,
    capability: Sm90Capability,
    *,
    target_name: str,
    policy_version: str,
    target_options: NttTargetOptions,
    implementation_model: TritonImplementationModel,
) -> None:
    points = {point.id: point for point in module.selection_points}
    for point in module.selection_points:
        for candidate in point.candidates:
            _verify_candidate(candidate, capability, module.stage)
            if point.kind in {"tir", "tir_microkernel"}:
                _verify_implementation_candidate(
                    candidate,
                    implementation_model,
                    module.stage,
                    embedded_identity=point.kind == "tir",
                )
    for selection in module.selections:
        point = points[selection.point_id]
        candidate = next(
            candidate
            for candidate in point.candidates
            if candidate.id == selection.candidate_id
        )
        _verify_candidate(candidate, capability, module.stage)
        if point.kind in {"tir", "tir_microkernel"}:
            _verify_implementation_candidate(
                candidate,
                implementation_model,
                module.stage,
                embedded_identity=point.kind == "tir",
            )

    tir_stages = {
        "selected_tir",
        "canonicalized_tir",
        "microkernel_candidates",
        "selected_microkernels",
        "packaged_tir",
        "memory_placed_tir",
        "allocated_tir",
        "scheduled_tir",
        "synchronized_tir",
        "bufferized_tir",
    }
    if module.stage in tir_stages:
        if module.metadata.get("target") != target_name:
            raise IRVerificationError(
                f"SM90 TIR target identity must be {target_name!r}.",
                stage=module.stage,
            )
        if module.metadata.get("target_policy") != policy_version:
            raise IRVerificationError(
                f"SM90 TIR policy must be {policy_version!r}.",
                stage=module.stage,
            )
        options_snapshot = module.metadata.get("target_options")
        try:
            snapshot_options = (
                NttTargetOptions.from_data(options_snapshot)
                if isinstance(options_snapshot, Mapping)
                else None
            )
        except (IRSchemaError, KeyError, TypeError, ValueError) as error:
            raise IRVerificationError(
                "SM90 TIR has an invalid NTT target-options snapshot.",
                stage=module.stage,
            ) from error
        if snapshot_options != target_options:
            raise IRVerificationError(
                "SM90 TIR NTT target-options snapshot differs from the active target.",
                stage=module.stage,
            )
        implementation_snapshot = module.metadata.get("target_implementation_model")
        if implementation_snapshot != implementation_model.snapshot():
            raise IRVerificationError(
                "SM90 TIR implementation-model snapshot differs from the active target.",
                stage=module.stage,
            )
        snapshot = module.metadata.get("target_capability")
        try:
            snapshot_capability = (
                Sm90Capability.from_data(snapshot)
                if isinstance(snapshot, Mapping)
                else None
            )
        except (IRSchemaError, KeyError, TypeError, ValueError) as error:
            raise IRVerificationError(
                "SM90 TIR has an invalid capability snapshot.",
                stage=module.stage,
            ) from error
        if snapshot_capability != capability:
            raise IRVerificationError(
                "SM90 TIR capability snapshot differs from the active target.",
                stage=module.stage,
            )
        launch = module.metadata.get("launch_contract")
        if not isinstance(launch, Mapping):
            raise IRVerificationError("SM90 TIR requires a launch contract.", stage=module.stage)
        num_warps = launch.get("num_warps")
        if (
            not isinstance(num_warps, int)
            or isinstance(num_warps, bool)
            or num_warps <= 0
            or num_warps * capability.warp_size > capability.max_threads_per_block
        ):
            raise IRVerificationError(
                f"SM90 launch num_warps {num_warps!r} exceeds machine limits.",
                stage=module.stage,
            )
        dispatches = tuple(
            dispatch
            for function in (*module.prim_functions, *module.kernel_definitions)
            if (dispatch := kernel_dispatch_of(function)) is not None
        )
        package_is_required = module.stage in {
            "packaged_tir", "memory_placed_tir", "allocated_tir", "scheduled_tir", "synchronized_tir", "bufferized_tir"
        }
        package_is_legacy_available = (
            module.stage == "selected_tir"
            and all(dispatch.microkernel is not None for dispatch in dispatches)
        )
        if package_is_required or package_is_legacy_available:
            package_plan = module.metadata.get("codegen_package_plan")
            expected_package_plan = sm90_codegen_package_plan(
                module,
                selected_kernel_nodes(module),
                capability,
                target_options,
            )
            if package_plan != expected_package_plan:
                raise IRVerificationError(
                    "SM90 codegen package plan differs from the selected TIR implementation set.",
                    stage=module.stage,
                )
        legal_ops = {
            "builtin.var",
            "builtin.weight",
            "builtin.const_asset",
            "builtin.get_item",
            "builtin.none",
            "builtin.tuple",
            "tir.buffer",
            "tir.buffer_view",
            "tir.call",
            "tir.scalar_const",
        }
        if module.stage not in {"synchronized_tir", "bufferized_tir"}:
            # The logical view is consumed by bufferization and becomes a
            # first-class tir.buffer_view once its MemSpan is known.
            legal_ops.add("distributed.sharded_view")
        illegal = [
            node.id
            for node in module.nodes
            if node.op not in legal_ops
        ]
        if illegal:
            raise IRVerificationError(
                f"SM90 semantic TIR contains non-lowered compute nodes: {illegal}.",
                stage=module.stage,
            )
        for node in module.nodes:
            if node.op == "tir.kernel":
                raise IRVerificationError(
                    "SM90 selected TIR must materialize typed KernelDefinition references.",
                    stage=module.stage,
                    node_id=node.id,
                )
        for function in (*module.prim_functions, *module.kernel_definitions):
            dispatch = kernel_dispatch_of(function)
            if dispatch is None:
                continue
            require_microkernel = module.stage in {
                "selected_microkernels",
                "packaged_tir",
                "memory_placed_tir",
                "allocated_tir",
                "scheduled_tir",
                "synchronized_tir",
                "bufferized_tir",
            }
            if dispatch.microkernel is None and not require_microkernel:
                continue
            if dispatch.microkernel is None:
                raise IRVerificationError(
                    f"SM90 PrimFunction @{function.name} has no selected microkernel.",
                    stage=module.stage,
                )
            implementation = implementation_model.implementation(
                dispatch.microkernel.implementation
            )
            if implementation is None or _implementation_mismatches(
                dispatch, implementation
            ):
                raise IRVerificationError(
                    f"SM90 PrimFunction @{function.name} does not match the active "
                    "Triton implementation model.",
                    stage=module.stage,
                )
            missing = capability.missing(_requirements(dispatch.facts.get("requires", ())))
            if missing:
                raise IRVerificationError(
                    f"SM90 PrimFunction @{function.name} requires unavailable features {missing}.",
                    stage=module.stage,
                )
            pipeline = dispatch.microkernel.transfer_pipeline
            body_fields = function.body.fields
            is_lowered = (
                len(body_fields) == 1
                and isinstance(body_fields[0], ProducerConsumerRegion)
            )
            if (module.stage == "bufferized_tir" and pipeline is not None and not is_lowered
                    and function.name in module.prim_function_map):
                raise IRVerificationError(
                    f"SM90 transfer-pipeline PrimFunction @{function.name} has no "
                    "explicit producer/consumer region.",
                    stage=module.stage,
                )
            if module.stage != "bufferized_tir" and is_lowered:
                raise IRVerificationError(
                    f"SM90 PrimFunction @{function.name} was pipeline-lowered before "
                    "the LowerTransferPipelineRegions stage.",
                    stage=module.stage,
                )

        if module.stage == "allocated_tir" and module.execution_functions:
            raise IRVerificationError(
                "Execution functions were materialized before their TIR pass.",
                stage=module.stage,
            )
        if module.stage in {"scheduled_tir", "synchronized_tir", "bufferized_tir"}:
            if set(module.execution_function_map) != set(module.function_map):
                raise IRVerificationError(
                    "Post-Bufferize TIR requires one ExecutionFunction per graph function.",
                    stage=module.stage,
                )
            pipeline_functions = _pipeline_execution_functions(module)
            for function in module.execution_functions:
                fields = function.body.fields
                is_lowered = (
                    len(fields) == 1
                    and isinstance(fields[0], ProducerConsumerRegion)
                )
                expected = function.name in pipeline_functions
                if module.stage == "bufferized_tir" and expected != is_lowered:
                    raise IRVerificationError(
                        f"ExecutionFunction @{function.name} pipeline-region state "
                        "does not match its call graph.",
                        stage=module.stage,
                    )
                if module.stage != "bufferized_tir" and is_lowered:
                    raise IRVerificationError(
                        f"ExecutionFunction @{function.name} was pipeline-lowered "
                        "before LowerTransferPipelineRegions.",
                        stage=module.stage,
                    )

    if module.stage in {"allocated_tir", "scheduled_tir", "synchronized_tir", "bufferized_tir"}:
        plan = verify_buffer_plan(module)
        if module.stage in {"synchronized_tir", "bufferized_tir"}:
            data = module.metadata.get("memory_synchronization")
            if not isinstance(data, Mapping) or data.get("schema") != SYNCHRONIZATION_SCHEMA:
                raise IRVerificationError(
                    f"bufferized_tir requires {SYNCHRONIZATION_SCHEMA} metadata.",
                    stage=module.stage,
                )
            actual = MemorySynchronizationPlan.from_data(data)
            materialized = memory_synchronization_from_execution_functions(
                module
            )
            if materialized != actual:
                raise IRVerificationError(
                    "ExecutionFunction barriers differ from the memory "
                    "synchronization diagnostic plan.",
                    stage=module.stage,
                )
            expected = plan_memory_synchronization(module, plan)
            if actual != expected:
                raise IRVerificationError(
                    "Memory synchronization plan does not match concrete buffer hazards.",
                    stage=module.stage,
                )


def _pipeline_execution_functions(module: IRModule) -> frozenset[str]:
    result = {
        function.name
        for function in module.execution_functions
        if any(
            (
                primitive := module.kernel_callable_map.get(call.callee)
            ) is not None
            and (dispatch := kernel_dispatch_of(primitive)) is not None
            and dispatch.microkernel is not None
            and dispatch.microkernel.transfer_pipeline is not None
            for call in execution_calls_of(function)
        )
    }
    changed = True
    while changed:
        changed = False
        for function in module.execution_functions:
            if function.name in result:
                continue
            if any(call.callee in result for call in execution_calls_of(function)):
                result.add(function.name)
                changed = True
    return frozenset(result)


def _verify_candidate(candidate, capability: Sm90Capability, stage: str) -> None:
    missing = capability.missing(candidate_requirements(candidate))
    if missing:
        raise IRVerificationError(
            f"SM90 candidate {candidate.id!r} requires unavailable features {missing}.",
            stage=stage,
        )


def _verify_implementation_candidate(
    candidate,
    implementation_model: TritonImplementationModel,
    stage: str,
    *,
    embedded_identity: bool,
) -> None:
    implementation = implementation_model.implementation(candidate.id)
    if implementation is None:
        raise IRVerificationError(
            f"SM90 candidate {candidate.id!r} is absent from the active Triton "
            "implementation model.",
            stage=stage,
        )
    mismatches = _implementation_mismatches(
        candidate, implementation, embedded_identity=embedded_identity
    )
    if mismatches:
        raise IRVerificationError(
            f"SM90 candidate {candidate.id!r} implementation parameters differ "
            f"from the active model: {mismatches}.",
            stage=stage,
        )
    stages = candidate.parameters.get("stages")
    if stages is not None and (
        not isinstance(stages, int)
        or isinstance(stages, bool)
        or stages < 1
        or stages > 8
    ):
        raise IRVerificationError(
            f"SM90 candidate {candidate.id!r} has invalid stage count {stages!r}.",
            stage=stage,
        )
    block_k = candidate.parameters.get("block_k")
    if block_k is not None and (
        not isinstance(block_k, int)
        or isinstance(block_k, bool)
        or block_k <= 0
    ):
        raise IRVerificationError(
            f"SM90 candidate {candidate.id!r} has invalid block_k {block_k!r}.",
            stage=stage,
        )


def _implementation_mismatches(
    value, implementation, *, embedded_identity: bool = True
) -> dict[str, object]:
    from triton.flagmega.ir import KernelDispatch

    if isinstance(value, KernelDispatch):
        microkernel = value.microkernel
        if microkernel is None:
            return {"microkernel": (implementation.id, None)}
        mismatches = {
            "family": (implementation.family, microkernel.family),
            "variant": (implementation.variant, microkernel.variant),
        }
        mismatches = {
            name: pair for name, pair in mismatches.items()
            if pair[0] != pair[1]
        }
        actual_parameters = microkernel.parameters
        actual_facts = microkernel.facts
        actual_requirements = set(microkernel.requires)
        if microkernel.shared_workspaces != implementation.shared_workspaces:
            mismatches["shared_workspaces"] = (
                implementation.shared_workspaces,
                microkernel.shared_workspaces,
            )
        if microkernel.transfer_pipeline != implementation.transfer_pipeline:
            mismatches["transfer_pipeline"] = (
                implementation.transfer_pipeline,
                microkernel.transfer_pipeline,
            )
    else:
        mismatches = {}
        actual_parameters = value.parameters
        actual_facts = value.facts
        actual_requirements = set(_requirements(value.facts.get("requires", ())))
    mismatches.update({
        name: (expected, actual_parameters.get(name))
        for name, expected in implementation.parameters.items()
        if actual_parameters.get(name) != expected
    })
    if embedded_identity and not isinstance(value, KernelDispatch):
        for name, expected in (
            ("family", implementation.family),
            ("variant", implementation.variant),
        ):
            if value.parameters.get(name) != expected:
                mismatches[name] = (expected, value.parameters.get(name))
    missing_requirements = tuple(
        requirement
        for requirement in implementation.requires
        if requirement not in actual_requirements
    )
    if missing_requirements:
        mismatches["requires"] = (implementation.requires, tuple(actual_requirements))
    for name, expected in implementation.facts.items():
        if actual_facts.get(name) != expected:
            mismatches[f"facts.{name}"] = (expected, actual_facts.get(name))
    return mismatches


def _requirements(value) -> tuple[str, ...]:
    if isinstance(value, str):
        return (value,)
    return tuple(str(item) for item in value)


__all__ = ["candidate_requirements", "verify_sm90_module"]
