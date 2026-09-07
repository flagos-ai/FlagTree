# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Agent-editable target microkernel selection over first-class semantic TIR."""

from __future__ import annotations

from dataclasses import replace
from triton.flagmega.ir.tir.kernel_definition import replace_kernel_dispatch, replace_kernel_callables

from triton.flagmega.errors import CodegenError, IRVerificationError
from triton.flagmega.ir import (
    IRModule,
    SelectionPoint,
    TIRMicroKernelSelection,
    kernel_dispatch_of,
)

from .core import TIRMicroKernelContext, TIRMicroKernelProviderRegistry
from .materialization import (
    materialize_shared_workspace_buffers,
    validate_transfer_pipeline,
)


class TritonMicroKernelSelectionPolicy:
    """Propose and materialize physical implementations after TIR canonicalization."""

    def __init__(self, registry: TIRMicroKernelProviderRegistry) -> None:
        self.registry = registry

    def propose(self, module: IRModule, target) -> IRModule:
        existing = {point.id for point in module.selection_points}
        points: list[SelectionPoint] = []
        for function in module.kernel_callable_map.values():
            dispatch = kernel_dispatch_of(function)
            if dispatch is None or dispatch.microkernel is not None:
                continue
            provider = self.registry.provider_for(dispatch.semantic_op)
            if provider is None:
                continue
            point_id = self.point_id(function.name)
            if point_id in existing:
                continue
            proposal = provider.propose(TIRMicroKernelContext(
                module,
                function,
                dispatch,
                target.triton_implementation_model,
            ))
            if proposal is None:
                continue
            points.append(SelectionPoint(
                point_id,
                "tir_microkernel",
                proposal.candidates,
                proposal.default_candidate,
                owner=None,
            ))
        return target.add_default_selections(
            module,
            tuple(points),
            "Select a target implementation after semantic TIR canonicalization.",
            policy_version=target.policy_version,
        )

    def apply(self, module: IRModule, target) -> IRModule:
        point_map = {point.id: point for point in module.selection_points}
        selection_map = module.selection_map
        functions = []
        for function in module.kernel_callable_map.values():
            dispatch = kernel_dispatch_of(function)
            if dispatch is None:
                functions.append(function)
                continue
            point_id = self.point_id(function.name)
            record = selection_map.get(point_id)
            if record is None:
                if (
                    dispatch.microkernel is None
                    and self.registry.provider_for(dispatch.semantic_op) is not None
                ):
                    raise IRVerificationError(
                        f"TIR microkernel point {point_id!r} has no selection record.",
                        stage=module.stage,
                    )
                functions.append(function)
                continue
            point = point_map[point_id]
            candidate = next(
                value for value in point.candidates
                if value.id == record.candidate_id
            )
            implementation = target.triton_implementation_model.implementation(
                candidate.id
            )
            if implementation is None:
                raise IRVerificationError(
                    f"TIR microkernel selection {candidate.id!r} is absent from the "
                    f"active implementation model.",
                    stage=module.stage,
                )
            expected_facts = dict(implementation.facts)
            if implementation.requires:
                expected_facts["requires"] = implementation.requires
            if (
                dict(candidate.parameters) != dict(implementation.parameters)
                or dict(candidate.facts) != expected_facts
            ):
                raise IRVerificationError(
                    f"TIR microkernel candidate {candidate.id!r} was edited away from "
                    "the active implementation catalog; add a named implementation "
                    "variant instead.",
                    stage=module.stage,
                )
            if (
                dispatch.microkernel is not None
                and dispatch.microkernel.implementation != implementation.id
            ):
                raise CodegenError(
                    f"PrimFunction @{function.name} already materializes microkernel "
                    f"{dispatch.microkernel.implementation!r}, but its selection record "
                    f"chooses {implementation.id!r}."
                )
            selected = TIRMicroKernelSelection(
                implementation=implementation.id,
                family=implementation.family,
                variant=implementation.variant,
                parameters=implementation.parameters,
                facts=implementation.facts,
                requires=implementation.requires,
                shared_workspaces=implementation.shared_workspaces,
                transfer_pipeline=implementation.transfer_pipeline,
            )
            validate_transfer_pipeline(
                function,
                dispatch,
                selected,
                stage=module.stage,
            )
            rewritten = replace(
                dispatch,
                microkernel=selected,
                shared_workspace_buffers=materialize_shared_workspace_buffers(
                    function, selected
                ),
            )
            functions.append(replace_kernel_dispatch(function, rewritten))
        return replace_kernel_callables(module, functions)

    @staticmethod
    def point_id(function_name: str) -> str:
        return f"microkernel.{function_name}"


__all__ = ["TritonMicroKernelSelectionPolicy"]
