# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Commit semantic TIR operand alignment contracts before storage/view planning."""

from dataclasses import replace

from triton.flagmega.codegen.triton.microkernels.core import TIRMicroKernelContext
from triton.flagmega.errors import IRVerificationError
from triton.flagmega.ir import kernel_dispatch_of, verify_module
from triton.flagmega.ir.tir.kernel_definition import replace_kernel_callables
from triton.flagmega.ir.tir.transfer_pipeline_validation import verify_transfer_sources


def implementation_alignments(function, implementation):
    """Read declared interface requirements, independent of candidate preference."""
    dispatch = kernel_dispatch_of(function)
    pipeline = implementation.transfer_pipeline
    requirements = {}
    if pipeline is None:
        return requirements
    verify_transfer_sources(function, dispatch, implementation, check_alignment=False)
    for channel in pipeline.channels:
        for index in channel.source_argument_indices:
            name = dispatch.arguments[index]
            requirements[name] = max(requirements.get(name, 1), channel.source_alignment_bytes)
    return requirements


def satisfies_alignment_contract(function, implementation):
    return all(function.parameter_map[name].alignment_bytes is None
               or required <= function.parameter_map[name].alignment_bytes
               for name, required in implementation_alignments(function, implementation).items())


def plan_storage_alignments(module, registry, implementation_model, *, capability=None):
    """Preserve all currently legal implementation interfaces in the planned ABI.

    No selection records/defaults are read or written. Existing explicit
    contracts remain authoritative, including when resuming on another target.
    """
    functions = []
    changed = False
    for function in module.kernel_callable_map.values():
        parameters = tuple(p for p in function.parameters if p.role.value != "metadata")
        dispatch = kernel_dispatch_of(function)
        if dispatch is None or all(p.alignment_bytes is not None for p in parameters):
            functions.append(function)
            continue
        implementations = ()
        if dispatch.microkernel is not None:
            # These are already concrete semantic-TIR operations, not a later
            # microkernel choice. Their interface is part of the incoming IR.
            implementations = (dispatch.microkernel,)
        elif (provider := registry.provider_for(dispatch.semantic_op)) is not None:
            proposal = provider.propose(TIRMicroKernelContext(module, function, dispatch, implementation_model))
            if proposal is not None:
                implementations = tuple(implementation_model.implementation(candidate.id) for candidate in proposal.candidates)
                if any(value is None for value in implementations):
                    raise IRVerificationError(f"Alignment planning for @{function.name} references an unknown implementation.")
        legal = tuple(value for value in implementations if satisfies_alignment_contract(function, value)
                      and (capability is None or capability.supports(value.requires)))
        if implementations and not legal:
            raise IRVerificationError(f"No implementation satisfies the declared alignment contract of @{function.name}.")
        requirements = {}
        for implementation in legal:
            for name, alignment in implementation_alignments(function, implementation).items():
                requirements[name] = max(requirements.get(name, 1), alignment)
        functions.append(replace(function, parameters=tuple(
            replace(parameter, alignment_bytes=requirements.get(parameter.name, 1))
            if parameter.role.value != "metadata" and parameter.alignment_bytes is None else parameter
            for parameter in function.parameters)))
        changed = True
    return verify_module(replace_kernel_callables(module, functions)) if changed else module


__all__ = ["implementation_alignments", "plan_storage_alignments", "satisfies_alignment_contract"]
