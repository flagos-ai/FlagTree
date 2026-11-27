# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT

from dataclasses import replace

import pytest

from triton.flagmega.codegen.triton.microkernels import (
    PagedAttentionSplitMicroKernelProvider,
    default_triton_microkernel_registry,
)


def test_split_attention_microkernel_provider_is_target_neutral_and_registered():
    provider = PagedAttentionSplitMicroKernelProvider()
    registry = default_triton_microkernel_registry()

    assert provider.op_names == frozenset({
        "ntt.paged_attention_partial",
        "ntt.paged_attention_combine",
        "ntt.paged_attention_gated_combine",
    })
    assert isinstance(
        registry.provider_for("ntt.paged_attention_partial"),
        PagedAttentionSplitMicroKernelProvider,
    )
    assert isinstance(
        registry.provider_for("ntt.paged_attention_combine"),
        PagedAttentionSplitMicroKernelProvider,
    )


@pytest.mark.parametrize("lanes", [1, 8])
@pytest.mark.parametrize("capacity,tile", [(16, 32), (32, 32), (64, 64), (128, 128), (256, 128)])
def test_gated_tile_follows_fixed_scalar_capacity_without_changing_sbp(capacity, tile, lanes):
    from python.test.flagmega.passes.tir.test_attention_gate_fusion import graph
    from triton.flagmega.compiler import Compiler
    from triton.flagmega.passes.tir.fuse_attention_gate import fuse_attention_gate
    from triton.flagmega.codegen.triton.microkernels.core import TIRMicroKernelContext
    compiler = Compiler()
    original = fuse_attention_gate(graph(distributed=True, lanes=lanes, dimension=capacity))
    module = compiler.compile(original, stop_after="plan-tir-alignments").module
    function = next(function for function in module.kernel_definitions
                    if function.dispatch.semantic_op == "ntt.paged_attention_gated_combine")
    context = TIRMicroKernelContext(module, function, function.dispatch, compiler.target.triton_implementation_model)
    provider = PagedAttentionSplitMicroKernelProvider()
    proposal = provider.propose(context)
    selected = next(candidate for candidate in proposal.candidates if candidate.id == proposal.default_candidate)
    assert selected.parameters["elements_per_program"] == tile
    assert function.parameter_map[function.dispatch.outputs[0]].type == original.node_map["result"].type
    assert {candidate.parameters["elements_per_program"] for candidate in proposal.candidates} == {32, 64, 128}

    # A target-specific preferred implementation is not overridden by this
    # portable masked-lane heuristic.
    model = context.implementation_model
    entries = tuple(replace(entry, facts={**entry.facts, "portable_triton": False})
                    if entry.id == "tir.paged_attention_gated_combine.decode" else entry for entry in model.implementations)
    custom = replace(context, implementation_model=replace(model, implementations=entries))
    assert provider.propose(custom).default_candidate == "tir.paged_attention_gated_combine.decode"
