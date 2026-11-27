# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""The normal workload target owns explicit state and reduction boundaries."""

from dataclasses import replace
from pathlib import Path

import pytest

from triton.flagmega import ir as fm
from python.test.flagmega.passes.functions.test_persistent_state_abi import state_module
from python.test.flagmega.passes.functions.test_reduction_rounding import graph


@pytest.fixture
def target(monkeypatch):
    tutorial = Path(__file__).parents[3] / "tutorials/flagmega/04-glm5.3-flash-fp8/nvidia-h800"
    monkeypatch.syspath_prepend(str(tutorial))
    from local_optimizations.native_tp4 import NativeTp4DistributionTarget, native_implementation_model

    return NativeTp4DistributionTarget(triton_implementation_model=native_implementation_model())


@pytest.mark.parametrize("lanes,mode", [((), "full"), ((8,), "full"), ((2, 4), "partial"), ((8,), "gather")])
def test_ordinary_proposal_preserves_rounding_without_standalone_conversions(target, lanes, mode):
    source = graph(lanes=lanes, mode=mode)
    proposal = target.propose_tir(source)
    assert not any(
        node.op in {"tensors.cast", "ntt.vectorized_cast", "tensors.pack", "tensors.unpack"} for node in proposal.nodes
    )
    assert proposal.node_map["output"].type == source.node_map["output"].type
    assert len(proposal.metadata["materialized_reduction_rounding"]) == 1


@pytest.mark.parametrize("lanes,mode", [((8,), "full"), ((2, 4), "partial"), ((8,), "gather")])
def test_rounding_pattern_survives_the_preceding_generic_fusion_pipeline(target, lanes, mode):
    from triton.flagmega.compiler import Compiler

    source = fm.verify_module(replace(graph(lanes=lanes, mode=mode, producer=True), stage="vector_contracts_lowered"))
    compiler = Compiler()
    compiler.target = target
    compiled = compiler.compile(source).module
    assert not any(
        definition.dispatch.semantic_op in {"tensors.cast", "ntt.vectorized_cast"}
        for definition in compiled.kernel_definitions
    )
    assert compiled.node_map["output"].type == source.node_map["output"].type
    fm.verify_buffer_plan(compiled)


@pytest.mark.parametrize("packed", [False, True])
def test_ordinary_proposal_exposes_the_declared_persistent_state_ownership(target, packed):
    source, pairs = state_module(packed=packed)
    source = fm.verify_module(replace(source, stage="frozen_constants"))
    proposal = target.propose_tir(source)
    assert not any(node.op in {"distributed.boxing", "distributed.sharded_view"} for node in proposal.nodes)
    for parameter, public in pairs:
        assert proposal.node_map[parameter].type == source.node_map[public.replace("output", "result")].type
    assert all(
        value in proposal.function_map["main"].outputs
        for state in proposal.metadata["composite_abi"]["tensor_states"]
        for value in state["output"]
    )
