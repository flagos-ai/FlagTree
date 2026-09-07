# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT

import inspect

from triton.flagmega.codegen.triton.lowering import TritonTirLoweringPolicy
from triton.flagmega.codegen.triton.selection import TritonTirSelectionPolicy
from triton.flagmega.codegen.triton.microkernels import (
    TritonMicroKernelSelectionPolicy,
)
from triton.flagmega.passes.auto_distributed.policy import NttDistributionPolicy
from triton.flagmega.passes.tir.bufferize import NttBufferizationPolicy
from triton.flagmega.rules.ntt.packing import NttPackingPolicy
from triton.flagmega.rules.ntt.vectorize.policy import NttVectorizationPolicy
from triton.flagmega.targets import (
    NttTarget,
    NttTargetOptions,
    NvidiaSm90Machine,
    NvidiaSm90Target,
    PyNttTarget,
)
from triton.flagmega.targets.selection import CapabilitySelectionPolicy


def test_pyntt_backend_is_a_policy_facade_over_a_machine():
    target = NvidiaSm90Target()

    assert isinstance(target, NttTarget)
    assert isinstance(target, PyNttTarget)
    assert isinstance(target.machine, NvidiaSm90Machine)
    assert isinstance(target.vectorization_policy, NttVectorizationPolicy)
    assert isinstance(target.distribution_policy, NttDistributionPolicy)
    assert type(target.packing_policy) is NttPackingPolicy
    assert isinstance(target.selection_policy, CapabilitySelectionPolicy)
    assert isinstance(target.tir_selection_policy, TritonTirSelectionPolicy)
    assert isinstance(target.tir_lowering_policy, TritonTirLoweringPolicy)
    assert isinstance(
        target.microkernel_selection_policy,
        TritonMicroKernelSelectionPolicy,
    )
    assert isinstance(target.bufferization_policy, NttBufferizationPolicy)
    source = inspect.getsource(PyNttTarget)
    assert "Candidate(" not in source
    assert "RewriteRule(" not in source
    assert "tir.kernel" not in source


def test_ntt_base_owns_generic_stage_orchestration():
    target_source = inspect.getsource(NvidiaSm90Target)
    base_source = inspect.getsource(NttTarget)

    assert "register_auto_vectorize_rules" not in target_source
    assert "register_auto_distributed_candidate_providers" not in target_source
    assert "propose_packing" not in target_source
    assert "propose_tir" not in target_source
    assert "register_auto_vectorize_rules" in base_source
    assert "register_auto_distributed_candidate_providers" in base_source
    assert "propose_packing" in base_source
    assert "propose_tir" in base_source
    assert "propose_microkernels" in base_source
    assert "select_microkernels" in base_source
    assert "NttPackingPolicy" not in base_source
    assert "packing_qkv_block_k" not in base_source
    assert "nvidia" not in base_source.lower()
    assert "sm90" not in base_source.lower()


def test_machine_profile_does_not_register_semantic_rules():
    source = inspect.getsource(NvidiaSm90Machine)

    for spelling in (
        "NttPackingPolicy",
        "NttVectorizationPolicy",
        "NttDistributionPolicy",
        "RewriteRule(",
        "Candidate(",
        "register_auto_",
    ):
        assert spelling not in source


def test_sm90_target_delegates_rule_registration_to_injected_ntt_policy():
    class VectorPolicy:
        def register_rules(self, registry):
            registry.append("rules")

        def register_propagation_rules(self, registry):
            registry.append("propagation")

    target = NvidiaSm90Target(vectorization_policy=VectorPolicy())
    registry = []
    target.register_auto_vectorize_rules(registry)
    target.register_pack_propagation_rules(registry)
    assert registry == ["rules", "propagation"]


def test_sm90_namespace_does_not_own_graph_packing_rules():
    target = NvidiaSm90Target()

    assert type(target.vectorization_policy).__module__.startswith(
        "triton.flagmega.rules.ntt"
    )
    assert type(target.packing_policy).__module__.startswith(
        "triton.flagmega.rules.ntt"
    )
    assert type(target.distribution_policy).__module__.startswith(
        "triton.flagmega.passes.auto_distributed"
    )
    assert type(target.tir_selection_policy).__module__.startswith(
        "triton.flagmega.codegen.triton"
    )


def test_sm90_target_registers_parameterized_generic_packing_rules():
    target = NvidiaSm90Target()
    backend_source = inspect.getsource(PyNttTarget)
    machine_source = inspect.getsource(NvidiaSm90Machine)

    assert type(target.packing_policy) is NttPackingPolicy
    assert "NttPackingPolicy(" in backend_source
    assert "NttPackingPolicy(" not in machine_source
    assert "Candidate(" not in backend_source
    assert "qkv_split_k" not in backend_source

    target_source = inspect.getsource(NvidiaSm90Target)
    assert "NvidiaSm90PackingPolicy" not in target_source
    assert "targets.nvidia.packing" not in target_source


def test_generic_ntt_packing_rule_does_not_own_qkv_machine_geometry():
    """Match nncase: K-major packing owns lanes, not micro-kernel tiles/mesh."""

    source = inspect.getsource(NttPackingPolicy)

    for spelling in (
        "qkv_block_k",
        "qkv_local_n_alignment",
        "distributed_placements",
        "context_mesh_size",
        "head_mesh_size",
        "mesh_y",
        "mesh_x",
    ):
        assert spelling not in source


def test_generic_ntt_options_do_not_serialize_sm90_qkv_asset_geometry():
    options_source = inspect.getsource(NttTargetOptions)
    snapshot = NvidiaSm90Target().options.to_data()

    assert "packing_qkv_block_k" not in options_source
    assert "packing_qkv_local_n_alignment" not in options_source
    assert "packing_qkv_block_k" not in snapshot
    assert "packing_qkv_local_n_alignment" not in snapshot


def test_generic_distribution_policy_does_not_own_machine_block_thresholds():
    source = inspect.getsource(NttDistributionPolicy)
    options_source = inspect.getsource(NttTargetOptions)

    assert "block_cyclic_output_n" not in source
    assert "distributed_block_cyclic_output_n" not in options_source


def test_sm90_does_not_override_graph_selection_or_split_candidates():
    source = inspect.getsource(NvidiaSm90Machine)

    assert "NvidiaSm90SelectionPolicy" not in source
    assert "NvidiaSm90DistributedSplitCandidateProvider" not in source
    assert "distributed_split_candidate_provider" not in source
    assert type(NvidiaSm90Target().selection_policy) is CapabilitySelectionPolicy
