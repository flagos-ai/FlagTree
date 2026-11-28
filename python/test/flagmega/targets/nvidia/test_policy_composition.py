# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT

from triton.flagmega.codegen.triton.lowering import TritonTirLoweringPolicy
from triton.flagmega.codegen.triton.selection import TritonTirSelectionPolicy
from triton.flagmega.codegen.triton.microkernels import (
    TritonMicroKernelSelectionPolicy, )
from triton.flagmega.passes.auto_distributed.policy import NttDistributionPolicy
from triton.flagmega.passes.tir.bufferize import NttBufferizationPolicy
from triton.flagmega.rules.ntt.packing import NttPackingPolicy
from triton.flagmega.rules.ntt.vectorize.policy import NttVectorizationPolicy
from triton.flagmega.targets import (
    NttTarget,
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


def test_generic_ntt_options_do_not_serialize_sm90_qkv_asset_geometry():
    snapshot = NvidiaSm90Target().options.to_data()

    assert "packing_qkv_block_k" not in snapshot
    assert "packing_qkv_local_n_alignment" not in snapshot
