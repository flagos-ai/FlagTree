# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT

import pytest

from triton.flagmega.rules.ntt.vectorize import VectorizeBinary, VectorizeRuleRegistry
from triton.flagmega.rules.ntt.vectorize.propagation import propagation_rules
from triton.flagmega.rules.ntt.vectorize.policy import NttVectorizationPolicy


def test_registry_preserves_target_registration_order_and_exposes_immutable_views():
    registry = VectorizeRuleRegistry()
    binary = VectorizeBinary()
    propagation = propagation_rules()[0]
    registry.add(binary)
    registry.add_propagation(propagation)
    assert registry.rules == (binary,)
    assert registry.propagation_rules == (propagation,)


def test_registry_rejects_duplicate_seed_and_propagation_names():
    registry = VectorizeRuleRegistry()
    registry.add(VectorizeBinary())
    with pytest.raises(ValueError, match="already registered"):
        registry.add(VectorizeBinary())

    first = propagation_rules()[0]
    registry.add_propagation(first)
    with pytest.raises(ValueError, match="already registered"):
        registry.add_propagation(first)


def test_ntt_policy_registers_seed_and_propagation_rules_explicitly():
    registry = VectorizeRuleRegistry()
    policy = NttVectorizationPolicy(lane_bytes=16, max_axes=2)
    policy.register_rules(registry)
    policy.register_propagation_rules(registry)
    assert [rule.name for rule in registry.rules] == [
        "VectorizeMatMul",
        "VectorizeRMSNorm",
        "VectorizeNormStats",
        "VectorizeNormApply",
        "VectorizeQKVRoPEWithCache",
        "VectorizeBinary",
        "VectorizeUnary",
    ]
    assert [rule.name for rule in registry.propagation_rules] == [
        "VectorizeBinaryPropagation",
        "BinaryDevectorizeLhsPropagation",
        "BinaryDevectorizeRhsPropagation",
        "VectorizeCastPropagation",
        "CastDevectorizePropagation",
        "FoldNopVectorizedCast",
        "VectorizeUnaryPropagation",
        "UnaryDevectorizePropagation",
        "VectorizeConcatPropagation",
        "ConcatDevectorizePropagation",
        "VectorizeTransposePropagation",
        "TransposeDevectorizePropagation",
        "VectorizePadPropagation",
        "PadDevectorizePropagation",
        "VectorizeSliceToShapePropagation",
        "SliceToShapeDevectorizePropagation",
        "VectorizeSlicePropagation",
        "SliceDevectorizePropagation",
        "VectorizeBroadcastPropagation",
        "BroadcastDevectorizePropagation",
        "VectorizeReshapePropagation",
        "ReshapeDevectorizePropagation",
        "VectorizeRoPEPropagation",
        "VectorizeSparseExpertsPropagation",
        "FoldPackUnpack",
        "FoldUnpackPack",
    ]
