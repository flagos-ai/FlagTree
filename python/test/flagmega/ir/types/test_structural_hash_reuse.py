# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Repeated distribution memo lookups must not rehash immutable type trees."""

from dataclasses import replace

from triton.flagmega import ir as fm
from triton.flagmega.passes.auto_distributed.realization import (
    DistributedReshardRealizationContext,
    DistributedReshardSourceKind,
    DistributedReshardUsageKind,
)


def test_distribution_context_reuses_nested_type_hashes(monkeypatch):
    tensor = fm.tensor_type("float32", (128,))
    placement = fm.Placement((8,), "x", "b")
    source = fm.DistributedType(tensor, (fm.SBP.broadcast(),), placement)
    target = fm.DistributedType(tensor, (fm.SBP.split_contiguous((0,), 16),), placement)
    context = DistributedReshardRealizationContext(
        source, target, DistributedReshardSourceKind.INTERNAL, DistributedReshardUsageKind.INTERNAL)
    calls = []
    original = fm.TensorType.__hash__

    def count(value):
        if value is tensor:
            calls.append(value)
        return original(value)

    monkeypatch.setattr(fm.TensorType, "__hash__", count)
    memo = {context: "proof"}
    for _ in range(100):
        assert memo[context] == "proof"
    assert len(calls) <= 2


def test_cached_hash_preserves_structural_equality_and_edited_type_keys():
    tensor = fm.tensor_type("float32", (128,))
    placement = fm.Placement((8,), "x", "b")
    source = fm.DistributedType(tensor, (fm.SBP.broadcast(),), placement)
    values = fm.TupleType((source, tensor))
    serialized = values.to_data()
    assert hash(tensor) == hash((tensor.dtype, tensor.shape, tensor.layout))
    assert hash(source) == hash((source.tensor, source.axis_policies, source.placement, source.partial, source.exclusive))
    assert hash(values) == hash((values.fields, values.is_variadic))
    edited = replace(source, axis_policies=(fm.SBP.split_contiguous((0,), 16),))
    memo = {source: "broadcast", edited: "split"}
    assert memo[replace(source)] == "broadcast"
    assert memo[replace(edited)] == "split"
    assert values.to_data() == serialized
