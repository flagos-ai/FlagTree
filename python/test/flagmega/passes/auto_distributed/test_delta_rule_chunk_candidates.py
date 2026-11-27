# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Long prefill's CP operations retain coupled grouped-head ownership."""

import pytest

from triton.flagmega import ir as fm
from triton.flagmega.passes.auto_distributed.candidates import (
    DistributedCandidateContext, DistributedCandidateProviderRegistry,
)
from triton.flagmega.passes.auto_distributed.policy import NttDistributionPolicy


@pytest.mark.parametrize("op,kind", [
    ("nn.delta_rule_chunk_transfer", "transfer"),
    ("nn.delta_rule_chunk_transfer", "state"),
    ("nn.delta_rule_chunk_fixup", None),
    ("nn.delta_rule_chunked_block_update", None),
])
def test_registered_chunk_provider_preserves_grouped_head_layout(op, kind):
    mesh = fm.Placement((2, 2), "yx", "bb")
    b = fm.SBP.broadcast()
    key_head = fm.SBP.split_contiguous((0,), 1)
    value_head = fm.SBP.split_contiguous((0,), 2)
    key = fm.tensor_type("bfloat16", (1024, 2, 128))
    value = fm.tensor_type("bfloat16", (1024, 4, 128))
    coeff = fm.tensor_type("bfloat16", (16, 4, 64, 64))
    prefix = fm.tensor_type("float32", (16, 4, 64))
    matrix = fm.tensor_type("float32", (2, 4, 128, 128))
    state = fm.RefType("State", (("matrix", fm.tensor_type("float32", (4, 128, 128))),))
    tensors = {
        "key": fm.DistributedType(key, (b, key_head, b), mesh),
        "value": fm.DistributedType(value, (b, value_head, b), mesh),
        "coeff": fm.DistributedType(coeff, (b, value_head, b, b), mesh),
        "prefix": fm.DistributedType(prefix, (b, value_head, b), mesh),
        "matrix": fm.DistributedType(matrix, (b, value_head, b, b), mesh),
        "state": state,
    }
    if op == "nn.delta_rule_chunk_transfer":
        names, attrs = ("key", "value", "coeff", "prefix"), {"matrix_kind": kind}
        expected = tensors["matrix"]
    elif op == "nn.delta_rule_chunk_fixup":
        names, attrs = ("matrix", "matrix", "state"), {}
        expected = tensors["matrix"]
    else:
        names, attrs = ("key", "key", "value", "coeff", "prefix", "state", "matrix"), {}
        expected = fm.TupleType((tensors["value"], state))
    distributed = tuple(tensors[name] for name in names)
    inputs = tuple(fm.Node(f"arg{i}", "builtin.var", (), fm.logical_type(t), attrs={"name": f"arg{i}"})
                   for i, t in enumerate(distributed))
    definition = fm.get_definition(op)
    attrs = definition.normalize_attrs(attrs)
    output = fm.Node("output", op, tuple(n.id for n in inputs),
                     definition.infer_type(inputs, attrs), attrs=attrs,
                     effect=definition.infer_effect(inputs, attrs))
    module = fm.IRModule("high_level", "packed", (*inputs, output),
                         (fm.Function("main", tuple(n.id for n in inputs), (output.id,)),), "main")
    registry = DistributedCandidateProviderRegistry()
    NttDistributionPolicy((mesh,)).register_candidate_providers(registry)
    provider = registry.try_get(op)
    assert provider is not None, f"No reviewed distribution provider for {op}"
    candidates = provider.get_candidates(DistributedCandidateContext(
        module, output, mesh, tuple((t,) for t in distributed)))
    assert any(c.input_types == distributed and c.return_type == expected for c in candidates)
    # Chunk/token and state-matrix feature axes remain complete for composition.
    for candidate in candidates:
        for t in candidate.input_types:
            if isinstance(t, fm.DistributedType):
                assert t.partial is None
                assert all(p == b for i, p in enumerate(t.axis_policies) if i != 1)
