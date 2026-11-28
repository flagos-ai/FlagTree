# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT

from triton.flagmega import ir as fm
from triton.flagmega.codegen.triton.candidates import (
    TritonCandidateProposal,
    TritonCandidateProviderRegistry,
)
from triton.flagmega.codegen.triton.selection import TritonTirSelectionPolicy
from triton.flagmega.targets import NvidiaSm90Target


def test_injected_provider_extends_selector_without_editing_central_policy():

    class UnitProvider:
        op_names = frozenset({"unit.identity"})

        def propose(self, node, context):
            del node, context
            return TritonCandidateProposal((fm.Candidate(
                "tir.unit.identity",
                {"family": "unit", "variant": "identity"},
                {"portable_triton": True},
            ), ), "tir.unit.identity")

    registry = TritonCandidateProviderRegistry()
    registry.add(UnitProvider())
    policy = TritonTirSelectionPolicy(lambda _node, values, _module, **_kwargs: values, registry)
    target = NvidiaSm90Target(tir_selection_policy=policy)
    builder = fm.IRBuilder(dialect="high_level", stage="frozen_constants")
    value_type = fm.tensor_type("float32", (16, ))
    value = builder.var("value", value_type, id="value")
    result = builder.call("unit.identity", (value, ), value_type, id="result")
    builder.function("main", (value, ), (result, ))

    proposed = target.propose_tir(builder.build(entry="main"))

    point = next(point for point in proposed.selection_points if point.id == "tir.result")
    assert point.default_candidate == "tir.unit.identity"
    assert proposed.selection_map[point.id].candidate_id == "tir.unit.identity"
