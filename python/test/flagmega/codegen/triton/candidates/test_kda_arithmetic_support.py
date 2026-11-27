# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""A numerical contract must have a target implementation before selection."""

import pytest
from triton.flagmega import ir as fm
from triton.flagmega.codegen.triton.candidates.core import TritonCandidateContext
from triton.flagmega.codegen.triton.candidates.routing import KdaCandidateProvider
from triton.flagmega.errors import CodegenError
from triton.flagmega.targets import NvidiaSm90Target
from triton.flagmega.targets.iluvatar_bi_v150 import IluvatarBiV150Target


def graph(arithmetic):
    class Graph(fm.Module):
        def forward(self):
            vectors = tuple(self.input(name, fm.tensor_type("bfloat16", (1, 2, 128)))
                            for name in ("q", "k", "v", "gate"))
            rate, bias = (self.input(name, fm.tensor_type("float32", (1, 2, 128)))
                          for name in ("rate", "bias"))
            beta = self.input("beta", fm.tensor_type("bfloat16", (1, 2)))
            state = self.input("state", fm.tensor_type("float32", (1, 2, 128, 128)))
            parameters = (*vectors, rate, bias, beta, state)
            if arithmetic == "phase":
                prefill = self.input("prefill", fm.tensor_type("bool", ()))
                parameters = (*parameters, prefill)
                result = fm.F.nn.kda_gated_step(*parameters, lower_bound=-5., name="core")
            else:
                result = fm.F.nn.kda_gated_recurrent_core(*parameters, lower_bound=-5.,
                                                         arithmetic=arithmetic, name="core")
            self.function("main", parameters, fm.F.tensors.get_items(result, 0, 1))
    return Graph(dialect="high_level", stage="imported", entry="main").build()


@pytest.mark.parametrize("arithmetic", ["recurrent", "chunk_bf16", "phase"])
@pytest.mark.parametrize("target_type", [NvidiaSm90Target, IluvatarBiV150Target])
def test_kda_arithmetic_requires_an_implemented_target_contract(arithmetic, target_type):
    module = graph(arithmetic)
    target = target_type()
    context = TritonCandidateContext(module, target, {}, {}, {}, frozenset(), True,
                                     target.triton_implementation_model)
    if arithmetic in {"chunk_bf16", "phase"} and target_type is IluvatarBiV150Target:
        with pytest.raises(CodegenError, match="chunk_bf16.*PTX"):
            KdaCandidateProvider().propose(module.node_map["core"], context)
    else:
        proposal = KdaCandidateProvider().propose(module.node_map["core"], context)
        assert proposal.default_candidate == "tir.kda_gated_recurrent_core.local"
