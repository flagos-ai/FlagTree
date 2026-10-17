# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Selection must preserve the requested projection arithmetic and platform."""

import pytest
from triton.flagmega.codegen.triton.candidates.core import TritonCandidateContext
from triton.flagmega.codegen.triton.candidates.routing import MhcCandidateProvider
from triton.flagmega.errors import CodegenError
from triton.flagmega.targets import NvidiaSm90Target
from triton.flagmega.targets.iluvatar_bi_v150 import IluvatarBiV150Target

from python.test.flagmega.ir.ops.nn.test_mhc_ordered_projection import module


@pytest.mark.parametrize("reduction", ["parallel", "stream_fma_128"])
@pytest.mark.parametrize("target_type", [NvidiaSm90Target, IluvatarBiV150Target])
def test_projection_selection_is_explicit_and_does_not_change_other_backends(reduction, target_type):
    graph = module(reduction=reduction, splits=1)
    node, = [value for value in graph.nodes if value.op == "nn.mhc_pre"]
    target = target_type()
    context = TritonCandidateContext(graph, target, {}, {}, {}, frozenset(), True,
                                     target.triton_implementation_model)
    if target_type is IluvatarBiV150Target and reduction == "stream_fma_128":
        with pytest.raises(CodegenError, match="CUDA 32-lane"):
            MhcCandidateProvider().propose(node, context)
    else:
        result = MhcCandidateProvider().propose(node, context)
        assert result.default_candidate == ("tir.mhc_pre.local" if reduction == "parallel"
                                             else "tir.mhc_pre.stream_fma")
