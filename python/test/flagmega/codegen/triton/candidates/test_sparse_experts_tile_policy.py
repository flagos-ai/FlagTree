# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT

from dataclasses import replace

import pytest

from triton.flagmega.codegen.triton.candidates.core import TritonCandidateContext
from triton.flagmega.codegen.triton.candidates.sparse_experts import SparseExpertsCandidateProvider
from triton.flagmega.errors import CodegenError
from triton.flagmega.ir.ops.nn.sparse_experts_gate_up import SparseExpertsGateUp
from triton.flagmega.targets import NvidiaSm90Target
from python.test.flagmega.codegen.triton.kernels.sparse_experts.helpers import stage_module


@pytest.mark.parametrize("n,k,distributed,expected", [(4, 2048, False, (4, 2048)),
                                                    (64, 2048, False, (64, 128)),
                                                    (128, 2048, True, (64, 128)),
                                                    (20, 72, False, (4, 2048))])
@pytest.mark.parametrize("packed", [False, True])
def test_static_tile_catalog_uses_scalar_owner_local_extents(n, k, distributed, expected, packed):
    module = stage_module(SparseExpertsGateUp, hidden=k, intermediate=n, packed=packed, tokens=1,
                          distribution="output" if distributed else None)
    node = module.node_map["expert_stage"]
    target = NvidiaSm90Target()
    model = tile_model()
    context = TritonCandidateContext(module, target, {}, {}, {}, frozenset(), True, model)
    before = module.semantic_hash
    proposal = SparseExpertsCandidateProvider().propose(node, context)
    candidate = next(c for c in proposal.candidates if c.id == proposal.default_candidate)
    assert (candidate.parameters["block_n"], candidate.parameters["block_k"]) == expected
    assert module.semantic_hash == before
    for candidate in proposal.candidates:
        assert all(candidate.parameters[name] == value
                   for name, value in model.implementation(candidate.id).parameters.items())
    from triton.flagmega.compiler import Compiler
    compiler = Compiler()
    compiler.target = NvidiaSm90Target(triton_implementation_model=model)
    assert compiler.compile(module).module.stage == "bufferized_tir"


def tile_model(minimum=64):
    model = NvidiaSm90Target().triton_implementation_model
    base = next(i for i in model.implementations if i.family == "sparse_experts_gate_up")
    base = replace(base, parameters={**base.parameters, "block_n": 4, "block_k": 2048})
    wide = replace(base, id=base.id + "_n64_k128", parameters={**base.parameters, "block_n": 64, "block_k": 128},
                   contract={**base.contract, "min_local_n": minimum})
    return replace(model, implementations=(*[base if i.id == base.id else i for i in model.implementations], wide),
                   preferences={**model.preferences, "sparse_experts_gate_up": (wide.id, base.id)})


@pytest.mark.parametrize("minimum", [-1, True, "64"])
def test_invalid_local_n_constraint_is_rejected(minimum):
    module = stage_module(SparseExpertsGateUp)
    context = TritonCandidateContext(module, NvidiaSm90Target(), {}, {}, {}, frozenset(), True, tile_model(minimum))
    with pytest.raises(CodegenError, match="min_local_n"):
        SparseExpertsCandidateProvider().propose(module.node_map["expert_stage"], context)


def test_bounded_dynamic_local_capacity_is_supported():
    from triton.flagmega import ir as fm
    module = stage_module(SparseExpertsGateUp, hidden=2048, intermediate=128, tokens=1)
    node = module.node_map["expert_stage"]
    node = replace(node, type=fm.tensor_type("bfloat16", (1, 3, fm.dim("n", minimum=1, maximum=128))))
    context = TritonCandidateContext(module, NvidiaSm90Target(), {}, {}, {}, frozenset(), True, tile_model())
    proposal = SparseExpertsCandidateProvider().propose(node, context)
    assert proposal.default_candidate.endswith("_n64_k128")
