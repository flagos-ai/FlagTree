# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT

from dataclasses import replace

import pytest
from triton.flagmega import ir as fm
from triton.flagmega.codegen.triton.candidates import ReductionCandidateProvider, TensorTransformCandidateProvider
from triton.flagmega.codegen.triton.candidates.core import TritonCandidateContext
from triton.flagmega.codegen.triton.implementation import TritonImplementation, TritonImplementationModel

from python.test.flagmega.ir.ops.primitive_helpers import primitive_module
from python.test.flagmega.ir.ops.tensors.test_concat_local_contract import concat_module


@pytest.mark.parametrize("op,attrs,provider", [
    ("nn.softmax", {}, ReductionCandidateProvider()),
    ("tensors.broadcast_to", {"shape": (2, 3, 7)}, TensorTransformCandidateProvider()),
])
def test_local_primitive_provider_respects_target_preference_over_catalog_order(op, attrs, provider):
    module = primitive_module(fm.get_definition(op), (fm.tensor_type("float32", (3, 7)), ), **attrs)
    family = op.split(".")[1]
    local = TritonImplementation(f"tir.{family}.local", family, "local", {"elements_per_program": 128},
                                 {"indexing": "local"})
    tuned = replace(local, id=f"tir.{family}.tuned", variant="tuned", parameters={"elements_per_program": 256})
    model = TritonImplementationModel((local, tuned), {family: (tuned.id, local.id)}, name="unit-tuned")
    context = TritonCandidateContext(module, None, {}, {}, {}, frozenset(), False, model)
    proposal = provider.propose(module.node_map["output"], context)
    assert tuple(candidate.id for candidate in proposal.candidates) == (local.id, tuned.id)
    assert proposal.default_candidate == tuned.id


def test_concat_wide_tile_is_selected_only_for_an_extent_that_covers_it():
    local128 = TritonImplementation("tir.concat.local128", "concat", "local", {"elements_per_program": 128},
                                    {"indexing": "local"})
    local2048 = TritonImplementation("tir.concat.local2048", "concat", "local", {"elements_per_program": 2048},
                                     {"indexing": "local"})
    model = TritonImplementationModel(
        (local128, local2048), {"concat": (local2048.id, local128.id)}, name="concat-tiles")

    def proposal(shape):
        module = concat_module(((1, shape), (1, shape)))
        context = TritonCandidateContext(module, None, {}, {}, {}, frozenset(), False, model)
        return TensorTransformCandidateProvider().propose(module.node_map["output"], context)

    assert tuple(candidate.id for candidate in proposal(32).candidates) == (local128.id,)
    assert tuple(candidate.id for candidate in proposal(1024).candidates) == (local128.id, local2048.id)
