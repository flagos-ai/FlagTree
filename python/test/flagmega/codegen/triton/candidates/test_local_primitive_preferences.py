# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT

from dataclasses import replace

import pytest

from triton.flagmega import ir as fm
from triton.flagmega.codegen.triton.candidates import ReductionCandidateProvider, TensorTransformCandidateProvider
from triton.flagmega.codegen.triton.candidates.core import TritonCandidateContext
from triton.flagmega.codegen.triton.implementation import TritonImplementation, TritonImplementationModel
from python.test.flagmega.ir.ops.primitive_helpers import primitive_module


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
