# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT

import pytest

from triton.flagmega import ir as fm
from triton.flagmega.errors import IRSchemaError


def _dispatch(*candidates):
    return fm.KernelDispatch(
        semantic_op="math.add",
        semantic_candidate="semantic.math.add",
        arguments=("lhs", "rhs"),
        outputs=("result",),
        inplace_alias_candidates=tuple(candidates),
        reads=("lhs", "rhs"),
        writes=("result",),
    )


def test_named_alias_candidate_round_trips_as_typed_tir():
    candidate = fm.InplaceAliasCandidate(output="result", input="lhs")
    dispatch = _dispatch(candidate)

    assert fm.tir_from_data(dispatch.to_data()) == dispatch


@pytest.mark.parametrize(
    "candidate, message",
    (
        (fm.InplaceAliasCandidate(output="unknown", input="lhs"), "outputs"),
        (fm.InplaceAliasCandidate(output="result", input="unknown"), "inputs"),
    ),
)
def test_alias_candidate_must_reference_the_dispatch_abi(candidate, message):
    with pytest.raises(IRSchemaError, match=message):
        _dispatch(candidate)


def test_duplicate_alias_candidate_is_rejected():
    candidate = fm.InplaceAliasCandidate(output="result", input="lhs")

    with pytest.raises(IRSchemaError, match="must be unique"):
        _dispatch(candidate, candidate)


def test_alias_candidate_requires_both_names():
    with pytest.raises(IRSchemaError, match="requires named output and input"):
        fm.InplaceAliasCandidate(output="result", input="")
