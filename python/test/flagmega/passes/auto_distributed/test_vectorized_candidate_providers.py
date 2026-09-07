# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT

from triton.flagmega import ir as fm
from triton.flagmega.passes.auto_distributed import DistributedCandidateContext
from triton.flagmega.passes.auto_distributed.inference_providers import (
    TypeInferenceCandidateProvider,
)
from triton.flagmega.passes.auto_distributed.providers import BinaryCandidateProvider


PLACEMENT = fm.Placement((2, 2), "yx", "bb")


def _var(name: str, value_type: fm.IRType) -> fm.Node:
    return fm.Node(name, "builtin.var", (), value_type, attrs={"name": name})


def _module(
    op: str,
    inputs: tuple[fm.Node, ...],
    output_type: fm.IRType,
    *,
    attrs: dict[str, object],
) -> tuple[fm.IRModule, fm.Node]:
    output = fm.Node(
        "output",
        op,
        tuple(value.id for value in inputs),
        output_type,
        attrs=attrs,
    )
    return (
        fm.IRModule(
            "ntt",
            "packed",
            (*inputs, output),
            (fm.Function("main", tuple(value.id for value in inputs), (output.id,)),),
            "main",
        ),
        output,
    )


def test_vectorized_binary_enumerates_exact_local_shard_candidates():
    vector = fm.tensor_type(fm.vector_type("bfloat16", (8,)), (1, 8))
    lhs = _var("lhs", vector)
    rhs = _var("rhs", vector)
    module, output = _module(
        "math.vectorized_binary",
        (lhs, rhs),
        vector,
        attrs={"binary_op": "add"},
    )

    candidates = BinaryCandidateProvider().get_candidates(
        DistributedCandidateContext(
            module,
            output,
            PLACEMENT,
            ((vector,), (vector,)),
        )
    )
    split = {
        candidate.return_type.axis_policies[-1].hierarchy_axes: candidate
        for candidate in candidates
        if candidate.reason == "binary-exact-output-sbp"
    }

    assert set(split) == {(0,), (1,), (0, 1)}
    assert all(
        candidate.input_types == (
            candidate.return_type,
            candidate.return_type,
        )
        for candidate in split.values()
    )
    assert split[(0, 1)].operation_cost == 2


def test_vectorized_unary_preserves_the_selected_input_distribution():
    vector = fm.tensor_type(fm.vector_type("bfloat16", (8,)), (1, 8))
    value = _var("value", vector)
    module, output = _module(
        "math.vectorized_unary",
        (value,),
        vector,
        attrs={"unary_op": "silu"},
    )
    split = fm.DistributedType(
        vector,
        (fm.SBP.broadcast(), fm.SBP.split_contiguous((0,), 1)),
        PLACEMENT,
    )
    provider = TypeInferenceCandidateProvider(frozenset({output.op}))

    candidates = provider.get_candidates(
        DistributedCandidateContext(module, output, PLACEMENT, ((split,),))
    )

    assert any(
        candidate.return_type == split and candidate.input_types == (split,)
        for candidate in candidates
    )


def test_vectorized_cast_scales_the_selected_split_in_physical_lane_units():
    source = fm.tensor_type(fm.vector_type("bfloat16", (8,)), (1, 8))
    result = fm.tensor_type(fm.vector_type("float32", (4,)), (1, 16))
    value = _var("value", source)
    module, output = _module(
        "ntt.vectorized_cast",
        (value,),
        result,
        attrs={
            "new_type": {
                "kind": "vector",
                "elem_type": "float32",
                "lanes": (4,),
            },
            "vectorize_axes": (1,),
        },
    )
    split = fm.DistributedType(
        source,
        (fm.SBP.broadcast(), fm.SBP.split_block_cyclic((0,), 2)),
        PLACEMENT,
    )
    expected = fm.DistributedType(
        result,
        (fm.SBP.broadcast(), fm.SBP.split_block_cyclic((0,), 4)),
        PLACEMENT,
    )
    provider = TypeInferenceCandidateProvider(frozenset({output.op}))

    candidates = provider.get_candidates(
        DistributedCandidateContext(module, output, PLACEMENT, ((split,),))
    )

    assert any(
        candidate.return_type == expected and candidate.input_types == (split,)
        for candidate in candidates
    )
