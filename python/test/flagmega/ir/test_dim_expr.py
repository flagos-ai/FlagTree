# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT

import pytest
from importlib import import_module

from triton.flagmega.errors import IRSchemaError

from triton.flagmega import ir as fm


def test_dynamic_calculation_simplification_and_python_round_trip(tmp_path):
    n = fm.dim("n", minimum=1, maximum=4096)
    expression = (n * 8 + 16) // 8

    assert expression.equivalent(n + 2)
    assert expression.evaluate({"n": 30}) == 32
    assert expression.minimum == 3
    assert expression.maximum == 4098
    assert fm.try_div_exactly(n * 32, 32) == n
    assert fm.dim_positive(-1, n).equivalent(n - 1)
    assert fm.dim_select(n, 0, 7, 9, fm.DimCompareOp.GREATER_THAN) == fm.dim(7)
    assert fm.dim_select(n, 8192, 7, 9, "greater_or_equal") == fm.dim(9)
    assert fm.unknown_dim().is_unknown
    assert fm.Dimension.from_data(expression.to_data()) == expression

    builder = fm.IRBuilder(dialect="high_level", stage="imported")
    value_type = fm.tensor_type("bfloat16", [n, expression])
    source = builder.var("source", value_type, id="source")
    builder.function("main", [source], [source])
    module = builder.build(entry="main")
    path = fm.emit_module(module, tmp_path / "dynamic.py")

    assert fm.load_module(path) == module
    assert "fm.dim_expr(" in path.read_text(encoding="utf-8")


def test_repeated_structural_expression_reuses_bounded_sympy_simplification(
    monkeypatch,
):
    dim_expr_module = import_module("triton.flagmega.ir.dim_expr")
    original_to_sympy = dim_expr_module._to_sympy
    conversions = []

    def observe(value):
        conversions.append(value)
        return original_to_sympy(value)

    monkeypatch.setattr(dim_expr_module, "_to_sympy", observe)
    dim_expr_module._simplify_sympy.cache_clear()
    dim_expr_module._dim_expr_cached.cache_clear()
    try:
        for _ in range(32):
            expression = fm.dim("shard_coord_0") * 32 + fm.dim("local_coord_0")
            assert str(expression) == "(local_coord_0 + (shard_coord_0 * 32))"

        assert dim_expr_module._simplify_sympy.cache_info().maxsize == 4096
        # `dim_expr` caches the whole validate/simplify step, so a repeated
        # expression is answered before `_simplify_sympy` is consulted again;
        # the property under test is that the repeats are served from *some*
        # cache rather than re-converted.
        outer = dim_expr_module._dim_expr_cached.cache_info()
        assert outer.hits > 0
        # There are only two structural intermediate expressions regardless of
        # how many call sites request the mapping.
        assert len(conversions) == 2
    finally:
        dim_expr_module._simplify_sympy.cache_clear()
        dim_expr_module._dim_expr_cached.cache_clear()


def test_dim_expr_caches_validation_and_structural_simplification():
    """A repeated expression must not re-validate or re-simplify.

    Distributed-type analyses re-derive the same owner coordinate arithmetic for
    every candidate: on the 45-layer GLM-5.3 graph `dim_expr` is called
    1,547,681 times for 4,960 distinct expressions. `_simplify_sympy` only
    caches the sympy step, so the repeats still paid for `_validate_expression`
    and `_simplify_local` on every call.
    """

    dim_expr_module = import_module("triton.flagmega.ir.dim_expr")
    dim_expr_module._dim_expr_cached.cache_clear()
    try:
        first = fm.dim_expr("add", fm.dim("a", 0, 7), fm.dim(3))
        before = dim_expr_module._dim_expr_cached.cache_info()
        repeats = tuple(
            fm.dim_expr("add", fm.dim("a", 0, 7), fm.dim(3)) for _ in range(16)
        )
        after = dim_expr_module._dim_expr_cached.cache_info()

        assert all(value == first for value in repeats)
        assert after.hits - before.hits == 16
        assert after.misses == before.misses, "a repeat recomputed the expression"
    finally:
        dim_expr_module._dim_expr_cached.cache_clear()


def test_dim_expr_cache_still_rejects_invalid_expressions():
    """Caching must not swallow validation.

    The cache wraps `_validate_expression`, so a malformed expression has to
    keep raising -- including when the same bad call is repeated and could be
    served from a cache entry.
    """

    dim_expr_module = import_module("triton.flagmega.ir.dim_expr")
    dim_expr_module._dim_expr_cached.cache_clear()
    try:
        # `add` legitimately accepts a single operand, so use an arity error
        # (`mod` needs two) and an unknown operator.
        for _ in range(3):
            with pytest.raises(IRSchemaError):
                fm.dim_expr("mod", fm.dim(5))
            with pytest.raises(IRSchemaError):
                fm.dim_expr("not_an_op", fm.dim(1), fm.dim(2))
    finally:
        dim_expr_module._dim_expr_cached.cache_clear()


def test_subview_rejects_a_larger_target_capacity_without_the_interval_proof():
    """A strictly larger target shard is non-containment by capacity alone.

    `is_local_shard_subview` ends in a symbolic interval proof that runs a
    simplification per tensor axis. Containment requires the target shard to
    fit inside the source shard, so a larger target capacity already decides
    the question: on the 45-layer GLM-5.3 graph that answers 25,147 of 39,827
    calls, all negative, and never rejects a containment the proof accepts.
    """

    from triton.flagmega.ir.distributed_type import is_local_shard_subview
    from triton.flagmega.ir.local_shard import local_shard_descriptor

    placement = fm.Placement((4, 8), "cy", "cb")
    tensor = fm.tensor_type("bfloat16", (512, 256))
    broadcast = fm.SBP.broadcast()
    # A split over both axes gives each owner 1/32 of the rows; a split over
    # one axis gives 1/4. The coarser target cannot fit inside the finer source.
    fine = fm.DistributedType(
        tensor, (fm.SBP.split_contiguous((0, 1)), broadcast), placement
    )
    coarse = fm.DistributedType(
        tensor, (fm.SBP.split_contiguous((0, )), broadcast), placement
    )

    coords = tuple(
        fm.dim(f"_owner_coord_{axis}", 0, extent - 1)
        for axis, extent in enumerate(placement.hierarchy)
    )
    fine_capacity = local_shard_descriptor(fine, coords).local_capacity_shape
    coarse_capacity = local_shard_descriptor(coarse, coords).local_capacity_shape
    assert coarse_capacity[0].fixed_value > fine_capacity[0].fixed_value, (
        "the fixture must present a strictly larger target capacity"
    )

    assert is_local_shard_subview(fine, coarse) is False
    # The reverse direction is a genuine containment and must still be proved.
    assert is_local_shard_subview(coarse, fine) is True


def test_frozen_mapping_get_does_not_use_the_mapping_abc_fallback():
    """Attribute probes must not pay the Mapping ABC's Python-level `get`.

    Every rewrite rule tests every node and each of its inputs for fused ops,
    so `attrs.get` runs 41.3 million times during the pre-distribution passes
    on the 45-layer GLM-5.3 graph. `Mapping.get` is a `try`/`except` around
    `__getitem__`; delegating to the underlying dict keeps it in C.
    """

    from collections.abc import Mapping

    from triton.flagmega.ir.model import _FrozenMapping

    values = {"a": 1, "b": None, "c": False}
    frozen = _FrozenMapping(values)

    # The override must not be inherited from the ABC.
    assert type(frozen).get is not Mapping.get

    # ...and must keep dict semantics, including falsy and absent values.
    for key in (*values, "missing"):
        assert frozen.get(key) == values.get(key)
        assert frozen.get(key, "default") == values.get(key, "default")
        assert (key in frozen) == (key in values)
    assert dict(frozen) == values
    assert len(frozen) == len(values)
