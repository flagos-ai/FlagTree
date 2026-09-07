# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT

from importlib import import_module

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
    try:
        for _ in range(32):
            expression = fm.dim("shard_coord_0") * 32 + fm.dim("local_coord_0")
            assert str(expression) == "(local_coord_0 + (shard_coord_0 * 32))"

        info = dim_expr_module._simplify_sympy.cache_info()
        assert info.maxsize == 4096
        assert info.hits > 0
        # There are only two structural intermediate expressions regardless of
        # how many call sites request the mapping.
        assert len(conversions) == 2
    finally:
        dim_expr_module._simplify_sympy.cache_clear()
