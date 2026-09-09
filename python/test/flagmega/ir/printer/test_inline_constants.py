# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT

import pytest

from triton.flagmega import ir as fm
from triton.flagmega.ir.printer import il_source, script_source
from triton.flagmega.passes import freeze_constant_islands
from .helpers import arithmetic_module


@pytest.mark.parametrize("render,prefix", ((il_source, ""), (script_source, "T.")))
def test_weights_and_splats_are_typed_inline_operands(render, prefix):
    module = arithmetic_module()
    source = render(module)
    assert f"Math.Add(%x, {prefix}WeightRef(f32[4], 'weight', Source: '/missing/checkpoint', Key: 'tensor.key'))" in source
    constant = "const" if not prefix else "T.Const"
    assert f"Math.Mul(%0, {constant}(f32[4] : splat(2.0)))" in source
    assert "%long_weight_id" not in source and "%long_constant_id" not in source
    assert sum(" = " in line for line in source.splitlines() if line.startswith("  %")) == 2


@pytest.mark.parametrize("render,prefix", ((il_source, "const"), (script_source, "T.Const")))
def test_returned_scalar_constants_inline_without_a_dangling_ssa(render, prefix):
    builder = fm.IRBuilder(dialect="high_level", stage="imported")
    values = [
        builder.node(op="builtin.scalar_const", type=fm.tensor_type(dtype, ()), attrs={"value": value}, id=name)
        for dtype, value, name in (("bool", True, "predicate"), ("int64", 3, "position"))
    ]
    builder.function("main", (), values)
    module = fm.verify_module(builder.build(entry="main"))
    source = render(module)
    assert f"{prefix}(bool[] : True), {prefix}(i64[] : 3)" in source
    assert "%predicate" not in source and "%position" not in source


@pytest.mark.parametrize("render", (il_source, script_source))
def test_constant_recipe_uses_the_same_inline_and_short_name_policy(render):
    builder = fm.IRBuilder(dialect="high_level", stage="canonical_constants")
    tensor = fm.tensor_type("float32", (4, ))
    weight = builder.weight("w", tensor, source="missing", key="w", id="weight")
    activated = builder.call("math.silu", (weight, ), tensor, id="activated_weight")
    builder.function("main", (), (activated, ))
    module = freeze_constant_islands(builder.build(entry="main"))
    source = render(module)
    assert "ConstAssetRef(f32[4], Recipe:" in source
    assert "Math.Silu(" in source and "WeightRef(f32[4], 'w', Source: 'missing')" in source
    assert "yield (%0)" in source
    assert "name='activated_weight'" in source
    assert "%activated_weight =" not in source and "%weight =" not in source


def test_readable_only_inlining_keeps_python_checkpoint_identity(tmp_path):
    module = arithmetic_module()
    before = fm.module_source(module)
    path = fm.emit_module(module, tmp_path / "module.py")
    assert path.read_text() == before
    assert "self.weight(" in before and "F.builtin.splat_const(" in before
    assert fm.load_module(path) == module
    assert "name='very_long_original_output_name'" in path.with_suffix(".il").read_text()


@pytest.mark.parametrize("render", (il_source, script_source))
def test_large_vector_splat_is_compact_and_weight_source_hash_is_preserved(render):
    builder = fm.IRBuilder(dialect="high_level", stage="imported")
    tensor = fm.tensor_type(fm.vector_type("bfloat16", (8, )), (1000000000, ))
    weight = builder.weight("weight", tensor, source="missing", key="weight", source_hash="abcd", id="w")
    constant = builder.node(op="builtin.splat_const", type=tensor, attrs={"value": 0.0}, id="zero")
    builder.function("main", (), (weight, constant))
    source = render(fm.verify_module(builder.build(entry="main")))
    assert "bf16<8>[1000000000]" in source and "splat(0.0)" in source
    assert "SourceHash: 'abcd'" in source and len(source) < 1000
    assert "%zero" not in source and "%w " not in source


def test_tir_readonly_buffers_and_scalar_literals_inline_but_not_mutable_buffers():
    builder = fm.IRBuilder(dialect="semantic_tir", stage="unit")
    tensor = fm.tensor_type("float32", ())
    readonly = builder.node(op="tir.buffer", type=tensor, id="readonly", attrs={"storage": "rdata", "key": "w"})
    mutable = builder.node(op="tir.buffer", type=tensor, id="scratch", attrs={"storage": "workspace"})
    literal = builder.node(op="tir.scalar_const", type=tensor, id="literal", attrs={"value": 1.0})
    builder.function("main", (), (readonly, mutable, literal))
    source = script_source(builder.build(entry="main"))
    assert "  %0 = T.Buffer(f32[], Name: 'scratch'" in source
    returned = next(line for line in source.splitlines() if "T.Return(" in line)
    assert "Name: 'readonly'" in returned and "%0, T.Const(f32[] : 1.0)" in returned
    assert "%readonly" not in source and "%literal" not in source
