# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT

import torch

from triton.flagmega import ir as fm
from triton.flagmega.evaluator import DictWeightResolver, TorchEvaluator, materialize_constant_assets
from triton.flagmega.passes.constants import freeze_constant_islands
from triton.flagmega.passes.functions import lift_constant_parameter_expressions as lift


def test_tuple_of_two_different_weights_is_one_typed_parameter_and_materializes(tmp_path):
    builder = fm.IRBuilder(dialect="nn", stage="canonical_constants")
    bf16, f32 = fm.tensor_type("bfloat16", (8, )), fm.tensor_type("float32", (8, ))
    pair_type = fm.TupleType((f32, f32))
    weights = [builder.weight(f"w{i}", bf16, source="memory", key=f"w{i}", id=f"w{i}") for i in range(4)]
    a, b = (builder.var(name, bf16, id=name) for name in ("a", "b"))
    casts = tuple(
        builder.call("tensors.cast", (value, ), f32, attrs={"dtype": "float32"}, id=f"c{i}")
        for i, value in enumerate((a, b)))
    pair = builder.call("builtin.tuple", casts, pair_type, id="pair")
    builder.function("worker", (a, b), (pair, ), attrs={"reusable": True})
    calls = [
        builder.call("builtin.call", weights[2 * i:2 * i + 2], pair_type, attrs={"callee": "worker"}, id=f"call{i}")
        for i in range(2)
    ]
    builder.function("main", (), calls)
    source = fm.verify_module(builder.build(entry="main"))
    result = lift(source)
    parameter, = result.function_map["worker"].parameters
    assert result.node_map[parameter].type == pair_type
    assert result.function_map["worker"].outputs == (parameter, )
    assert fm.load_module(fm.emit_module(result, tmp_path / "tuple.py")) == result
    values = {f"w{i}": (torch.arange(8) * .07 + i).bfloat16() for i in range(4)}
    resolver = DictWeightResolver(values)
    expected = TorchEvaluator(resolver).run(source, {})
    torch.testing.assert_close(TorchEvaluator(resolver).run(result, {}), expected, rtol=0, atol=0)
    frozen = freeze_constant_islands(result)
    assets = materialize_constant_assets(frozen, resolver)
    for i in range(2):
        actual, = frozen.node_map[f"call{i}"].inputs
        torch.testing.assert_close(assets[actual], expected[i], rtol=0, atol=0)
