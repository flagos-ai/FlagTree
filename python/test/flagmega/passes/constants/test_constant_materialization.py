# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT

import torch
from dataclasses import replace

from triton.flagmega import ir as fm
from triton.flagmega.evaluator import (
    DictWeightResolver,
    TorchEvaluator,
    iter_numpy_materialized_constant_assets,
    materialize_constant_assets,
)
from triton.flagmega.importer import MemoryCheckpoint, TensorInfo
from triton.flagmega.passes import freeze_constant_islands


def _packed_constant_output():
    builder = fm.IRBuilder(dialect="high_level", stage="canonical_constants")
    logical_type = fm.tensor_type("float32", (2, 8))
    weight = builder.weight("weight", logical_type, source="memory", key="weight", id="weight")
    packed_type = fm.tensor_type(fm.vector_type("float32", (2, 2)), (2, 2))
    packed = builder.call(
        "tensors.pack",
        (weight,),
        packed_type,
        id="packed",
        attrs={"lanes": (2, 2), "axes": (1, 1)},
    )
    builder.function("main", (), (packed,))
    return freeze_constant_islands(builder.build(entry="main"))


def _numeric_constant_output():
    builder = fm.IRBuilder(dialect="high_level", stage="canonical_constants")
    value_type = fm.tensor_type("float32", (2, 4))
    lhs = builder.weight("lhs", value_type, source="memory", key="lhs", id="lhs")
    rhs = builder.weight("rhs", value_type, source="memory", key="rhs", id="rhs")
    product = builder.call("math.mul", (lhs, rhs), value_type, id="product")
    builder.function("main", (), (product,))
    return freeze_constant_islands(builder.build(entry="main"))


def test_materialization_executes_the_normal_recipe_only_when_requested():
    module = _packed_constant_output()
    weight = torch.arange(16, dtype=torch.float32).reshape(2, 8)
    resolver = DictWeightResolver({"weight": weight})

    assets = materialize_constant_assets(module, resolver)

    assert tuple(assets["packed"].shape) == (2, 2, 2, 2)
    torch.testing.assert_close(TorchEvaluator(resolver).run(module, {})[0], assets["packed"])


def test_numpy_layout_materializer_matches_torch_recipe_bytes():
    module = _packed_constant_output()
    weight = torch.arange(16, dtype=torch.float32).reshape(2, 8)
    checkpoint = MemoryCheckpoint(
        {},
        {"weight": TensorInfo("weight", fm.DType.FLOAT32, (2, 8), "memory")},
        {"weight": weight},
    )

    [(name, actual)] = iter_numpy_materialized_constant_assets(module, checkpoint)
    expected = materialize_constant_assets(module, DictWeightResolver({"weight": weight}))[name]

    assert actual.flags.c_contiguous
    assert actual.tobytes() == expected.view(torch.uint8).numpy().tobytes()


def test_numpy_materializer_falls_back_one_numeric_recipe_to_torch():
    module = _numeric_constant_output()
    lhs = torch.arange(8, dtype=torch.float32).reshape(2, 4)
    rhs = torch.full((2, 4), 3.0)
    checkpoint = MemoryCheckpoint(
        {},
        {
            "lhs": TensorInfo("lhs", fm.DType.FLOAT32, (2, 4), "memory"),
            "rhs": TensorInfo("rhs", fm.DType.FLOAT32, (2, 4), "memory"),
        },
        {"lhs": lhs, "rhs": rhs},
    )

    [(name, actual)] = iter_numpy_materialized_constant_assets(module, checkpoint)

    assert name == "product"
    torch.testing.assert_close(actual, lhs * rhs)


def test_constant_recipe_verification_does_not_refreeze_module_metadata(monkeypatch):
    import triton.flagmega.ir.model as model_module

    module = replace(
        (source := _packed_constant_output()),
        metadata={
            **dict(source.metadata),
            "large": tuple({"value": index} for index in range(128)),
        },
    )
    original = model_module._freeze_mapping

    def reject_module_metadata(value):
        if value is module.metadata:
            raise AssertionError("recipe verification must not clone full module metadata")
        return original(value)

    monkeypatch.setattr(model_module, "_freeze_mapping", reject_module_metadata)

    assert fm.verify_module(module) is module
