# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Uniformity proofs follow values and immutable storage, never weight names."""

from dataclasses import replace
import struct

import pytest

from triton.flagmega import ir as fm
from triton.flagmega.codegen.triton.readonly_scalars import ReadonlyScalarAnalysis
from triton.flagmega.passes.constants import freeze_constant_islands


def case(transform=lambda value: value, *, value=1.0, dtype="float32", weight=False):
    class Graph(fm.Module):
        def forward(self):
            value_type = fm.tensor_type(dtype, (2, 4))
            literal = (self.weight("unit_scale", value_type, source="unread.safetensors", key="unit_scale", id="literal")
                       if weight else fm.F.builtin.splat_const(value_type, value, name="literal"))
            result = transform(literal)
            self.function("main", (), (result,))

    module = Graph(dialect="nn", stage="frozen_constants", entry="main").build()
    frozen = freeze_constant_islands(module)
    plan = fm.make_buffer_plan(frozen)
    output = frozen.function_map["main"].outputs[0]
    return frozen, plan, output


@pytest.mark.parametrize("transform", [
    lambda x: x,
    lambda x: fm.F.tensors.reshape(x, shape=(4, 2)),
    lambda x: fm.F.tensors.permute(x, (1, 0)),
    lambda x: fm.F.tensors.broadcast_to(x, shape=(3, 2, 4)),
    lambda x: fm.F.tensors.slice(x, starts=(1,), ends=(3,), axes=(1,)),
    lambda x: fm.F.distributed.sharded_view(x, fm.DistributedType(
        x.type, (fm.SBP.broadcast(), fm.SBP.split_contiguous((0,))), fm.Placement((2,), "x", "b"))),
])
def test_value_preserving_recipe_views_keep_uniformity(transform):
    module, plan, output = case(transform, value=-0.0)
    actual = ReadonlyScalarAnalysis(module, plan).float32_splat(output)
    assert actual is not None
    assert struct.pack("f", actual) == struct.pack("f", -0.0)


@pytest.mark.parametrize("transform", [
    lambda x: fm.F.tensors.cast(x, "bfloat16"),
    lambda x: fm.F.tensors.cast(fm.F.tensors.cast(x, "bfloat16"), "float32"),
    lambda x: fm.F.math.add(x, x),
    lambda x: fm.F.tensors.pack(x, (2,), axes=(1,)),
    lambda x: fm.F.tensors.pad(x, (0, 1), pad_value=0.0),
])
def test_arithmetic_cast_and_packed_storage_are_not_assumed_transparent(transform):
    module, plan, output = case(transform)
    assert ReadonlyScalarAnalysis(module, plan).float32_splat(output) is None


def test_runtime_storage_never_inherits_a_literal_proof():
    module, plan, output = case()
    buffer = plan.buffer_map[output]
    altered = replace(plan, buffers=tuple(replace(value, storage="external") if value.id == output else value
                                          for value in plan.buffers))
    assert buffer.source_node == output
    assert ReadonlyScalarAnalysis(module, altered).float32_splat(output) is None


def test_edited_recipe_gets_a_fresh_value_not_a_process_global_cache():
    first, first_plan, first_output = case(value=1.0)
    second, second_plan, second_output = case(value=0.5)
    assert first_output == second_output
    assert ReadonlyScalarAnalysis(first, first_plan).float32_splat(first_output) == 1.0
    assert ReadonlyScalarAnalysis(second, second_plan).float32_splat(second_output) == 0.5


@pytest.mark.parametrize("dtype", ["bfloat16", "int32"])
def test_non_float32_storage_is_not_reinterpreted(dtype):
    module, plan, output = case(dtype=dtype)
    assert ReadonlyScalarAnalysis(module, plan).float32_splat(output) is None


def test_weight_named_unit_scale_is_not_a_proven_literal():
    module, plan, output = case(weight=True)
    assert ReadonlyScalarAnalysis(module, plan).float32_splat(output) is None


def test_partial_components_are_not_logical_splat_values():
    def partial(value):
        placement = fm.Placement((2,), "x", "b")
        target = fm.DistributedType(value.type, (fm.SBP.broadcast(),) * 2, placement, fm.SBP.partial((0,)))
        return fm.F.distributed.boxing(value, target)

    module, plan, output = case(partial)
    assert ReadonlyScalarAnalysis(module, plan).float32_splat(output) is None
