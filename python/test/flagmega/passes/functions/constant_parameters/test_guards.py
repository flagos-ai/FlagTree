# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT

from dataclasses import replace
import pytest

from triton.flagmega import ir as fm
from triton.flagmega.errors import IRVerificationError, StageError
from triton.flagmega.ir.ops.tensors.cast import Cast
from triton.flagmega.passes.constants import freeze_constant_islands
from triton.flagmega.passes.functions import lift_constant_parameter_expressions as lift
from triton.flagmega.passes.functions.lift_constant_expressions import _storage_value
from .helpers import module


def test_one_dynamic_actual_keeps_the_common_callee_unspecialized():
    source = module(dynamic=True, nested=True)
    assert lift(source) == source


@pytest.mark.parametrize("attribute", ["const_evaluable", "deterministic"])
def test_explicit_op_contract_required_not_purity_alone(monkeypatch, attribute):
    source = module()
    monkeypatch.setattr(Cast, attribute, False)
    assert lift(source) == source


def test_effectful_read_does_not_become_an_offline_expression(monkeypatch):
    source = module()
    monkeypatch.setattr(Cast, "infer_effect", classmethod(lambda cls, inputs, attrs: fm.Effect("read", "mutable")))
    source = replace(
        source,
        nodes=tuple(replace(n, effect=fm.Effect("read", "mutable")) if n.id == "cast" else n for n in source.nodes))
    assert lift(source) == source


def test_frozen_recipes_cannot_be_thawed_by_the_pass():
    with pytest.raises(StageError, match="constants_open"):
        lift(freeze_constant_islands(module()))


def test_new_parameter_name_collision_is_explicit():
    source = module()
    collision = fm.Node("transformed.constant_parameter", "builtin.var", (), source.node_map["runtime"].type,
                        attrs={"name": "collision"})
    source = replace(
        source, nodes=(*source.nodes, collision), functions=tuple(
            replace(f, parameters=(*f.parameters, collision.id)) if f.name == "main" else f for f in source.functions))
    with pytest.raises(IRVerificationError, match="collides"):
        lift(source)


@pytest.mark.parametrize("value", [
    fm.RefType("state"),
    fm.tensor_type(fm.PointerType(fm.DType.FLOAT32), (8, )),
    fm.tensor_type("float32", ("tokens", 8)),
    fm.TupleType((fm.tensor_type("float32", (8, )), fm.RefType("state")))
])
def test_nonmaterializable_state_pointer_or_dynamic_payload_is_not_an_abi_candidate(value):
    assert not _storage_value(value)
