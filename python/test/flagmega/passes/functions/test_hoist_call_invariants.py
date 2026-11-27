# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Repeated immutable arguments may share computation, not mutable state reads."""

from dataclasses import replace

import pytest
import torch

from triton.flagmega import ir as fm
from triton.flagmega.evaluator import DictWeightResolver, TorchEvaluator
from triton.flagmega.passes.functions import function_nodes


def module(*, same=True, calls=2, export_source=False):
    b = fm.IRBuilder(dialect="high_level", stage="decomposed")
    f32 = fm.tensor_type("float32", (8,))
    bf16 = fm.tensor_type("bfloat16", (8,))
    a = b.var("a", f32, id="a")
    other = b.var("other", f32, id="other")
    x = b.var("x", bf16, id="x")
    p = b.var("p", f32, id="p")
    q = b.var("q", bf16, id="q")
    rounded = b.call("tensors.cast", (p,), bf16, attrs={"dtype": "bfloat16"}, id="rounded")
    result = b.call("math.add", (rounded, q), bf16, id="result")
    outputs = (result, p) if export_source else (result,)
    result_type = fm.TupleType((bf16, f32)) if export_source else bf16
    values = [b.call("builtin.call", (a if same or i == 0 else other, x), result_type,
                     attrs={"callee": "shared"}, id=f"call{i}") for i in range(calls)]
    b.function("shared", (p, q), outputs, attrs={"reusable": True, "calling_convention": "device"})
    b.function("main", (a, other, x), values)
    return fm.verify_module(b.build(entry="main"))


def transform(value):
    # Access through the public namespace so fail-before is a missing pass,
    # without preventing collection of the neighboring regression cases.
    from triton.flagmega import passes
    return passes.functions.hoist_call_invariant_expressions(value)


@pytest.mark.parametrize("export_source", [False, True])
def test_shared_transform_moves_to_caller_once_and_roundtrips(tmp_path, export_source):
    original = module(export_source=export_source)
    result = fm.verify_module(transform(original))
    callee = result.function_map["shared"]
    assert not any(n.op == "tensors.cast" for n in function_nodes(result, callee))
    casts = [n for n in function_nodes(result, result.function_map["main"]) if n.op == "tensors.cast"]
    assert len(casts) == 1
    assert casts[0].inputs == ("a",)
    assert ("p" in callee.parameters) == export_source
    assert result.node_map["call0"].inputs == result.node_map["call1"].inputs
    torch.manual_seed(712)
    inputs = {"a": torch.randn(8), "other": torch.randn(8), "x": torch.randn(8).bfloat16()}
    evaluator = TorchEvaluator(DictWeightResolver({}))
    torch.testing.assert_close(evaluator.run(result, inputs), evaluator.run(original, inputs), rtol=0, atol=0)
    assert transform(result).semantic_hash == result.semantic_hash
    path = fm.emit_module(result, tmp_path / "hoisted.py")
    assert fm.load_module(path).semantic_hash == result.semantic_hash


@pytest.mark.parametrize("same,calls", [(False, 2), (True, 1)])
def test_different_arguments_or_single_invocation_do_not_hoist(same, calls):
    original = module(same=same, calls=calls)
    # q is invariant, but its Add also depends on the varying p.
    assert transform(original).semantic_hash == original.semantic_hash


def test_effectful_expression_is_not_moved():
    original = module()
    rounded = original.node_map["rounded"]
    effectful = replace(rounded, effect=fm.Effect("read", "test_state"))
    original = replace(original, nodes=tuple(effectful if n.id == rounded.id else n for n in original.nodes))
    assert transform(original).semantic_hash == original.semantic_hash


def test_invariance_never_treats_pointer_or_reference_payload_as_immutable_data():
    from triton.flagmega.passes.functions.hoist_call_invariants import _immutable
    reference = fm.RefType("state")
    pointer = fm.tensor_type(fm.PointerType(fm.DType.FLOAT32), (8,))
    assert not _immutable(reference)
    assert not _immutable(pointer)
    assert not _immutable(fm.TupleType((fm.tensor_type("float32", (8,)), reference)))


def test_varying_data_keeps_the_callee_computation_and_only_hoists_shared_cast():
    original = module()
    nodes = original.node_map
    extra = fm.Node("next_x", "math.add", ("x", "x"), nodes["x"].type)
    second = replace(nodes["call1"], inputs=("a", extra.id))
    rebuilt = []
    for node in original.nodes:
        if node.id == second.id:
            rebuilt.extend((extra, second))
        else:
            rebuilt.append(node)
    original = fm.verify_module(replace(original, nodes=tuple(rebuilt)))
    result = fm.verify_module(transform(original))
    callee = result.function_map["shared"]
    assert any(node.op == "math.add" for node in function_nodes(result, callee))
    assert not any(node.op == "tensors.cast" for node in function_nodes(result, callee))
    first_args, second_args = (result.node_map[f"call{i}"].inputs for i in range(2))
    assert first_args[-1] == second_args[-1]
    assert first_args[0] != second_args[0]
    values = {"a": torch.randn(8), "other": torch.randn(8), "x": torch.randn(8).bfloat16()}
    evaluator = TorchEvaluator(DictWeightResolver({}))
    torch.testing.assert_close(evaluator.run(result, values), evaluator.run(original, values), rtol=0, atol=0)
