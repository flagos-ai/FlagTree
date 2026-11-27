# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Reusable stage outlining retains SSA, numerical boundaries and call effects."""
from dataclasses import replace

import pytest

from triton.flagmega import ir as fm
from triton.flagmega.errors import IRVerificationError
from triton.flagmega.evaluator import DictWeightResolver, TorchEvaluator
from triton.flagmega.passes.functions.graph import function_nodes
from triton.flagmega.passes.functions.outline_stages import FunctionStage, outline_function_stages


STAGES = tuple(FunctionStage(name, (node,)) for name, node in (
    ("prefix", "w_prefix"), ("routed_up", "w_left0"), ("routed_down", "w_left1"),
    ("shared_up", "w_right0"), ("shared_down", "w_right1"), ("join", "w_joined"),
))


def reusable_graph(*, single_result=False, packed=False, effectful=False):
    dtype = fm.vector_type("bfloat16", (8,)) if packed else "bfloat16"
    # Elementwise graphs exercise packed tuple fields without defining a new
    # numerical matmul contract; the ordinary scalar graph is a real fork/join.
    kind = fm.tensor_type(dtype, (1, 16 if packed else 128))

    class Graph(fm.Module):
        def forward(self):
            x = self.input("wx", kind, id="wx")
            bias = self.input("wbias", kind, id="wbias")
            if packed:
                weights = [self.input("w" + s, kind, id="w" + s) for s in "abcd"]
            else:
                weights = [self.input("w" + s, fm.tensor_type("bfloat16", shape), id="w" + s)
                           for s, shape in zip("abcd", ((64, 128), (32, 64), (64, 128), (32, 64)))]
            prefix = fm.F.math.add(x, bias, name="w_prefix")
            if packed:
                left0 = fm.F.math.add(prefix, weights[0], name="w_left0")
                left1 = fm.F.math.mul(left0, weights[1], name="w_left1")
                right0 = fm.F.math.add(prefix, weights[2], name="w_right0")
                right1 = fm.F.math.mul(right0, weights[3], name="w_right1")
            else:
                left0 = fm.F.math.matmul(prefix, weights[0], transpose_b=True, name="w_left0")
                left1 = fm.F.math.matmul(left0, weights[1], transpose_b=True, name="w_left1")
                right0 = fm.F.math.matmul(prefix, weights[2], transpose_b=True, name="w_right0")
                right1 = fm.F.math.matmul(right0, weights[3], transpose_b=True, name="w_right1")
            joined = fm.F.math.add(left1, right1, name="w_joined")
            outs = (joined,) if single_result else (left1, right1, joined)
            self.function("worker", (x, bias, *weights), outs,
                          attrs={"reusable": True, "noinline": True, "calling_convention": "device",
                                 "function_boundary_layout": {"inputs": (0,), "outputs": (0,)}})
            x0 = self.input("x0", kind, id="x0")
            x1 = self.input("x1", kind, id="x1")
            args = [self.input("bias", kind, id="bias")]
            args.extend(self.input(s, w.type, id=s) for s, w in zip("abcd", weights))
            result_type = joined.type if single_result else fm.TupleType(tuple(o.type for o in outs))
            effect = fm.effect("read_write", "state") if effectful else fm.PURE
            calls = [fm.F.builtin.call(value, *args, result_type=result_type, callee="worker", effect=effect,
                                       name=f"call{index}") for index, value in enumerate((x0, x1))]
            self.function("main", (x0, x1, *args), tuple(calls))
    return Graph(dialect="nn", stage="frozen_constants", entry="main").build()


@pytest.mark.parametrize("single_result,packed", [(False, False), (True, False), (False, True)])
def test_outlining_reuses_one_body_for_repeated_calls_and_round_trips(tmp_path, single_result, packed):
    torch = pytest.importorskip("torch")
    original = reusable_graph(single_result=single_result, packed=packed)
    result = outline_function_stages(original, "worker", STAGES)
    assert len(result.functions) == 7
    assert {f.name for f in result.functions} == {"main", *("worker." + s.name for s in STAGES)}
    assert all(sum(n.id == s.nodes[0] for n in result.nodes) == 1 for s in STAGES)
    assert sum(n.op == "builtin.call" for n in result.nodes) == 12
    assert all("function_boundary_layout" not in f.attrs for f in result.functions)
    assert result.metadata["outlined_function_stages"][0]["invalidated_boundary_layouts"]
    assert sum(n.op == "math.matmul" for n in result.nodes) == (0 if packed else 4)
    feeds = {}
    generator = torch.Generator().manual_seed(391)
    for name in original.function_map["main"].parameters:
        kind = original.node_map[name].type
        shape = tuple(d.fixed_value for d in kind.shape) + tuple(getattr(kind.dtype, "lanes", ()))
        feeds[name] = (torch.randint(-4, 5, shape, generator=generator).float() / 16).bfloat16()
    evaluator = TorchEvaluator(DictWeightResolver({}))
    torch.testing.assert_close(evaluator.run(result, feeds), evaluator.run(original, feeds), rtol=0, atol=0)
    reloaded = fm.load_module(fm.emit_module(result, tmp_path / "outlined.py"))
    assert reloaded.semantic_hash == result.semantic_hash
    torch.testing.assert_close(evaluator.run(reloaded, feeds), evaluator.run(original, feeds), rtol=0, atol=0)
    if single_result:
        assert result.node_map["call0"].op == "builtin.call"
    else:
        assert result.node_map["call0"].op == "builtin.tuple"


def test_original_caller_effect_is_kept_once_on_each_prefix_call():
    original = reusable_graph(effectful=True)
    result = outline_function_stages(original, "worker", STAGES)
    calls = [n for n in result.nodes if n.op == "builtin.call"]
    writes = [n for n in calls if not n.effect.is_pure]
    assert [(n.id, n.effect) for n in writes] == [(f"call{i}.stage0", fm.effect("read_write", "state")) for i in (0, 1)]
    assert all(n.attrs["callee"] == "worker.prefix" for n in writes)


def test_globally_shared_constant_result_keeps_its_original_identity():
    original = reusable_graph()
    literal = fm.Node("shared_literal", "builtin.splat_const", (), original.node_map["w_prefix"].type,
                      attrs={"value": 0.25})
    result_type = fm.TupleType((*original.node_map["call0"].type.fields, literal.type))
    source = replace(original, nodes=(literal, *tuple(
        replace(n, type=result_type) if n.id in {"call0", "call1"} else n for n in original.nodes)),
        functions=tuple(replace(f, outputs=(*f.outputs, literal.id)) for f in original.functions))
    fm.verify_module(source)
    result = outline_function_stages(source, "worker", STAGES)
    assert result.node_map["call0"].inputs[-1] == literal.id
    assert result.node_map["call1"].inputs[-1] == literal.id
    assert result.node_map[literal.id] is literal


@pytest.mark.parametrize("change,message", [
    ("missing", "exactly partition"), ("duplicate", "exactly partition"),
    ("forward", "later stage"), ("effect", "stateful/reference"),
    ("attributes", "semantic function attributes"), ("planned", "pre-allocation"),
])
def test_outlining_rejects_incomplete_or_unsound_cuts(change, message):
    source = reusable_graph()
    stages = STAGES
    if change == "missing":
        stages = STAGES[:-1]
    elif change == "duplicate":
        stages = (*STAGES, FunctionStage("again", ("w_prefix",)))
    elif change == "forward":
        stages = (STAGES[1], STAGES[0], *STAGES[2:])
    elif change == "effect":
        # Use a typed call with a real effect instead of corrupting a pure op.
        original = source.node_map["w_left0"]
        parameter = fm.Node("effect_parameter", "builtin.var", (), source.node_map["w_prefix"].type,
                            attrs={"name": "effect_parameter"})
        body = replace(original, id="effect_body", inputs=("effect_parameter", "wa"))
        # The new leaf receives all its inputs as parameters; the shared weight
        # formal is legal and keeps this an actual effectful call fixture.
        source = replace(source, nodes=(*tuple(
            replace(n, op="builtin.call", effect=fm.effect("write", "x"), attrs={"callee": "effect_leaf"})
            if n.id == "w_left0" else n for n in source.nodes), parameter, body),
            functions=(*source.functions, fm.Function("effect_leaf", ("effect_parameter", "wa"), ("effect_body",))))
    elif change == "attributes":
        source = replace(source, functions=tuple(replace(f, attrs={**f.attrs, "unknown_contract": True})
                                                if f.name == "worker" else f for f in source.functions))
    else:
        source = replace(source, metadata={"codegen_package_plan": {"schema": "already-planned"}})
    fm.verify_module(source)
    with pytest.raises(IRVerificationError, match=message):
        outline_function_stages(source, "worker", stages)
