# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Consumer packing must reach only MhcPre's collapsed tuple field."""

import pytest
import torch

from triton.flagmega import ir as fm
from triton.flagmega.evaluator import DictWeightResolver, TorchEvaluator
from triton.flagmega.passes.functions import post_function_boundary_pack_propagation
from triton.flagmega.targets import NvidiaSm90Target

from python.test.flagmega.ir.ops.nn.test_mhc import (
    COEFFICIENTS, HC, HIDDEN, TOKENS, _operands, _pre_attrs,
)


class MhcConsumers(fm.Module):
    def __init__(self, lanes=(8,), *, scalar_users=False, mixed_users=False,
                 output_lanes=(), demand_field=2, axis=-1, large=False, collapsed_dtype=None,
                 collapse_input_dtype=None, collapse_reduction="sum"):
        super().__init__(dialect="high_level", stage="packed", entry="main")
        self.lanes, self.scalar_users, self.mixed_users = lanes, scalar_users, mixed_users
        self.output_lanes, self.demand_field, self.axis = output_lanes, demand_field, axis
        self.large = large
        self.collapsed_dtype = collapsed_dtype
        self.collapse_input_dtype, self.collapse_reduction = collapse_input_dtype, collapse_reduction

    def forward(self):
        types = {"streams": fm.tensor_type("bfloat16", (TOKENS, HC, HIDDEN)),
                 "projection": fm.tensor_type("float32", (COEFFICIENTS, HC * HIDDEN)),
                 "base": fm.tensor_type("float32", (COEFFICIENTS,)),
                 "scale": fm.tensor_type("float32", (3,))}
        operands = tuple(self.input(name, value, id=name) for name, value in types.items())
        attrs = _pre_attrs()
        if self.output_lanes:
            attrs["output_lanes"] = self.output_lanes
        pre = fm.F.nn.mhc_pre(*operands, name="pre", collapsed_dtype=self.collapsed_dtype,
                              collapse_input_dtype=self.collapse_input_dtype,
                              collapse_reduction=self.collapse_reduction, **attrs)
        fields = fm.F.tensors.get_items(pre, 0, 1, 2, name_prefix="field")
        outputs = [fields[0], fields[1]]
        for index in range(2):
            # Separate GetItems of one producer must share the packed result.
            field = fm.F.tensors.get_item(pre, self.demand_field, name=f"demand_{index}")
            outputs.append(fm.F.tensors.pack(field, self.lanes,
                                            axes=(self.axis,) * len(self.lanes)))
        if self.scalar_users:
            outputs.extend((fields[2], pre))
        if self.mixed_users:
            outputs.append(fm.F.tensors.pack(fields[2], (4,), axes=(-1,)))
        if self.large:
            # Keep a genuine reachable >20k-node graph on the dataflow path.
            padding = operands[-1]
            for _ in range(20_001):
                padding = fm.F.math.add(padding, operands[-1])
            outputs.append(padding)
        self.function("main", operands, tuple(outputs))


@pytest.mark.parametrize("lanes", [(8,), (2, 4), (4, 2)])
@pytest.mark.parametrize("scalar_users", [False, True])
def test_mhc_packing_reaches_single_producer_and_preserves_tuple_users(lanes, scalar_users):
    original = MhcConsumers(lanes, scalar_users=scalar_users).build()
    rewritten = post_function_boundary_pack_propagation(original, NvidiaSm90Target())
    pre, = (node for node in rewritten.nodes if node.op == "nn.mhc_pre")
    assert pre.attrs.get("output_lanes") == lanes
    assert pre.type.fields[:2] == original.node_map["pre"].type.fields[:2]
    assert pre.type.fields[2] == fm.tensor_type(fm.vector_type("bfloat16", lanes),
                                              (TOKENS, HIDDEN // 8))
    assert not any(node.op == "tensors.pack" for node in rewritten.nodes)
    values = {name: value for name, value in _operands().items() if name != "value"}
    evaluator = TorchEvaluator(DictWeightResolver({}))
    torch.testing.assert_close(evaluator.run(rewritten, values), evaluator.run(original, values), rtol=0, atol=0)
    assert post_function_boundary_pack_propagation(rewritten, NvidiaSm90Target()).semantic_hash == rewritten.semantic_hash


def test_mhc_mixed_layout_consumers_keep_their_types_without_duplicating_compute():
    original = MhcConsumers(scalar_users=True, mixed_users=True).build()
    rewritten = post_function_boundary_pack_propagation(original, NvidiaSm90Target())
    pre, = (node for node in rewritten.nodes if node.op == "nn.mhc_pre")
    # Identical Packs are CSE'd first, leaving a stable tie between 4 and 8.
    assert pre.attrs.get("output_lanes") == (4,)
    assert tuple(rewritten.node_map[name].type for name in rewritten.functions[0].outputs) == tuple(
        original.node_map[name].type for name in original.functions[0].outputs)
    values = {name: value for name, value in _operands().items() if name != "value"}
    evaluator = TorchEvaluator(DictWeightResolver({}))
    torch.testing.assert_close(evaluator.run(rewritten, values), evaluator.run(original, values), rtol=0, atol=0)


def test_mhc_packing_composes_with_an_already_packed_output():
    original = MhcConsumers((4,), output_lanes=(2,), scalar_users=True).build()
    rewritten = post_function_boundary_pack_propagation(original, NvidiaSm90Target())
    pre, = (node for node in rewritten.nodes if node.op == "nn.mhc_pre")
    assert pre.attrs["output_lanes"] == (4, 2)
    assert not any(node.op == "tensors.pack" for node in rewritten.nodes)
    values = {name: value for name, value in _operands().items() if name != "value"}
    evaluator = TorchEvaluator(DictWeightResolver({}))
    torch.testing.assert_close(evaluator.run(rewritten, values), evaluator.run(original, values), rtol=0, atol=0)


@pytest.mark.parametrize("kwargs", [{"lanes": (4,), "demand_field": 0},
                                    {"lanes": (3,), "axis": 0}])
def test_mhc_does_not_take_packing_demands_from_coefficients_or_token_axis(kwargs):
    original = MhcConsumers(**kwargs).build()
    rewritten = post_function_boundary_pack_propagation(original, NvidiaSm90Target())
    pre, = (node for node in rewritten.nodes if node.op == "nn.mhc_pre")
    assert not pre.attrs.get("output_lanes")
    assert pre.type == original.node_map["pre"].type


def test_mhc_producer_packing_is_registered_on_the_large_graph_path():
    original = MhcConsumers(large=True).build()
    assert len(original.nodes) > 20_000
    rewritten = post_function_boundary_pack_propagation(original, NvidiaSm90Target())
    pre, = (node for node in rewritten.nodes if node.op == "nn.mhc_pre")
    assert pre.attrs.get("output_lanes") == (8,)
    assert not any(node.op == "tensors.pack" for node in rewritten.nodes)


@pytest.mark.parametrize("large", [False, True])
def test_wide_collapse_survives_packing_with_scalar_and_tuple_users(large):
    original = MhcConsumers(collapsed_dtype="float32", scalar_users=True, large=large,
                           collapse_input_dtype="bfloat16", collapse_reduction="sequential_fma").build()
    rewritten = post_function_boundary_pack_propagation(original, NvidiaSm90Target())
    pre, = (node for node in rewritten.nodes if node.op == "nn.mhc_pre")
    assert pre.attrs["collapsed_dtype"] == "float32"
    assert pre.attrs["collapse_input_dtype"] == "bfloat16"
    assert pre.attrs["collapse_reduction"] == "sequential_fma"
    assert pre.type.fields[2] == fm.tensor_type(fm.vector_type("float32", (8,)), (TOKENS, HIDDEN // 8))
    assert not any(node.op == "tensors.pack" for node in rewritten.nodes)
    values = {name: value for name, value in _operands().items() if name != "value"}
    evaluator = TorchEvaluator(DictWeightResolver({}))

    def evaluable(module):
        if not large:
            return module
        from dataclasses import replace

        # The real 20k-node branch exercises packing's large-graph path above.
        # Evaluate every MHC result without recursively interpreting that
        # independent padding branch, which exceeds the evaluator's stack.
        return replace(module, functions=tuple(replace(f, outputs=f.outputs[:-1])
                       if f.name == module.entry else f for f in module.functions))

    torch.testing.assert_close(evaluator.run(evaluable(rewritten), values),
                               evaluator.run(evaluable(original), values), atol=0, rtol=0)
    assert post_function_boundary_pack_propagation(rewritten, NvidiaSm90Target()).semantic_hash == rewritten.semantic_hash


@pytest.mark.parametrize("collapsed_dtype", [None, "float32"])
def test_callee_normalization_demand_reaches_mhc_and_both_reusable_calls(collapsed_dtype):
    class Graph(fm.Module):
        def forward(self):
            parameter = self.input("parameter", fm.tensor_type(collapsed_dtype or "bfloat16", (TOKENS, HIDDEN)),
                                   id="parameter")
            packed = fm.F.tensors.pack(parameter, (8,), axes=(-1,))
            stats = fm.F.nn.norm_stats(packed, axis=1, use_mean=False)
            self.function("stats", (parameter,), (stats,), attrs={"reusable": True})
            types = {"streams": fm.tensor_type("bfloat16", (TOKENS, HC, HIDDEN)),
                     "projection": fm.tensor_type("float32", (COEFFICIENTS, HC * HIDDEN)),
                     "base": fm.tensor_type("float32", (COEFFICIENTS,)), "scale": fm.tensor_type("float32", (3,))}
            operands = tuple(self.input(name, value, id=name) for name, value in types.items())
            pre = fm.F.nn.mhc_pre(*operands, name="pre", collapsed_dtype=collapsed_dtype, **_pre_attrs())
            collapsed = fm.F.tensors.get_item(pre, 2)
            first = fm.F.builtin.call(collapsed, callee="stats", result_type=stats.type)
            added = fm.F.math.add(collapsed, collapsed)
            second = fm.F.builtin.call(added, callee="stats", result_type=stats.type)
            self.function("main", operands, (first, second))

    original = Graph(dialect="high_level", stage="packed", entry="main").build()
    rewritten = post_function_boundary_pack_propagation(original, NvidiaSm90Target())
    pre, = (node for node in rewritten.nodes if node.op == "nn.mhc_pre")
    assert pre.attrs.get("output_lanes") == (8,)
    parameter, = rewritten.function_map["stats"].parameters
    assert rewritten.node_map[parameter].type == pre.type.fields[2]
    assert not any(node.op == "tensors.pack" for node in rewritten.nodes)
    for call in (node for node in rewritten.nodes if node.op == "builtin.call"):
        assert rewritten.node_map[call.inputs[0]].type == rewritten.node_map[parameter].type
    values = {name: value for name, value in _operands().items() if name != "value"}
    evaluator = TorchEvaluator(DictWeightResolver({}))
    torch.testing.assert_close(evaluator.run(rewritten, values), evaluator.run(original, values), rtol=0, atol=0)
    assert post_function_boundary_pack_propagation(rewritten, NvidiaSm90Target()).semantic_hash == rewritten.semantic_hash
