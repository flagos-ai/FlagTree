# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT

import pytest

from triton.flagmega import ir as fm
from triton.flagmega.compiler import Compiler
from triton.flagmega.ir.ops.nn._gdn_state import GatedDeltaNetStateConfig
from triton.flagmega.options import CompileOptions


def state_slice_graph(index=None, *, reusable=False):
    config = GatedDeltaNetStateConfig(3, 1, 2, 4, 4, 4, 16)

    class Graph(fm.Module):

        def forward(self):
            qkv = self.input("qkv", fm.tensor_type("bfloat16", (1, 16)), id="qkv")
            state = self.input("state", config.ref_type, id="state")
            weight = self.input("weight", fm.tensor_type("bfloat16", (16, 1, 4)), id="weight")
            layer = (self.input("layer", fm.tensor_type("int32", ()), id="layer")
                     if index is None else fm.F.builtin.scalar_const(fm.tensor_type("int32", ()), index, name="layer"))
            view = fm.F.nn.gated_delta_net_state_slice(state, layer, name="view")
            convolution = fm.F.nn.gated_delta_net_convolution(qkv, view, weight, conv_kernel_size=4, name="conv")
            result = fm.F.tensors.get_item(convolution, 0)
            parameters = (qkv, state, weight, layer) if index is None else (qkv, state, weight)
            if not reusable:
                self.function("main", parameters, (result, state))
                return
            self.function("worker", parameters, (result, ), attrs={"noinline": True, "reusable": True})
            entry = tuple(self.input("entry_" + node.id, node.type, id="entry_" + node.id) for node in parameters)
            first = fm.F.builtin.call(*entry, result_type=result.type, callee="worker",
                                      effect=fm.effect("read_write", "gated_delta_net_state"), name="first")
            second = fm.F.builtin.call(first, *entry[1:], result_type=result.type, callee="worker",
                                       effect=fm.effect("read_write", "gated_delta_net_state"), name="second")
            self.function("main", entry, (second, entry[1]))

    return Graph(dialect="high_level", stage="frozen_constants", entry="main").build()


@pytest.mark.parametrize("index", [None, 0, 1, 2])
@pytest.mark.parametrize("level", ["fast", "optimized"])
def test_gdn_state_slice_bufferizes_as_one_layer_memspan_without_allocation(index, level):
    compiled = Compiler(CompileOptions(bufferize_opt_level=level)).compile(state_slice_graph(index)).module
    plan = fm.verify_buffer_plan(compiled)
    assert plan.optimization_level == level
    bindings = dict(plan.function_map["main"].values)
    parent = tuple(plan.buffer_map[value] for value in bindings["state"])
    views = tuple(plan.buffer_map[value] for value in bindings["view"])
    assert compiled.node_map["view"].op == "tir.ref_slice"
    for original, view in zip(parent, views):
        assert view.physical_id == original.physical_id
        assert view.alias.source == original.id and view.alias.kind.value == "view"
        assert view.mem_span.size * 3 == original.mem_span.size
        assert view.shape == (1, *original.shape[1:])
        assert view.mem_span.is_within(original.mem_span)
        assert view.mem_span.may_alias(original.mem_span)
        if index is not None:
            assert view.byte_offset == index * view.nbytes
        else:
            assert view.mem_span.start.minimum == 0
            assert view.mem_span.start.maximum == 2 * view.nbytes
            assert view.offset_bindings
    assert dict(plan.function_map["main"].outputs)["state"] == bindings["state"]
