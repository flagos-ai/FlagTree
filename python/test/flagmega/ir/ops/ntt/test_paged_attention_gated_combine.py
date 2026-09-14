# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT

from dataclasses import replace

import pytest
import torch

from triton.flagmega import ir as fm
from triton.flagmega.errors import IRSchemaError
from triton.flagmega.evaluator import DictWeightResolver, TorchEvaluator
from triton.flagmega.ir.ops.ntt.paged_attention_gated_combine import PagedAttentionGatedCombine
from python.test.flagmega.passes.tir.test_attention_gate_fusion import graph
from triton.flagmega.passes.tir.fuse_attention_gate import fuse_attention_gate


@pytest.mark.parametrize("dtype", ["bfloat16", "float16", "float32"])
@pytest.mark.parametrize("lanes", [(), (2, 2)])
@pytest.mark.parametrize("layout", [("seq", "head", "dim"), ("dim", "head", "seq")])
def test_explicit_op_preserves_rounding_for_float_types_and_axis_orders(dtype, lanes, layout):
    scalar_shape = tuple({"seq": 2, "head": 3, "dim": 8}[axis] for axis in layout)
    stats_shape = tuple(1 if axis == "dim" else size for axis, size in zip(layout, scalar_shape))
    output_shape = tuple(size // (4 if lanes else 1) if axis == "dim" else size for axis, size in zip(layout, scalar_shape))
    tensor = fm.tensor_type(fm.vector_type(dtype, lanes) if lanes else dtype, output_shape)

    class Graph(fm.Module):
        def forward(self):
            maximum = self.input("maximum", fm.tensor_type("float32", stats_shape))
            total = self.input("total", maximum.type)
            acc = self.input("acc", fm.tensor_type("float32", scalar_shape))
            gate = self.input("gate", tensor)
            attrs = dict(layout=layout, hidden_size=24, output_data_type=tensor.dtype,
                         output_type=tensor, split_hierarchy_axis=0, split_count=2)
            fused = fm.F.ntt.paged_attention_gated_combine(maximum, total, acc, gate, **attrs)
            attention = fm.F.ntt.paged_attention_combine(maximum, total, acc, **attrs)
            separate = attention
            # Standalone Sigmoid currently accepts BF16/FP32, not FP16.
            if dtype != "float16":
                sigmoid = (fm.F.math.vectorized_unary(gate, unary_op="sigmoid") if lanes else fm.F.math.sigmoid(gate))
                separate = (fm.F.math.vectorized_binary(attention, sigmoid, binary_op="mul") if lanes else fm.F.math.mul(attention, sigmoid))
            self.function("main", (maximum, total, acc, gate), (fused, separate))

    module = Graph(dialect="ntt", stage="packed", entry="main").build()
    torch.manual_seed(72)
    values = {"maximum": torch.zeros(stats_shape), "total": torch.full(stats_shape, 2.),
              "acc": torch.full(scalar_shape, 2. + 2. ** -7),
              "gate": torch.randn((*output_shape, *lanes)).to(getattr(torch, dtype))}
    actual, expected = TorchEvaluator(DictWeightResolver({})).run(module, values)
    if dtype == "float16":
        expected = expected * values["gate"].float().sigmoid().half()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_gate_is_owner_local_while_all_three_states_are_partial_group_reads():
    assert tuple(parameter.memory_effect.owner_access for parameter in PagedAttentionGatedCombine.input_parameters) == (
        *(fm.MemoryOwnerAccess.PARTIAL_GROUP,) * 3, fm.MemoryOwnerAccess.LOCAL)


@pytest.mark.parametrize("mismatch", ["dtype", "owners", "partial"])
def test_gate_contract_rejects_mismatched_dtype_owners_or_partial(mismatch):
    module = fuse_attention_gate(graph(distributed=True))
    root = module.node_map["result"]
    arguments = [module.node_map[key] for key in root.inputs]
    source = arguments[3].type
    if mismatch == "dtype":
        invalid = replace(source, tensor=fm.tensor_type("float32", source.tensor.shape))
    else:
        policies = (*source.axis_policies[:2], fm.SBP.broadcast())
        invalid = fm.DistributedType(source.tensor, policies, source.placement,
                                    fm.SBP.partial((0,), fm.ReduceOp.SUM) if mismatch == "partial" else None)
    arguments[3] = replace(arguments[3], type=invalid)
    with pytest.raises(IRSchemaError, match="identical types and owners"):
        PagedAttentionGatedCombine.prepare(arguments, root.attrs)
