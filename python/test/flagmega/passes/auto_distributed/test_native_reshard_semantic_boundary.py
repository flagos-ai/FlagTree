# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT

import pytest
import torch

from triton.flagmega import ir as fm
from triton.flagmega.evaluator import DictWeightResolver, TorchEvaluator
from triton.flagmega.passes.auto_distributed.policy import lower_vectorization_contracts


@pytest.mark.parametrize("adapter", ["distributed.boxing", "distributed.sharded_view"])
@pytest.mark.parametrize("multi_axis", [False, True])
def test_native_vector_reshard_can_feed_scalar_reshape_without_losing_shared_vector_use(adapter, multi_axis):
    placement = fm.Placement((2, 2), "xy", "bb")
    lanes, axes, shape = ((2, 4), (0, 1), (2, 4)) if multi_axis else ((8, ), (1, ), (1, 4))
    reshaped_shape, result_axes = ((1, 2, 4), (1, 2)) if multi_axis else ((1, 2, 2), (2, ))
    logical_shape = (1, 4, 16) if multi_axis else (1, 2, 16)
    vector = fm.tensor_type(fm.vector_type("bfloat16", lanes), shape)
    source_type = fm.DistributedType(vector, (fm.SBP.broadcast(), fm.SBP.split_contiguous((0, ))), placement)
    target_type = fm.DistributedType(vector, (fm.SBP.broadcast(), fm.SBP.broadcast()), placement)

    class Graph(fm.Module):

        def forward(self):
            source = self.input("source", source_type, id="source")
            native = fm.F.math.vectorized_unary(
                source, unary_op="silu", name="native.compute", metadata={
                    "vectorization_root": "native",
                    "vectorization_role": "compute",
                    "vectorization_internal": True,
                    "vectorization_candidate": "vectorization.unary",
                    "vector_axes": axes,
                    "vector_lanes": lanes,
                    "vectorization_attrs": {},
                    "vectorized_from": "math.silu",
                })
            bridge = fm.get_definition(adapter).construct(native, new_type=target_type, name="bridge")
            reshaped = fm.F.tensors.reshape(
                bridge, reshaped_shape, name="reshape.compute", metadata={
                    "vectorization_root": "native",
                    "vectorization_semantic_id": "reshaped",
                    "vectorization_role": "propagated-reshape",
                    "vectorization_internal": True,
                    "vectorization_candidate": "vectorization.unary",
                    "vector_axes": result_axes,
                    "vector_lanes": lanes,
                    "vectorization_attrs": {"shape": logical_shape},
                    "vectorized_from": "tensors.reshape",
                })
            logical = fm.F.tensors.unpack(
                reshaped, axes=result_axes, name="reshaped", metadata={
                    "vectorization_root": "reshaped",
                    "vectorization_candidate": "vectorization.unary",
                    "vector_axes": result_axes,
                    "vector_lanes": lanes,
                    "vectorization_attrs": {"shape": logical_shape},
                    "vectorized_from": "tensors.reshape",
                })
            self.function("main", (source, ), (logical, bridge))

    original = Graph(dialect="ntt", stage="matmul_norm_stats_lowered", entry="main").build()
    fm.verify_module(original)
    lowered = lower_vectorization_contracts(original)
    fm.verify_module(lowered)
    assert lowered.node_map["bridge"].op == adapter and lowered.node_map["bridge"].type == target_type
    result = lowered.node_map["reshaped"]
    assert result.op == "tensors.reshape" and result.type.tensor.dtype == fm.DType.BFLOAT16
    boundary = lowered.node_map[result.inputs[0]]
    assert boundary.op == "tensors.unpack" and boundary.inputs == ("bridge", )
    assert boundary.type.tensor.shape[-1].fixed_value == (16 if multi_axis else 32)
    # Keep the physical native producer and shared vector-valued output alive.
    assert lowered.node_map["native.compute"].op == "math.vectorized_unary"
    inputs = {"source": torch.linspace(-3, 3, 64 if multi_axis else 32).reshape(*shape, *lanes).bfloat16()}
    before = TorchEvaluator(DictWeightResolver({})).run(original, inputs)
    after = TorchEvaluator(DictWeightResolver({})).run(lowered, inputs)
    for actual, expected in zip(after, before):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
