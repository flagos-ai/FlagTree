# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""A distribution bridge does not turn a result bitcast into a compute op."""

import pytest

from triton.flagmega import ir as fm
from triton.flagmega.passes.auto_distributed.policy import lower_vectorization_contracts


@pytest.mark.parametrize("adapter", ["boxing", "force_boxing", "sharded_view"])
@pytest.mark.parametrize("bridge_count", [1, 2])
def test_cloned_norm_result_keeps_exact_compute_through_reshard(adapter, bridge_count):
    placement = fm.Placement((2, 2), "yx", "bb")
    b = fm.SBP.broadcast()
    vector = fm.vector_type("float32", (4,))
    value_type = fm.DistributedType(fm.tensor_type(vector, (1, 8)), (b, b), placement)
    parameter_type = fm.DistributedType(fm.tensor_type(vector, (8,)), (b,), placement)
    stats_type = fm.DistributedType(fm.tensor_type("float32", (1, 1, 1)), (b, b, b), placement)
    attrs = {"axis": -1, "epsilon": 1e-5, "use_mean": False}
    provenance = {
        "cloned_for_function_variant": "norm_layout_variant",
        "vectorization_candidate": "vectorization.norm_apply.reduction_axis",
        "vectorization_attrs": attrs,
        "vector_axes": (1,), "vector_lanes": (4,),
        "vectorized_from": "nn.norm_apply",
    }

    class Graph(fm.Module):
        def forward(self):
            value = self.input("value", value_type)
            stats = self.input("stats", stats_type)
            scale = self.input("scale", parameter_type)
            bias = self.input("bias", parameter_type)
            compute = fm.F.nn.norm_apply(value, stats, scale, bias, **attrs,
                name="compute.variant", metadata={
                    **provenance, "vectorization_internal": True,
                    "vectorization_role": "compute", "vectorization_root": "original_norm",
                })
            bridge = compute
            for index in range(bridge_count):
                target = fm.DistributedType(
                    value_type.tensor,
                    (b, fm.SBP.split_contiguous(tuple(range(index + 1)))), placement,
                )
                bridge = getattr(fm.F.distributed, adapter)(bridge, target, name=f"bridge_{index}")
            result = fm.F.tensors.bitcast(bridge, "float32", name="norm.variant", metadata=provenance)
            self.function("main", (value, stats, scale, bias), (result,))

    original = Graph(dialect="ntt", stage="norm_bindings_finalized", entry="main").build()
    lowered = lower_vectorization_contracts(original)
    fm.verify_module(lowered)
    assert lowered.node_map["compute.variant"].inputs == original.node_map["compute.variant"].inputs
    assert lowered.node_map["compute.variant"].type == value_type
    assert lowered.node_map["norm.variant"].op == "tensors.bitcast"
    assert lowered.node_map["norm.variant"].inputs == (f"bridge_{bridge_count - 1}",)
    for index in range(bridge_count):
        assert lowered.node_map[f"bridge_{index}"].op == f"distributed.{adapter}"
