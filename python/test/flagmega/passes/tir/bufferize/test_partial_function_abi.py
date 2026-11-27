# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT

from math import prod

import pytest

from triton.flagmega import ir as fm


@pytest.mark.parametrize("hierarchy", [(2, 2), (2, 4, 2)])
def test_partial_statistics_preserve_owner_components_across_function_abi(hierarchy):
    placement = fm.Placement(hierarchy, "abc"[:len(hierarchy)], "b" * len(hierarchy))
    axes = tuple(range(len(hierarchy)))
    split = fm.DistributedType(
        fm.tensor_type("float32", (1, 64)),
        (fm.SBP.broadcast(), fm.SBP.split_contiguous(axes)), placement,
    )
    stats_tensor = fm.tensor_type("float32", (1, 1, 1))
    partial = fm.DistributedType(
        stats_tensor, (fm.SBP.broadcast(),) * 3, placement, fm.SBP.partial(axes)
    )
    complete = fm.DistributedType(stats_tensor, (fm.SBP.broadcast(),) * 3, placement)

    class Graph(fm.Module):
        def __init__(self):
            super().__init__(dialect="tir", stage="selected_tir", entry="main")

        def forward(self):
            parameter = self.input("parameter", split, id="parameter")
            stats = fm.F.nn.norm_stats(parameter, axis=-1, use_mean=False, name="stats")
            self.function("producer", (parameter,), (stats,), attrs={"reusable": True, "noinline": True})
            forwarded = self.input("forwarded", partial, id="forwarded")
            self.function("forward", (forwarded,), (forwarded,), attrs={"reusable": True, "noinline": True})
            value = self.input("value", split, id="value")
            first = fm.F.builtin.call(value, result_type=partial, callee="producer", name="first")
            second = fm.F.builtin.call(first, result_type=partial, callee="forward", name="second")
            result = fm.F.distributed.boxing(second, complete, name="result")
            self.function("main", (value,), (result,))

    plan = fm.make_buffer_plan(Graph().build())
    expected_kind = fm.DistributedBufferStorageKind.COMPACT_PER_OWNER
    for buffer_id in ("stats", "forwarded", "first"):
        descriptor = plan.buffer_map[buffer_id]
        assert descriptor.distributed_storage_kind is expected_kind
        assert descriptor.nbytes == 4
        assert descriptor.mem_span.buffer.nbytes == 4 * prod(hierarchy)
    formal = plan.buffer_map["stats"]
    actual = plan.buffer_map[dict(plan.call_map["first"].results)["stats"]]
    assert formal.distributed_type == actual.distributed_type == partial
    assert formal.strides == actual.strides
    assert dict(plan.call_map["second"].arguments)["forwarded"] == actual.id
    assert plan.buffer_map[dict(plan.call_map["second"].results)["forwarded"]].mem_span.must_alias(actual.mem_span)
