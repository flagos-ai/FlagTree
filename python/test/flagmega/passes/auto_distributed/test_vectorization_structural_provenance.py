# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT

import pytest

from triton.flagmega import ir as fm
from triton.flagmega.passes.auto_distributed.policy import lower_vectorization_contracts


@pytest.mark.parametrize("adapter", ["boxing", "sharded_view"])
@pytest.mark.parametrize("inherited_internal", [False, True])
def test_layout_publication_is_not_reexecuted_as_its_semantic_provenance(adapter, inherited_internal):
    placement = fm.Placement((2, 2), "yx", "bb")
    tensor = fm.tensor_type("float32", (1, 1, 4))
    broadcast = fm.DistributedType(tensor, (fm.SBP.broadcast(),) * 3, placement)
    source_type = (
        fm.DistributedType(tensor, broadcast.axis_policies, placement, fm.SBP.partial((0, 1)))
        if adapter == "boxing" else
        fm.DistributedType(
            tensor, (fm.SBP.broadcast(), fm.SBP.broadcast(), fm.SBP.split_contiguous((0, 1))),
            placement,
        )
    )

    class Graph(fm.Module):
        def __init__(self):
            super().__init__(dialect="ntt", stage="norm_bindings_finalized", entry="main")

        def forward(self):
            stats = self.input("stats", source_type, id="stats")
            publication = getattr(fm.F.distributed, adapter)(
                stats, broadcast, name="published_stats", metadata={
                    "introduced_by": "PropagatePostAutoDistributedFunctionBoundaryLayouts",
                    "vectorized_from": "nn.norm_stats",
                    "vectorization_internal": inherited_internal,
                    "vectorization_candidate": "vectorization.norm_stats.reduction_axis",
                    "vectorization_attrs": {"axis": -1, "use_mean": False},
                    "vectorization_inputs": ("original_activation",),
                    "vector_axes": (1,),
                    "vector_lanes": (8,),
                },
            )
            self.function("main", (stats,), (publication,))

    original = Graph().build()
    lowered = lower_vectorization_contracts(original)

    fm.verify_module(lowered)
    assert lowered.node_map["published_stats"] == original.node_map["published_stats"]
    assert all(node.op != "nn.norm_stats" for node in lowered.nodes)
