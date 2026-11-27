# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""A single tensor map must not be selected for a chip-split weight.

One `host_tensor_descriptor` spans the canonical weight shape, so its base
address has to be canonical global: `_canonical_global_values` then pins that
rdata root and every chip stores a whole copy. When the weight is split across
chips that copy is pure replication, and a descriptor *table* reads the same
weight from chip-local storage instead.

`DenseMatmulCandidateProvider` already enforces this through
`_supports_descriptor_weight_storage`. Any other provider that can select the
same `host_tensor_descriptor` implementations has to apply the same rule, or the
storage cost reappears through whichever op that provider owns. On the 45-layer
GLM-5.3 artifact the gap cost 2.1 GiB per chip through `ntt.matmul_norm_stats`.
"""

from triton.flagmega import ir as fm
from triton.flagmega.targets import NvidiaSm90Target

# Four chips over the `c` level, matching the GLM-5.3 launch placement that
# exposed this.
PLACEMENT = fm.Placement((4, 8, 16), "cyx", "cbb")

K = 4096
# 512 owners x 8 packed elements each: the local output tile is 64 scalars,
# which is the `tile_n` the descriptor implementations declare.
N = 32768


class _ChipSplitProjectionNormStats(fm.Module):
    """`ntt.matmul_norm_stats` whose packed weight is split across chips.

    The output axis carries the chip split, so a single canonical map over this
    weight would oblige all four chips to hold the complete tensor. The
    reduction axis stays broadcast to keep the projection free of a partial, so
    the fixture isolates the storage question from reduction materialization.
    """

    def __init__(self):
        # Deliberately *no* `launch_contract`: the real pipeline installs that
        # only in `_finalize_tir_package`, long after candidates are proposed.
        # A fixture that declares it would let the rule read a chip count the
        # pipeline cannot see, and pass while production stays broken.
        super().__init__(dialect="ntt", stage="frozen_constants", entry="main")

    def forward(self):
        broadcast = fm.SBP.broadcast()
        # Hierarchy axis 0 is the `c` level, so this policy splits the weight's
        # N axis across chips.  Covering every axis is what makes each owner's
        # statistics its own partial, which is the regime the descriptor
        # implementations declare (`statistics_kind="owner_partial"`).
        n_policy = fm.SBP.split_contiguous((0, 1, 2), 8)
        lhs = self.input(
            "lhs",
            fm.DistributedType(
                fm.tensor_type("bfloat16", (1, K)),
                (broadcast, broadcast),
                PLACEMENT,
            ),
            id="lhs",
        )
        rhs = self.input(
            "rhs",
            fm.DistributedType(
                fm.tensor_type(
                    fm.vector_type("bfloat16", (8, 2, 8)), (K // 16, N // 8)
                ),
                (broadcast, n_policy),
                PLACEMENT,
            ),
            id="rhs",
        )
        # The addend is the projection's own local shard: 8 bf16 lanes per
        # element, carrying the same N split.
        addend = self.input(
            "addend",
            fm.DistributedType(
                fm.tensor_type(fm.vector_type("bfloat16", (8, )), (1, N // 8)),
                (broadcast, n_policy),
                PLACEMENT,
            ),
            id="addend",
        )
        result = fm.F.ntt.matmul_norm_stats(
            lhs,
            rhs,
            addend,
            rhs_layout="k_major",
            axis=1,
            use_mean=False,
            name="projection",
            # The descriptor implementations declare an output-axis vector
            # contract; without the selected vectorization the synthetic matmul
            # reports `scalar` and no descriptor variant matches at all.
            metadata={
                "selected_vectorization": "vectorization.matmul.n",
                "selected_vector_axes": (1, ),
                "selected_vector_lanes": (8, ),
            },
        )
        self.function("main", (lhs, rhs, addend), (result, ))


def _single_map_candidates(point):
    """Candidate ids that pin a canonical weight base address.

    `descriptor_kind="table"` emits one map per resident owner and addresses
    chip-local storage, so only the single-map form is a problem here. The
    non-descriptor SIMT variant does not build a tensor map at all, so it is
    identified by name rather than by the absent `descriptor_kind`.
    """

    return {
        candidate.id
        for candidate in point.candidates
        if "tensor_descriptor" in candidate.id
        and candidate.parameters.get("descriptor_kind") != "table"
    }


def test_matmul_norm_stats_rejects_a_single_map_over_a_chip_split_weight():
    """The norm-stats provider must apply the descriptor storage rule too.

    `MatMulNormStatsCandidateProvider.propose` reaches the same
    `host_tensor_descriptor` implementations as `DenseMatmulCandidateProvider`
    but did not consult `_supports_descriptor_weight_storage`, so a chip-split
    weight still got a canonical single map through the fused epilogue op.
    """

    proposed = NvidiaSm90Target().propose_tir(_ChipSplitProjectionNormStats().build())
    point = next(
        point for point in proposed.selection_points if point.owner == "projection"
    )
    assert point.candidates, "the chip-split projection must still have candidates"
    offenders = _single_map_candidates(point)
    assert not offenders, (
        "a single tensor map forces every chip to store the whole weight: "
        f"{sorted(offenders)}"
    )
