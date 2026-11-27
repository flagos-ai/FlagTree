# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""A chip-split GLU weight must reach a renderable descriptor candidate.

One `host_tensor_descriptor` spans the canonical weight shape, so it needs a
canonical global base address and obliges every chip to hold the whole weight.
A `descriptor_kind="table"` variant names one resident owner component per
entry, so it addresses whatever storage the weight already has.

Candidate selection and codegen have to agree on that. If the provider rejects
the single map for a chip-split weight while the renderer still demands
canonical storage from the table variant, a graph that used to compile stops
compiling instead of getting smaller.
"""

import tempfile
from pathlib import Path

from triton.flagmega import ir as fm
from triton.flagmega.codegen.triton import render_triton_package
from triton.flagmega.compiler import Compiler
from triton.flagmega.targets import NvidiaSm90Target

# A chip axis plus two block axes.
PLACEMENT = fm.Placement((4, 2, 2), "cyx", "cbb")

K = 2048
N = 4096


class _PackedGlu(fm.Module):
    """`nn.packed_dense_matmul_glu` whose N axis is split over `feature_axes`."""

    def __init__(self, feature_axes):
        super().__init__(
            dialect="ntt",
            stage="frozen_constants",
            entry="main",
            # `auto_distribution` is needed for SM90 package planning.  No
            # `launch_contract`: the pipeline installs that only in
            # `_finalize_tir_package`, so a fixture declaring it would let the
            # storage rule read a chip count proposal time cannot see.
            metadata={"auto_distribution": {"placement": PLACEMENT.to_data()}},
        )
        self.feature_axes = feature_axes

    def forward(self):
        broadcast = fm.SBP.broadcast()
        n_policy = fm.SBP.split_contiguous(self.feature_axes)
        value = self.input(
            "value",
            fm.DistributedType(
                fm.tensor_type("bfloat16", (1, K)), (broadcast, broadcast), PLACEMENT
            ),
        )
        weight_type = fm.DistributedType(
            fm.tensor_type("bfloat16", (K // 16, N // 8, 2, 64)),
            (broadcast, n_policy, broadcast, broadcast),
            PLACEMENT,
        )
        gate = self.input("gate_weight", weight_type)
        up = self.input("up_weight", weight_type)
        result = fm.F.nn.packed_dense_matmul_glu(
            value,
            gate,
            up,
            packed_layout="k_major_n8_k16",
            name="glu",
            metadata={
                "selected_vectorization": "vectorization.matmul.n",
                "selected_vector_axes": (1, ),
                "selected_vector_lanes": (8, ),
            },
        )
        self.function("main", (value, gate, up), (result, ))


def _descriptor_candidates(feature_axes):
    """Descriptor candidate ids proposed for the GLU node, by kind."""

    module = _PackedGlu(feature_axes).build()
    proposed = NvidiaSm90Target().propose_tir(module)
    point = next(point for point in proposed.selection_points if point.owner == "glu")
    single, table = set(), set()
    for candidate in point.candidates:
        if "tensor_descriptor" not in candidate.id:
            continue
        if candidate.parameters.get("descriptor_kind") == "table":
            table.add(candidate.id)
        else:
            single.add(candidate.id)
    return single, table


def test_chip_split_glu_drops_the_single_map_and_keeps_a_table():
    """Selection must leave the table variant, not just remove the single map."""

    single, table = _descriptor_candidates((0, 1, 2))
    assert not single, (
        f"a single tensor map replicates the whole GLU weight: {sorted(single)}"
    )
    assert table, "a chip-split GLU weight must still have a table candidate"


def test_block_only_glu_split_keeps_the_single_map():
    """A split that stays inside one chip has nothing to gain from a table.

    This pins the contrast so the test above cannot pass by removing every
    descriptor candidate.
    """

    single, _ = _descriptor_candidates((1, 2))
    assert single, "a chip-local GLU weight should still offer a single map"


def test_chip_split_glu_table_renders():
    """Codegen must accept the table variant the provider now steers onto.

    `_packed_dense_matmul_glu_call` required canonical-global weight storage for
    *both* descriptor kinds. Rejecting the single map without this carve-out
    turned a compiling graph into a `CodegenError`.
    """

    compiled = Compiler().compile(_PackedGlu((0, 1, 2)).build()).module
    with tempfile.TemporaryDirectory() as directory:
        render_triton_package(compiled, Path(directory) / "generated")
        source = (Path(directory) / "generated" / "generated_kernels.py").read_text()
    assert source, "the chip-split GLU package must render"
