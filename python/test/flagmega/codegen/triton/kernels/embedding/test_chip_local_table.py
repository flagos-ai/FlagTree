# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""A chip-split embedding feature axis must not force a canonical table.

The decode lookup reads `weight[token_id, feature]`. The *vocabulary* axis is
genuinely global: a token id is a runtime value, so a chip that owns only some
rows cannot answer for the rest, which is why
`DistributedEmbeddingProvider` keeps that axis broadcast unconditionally. The
*feature* axis is different -- the provider projects it exactly onto the
output's feature policy, so each owner reads only the features it also writes
and nothing crosses a chip boundary.

`canonical_global_arguments = ("weight",)` pins the whole table regardless, so
on a 4-chip launch every chip stores a full copy. For GLM-5.3 that table is
1.18 GiB against a 2.68 GiB per-chip overflow.
"""

import pytest

from triton.flagmega import ir as fm
from triton.flagmega.compiler import Compiler
from triton.flagmega.ir import DistributedBufferStorageKind as StorageKind

# A chip axis plus two block axes, matching the GLM-5.3 launch placement.
PLACEMENT = fm.Placement((4, 2, 2), "cyx", "cbb")

VOCAB = 17
FEATURES = 64


def _embedding_module(*, feature_axes):
    """`nn.embedding` whose feature axis is split over `feature_axes`.

    The weight and output carry the same feature policy, which is the contract
    the distributed provider enumerates.
    """

    class Embedding(fm.Module):

        def forward(self):
            policy = fm.SBP.split_contiguous(feature_axes)
            indices = self.input(
                "indices",
                fm.DistributedType(
                    fm.tensor_type("int32", (1, )), (fm.SBP.broadcast(), ), PLACEMENT
                ),
            )
            weight = self.input(
                "weight",
                fm.DistributedType(
                    fm.tensor_type("bfloat16", (VOCAB, FEATURES)),
                    (fm.SBP.broadcast(), policy),
                    PLACEMENT,
                ),
            )
            output = fm.F.nn.embedding(indices, weight, padding_idx=0)
            self.function("main", (indices, weight), (output, ))

    return Embedding(
        dialect="distributed",
        stage="frozen_constants",
        entry="main",
        metadata={
            "auto_distribution": {"placement": PLACEMENT.to_data()},
            "launch_contract": {"grid_mesh": PLACEMENT.to_data()},
        },
    ).build()


def _weight_buffer(plan):
    """The plan's descriptor for the embedding table.

    Bufferization renames values, so identify the table by shape rather than by
    name: it is the only `VOCAB x FEATURES` buffer.
    """

    matches = [
        buffer
        for buffer in plan.buffers
        if tuple(buffer.shape) == (VOCAB, FEATURES)
    ]
    assert len(matches) == 1, (
        f"expected exactly one {VOCAB}x{FEATURES} table, got "
        f"{[buffer.id for buffer in matches]}"
    )
    return matches[0]


def test_chip_split_feature_axis_keeps_the_table_chip_local():
    """Splitting features across chips must not replicate the whole table.

    Hierarchy axis 0 is the `c` level, so this policy gives each chip a
    quarter of every row. Storing it `CANONICAL_GLOBAL` would put the complete
    table on all four chips.
    """

    from triton.flagmega.ir import verify_buffer_plan

    compiled = Compiler().compile(_embedding_module(feature_axes=(0, 1, 2))).module
    weight = _weight_buffer(verify_buffer_plan(compiled))
    assert weight.distributed_storage_kind is not StorageKind.CANONICAL_GLOBAL, (
        "a chip-split embedding table must not be stored canonical global: "
        f"{weight.id} is {weight.distributed_storage_kind}"
    )


def test_block_only_feature_split_still_uses_the_canonical_table():
    """A split that stays inside one chip has nothing to gain from compaction.

    Every chip already owns the complete table here, so this pins the contrast
    and keeps the test above from passing for the wrong reason.
    """

    from triton.flagmega.ir import verify_buffer_plan

    compiled = Compiler().compile(_embedding_module(feature_axes=(1, 2))).module
    weight = _weight_buffer(verify_buffer_plan(compiled))
    assert weight.distributed_storage_kind is StorageKind.CANONICAL_GLOBAL


def test_chip_local_table_is_addressed_without_the_chip_coordinate():
    """A compact table must be indexed by local features, not global ones.

    Each chip holds only its own feature slice, so its row stride is the local
    width and its base offset selects the block owner inside that slice. The
    chip coordinate must not appear: the table is already chip-private, and
    including it would address past the end of the segment.

    A device test cannot cover this -- a `c`-level mesh needs a 4-rank
    launcher -- so the generated addressing is checked directly.
    """

    import re
    import tempfile
    from pathlib import Path

    from triton.flagmega.codegen.triton import render_triton_package

    compiled = Compiler().compile(_embedding_module(feature_axes=(0, 1, 2))).module
    with tempfile.TemporaryDirectory() as directory:
        render_triton_package(compiled, Path(directory) / "generated")
        source = (Path(directory) / "generated" / "generated_kernels.py").read_text()
    loads = [line for line in source.splitlines() if "(token_id) *" in line]
    assert len(loads) == 1, f"expected one table load, got {loads}"
    load = loads[0]
    # 16 block owners share each row of 64 features.
    assert "(token_id) * 4" in load, f"row stride is not the local width: {load}"
    assert "shard_coord0" not in load, (
        f"a chip-private table must not be indexed by the chip axis: {load}"
    )
    assert re.search(r"shard_coord[12]", load), (
        f"the block owner's slice must still be selected: {load}"
    )
