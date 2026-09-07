# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT

import pytest

from triton.flagmega.codegen.triton.templates import TritonTemplateRegistry
from triton.flagmega.codegen.triton.tir_package import _mesh_context


@pytest.mark.parametrize("hierarchy", [(2, 4), (2, 2, 2)])
def test_argmax_phases_keep_grid_scratch_in_entry(
    hierarchy, assert_entry_owned_phase_schedule,
):
    mesh = _mesh_context({
        "hierarchy": hierarchy,
        "hierarchy_levels": "b" * len(hierarchy),
        "name": "abc"[:len(hierarchy)],
    })
    source = TritonTemplateRegistry().render(
        "kernels/greedy_sample/_function.py.jinja",
        {
            **mesh,
            "distributed_entry": True,
            "render_calls": ({
                "symbol": "sample",
                "signature": "logits, maximum, index, result",
                "family": "greedy_sample",
                "variant": "distributed_argmax",
                "execution_kind": "collective",
                "internal_grid_barriers": 1,
                "vocab_size": 64,
                "tile": 8,
                "local_capacity": 8,
                "batch_capacity": 1,
                "result_capacity": 1,
                "partial_batch_logical": "0",
                "result_logical_batch": "0",
                "result_offset": "0",
                "result_active": "True",
                "result_writer": "shard_index == 0",
                "active": "True",
                "logits": "logits",
                "logits_offset": "shard_index * 8 + vocab_offsets",
                "logical_index": "shard_index * 8 + vocab_offsets",
                "partial_max": "maximum",
                "partial_index": "index",
                "result": "result",
            },),
        },
    )
    assert_entry_owned_phase_schedule(source, "sample")
