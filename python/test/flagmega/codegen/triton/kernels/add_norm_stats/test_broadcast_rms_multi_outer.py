# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""The broadcast RMS device functions reduce each outer row independently."""

import pytest

from triton.flagmega.codegen.triton.templates import TritonTemplateRegistry


@pytest.mark.parametrize(
    "template",
    [
        "kernels/add_norm_stats/persistent_rms.py.jinja",
        "kernels/add_norm_stats/rms.py.jinja",
    ],
)
def test_broadcast_rms_reduces_each_outer_row_independently(template):
    # The dimensions before the normalized axis are independent statistics rows
    # (tokens). A >=2-outer-dim input must produce one statistic per outer row
    # over its own NORM_SIZE suffix, not one collapsed statistic over all
    # ELEMENTS (which overwrote stats[1:] with zeros). The device functions
    # iterate ELEMENTS // NORM_SIZE rows and store stats_output + norm_row.
    source = TritonTemplateRegistry().render(template, {"distributed_entry": False})
    assert "NORM_SIZE: tl.constexpr" in source
    assert "for norm_row in tl.range(0, ELEMENTS // NORM_SIZE)" in source
    assert "stats_output + norm_row" in source
    # The per-row reduction scopes offsets to the row's own suffix.
    assert "norm_row * NORM_SIZE" in source
