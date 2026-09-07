# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT

import pytest

from triton.flagmega.errors import IRVerificationError
from triton.flagmega.passes.tir.bufferize import AliasAnalysis, AliasKind


def test_alias_analysis_distinguishes_must_alias_from_overlapping_views():
    analysis = AliasAnalysis()
    analysis.define("root", "workspace:0", 256)
    analysis.add_alias("inplace", "root")
    analysis.add_alias("low", "root", byte_offset=0, nbytes=128, kind=AliasKind.VIEW)
    analysis.add_alias("high", "root", byte_offset=128, nbytes=128, kind=AliasKind.VIEW)

    assert analysis.must_alias("root", "inplace")
    assert analysis.may_alias("root", "low")
    assert not analysis.must_alias("root", "low")
    assert not analysis.may_alias("low", "high")
    assert analysis.groups() == (("workspace:0", ("root", "inplace", "low", "high")),)


def test_alias_analysis_rejects_a_view_outside_the_source_range():
    analysis = AliasAnalysis()
    analysis.define("root", "workspace:0", 64)
    with pytest.raises(IRVerificationError, match="exceeds source"):
        analysis.add_alias("bad", "root", byte_offset=32, nbytes=64, kind=AliasKind.VIEW)
