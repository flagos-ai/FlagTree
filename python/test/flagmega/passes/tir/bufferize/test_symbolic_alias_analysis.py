# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT

import pytest

from triton.flagmega import ir as fm
from triton.flagmega.errors import IRVerificationError
from triton.flagmega.passes.tir.bufferize import AliasAnalysis, AliasKind


def test_bounded_runtime_view_may_alias_each_static_layer_but_does_not_must_alias():
    analysis = AliasAnalysis()
    analysis.define("root", "state", 256)
    index = fm.dim("layer", minimum=0, maximum=3)
    analysis.add_alias("selected", "root", byte_offset=index * 64, nbytes=64, kind=AliasKind.VIEW)
    for layer in range(4):
        analysis.add_alias(str(layer), "root", byte_offset=layer * 64, nbytes=64, kind=AliasKind.VIEW)
        assert analysis.may_alias("selected", str(layer))
        assert not analysis.must_alias("selected", str(layer))
    assert not analysis.may_alias("0", "1")


def test_symbolic_tail_and_nested_subview_preserve_dimexpr_without_int_conversion():
    analysis = AliasAnalysis()
    analysis.define("root", "state", 256)
    offset = fm.dim("layer", minimum=0, maximum=3) * 64
    tail = analysis.add_alias("tail", "root", byte_offset=offset, kind=AliasKind.VIEW)
    assert tail.mem_span.size == 256 - offset
    nested = analysis.add_alias("first", "tail", nbytes=64, kind=AliasKind.VIEW)
    assert nested.mem_span.start == offset
    assert nested.mem_span.is_within(tail.mem_span)


@pytest.mark.parametrize("lower,upper", [(-1, 3), (0, 4), (None, 3), (0, None)])
def test_view_requires_a_proof_of_bounds_not_only_a_plausible_default(lower, upper):
    analysis = AliasAnalysis()
    analysis.define("root", "state", 256)
    with pytest.raises(IRVerificationError, match="exceeds source"):
        analysis.add_alias("selected", "root", byte_offset=fm.dim("layer", minimum=lower, maximum=upper) * 64,
                           nbytes=64, kind=AliasKind.VIEW)
