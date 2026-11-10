# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Positive owner-domain ceil division is identical in both code generators."""
import pytest
from triton.flagmega import ir as fm
from triton.flagmega.codegen.triton.kernel_call_renderers import _invert_split_axis
from triton.flagmega.codegen.cpp.expressions import translate


@pytest.mark.parametrize('extent,owners',[(24,8),(25,8),(192,2)])
def test_static_owner_capacity_uses_exact_positive_ceil(extent,owners):
    policy=fm.SBP.split_contiguous((0,)).to_data()
    local,mesh=_invert_split_axis('index',str(extent),policy,(owners,))
    assert '-(-' not in local
    translate(local)
    capacity=(extent+owners-1)//owners
    for index in range(extent):
        assert eval(local,{'__builtins__':{}},{'index':index})==index%capacity
        assert eval(mesh[0][1],{'__builtins__':{}},{'index':index})==index//capacity


def test_staged_active_owner_extents_keep_positive_ceil_arithmetic():
    policy=fm.SBP.split(fm.SplitStage.contiguous((0,)),fm.SplitStage.contiguous((1,))).to_data()
    local,mesh=_invert_split_axis('index','24',policy,(2,4))
    assert '-(-' not in local
    assert len(mesh)==2
    translate(local)
