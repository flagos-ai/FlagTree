# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT

import pytest

from triton.flagmega import ir as fm
from triton.flagmega.errors import IRSchemaError
from triton.flagmega.ir.ops.tir.buffer_subspan import BufferSubspan


@pytest.mark.parametrize("shape,offsets,result", [((3, 8), (1, 0), (2, 8)), ((1, 8), (0, 2), (1, 3)),
                                               ((2, 3, 8), (1, 1, 0), (1, 2, 8)), ((3, 8), (0, 0), (0, 8))])
def test_subspan_infers_contiguous_intervals(shape, offsets, result):
    source = fm.Node("x", "builtin.var", (), fm.tensor_type("float32", shape), attrs={"name": "x"})
    call = BufferSubspan.prepare((source,), {"shape": result, "offsets": offsets})
    assert call.result_type == fm.tensor_type("float32", result)


@pytest.mark.parametrize("offsets,shape", [((0, 2), (3, 3)), ((2, 0), (2, 8)), ((-1, 0), (1, 8)),
                                        ((False, 0), (1, 8)), ((0,), (1, 8))])
def test_invalid_or_strided_subspans_are_rejected(offsets, shape):
    source = fm.Node("x", "builtin.var", (), fm.tensor_type("float32", (3, 8)), attrs={"name": "x"})
    with pytest.raises(IRSchemaError):
        BufferSubspan.prepare((source,), {"shape": shape, "offsets": offsets})


def test_subspan_cannot_change_distributed_ownership():
    source_type = fm.DistributedType(fm.tensor_type("float32", (4, 8)),
                                     (fm.SBP.split_contiguous((0,)), fm.SBP.broadcast()),
                                     fm.Placement((2, 2), "yx", "bb"))
    source = fm.Node("x", "builtin.var", (), source_type, attrs={"name": "x"})
    with pytest.raises(IRSchemaError, match="ownership"):
        BufferSubspan.prepare((source,), {"shape": (1, 8), "offsets": (0, 0)})
