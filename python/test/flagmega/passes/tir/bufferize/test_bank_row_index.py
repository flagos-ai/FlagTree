# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Bank indexing must select storage rather than silently read row zero."""

import pytest
from triton.flagmega import ir as fm
from triton.flagmega.errors import CodegenError, IRVerificationError
from triton.flagmega.passes.tir.bufferize import plan_buffers


def bank_read(index):
    b = fm.IRBuilder(dialect="semantic_tir", stage="canonicalized_tir")
    ty = fm.tensor_type("float32", (1, 16))
    schema = fm.ParameterBankSchema("rows", (fm.BankField("value", ty, "checkpoint"), ))
    bank = b.node(op="tir.parameter_bank", type=schema.type, id="bank",
                  attrs={"schema": schema, "source": "fixture", "rows": (("row0", ), ("row1", ), ("row2", ))})
    if index is None:
        idx = b.var("index", fm.tensor_type("int32", ()), id="index")
    else:
        idx = b.node(op="tir.scalar_const", type=fm.tensor_type("int32", ()), id="index", attrs={"value": index})
    row = b.call("tir.bank_get", (bank, idx), schema.element_type, id="row")
    value = b.call("builtin.get_item", (row, ), ty, id="value", attrs={"index": 0})
    x = b.var("x", ty, id="x")
    # A real data consumer, not just an unobserved metadata node.
    out = b.call("math.add", (value, x), ty, id="out")
    b.function("main", (x, idx) if index is None else (x, ), (out, ))
    return fm.verify_module(b.build(entry="main"))


@pytest.mark.parametrize("index", [0, 1, 2])
def test_constant_bank_get_selects_exact_physical_row(index):
    p = plan_buffers(bank_read(index))
    values = dict(p.function_map["main"].values)
    assert p.buffer_map[values["value"][0]].weight_key == f"row{index}"


@pytest.mark.parametrize("index", [-1, 3])
def test_constant_bank_get_rejects_outside_stored_domain(index):
    with pytest.raises(IRVerificationError, match="bank.*index|Bank.*index"):
        plan_buffers(bank_read(index))


def test_dynamic_direct_bank_leaf_cannot_silently_read_row_zero():
    with pytest.raises((CodegenError, IRVerificationError), match="dynamic bank|Dynamic bank"):
        plan_buffers(bank_read(None))
