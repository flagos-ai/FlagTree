# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Original early checkpoints reject runtime adapters, including loop bodies."""

import pytest
from triton.flagmega import ir as fm
from triton.flagmega.errors import IRVerificationError
from triton.flagmega.ir.ops.core import construction_scope
from triton.flagmega.passes.g3_adapter_inventory import check_g3_adapters, g3_adapter_inventory
from triton.flagmega.stages import get_stage
from triton.flagmega.targets import NvidiaSm90Target


def adapter_graph(family, *, constant=False):

    class Graph(fm.Module):

        def forward(self):
            value_type = fm.tensor_type("float32", (4, 4))
            value = (fm.F.builtin.splat_const(value_type, 1., name="source") if constant else self.input(
                "source", value_type))
            if family == "Pack":
                result = fm.F.tensors.pack(value, (2, ), axes=(1, ), name="adapter")
            elif family == "Unpack":
                packed = fm.F.tensors.pack(value, (2, ), axes=(1, ), name="packed")
                result = fm.F.tensors.unpack(packed, axes=(1, ), name="adapter")
            elif family == "Cast":
                result = fm.F.tensors.cast(value, "bfloat16", name="adapter")
            elif family == "Slice":
                result = fm.F.tensors.slice(value, starts=(0, ), ends=(2, ), axes=(0, ), name="adapter")
            else:
                result = fm.F.tensors.concat(value, value, axis=0, name="adapter")
            self.function("main", () if constant else (value, ), (result, ))

    return Graph(dialect="high_level", stage="stats_combined", entry="main").build()


@pytest.mark.parametrize("family", ("Pack", "Slice", "Cast", "Concat", "Unpack"))
def test_original_gates_reject_runtime_boundaries_and_report_uses(family):
    source = adapter_graph(family)
    inventory = g3_adapter_inventory(source, checkpoint="02_AutoPackingPass/After")
    assert inventory["runtime_counts"][family] == 1
    with pytest.raises(IRVerificationError, match="G3.*producers=.*consumers="):
        get_stage("remove-unused-functions").run(source, NvidiaSm90Target())
    with pytest.raises(IRVerificationError, match="G3.*03_AutoDistributedPass/After"):
        check_g3_adapters(source, checkpoint="03_AutoDistributedPass/After")


def test_constant_island_uses_real_partition_proof_without_mutation():
    source = adapter_graph("Pack", constant=True)
    before = source.semantic_hash
    inventory = check_g3_adapters(source, checkpoint="02_AutoPackingPass/After")
    assert inventory["runtime_counts"]["Pack"] == 0
    assert inventory["offline_counts"]["Pack"] == 1
    assert inventory["adapters"][0]["offline_recipe"]["fingerprint"]
    assert source.semantic_hash == before and not source.constant_recipes


def test_loop_body_and_repeated_callee_are_not_hidden_by_entry_calls(tmp_path):
    builder = fm.IRBuilder(dialect="high_level", stage="stats_combined")
    tensor, index = fm.tensor_type("float32", (4, )), fm.tensor_type("int64", ())
    iv, carried = builder.var("iv", index), builder.var("carried", tensor)
    with construction_scope(builder):
        narrow = fm.F.tensors.cast(carried, "bfloat16", name="narrow")
        wide = fm.F.tensors.cast(narrow, "float32", name="wide")
    body = builder.region("body", (iv, carried), (wide, ), iter_count=1)
    x = builder.var("x", tensor)
    with construction_scope(builder):
        lo = fm.F.builtin.scalar_const(index, 0)
        hi = fm.F.builtin.scalar_const(index, 2)
        step = fm.F.builtin.scalar_const(index, 1)
        loop = fm.F.control.for_loop(lo, hi, step, x, body=body)
        output = fm.F.tensors.get_item(loop, 0)
    builder.function("main", (x, ), (output, ))
    source = builder.build(entry="main")
    inventory = g3_adapter_inventory(source, checkpoint="02_AutoPackingPass/After")
    assert inventory["reachable_functions"] == ["body", "main"]
    assert inventory["scalar_casts"] == 2
    restored = fm.load_module(fm.emit_module(source, tmp_path / "loop.py"))
    with pytest.raises(IRVerificationError, match="G3"):
        check_g3_adapters(restored, checkpoint="03_AutoDistributedPass/After")
