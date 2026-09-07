# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT

import json

from triton.flagmega import ir as fm
from triton.flagmega.diagnostics import DumpFlags, DumpManager, DumpScope
from triton.flagmega.passes.functions import (
    post_function_boundary_pack_propagation,
    propagate_function_boundary_layouts,
)
from triton.flagmega.targets import NvidiaSm90Target



class _DumpModule(fm.Module):
    def __init__(self):
        super().__init__(dialect="ntt", stage="packed", entry="main")

    def forward(self):
        value_type = fm.tensor_type("float32", (2, 8))
        parameter = self.input("parameter", value_type, id="parameter")
        packed = fm.F.tensors.pack(
            parameter, (4,), axes=(1,), name="packed")
        result = fm.F.tensors.unpack(packed, axes=(1,), name="result")
        value = self.input("value", value_type, id="value")
        call = fm.F.builtin.call(
            value, result_type=value_type, callee="layer", name="call")
        self.function("main", (value,), (call,))
        self.function("layer", (parameter,), (result,), attrs={"reusable": True})


def test_post_boundary_pass_uses_construct_rules_extract_and_dumps_pick(tmp_path):
    module = propagate_function_boundary_layouts(
        _DumpModule().build())
    root = DumpManager(
        tmp_path,
        DumpFlags.PASS_IR | DumpFlags.REWRITE | DumpFlags.EGRAPH_COST,
    ).root

    with DumpScope(root):
        post_function_boundary_pack_propagation(module, NvidiaSm90Target())

    assert tuple((tmp_path / "00_EGraphConstructPass" / "End").glob("V*.dot"))
    assert (
        tmp_path
        / "02_EGraphExtractPass"
        / "Costs"
        / "Pick.txt"
    ).is_file()
    entries = json.loads(
        (tmp_path / "artifacts.json").read_text(encoding="utf-8")
    )["artifacts"]
    assert any(
        entry["producer"] == "EGraphExtractor"
        and entry["kind"] == "selection-report"
        for entry in entries
    )
