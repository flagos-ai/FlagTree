# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT

from dataclasses import replace
import hashlib
import json

from triton.flagmega import ir as fm
from triton.flagmega.diagnostics import DumpFlags, DumpManager
from triton.flagmega.passes import EGraphRulesPass, PassManager
from triton.flagmega.rules import RewriteRule


def _module():
    builder = fm.IRBuilder(dialect="high_level", stage="imported")
    value_type = fm.tensor_type("float32", (4,))
    lhs = builder.var("lhs", value_type, id="lhs")
    rhs = builder.var("rhs", value_type, id="rhs")
    output = builder.call("math.add", (lhs, rhs), value_type, id="output")
    builder.function("main", (lhs, rhs), (output,))
    return fm.verify_module(builder.build(entry="main"))


def test_egraph_lifecycle_rewrite_cost_and_pick_dumps(tmp_path):
    flags = DumpFlags.PASS_IR | DumpFlags.REWRITE | DumpFlags.EGRAPH_COST
    dumper = DumpManager(tmp_path, flags).root
    rule = RewriteRule(
        "commute",
        lambda node, _module: node.id == "output",
        lambda node, _module: replace(node, inputs=tuple(reversed(node.inputs))),
    )

    result = PassManager("dump", dumper=dumper).add(EGraphRulesPass("Rules", (rule,))).run(_module())

    assert result.executed == ("EGraphConstructPass", "Rules", "EGraphExtractPass")
    construct = tmp_path / "00_EGraphConstructPass"
    rules = tmp_path / "01_Rules"
    extract = tmp_path / "02_EGraphExtractPass"
    assert len(tuple((construct / "End").glob("V*.dot"))) == 1
    assert len(tuple((rules / "Start").glob("V*.dot"))) == 1
    assert len(tuple((rules / "End").glob("V*.dot"))) == 1
    assert len(tuple((rules / "Matches").glob("V*.txt"))) >= 1
    assert len(tuple((rules / "Rebuild").glob("V*.dot"))) >= 1
    assert len(tuple((extract / "Start").glob("V*.dot"))) == 1
    for name in ("Cost.dot", "Cost.txt", "Solve.txt", "Pick.dot", "Pick.txt"):
        assert (extract / "Costs" / name).is_file()
    assert "Status : OPTIMAL" in (extract / "Costs" / "Solve.txt").read_text(encoding="utf-8")
    assert "PICK" in (extract / "Costs" / "Pick.dot").read_text(encoding="utf-8")
    assert "model: structural-unit/v1" in (extract / "Costs" / "Pick.txt").read_text(encoding="utf-8")
    manifest = json.loads((tmp_path / "artifacts.json").read_text(encoding="utf-8"))
    pick = next(
        value for value in manifest["artifacts"]
        if value["relative_path"].endswith("02_EGraphExtractPass/Costs/Pick.txt")
    )
    assert pick["kind"] == "selection-report"
    assert pick["sha256"] == hashlib.sha256(
        (extract / "Costs" / "Pick.txt").read_bytes()
    ).hexdigest()
    assert (construct / "Before" / "main.py").is_file()
    assert (extract / "After" / "main.py").is_file()


def test_egraph_cost_flag_does_not_implicitly_enable_pass_ir(tmp_path):
    dumper = DumpManager(tmp_path, DumpFlags.EGRAPH_COST).root
    PassManager("cost-only", dumper=dumper).add(EGraphRulesPass("Rules", ())).run(_module())

    assert not (tmp_path / "00_EGraphConstructPass" / "End").exists()
    assert (tmp_path / "02_EGraphExtractPass" / "Costs" / "Cost.dot").is_file()
    assert (tmp_path / "02_EGraphExtractPass" / "Costs" / "Pick.dot").is_file()
