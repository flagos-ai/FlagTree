# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT

from triton.flagmega.diagnostics import DumpFlags, DumpManager, DumpScope
from triton.flagmega.passes.auto_distributed import AutoDistributedPass
from triton.flagmega.targets import NvidiaSm90Target

from .helpers import packed_matmul_module


def test_search_emits_nncase_graph_solve_cost_and_pick_dumps(tmp_path):
    root = DumpManager(tmp_path, DumpFlags.COMPILE | DumpFlags.EGRAPH_COST).root

    with DumpScope(root):
        AutoDistributedPass.run(packed_matmul_module(), NvidiaSm90Target())

    graph = (tmp_path / "Proposal" / "DistributedSearchGraph.dot").read_text(
        encoding="utf-8"
    )
    assert "shape=diamond" in graph
    assert "reshard internal" in graph
    for phase in ("Proposal", "Applied"):
        assert (tmp_path / phase / "DistributedSearchGraph.dot").is_file()
        assert (tmp_path / phase / "Costs" / "Solve.txt").is_file()
        assert (tmp_path / phase / "Costs" / "Pick.dot").is_file()
        assert (tmp_path / phase / "Costs" / "Pick.txt").is_file()
    assert "Status : OPTIMAL" in (
        tmp_path / "Applied" / "Costs" / "Solve.txt"
    ).read_text(encoding="utf-8")
    picks = (tmp_path / "Applied" / "Costs" / "Pick.txt").read_text(
        encoding="utf-8"
    )
    assert "distribution.output." in picks
    assert ".in_" in picks and ".out_" in picks
    assert "reason=matmul-" in picks
    assert "Reshards:" in picks
    assert " -> " in picks
