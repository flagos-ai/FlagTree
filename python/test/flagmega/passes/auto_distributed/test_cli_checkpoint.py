# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT

import json
from dataclasses import replace

from triton.flagmega import ir as fm
from triton.flagmega.cli import main
from triton.flagmega.compiler import Compiler
from triton.flagmega.passes.auto_distributed import AutoDistributedPass
from triton.flagmega.targets import NvidiaSm90Target

from .helpers import packed_matmul_module


def test_cli_inspect_serializes_frozen_distributed_metadata(tmp_path, capsys):
    module = AutoDistributedPass.run(packed_matmul_module(), NvidiaSm90Target())
    checkpoint = fm.emit_module(module, tmp_path / "distributed.py")

    assert main(["inspect", str(checkpoint), "--json"]) == 0
    result = json.loads(capsys.readouterr().out)

    assert result["ok"] is True
    assert result["metadata"]["auto_distribution"]["placement"] == {
        "hierarchy": [8, 16],
        "hierarchy_levels": "bb",
        "name": "yx",
    }
    point = next(value for value in result["selection_points"] if value["id"] == "distribution.output")
    assert all(value["parameters"]["target_op"] == "math.packed_block_scaled_matmul" for value in point["alternatives"])


def test_cli_can_edit_proposal_then_apply_distribution(tmp_path, capsys):
    compiler = Compiler()
    current = replace(
        packed_matmul_module(), stage="unused_functions_removed"
    )
    proposed = compiler.run_stage(
        current,
        "propose-distribution",
        output=tmp_path / "distribution_candidates.py",
    ).module
    point = next(
        point
        for point in proposed.selection_points
        if point.id == "distribution.output"
    )
    plan = tmp_path / "selection.py"

    assert main([
        "select",
        str(tmp_path / "distribution_candidates.py"),
        "--point",
        point.id,
        "--candidate",
        point.default_candidate,
        "--rationale",
        "Measured choice retained by agent.",
        "--output",
        str(plan),
        "--json",
    ]) == 0
    capsys.readouterr()
    assert main([
        "stage",
        "auto-distributed",
        "--input",
        str(tmp_path / "distribution_candidates.py"),
        "--decisions",
        str(plan),
        "--output",
        str(tmp_path / "distributed.py"),
        "--json",
    ]) == 0
    result = json.loads(capsys.readouterr().out)
    distributed = fm.load_module(tmp_path / "distributed.py")

    assert result["module"]["stage"] == "distributed"
    assert distributed.selection_map[point.id].origin == "agent"
    assert any(
        node.op.startswith("distributed.") for node in distributed.nodes
    )
