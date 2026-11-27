# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Pinned native softmax routing arithmetic with independent CUDA gold."""

import json
from pathlib import Path

import pytest
import torch

from triton.flagmega import ir as fm
from triton.flagmega.artifacts import write_artifact
from triton.flagmega.compiler import Compiler
from triton.flagmega.evaluator import DictWeightResolver, TorchEvaluator
from triton.flagmega.runtime import load

MODE = "exp_grouped_reciprocal"


def module(*, rows=3, experts=256, k=8):
    class Graph(fm.Module):
        def forward(self):
            logits = self.input("logits", fm.tensor_type("float32", (rows, experts)), id="logits")
            route = fm.F.nn.staged_routing(
                logits, fm.F.builtin.none(), k=k, index_dtype="int32", arithmetic=MODE)
            self.function("main", (logits,), fm.F.tensors.get_items(route, 0, 1))

    return Graph(dialect="high_level", stage="imported", entry="main").build()


def gold(device="cpu"):
    data = json.loads((Path(__file__).parent / "data/native_grouped_softmax_routing.json").read_text())
    return (
        torch.tensor(data["logits"], device=device),
        torch.tensor(data["weights"], device=device),
        torch.tensor(data["ids"], device=device, dtype=torch.int32),
    )


def test_grouped_softmax_contract_survives_edit_resume(tmp_path):
    graph = module()
    assert fm.load_module(fm.emit_module(graph, tmp_path / "source.py")) == graph


@pytest.mark.parametrize("experts", (1, 2, 4))
def test_grouped_softmax_short_rows_keep_selection_order(experts):
    logits = torch.zeros((1, experts))
    weights, ids = TorchEvaluator(DictWeightResolver({})).run(
        module(rows=1, experts=experts, k=experts), {"logits": logits})
    torch.testing.assert_close(weights, torch.full_like(weights, 1. / experts), rtol=0, atol=0)
    torch.testing.assert_close(ids, torch.arange(experts, dtype=torch.int32).unsqueeze(0))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_grouped_softmax_routing_matches_native_cuda(tmp_path):
    logits, expected, expected_ids = gold("cuda")
    artifact = write_artifact(
        Compiler().compile(module()).module, tmp_path / "artifact",
        target="nvidia-sm90", emit_executable=True)
    runtime = load(artifact, device="cuda:0")
    weights, ids = torch.empty_like(expected), torch.empty_like(expected_ids)
    try:
        runtime.prepare(logits, weights, ids)
        runtime.run_into(logits, weights, ids)
        torch.testing.assert_close(weights, expected, rtol=0, atol=0)
        torch.testing.assert_close(ids, expected_ids, rtol=0, atol=0)
        reports = runtime.resource_report
        if not isinstance(reports, tuple):
            reports = (reports,)
        assert all(report["spill_bytes"] == 0 for report in reports)
    finally:
        runtime.close()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_grouped_softmax_eager_matches_native_cuda():
    logits, expected, ids = gold("cuda")
    actual = TorchEvaluator(DictWeightResolver({})).run(module(), {"logits": logits})
    torch.testing.assert_close(actual[0], expected, rtol=0, atol=0)
    torch.testing.assert_close(actual[1], ids, rtol=0, atol=0)
