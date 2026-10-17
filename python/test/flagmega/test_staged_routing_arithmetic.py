# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Explicit routing arithmetic, independent pinned CUDA gold, and old defaults."""

from pathlib import Path

import pytest
import torch
from triton.flagmega import ir as fm
from triton.flagmega.artifacts import write_artifact
from triton.flagmega.codegen.triton.candidates.core import TritonCandidateContext
from triton.flagmega.codegen.triton.candidates.routing import StagedRoutingCandidateProvider
from triton.flagmega.compiler import Compiler
from triton.flagmega.errors import CodegenError, IRSchemaError
from triton.flagmega.evaluator import DictWeightResolver, TorchEvaluator
from triton.flagmega.runtime import load
from triton.flagmega.targets import NvidiaSm90Target
from triton.flagmega.targets.iluvatar_bi_v150 import IluvatarBiV150Target

MODE = "tanh_scaled_reciprocal"
# Selected logits from a real E288 router, with synthetic biases fixing order.
# Gold: unmodified vLLM 0.31.0 grouped_topk, E288/k8/renormalize/scale2.5,
# CUDA 13.0, SM90. No checkpoint or vLLM dependency is needed to reproduce.
LOGITS = [-0.08202171325683594, -0.7358289361000061, -0.3879976272583008, -1.284753441810608,
          -1.4504410028457642, -0.8667342066764832, -1.9413106441497803, -2.0344033241271973]
GOLD = [0.5572090148925781, 0.3764069676399231, 0.46969908475875854, 0.2518647611141205,
        0.2207120656967163, 0.3438898026943207, 0.1458415985107422, 0.13437670469284058]


def module(arithmetic=None, *, normalize=True, scoring="sigmoid", k=8):
    class Graph(fm.Module):
        def forward(self):
            value = self.input("logits", fm.tensor_type("float32", (1, 288)), id="logits")
            bias = self.input("bias", fm.tensor_type("float32", (288,)), id="bias")
            attrs = {} if arithmetic is None else {"arithmetic": arithmetic}
            routing = fm.F.nn.staged_routing(value, bias, k=k, index_dtype="int32", scoring=scoring,
                                             normalize=normalize, scaling=2.5, **attrs)
            outputs = fm.F.tensors.get_items(routing, 0, 1)
            self.function("main", (value, bias), outputs)
    return Graph(dialect="high_level", stage="imported", entry="main").build()


def inputs():
    return {"logits": torch.tensor([LOGITS + [-100.0] * 280]),
            "bias": torch.tensor(list(range(8, 0, -1)) + [-100.0] * 280)}


def test_default_attributes_and_hash_are_unchanged():
    old, explicit = module(), module("standard")
    assert old.semantic_hash == explicit.semantic_hash
    assert all("arithmetic" not in n.attrs for n in explicit.nodes if n.op == "nn.staged_routing")
    assert module(MODE).semantic_hash != old.semantic_hash


def test_arithmetic_survives_python_edit_resume(tmp_path):
    graph = module(MODE)
    path = tmp_path / "routing.py"
    fm.emit_module(graph, path)
    restored = fm.load_module(path)
    assert restored.semantic_hash == graph.semantic_hash
    assert next(n for n in restored.nodes if n.op == "nn.staged_routing").attrs["arithmetic"] == MODE


@pytest.mark.parametrize("attrs", [{"arithmetic": "unknown"}, {"arithmetic": MODE, "scoring": "softmax"},
                                  {"arithmetic": MODE, "k": 33}])
def test_unsupported_arithmetic_is_rejected(attrs):
    with pytest.raises(IRSchemaError, match="arithmetic|sigmoid scoring"):
        module(**attrs)


@pytest.mark.parametrize("normalize", [True, False])
def test_explicit_evaluator_formula_and_underflow(normalize):
    graph, values = module(MODE, normalize=normalize), inputs()
    actual = TorchEvaluator(DictWeightResolver({})).run(graph, values)
    selected = (torch.tanh(values["logits"][..., :8] * 0.5) * 0.5 + 0.5)
    total = selected
    for half in (4, 2, 1):
        total = total[..., :half] + total[..., half:]
    scale = torch.full_like(total, 2.5) / (total + 1e-20) if normalize else 2.5
    torch.testing.assert_close(actual[0], selected * scale, rtol=0, atol=0)
    torch.testing.assert_close(actual[1], torch.arange(8, dtype=torch.int32).reshape(1, 8))
    values["logits"].fill_(-100.0)
    underflow = TorchEvaluator(DictWeightResolver({})).run(graph, values)
    assert torch.isfinite(underflow[0]).all() and torch.count_nonzero(underflow[0]) == 0


@pytest.mark.parametrize("arithmetic", ["standard", MODE])
@pytest.mark.parametrize("target_type", [NvidiaSm90Target, IluvatarBiV150Target])
def test_candidate_platform_contract(arithmetic, target_type):
    graph, target = module(arithmetic), target_type()
    node = next(n for n in graph.nodes if n.op == "nn.staged_routing")
    context = TritonCandidateContext(graph, target, {}, {}, {}, frozenset(), True,
                                    target.triton_implementation_model)
    if arithmetic == MODE and target_type is IluvatarBiV150Target:
        with pytest.raises(CodegenError, match="CUDA 32-lane"):
            StagedRoutingCandidateProvider().propose(node, context)
    else:
        assert StagedRoutingCandidateProvider().propose(node, context).candidates


@pytest.mark.cuda
def test_explicit_cuda_routing_matches_pinned_gold(tmp_path: Path):
    compiled = Compiler().compile(module(MODE)).module
    artifact = write_artifact(compiled, tmp_path / "artifact", target="nvidia-sm90", emit_executable=True)
    runtime = load(artifact, device="cuda:0")
    try:
        values = {name: value.cuda() for name, value in inputs().items()}
        outputs = (torch.empty((1, 8), device="cuda"), torch.empty((1, 8), device="cuda", dtype=torch.int32))
        args = (values["logits"], values["bias"], *outputs)
        runtime.prepare(*args)
        runtime.run_into(*args)
        torch.testing.assert_close(outputs[0].cpu(), torch.tensor([GOLD]), rtol=0, atol=0)
        torch.testing.assert_close(outputs[1].cpu(), torch.arange(8, dtype=torch.int32).reshape(1, 8))
        assert runtime.resource_report["spill_bytes"] == 0
    finally:
        runtime.close()
