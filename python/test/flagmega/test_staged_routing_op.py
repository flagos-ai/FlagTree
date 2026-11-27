# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""nn.staged_routing: both scorings, bias/tie semantics, and the executable kernel."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
import torch
from triton.flagmega.artifacts import write_artifact
from triton.flagmega.compiler import Compiler
from triton.flagmega.errors import CodegenError
from triton.flagmega.evaluator import DictWeightResolver, TorchEvaluator
from triton.flagmega.runtime import load

ROWS = 8
EXPERTS = 288
TOP_K = 8


def _routing_module(*, scoring: str, with_bias: bool, normalize: bool, scaling: float, experts: int = EXPERTS,
                    k: int = TOP_K, rows: int = ROWS):
    from triton.flagmega.ir import F, Module, tensor_type

    class Graph(Module):
        def forward(self):
            logits = self.input("logits", tensor_type("float32", (rows, experts)), id="logits")
            bias = self.input("bias", tensor_type("float32", (experts, )), id="bias") if with_bias \
                else F.builtin.none()
            routing = F.nn.staged_routing(logits, bias, k=k, axis=-1, index_dtype="int32", scoring=scoring,
                                          normalize=normalize, scaling=scaling, name="routing")
            weights, indices = F.tensors.get_items(routing, 0, 1, name_prefix="routing")
            self.function("main", ([logits, bias] if with_bias else [logits]), (weights, indices))

    return Graph(dialect="high_level", stage="imported", entry="main", metadata={"model": "staged-routing"}).build()


def _reference(logits, bias, *, scoring, normalize, scaling):
    scores = torch.softmax(logits.float(), -1) if scoring == "softmax" else torch.sigmoid(logits.float())
    choice = scores if bias is None else scores + bias.float()
    order = torch.argsort(choice, dim=-1, descending=True, stable=True)[..., :TOP_K]
    selected = scores.gather(-1, order)
    weights = selected
    if normalize:
        weights = weights / weights.sum(-1, keepdim=True)
    return weights * scaling, order.to(torch.int32)


@pytest.mark.parametrize("scoring,with_bias,normalize,scaling", [
    ("softmax", False, True, 1.0),   # Qwen3.5 staged routing
    ("softmax", True, True, 1.0),
    ("sigmoid", True, True, 2.5),    # GLM noaux_tc
    ("sigmoid", False, False, 1.0),
])
def test_staged_routing_evaluator_matches_reference(scoring, with_bias, normalize, scaling):
    module = _routing_module(scoring=scoring, with_bias=with_bias, normalize=normalize, scaling=scaling)
    generator = torch.Generator().manual_seed(3)
    logits = torch.randn(ROWS, EXPERTS, generator=generator)
    bias = torch.randn(EXPERTS, generator=generator) if with_bias else None
    outputs = TorchEvaluator(DictWeightResolver({})).run(
        module, {"logits": logits, **({"bias": bias} if with_bias else {})})
    expected_weights, expected_indices = _reference(logits, bias, scoring=scoring, normalize=normalize,
                                                    scaling=scaling)
    torch.testing.assert_close(outputs[0], expected_weights)
    torch.testing.assert_close(outputs[1], expected_indices)


def test_staged_routing_prefers_lower_index_on_ties():
    module = _routing_module(scoring="sigmoid", with_bias=False, normalize=False, scaling=1.0, experts=4, k=2,
                             rows=1)
    logits = torch.zeros(1, 4)
    outputs = TorchEvaluator(DictWeightResolver({})).run(module, {"logits": logits})
    # All-equal scores must select the first k ids in order.
    torch.testing.assert_close(outputs[1], torch.arange(2, dtype=torch.int32).unsqueeze(0))


def test_staged_routing_reports_unsupported_extent_at_candidate_selection():
    module = _routing_module(scoring="sigmoid", with_bias=False, normalize=True, scaling=1.0,
                             experts=2048)
    with pytest.raises(CodegenError, match="no complete local tile"):
        Compiler().compile(module)


@pytest.mark.cuda
def test_staged_routing_executable_matches_reference():
    module = _routing_module(scoring="sigmoid", with_bias=True, normalize=True, scaling=2.5)
    compiled = Compiler().compile(module).module
    artifact = write_artifact(compiled, Path(tempfile.mkdtemp(prefix="flagmega-routing-")), target="nvidia-sm90",
                              emit_executable=True)
    runtime = load(artifact, device="cuda:0")
    generator = torch.Generator().manual_seed(11)
    logits = torch.randn(ROWS, EXPERTS, generator=generator)
    bias = torch.randn(EXPERTS, generator=generator)
    weights = torch.empty((ROWS, TOP_K), dtype=torch.float32, device="cuda:0")
    indices = torch.empty((ROWS, TOP_K), dtype=torch.int32, device="cuda:0")
    runtime.prepare(logits.to("cuda:0"), bias.to("cuda:0"), weights, indices)
    runtime.run_into(logits.to("cuda:0"), bias.to("cuda:0"), weights, indices)
    torch.cuda.synchronize()
    expected_weights, expected_indices = _reference(logits, bias, scoring="sigmoid", normalize=True, scaling=2.5)
    torch.testing.assert_close(weights.cpu(), expected_weights, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(indices.cpu(), expected_indices)
