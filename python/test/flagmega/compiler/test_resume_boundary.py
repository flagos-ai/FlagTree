# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Resuming an already reached stop boundary must not consume later stages."""

import pytest

from triton.flagmega import ir as fm
from triton.flagmega.compiler import Compiler
from triton.flagmega.options import CompileOptions


@pytest.mark.parametrize("stop", ["decompose-gdn", "propose-distribution"])
@pytest.mark.parametrize("use_output_stage", [False, True])
def test_resume_at_existing_boundary_is_identity_and_emits_checkpoint(tmp_path, stop, use_output_stage):
    class Graph(fm.Module):
        def forward(self):
            x = self.input("x", fm.tensor_type("bfloat16", (1, 64)))
            self.function("main", (x,), (fm.F.math.silu(x),))

    original = Graph(dialect="high_level", stage="imported", entry="main").build()
    checkpoint = Compiler().compile(original, stop_after=stop).module
    source = fm.load_module(fm.emit_module(checkpoint, tmp_path / "input.py"))
    result = Compiler(CompileOptions(work_dir=tmp_path / "resume")).compile(
        source, stop_after=source.stage if use_output_stage else stop)
    assert result.module == source
    assert result.reports == ()
    assert fm.load_module(result.checkpoint) == source


def test_resume_before_vector_contract_lowering_does_not_skip_into_tir(monkeypatch):

    class Graph(fm.Module):

        def forward(self):
            value = self.input("value", fm.tensor_type("bfloat16", (1, 16)))
            self.function("main", (value, ), (fm.F.math.silu(value), ))

    original = Graph(dialect="ntt", stage="matmul_norm_stats_lowered", entry="main").build()

    def unexpected_tir(*args, **kwargs):
        pytest.fail("Resume skipped LowerVectorizationContracts and entered TIR selection")

    monkeypatch.setattr("triton.flagmega.codegen.triton.selection.TritonTirSelectionPolicy.propose", unexpected_tir)
    result = Compiler().compile(original, stop_after="lower-vectorization-contracts")
    assert result.module.stage == "vector_contracts_lowered"
    assert len(result.reports) == 1
    assert result.reports[0].stage == "AutoDistributedPass"
