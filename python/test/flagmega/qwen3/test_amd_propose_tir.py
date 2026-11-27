# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""FlagMega-on-Radeon Target T1: a real multi-op bf16 Qwen3 model (embedding,
RMS norm, rotary, fused QKV, paged attention, MLP, matmul) reaches
--stop-after propose-tir on amd-gfx1100 / amd-gfx1201 using the NV-reused
planner services. Every selection point keeps at least one AMD-capability
supported candidate -- the bf16 catalog has no gfx1100 coverage hole."""

import pytest

from triton.flagmega.compiler import Compiler
from triton.flagmega.options import CompileOptions
from triton.flagmega.importer import import_qwen3_model

from .helpers import full_checkpoint

AMD_TARGETS = ("amd-gfx1100", "amd-gfx1201")


@pytest.mark.parametrize("target_name", AMD_TARGETS)
def test_qwen3_bf16_reaches_propose_tir_on_amd(target_name):
    module = import_qwen3_model(full_checkpoint(num_hidden_layers=2))

    result = Compiler(CompileOptions(target=target_name)).compile(
        module, stop_after="propose-tir"
    ).module

    assert result.stage == "selected_tir_variants"

    points = list(result.selection_points)
    assert points
    empty = [point.id for point in points if not point.candidates]
    assert not empty, (
        f"{target_name}: selection points with no capability-supported "
        f"candidate: {empty}"
    )
