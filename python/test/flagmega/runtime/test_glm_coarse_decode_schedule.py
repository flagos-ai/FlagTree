# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""The ordinary workload target selects the measured complete decode stages."""

from pathlib import Path

import pytest

from triton.flagmega import ir as fm
from triton.flagmega.compiler import Compiler
from triton.flagmega.options import CompileOptions
from python.test.flagmega.codegen.triton.kernels.mhc.test_cooperative_contiguous_tf32 import graph as mhc_graph
from python.test.flagmega.codegen.triton.kernels.sparse_experts.test_cooperative_wgmma import stage_graph


@pytest.fixture
def target(monkeypatch):
    tutorial = Path(__file__).parents[3] / "tutorials/flagmega/04-glm5.3-flash-fp8/nvidia-h800"
    monkeypatch.syspath_prepend(str(tutorial))
    from local_optimizations.native_tp4 import NativeTp4DistributionTarget, native_implementation_model

    return NativeTp4DistributionTarget(triton_implementation_model=native_implementation_model())


def test_normal_catalog_registers_complete_k_and_preserves_the_original(target):
    catalog = target.triton_implementation_model
    complete = catalog.implementation("tir.sparse_experts_gate_up.complete_k_wgmma_n32")
    assert complete is not None
    assert complete.parameters["num_stages"] == 3
    assert catalog.implementation("tir.sparse_experts_gate_up.cooperative_wgmma") is not None
    assert catalog.implementation("tir.partitioned_sparse_experts_down.wgmma").parameters["output_tile_n"] == 256


def test_normal_mhc_catalog_declares_the_actual_publication_count(target):
    stream = target.triton_implementation_model.implementation("tir.mhc_pre.cooperative_stream_fma")
    assert stream.parameters.get("independent_collapse_pre") is True
    assert stream.facts["internal_grid_barriers"] == 1


def test_ordinary_compile_uses_cooperative_initial_tf32_with_unchanged_splits(target):
    source, placement = mhc_graph(hidden=512, splits=8)
    compiler = Compiler(CompileOptions(placement=placement))
    compiler.target = target.with_placement(placement)
    compiled = compiler.compile(source).module
    (dispatch,) = (d.dispatch for d in compiled.kernel_definitions if d.dispatch.semantic_op == "nn.mhc_pre")
    assert dispatch.microkernel.variant == "cooperative_contiguous_tf32"
    assert dispatch.semantic_attrs["projection_splits"] == 8
    assert dispatch.microkernel.facts["internal_grid_barriers"] == 2
    fm.verify_buffer_plan(compiled)


def test_ordinary_compile_uses_complete_k_without_projection_partials(target):
    placement = fm.Placement((4, 8, 16), "cyx", "cbb")
    source = stage_graph(placement=placement, packed=True, rows=2048, reduction=4096)
    compiler = Compiler(CompileOptions(placement=placement))
    compiler.target = target.with_placement(placement)
    compiled = compiler.compile(source).module
    (dispatch,) = (
        d.dispatch for d in compiled.kernel_definitions if d.dispatch.semantic_op == "nn.sparse_experts_gate_up"
    )
    assert dispatch.microkernel.variant == "complete_k_wgmma"
    assert dispatch.microkernel.parameters["task_n"] == 32
    assert dispatch.microkernel.parameters["num_stages"] == 3
    assert not any(workspace.name == "projection_partials" for workspace in dispatch.workspaces)
    fm.verify_buffer_plan(compiled)
