# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Block-FP8 projections must own M=1..5 owner-local rows in one kernel.

The decode path is a per-row GEMV: every weight tile is walked in the same K
order decode-1 uses, one activation row at a time.  A bucketed T>1 call must
therefore iterate its owner-local rows and never fold them into one larger
reduction, which would change every row's FP32 accumulation tree.
"""

import hashlib
from math import ceil
from pathlib import Path

import pytest

from triton.flagmega import ir as fm
from triton.flagmega.artifacts import write_artifact
from triton.flagmega.compiler import Compiler
from triton.flagmega.options import CompileOptions
from triton.flagmega.selection import apply_plan, override_plan

N, K = 56, 160

# Byte-identity guard for the M=1 render: the multi-row machinery must not
# change one line of the single-row kernel.  Update only with the template.
ROWS_ONE_SOURCE_SHA256 = (
    "8d01b154c506d653f1487a23fcd303701da29ac964c3d853fd53969e28d9cf02"
)


def linear_module(rows: int) -> fm.IRModule:
    class Graph(fm.Module):
        def forward(self):
            value = self.input("value", fm.tensor_type("bfloat16", (rows, K)), id="value")
            weight = self.input("weight", fm.tensor_type("float8_e4m3fn", (N, K)), id="weight")
            scale = self.input(
                "weight_scale",
                fm.tensor_type("float32", (ceil(N / 16), ceil(K / 128))),
                id="weight_scale",
            )
            output = fm.F.math.block_scaled_matmul(
                value, weight, scale, weight_block_n=16, weight_block_k=128,
                name="projection",
            )
            self.function("main", (value, weight, scale), (output,))

    return Graph(dialect="high_level", stage="imported", entry="main").build()


def build(tmp_path: Path, rows: int) -> Path:
    compiler = Compiler(CompileOptions())
    proposed = compiler.compile(linear_module(rows), stop_after="propose-tir").module
    choices = [
        (point.id, "tir.block_fp8.simt")
        for point in proposed.selection_points
        if any(candidate.id == "tir.block_fp8.simt" for candidate in point.candidates)
    ]
    assert choices, "the SIMT block-FP8 implementation must actually be tested"
    selected = apply_plan(proposed, override_plan(proposed, choices))
    return write_artifact(
        Compiler(CompileOptions()).compile(selected).module,
        tmp_path / f"rows{rows}", target="nvidia-sm90", emit_executable=True,
    )


def source(artifact: Path) -> str:
    return (artifact / "generated_kernels.py").read_text()


def test_single_row_render_is_byte_identical(tmp_path):
    rendered = source(build(tmp_path, 1))
    assert "linear_row" not in rendered
    assert hashlib.sha256(rendered.encode()).hexdigest() == ROWS_ONE_SOURCE_SHA256


@pytest.mark.parametrize("rows", [2, 3, 4, 5])
def test_multirow_projection_iterates_owner_local_rows(tmp_path, rows):
    rendered = source(build(tmp_path, rows))
    # One kernel owns every row: an explicit row loop, not one launch per row.
    assert f"for linear_row in tl.range(0, {rows}):" in rendered
    # Row r reads its own activation row and writes its own result row.
    assert f"(value + (((linear_row)) * {K}" in rendered
    assert f"destination_offsets = ((linear_row)) * {N}" in rendered
    # The row coordinate also gates the store mask.
    assert f"n_mask = (linear_row < ({rows}))" in rendered
    # The reduction contract is untouched: one row per accumulator, one
    # reduction per K tile, and rows are never folded into the K axis.
    assert "accumulator = tl.zeros((16,), tl.float32)" in rendered
    assert rendered.count("tl.sum(weight_value * source_value[None, :], axis=1)") == 1
    assert "tl.sum(weight_value * source_value[None, :], axis=0)" not in rendered


def test_multirow_beyond_five_rows_is_rejected(tmp_path):
    from triton.flagmega.errors import CodegenError

    with pytest.raises(CodegenError, match="at most 5 owner-local rows"):
        build(tmp_path, 6)


def _torch():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    return torch


def _target_for(torch) -> str:
    major, minor = torch.cuda.get_device_capability()
    return "nvidia-sm120" if (major, minor) == (12, 0) else "nvidia-sm90"


def _artifact_for(tmp_path, rows, target):
    compiler = Compiler(CompileOptions(target=target))
    proposed = compiler.compile(linear_module(rows), stop_after="propose-tir").module
    choices = [
        (point.id, "tir.block_fp8.simt")
        for point in proposed.selection_points
        if any(candidate.id == "tir.block_fp8.simt" for candidate in point.candidates)
    ]
    selected = apply_plan(proposed, override_plan(proposed, choices))
    return write_artifact(
        Compiler(CompileOptions(target=target)).compile(selected).module,
        tmp_path / f"gpu-rows{rows}", target=target, emit_executable=True,
    )


def _values(rows, torch, generator):
    value = torch.randn(rows, K, generator=generator).to(torch.bfloat16).cuda()
    weight = (
        torch.randn(N, K, generator=generator)
        .to(torch.float8_e4m3fn)
        .reshape(N, K // 32, 2, 16)
        .cuda()
    )
    scale = torch.randn(ceil(N / 16), ceil(K / 128), generator=generator).to(torch.float32).cuda()
    return value, weight, scale


def _run(artifact, value, weight, scale, torch):
    from triton.flagmega.runtime import load

    runtime = load(artifact, device="cuda:0")
    try:
        output = torch.empty(
            (value.shape[0], N), dtype=torch.bfloat16, device="cuda"
        )
        runtime.prepare(value, weight, scale, output=output)
        runtime.run_into(output, value, weight, scale)
        torch.cuda.synchronize()
        return output.clone()
    finally:
        runtime.close()


@pytest.mark.parametrize("rows", [2, 5])
def test_multirow_projection_is_bitwise_equal_to_sequential_rows(tmp_path, rows):
    torch = _torch()
    target = _target_for(torch)
    generator = torch.Generator().manual_seed(20261120)
    value, weight, scale = _values(rows, torch, generator)
    multi = _run(_artifact_for(tmp_path, rows, target), value, weight, scale, torch)
    single_artifact = _artifact_for(tmp_path, 1, target)
    sequential = []
    for row in range(rows):
        sequential.append(
            _run(single_artifact, value[row:row + 1].contiguous(), weight, scale, torch)
        )
    expected = torch.cat(sequential, 0)
    # Bitwise, not a tolerance: the multi-row kernel replays decode-1's
    # per-row K order, so every bit must agree.
    assert torch.equal(multi, expected), (
        (multi.float() - expected.float()).abs().max().item()
    )
