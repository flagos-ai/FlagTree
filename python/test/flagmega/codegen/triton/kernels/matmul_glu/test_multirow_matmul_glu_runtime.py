# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""FP8 MatMulGlu must own M=1..5 owner-local rows without changing the reduction.

The decode GLU is a per-row GEMV. A bucketed T>1 call iterates its owner-local
rows, each replaying decode-1's K order; folding rows into one reduction would
silently change every row's FP32 accumulation tree.
"""

import hashlib
from pathlib import Path

import pytest

from triton.flagmega import ir as fm
from triton.flagmega.artifacts import write_artifact
from triton.flagmega.compiler import Compiler
from triton.flagmega.options import CompileOptions
from triton.flagmega.selection import apply_plan, override_plan

K = 128
N = 192

ROWS_ONE_SOURCE_SHA256 = (
    "4366b2c9308068c6eab69b9dd366e1e223c76b8f165f911a742a26033e0439d4"
)


def glu_module(rows: int) -> fm.IRModule:
    value = fm.tensor_type("bfloat16", (rows, K))
    weight = fm.tensor_type("float8_e4m3fn", (N, K))
    scale = fm.tensor_type("float32", (N // 16, 1))

    class Graph(fm.Module):
        def forward(self):
            x = self.input("value", value, id="value")
            gate = self.input("gate", weight, id="gate")
            up = self.input("up", weight, id="up")
            scales = self.input("scales", scale, id="scales")
            result = fm.F.nn.matmul_glu(
                x, gate, up, scales, scales, activation="silu",
                weight_block_n=16, weight_block_k=128, name="glu",
            )
            self.function("main", (x, gate, up, scales), (result,))

    return Graph(dialect="high_level", stage="imported", entry="main").build()


def _options(target):
    return CompileOptions() if target is None else CompileOptions(target=target)


def _select(module, target):
    proposed = Compiler(_options(target)).compile(module, stop_after="propose-tir").module
    choices = [
        (point.id, "tir.matmul_glu.simt")
        for point in proposed.selection_points
        if any(candidate.id == "tir.matmul_glu.simt" for candidate in point.candidates)
    ]
    assert choices, "the SIMT MatMulGlu implementation must actually be tested"
    return Compiler(_options(target)).compile(
        apply_plan(proposed, override_plan(proposed, choices))
    ).module


def build(tmp_path: Path, rows: int) -> Path:
    return write_artifact(
        _select(glu_module(rows), None), tmp_path / f"glu{rows}",
        target="nvidia-sm90", emit_executable=True,
    )


def source(artifact: Path) -> str:
    return (artifact / "generated_kernels.py").read_text()


def test_single_row_render_is_byte_identical(tmp_path):
    rendered = source(build(tmp_path, 1))
    assert "linear_row" not in rendered
    assert hashlib.sha256(rendered.encode()).hexdigest() == ROWS_ONE_SOURCE_SHA256


@pytest.mark.parametrize("rows", [2, 3, 4, 5])
def test_multirow_glu_iterates_owner_local_rows(tmp_path, rows):
    rendered = source(build(tmp_path, rows))
    assert f"for linear_row in tl.range(0, {rows}):" in rendered
    assert f"(value + (((linear_row)) * {K}" in rendered
    assert f"destination_offsets = ((linear_row)) * {N}" in rendered
    assert f"n_mask = (linear_row < ({rows}))" in rendered
    # Same reduction contract as decode-1: one row per accumulator and the
    # gate/up dot is still reduced along K, never along rows.
    assert "tl.sum(" in rendered
    assert "axis=1" in rendered
    assert "axis=0" not in rendered


def test_multirow_beyond_five_rows_is_rejected(tmp_path):
    from triton.flagmega.errors import CodegenError

    with pytest.raises(CodegenError, match="at most 5 owner-local rows"):
        build(tmp_path, 6)


def _torch():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    return torch


def _target(torch) -> str:
    return "nvidia-sm120" if torch.cuda.get_device_capability() == (12, 0) else "nvidia-sm90"


def _values(rows, torch, generator):
    def packed():
        return (
            torch.randn(N, K, generator=generator)
            .to(torch.float8_e4m3fn)
            .reshape(N, K // 32, 2, 16)
            .cuda()
        )

    value = torch.randn(rows, K, generator=generator).to(torch.bfloat16).cuda()
    scales = torch.rand(N // 16, 1, generator=generator).to(torch.float32).cuda()
    return value, packed(), packed(), scales


def _run(artifact, value, gate, up, scales, torch):
    from triton.flagmega.runtime import load

    runtime = load(artifact, device="cuda:0")
    try:
        output = torch.empty((value.shape[0], N), dtype=torch.bfloat16, device="cuda")
        runtime.prepare(value, gate, up, scales, output=output)
        runtime.run_into(output, value, gate, up, scales)
        torch.cuda.synchronize()
        return output.clone()
    finally:
        runtime.close()


@pytest.mark.parametrize("rows", [2, 5])
def test_multirow_glu_is_bitwise_equal_to_sequential_rows(tmp_path, rows):
    torch = _torch()
    target = _target(torch)
    generator = torch.Generator().manual_seed(20261120)
    value, gate, up, scales = _values(rows, torch, generator)
    multi = _run(
        write_artifact(_select(glu_module(rows), target), tmp_path / f"gpu{rows}",
                       target=target, emit_executable=True),
        value, gate, up, scales, torch,
    )
    single = write_artifact(_select(glu_module(1), target), tmp_path / "gpu1",
                            target=target, emit_executable=True)
    expected = torch.cat([
        _run(single, value[row:row + 1].contiguous(), gate, up, scales, torch)
        for row in range(rows)
    ], 0)
    assert torch.equal(multi, expected), (
        (multi.float() - expected.float()).abs().max().item()
    )
