# Copyright 2026 FlagOS Contributors

"""Pass-level regression for repeated transitive-slice construction.

No GPU or model is needed: a load-dependent pointer chain is sufficient. The
timeout is a generous complexity guard, not a microbenchmark latency target.
"""

import os
from pathlib import Path
import subprocess

import pytest


def _triton_opt():
    configured = os.environ.get("TRITON_OPT_PATH")
    if configured:
        binary = configured
    else:
        root = Path(__file__).resolve().parents[4]
        candidates = sorted(root.glob("build/cmake.*/bin/triton-opt"))
        if not candidates:
            pytest.skip("requires a local triton-opt build")
        binary = str(candidates[0])
    help_text = subprocess.run(
        [binary, "--help"], capture_output=True, text=True, timeout=10, check=True,
    ).stdout
    if "tle-lower" not in help_text:
        pytest.skip("requires a TLE-enabled triton-opt build")
    return binary


def pointer_chain(count):
    tensor = "tensor<128xi32, #blocked>"
    pointers = "tensor<128x!tt.ptr<i32>, #blocked>"
    lines = [
        "#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], "
        "warpsPerCTA = [4], order = [0]}>",
        'module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, '
        'ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {',
        "tt.func @load_dependent_chain(%base: !tt.ptr<i32>) {",
        f"%ptr = tt.splat %base : !tt.ptr<i32> -> {pointers}",
        f"%i = tt.make_range {{start = 0 : i32, end = 128 : i32}} : {tensor}",
        f"%p0 = tt.addptr %ptr, %i : {pointers}, {tensor}",
        f"%v0 = tt.load %p0 : {pointers}",
    ]
    for index in range(1, count):
        lines.extend([
            f"%p{index} = tt.addptr %p{index - 1}, %v{index - 1} : {pointers}, {tensor}",
            f"%v{index} = tt.load %p{index} : {pointers}",
        ])
    lines.extend(["tt.return", "}", "}"])
    return "\n".join(lines)


def test_coalesce_long_load_dependent_chain():
    result = subprocess.run(
        [_triton_opt(), "-tritongpu-coalesce"], input=pointer_chain(512),
        capture_output=True, text=True, timeout=30, check=True,
    )
    assert result.stdout.count("tt.load") == 512
    assert result.stdout.count("tt.addptr") == 512
    assert "sizePerThread = [1]" in result.stdout
    # Coalesce emits pointer conversions even when the layout is unchanged;
    # canonicalization is deliberately not part of this pass-level test.
    conversions = [line for line in result.stdout.splitlines() if "ttg.convert_layout" in line]
    assert len(conversions) == 512
    for line in conversions:
        source, target = line.split(" : ", 1)[1].split(" -> ")
        assert source == target
