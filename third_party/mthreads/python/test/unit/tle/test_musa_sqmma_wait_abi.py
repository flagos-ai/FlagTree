"""Fail-closed ABI checks for the bundled MTGPU SQMMA wait intrinsic."""

from __future__ import annotations

import subprocess
from pathlib import Path


REPO = Path(__file__).resolve().parents[6]
LLC = REPO / "python/triton/backends/mthreads/bin/llc"
COMMON_ARGS = ("-march=mtgpu", "-mcpu=mp_31", "--opaque-pointers", "-O3", "-filetype=obj")


def _run_llc(tmp_path: Path, name: str, declaration: str, call: str):
    llir = tmp_path / f"{name}.ll"
    obj = tmp_path / f"{name}.o"
    llir.write_text(
        f"""define void @{name}() {{
entry:
  {call}
  ret void
}}

{declaration}
"""
    )
    return subprocess.run(
        [str(LLC), str(llir), *COMMON_ARGS, "-o", str(obj)],
        check=False,
        capture_output=True,
        text=True,
    ), obj


def test_parameterless_sqmma_wait_is_the_supported_abi(tmp_path):
    result, obj = _run_llc(
        tmp_path,
        "sqmma_wait0_probe",
        "declare void @llvm.musa.sqmma.wait()",
        "call void @llvm.musa.sqmma.wait()",
    )
    assert result.returncode == 0, result.stderr
    assert obj.stat().st_size > 0


def test_sqmma_wait_pending_count_is_rejected_by_bundled_llc(tmp_path):
    result, _ = _run_llc(
        tmp_path,
        "sqmma_wait1_probe",
        "declare void @llvm.musa.sqmma.wait(i32)",
        "call void @llvm.musa.sqmma.wait(i32 1)",
    )
    assert result.returncode != 0
    assert "Intrinsic has incorrect argument type" in result.stderr
