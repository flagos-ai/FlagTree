"""Membar index proofs must describe the values after integer operations."""
from pathlib import Path
import re

import pytest
import triton
from triton.backends.compiler import GPUTarget


COUNTEREXAMPLE = (Path(__file__).resolve().parents[4]
                  / "third_party/tle/test/GPU/membar-index-cast-wrap.mlir")


@pytest.mark.parametrize("case", ["trunc_extui", "trunc_extsi", "unknown", "depth", "i32_wrap"])
def test_index_coverage_retains_cross_warp_raw(tmp_path, monkeypatch, case):
    source = COUNTEREXAMPLE.read_text()
    if case == "trunc_extsi":
        # Signed i8 extension maps [128,256) to [-128,0). Adding 128
        # gives exactly the reader's [0,128), with different warp ownership.
        source = source.replace("start = 256 : i32, end = 384", "start = 128 : i32, end = 256")
        source = source.replace("arith.extui", "arith.extsi")
        source = source.replace("%two = arith.constant dense<2>", "%two = arith.constant dense<128>")
        source = source.replace("%i = arith.muli", "%i = arith.addi")
        source = source.replace("%two_r = arith.constant dense<2>", "%two_r = arith.constant dense<1>")
    elif case in ("unknown", "depth", "i32_wrap"):
        source = source.replace("%i = arith.muli %wide, %two", "%i = arith.muli %final, %two")
        ty = "tensor<128xi32, #write>"
        if case == "unknown":
            ops = f"%mask = arith.constant dense<127> : {ty}\n    %final = arith.andi %wr, %mask : {ty}"
        elif case == "depth":
            ops = f"%zero = arith.constant dense<0> : {ty}\n"
            previous = "%wr"
            for i in range(20):
                current = "%final" if i == 19 else f"%deep{i}"
                ops += f"    {current} = arith.addi {previous}, %zero : {ty}\n"
                previous = current
        else:
            # Arithmetic wraps in the intermediate, then cancels back to wr.
            ops = (f"%large = arith.constant dense<2147483647> : {ty}\n"
                   f"    %wrapped = arith.addi %wr, %large : {ty}\n"
                   f"    %final = arith.subi %wrapped, %large : {ty}")
        source = source.replace("    %two =", "    " + ops + "\n    %two =", 1)
    path = tmp_path / "counterexample.ttgir"
    path.write_text(source)
    monkeypatch.setenv("TRITON_CACHE_DIR", str(tmp_path / "fresh-cache"))
    compiled = triton.compile(str(path), target=GPUTarget("cuda", 120, 32),
                              options=dict(num_warps=4, num_ctas=1, ptx_options="--minnctapersm=1"))
    ptx = compiled.asm["ptx"]
    assert re.search(r"st\.shared[\s\S]*bar\.sync[\s\S]*ld\.shared", ptx)
    assert ptx.count("bar.sync") == 1
    assert compiled.metadata.shared == 4096
    assert compiled.metadata.static_shared == 0
