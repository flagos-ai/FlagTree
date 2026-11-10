"""Hardware EXP2 is pure even when the LLVM middle end lacks MUSA IDs."""
import re

import pytest
import triton
import triton.language as tl
from test_tle_utils import compile_musa, require_mthreads_libtriton

require_mthreads_libtriton()


@triton.jit
def _exp2_memory_effects(x, out, BLOCK: tl.constexpr):
    i = tl.arange(0, BLOCK)
    tl.store(out + i, tl.exp2(tl.load(x + i)))


@pytest.mark.parametrize("width", [2, 4])
@pytest.mark.parametrize("native", [False, True])
def test_exp2_declaration_is_pure(width, native):
    compiled = compile_musa(
        _exp2_memory_effects,
        {"x": "*fp32", "out": "*fp32", "BLOCK": "constexpr"},
        {"BLOCK": 1024},
        options={"num_warps": 4, "exp2_vector_width": width, "exp2_native": native},
    )
    ir = compiled.asm["llir"]
    declaration = next(line for line in ir.splitlines()
                       if line.startswith("declare") and f"@llvm.musa.exp2.f.bst{width}(" in line)
    group = re.search(r"#(\d+)\s*$", declaration)
    assert group, declaration
    attributes = next(line for line in ir.splitlines() if line.startswith(f"attributes #{group[1]} ="))
    assert "memory(none)" in attributes or "readnone" in attributes
    assert "nounwind" in attributes
