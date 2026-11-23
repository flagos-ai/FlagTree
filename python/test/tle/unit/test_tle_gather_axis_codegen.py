# Copyright 2025-     FlagOS Contributors
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
"""Gather row-offset facts survive the complete GPU compilation pipeline."""
import re

import pytest
import triton
import triton.language as tl
from triton._C.libtriton import ir
from triton.backends import backends
from triton.backends.compiler import GPUTarget
from triton.compiler import ASTSource

pytestmark = pytest.mark.skipif(not hasattr(ir.builder, "ensure_ttg_layout_attrs"), reason="requires __TLE__ build")
_TARGETS = [
    target for target in (GPUTarget("cuda", 90, 32), GPUTarget("hip", "gfx1100", 32)) if any(
        backend.compiler.supports_target(target) for backend in backends.values())
]


@triton.jit
def _gather_rows(source, indices, output):
    rows = tl.arange(0, 128)
    selected = tl.load(indices + rows)
    offsets = tl.gather(rows * 512, selected, axis=0)
    columns = tl.arange(0, 32)
    values = tl.load(tl.multiple_of(source, 16) + offsets[:, None] + columns[None, :])
    tl.store(output + rows[:, None] * 32 + columns[None, :], values)


@pytest.mark.parametrize("target", _TARGETS, ids=lambda target: target.backend)
def test_gather_row_offsets_preserve_vector_loads(target):
    compiled = triton.compile(
        ASTSource(_gather_rows, {"source": "*bf16", "indices": "*i32", "output": "*bf16"}, attrs={}),
        target=target,
        options={"num_warps": 4},
    )
    assembly = compiled.asm["ptx" if target.backend == "cuda" else "amdgcn"]
    if target.backend == "cuda":
        assert "ld.global.v4.b32" in assembly
    else:
        assert re.search(r"\b(?:buffer|global|flat)_load_b128\b", assembly)
