"""gfx936 ALT reads retain the affine subview when folding row offsets."""
import re

import pytest
import triton
from triton._C import libtriton
from triton.backends.compiler import GPUTarget
from triton.experimental import gluon as g
from triton.experimental.gluon import language as gl
from triton.experimental.gluon._runtime import GluonASTSource

if not hasattr(libtriton, "hcu"):
    pytest.skip("HCU compiler required", allow_module_level=True)


@g.jit
def _read_slice(A, Out, START: gl.constexpr):
    blocked: gl.constexpr = gl.BlockedLayout([1, 8], [4, 16], [4, 1], [1, 0])
    shared: gl.constexpr = gl.SwizzledSharedLayout(32, 1, 4, [1, 0])
    fragment: gl.constexpr = gl.DistributedLinearLayout(
        [[1, 0], [2, 0], [4, 0], [0, 1], [0, 32], [0, 64]],
        [[0, 2], [0, 4], [0, 8], [0, 16], [8, 0], [16, 0]],
        [[0, 0], [0, 0]], [], [32, 128],
    )
    rows = gl.arange(0, 64, layout=gl.SliceLayout(1, blocked))
    cols = gl.arange(0, 128, layout=gl.SliceLayout(0, blocked))
    owner = gl.allocate_shared_memory(
        gl.bfloat16, [64, 128], shared, gl.load(A + rows[:, None] * 128 + cols[None, :])
    )
    values = owner.slice(START, 32).load(fragment)
    rows = gl.arange(0, 32, layout=gl.SliceLayout(1, fragment))
    cols = gl.arange(0, 128, layout=gl.SliceLayout(0, fragment))
    gl.store(Out + rows[:, None] * 128 + cols[None, :], values)


@pytest.mark.parametrize("start", [0, 32])
def test_alt_matrix_read_reuses_base_for_aligned_subslice(start):
    source = GluonASTSource(
        _read_slice, {"A": "*bf16", "Out": "*bf16", "START": "constexpr"},
        {"START": start},
        attrs={(i,): [("tt.divisibility", 16)] for i in range(2)},
    )
    kernel = triton.compile(source, target=GPUTarget("hip", "gfx936", 64),
                            options={"num_warps": 4})
    reads = re.findall(
        r"^\s*ds_read_m32x16_b16_alt\s+v\[\d+:\d+\],\s+(v\d+)(?:\s+offset:(\d+))?",
        kernel.asm["amdgcn"], re.M,
    )
    by_base = {}
    for base, offset in reads:
        by_base.setdefault(base, set()).add(int(offset or 0))
    assert len(reads) == 8
    assert len(by_base) == 4 and all(offsets == {0, 1024} for offsets in by_base.values())
