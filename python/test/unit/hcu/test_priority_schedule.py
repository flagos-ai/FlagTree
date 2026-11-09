"""Actual target lowering tests for the first-MMAC priority boundary."""
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

from triton.experimental.gluon.language.amd import AMDMFMALayout
from triton.experimental.gluon.language.amd.cdna3 import mfma
from triton.experimental.gluon.language.hcu import accumulator_fence


@g.jit
def _priority_dot(A, B, C, SIDE, M: gl.constexpr, EXTRA_USE: gl.constexpr):
    load: gl.constexpr = gl.BlockedLayout([1, 4], [4, 16], [2, 2], [1, 0])
    mma: gl.constexpr = AMDMFMALayout(
        3, [16, 16, 16], True, [2, 2], tiles_per_warp=[1, 1], mmac_layout=1
    )
    x = gl.arange(0, M, layout=gl.SliceLayout(1, load))
    k = gl.arange(0, 32, layout=gl.SliceLayout(0, load))
    a = gl.load(A + x[:, None] * 32 + k[None, :])
    b = gl.permute(gl.load(B + x[:, None] * 32 + k[None, :]), (1, 0))
    a = gl.convert_layout(a, gl.DotOperandLayout(0, mma, 8))
    b = gl.convert_layout(b, gl.DotOperandLayout(1, mma, 8))
    acc = gl.full((M, M), 0, gl.float32, mma)
    acc = accumulator_fence(acc, 1)
    if EXTRA_USE:
        gl.store(SIDE, gl.sum(gl.sum(acc, 0), 0))
    acc = mfma(a, b, acc)
    acc = accumulator_fence(acc, 0)
    out: gl.constexpr = gl.BlockedLayout([1, 4], [4, 16], [2, 2], [1, 0])
    i = gl.arange(0, M, layout=gl.SliceLayout(1, out))
    j = gl.arange(0, M, layout=gl.SliceLayout(0, out))
    gl.store(C + i[:, None] * M + j[None, :], gl.convert_layout(acc, out))


@pytest.mark.parametrize("dtype", ["fp16", "bf16"])
@pytest.mark.parametrize("tile", [32, 64])
@pytest.mark.parametrize("defer,extra_use", [(False, False), (True, False), (True, True)])
def test_first_mmac_priority_has_actual_ssa_cohort(dtype, tile, defer, extra_use):
    source = GluonASTSource(
        _priority_dot,
        {"A": "*" + dtype, "B": "*" + dtype, "C": "*fp32", "SIDE": "*fp32",
         "M": "constexpr", "EXTRA_USE": "constexpr"},
        {"M": tile, "EXTRA_USE": extra_use},
    )
    kernel = triton.compile(
        source, target=GPUTarget("hip", "gfx936", 64),
        options=dict(num_warps=4, num_stages=1, priority_after_first_mmac=defer),
    )
    assembly = kernel.asm["amdgcn"]
    high = assembly.index("s_setprio 1")
    preceding = re.findall(r"^\s*v_mmac_", assembly[:high], re.M)
    assert len(preceding) == (1 if defer and not extra_use else 0)
    assert assembly.count("s_setprio 1") == assembly.count("s_setprio 0") == 1
    assert kernel.asm["ttgir"].count("hcug.accumulator_fence") == 2
    assert "elementwise_inline_asm" not in kernel.asm["ttgir"]
    assert "~{memory}" in kernel.asm["llir"]
