# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""mul_rn on RDNA must keep the product's rounding boundary that RoPE relies on:
a following add must not contract into an FMA under enable_fp_fusion, and fp64
inputs must still resolve. These were regressions from the earlier
_semantic.mul / fma(a, b, 0) implementations; mul_rn now emits the hardware
multiply via inline asm, which the contraction pass cannot see through."""

import pytest
import triton
import triton.language as tl
from triton.backends.compiler import GPUTarget
from triton.language.extra.hip import libdevice

ARCHS = ("gfx1100", "gfx1201")


@triton.jit
def _rounded_mul_add(X, Y, C, Z):
    tl.store(Z, libdevice.mul_rn(tl.load(X), tl.load(Y)) + tl.load(C))


@triton.jit
def _mul_only(X, Y, Z):
    tl.store(Z, libdevice.mul_rn(tl.load(X), tl.load(Y)))


def _amdgcn(fn, signature, arch):
    src = triton.compiler.ASTSource(fn=fn, signature=signature, constexprs={})
    return triton.compile(src, target=GPUTarget("hip", arch, 32)).asm["amdgcn"]


@pytest.mark.parametrize("arch", ARCHS)
def test_mul_rn_does_not_contract_into_fma(arch):
    asm = _amdgcn(
        _rounded_mul_add,
        {"X": "*fp32", "Y": "*fp32", "C": "*fp32", "Z": "*fp32"},
        arch,
    )
    assert "v_fmac" not in asm and "v_fma_f32" not in asm, \
        "mul_rn(a, b) + c contracted into an FMA"
    assert "v_mul_f32" in asm


@pytest.mark.parametrize("arch", ARCHS)
def test_mul_rn_resolves_fp64(arch):
    # Regression: fma(a, b, 0.0) handed the fp64 overload an fp32 addend.
    _amdgcn(_mul_only, {"X": "*fp64", "Y": "*fp64", "Z": "*fp64"}, arch)


@pytest.mark.parametrize("arch", ARCHS)
@pytest.mark.parametrize("sx,sy", [("*fp32", "*fp64"), ("*fp64", "*fp32")])
def test_mul_rn_promotes_mixed_operands(arch, sx, sy):
    # Regression: the inline-asm multiply was selected from arg0.dtype alone, so
    # a mixed fp32/fp64 call emitted mismatched operand widths that failed to
    # assemble. Both operands must be promoted to the common type first.
    _amdgcn(_mul_only, {"X": sx, "Y": sy, "Z": "*fp64"}, arch)
