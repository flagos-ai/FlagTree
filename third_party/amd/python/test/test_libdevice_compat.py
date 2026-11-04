"""Native AMD math compatibility APIs compile without a GPU launch."""
import pytest
import triton
import triton.language as tl
from triton.language.extra.hip import libdevice
from triton.compiler import ASTSource
from triton.backends.compiler import GPUTarget

@triton.jit
def _unary(A, Out, FN: tl.constexpr):
    i = tl.arange(0, 64)
    tl.store(Out + i, FN(tl.load(A + i)))

@triton.jit
def _rounded_product(A, B, C, Out):
    i = tl.arange(0, 64)
    product = libdevice.mul_rn(tl.load(A + i), tl.load(B + i))
    tl.store(Out + i, product + tl.load(C + i))

@pytest.mark.parametrize("name,symbol", [
    ("fast_cosf", "__ocml_cos_f32"), ("fast_sinf", "__ocml_sin_f32")])
def test_fast_math_compatibility_symbols(name, symbol):
    compiled = triton.compile(ASTSource(_unary,
        signature={"A": "*fp32", "Out": "*fp32"},
        constexprs={"FN": getattr(libdevice, name)}),
        target=GPUTarget("hip", "gfx1100", 32), options={"num_warps": 4})
    assert symbol in compiled.asm["ttir"]

@pytest.mark.parametrize("left,right,result,instruction", [
    ("fp32", "fp32", "fp32", "v_mul_f32"),
    ("fp64", "fp64", "fp64", "v_mul_f64"),
    ("fp32", "fp64", "fp64", "v_mul_f64")])
def test_mul_rn_keeps_a_separately_rounded_product(left, right, result, instruction):
    compiled = triton.compile(ASTSource(_rounded_product,
        signature={"A": "*" + left, "B": "*" + right,
                   "C": "*" + result, "Out": "*" + result}),
        target=GPUTarget("hip", "gfx1100", 32),
        options={"num_warps": 4, "enable_fp_fusion": True})
    assembly = compiled.asm["amdgcn"]
    assert instruction in assembly
    assert "v_fma_f" not in assembly and "v_mad_f" not in assembly
