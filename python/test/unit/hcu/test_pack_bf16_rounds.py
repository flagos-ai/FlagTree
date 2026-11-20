"""Check that the HCU pack preserves software BF16 rounding semantics."""

import re

import pytest
import triton
from triton._C.libtriton import llvm
from triton.backends.compiler import GPUTarget
from triton.experimental import gluon as g
from triton.experimental.gluon import language as gl
from triton.experimental.gluon._runtime import GluonASTSource
from triton.experimental.gluon.language.amd.cdna3 import buffer_store

pytestmark = pytest.mark.skipif(not hasattr(llvm, "pack_hcu_bf16_rounds"), reason="HCU compiler required")


def _rounding(prefix):
    return f"""
      %{prefix}.bits = bitcast float %{prefix} to i32
      %{prefix}.nan = fcmp uno float %{prefix}, 0.0
      %{prefix}.hi = lshr i32 %{prefix}.bits, 16
      %{prefix}.parity = and i32 %{prefix}.hi, 1
      %{prefix}.bias = add i32 %{prefix}.bits, 32767
      %{prefix}.rounded = add i32 %{prefix}.bias, %{prefix}.parity
      %{prefix}.shift = lshr i32 %{prefix}.rounded, 16
      %{prefix}.short = trunc i32 %{prefix}.shift to i16
      %{prefix}.bf = bitcast i16 %{prefix}.short to bfloat
      %{prefix}.res = select i1 %{prefix}.nan, bfloat 0xR7FFF, bfloat %{prefix}.bf
    """


def _packed_ir(extra=""):
    return f"""
    target datalayout = "e"
    target triple = "amdgcn-amd-amdhsa"
    define i32 @test(float %x, float %y, ptr %out) {{
      {_rounding("x")}
      {_rounding("y")}
      {extra}
      %lo = insertelement <2 x bfloat> poison, bfloat %x.res, i64 0
      %hi = insertelement <2 x bfloat> %lo, bfloat %y.res, i64 1
      %packed = bitcast <2 x bfloat> %hi to i32
      ret i32 %packed
    }}
    """


def test_packs_two_rounded_high_halfwords():
    result, sites = llvm.pack_hcu_bf16_rounds(_packed_ir())
    assert sites == 1
    assert "call i32 @llvm.amdgcn.perm" in result
    assert "i32 117834498)" in result  # Select high halfword from each input.
    assert result.count("i32 2147418112") == 2
    assert "%x.shift =" not in result and "%y.shift =" not in result
    assert result.index("select i1 %x.nan") < result.index("%y.bits =")
    assert llvm.pack_hcu_bf16_rounds(result) == (result, 0)


@pytest.mark.parametrize(
    ("original", "replacement"),
    [
        ("bfloat 0xR7FFF", "bfloat 0xR7FC0"),
        ("i32 %x.bits, 32767", "i32 %x.bits, 32768"),
        ("fcmp uno float %x, 0.0", "fcmp uno float %y, 0.0"),
        ("lshr i32 %x.rounded", "lshr exact i32 %x.rounded"),
        ("bfloat %y.res, i64 1", "bfloat %y.res, i64 0"),
        ('datalayout = "e"', 'datalayout = "E"'),
    ],
)
def test_incompatible_pack_is_unchanged(original, replacement):
    source = _packed_ir().replace(original, replacement, 1)
    assert llvm.pack_hcu_bf16_rounds(source) == (source, 0)


def test_keeps_conversion_with_another_user():
    source = _packed_ir("store bfloat %x.res, ptr %out")
    assert llvm.pack_hcu_bf16_rounds(source) == (source, 0)


@g.jit
def _cast_store(src, dst):
    layout: gl.constexpr = gl.DistributedLinearLayout([[1]], [[2], [4], [8], [16], [32], [64]], [], [], [128])
    index = gl.arange(0, 128, layout=layout)
    x = gl.load(src + index)
    buffer_store(x.to(gl.bfloat16), dst, index)


def test_packs_elementwise_conversion_without_matrix_layout():
    source = GluonASTSource(
        _cast_store,
        {"src": "*fp32", "dst": "*bf16"},
        {},
        attrs={(0, ): [("tt.divisibility", 16)], (1, ): [("tt.divisibility", 16)]},
    )
    target = GPUTarget("hip", "gfx936", 64)
    before = triton.compile(source, target=target, options={"num_warps": 1})
    after = triton.compile(source, target=target, options={"num_warps": 1, "pack_bf16_rounds": True})
    assert after.metadata.hcu_packed_bf16_round_sites == 1
    shifts = r"^\s*v_lshrrev_b32(?:_e32|_e64)?\s"
    assert len(re.findall(shifts, before.asm["amdgcn"], re.M)) == 2
    assert not re.findall(shifts, after.asm["amdgcn"], re.M)
