"""SDK descriptor-ABI fixtures based on crt/sqmma.h device-only output."""

import pytest
from triton.backends.mthreads.compiler import _sdk_sqmma_descriptor_compat


@pytest.mark.parametrize("suffix,scalar,alignment", [("i8", "i8", 1), ("half", "i16", 2), ("fp32", "float", 4)])
def test_pointer_call_and_declaration(suffix, scalar, alignment):
    ir = (
        f"  %7 = call i32 @llvm.musa.sqmma.desc.{suffix}(ptr addrspace(3) %p, i32 128, i32 1, i32 3), !dbg !9\n"
        f"declare i32 @llvm.musa.sqmma.desc.{suffix}(ptr addrspace(3), i32, i32, i32) #1\n"
    )
    out = _sdk_sqmma_descriptor_compat(ir)
    assert f"load {scalar}, ptr addrspace(3) %p, align {alignment}" in out
    assert f"desc.{suffix}({scalar} %musa.sdk.desc.0, i32 128, i32 1, i32 3), !dbg !9" in out
    assert f"desc.{suffix}({scalar}, i32, i32, i32) #1" in out
    assert "%7 = call" in out
    assert _sdk_sqmma_descriptor_compat(out) == out


def test_constant_gep_and_existing_name():
    ir = (
        "  %musa.sdk.desc.0 = add i32 0, 1\n"
        "  %7 = tail call i32 @llvm.musa.sqmma.desc.i8(ptr addrspace(3) getelementptr (i8, ptr addrspace(3) @smem, i32 81920), i32 128, i32 0, i32 3)\n"
    )
    out = _sdk_sqmma_descriptor_compat(ir)
    assert (
        "%musa.sdk.desc.1 = load i8, ptr addrspace(3) getelementptr (i8, ptr addrspace(3) @smem, i32 81920), align 1"
        in out
    )
    assert "tail call i32 @llvm.musa.sqmma.desc.i8(i8 %musa.sdk.desc.1, i32 128, i32 0, i32 3)" in out


@pytest.mark.parametrize(
    "operand", ["ptr %global", "ptr addrspace(1) %global", "ptr addrspace(3) poison", "ptr addrspace(3) nonnull %p"]
)
def test_unsupported_operand_fails_explicitly(operand):
    with pytest.raises(ValueError, match="Unsupported SQMMA descriptor"):
        _sdk_sqmma_descriptor_compat(f"%r = call i32 @llvm.musa.sqmma.desc.i8({operand}, i32 128, i32 0, i32 3)\n")


def test_unrelated_ir_unchanged():
    ir = "declare i32 @llvm.musa.other(ptr addrspace(3))\n%v = load i8, ptr addrspace(3) %p\n"
    assert _sdk_sqmma_descriptor_compat(ir) == ir
