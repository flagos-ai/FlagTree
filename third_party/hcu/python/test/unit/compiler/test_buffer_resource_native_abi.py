import pytest
from triton._C.libtriton import llvm


INTRINSIC = "@llvm.amdgcn.make.buffer.rsrc.p8.p1"


def resource_module(count):
    return f"""
declare ptr addrspace(8) {INTRINSIC}(ptr addrspace(1), i16, i64, i32)

define ptr addrspace(8) @resource(ptr addrspace(1) %base, i32 %count) {{
  %wide = zext i32 %count to i64
  %resource = call ptr addrspace(8) {INTRINSIC}(ptr addrspace(1) %base, i16 0, i64 {count}, i32 0)
  ret ptr addrspace(8) %resource
}}
"""


@pytest.mark.parametrize("count", ["%wide", "4294967295"])
def test_clang18_buffer_resource_count_is_narrowed_when_proved(count):
    source = resource_module(count)
    converted = llvm.adapt_hcu_buffer_resources_for_clang18(source)
    declaration = next(line for line in converted.splitlines() if line.startswith("declare") and INTRINSIC in line)
    assert ", i16, i32, i32)" in declaration
    call = next(line for line in converted.splitlines() if f"call ptr addrspace(8) {INTRINSIC}" in line)
    assert ", i64 " not in call


def test_clang18_buffer_resource_count_cannot_drop_nonzero_bits():
    with pytest.raises(ValueError, match="32-bit buffer resource count"):
        llvm.adapt_hcu_buffer_resources_for_clang18(resource_module("4294967296"))
