"""Exercise the HCU common VOFFSET transform and its safety conditions."""

import pytest
from triton._C.libtriton import llvm

pytestmark = pytest.mark.skipif(not hasattr(llvm, "factor_hcu_buffer_store_offsets"), reason="HCU compiler required")

INTRINSIC = "llvm.amdgcn.raw.ptr.buffer.store.i32"


def _fixture(
    mask=255,
    delta=12288,
    masked=False,
    distinct=False,
    soffset=0,
    separate_blocks=False,
    payload="i32",
    suffix="i32",
    cache=0,
):
    intrinsic = "llvm.amdgcn.raw.ptr.buffer.store." + suffix

    def store(resource, offset):
        return (f"  call void @{intrinsic}({payload} %value, ptr addrspace(8) %{resource}, "
                f"i32 %{offset}, i32 {soffset}, i32 {cache})\n")

    ir = f"""define amdgpu_kernel void @mm(ptr addrspace(8) %r, ptr addrspace(8) %r2,
                                  i32 %row, i32 %col, {payload} %value, i1 %valid) {{
entry:
  %left = mul i32 %row, 3072
  %right = and i32 %col, {mask}
  %next = add i32 %left, {delta}
  %p0 = or i32 %left, %right
  %p1 = or i32 %next, %right
"""
    if masked:
        ir += "  %s0 = select i1 %valid, i32 %p0, i32 -1\n  %s1 = select i1 %valid, i32 %p1, i32 -1\n"
    ir += store("r", "s0" if masked else "p0")
    if separate_blocks:
        ir += "  br label %next_block\nnext_block:\n"
    ir += store("r2" if distinct else "r", "s1" if masked else "p1")
    return ir + f"""  ret void
}}
declare void @{intrinsic}({payload}, ptr addrspace(8), i32, i32, i32 immarg)
"""


@pytest.mark.parametrize("masked", [False, True])
def test_factors_disjoint_shared_expressions(masked):
    transformed, count = llvm.factor_hcu_buffer_store_offsets(_fixture(masked=masked), "gfx936")
    assert count == 2
    stores = [line.strip() for line in transformed.splitlines() if f"call void @{INTRINSIC}" in line]
    assert len(stores) == 2
    assert all("i32 %value, ptr addrspace(8) %r" in line and line.endswith("i32 0, i32 0)") for line in stores)
    assert transformed.count("select i1 %valid") == (2 if masked else 0)
    assert transformed.count("i32 -1") == (2 if masked else 0)
    assert "or i32" not in transformed
    assert transformed.count(", 12288") == 1


@pytest.mark.parametrize(
    "arch, kwargs",
    [
        ("gfx936", {"mask": 4095}),
        ("gfx936", {"delta": 12289}),
        ("gfx936", {"distinct": True}),
        ("gfx936", {"separate_blocks": True}),
        ("gfx942", {}),
    ],
)
def test_keeps_unproved_or_unsupported_stores(arch, kwargs):
    source = _fixture(**kwargs)
    assert llvm.factor_hcu_buffer_store_offsets(source, arch) == (source, 0)


@pytest.mark.parametrize("masked", [False, True])
@pytest.mark.parametrize("payload,suffix", [
    ("i16", "i16"),
    ("half", "f16"),
    ("float", "f32"),
    ("<2 x i32>", "v2i32"),
    ("<4 x i32>", "v4i32"),
    ("<2 x float>", "v2f32"),
])
def test_address_factoring_does_not_depend_on_payload_type(masked, payload, suffix):
    source = _fixture(masked=masked, payload=payload, suffix=suffix)
    transformed, count = llvm.factor_hcu_buffer_store_offsets(source, "gfx936")
    assert count == 2
    # The same modulo-i32 offset equality applies to every payload overload;
    # neither data, descriptor, mask, cache policy nor store order changes.
    intrinsic = "llvm.amdgcn.raw.ptr.buffer.store." + suffix
    stores = [line for line in transformed.splitlines() if f"call void @{intrinsic}" in line]
    assert len(stores) == 2 and all(f"{payload} %value" in line for line in stores)
    assert transformed.count("select i1 %valid") == (2 if masked else 0)
    assert llvm.factor_hcu_buffer_store_offsets(transformed, "gfx936") == (transformed, 0)


@pytest.mark.parametrize("soffset", [0, 4096, -16])
@pytest.mark.parametrize("cache", [0, 1, 2, 3])
def test_preserves_other_address_and_cache_fields(soffset, cache):
    transformed, count = llvm.factor_hcu_buffer_store_offsets(_fixture(soffset=soffset, cache=cache), "gfx936")
    assert count == 2
    stores = [line for line in transformed.splitlines() if f"call void @{INTRINSIC}" in line]
    assert len(stores) == 2
    assert all(line.endswith(f"i32 {soffset}, i32 {cache})") for line in stores)
