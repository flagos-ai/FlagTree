"""Both typed G2L and matrix-read forms retain ordinary physical object scopes."""
import re
import pytest
from triton._C import libtriton
from triton.backends.compiler import GPUTarget

if not hasattr(libtriton,"hcu"):
    pytest.skip("HCU bindings required",allow_module_level=True)
compiler=pytest.importorskip("triton.backends.hcu.compiler")


def module_source(structured,alt,unknown=False,size=16):
    name="llvm.amdgcn.struct.ptr.buffer.load.lds" if structured else "llvm.amdgcn.raw.ptr.buffer.load.lds"
    args="!llvm.ptr<8>, !llvm.ptr<3>, i32, i32, i32, i32, i32"+(", i32" if structured else "")
    values="%rsrc, %a, %size, %zero, %zero, %zero, %zero"+(", %zero" if structured else "")
    values2=values.replace("%a,","%b,")
    read="llvm.hcu.ds.read.m32x16.b16."+("alt." if alt else "")+"v4i32.p3"
    ptra="%a = llvm.inttoptr %opaque : i32 to !llvm.ptr<3>" if unknown else "%a = llvm.getelementptr %root[%zero] : (!llvm.ptr<3>, i32) -> !llvm.ptr<3>, i8"
    return f'''#domain = #llvm.alias_scope_domain<id = "hcug.AsyncOps">
#copy = #llvm.alias_scope<id = "hcug.AsyncCopies",domain = #domain>
#read = #llvm.alias_scope<id = "hcug.LocalLoads",domain = #domain>
module {{
 llvm.mlir.global external @global_smem() {{addr_space = 3 : i32,alignment = 16 : i64}} : !llvm.array<0 x i8>
 llvm.func @{name}({args})
 llvm.func @{read}(!llvm.ptr<3>,i32) -> vector<4xi32>
 llvm.func @scopes(%rsrc:!llvm.ptr<8>,%opaque:i32) -> vector<4xi32> {{
  %zero = llvm.mlir.constant(0 : i32) : i32
  %size = llvm.mlir.constant({size} : i32) : i32
  %offset = llvm.mlir.constant(8192 : i32) : i32
  %root = llvm.mlir.addressof @global_smem : !llvm.ptr<3>
  {ptra}
  %b = llvm.getelementptr %root[%offset] : (!llvm.ptr<3>,i32) -> !llvm.ptr<3>,i8
  llvm.call @{name}({values}) {{alias_scopes = [#copy]}} : ({args}) -> ()
  llvm.call @{name}({values2}) {{alias_scopes = [#copy]}} : ({args}) -> ()
  %value = llvm.call @{read}(%a,%zero) {{alias_scopes = [#read],noalias_scopes = [#copy]}} : (!llvm.ptr<3>,i32) -> vector<4xi32>
  llvm.return %value : vector<4xi32>
 }}
}}'''


def lower(tmp_path,structured,alt,unknown=False,size=16):
    path=tmp_path/'scopes.mlir';path.write_text(module_source(structured,alt,unknown,size))
    context=libtriton.ir.context();libtriton.ir.load_dialects(context)
    compiler.HIPBackend(GPUTarget("hip","gfx936",64)).load_dialects(context)
    module=libtriton.ir.parse_mlir_module(str(path),context)
    libtriton.llvm.init_targets()
    llctx=libtriton.llvm.context();ll=libtriton.llvm.to_module(module,llctx)
    libtriton.hcu.attach_target_triple(ll)
    libtriton.llvm.attach_datalayout(ll,libtriton.hcu.TARGET_TRIPLE,"gfx936","")
    changed=libtriton.hcu.annotate_lds_object_scopes(ll,"gfx936",64)
    return changed,str(ll)


@pytest.mark.parametrize("structured",[False,True])
@pytest.mark.parametrize("alt",[False,True])
def test_typed_copy_and_gather_object_scopes(tmp_path,structured,alt):
    changed,ir=lower(tmp_path,structured,alt)
    assert changed==3
    copies=[x for x in ir.splitlines() if "call void @llvm.amdgcn." in x]
    assert len(copies)==2 and all("!alias.scope" in x for x in copies)
    scopes=[re.search(r"!alias.scope !(\d+)",x).group(1) for x in copies]
    assert scopes[0]!=scopes[1]
    read=next(x for x in ir.splitlines() if "call <4 x i32> @llvm.hcu.ds.read" in x)
    assert "!noalias" in read and "gfx936 LDS objects" in ir


def test_unknown_destination_keeps_conservative_scopes(tmp_path):
    changed,ir=lower(tmp_path,True,True,unknown=True)
    assert changed==0 and "gfx936 LDS objects" not in ir


def test_unverified_structured_8byte_footprint_stays_conservative(tmp_path):
    changed,ir=lower(tmp_path,True,True,size=8)
    assert changed==0 and "gfx936 LDS objects" not in ir


@pytest.mark.parametrize("kind",["bounded","unknown","outside","wrapping"])
def test_vector_read_add_over_deep_layout_bits(tmp_path,kind):
    # Ordinary vector loads can add a fragment offset over a deep layout OR
    # tree. Exhausted generic ValueTracking depth must not hide that bound;
    # unknown, outside-LDS and wrapping addresses must still stay conservative.
    source=module_source(True,False)
    lines=[];previous="%zero"
    for i in range(8):
        lines += [f"  %mask{i} = llvm.mlir.constant({16 << (i%6)} : i32) : i32",
                  f"  %part{i} = llvm.and %opaque, %mask{i} : i32",
                  f"  %tree{i} = llvm.or {previous}, %part{i} : i32"]
        previous=f"%tree{i}"
    base="%opaque" if kind=="unknown" else previous
    displacement={"bounded":256,"unknown":256,"outside":65536,"wrapping":2147483647}[kind]
    lines += [f"  %fragment = llvm.mlir.constant({displacement} : i32) : i32",
              f"  %index = llvm.add {base}, %fragment : i32",
              "  %readptr = llvm.getelementptr %root[%index] : (!llvm.ptr<3>,i32) -> !llvm.ptr<3>,i8",
              "  %value = llvm.load %readptr {alias_scopes = [#read],noalias_scopes = [#copy],alignment = 16 : i64} : !llvm.ptr<3> -> vector<4xi32>"]
    original=next(line for line in source.splitlines() if "%value = llvm.call" in line)
    source=source.replace(original,"\n".join(lines))
    path=tmp_path/'vector-scopes.mlir';path.write_text(source)
    context=libtriton.ir.context();libtriton.ir.load_dialects(context)
    compiler.HIPBackend(GPUTarget("hip","gfx936",64)).load_dialects(context)
    module=libtriton.ir.parse_mlir_module(str(path),context)
    llctx=libtriton.llvm.context();ll=libtriton.llvm.to_module(module,llctx)
    libtriton.hcu.attach_target_triple(ll)
    libtriton.llvm.attach_datalayout(ll,libtriton.hcu.TARGET_TRIPLE,"gfx936","")
    changed=libtriton.hcu.annotate_lds_object_scopes(ll,"gfx936",64)
    assert changed==(3 if kind=="bounded" else 0)
    assert ("gfx936 LDS objects" in str(ll))==(kind=="bounded")


@pytest.mark.parametrize("factor", [520, -2147483128])
@pytest.mark.parametrize("kind", ["bounded", "unknown", "outside", "wrapping"])
def test_masked_mul_in_last_lds_object(tmp_path, kind, factor):
    # This affine/XOR chart depends only on the low eight input bits. Enumerate
    # that complete domain to check the intended 16-byte access ends at 64KiB.
    # The negative factor has the same low product for this even mask,
    # despite unsigned overflow; the analysis must not assume no-wrap.
    offsets = []
    for tid in range(256):
        index = (((tid & 6) * factor) & 0xffffffff) ^ (tid & 48)
        index += (tid & 128) << 2
        index += ((tid << 6) & 64) | ((tid << 4) & 128)
        offsets.append(index | 4352)
    assert min(offsets) == 4352 and max(offsets) == 8176
    source = module_source(True, False).replace(
        "constant(8192 : i32)", "constant(61440 : i32)"
    )
    plane = {"bounded": 57344, "unknown": 57344,
             "outside": 65536, "wrapping": 2147483647}[kind]
    base = ("%plane = llvm.inttoptr %opaque : i32 to !llvm.ptr<3>"
            if kind == "unknown" else
            "%plane = llvm.getelementptr %root[%plane_offset] : "
            "(!llvm.ptr<3>, i32) -> !llvm.ptr<3>, i8")
    body = f"""
  %c2 = llvm.mlir.constant(2 : i32) : i32
  %c4 = llvm.mlir.constant(4 : i32) : i32
  %c6 = llvm.mlir.constant(6 : i32) : i32
  %c48 = llvm.mlir.constant(48 : i32) : i32
  %c64 = llvm.mlir.constant(64 : i32) : i32
  %c128 = llvm.mlir.constant(128 : i32) : i32
  %c520 = llvm.mlir.constant({factor} : i32) : i32
  %c4352 = llvm.mlir.constant(4352 : i32) : i32
  %plane_offset = llvm.mlir.constant({plane} : i32) : i32
  %mask6 = llvm.and %opaque, %c6 : i32
  %scaled = llvm.mul %mask6, %c520 : i32
  %mask48 = llvm.and %opaque, %c48 : i32
  %xored = llvm.xor %scaled, %mask48 : i32
  %mask128 = llvm.and %opaque, %c128 : i32
  %warp = llvm.shl %mask128, %c2 : i32
  %lane_shift6 = llvm.shl %opaque, %c6 : i32
  %lane64 = llvm.and %lane_shift6, %c64 : i32
  %lane_shift4 = llvm.shl %opaque, %c4 : i32
  %lane128 = llvm.and %lane_shift4, %c128 : i32
  %lanes = llvm.or %lane64, %lane128 : i32
  %with_warp = llvm.add %xored, %warp : i32
  %sum = llvm.add %with_warp, %lanes : i32
  %index = llvm.or %sum, %c4352 : i32
  {base}
  %readptr = llvm.getelementptr %plane[%index] : (!llvm.ptr<3>, i32) -> !llvm.ptr<3>, i8
  %value = llvm.load %readptr {{alias_scopes = [#read],noalias_scopes = [#copy],alignment = 16 : i64}} : !llvm.ptr<3> -> vector<4xi32>
"""
    original = next(line for line in source.splitlines() if "%value = llvm.call" in line)
    source = source.replace(original, body)
    path = tmp_path / "masked-mul-last-object.mlir"
    path.write_text(source)
    context = libtriton.ir.context()
    libtriton.ir.load_dialects(context)
    compiler.HIPBackend(GPUTarget("hip", "gfx936", 64)).load_dialects(context)
    module = libtriton.ir.parse_mlir_module(str(path), context)
    libtriton.llvm.init_targets()
    llctx = libtriton.llvm.context()
    ll = libtriton.llvm.to_module(module, llctx)
    libtriton.hcu.attach_target_triple(ll)
    libtriton.llvm.attach_datalayout(ll, libtriton.hcu.TARGET_TRIPLE, "gfx936", "")
    changed = libtriton.hcu.annotate_lds_object_scopes(ll, "gfx936", 64)
    text = str(ll)
    assert changed == (3 if kind == "bounded" else 0)
    if kind == "bounded":
        copies = [line for line in text.splitlines() if "call void @llvm.amdgcn." in line]
        read = next(line for line in text.splitlines() if "load <4 x i32>" in line)
        def members(number):
            line = next(line for line in text.splitlines() if line.startswith("!" + number + " = "))
            return set(re.findall(r"!(\d+)", line.split(" = ", 1)[1]))
        scopes = [members(re.search(r"!alias.scope !(\d+)", line).group(1)) for line in copies]
        excluded = members(re.search(r"!noalias !(\d+)", read).group(1))
        assert scopes[0] <= excluded
        assert not (scopes[1] & excluded), "Do not exclude the overlapping last object"
