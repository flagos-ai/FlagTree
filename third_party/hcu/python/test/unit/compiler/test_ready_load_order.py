from ._helpers import load_fixture as _ready_shared_loads_llvm_load_fixture
# ready shared loads
import pytest
from triton._C import libtriton
from triton.backends.compiler import GPUTarget

if not hasattr(libtriton, "hcu"):
    pytest.skip("HCU compiler required", allow_module_level=True)

from triton.backends.hcu.compiler import HIPBackend


def _ready_shared_loads_source(native=False, negative=None, structured=False):
    read_name = "llvm.hcu.ds.read.m32x16.b16.v4i32.p3"
    read_decl = f"llvm.func @{read_name}(!llvm.ptr<3>, i32) -> vector<4xi32>"
    parts = []
    for slot in range(2):
        exclude = "" if negative == "noalias" else f", noalias_scopes = [#copy{slot}]"
        # Excluding a different producer cannot authorize movement.
        if negative == "wrong_scope":
            exclude = f", noalias_scopes = [#copy{1-slot}]"
        attributes = f"{{alias_scopes = [#read]{exclude}}}"
        if native:
            read = (f"%value{slot} = llvm.call @{read_name}(%read, %zero) {attributes}"
                    " : (!llvm.ptr<3>, i32) -> vector<4xi32>")
        else:
            read = (f"%value{slot} = llvm.load %read {attributes}"
                    " : !llvm.ptr<3> -> vector<4xi32>")
        boundary = {
            "barrier": "llvm.call @llvm.amdgcn.s.barrier() : () -> ()",
            "wait": "llvm.call @llvm.amdgcn.s.waitcnt(%zero) : (i32) -> ()",
            "unknown_call": "llvm.call @opaque() : () -> ()",
            "unknown_store": "llvm.store %zero, %read : i32, !llvm.ptr<3>",
            "storage_fact": 'llvm.intr.assume %true ["dereferenceable"(%write, %extent : !llvm.ptr<3>, i64)] : i1',
            "conditional_assume": 'llvm.intr.assume %more : i1',
        }.get(negative, "")
        parts += [
            f"""
    llvm.call @llvm.amdgcn.raw.ptr.buffer.load.lds(%rsrc, %write, %size,
      %zero, %zero, %zero, %zero) {{alias_scopes = [#copy{slot}]}}
      : (!llvm.ptr<8>, !llvm.ptr<3>, i32, i32, i32, i32, i32) -> ()
    {boundary}
    {read}
"""
        ]
        if slot == 0:
            parts += ["llvm.call @llvm.amdgcn.s.barrier() : () -> ()"]
    text = """#domain = #llvm.alias_scope_domain<id = "test.ready">
#copy0 = #llvm.alias_scope<id = "copy0", domain = #domain>
#copy1 = #llvm.alias_scope<id = "copy1", domain = #domain>
#read = #llvm.alias_scope<id = "read", domain = #domain>
module {
  llvm.func @llvm.amdgcn.raw.ptr.buffer.load.lds(
    !llvm.ptr<8>, !llvm.ptr<3>, i32, i32, i32, i32, i32)
  llvm.func @llvm.amdgcn.s.barrier()
  llvm.func @llvm.amdgcn.s.waitcnt(i32)
  llvm.func @opaque()
  """ + read_decl + """
  llvm.func @ready(%rsrc: !llvm.ptr<8>, %write: !llvm.ptr<3>,
                   %read: !llvm.ptr<3>, %more: i1) {
    %zero = llvm.mlir.constant(0 : i32) : i32
    %size = llvm.mlir.constant(16 : i32) : i32
    %true = llvm.mlir.constant(true) : i1
    %extent = llvm.mlir.constant(4096 : i64) : i64
    llvm.br ^loop
  ^loop:
""" + "\n".join(parts) + """
    llvm.cond_br %more, ^loop, ^exit
  ^exit:
    llvm.return
  }
}
"""
    if structured:
        text = text.replace("llvm.amdgcn.raw.ptr.buffer.load.lds", "llvm.amdgcn.struct.ptr.buffer.load.lds")
        text = text.replace("!llvm.ptr<8>, !llvm.ptr<3>, i32, i32, i32, i32, i32)",
                            "!llvm.ptr<8>, !llvm.ptr<3>, i32, i32, i32, i32, i32, i32)")
        text = text.replace("%zero, %zero, %zero, %zero)", "%zero, %zero, %zero, %zero, %zero)")
    return text


def _ready_shared_loads_lower(tmp_path, native=False, negative=None, arch="gfx936", kernel=True, structured=False):
    path = tmp_path / "ready.mlir"
    path.write_text(_ready_shared_loads_source(native, negative, structured))
    context = libtriton.ir.context()
    libtriton.ir.load_dialects(context)
    HIPBackend(GPUTarget("hip", "gfx936", 64)).load_dialects(context)
    module = libtriton.ir.parse_mlir_module(str(path), context)
    llctx = libtriton.llvm.context()
    ll = libtriton.llvm.to_module(module, llctx)
    if kernel:
        function = next(f for f in ll.get_functions() if f.name == "ready")
        function.set_calling_conv(libtriton.hcu.CALLING_CONV_AMDGPU_KERNEL)
    before = str(ll)
    changed = libtriton.hcu.prefer_ready_shared_loads(ll, arch)
    return changed, before, str(ll)


@pytest.mark.parametrize("native", [False, True])
def test_separate_epochs_keep_their_own_slot_proofs(tmp_path, native):
    changed, before, after = _ready_shared_loads_lower(tmp_path, native)
    assert changed == 2
    assert after.count("call void @llvm.amdgcn.s.barrier()") == 1
    # The producer order and each existing barrier remain intact. A schedule
    # fence separates the completed reads from the future-slot copies.
    body = after.split("define amdgpu_kernel void @ready", 1)[1]
    copy = "call void @llvm.amdgcn.raw.ptr.buffer.load.lds"
    read = "call <4 x i32> @llvm.hcu.ds.read" if native else "load <4 x i32>"
    halves = body.split("call void @llvm.amdgcn.s.barrier()")
    for half in halves:
        assert half.index(read) < half.index("call void @llvm.amdgcn.sched.barrier") < half.index(copy)
    assert after.count(copy) == before.count(copy) == 2


@pytest.mark.parametrize(
    "negative", ["noalias", "wrong_scope", "barrier", "wait", "unknown_call", "unknown_store", "conditional_assume"])
@pytest.mark.parametrize("native", [False, True])
def test_no_crossing_unproved_alias_or_memory_effect(tmp_path, native, negative):
    changed, before, after = _ready_shared_loads_lower(tmp_path, native, negative)
    assert changed == 0 and after == before


@pytest.mark.parametrize("arch,kernel", [("gfx942", True), ("gfx936", False)])
def test_target_and_kernel_scope_remain_exact_off(tmp_path, arch, kernel):
    changed, before, after = _ready_shared_loads_lower(tmp_path, arch=arch, kernel=kernel)
    assert changed == 0 and after == before


@pytest.mark.parametrize("native", [False, True])
def test_true_storage_assume_is_not_a_memory_boundary(tmp_path, native):
    changed, before, after = _ready_shared_loads_lower(tmp_path, native, "storage_fact")
    assert changed == 2
    assert after.count('"dereferenceable"') == before.count('"dereferenceable"') == 2
    assert after.count("call void @llvm.amdgcn.s.barrier()") == 1


@pytest.mark.parametrize("native", [False, True])
@pytest.mark.parametrize("boundary", [
    None, "storage_fact", "noalias", "wrong_scope", "barrier", "wait", "unknown_call", "unknown_store",
    "conditional_assume"
])
def test_structured_producer_obeys_same_epoch_and_complete_alias_proof(tmp_path, native, boundary):
    changed, before, after = _ready_shared_loads_lower(tmp_path, native, boundary, structured=True)
    if boundary in (None, "storage_fact"):
        assert changed == 2
        copy = "call void @llvm.amdgcn.struct.ptr.buffer.load.lds"
        read = "call <4 x i32> @llvm.hcu.ds.read" if native else "load <4 x i32>"
        body = after.split("define amdgpu_kernel void @ready", 1)[1]
        for part in body.split("call void @llvm.amdgcn.s.barrier()"):
            assert part.index(read) < part.index("call void @llvm.amdgcn.sched.barrier") < part.index(copy)
        assert after.count(copy) == before.count(copy) == 2
    else:
        assert changed == 0 and before == after


def _ready_shared_loads_readonly_interleave_fixture(tmp_path, native, offset=0, boundary="", volatile=False,
                                                    dependent=False):
    # The anchor takes the address instead of calling an AMDGPU kernel.
    context = libtriton.ir.context()
    libtriton.ir.load_dialects(context)
    libtriton.hcu.load_dialects(context)
    stub = tmp_path / "readonly-anchor.mlir"
    stub.write_text('''module {
      llvm.func @readonly_fixture(!llvm.ptr<1>, !llvm.ptr<3>, !llvm.ptr<1>)
      llvm.func @anchor() -> !llvm.ptr {
        %f = llvm.mlir.addressof @readonly_fixture : !llvm.ptr
        llvm.return %f : !llvm.ptr
      }
    }''')
    mlir = libtriton.ir.parse_mlir_module(str(stub), context)
    llvm_context = libtriton.llvm.context()
    module = libtriton.llvm.to_module(mlir, llvm_context)
    name = "llvm.hcu.ds.read.m32x16.b16." + ("alt." if native == "alt" else "") + "v4i32.p3"
    if native == "unknown":
        name = "unknown_shared_reader"
    declaration = f"declare <4 x i32> @{name}(ptr addrspace(3), i32) convergent nounwind memory(read)"
    if native == "load":
        pointer = "%s_offset" if offset else "%s"
        read = lambda value: f"%{value} = load {'volatile ' if volatile else ''}<4 x i32>, ptr addrspace(3) {pointer}, align 16"
    else:
        pointer = "%dependent" if dependent else "%s"
        read = lambda value: f"%{value} = call <4 x i32> @{name}(ptr addrspace(3) {pointer}, i32 {offset})"
    address = """%index = extractelement <4 x i32> %global, i32 0
        %dependent = getelementptr i8, ptr addrspace(3) %s, i32 %index""" if dependent else ""
    if native == "load" and offset:
        address = f"%s_offset = getelementptr i8, ptr addrspace(3) %s, i32 {offset}"
    library = tmp_path / "readonly-fixture.ll"
    library.write_text(f'''{declaration}
      declare void @llvm.amdgcn.s.waitcnt(i32 immarg)
      declare void @llvm.amdgcn.s.barrier()
      declare void @opaque()
      define void @readonly_fixture(ptr addrspace(1) %input, ptr addrspace(3) %s,
                                    ptr addrspace(1) %output) {{
      entry:
        %global = load <4 x i32>, ptr addrspace(1) %input, align 16
        {address}
        {boundary}
        {read("first")}
        {read("second")}
        %sum = add <4 x i32> %global, %first
        %value = add <4 x i32> %sum, %second
        store <4 x i32> %value, ptr addrspace(1) %output, align 16
        ret void
      }}''')
    libtriton.llvm.link_extern_libs(module, [str(library)])
    function = next(f for f in module.get_functions() if f.name == "readonly_fixture")
    function.set_calling_conv(libtriton.hcu.CALLING_CONV_AMDGPU_KERNEL)
    return module, (context, mlir, llvm_context)


@pytest.mark.parametrize("native", ["load", "normal", "alt"])
@pytest.mark.parametrize("offset", [0, 2048])
@pytest.mark.parametrize("fences", [False, True])
def test_readonly_interleave_covers_native_shared_reads(tmp_path, native, offset, fences):
    module, keepalive = _ready_shared_loads_readonly_interleave_fixture(tmp_path, native, offset)
    result = libtriton.hcu.interleave_global_lds_reads(module, "gfx936", fences)
    assert result["transformed_windows"] == 1 and result["moved_global_reads"] == 1 and result["split_lds_reads"] == 2
    assert result["sched_barriers_added"] == (2 if fences else 0)
    text = str(module)
    assert text.index("%first =") < text.index("%global =") < text.index("%second =")
    again = libtriton.hcu.interleave_global_lds_reads(module, "gfx936", fences)
    assert again["transformed_windows"] == 0 and str(module) == text


@pytest.mark.parametrize("boundary", [
    "call void @llvm.amdgcn.s.waitcnt(i32 0)",
    "call void @llvm.amdgcn.s.barrier()",
    "store i32 1, ptr addrspace(3) %s, align 4",
    "store i32 1, ptr addrspace(1) %output, align 4",
    "call void @opaque()",
    'call void asm sideeffect "", "~{memory}"()',
])
@pytest.mark.parametrize("native", ["normal", "alt"])
def test_readonly_interleave_native_reads_keep_effect_boundaries(tmp_path, native, boundary):
    module, keepalive = _ready_shared_loads_readonly_interleave_fixture(tmp_path, native, boundary=boundary)
    before = str(module)
    result = libtriton.hcu.interleave_global_lds_reads(module, "gfx936", True)
    assert result["transformed_windows"] == 0 and str(module) == before


@pytest.mark.parametrize("native,volatile,arch,kernel", [
    ("unknown", False, "gfx936", True),
    ("load", True, "gfx936", True),
    ("alt", False, "gfx942", True),
    ("alt", False, "gfx936", False),
])
def test_readonly_interleave_native_reads_retain_fallback(tmp_path, native, volatile, arch, kernel):
    module, keepalive = _ready_shared_loads_readonly_interleave_fixture(tmp_path, native, volatile=volatile)
    if not kernel:
        function = next(f for f in module.get_functions() if f.name == "readonly_fixture")
        function.set_calling_conv(0)
    before = str(module)
    result = libtriton.hcu.interleave_global_lds_reads(module, arch, True)
    assert result["transformed_windows"] == 0 and str(module) == before


@pytest.mark.parametrize("native", ["normal", "alt"])
def test_readonly_interleave_native_reads_preserve_global_address_dependency(tmp_path, native):
    module, keepalive = _ready_shared_loads_readonly_interleave_fixture(tmp_path, native, dependent=True)
    before = str(module)
    result = libtriton.hcu.interleave_global_lds_reads(module, "gfx936", True)
    assert result["transformed_windows"] == 0 and result["rejected_dependencies"] == 1
    assert str(module) == before


# ready shared loads
import re
from triton._C.libtriton import ir, llvm

try:
    from triton._C.libtriton import hcu
except ImportError:
    pytest.skip("Requires the HCU backend", allow_module_level=True)

_ready_shared_loads_llvm_BASE = '''
declare void @llvm.amdgcn.raw.ptr.buffer.load.lds(ptr addrspace(8), ptr addrspace(3), i32 immarg, i32, i32, i32 immarg, i32 immarg)
declare <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3), i32)
declare void @llvm.amdgcn.s.waitcnt(i32 immarg)
declare void @llvm.amdgcn.s.barrier()
define void @fixture() {
entry:
  br label %loop
loop:
  call void @llvm.amdgcn.raw.ptr.buffer.load.lds(ptr addrspace(8) poison, ptr addrspace(3) poison, i32 8, i32 0, i32 0, i32 0, i32 0), !alias.scope !0
  call void @llvm.amdgcn.raw.ptr.buffer.load.lds(ptr addrspace(8) poison, ptr addrspace(3) poison, i32 8, i32 8, i32 0, i32 0, i32 0), !alias.scope !0
  %a = load <4 x i16>, ptr addrspace(3) poison, align 8, !noalias !0
  %b = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) poison, i32 0), !noalias !0
  br i1 true, label %loop, label %exit
exit:
  ret void
}
!0 = !{!1}
!1 = distinct !{!1, !2, !"async"}
!2 = distinct !{!2, !"pipeline"}
!3 = !{!4}
!4 = distinct !{!4, !2, !"different"}
'''


@pytest.mark.parametrize("case", ["mixed_reads", "vector_only", "single_copy"])
def test_ready_read_order_and_idempotence(tmp_path, case):
    text = _ready_shared_loads_llvm_BASE
    final_read = "%b = call"
    copy_count = 2
    if case == "vector_only":
        text = re.sub(r"^  %b = call.*\n", "", text, flags=re.M)
        final_read = "%a = load"
    elif case == "single_copy":
        text = re.sub(r"^  call void @llvm\.amdgcn\.raw\.ptr\.buffer\.load\.lds.*i32 8, i32 8,.*\n", "", text,
                      flags=re.M)
        copy_count = 1
    module, keep = _ready_shared_loads_llvm_load_fixture(tmp_path, text)
    assert hcu.prefer_ready_shared_loads(module, "gfx936") == 1
    after = str(module)
    body = after.split("define internal amdgpu_kernel", 1)[1].split("exit:", 1)[0]
    barrier = "call void @llvm.amdgcn.sched.barrier"
    copy = "call void @llvm.amdgcn.raw.ptr.buffer.load.lds"
    assert body.index(final_read) < body.index(barrier) < body.index(copy)
    assert body.count(copy) == copy_count
    assert body.count(barrier) == 1
    assert hcu.prefer_ready_shared_loads(module, "gfx936") == 0
    assert str(module) == after


@pytest.mark.parametrize(
    ("case", "expected"),
    [
        ("separate_wide_groups", [0, "lds", 16, 32]),
        ("larger_companion_group", [0, "lds", 16, 32, 48]),
        ("larger_single_scope", ["lds", 0, 16, 32, 48]),
        ("single_scope", ["lds", 0, 16, 32]),
        ("short_copy", ["lds", 0, 16, 32]),
        ("missing_exclusion", [0, 16, 32, "lds"]),
    ],
)
def test_single_copy_before_ready_reads(tmp_path, case, expected):
    # Preserve only a singleton wide copy when its scope differs from the
    # larger companion group and every read excludes both producer scopes.
    copy = next(line for line in _ready_shared_loads_llvm_BASE.splitlines()
                if line.startswith("  call void @llvm.amdgcn.raw.ptr.buffer.load.lds"))
    width = 8 if case == "short_copy" else 16
    copies = []
    offsets = (0, 16, 32, 48) if case.startswith("larger_") else (0, 16, 32)
    for offset in offsets:
        scope = "!0" if offset == 0 or case in ("single_scope", "larger_single_scope") else "!3"
        copies.append(
            copy.replace("i32 8, i32 0,", f"i32 {width}, i32 {offset},").replace("!alias.scope !0",
                                                                                 f"!alias.scope {scope}"))
    text = re.sub(
        r"^  call void @llvm\.amdgcn\.raw\.ptr\.buffer\.load\.lds[^\n]*\n",
        "",
        _ready_shared_loads_llvm_BASE,
        flags=re.M,
    ).replace("loop:\n", "loop:\n" + "\n".join(copies) + "\n", 1)
    if case != "missing_exclusion":
        text = text.replace("!noalias !0", "!noalias !5") + "\n!5 = !{!1, !4}\n"
    module, keep = _ready_shared_loads_llvm_load_fixture(tmp_path, text)
    changed = hcu.prefer_ready_shared_loads(module, "gfx936")
    assert changed == (case != "missing_exclusion")
    result = str(module)
    events = []
    for line in result.splitlines():
        if "load <4 x i16>" in line:
            events.append("lds")
        elif "call void @llvm.amdgcn.raw.ptr.buffer.load.lds(" in line:
            events.append(int(re.search(r"i32 \d+, i32 (\d+),", line)[1]))
    assert events == expected
    assert hcu.prefer_ready_shared_loads(module, "gfx936") == 0
    assert str(module) == result


@pytest.mark.parametrize('case', [
    'other_arch', 'missing_copy_scope', 'missing_read_noalias', 'different_scope', 'incomplete_scope', 'wait',
    'barrier', 'global_store', 'volatile', 'atomic', 'unknown_call', 'inline_asm', 'already_ordered', 'no_backedge'
])
def test_unsafe_or_inapplicable_unchanged(tmp_path, case):
    text = _ready_shared_loads_llvm_BASE
    if case == 'missing_copy_scope': text = text.replace(', !alias.scope !0', '')
    elif case == 'missing_read_noalias': text = text.replace(', !noalias !0', '')
    elif case == 'different_scope': text = text.replace('!noalias !0', '!noalias !3')
    elif case == 'incomplete_scope':
        text = text.replace('!0 = !{!1}', '!0 = !{!1, !4}').replace('!noalias !0', '!noalias !5') + '\n!5 = !{!1}\n'
    elif case in ['wait', 'barrier', 'global_store', 'unknown_call', 'inline_asm']:
        effect = {
            'wait': 'call void @llvm.amdgcn.s.waitcnt(i32 0)', 'barrier': 'call void @llvm.amdgcn.s.barrier()',
            'global_store': 'store i32 1, ptr addrspace(1) poison', 'unknown_call': 'call void @unknown()',
            'inline_asm': 'call void asm sideeffect "", ""()'
        }[case]
        text = text.replace('  %a = load', '  ' + effect + '\n  %a = load')
        if case == 'unknown_call': text = 'declare void @unknown()\n' + text
    elif case == 'volatile': text = text.replace('%a = load ', '%a = load volatile ')
    elif case == 'atomic':
        text = text.replace('%a = load <4 x i16>, ptr addrspace(3) poison, align 8',
                            '%a = load atomic i64, ptr addrspace(3) poison monotonic, align 8')
    elif case == 'already_ordered':
        lines = text.splitlines()
        copies = [line for line in lines if line.startswith('  call void @llvm.amdgcn.raw.ptr.buffer.load.lds')]
        lines = [line for line in lines if line not in copies]
        index = next(i for i, line in enumerate(lines) if line.startswith('  br i1'))
        text = '\n'.join(lines[:index] + copies + lines[index:])
    elif case == 'no_backedge':
        text = text.replace('br i1 true, label %loop, label %exit', 'br label %exit')
    module, keep = _ready_shared_loads_llvm_load_fixture(tmp_path, text)
    before = str(module)
    assert hcu.prefer_ready_shared_loads(module, 'gfx942' if case == 'other_arch' else 'gfx936') == 0
    assert str(module) == before


@pytest.mark.parametrize("structured", [(True, False, False), (True, True, True), (False, False, False)])
def test_issue_group_position(tmp_path, structured):
    text = _ready_shared_loads_llvm_BASE + "\n!5 = !{!1, !4}\n"
    copy = next(line for line in _ready_shared_loads_llvm_BASE.splitlines()
                if line.startswith("  call void @llvm.amdgcn.raw.ptr.buffer.load.lds"))
    decl = ("declare void @llvm.amdgcn.struct.ptr.buffer.load.lds("
            "ptr addrspace(8), ptr addrspace(3), i32 immarg, i32, i32, i32, i32 immarg, i32 immarg)\n")
    text = decl + re.sub(r"^  call void @llvm.amdgcn.raw.ptr.buffer.load.lds[^\n]*\n", "", text, flags=re.M)
    lines = []
    for offset, indexed in zip((0, 16, 32), structured):
        candidate = copy.replace("i32 8, i32 0,", f"i32 16, i32 {offset},")
        if offset:
            candidate = candidate.replace("!alias.scope !0", "!alias.scope !3")
        if indexed:
            candidate = candidate.replace("raw.ptr.buffer", "struct.ptr.buffer")
            candidate = candidate.replace("), !alias.scope", ", i32 0), !alias.scope")
        lines.append(candidate)
    text = text.replace("loop:\n", "loop:\n" + "\n".join(lines) + "\n").replace("!noalias !0", "!noalias !5")
    mod, live = _ready_shared_loads_llvm_load_fixture(tmp_path, text)
    assert hcu.prefer_ready_shared_loads(mod, "gfx936") == 1
    after = str(mod)
    events = []
    for line in after.splitlines():
        if "load <4 x i16>" in line:
            events.append("load")
        elif "= call <4 x i32> @llvm.hcu.ds.read" in line:
            events.append("read")
        elif "call void @llvm.amdgcn.sched.barrier" in line:
            events.append("fence")
        elif re.search(r"call void @llvm.amdgcn.(raw|struct).ptr.buffer.load.lds", line):
            events.append(int(re.search(r"i32 16, i32 (\d+),", line)[1]))
    if structured == (True, False, False):
        assert events == ["load", "fence", 0, "fence", "read", "fence", 16, 32]
    else:
        assert events == [0, "fence", "load", "read", "fence", 16, 32]
    assert hcu.prefer_ready_shared_loads(mod, "gfx936") == 0
    assert str(mod) == after


# local read prefetch order

_local_read_prefetch_order_LAYOUTS = r"""
#mma = #ttg.amd_mfma<{version = 3, warpsPerCTA = [2, 2], instrShape = [16, 16, 16], isTransposed = true, mmacLayout = 1}>
#shared = #ttg.shared_linear<{offset = [[0, 1], [0, 2], [0, 4], [0, 8], [0, 16], [1, 0], [2, 8], [4, 16], [8, 0], [16, 0], [32, 0]]}, alignment = 16>
#shared1 = #ttg.shared_linear<{offset = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0], [0, 1], [8, 2], [16, 4], [0, 8], [0, 16], [0, 32], [0, 64]]}, alignment = 16>
#smem = #ttg.shared_memory
"""


@pytest.mark.parametrize("dtype", ["bf16", "f16"])
@pytest.mark.parametrize("case", [
    "eligible", "one_side", "chain", "noncarried", "nohint", "wrong_target", "early_use", "barrier", "wait",
    "unknown_call"
])
def test_prefetched_reads_precede_math_without_moving_memory(tmp_path, dtype, case):
    a = f"tensor<64x32x{dtype}, #ttg.dot_op<{{opIdx = 0, parent = #mma, kWidth = 8}}>>"
    b = f"tensor<32x128x{dtype}, #ttg.dot_op<{{opIdx = 1, parent = #mma, kWidth = 8}}>>"
    acc = "tensor<64x128xf32, #mma>"
    ma = f"!ttg.memdesc<64x32x{dtype}, #shared, #smem, mutable>"
    mb = f"!ttg.memdesc<32x128x{dtype}, #shared1, #smem, mutable>"
    hint = "" if case == "nohint" else ", ttg.hcu.local_prefetch = 1 : i32"
    target = "gfx90a" if case == "wrong_target" else "gfx936"
    left = "%initialA" if case == "noncarried" else "%a"
    right = "%initialB" if case in ("noncarried", "one_side") else "%b"
    math = f"%current = tt.dot {left}, {right}, %held : {a} * {b} -> {acc}"
    result = "%current"
    if case == "chain":
        math += f"\n%second = tt.dot {left}, {right}, %current : {a} * {b} -> {acc}"
        result = "%second"
    if case == "early_use":
        math += f"\n%observed = arith.addf %current, %held : {acc}"
        result = "%observed"
    boundary = {
        "barrier": "ttg.local_barrier",
        "wait": "%wait = ttg.async_wait %ready {num = 0 : i32}",
        "unknown_call": "tt.call @opaque() : () -> ()",
    }.get(case, "")
    token = "%wait" if case == "wait" else "%ready"
    text = _local_read_prefetch_order_LAYOUTS + f"""
module attributes {{ttg.target = "hip:{target}", "ttg.num-ctas" = 1 : i32,
    "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 64 : i32{hint}}} {{
 tt.func private @opaque()
 tt.func public @prefetch(%memA: {ma}, %memB: {mb},
     %initialA: {a}, %initialB: {b}, %initial: {acc},
     %ready: !ttg.async.token, %count: i32) -> ({acc}, {a}, {b}) {{
  %zero = arith.constant 0 : i32
  %one = arith.constant 1 : i32
  %out:3 = scf.for %i = %zero to %count step %one
      iter_args(%held = %initial, %a = %initialA, %b = %initialB)
      -> ({acc}, {a}, {b}) : i32 {{
   {math}
   {boundary}
   ttg.local_store %initialA, %memA : {a} -> {ma}
   %nextA = ttg.local_load %memA token {token} : {ma} -> {a}
   ttg.local_store %initialB, %memB : {b} -> {mb}
   %nextB = ttg.local_load %memB token {token} : {mb} -> {b}
   scf.yield {result}, %nextA, %nextB : {acc}, {a}, {b}
  }}
  tt.return %out#0, %out#1, %out#2 : {acc}, {a}, {b}
 }}
}}
"""
    path = tmp_path / "prefetch.mlir"
    path.write_text(text)
    context = libtriton.ir.context()
    libtriton.ir.load_dialects(context)
    HIPBackend(GPUTarget("hip", "gfx936", 64)).load_dialects(context)
    module = libtriton.ir.parse_mlir_module(str(path), context)
    before = str(module)
    manager = libtriton.ir.pass_manager(context)
    libtriton.hcu.passes.ttgpuir.add_mls_lowering_pass(manager)
    manager.run(module, "loop_carried_local_prefetch_order")
    after = str(module)
    dots = [x.start() for x in re.finditer(r"= tt.dot ", after)]
    reads = [x.start() for x in re.finditer(r"= ttg.local_load ", after)]
    assert len(dots) == (2 if case == "chain" else 1)
    assert len(reads) == 2
    if case in ("eligible", "chain"):
        assert all(dot > reads[-1] for dot in dots)
    elif case == "one_side":
        assert dots[0] > reads[0]
    else:
        assert dots[0] < reads[0]
    memory = r"ttg\.(?:local_store|local_load|local_barrier|async_wait)|tt\.call"
    assert re.findall(memory, before) == re.findall(memory, after)
    assert "rocdl.sched.barrier" not in after


# tle local pointer read order

if not hasattr(libtriton, "hcu") or not hasattr(libtriton, "tle"):
    pytest.skip("HCU/TLE compiler required", allow_module_level=True)


@pytest.mark.parametrize("warp_size", [32, 64])
@pytest.mark.parametrize("paired_reads", [False, True])
def test_layout_rematerialization_keeps_read_before_overwrite(tmp_path, warp_size, paired_reads):
    target = "cuda:90" if warp_size == 32 else "hip:gfx936"
    second_read = """
    %later = tt.load %ptr : tensor<128x!tt.ptr<i32, 3>, #blocked>
    %sum = arith.addi %first, %later : tensor<128xi32, #blocked>
""" if paired_reads else ""
    value = "%sum" if paired_reads else "%first"
    source = f"""
#blocked = #ttg.blocked<{{sizePerThread = [1], threadsPerWarp = [{warp_size}], warpsPerCTA = [2], order = [0]}}>
#other = #ttg.blocked<{{sizePerThread = [2], threadsPerWarp = [{warp_size}], warpsPerCTA = [2], order = [0]}}>
#shared = #ttg.swizzled_shared<{{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}}>
#smem = #ttg.shared_memory
module attributes {{"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 2 : i32, ttg.target = "{target}", "ttg.threads-per-warp" = {warp_size} : i32}} {{
  tt.func public @read_before_overwrite(%mem: !ttg.memdesc<2x128xi32, #shared, #smem, mutable>, %slot: i32) -> tensor<128xi32, #other> {{
    %row = tt.splat %slot : i32 -> tensor<128xi32, #blocked>
    %col = tt.make_range {{end = 128 : i32, start = 0 : i32}} : tensor<128xi32, #blocked>
    %ptr = "tle.local_pointers"(%mem, %row, %col) : (!ttg.memdesc<2x128xi32, #shared, #smem, mutable>, tensor<128xi32, #blocked>, tensor<128xi32, #blocked>) -> tensor<128x!tt.ptr<i32, 3>, #blocked>
    %first = tt.load %ptr : tensor<128x!tt.ptr<i32, 3>, #blocked>
    ttg.local_barrier
    %zero = arith.constant dense<0> : tensor<128xi32, #blocked>
    tt.store %ptr, %zero : tensor<128x!tt.ptr<i32, 3>, #blocked>
    {second_read}
    %out = ttg.convert_layout {value} : tensor<128xi32, #blocked> -> tensor<128xi32, #other>
    tt.return %out : tensor<128xi32, #other>
  }}
}}
"""
    path = tmp_path / "read_order.mlir"
    path.write_text(source)
    context = libtriton.ir.context()
    libtriton.ir.load_dialects(context)
    HIPBackend(GPUTarget("hip", "gfx936", 64)).load_dialects(context)
    libtriton.tle.load_dialects(context)
    module = libtriton.ir.parse_mlir_module(str(path), context)
    manager = libtriton.ir.pass_manager(context)
    libtriton.tle.passes.add_optimize_local_pointer_loads(manager)
    manager.run(module, "preserve_shared_read_position")
    after = str(module)
    loads = [m.start() for m in re.finditer(r"\btt.load ", after)]
    barrier = after.index("ttg.local_barrier")
    store = after.index("tt.store")
    assert len(loads) == (2 if paired_reads else 1)
    assert loads[0] < barrier < store
    if paired_reads:
        assert store < loads[1]
    assert "ttg.convert_layout" not in after
