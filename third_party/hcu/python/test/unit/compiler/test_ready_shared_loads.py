"""Exercise the real LLVM helper with positive and unsafe-to-move fixtures."""
import re
import pytest
from triton._C.libtriton import ir, llvm

try:
    from triton._C.libtriton import hcu
except ImportError:
    pytest.skip("Requires the HCU backend", allow_module_level=True)

BASE = '''
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


def load_fixture(tmp_path, text):
    # Use existing public LLVM-library import: this exercises the built helper,
    # not a Python implementation of its legality checks.
    stub = tmp_path / 'root.mlir'
    stub.write_text('module { llvm.func @fixture() llvm.func @root() { llvm.call @fixture() : () -> () llvm.return } }')
    source = tmp_path / 'fixture.ll'
    source.write_text(text)
    context = ir.context()
    ir.load_dialects(context)
    hcu.load_dialects(context)
    mlir = ir.parse_mlir_module(str(stub), context)
    llvm_context = llvm.context()
    module = llvm.to_module(mlir, llvm_context)
    llvm.link_extern_libs(module, [str(source)])
    fixture = next(fn for fn in module.get_functions() if fn.name == 'fixture')
    fixture.set_calling_conv(hcu.CALLING_CONV_AMDGPU_KERNEL)
    return module, (context, mlir, llvm_context)


@pytest.mark.parametrize("case", ["mixed_reads", "vector_only", "single_copy"])
def test_ready_read_order_and_idempotence(tmp_path, case):
    text = BASE
    final_read = "%b = call"
    copy_count = 2
    if case == "vector_only":
        text = re.sub(r"^  %b = call.*\n", "", text, flags=re.M)
        final_read = "%a = load"
    elif case == "single_copy":
        text = re.sub(r"^  call void @llvm\.amdgcn\.raw\.ptr\.buffer\.load\.lds.*i32 8, i32 8,.*\n", "", text,
                      flags=re.M)
        copy_count = 1
    module, keep = load_fixture(tmp_path, text)
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
        ("single_scope", ["lds", 0, 16, 32]),
        ("short_copy", ["lds", 0, 16, 32]),
        ("missing_exclusion", [0, 16, 32, "lds"]),
    ],
)
def test_single_copy_before_ready_reads(tmp_path, case, expected):
    # Preserve only a singleton wide copy when its scope differs from the
    # following two and every read excludes all three copy scopes.
    copy = next(line for line in BASE.splitlines()
                if line.startswith("  call void @llvm.amdgcn.raw.ptr.buffer.load.lds"))
    width = 8 if case == "short_copy" else 16
    copies = []
    for offset in (0, 16, 32):
        scope = "!0" if offset == 0 or case == "single_scope" else "!3"
        copies.append(
            copy.replace("i32 8, i32 0,", f"i32 {width}, i32 {offset},").replace("!alias.scope !0",
                                                                                 f"!alias.scope {scope}"))
    text = re.sub(
        r"^  call void @llvm\.amdgcn\.raw\.ptr\.buffer\.load\.lds[^\n]*\n",
        "",
        BASE,
        flags=re.M,
    ).replace("loop:\n", "loop:\n" + "\n".join(copies) + "\n", 1)
    if case != "missing_exclusion":
        text = text.replace("!noalias !0", "!noalias !5") + "\n!5 = !{!1, !4}\n"
    module, keep = load_fixture(tmp_path, text)
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
    text = BASE
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
    module, keep = load_fixture(tmp_path, text)
    before = str(module)
    assert hcu.prefer_ready_shared_loads(module, 'gfx942' if case == 'other_arch' else 'gfx936') == 0
    assert str(module) == before


@pytest.mark.parametrize("structured", [(True, False, False), (True, True, True), (False, False, False)])
def test_issue_group_position(tmp_path, structured):
    text = BASE + "\n!5 = !{!1, !4}\n"
    copy = next(line for line in BASE.splitlines()
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
    mod, live = load_fixture(tmp_path, text)
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
