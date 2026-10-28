"""Unit tests for the LLVM 22 -> 19 IR downgrade.

The downgrade is pure text in / text out, and every rewrite it performs is
claimed to be semantics-preserving.  These tests pin that claim down: each
rewrite gets an input/output fixture, and the "detected but not rewritten"
constructs get a fixture asserting the tool complains instead of silently
passing them through.

Run with:  pytest python/test/unit/backends/test_llvm_ir_downgrade.py
"""

import os
import subprocess
import sys
from pathlib import Path

import pytest

# Import the module directly from the source tree so the test does not depend
# on triton being installed (it has no triton imports of its own).
#   .../triton/python/test/unit/backends/test_llvm_ir_downgrade.py
#   parents[0]=backends  [1]=unit  [2]=test  [3]=python  [4]=<repo root>
_REPO_ROOT = Path(__file__).resolve().parents[4]
_BACKENDS_DIR = _REPO_ROOT / "python" / "triton" / "backends"
sys.path.insert(0, str(_BACKENDS_DIR))
import llvm_ir_downgrade as dg  # noqa: E402
import llvm19_toolchain  # noqa: E402


def _down(ir: str) -> str:
    out, warnings = dg.downgrade(ir)
    assert not warnings, f"unexpected warnings: {warnings}"
    return out


# --------------------------------------------------------------------------
# 1. captures(...) parameter attribute (LLVM 20+)
# --------------------------------------------------------------------------


def test_captures_none_mapped_to_nocapture():
    """LLVM 19 spells the very same fact `nocapture`."""
    ir = "define void @f(ptr addrspace(1) readonly captures(none) %0) {\n"
    assert _down(ir) == "define void @f(ptr addrspace(1) readonly nocapture %0) {\n"


def test_captures_none_only_form_mapped():
    ir = "define void @f(ptr captures(none) %0) {\n"
    assert _down(ir) == "define void @f(ptr nocapture %0) {\n"


def test_captures_none_before_comma_gains_no_space():
    # 73 call sites in the recorded corpus end the attribute with `,`; the
    # rewrite must not leave a space in front of it.
    ir = "define void @f(ptr captures(none), i32 %n) {\n"
    assert _down(ir) == "define void @f(ptr nocapture, i32 %n) {\n"


def test_nontrivial_captures_kept_and_reported():
    """A non-none payload changes semantics if stripped -> must not be silent."""
    ir = "define void @f(ptr captures(address) %0) {\n"
    out, warnings = dg.downgrade(ir)
    assert "captures(address)" in out
    assert any("captures(address)" in w for w in warnings)


# --------------------------------------------------------------------------
# 2. LLVM 20+ no-op parameter attributes
# --------------------------------------------------------------------------


@pytest.mark.parametrize("attr", ["dead_on_return", "nocreateundeforpoison", "noext"])
def test_noop_param_attrs_dropped(attr):
    ir = f"define void @f(ptr {attr} %0) {{\n"
    assert attr not in _down(ir)


# --------------------------------------------------------------------------
# 3. getelementptr nuw / nusw flags (LLVM 20+)
# --------------------------------------------------------------------------


def test_gep_inbounds_nuw_dropped():
    ir = "  %p = getelementptr inbounds nuw i8, ptr addrspace(3) @g, i32 %x\n"
    out = _down(ir)
    assert out == "  %p = getelementptr inbounds i8, ptr addrspace(3) @g, i32 %x\n"


def test_gep_bare_nuw_dropped():
    ir = "  %p = getelementptr nuw i8, ptr %b, i32 %x\n"
    assert _down(ir) == "  %p = getelementptr i8, ptr %b, i32 %x\n"


def test_gep_nusw_dropped():
    ir = "  %p = getelementptr inbounds nusw i32, ptr %b, i32 %x\n"
    assert "nusw" not in _down(ir)


def test_add_nuw_untouched():
    """The same token on a non-gep instruction must survive."""
    ir = "  %s = add nuw i32 %a, %b\n"
    assert _down(ir) == ir


# --------------------------------------------------------------------------
# 4. icmp samesign (LLVM 20+)
# --------------------------------------------------------------------------


def test_icmp_samesign_dropped():
    ir = "  %c = icmp samesign ult i32 %11, 4\n"
    assert _down(ir) == "  %c = icmp ult i32 %11, 4\n"


def test_icmp_without_samesign_untouched():
    ir = "  %c = icmp ult i32 %11, 4\n"
    assert _down(ir) == ir


# --------------------------------------------------------------------------
# 5. memory(...) locations LLVM 19 has no spelling for (LLVM 20+/21+)
# --------------------------------------------------------------------------


def test_target_mem_locations_stripped():
    ir = "attributes #0 = { memory(none, target_mem0: none, target_mem1: none) }\n"
    out = _down(ir)
    assert "target_mem" not in out
    assert "memory(none)" in out


def test_memory_attr_without_target_mem_untouched():
    ir = "attributes #0 = { memory(argmem: read) }\n"
    assert _down(ir) == ir


@pytest.mark.parametrize("payload,expected", [
    ("memory(read, errnomem: none)", "memory(read)"),
    ("memory(write, errnomem: none)", "memory(write)"),
])
def test_errno_memory_location_is_rewritten(payload, expected):
    """The XPU frontend prints the errno location on the TLE vector load/store
    declarations (`llvm.xpu.vload_mz` / `vstore_mh`) and the LLVM 19 reader
    rejects it outright -- both shapes below are from a real module."""
    ir = f"attributes #0 = {{ {payload} }}\n"
    assert _down(ir) == f"attributes #0 = {{ {expected} }}\n"


def test_extra_memory_location_widens_the_default():
    """LLVM 19's "other" is the location that covers errno/target memory, so a
    non-none value there must widen the default, not vanish with the entry."""
    assert _down("attributes #0 = { memory(read, errnomem: write) }\n") == \
        "attributes #0 = { memory(readwrite) }\n"
    assert _down("attributes #0 = { memory(none, target_mem0: read) }\n") == \
        "attributes #0 = { memory(read) }\n"


def test_extra_memory_location_keeps_the_named_locations():
    """Folding must not disturb argmem/inaccessiblemem -- and when the payload
    had no leading behaviour, the folded one has to be printed, since an absent
    default reads as `none`."""
    assert _down("attributes #0 = { memory(argmem: readwrite, errnomem: read) }\n") == \
        "attributes #0 = { memory(read, argmem: readwrite) }\n"


# --------------------------------------------------------------------------
# 6. Calling-convention spellings (numeric collision, text differs)
# --------------------------------------------------------------------------


def test_xcn_kernel_cc_spelling_restored():
    ir = "define cheriot_compartmentcallcc void @k() {\n"
    assert _down(ir) == "define xcn_kernel void @k() {\n"


def test_xpu_kernel_cc_spelling_restored():
    ir = "define amdgpu_gfx_whole_wave void @k() {\n"
    assert _down(ir) == "define xpu_kernel void @k() {\n"


# --------------------------------------------------------------------------
# 7. "; Unknown intrinsic" comment stripping (cosmetic)
# --------------------------------------------------------------------------


def test_unknown_intrinsic_comment_stripped():
    ir = "; Unknown intrinsic\ndeclare i32 @llvm.xcn.rcp(i32)\n"
    out = _down(ir)
    assert "Unknown intrinsic" not in out
    assert "@llvm.xcn.rcp" in out


# --------------------------------------------------------------------------
# bf16-family name restore: whole-module invariant (asserted on both legs)
#
# The private `llvm.<vendor>.v.*bf16*` family are the packed-pair and scalar bf16 intrinsics.  The
# XTDK-frontend leg emits them *unprefixed* -- its LLVM 22 tables reject a
# prefixed declaration, the LLVM 19 llc rejects an unprefixed call -- so that
# leg's downgrade restores the prefix (step 4c); on the public-LLVM22 leg the
# frontend emits the prefixed name itself and the step is a no-op.  The
# invariant below keeps that round trip honest: after the downgrade no
# bf16-family name may still be unprefixed, and none may be dropped or
# double-prefixed.  An unprefixed private name is not inert -- llc turns it
# into an ordinary external function, so it would only surface at link or run
# time -- which is why the invariant is asserted at module granularity rather
# than name by name.
# --------------------------------------------------------------------------

_BF16_PLAIN_MODULE = """\
declare <2 x bfloat> @xcn.v.pk.add.bf16(<2 x bfloat>, <2 x bfloat>) local_unnamed_addr
declare bfloat @xcn.v.add.bf16(bfloat, bfloat) local_unnamed_addr

define void @bf16_add(<2 x bfloat> %a, <2 x bfloat> %b) {
entry:
  %p = tail call <2 x bfloat> @xcn.v.pk.add.bf16(<2 x bfloat> %a, <2 x bfloat> %b)
  %s = tail call bfloat @xcn.v.add.bf16(bfloat 1.000000e+00, bfloat 2.000000e+00)
  ret void
}
"""

# What the *other* leg's frontend hands the downgrade: the same module with the
# intrinsic prefix already on the names, and no restore step to perform.
_BF16_MODULE = (_BF16_PLAIN_MODULE if hasattr(dg, "_XCN_PLAIN_BF16_RE") else _BF16_PLAIN_MODULE.replace(
    "@xcn.v.", "@llvm.xcn.v."))


def test_bf16_family_names_are_all_prefixed_after_downgrade():
    out = _down(_BF16_MODULE)
    # No unprefixed private name survives, in a declaration or in a call.
    assert "@xcn." not in out, out
    # Every name that went in came out with the intrinsic prefix: none was
    # dropped and none was prefixed twice.
    assert out.count("@llvm.xcn.v.") == _BF16_MODULE.count("xcn.v.")
    # The restore is name-only: no line is added or removed.
    assert out.count("\n") == _BF16_MODULE.count("\n")


# --------------------------------------------------------------------------
# 9. Constructs that must fail loudly rather than be rewritten
# --------------------------------------------------------------------------


def test_ptrtoaddr_reported():
    ir = "  %a = ptrtoaddr ptr %p to i64\n"
    _, warnings = dg.downgrade(ir)
    assert any("ptrtoaddr" in w for w in warnings)


@pytest.mark.parametrize("intr,family", [
    ("llvm.atan2.f32", "llvm.atan2"),
    ("llvm.vector.interleave2.v4i32", "llvm.vector.interleave"),
    ("llvm.vector.deinterleave2.v4i32", "llvm.vector.deinterleave"),
])
def test_unsupported_intrinsics_reported(intr, family):
    """Detection is by family prefix, so overloaded variants are all caught."""
    ir = f"  %r = call float @{intr}(float %a)\n"
    _, warnings = dg.downgrade(ir)
    assert any(family in w for w in warnings), warnings


def test_cli_exits_nonzero_on_unsupported(tmp_path):
    src = tmp_path / "in.ll"
    src.write_text("  %a = ptrtoaddr ptr %p to i64\n")
    # There is no `llvm_ir_downgrade.py` to hand to the interpreter any more, and
    # `-m` is not an option either: runpy needs the module's *code object*, which
    # a compiled extension does not have (`No code object available for ...`).
    # A plain import plus an explicit `main()` is the entry that survives.
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join([str(_BACKENDS_DIR), env.get("PYTHONPATH", "")]).rstrip(os.pathsep)
    r = subprocess.run(
        [sys.executable, "-c", "import sys, llvm_ir_downgrade as m; sys.exit(m.main())",
         str(src)],
        capture_output=True,
        text=True,
        env=env,
    )
    assert r.returncode == 1
    assert "ptrtoaddr" in r.stderr


# --------------------------------------------------------------------------
# 10. Whole-module smoke: nothing else is disturbed
# --------------------------------------------------------------------------

_SAMPLE = """\
; ModuleID = 'add_kernel'
target datalayout = "e-p:64:64-i64:64-n32:64-S32-A5-G1-ni:9"
target triple = "xcn-xcn-xcnpkg"

declare i32 @llvm.xcn.workitem.id.x() #0

define xcn_kernel void @add(ptr addrspace(1) readonly captures(none) %in,
                            ptr addrspace(1) %out, i32 %n) #1 {
entry:
  %tid = call i32 @llvm.xcn.workitem.id.x()
  %cmp = icmp samesign ult i32 %tid, %n
  br i1 %cmp, label %body, label %exit

body:
  %gep = getelementptr inbounds nuw float, ptr addrspace(1) %in, i32 %tid
  %v = load float, ptr addrspace(1) %gep, align 4
  %s = fadd float %v, 1.0
  %o = getelementptr inbounds nuw float, ptr addrspace(1) %out, i32 %tid
  store float %s, ptr addrspace(1) %o, align 4
  br label %exit

exit:
  ret void
}

attributes #0 = { memory(none, target_mem0: none, target_mem1: none) }
attributes #1 = { "target-cpu"="xcn" }
attributes #2 = { nofree nounwind memory(read, errnomem: none) }
"""


def test_sample_module_fully_downgraded():
    out = _down(_SAMPLE)
    # No LLVM 20+ surface syntax survives.
    for token in ("captures(", "samesign", "target_mem", "errnomem"):
        assert token not in out, f"{token!r} survived the downgrade"
    assert "getelementptr inbounds nuw" not in out
    # Everything else is preserved verbatim.
    assert 'target triple = "xcn-xcn-xcnpkg"' in out
    assert "define xcn_kernel void @add(" in out
    assert "%s = fadd float %v, 1.0" in out
    assert '"target-cpu"="xcn"' in out
    # Line count is unchanged: rewrites are in-place, never line-structural.
    assert out.count("\n") == _SAMPLE.count("\n")


def test_sample_module_is_idempotent():
    once = _down(_SAMPLE)
    assert _down(once) == once


# --------------------------------------------------------------------------
# 11. Kernel calling-convention guard (llvm19_toolchain.check_kernel_cc_downgraded)
#
# The guard is the fail-loud backstop for the amdgpu_gfx_whole_wave -> xpu_kernel
# / cheriot_compartmentcallcc -> cluster-kernel text rewrite.  If a future LLVM
# prebuilt renames cc 124/125, the rewrite stops matching and every kernel would
# build but fail to launch.  The guard raises instead.
# --------------------------------------------------------------------------


def test_kernel_cc_guard_passes_when_rewritten():
    # Public spelling in, XTDK spelling out -> no error.
    ir = "define amdgpu_gfx_whole_wave void @k() {\n"
    out = "define xpu_kernel void @k() {\n"
    llvm19_toolchain.check_kernel_cc_downgraded(ir, out)


def test_kernel_cc_guard_raises_when_rewrite_stopped():
    # The scenario the guard exists for: the rewrite no longer rewrote the
    # public spelling (e.g. the LLVM prebuilt now prints a different name),
    # so the output still carries the public spelling.
    ir = "define amdgpu_gfx_whole_wave void @k() {\n"
    out = "define amdgpu_gfx_whole_wave void @k() {\n"
    with pytest.raises(RuntimeError, match="kernel calling convention"):
        llvm19_toolchain.check_kernel_cc_downgraded(ir, out)


def test_kernel_cc_guard_covers_wavecc_suffix():
    # The downgrade rewrite matches the `cc`-suffixed form
    # (amdgpu_gfx_whole_wavecc); the guard must arm on it too.
    ir = "define amdgpu_gfx_whole_wavecc void @k() {\n"
    out = "define xpu_kernel void @k() {\n"
    llvm19_toolchain.check_kernel_cc_downgraded(ir, out)


def test_kernel_cc_guard_xcn_kernel():
    ir = "define cheriot_compartmentcallcc void @k() {\n"
    out = "define xcn_kernel void @k() {\n"
    llvm19_toolchain.check_kernel_cc_downgraded(ir, out)


def test_kernel_cc_guard_no_public_spelling_is_noop():
    # No public spelling in the input -> the guard must not raise, even if the
    # output (for an unrelated reason) also lacks the XTDK spelling.
    llvm19_toolchain.check_kernel_cc_downgraded("define void @k() {\n", "define void @k() {\n")


def test_cc_bridge_is_single_source_of_truth():
    # The rewrite, the guard regexes, and the bridge table must be mutually
    # consistent: every public spelling in CALLING_CONV_BRIDGE is rewritten to
    # its XTDK name, and the guard arms on exactly those spellings.  This pins
    # the "single source of truth" contract so a future edit can't desync them.
    for public, xtdk in dg.CALLING_CONV_BRIDGE.items():
        assert dg.rewrite_calling_convs(f"define {public} void @k() {{\n") == (f"define {xtdk} void @k() {{\n")
        # The guard must raise when the rewrite is (hypothetically) a no-op.
        with pytest.raises(RuntimeError, match="kernel calling convention"):
            llvm19_toolchain.check_kernel_cc_downgraded(
                f"define {public} void @k() {{\n",
                f"define {public} void @k() {{\n",
            )
        # The guard regexes are derived from the same table.
        assert dg._CC_PUBLIC_SPELLING_RE.search(f"define {public} void")
        assert dg._CC_XTDK_NAME_RE.search(f"define {xtdk} void")


def test_cc_rewrite_matches_downgrade_path():
    # downgrade() itself goes through rewrite_calling_convs (not a parallel
    # inline regex), so the whole-module path and the direct helper agree.
    ir = "define amdgpu_gfx_whole_wave void @k() {\n"
    out, warnings = dg.downgrade(ir)
    assert not warnings
    assert out == dg.rewrite_calling_convs(ir)


def test_llvm19_env_prepends_staged_lib(tmp_path, monkeypatch):
    """Staged LLVM 19 tools must resolve their own libLLVM.so.19.1.

    The private cluster intrinsics the frontend emits as plain calls are lowered by
    code inside libLLVM (llc is only a driver), and DT_RUNPATH loses to
    LD_LIBRARY_PATH.  Ambient XPU runtimes ship an older libLLVM.so.19.1 there,
    whose cluster target leaves those calls as external symbols, so every affected
    kernel dies at the link step with
    "ld.lld: error: undefined symbol: llvm.xcn.low.prec.fdiv".  Prepending the
    staged lib dir is what keeps the toolchain self-consistent.
    """
    bin_dir = tmp_path / "llvm19" / "bin"
    lib_dir = tmp_path / "llvm19" / "lib"
    bin_dir.mkdir(parents=True)
    lib_dir.mkdir()
    monkeypatch.setenv(llvm19_toolchain._LLVM19_BIN_DIR_ENV, str(bin_dir))

    # A per-backend copy only participates when it actually holds the shlib
    # (setup.py stages it once under the shared dir otherwise), so give this one
    # a file: the test then pins the *ordering* rule instead of the layout.
    (lib_dir / "libLLVM.so.19.1").write_bytes(b"")
    monkeypatch.setenv("LD_LIBRARY_PATH", "/ambient/old-libLLVM")
    env = llvm19_toolchain.llvm19_env("xcn")
    entries = env["LD_LIBRARY_PATH"].split(":")
    assert entries[0] == str(lib_dir)  # staged copy wins
    assert entries[-1] == "/ambient/old-libLLVM"  # ambient stays last
    # The rest of the environment is passed through untouched.
    monkeypatch.setenv("TRITON_CACHE_DIR", "/tmp/llvm19-env-probe")
    assert llvm19_toolchain.llvm19_env("xcn")["TRITON_CACHE_DIR"] == "/tmp/llvm19-env-probe"

    # No ambient entry: no leading empty path element either.
    monkeypatch.delenv("LD_LIBRARY_PATH")
    assert llvm19_toolchain.llvm19_env("xcn")["LD_LIBRARY_PATH"].split(":")[0] == str(lib_dir)


def test_llvm19_env_falls_back_to_the_shared_runtime(tmp_path, monkeypatch):
    """The default layout stages the shlib once; that dir must be prepended.

    This is the contract setup.py::_stage_shared_llvm19 relies on: no
    per-backend copy exists any more, so the tools resolve libLLVM through this
    entry (and through the $ORIGIN-relative RPATH the staging forces).
    """
    bin_dir = tmp_path / "llvm19" / "bin"
    bin_dir.mkdir(parents=True)
    (tmp_path / "llvm19" / "lib").mkdir()  # empty: no per-dir copy
    monkeypatch.setenv(llvm19_toolchain._LLVM19_BIN_DIR_ENV, str(bin_dir))
    shared = llvm19_toolchain._shared_llvm19_lib_dir("xcn")
    if shared is None:
        pytest.skip("no shared LLVM 19 runtime staged in this tree")

    monkeypatch.setenv("LD_LIBRARY_PATH", "/ambient/old-libLLVM")
    entries = llvm19_toolchain.llvm19_env("xcn")["LD_LIBRARY_PATH"].split(":")
    assert entries[0] == str(shared)  # nothing staged ahead of it
    assert entries[-1] == "/ambient/old-libLLVM"


# --------------------------------------------------------------------------
# 12. Workgroup barrier fence wrapping
#
# The public LLVM 22 frontend does not emit the release/acquire fence pair
# around the cluster barrier intrinsics that XTDK's own lowering produces, so the
# downgrade adds it.  The XTDK leg *does* emit the pair itself, so the wrap
# has to be line-anchored (a comment that mentions a barrier call is not an
# instruction) and idempotent (a call already between the pair must be left
# alone, and a one-sided fence must be completed rather than taken for a
# finished pair).  The first two properties were violated before this section
# existed: a comment injected real fence instructions, and every re-run doubled
# them.
# --------------------------------------------------------------------------


def test_barrier_call_gets_fence_pair():
    ir = "define void @k() {\n  call void @llvm.xcn.s.barrier()\n  ret void\n}\n"
    assert _down(ir) == ("define void @k() {\n"
                         '  fence syncscope("workgroup") release\n'
                         "  call void @llvm.xcn.s.barrier()\n"
                         '  fence syncscope("workgroup") acquire\n'
                         "  ret void\n}\n")


def test_barrier_fence_wrap_is_idempotent():
    # The XTDK-leg shape: the pair is already there, so nothing is added and a
    # second pass changes nothing.
    ir = ("define void @k() {\n"
          '  fence syncscope("workgroup") release\n'
          "  tail call void @llvm.xcn.s.barrier() #2\n"
          '  fence syncscope("workgroup") acquire\n'
          "  ret void\n}\n")
    once = _down(ir)
    assert once == ir
    assert once.count('fence syncscope("workgroup")') == 2
    assert _down(once) == once


def test_barrier_fence_half_pair_is_completed():
    """One-sided fence: the missing side is added, the present one is reused.

    The skip criterion reads *both* sides of the call.  Checking only the
    release side (as the first version did) accepted `release + call` with no
    acquire and left that barrier without the acquire fence, which is the same
    reordering hazard the pair exists to prevent -- one fence short rather than
    one fence too many.  Completing the pair with just the missing fence also
    keeps `fences == 2 * barriers` true of the output.
    """
    release = '  fence syncscope("workgroup") release\n'
    call = "  call void @llvm.xcn.s.barrier()\n"
    acquire = '  fence syncscope("workgroup") acquire\n'
    head, tail = "define void @k() {\n", "  ret void\n}\n"
    want = head + release + call + acquire + tail
    for src in (head + release + call + tail,  # acquire side missing
                head + call + acquire + tail,  # release side missing
                ):
        out = _down(src)
        assert out == want
        assert out.count('fence syncscope("workgroup")') == 2
        assert _down(out) == out  # completing is idempotent too


def test_barrier_mentioned_in_a_comment_is_not_an_instruction():
    ir = "; see call void @llvm.xcn.s.barrier() above\ndefine void @k() {\n  ret void\n}\n"
    assert _down(ir) == ir
