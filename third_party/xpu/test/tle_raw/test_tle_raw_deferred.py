"""Tests for how a `tle.raw` payload reaches the kernel: the source store and the
LLVM 19 merge.

A payload records only a content-addressed id at trace time. The backend
compiles it for the arch it is building for and merges the text into the kernel
module right before `llc` (`triton/experimental/tle/raw/merge.py`), which is what
keeps the payload inlined: the call the raw op lowers to carries `alwaysinline`,
and LLVM only honours that within one module.

None of this needs an XPU device: the compiler-only parts run anywhere, the
merge tests feed hand-written LLVM 19 IR to the LLVM 19 `llvm-link` the backend
ships, and the payload frontend tests use stub handles.
"""

import os

import pytest

from triton._C.libtriton import gluon_ir, ir, xpu
from triton.experimental.tle.raw import registry
from triton.experimental.tle.raw.merge import (
    TLE_RAW_EXTERN_LIBS_KEY,
    TLE_RAW_PAYLOADS_KEY,
    merge_raw_payloads,
    record_raw_extern_libs,
    record_raw_payloads,
)
from triton.experimental.tle.raw.runtime import XCNJITFunction, XPUJITFunction
from triton.experimental.tle.raw.source_store import (
    clear_pending_sources,
    list_pending_sources,
    payload_ll,
    record_eager,
    register_source,
)

pytestmark = pytest.mark.no_xpu_required

XPU_ARCH = 5
BACKEND = "xpu"

PAYLOAD_IR = '''
target datalayout = "e-p:64:64"
target triple = "xcn-xcn-xcnpkg"

@llvm.compiler.used = appending addrspace(1) global [1 x ptr] [ptr @my_scale], section "llvm.metadata"

define internal void @my_scale(ptr addrspace(1) %out, i32 %n) #0 {
  ret void
}

attributes #0 = { nounwind }
'''

KERNEL_IR = '''
target datalayout = "e-m:e-p:32:32-p1:64:64"
target triple = "xcn-xcn-xcnpkg"

declare void @my_scale(ptr addrspace(1), i32)

define xcn_kernel void @raw_kernel(ptr addrspace(1) %0, i32 %1) {
  call void @my_scale(ptr addrspace(1) %0, i32 %1) #0
  ret void
}

attributes #0 = { alwaysinline }
'''

# The same kernel, with the payload's last parameter declared as i64. `llvm-link`
# links this happily and `llc` codegens it with an out-of-line call, so nothing
# downstream catches it.
MISMATCHED_KERNEL_IR = KERNEL_IR.replace("declare void @my_scale(ptr addrspace(1), i32)",
                                         "declare void @my_scale(ptr addrspace(1), i64)").replace(
                                             "ptr addrspace(1) %0, i32 %1) {\n  call",
                                             "ptr addrspace(1) %0, i64 %1) {\n  call").replace(
                                                 "i32 %1) #0", "i64 %1) #0")


@pytest.fixture(autouse=True)
def _clean_source_store():
    clear_pending_sources()
    yield
    clear_pending_sources()


class _StubHandle:
    """Payload handle that needs no toolchain; records the arch it got."""

    def __init__(self, llvm_ir=PAYLOAD_IR):
        self.arches = []
        self.llvm_ir = llvm_ir

    def make_llvm(self, context=None, arch=None):
        self.arches.append(arch)
        return self.llvm_ir


def _handle(tmp_path, dialect="xpu3", source='extern "C" __device__ void my_scale() {}', **kwargs):
    tmp_path.mkdir(parents=True, exist_ok=True)
    path = tmp_path / "my_scale.xpu"
    path.write_text(source)
    cls = registry[dialect]

    def my_scale(out, inp):
        ...

    return cls(my_scale, file=path, **kwargs)


def _refs(*pairs):
    return {TLE_RAW_PAYLOADS_KEY: list(pairs)}


# -- frontend: dialects, arch and toolchain resolution ----------------------


def test_registry_maps_chip_generations_to_their_stack():
    """Names are chip generations: xpu3 is the XPU stack, xpu4/xpu5 are the alternate one."""
    assert registry["xpu3"] is XPUJITFunction
    assert registry["xpu4"] is XCNJITFunction
    assert registry["xpu5"] is XCNJITFunction


def test_aliases_of_one_stack_share_a_payload_id(tmp_path):
    """`dialect_name` drives the id, so an alias does not cause a second compile."""
    assert (_handle(tmp_path, "xpu4").register_payload() == _handle(tmp_path, "xpu5").register_payload())


def test_arch_resolution_precedence(tmp_path, monkeypatch):
    monkeypatch.delenv("TRITON_XPU_ARCH", raising=False)
    monkeypatch.delenv("TRITON_XCN_ARCH", raising=False)
    assert _handle(tmp_path, "xpu3").resolve_arch() == 3
    assert _handle(tmp_path, "xpu4").resolve_arch() == 4
    # Explicit arch wins over the env var, and the caller (the backend, during
    # the merge) wins over both.
    monkeypatch.setenv("TRITON_XPU_ARCH", "4")
    assert _handle(tmp_path, "xpu3").resolve_arch() == 4
    monkeypatch.setenv("TRITON_XCN_ARCH", "6")
    assert _handle(tmp_path, "xpu4").resolve_arch() == 6
    assert _handle(tmp_path, "xpu3", arch=3).resolve_arch() == 3
    assert _handle(tmp_path, "xpu3", arch=3).resolve_arch(5) == 5


def test_toolchain_dir_is_the_packaged_per_arch_one(tmp_path):
    """XPUBackend keys the clang dir off `arch`; handing it the wrong option
    object would raise AttributeError instead."""
    if "TRITON_XPU_CLANG_PATH" in os.environ:
        pytest.skip("TRITON_XPU_CLANG_PATH overrides the packaged toolchain layout")
    clang_dir = _handle(tmp_path, "xpu3")._backend_clang_dir(3)
    assert clang_dir.name == "bin" and "xpu3" in str(clang_dir)


def test_alternate_stack_honors_launcher_arch(tmp_path, monkeypatch):
    """The alternate stack honors the XPU arch variable set by its launcher."""
    monkeypatch.delenv("TRITON_XCN_ARCH", raising=False)
    monkeypatch.setenv("TRITON_XPU_ARCH", "5")
    assert _handle(tmp_path, "xpu4").resolve_arch() == 5
    monkeypatch.setenv("TRITON_XCN_ARCH", "6")
    assert _handle(tmp_path, "xpu4").resolve_arch() == 6


def test_stack_toolchains_are_looked_up_by_arch(tmp_path):
    """Each stack keys its packaged clang directory by the selected arch."""
    if "TRITON_XPU_CLANG_PATH" in os.environ:
        pytest.skip("TRITON_XPU_CLANG_PATH overrides the packaged toolchain layout")
    xpu_dir = _handle(tmp_path, "xpu3")._backend_clang_dir(4)
    assert xpu_dir.name == "bin" and "xpu4" in str(xpu_dir)
    try:
        alt_dir = _handle(tmp_path / "alternate", "xpu4")._backend_clang_dir(4)
    except Exception as exc:
        pytest.skip(f"no packaged alternate toolchain in this checkout: {exc}")
    assert alt_dir.name == "bin" and "xpu4" in str(alt_dir)


def test_a_payload_without_file_or_source_is_rejected():

    def my_scale():
        ...

    with pytest.raises(ValueError, match="needs either `source` or `file`"):
        XPUJITFunction(my_scale)


# -- cache key --------------------------------------------------------------


def test_editing_the_payload_changes_the_cache_key(tmp_path):
    handle = _handle(tmp_path, "xpu3")
    before = handle.cache_key
    handle.file.write_text('extern "C" __device__ void my_scale() { /* v2 */ }')
    assert handle.cache_key != before


def test_dependencies_finder_folds_the_payload_in(tmp_path):
    from triton.runtime.jit import DependenciesFinder

    handle = _handle(tmp_path, "xpu3")
    finder = DependenciesFinder("kernel", {}, {}, "def kernel():\n    pass\n")
    baseline = finder.ret
    finder.record_reference(handle, {}, "my_scale")
    assert finder.ret != baseline
    # Hashed, not tracked as a mutable global (which would deepcopy the handle).
    assert not finder.used_global_vals


# -- source store -----------------------------------------------------------


def test_payload_sources_are_content_addressed(tmp_path):
    first = _handle(tmp_path, "xpu3").register_payload()
    second = _handle(tmp_path, "xpu3").register_payload()
    assert first == second
    assert list(list_pending_sources()) == [first]
    assert list_pending_sources()[first]["dialect"] == "xpu"


def test_payload_id_tracks_dialect_and_source(tmp_path):
    xpu_id = _handle(tmp_path / "a", "xpu3").register_payload()
    alt_id = _handle(tmp_path / "b", "xpu4").register_payload()
    edited = _handle(tmp_path / "c", "xpu3", source='extern "C" __device__ void my_scale() { }')
    assert len({xpu_id, alt_id, edited.register_payload()}) == 3


def test_deferred_payload_is_compiled_for_the_requested_arch():
    stub = _StubHandle()
    sid = register_source(dialect="xcn", callee="my_scale", source="src", handle=stub)

    assert payload_ll(sid, 4) == PAYLOAD_IR
    assert payload_ll(sid, 4) == PAYLOAD_IR
    # Compiled once per arch, not once per request.
    assert stub.arches == [4]
    assert payload_ll(sid, 6) == PAYLOAD_IR
    assert stub.arches == [4, 6]


def test_eager_payload_is_pinned_to_its_own_arch():
    """`deferred=False` compiles at trace time and that text wins from then on."""
    stub = _StubHandle()
    sid = register_source(dialect="xcn", callee="my_scale", source="src", handle=stub)
    record_eager(sid, "pinned")

    assert payload_ll(sid, 4) == "pinned"
    assert payload_ll(sid, 8) == "pinned"
    assert stub.arches == []


def test_unregistered_payload_is_reported():
    with pytest.raises(RuntimeError, match="is not registered"):
        payload_ll("deadbeef", 4)


def test_source_store_evicts_the_least_recently_used(monkeypatch):
    """A process tracing many distinct payloads must not grow the store without
    bound, and eviction must not drop something still in use."""
    monkeypatch.setenv("TRITON_TLE_RAW_MAX_SOURCES", "3")
    ids = [register_source(dialect="xcn", callee=f"k{i}", source=f"src{i}", handle=_StubHandle()) for i in range(3)]

    # Using an entry makes it the most recently used, so the entry evicted by the
    # next insert is a different one -- the oldest, not the one just touched.
    payload_ll(ids[0], 4)
    newest = register_source(dialect="xcn", callee="k3", source="src3", handle=_StubHandle())

    assert set(list_pending_sources()) == {ids[0], ids[2], newest}


def test_source_store_rejects_a_nonsense_cap(monkeypatch):
    monkeypatch.setenv("TRITON_TLE_RAW_MAX_SOURCES", "0")
    with pytest.raises(RuntimeError, match="at least 1"):
        register_source(dialect="xcn", callee="k", source="src")


# -- merge ------------------------------------------------------------------


def _merge(kernel, callees=("my_scale", ), arch=4):
    """Merge a stub payload into `kernel`; the stub defines `my_scale`."""
    stub = _StubHandle()
    sid = register_source(dialect="xcn", callee="my_scale", source="src", handle=stub)
    refs = [(callee, sid) for callee in callees]
    return merge_raw_payloads(kernel, _refs(*refs), BACKEND, arch)


def test_merge_binds_the_call_and_keeps_the_payload_inlinable():
    merged = _merge(KERNEL_IR)

    # The kernel's declaration is satisfied by the payload's definition, so the
    # call binds within the module -- which is what lets `alwaysinline` fire.
    assert "declare void @my_scale" not in merged
    assert "define internal void @my_scale" in merged
    assert "alwaysinline" in merged
    assert "raw_kernel" in merged
    # The payload's keep-alive array would pin its body into the object file
    # after inlining made it dead code; the kernel's call is the root instead.
    assert "compiler.used" not in merged


def test_merge_leaves_a_kernel_without_raw_ops_alone():
    kernel = KERNEL_IR.replace("  call void @my_scale(ptr addrspace(1) %0, i32 %1) #0\n", "")
    assert merge_raw_payloads(kernel, {}, BACKEND, 4) == kernel


def test_merge_reports_a_signature_mismatch():
    with pytest.raises(RuntimeError) as excinfo:
        _merge(MISMATCHED_KERNEL_IR)
    message = str(excinfo.value)
    assert "does not match the call site" in message
    assert "i64" in message and "i32" in message


def test_merge_reports_a_payload_that_was_never_registered():
    """An IR entry point (or a stage override) compiles without tracing."""
    with pytest.raises(RuntimeError, match="is not in the kernel module"):
        merge_raw_payloads(KERNEL_IR, {}, BACKEND, 4)


def test_merge_reports_a_payload_without_the_entry():
    with pytest.raises(RuntimeError, match="does not define 'vec_add' exactly once"):
        _merge(KERNEL_IR, callees=("vec_add", ))


# -- harvest ----------------------------------------------------------------


def test_record_raw_payloads_harvests_the_module():
    context = ir.context()
    ir.load_dialects(context)
    xpu.load_dialects(context)
    builder = gluon_ir.GluonOpBuilder(context)
    mod = builder.create_module()
    ptr_ty = builder.get_ptr_ty(builder.get_float_ty(), 1)
    i32_ty = builder.get_int32_ty()
    fn = builder.get_or_insert_function(mod, "kernel", builder.get_function_ty([ptr_ty, i32_ty], []), "public", False)
    mod.push_back(fn)
    builder.set_insertion_point_to_start(fn.add_entry_block())
    builder.create_xpu_raw("my_scale", "deadbeef", [fn.args(0), fn.args(1)])
    builder.ret([])

    metadata = {}
    record_raw_payloads(mod, metadata)
    record_raw_extern_libs(metadata, ("/lib/ockl.bc", ))
    assert metadata[TLE_RAW_PAYLOADS_KEY] == [("my_scale", "deadbeef")]
    assert metadata[TLE_RAW_EXTERN_LIBS_KEY] == ["/lib/ockl.bc"]


def test_eager_payload_compiles_at_trace_time():
    """`deferred=False` compiles the payload right away, so a broken payload is
    reported when `tle.raw.call` runs instead of when the backend merges it."""
    import triton.experimental.tle as tle

    @tle.raw.dialect("xpu3", source='extern "C" void my_scale(float* out) {}', arch=XPU_ARCH, deferred=False)
    def my_scale(out):
        ...

    with pytest.raises(RuntimeError, match="defines no 'my_scale'"):
        my_scale.register_payload()


def test_only_payloads_with_library_calls_pull_in_a_device_library():
    """A payload that talks to the device library (`__ockl_*`, what a cluster
    payload calls for its thread id) needs it linked in; an intrinsic-only payload
    must not, or the whole `xpu::print*`/printf cluster comes along and `llc`
    cannot drop it."""
    from triton.experimental.tle.raw.merge import _payloads_need_device_libs

    kernel = "define xcn_kernel void @k() {\n  ret void\n}\n"
    libcall = "declare i64 @__ockl_get_group_id(i32)\n" \
              "define internal void @p() {\n  %r = call i64 @__ockl_get_group_id(i32 0)\n  ret void\n}\n"
    intrinsic = "define internal void @p() {\n  %r = call i32 @llvm.xcn.workgroup.id.x()\n  ret void\n}\n"
    local = "define internal void @p() {\n  call void @helper()\n  ret void\n}\n" \
            "define internal void @helper() {\n  ret void\n}\n"

    assert _payloads_need_device_libs(kernel, [libcall]) is True
    assert _payloads_need_device_libs(kernel, [intrinsic]) is False
    assert _payloads_need_device_libs(kernel, [local]) is False


OTHER_PAYLOAD_IR = PAYLOAD_IR.replace("my_scale", "other")

# Two payload calls in one kernel: the merge has to bind each one.
TWO_PAYLOAD_KERNEL_IR = KERNEL_IR.replace(
    "declare void @my_scale(ptr addrspace(1), i32)",
    "declare void @my_scale(ptr addrspace(1), i32)\ndeclare void @other(ptr addrspace(1), i32)").replace(
        "  call void @my_scale(ptr addrspace(1) %0, i32 %1) #0\n",
        "  call void @my_scale(ptr addrspace(1) %0, i32 %1) #0\n"
        "  call void @other(ptr addrspace(1) %0, i32 %1) #0\n")


def test_merge_links_every_payload_of_a_kernel():
    """A kernel can call payloads with different source ids; each is linked in and
    bound, and none is left as a declaration."""
    refs = []
    for callee, text in (("my_scale", PAYLOAD_IR), ("other", OTHER_PAYLOAD_IR)):
        stub = _StubHandle(text)
        refs.append((callee, register_source(dialect="xcn", callee=callee, source=callee, handle=stub)))

    merged = merge_raw_payloads(TWO_PAYLOAD_KERNEL_IR, _refs(*refs), BACKEND, 4)

    assert "define internal void @my_scale" in merged
    assert "define internal void @other" in merged
    assert "declare void @my_scale" not in merged
    assert "declare void @other" not in merged


def test_merge_links_a_payload_called_twice_only_once():
    """Several raw ops on one payload are a single definition to link in; linking
    the text in twice would leave `llvm-link` renaming the second copy and pinning
    a duplicate of the body into the object file."""
    stub = _StubHandle()
    sid = register_source(dialect="xcn", callee="my_scale", source="src", handle=stub)

    merged = merge_raw_payloads(KERNEL_IR, _refs(("my_scale", sid), ("my_scale", sid)), BACKEND, 4)

    assert merged.count("define internal void @my_scale(") == 1
    assert "my_scale." not in merged


def test_merge_reports_two_payloads_sharing_one_entry_name():
    """LLVM cannot hold two definitions of `my_scale`: `llvm-link` would rename one
    of them, silently binding the call to whichever survived."""
    stub_a = _StubHandle(PAYLOAD_IR)
    stub_b = _StubHandle(PAYLOAD_IR.replace("ret void", "ret void ; v2"))
    sid_a = register_source(dialect="xcn", callee="my_scale", source="a", handle=stub_a)
    sid_b = register_source(dialect="xcn", callee="my_scale", source="b", handle=stub_b)

    with pytest.raises(RuntimeError, match="two different payloads both define 'my_scale'"):
        merge_raw_payloads(KERNEL_IR, _refs(("my_scale", sid_a), ("my_scale", sid_b)), BACKEND, 4)


def test_merge_reports_a_payload_that_is_not_llvm_ir():
    """The payload is linked in as LLVM IR text, so the MLIR-dialect form the old
    MLIR-import path accepted can no longer be merged -- and says so."""
    mlir_payload = ("llvm.func @my_scale(%arg0: !llvm.ptr, %arg1: i32) "
                    "{ llvm.return }\n")
    stub = _StubHandle(mlir_payload)
    sid = register_source(dialect="xcn", callee="my_scale", source="src", handle=stub)

    with pytest.raises(RuntimeError, match="linked in as LLVM IR text"):
        merge_raw_payloads(KERNEL_IR, _refs(("my_scale", sid)), BACKEND, 4)


# What the XTDK clang actually emits for a payload: `hidden` visibility and
# `local_unnamed_addr` rather than `internal`, plus the non-ABI parameter
# attributes it adds on its own. The kernel only ever declares bare types, so the
# signature check has to look past all of that.
ATTRIBUTED_PAYLOAD_IR = PAYLOAD_IR.replace(
    "define internal void @my_scale(ptr addrspace(1) %out, i32 %n) #0 {",
    "define hidden void @my_scale(ptr addrspace(1) nocapture noundef %out, i32 noundef %n) "
    "local_unnamed_addr #0 {")

# `llvm-link` folds every module's `!llvm.ident` into one multi-node list; `llc`
# turns those version strings into a `.comment` section per kernel.
IDENT_PAYLOAD_IR = PAYLOAD_IR + '\n!llvm.ident = !{!0}\n!0 = !{!"clang version 19.1.7"}\n'


def test_merge_looks_past_the_payloads_non_abi_attributes():
    """A payload carrying clang's attributes and a kernel declaring bare types is
    the normal case, not a signature mismatch."""
    stub = _StubHandle(ATTRIBUTED_PAYLOAD_IR)
    sid = register_source(dialect="xcn", callee="my_scale", source="src", handle=stub)

    merged = merge_raw_payloads(KERNEL_IR, _refs(("my_scale", sid)), BACKEND, 4)

    assert "define internal void @my_scale" in merged
    assert "declare void @my_scale" not in merged


def test_merge_still_catches_a_mismatch_behind_the_attributes():
    """The other direction: the attributes must not make the comparison so
    forgiving that the type mismatch the check exists for slips through."""
    stub = _StubHandle(ATTRIBUTED_PAYLOAD_IR.replace("i32 noundef %n", "i64 noundef %n"))
    sid = register_source(dialect="xcn", callee="my_scale", source="src", handle=stub)

    with pytest.raises(RuntimeError, match="does not match the call site"):
        merge_raw_payloads(KERNEL_IR, _refs(("my_scale", sid)), BACKEND, 4)


def test_merge_strips_the_compiler_ident_banner():
    """The merged `!llvm.ident` and the node it points at are dropped, so `llc`
    does not turn the compiler version string into a `.comment` section."""
    stub = _StubHandle(IDENT_PAYLOAD_IR)
    sid = register_source(dialect="xcn", callee="my_scale", source="src", handle=stub)

    merged = merge_raw_payloads(KERNEL_IR, _refs(("my_scale", sid)), BACKEND, 4)

    assert "llvm.ident" not in merged
    assert "clang version" not in merged
