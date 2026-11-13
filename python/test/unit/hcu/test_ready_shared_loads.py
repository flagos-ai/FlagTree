"""Actual LLVM scheduling: slot scopes, synchronization epochs, safe fallback."""
import pytest
from triton._C import libtriton
from triton.backends.compiler import GPUTarget

if not hasattr(libtriton, "hcu"):
    pytest.skip("HCU compiler required", allow_module_level=True)

from triton.backends.hcu.compiler import HIPBackend


def source(native=False, negative=None, structured=False):
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
        parts += [f"""
    llvm.call @llvm.amdgcn.raw.ptr.buffer.load.lds(%rsrc, %write, %size,
      %zero, %zero, %zero, %zero) {{alias_scopes = [#copy{slot}]}}
      : (!llvm.ptr<8>, !llvm.ptr<3>, i32, i32, i32, i32, i32) -> ()
    {boundary}
    {read}
"""]
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
        text = text.replace("llvm.amdgcn.raw.ptr.buffer.load.lds",
                            "llvm.amdgcn.struct.ptr.buffer.load.lds")
        text = text.replace("!llvm.ptr<8>, !llvm.ptr<3>, i32, i32, i32, i32, i32)",
                            "!llvm.ptr<8>, !llvm.ptr<3>, i32, i32, i32, i32, i32, i32)")
        text = text.replace("%zero, %zero, %zero, %zero)",
                            "%zero, %zero, %zero, %zero, %zero)")
    return text


def lower(tmp_path, native=False, negative=None, arch="gfx936", kernel=True, structured=False):
    path = tmp_path / "ready.mlir"
    path.write_text(source(native, negative, structured))
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
    changed, before, after = lower(tmp_path, native)
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


@pytest.mark.parametrize("negative", ["noalias", "wrong_scope", "barrier", "wait",
                                     "unknown_call", "unknown_store", "conditional_assume"])
@pytest.mark.parametrize("native", [False, True])
def test_no_crossing_unproved_alias_or_memory_effect(tmp_path, native, negative):
    changed, before, after = lower(tmp_path, native, negative)
    assert changed == 0 and after == before


@pytest.mark.parametrize("arch,kernel", [("gfx942", True), ("gfx936", False)])
def test_target_and_kernel_scope_remain_exact_off(tmp_path, arch, kernel):
    changed, before, after = lower(tmp_path, arch=arch, kernel=kernel)
    assert changed == 0 and after == before


@pytest.mark.parametrize("native", [False, True])
def test_true_storage_assume_is_not_a_memory_boundary(tmp_path, native):
    changed, before, after = lower(tmp_path, native, "storage_fact")
    assert changed == 2
    assert after.count('"dereferenceable"') == before.count('"dereferenceable"') == 2
    assert after.count("call void @llvm.amdgcn.s.barrier()") == 1


@pytest.mark.parametrize("native", [False, True])
@pytest.mark.parametrize("boundary", [None, "storage_fact", "noalias", "wrong_scope",
                                     "barrier", "wait", "unknown_call",
                                     "unknown_store", "conditional_assume"])
def test_structured_producer_obeys_same_epoch_and_complete_alias_proof(tmp_path, native, boundary):
    changed, before, after = lower(tmp_path, native, boundary, structured=True)
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
